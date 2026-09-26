import argparse
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm

import json
import joblib
from ..config import (
    BLOCKING_TOP_K,
    CANDIDATE_PAIRS_PATH,
    CHECKPOINT_DIR,
    MATCHING_RESULTS_PATH,
    MODEL_THRESHOLD,
    OUTPUT_DIR,
    TEST_DIR,
    TRAIN_DIR,
)
from ..data.loader import load_dataset_split, load_ground_truth
from ..data.preprocessor import preprocess_dataframe
from ..blocking.candidate_generator import CandidateGenerator
from ..features.feature_extractor import extract_features_for_pairs
from ..models.matcher import EntityMatcher, FEATURE_COLUMNS


def save_candidates_tsv(candidates_dict: dict, out_path: Path):
    """Saves candidate pairs in official challenge TSV format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s1_id, cand_list in candidates_dict.items():
        cand_str = ",".join(cand_list)
        rows.append(f"{s1_id}\t{cand_str}")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("\n".join(rows) + "\n")


def save_matches_tsv(matches_dict: dict, out_path: Path):
    """Saves final matches in official challenge TSV format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s1_id, match_list in matches_dict.items():
        match_str = ",".join(match_list)
        rows.append(f"{s1_id}\t{match_str}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("\n".join(rows) + "\n")


def run_pipeline(split: str = "test", top_k: int = BLOCKING_TOP_K, threshold_override: float = None):
    data_dir = TEST_DIR if split == "test" else TRAIN_DIR
    print(f"Loading {split} data from {data_dir}...")
    s1_df, s2_df, s3_df = load_dataset_split(data_dir, split_prefix=split)

    print("Preprocessing text...")
    s1_df = preprocess_dataframe(s1_df)
    s2_df = preprocess_dataframe(s2_df)
    s3_df = preprocess_dataframe(s3_df)

    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)

    if split == "test" and CANDIDATE_PAIRS_PATH.exists() and CANDIDATE_PAIRS_PATH.stat().st_size > 1000000:
        print(f"Loading existing candidate pairs from {CANDIDATE_PAIRS_PATH}...", flush=True)
        candidates = {}
        cand_df = pd.read_csv(CANDIDATE_PAIRS_PATH, sep="\t", keep_default_na=False)
        for s1_id, c_str in zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"]):
            candidates[s1_id] = [c.strip() for c in c_str.split(",") if c.strip()]
        print(f"Loaded candidates for {len(candidates):,} entities!", flush=True)
    else:
        print("Generating candidate pairs (blocking)...", flush=True)
        generator = CandidateGenerator(top_k=top_k)
        candidates = generator.generate_candidates_by_country(s1_df, targets_df)
        print(f"Saving candidate pairs to {CANDIDATE_PAIRS_PATH}...", flush=True)
        save_candidates_tsv(candidates, CANDIDATE_PAIRS_PATH)

    # Lookup dictionaries for fast feature extraction
    records_s1 = dict(
        zip(
            s1_df["entity_id"],
            zip(s1_df["clean_name"], s1_df["clean_address"], s1_df["clean_country"]),
        )
    )
    records_target = dict(
        zip(
            targets_df["entity_id"],
            zip(targets_df["clean_name"], targets_df["clean_address"], targets_df["clean_country"]),
        )
    )

    # Load trained model checkpoint
    print("Setting up LightGBM model matcher...")
    model_path = CHECKPOINT_DIR / "lgbm_matcher.joblib"
    meta_path = CHECKPOINT_DIR / "model_metadata.json"

    if model_path.exists() and meta_path.exists():
        print(f"Loading trained model checkpoint from {model_path}...")
        model = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        threshold = threshold_override if threshold_override is not None else metadata.get("optimal_threshold", 0.60)
        print(f"Applying calibrated F_0.5 decision threshold: {threshold:.2f}")
    else:
        model = None
        threshold = threshold_override if threshold_override is not None else MODEL_THRESHOLD
        print("Warning: No model checkpoint found, using rule-based thresholding.")

    # Score candidate pairs in stream chunks to keep memory usage bounded (< 2 GB)
    print("Scoring candidate pairs in memory-efficient stream chunks...", flush=True)
    matches = {s1_id: [] for s1_id in s1_df["entity_id"]}
    s1_id_list = list(s1_df["entity_id"])
    chunk_size = 25000
    total_chunks = (len(s1_id_list) + chunk_size - 1) // chunk_size

    for chunk_idx, chunk_start in enumerate(range(0, len(s1_id_list), chunk_size)):
        if (chunk_idx + 1) % 5 == 0 or chunk_idx == 0 or (chunk_idx + 1) == total_chunks:
            print(f"  [Matching] Scoring chunk {chunk_idx + 1} / {total_chunks} ({chunk_start:,} / {len(s1_id_list):,} entities)...", flush=True)

        chunk_s1_ids = s1_id_list[chunk_start : chunk_start + chunk_size]
        chunk_pairs = []
        for s1_id in chunk_s1_ids:
            for cand_id in candidates.get(s1_id, []):
                chunk_pairs.append((s1_id, cand_id))

        if not chunk_pairs:
            continue

        chunk_feat_df = extract_features_for_pairs(chunk_pairs, records_s1, records_target)
        if chunk_feat_df.empty:
            continue

        if model is not None:
            X = chunk_feat_df[FEATURE_COLUMNS]
            probs = model.predict_proba(X)[:, 1]
            valid_mask = (
                (probs >= threshold)
                & (chunk_feat_df["num_conflict"].values == 0.0)
                & (chunk_feat_df["pin_conflict"].values == 0.0)
            )
            selected = chunk_feat_df[valid_mask]
        else:
            confidence_mask = (
                (chunk_feat_df["name_token_set"] >= 0.85)
                & (chunk_feat_df["addr_token_set"] >= 0.60)
                & (chunk_feat_df["same_country"] == 1.0)
            ) | (chunk_feat_df["exact_name"] == 1.0)
            selected = chunk_feat_df[confidence_mask]

        for s1_m, cand_m in zip(selected["source1_entity_id"], selected["candidate_entity_id"]):
            matches[s1_m].append(cand_m)

    # Ensure uniqueness
    for s1_id in matches:
        matches[s1_id] = list(dict.fromkeys(matches[s1_id]))

    print(f"Saving final matches to {MATCHING_RESULTS_PATH}...")
    save_matches_tsv(matches, MATCHING_RESULTS_PATH)
    print("Pipeline execution complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Entity Resolution Pipeline")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--top_k", type=int, default=BLOCKING_TOP_K)
    parser.add_argument("--threshold", type=float, default=0.70, help="Probability threshold for matching")
    args = parser.parse_args()

    run_pipeline(split=args.split, top_k=args.top_k, threshold_override=args.threshold)
