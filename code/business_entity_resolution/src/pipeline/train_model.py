import argparse
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CODE_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = CODE_DIR.parent.parent
sys.path.insert(0, str(CODE_DIR))

from src.config import CHECKPOINT_DIR
from src.data.loader import load_source_tsv, load_ground_truth
from src.data.preprocessor import preprocess_dataframe
from src.blocking.candidate_generator import CandidateGenerator
from src.features.feature_extractor import extract_features_for_pairs
from src.models.matcher import EntityMatcher, FEATURE_COLUMNS
from src.evaluation.metrics import compute_macro_f05


def train_and_save_model(data_dir: Path, out_checkpoint_dir: Path = CHECKPOINT_DIR):
    out_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("STARTING MODEL TRAINING & CALIBRATION PIPELINE")
    print(f"Data Source: {data_dir}")
    print(f"Checkpoint Target: {out_checkpoint_dir}")
    print("=" * 60)

    # 1. Load data
    s1_files = list(data_dir.glob("*_source1.tsv"))
    s2_files = list(data_dir.glob("*_source2.tsv"))
    s3_files = list(data_dir.glob("*_source3.tsv"))
    gt_files = list(data_dir.glob("*_ground_truth.tsv"))

    if not (s1_files and s2_files and s3_files and gt_files):
        raise FileNotFoundError(f"Missing required TSV files in {data_dir}")

    print("\n[Step 1] Loading dataset...")
    s1_df = load_source_tsv(s1_files[0])
    s2_df = load_source_tsv(s2_files[0])
    s3_df = load_source_tsv(s3_files[0])
    ground_truth = load_ground_truth(gt_files[0])

    print(f"Loaded: S1={len(s1_df):,}, S2={len(s2_df):,}, S3={len(s3_df):,}, GT={len(ground_truth):,}")

    # 2. Preprocess
    print("\n[Step 2] Normalizing text & addresses...")
    s1_df = preprocess_dataframe(s1_df)
    s2_df = preprocess_dataframe(s2_df)
    s3_df = preprocess_dataframe(s3_df)
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)

    # 3. Blocking
    print("\n[Step 3] Candidate Generation (Blocking)...")
    generator = CandidateGenerator(top_k=30, min_sim=0.10)
    candidates = generator.generate_candidates_by_country(s1_df, targets_df)

    # 4. Feature Extraction
    print("\n[Step 4] Extracting Pairwise Features...")
    pairs = []
    labels = []
    for s1_id, cand_list in candidates.items():
        true_matches_set = set(ground_truth.get(s1_id, []))
        for cand_id in cand_list:
            pairs.append((s1_id, cand_id))
            labels.append(1 if cand_id in true_matches_set else 0)

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

    feature_df = extract_features_for_pairs(pairs, records_s1, records_target)
    y = np.array(labels)

    # 5. Train / Validation Split (by entity to avoid leakage)
    print("\n[Step 5] Fitting LightGBM Model...")
    s1_ids = list(s1_df["entity_id"])
    train_ids, val_ids = train_test_split(s1_ids, test_size=0.25, random_state=42)

    train_mask = feature_df["source1_entity_id"].isin(set(train_ids))
    val_mask = feature_df["source1_entity_id"].isin(set(val_ids))

    X_train, y_train = feature_df[train_mask], y[train_mask]
    X_val, y_val = feature_df[val_mask], y[val_mask]

    matcher = EntityMatcher()
    matcher.fit(X_train, y_train)

    # 6. Optimize threshold for F_0.5
    print("\n[Step 6] Calibrating Threshold on Validation Set for F_0.5...")
    val_probs = matcher.predict_proba(X_val)
    val_gt = {s1: ground_truth[s1] for s1 in val_ids}

    thresholds = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    best_thresh = matcher.optimize_threshold(X_val, val_probs, val_gt, thresholds=thresholds)

    # Evaluate final validation metric
    val_df_eval = X_val.copy()
    val_df_eval["prob"] = val_probs
    matched = val_df_eval[val_df_eval["prob"] >= best_thresh]

    preds = {s1: [] for s1 in val_ids}
    for _, row in matched.iterrows():
        preds[row["source1_entity_id"]].append(row["candidate_entity_id"])
    for s1 in preds:
        preds[s1] = list(dict.fromkeys(preds[s1]))

    val_f05 = compute_macro_f05(preds, val_gt)

    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"Validation Macro F_0.5 Score: {val_f05:.4f}")

    # 7. Save model checkpoint & metadata
    model_path = out_checkpoint_dir / "lgbm_matcher.joblib"
    meta_path = out_checkpoint_dir / "model_metadata.json"

    joblib.dump(matcher.model, model_path)
    metadata = {
        "model_type": "LightGBM",
        "optimal_threshold": float(best_thresh),
        "validation_macro_f05": float(val_f05),
        "feature_columns": FEATURE_COLUMNS,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Step 7] Model successfully saved to {model_path}!")
    print(f"Metadata saved to {meta_path}!")
    return model_path, best_thresh, val_f05


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "sample_val",
        help="Path to training data directory",
    )
    args = parser.parse_args()
    train_and_save_model(args.data_dir)
