import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(CODE_DIR))

from src.config import CHECKPOINT_DIR, TRAIN_DIR
from src.data.loader import load_source_tsv, load_ground_truth
from src.data.preprocessor import preprocess_dataframe
from src.blocking.candidate_generator import CandidateGenerator
from src.features.feature_extractor import extract_features_for_pairs
from src.models.matcher import EntityMatcher, FEATURE_COLUMNS
from src.evaluation.metrics import compute_macro_f05


def train_model_on_hard_negatives(sample_s1_size: int = 20000):
    print("=" * 60)
    print(f"TRAINING LIGHTGBM MATCHER WITH HARD NEGATIVE MINING ({sample_s1_size:,} S1 entities)")
    print("=" * 60)

    # 1. Load training data
    print("\n[Step 1] Loading subset of train data...")
    s1_all = load_source_tsv(TRAIN_DIR / "train_source1.tsv")
    gt_all = load_ground_truth(TRAIN_DIR / "train_ground_truth.tsv")

    # Sample S1 entities stratified by singleton
    is_singleton = s1_all["entity_id"].apply(lambda sid: len(gt_all.get(sid, [])) == 0)
    singletons = s1_all[is_singleton].sample(n=min(int(sample_s1_size * 0.08), is_singleton.sum()), random_state=42)
    non_singletons = s1_all[~is_singleton].sample(n=sample_s1_size - len(singletons), random_state=42)
    s1_df = pd.concat([singletons, non_singletons]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Sampled S1: {len(s1_df):,} entities ({len(singletons):,} singletons)")

    # Collect true match IDs needed
    s1_ids_set = set(s1_df["entity_id"])
    true_target_ids = set()
    for sid in s1_ids_set:
        for tid in gt_all.get(sid, []):
            true_target_ids.add(tid)

    # Load targets (include all true matches + distractors)
    print("\n[Step 2] Loading targets with distractors...")
    s2_all = load_source_tsv(TRAIN_DIR / "train_source2.tsv")
    s3_all = load_source_tsv(TRAIN_DIR / "train_source3.tsv")

    s2_true = s2_all[s2_all["entity_id"].isin(true_target_ids)]
    s2_dist = s2_all[~s2_all["entity_id"].isin(true_target_ids)].sample(n=min(len(s2_true) * 3, 150000), random_state=42)
    s2_df = pd.concat([s2_true, s2_dist]).reset_index(drop=True)

    s3_true = s3_all[s3_all["entity_id"].isin(true_target_ids)]
    s3_dist = s3_all[~s3_all["entity_id"].isin(true_target_ids)].sample(n=min(len(s3_true) * 3, 150000), random_state=42)
    s3_df = pd.concat([s3_true, s3_dist]).reset_index(drop=True)

    targets_all = pd.concat([s2_df, s3_df], ignore_index=True)
    print(f"Total Targets pool: {len(targets_all):,} ({len(true_target_ids):,} true matches + {len(targets_all) - len(true_target_ids):,} distractors)")

    # 3. Preprocess
    print("\n[Step 3] Preprocessing text...")
    s1_df = preprocess_dataframe(s1_df)
    targets_all = preprocess_dataframe(targets_all)

    # 4. Blocking / Candidate Generation
    print("\n[Step 4] Mining Hard Negative Candidates via Blocking...")
    generator = CandidateGenerator(top_k=12)
    candidates = generator.generate_candidates_by_country(s1_df, targets_all)

    # 5. Build Training Pairs
    print("\n[Step 5] Building Pairwise Dataset with Hard Negatives...")
    pairs = []
    labels = []
    for sid, cand_list in candidates.items():
        true_set = set(gt_all.get(sid, []))
        for cid in cand_list:
            pairs.append((sid, cid))
            labels.append(1 if cid in true_set else 0)

    y = np.array(labels)
    print(f"Total Candidate Pairs: {len(pairs):,}")
    print(f"  Positive matches (1): {y.sum():,} ({y.mean() * 100:.2f}%)")
    print(f"  Hard negatives   (0): {(1 - y).sum():,} ({(1 - y.mean()) * 100:.2f}%)")

    records_s1 = dict(zip(s1_df["entity_id"], zip(s1_df["clean_name"], s1_df["clean_address"], s1_df["clean_country"])))
    records_target = dict(zip(targets_all["entity_id"], zip(targets_all["clean_name"], targets_all["clean_address"], targets_all["clean_country"])))

    print("\n[Step 6] Extracting Discriminatory Features...")
    feature_df = extract_features_for_pairs(pairs, records_s1, records_target)

    # 7. Train / Validation Split (by entity to prevent leakage)
    print("\n[Step 7] Training LightGBM Model on Hard Negatives...")
    s1_ids = list(s1_df["entity_id"])
    train_ids, val_ids = train_test_split(s1_ids, test_size=0.25, random_state=42)

    train_mask = feature_df["source1_entity_id"].isin(set(train_ids))
    val_mask = feature_df["source1_entity_id"].isin(set(val_ids))

    X_train, y_train = feature_df[train_mask], y[train_mask]
    X_val, y_val = feature_df[val_mask], y[val_mask]

    matcher = EntityMatcher()
    matcher.fit(X_train, y_train)

    # 8. Calibrate Threshold for High-Precision Macro F_0.5
    print("\n[Step 8] Calibrating Threshold on Holdout Entities...")
    val_probs = matcher.predict_proba(X_val)
    val_gt = {s1: gt_all.get(s1, []) for s1 in val_ids}

    thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85, 0.88, 0.90]
    best_thresh = matcher.optimize_threshold(X_val, val_probs, val_gt, thresholds=thresholds)

    # Evaluate validation Macro F0.5 with guardrails
    val_df_eval = X_val.copy()
    val_df_eval["prob"] = val_probs
    valid_mask = (
        (val_df_eval["prob"] >= best_thresh)
        & (val_df_eval["num_conflict"] == 0.0)
        & (val_df_eval["branch_mismatch"] == 0.0)
    )
    matched = val_df_eval[valid_mask]

    preds = {s1: [] for s1 in val_ids}
    for _, row in matched.iterrows():
        preds[row["source1_entity_id"]].append(row["candidate_entity_id"])
    for s1 in preds:
        preds[s1] = list(dict.fromkeys(preds[s1]))

    val_f05 = compute_macro_f05(preds, val_gt)
    print(f"\n============================================================")
    print(f"Optimal Threshold:            {best_thresh:.2f}")
    print(f"Validation Macro F_0.5 Score: {val_f05:.4f}")
    print(f"============================================================")

    # 9. Save Checkpoint
    out_checkpoint_dir = CHECKPOINT_DIR
    out_checkpoint_dir.mkdir(parents=True, exist_ok=True)
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

    print(f"Model saved to {model_path}!")
    print(f"Metadata saved to {meta_path}!")


if __name__ == "__main__":
    train_model_on_hard_negatives(sample_s1_size=25000)
