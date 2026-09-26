import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root and code dir to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution"))

from src.data.loader import load_source_tsv, load_ground_truth
from src.data.preprocessor import preprocess_dataframe
from src.blocking.candidate_generator import CandidateGenerator
from src.features.feature_extractor import extract_features_for_pairs
from src.models.matcher import EntityMatcher, FEATURE_COLUMNS
from src.evaluation.metrics import compute_macro_f05, compute_entity_f05

def run_benchmark():
    val_dir = PROJECT_ROOT / "dataset" / "sample_val"
    print("=" * 60)
    print("PHASE 1: RUNNING VALIDATION BENCHMARK EVALUATION")
    print("=" * 60)
    
    # 1. Load sample dataset
    print("\n[Step 1] Loading validation split...")
    s1_df = load_source_tsv(val_dir / "val_source1.tsv")
    s2_df = load_source_tsv(val_dir / "val_source2.tsv")
    s3_df = load_source_tsv(val_dir / "val_source3.tsv")
    ground_truth = load_ground_truth(val_dir / "val_ground_truth.tsv")
    
    print(f"Loaded: S1={len(s1_df):,}, S2={len(s2_df):,}, S3={len(s3_df):,}, GT={len(ground_truth):,}")
    
    # 2. Preprocess text
    print("\n[Step 2] Preprocessing names and addresses...")
    s1_df = preprocess_dataframe(s1_df)
    s2_df = preprocess_dataframe(s2_df)
    s3_df = preprocess_dataframe(s3_df)
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    # 3. Blocking / Candidate Generation
    print("\n[Step 3] Running Candidate Generation (Blocking)...")
    generator = CandidateGenerator(top_k=30, min_sim=0.10)
    candidates = generator.generate_candidates_by_country(s1_df, targets_df)
    
    # Calculate Candidate Recall & Reduction Ratio
    total_true_matches = sum(len(matches) for matches in ground_truth.values())
    captured_matches = 0
    candidate_counts = []
    
    for s1_id, gt_matches in ground_truth.items():
        cand_set = set(candidates.get(s1_id, []))
        captured_matches += len(set(gt_matches) & cand_set)
        candidate_counts.append(len(cand_set))
        
    candidate_recall = captured_matches / total_true_matches if total_true_matches > 0 else 0.0
    avg_cands = np.mean(candidate_counts)
    
    print("-" * 50)
    print(f"BLOCKING PERFORMANCE:")
    print(f"  Total True Matches: {total_true_matches:,}")
    print(f"  Captured in Candidates: {captured_matches:,}")
    print(f"  Candidate Recall Ceiling: {candidate_recall * 100:.2f}%")
    print(f"  Avg Candidates per Entity: {avg_cands:.1f}")
    print("-" * 50)
    
    # 4. Feature Extraction for Candidate Pairs
    print("\n[Step 4] Extracting Pairwise Features on Candidates...")
    pairs = []
    pair_labels = []
    
    for s1_id, cand_list in candidates.items():
        true_matches_set = set(ground_truth.get(s1_id, []))
        for cand_id in cand_list:
            pairs.append((s1_id, cand_id))
            pair_labels.append(1 if cand_id in true_matches_set else 0)
            
    records_s1 = {
        row["entity_id"]: (row["clean_name"], row["clean_address"], row["clean_country"])
        for _, row in s1_df.iterrows()
    }
    records_target = {
        row["entity_id"]: (row["clean_name"], row["clean_address"], row["clean_country"])
        for _, row in targets_df.iterrows()
    }
    
    feature_df = extract_features_for_pairs(pairs, records_s1, records_target)
    y = np.array(pair_labels)
    
    print(f"Total candidate pairs generated: {len(feature_df):,}")
    print(f"Positive match pairs (class 1): {y.sum():,} ({y.mean() * 100:.2f}%)")
    print(f"Negative distractor pairs (class 0): {(1 - y).sum():,} ({(1 - y).mean() * 100:.2f}%)")
    
    # 5. Train/Val Split by Entity ID (prevents data leakage)
    print("\n[Step 5] Training LightGBM Matcher (Entity-stratified split)...")
    s1_all_ids = list(s1_df["entity_id"])
    train_s1_ids, test_s1_ids = train_test_split(s1_all_ids, test_size=0.30, random_state=42)
    
    train_mask = feature_df["source1_entity_id"].isin(set(train_s1_ids))
    test_mask = feature_df["source1_entity_id"].isin(set(test_s1_ids))
    
    X_train, y_train = feature_df[train_mask], y[train_mask]
    X_test, y_test = feature_df[test_mask], y[test_mask]
    
    print(f"Training pairs: {len(X_train):,}, Validation pairs: {len(X_test):,}")
    
    matcher = EntityMatcher()
    matcher.fit(X_train, y_train)
    
    # 6. Optimize Threshold for Macro F_0.5
    print("\n[Step 6] Optimizing decision threshold for Macro F_0.5...")
    val_probs = matcher.predict_proba(X_test)
    test_gt = {s1_id: ground_truth[s1_id] for s1_id in test_s1_ids}
    
    thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    best_thresh = matcher.optimize_threshold(X_test, val_probs, test_gt, thresholds=thresholds)
    
    # Evaluate at best threshold
    X_test_eval = X_test.copy()
    X_test_eval["prob"] = val_probs
    matched = X_test_eval[X_test_eval["prob"] >= best_thresh]
    
    preds = {s1_id: [] for s1_id in test_s1_ids}
    for _, row in matched.iterrows():
        preds[row["source1_entity_id"]].append(row["candidate_entity_id"])
    for s1 in preds:
        preds[s1] = list(dict.fromkeys(preds[s1]))
        
    final_f05 = compute_macro_f05(preds, test_gt)
    
    # Singletons vs Non-singletons
    singleton_scores = []
    match_scores = []
    for s1_id, gt_list in test_gt.items():
        score = compute_entity_f05(preds[s1_id], gt_list)
        if len(gt_list) == 0:
            singleton_scores.append(score)
        else:
            match_scores.append(score)
            
    print("\n" + "=" * 60)
    print("FINAL VALIDATION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Optimal Probability Threshold: {best_thresh:.2f}")
    print(f"  Macro F_0.5 Score:             {final_f05:.4f}")
    print(f"  Singleton Accuracy:            {np.mean(singleton_scores) * 100:.2f}%")
    print(f"  Non-Singleton F_0.5:           {np.mean(match_scores):.4f}")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
