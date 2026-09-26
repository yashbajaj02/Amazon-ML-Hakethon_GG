import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def create_sample(sample_size: int = 10000, distractor_ratio: float = 2.0, random_seed: int = 42):
    np.random.seed(random_seed)
    
    project_root = Path(__file__).resolve().parent.parent
    train_dir = project_root / "dataset" / "train"
    out_dir = project_root / "dataset" / "sample_val"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading ground truth from {train_dir / 'train_ground_truth.tsv'}...")
    gt_df = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", keep_default_na=False)
    
    # Stratified sample by singleton status
    is_singleton = gt_df["matched_entity_ids"].str.strip() == ""
    singletons = gt_df[is_singleton]
    non_singletons = gt_df[~is_singleton]
    
    singleton_prop = len(singletons) / len(gt_df)
    n_singletons = int(round(sample_size * singleton_prop))
    n_non_singletons = sample_size - n_singletons
    
    sample_singletons = singletons.sample(n=n_singletons, random_state=random_seed)
    sample_non_singletons = non_singletons.sample(n=n_non_singletons, random_state=random_seed)
    
    sample_gt = pd.concat([sample_singletons, sample_non_singletons]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    s1_ids = set(sample_gt["source1_entity_id"])
    
    # Collect all true matched target IDs
    true_target_ids = set()
    for raw in sample_gt["matched_entity_ids"]:
        if raw.strip():
            for m in raw.split(","):
                if m.strip():
                    true_target_ids.add(m.strip())
                    
    true_s2_ids = {i for i in true_target_ids if i.startswith("S2-")}
    true_s3_ids = {i for i in true_target_ids if i.startswith("S3-")}
    
    print(f"Sampled {len(sample_gt):,} S1 entities ({n_singletons} singletons, {n_non_singletons} with matches).")
    print(f"Requires {len(true_s2_ids):,} S2 true matches and {len(true_s3_ids):,} S3 true matches.")
    
    # Save sampled ground truth
    sample_gt.to_csv(out_dir / "val_ground_truth.tsv", sep="\t", index=False)
    
    # Load and filter Source 1
    print("Filtering Source 1...")
    s1_df = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", keep_default_na=False)
    s1_sub = s1_df[s1_df["entity_id"].isin(s1_ids)].copy()
    s1_sub.to_csv(out_dir / "val_source1.tsv", sep="\t", index=False)
    del s1_df
    
    # Load and filter Source 2 with distractors
    print("Filtering Source 2...")
    s2_df = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", keep_default_na=False)
    s2_true = s2_df[s2_df["entity_id"].isin(true_s2_ids)]
    s2_remaining = s2_df[~s2_df["entity_id"].isin(true_s2_ids)]
    n_s2_distractors = int(len(s2_true) * distractor_ratio)
    s2_dist = s2_remaining.sample(n=min(n_s2_distractors, len(s2_remaining)), random_state=random_seed)
    s2_sub = pd.concat([s2_true, s2_dist]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    s2_sub.to_csv(out_dir / "val_source2.tsv", sep="\t", index=False)
    del s2_df, s2_true, s2_remaining, s2_dist
    
    # Load and filter Source 3 with distractors
    print("Filtering Source 3...")
    s3_df = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", keep_default_na=False)
    s3_true = s3_df[s3_df["entity_id"].isin(true_s3_ids)]
    s3_remaining = s3_df[~s3_df["entity_id"].isin(true_s3_ids)]
    n_s3_distractors = int(len(s3_true) * distractor_ratio)
    s3_dist = s3_remaining.sample(n=min(n_s3_distractors, len(s3_remaining)), random_state=random_seed)
    s3_sub = pd.concat([s3_true, s3_dist]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    s3_sub.to_csv(out_dir / "val_source3.tsv", sep="\t", index=False)
    del s3_df, s3_true, s3_remaining, s3_dist
    
    print(f"Validation sample created successfully in {out_dir}:")
    print(f"  Source 1: {len(s1_sub):,} records")
    print(f"  Source 2: {len(s2_sub):,} records")
    print(f"  Source 3: {len(s3_sub):,} records")
    print(f"  Ground Truth: {len(sample_gt):,} records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10000, help="Number of S1 entities")
    parser.add_argument("--distractors", type=float, default=2.0, help="Distractor multiplier")
    args = parser.parse_args()
    create_sample(sample_size=args.size, distractor_ratio=args.distractors)
