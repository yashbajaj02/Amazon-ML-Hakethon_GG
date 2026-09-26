import sys
from pathlib import Path
import pandas as pd
import joblib
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution"))

from src.models.matcher import FEATURE_COLUMNS
from src.features.feature_extractor import compute_pair_features

NUM_REGEX = re.compile(r"\b\d+\b")
POSTAL_REGEX = re.compile(r"\b[1-9]\d{4,5}\b")

# Load model
model = joblib.load(PROJECT_ROOT / "code" / "business_entity_resolution" / "checkpoints" / "lgbm_matcher.joblib")

# Load 0.709 results
res_df = pd.read_csv(PROJECT_ROOT / "matching_results_0.709.tsv", sep="\t")
empty_s1 = set(res_df[res_df["matched_entity_ids"].isna() | (res_df["matched_entity_ids"].str.strip() == "")]["source1_entity_id"])
print(f"Total empty S1 in 0.709: {len(empty_s1):,}")

# Take a sample of 2,000 empty S1
sample_empty = list(empty_s1)[:2000]
sample_set = set(sample_empty)

# Load S1 records
s1_records = {}
test_dir = Path("/home/khushu/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/test")
with open(test_dir / "test_source1.tsv", "r", encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        p = line.strip().split("\t")
        if len(p) >= 4 and p[0] in sample_set:
            s1_records[p[0]] = (p[1], p[2], p[3])

# Load candidate pairs
cand_pairs = {}
all_cand_ids = set()
with open(PROJECT_ROOT / "output" / "candidate_pairs.tsv", "r", encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        p = line.strip().split("\t")
        if len(p) >= 2 and p[0] in sample_set:
            cands = p[1].split(",") if p[1] else []
            cand_pairs[p[0]] = cands
            all_cand_ids.update(cands)

# Load candidate records
cand_records = {}
for sf in ["test_source2.tsv", "test_source3.tsv"]:
    with open(test_dir / sf, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 4 and p[0] in all_cand_ids:
                cand_records[p[0]] = (p[1], p[2], p[3])

unblocked = []
for s1_id in sample_empty:
    cands = cand_pairs.get(s1_id, [])
    if not cands:
        continue
    n1, a1, c1 = s1_records.get(s1_id, ("", "", ""))
    nums1_int = {int(x) for x in NUM_REGEX.findall(a1)}
    
    for cand_id in cands:
        n2, a2, c2 = cand_records.get(cand_id, ("", "", ""))
        feats = compute_pair_features(n1, a1, c1, n2, a2, c2)
        X = pd.DataFrame([feats])[FEATURE_COLUMNS]
        prob = model.predict_proba(X)[0, 1]

        nums2_int = {int(x) for x in NUM_REGEX.findall(a2)}
        int_num_conflict = 1.0 if (nums1_int and nums2_int and not (nums1_int & nums2_int)) else 0.0

        # Ultra-safe criteria
        if (
            prob >= 0.90
            and feats["pin_conflict"] == 0.0
            and int_num_conflict == 0.0
            and feats["core_name_ratio"] >= 0.85
            and feats["name_token_set"] >= 0.90
            and feats["same_country"] == 1.0
        ):
            unblocked.append((s1_id, cand_id, prob, n1, a1, n2, a2))

print(f"\nUnblocked {len(unblocked)} high-confidence matches out of 2,000 sampled empty entities ({len(unblocked)/2000*100:.2f}%):")
for ex in unblocked[:10]:
    print(f"\n[Prob: {ex[2]:.4f}] S1: {ex[0]} -> Cand: {ex[1]}")
    print(f"  S1:   {ex[3]} | {ex[4]}")
    print(f"  Cand: {ex[5]} | {ex[6]}")
