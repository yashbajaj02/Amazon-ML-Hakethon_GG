import sys
import time
from pathlib import Path
import re
import pandas as pd
import joblib
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution"))

from src.models.matcher import FEATURE_COLUMNS
from src.features.feature_extractor import compute_pair_features

NUM_REGEX = re.compile(r"\b\d+\b")
POSTAL_REGEX = re.compile(r"\b[1-9]\d{4,5}\b")
LEGAL_STRIP = re.compile(
    r"\b(corporation|incorporated|limited|private|llc|llp|company|sarl|sas|sasu|eurl|sci|sa|corp|inc|ltd|pvt|co)\b",
    flags=re.IGNORECASE,
)
BRANCH_KEYWORDS = {
    "central", "west", "east", "north", "south", "branch",
    "services", "capital", "holdings", "group", "international", "global"
}

def main():
    print("=" * 60)
    print("ULTRA-SAFE HIGH-PRECISION UNBLOCKING PASS ON EMPTY ENTITIES")
    print("=" * 60)
    start_time = time.time()

    # 1. Load baseline results
    print("\n[Step 1] Loading baseline 0.709 results...")
    base_file = PROJECT_ROOT / "matching_results_0.709.tsv"
    base_df = pd.read_csv(base_file, sep="\t", keep_default_na=False)
    
    matches_dict = {}
    empty_s1 = set()
    for sid, mstr in zip(base_df["source1_entity_id"], base_df["matched_entity_ids"]):
        if mstr.strip():
            matches_dict[sid] = [x.strip() for x in mstr.split(",") if x.strip()]
        else:
            matches_dict[sid] = []
            empty_s1.add(sid)

    print(f"Total S1 entities: {len(base_df):,}")
    print(f"Non-empty entities (preserved 100%): {len(matches_dict) - len(empty_s1):,}")
    print(f"Empty entities to evaluate: {len(empty_s1):,}")

    # 2. Load candidates for empty entities
    print("\n[Step 2] Loading candidates for empty entities from candidate_pairs.tsv...")
    cand_dict = {}
    needed_cands = set()
    with open(PROJECT_ROOT / "output" / "candidate_pairs.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0] in empty_s1:
                cands = [c.strip() for c in p[1].split(",") if c.strip()]
                if cands:
                    cand_dict[p[0]] = cands
                    needed_cands.update(cands)

    print(f"Empty entities with candidates: {len(cand_dict):,}")
    print(f"Unique candidate target records needed: {len(needed_cands):,}")

    # 3. Load text records
    print("\n[Step 3] Loading record strings from dataset...")
    test_dir = Path("/home/khushu/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/test")

    s1_records = {}
    with open(test_dir / "test_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4 and p[0] in empty_s1:
                s1_records[p[0]] = (p[1], p[2], p[3])
    print(f"  Loaded {len(s1_records):,} Source-1 records.")

    target_records = {}
    for sf in ["test_source2.tsv", "test_source3.tsv"]:
        with open(test_dir / sf, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4 and p[0] in needed_cands:
                    target_records[p[0]] = (p[1], p[2], p[3])
    print(f"  Loaded {len(target_records):,} Target records (Source 2/3).")

    # 4. Load trained model
    print("\n[Step 4] Loading LightGBM model...")
    model = joblib.load(PROJECT_ROOT / "code" / "business_entity_resolution" / "checkpoints" / "lgbm_matcher.joblib")

    # 5. Fast pre-filter and high-precision evaluation
    print("\n[Step 5] Screening candidate pairs with ultra-strict criteria...")
    candidate_pool = []
    
    for s1_id, cands in cand_dict.items():
        n1, a1, c1 = s1_records.get(s1_id, ("", "", ""))
        if not n1:
            continue
        c1_up = c1.strip().upper()
        core1 = LEGAL_STRIP.sub(" ", n1).strip()
        tok1 = set(n1.lower().split())
        b1 = tok1 & BRANCH_KEYWORDS
        nums1_int = {int(x) for x in NUM_REGEX.findall(a1)}
        pins1 = set(POSTAL_REGEX.findall(a1))

        for cand_id in cands:
            n2, a2, c2 = target_records.get(cand_id, ("", "", ""))
            if not n2:
                continue

            # Check 1: Same country
            if c1_up != c2.strip().upper():
                continue

            # Check 2: Branch keywords must match exactly
            tok2 = set(n2.lower().split())
            b2 = tok2 & BRANCH_KEYWORDS
            if b1 != b2:
                continue

            # Check 3: Token set ratio >= 88
            if fuzz.token_set_ratio(n1, n2) < 88:
                continue

            # Check 4: Core name ratio >= 85
            core2 = LEGAL_STRIP.sub(" ", n2).strip()
            if fuzz.ratio(core1, core2) < 85:
                continue

            # Check 5: Integer number conflict check
            nums2_int = {int(x) for x in NUM_REGEX.findall(a2)}
            if nums1_int and nums2_int and not (nums1_int & nums2_int):
                continue

            # Check 6: PIN conflict check
            pins2 = set(POSTAL_REGEX.findall(a2))
            if pins1 and pins2 and not (pins1 & pins2):
                continue

            candidate_pool.append((s1_id, cand_id, n1, a1, c1, n2, a2, c2))

    print(f"Qualified candidate pairs for model inference: {len(candidate_pool):,}")

    # 6. Feature extraction & Model prediction in batches
    batch_size = 50000
    unblocked_count = 0
    new_matches = {}

    for i in range(0, len(candidate_pool), batch_size):
        batch = candidate_pool[i : i + batch_size]
        rows = []
        for s1_id, cand_id, n1, a1, c1, n2, a2, c2 in batch:
            fdict = compute_pair_features(n1, a1, c1, n2, a2, c2)
            fdict["s1_id"] = s1_id
            fdict["cand_id"] = cand_id
            rows.append(fdict)

        bdf = pd.DataFrame(rows)
        X = bdf[FEATURE_COLUMNS]
        probs = model.predict_proba(X)[:, 1]

        # Strictest threshold: P >= 0.90
        confident_mask = probs >= 0.90
        passed_df = bdf[confident_mask]

        for s1_m, cand_m in zip(passed_df["s1_id"], passed_df["cand_id"]):
            if s1_m not in new_matches:
                new_matches[s1_m] = []
            new_matches[s1_m].append(cand_m)

    # 7. Merge unblocked entities
    unblocked_entities = 0
    for s1_id, c_list in new_matches.items():
        if not matches_dict[s1_id]:  # double-check it was indeed empty
            matches_dict[s1_id] = list(dict.fromkeys(c_list))
            unblocked_entities += 1

    print(f"\nSuccessfully unblocked {unblocked_entities:,} authentic entities with P >= 0.90!")
    print(f"Remaining empty singletons: {len(empty_s1) - unblocked_entities:,} (protected full 1.0 score)")

    # 8. Save output
    out_path = PROJECT_ROOT / "matching_results.tsv"
    print(f"\n[Step 8] Writing updated matches to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, m_list in matches_dict.items():
            f.write(f"{s1_id}\t{','.join(m_list)}\n")

    print(f"Completed in {time.time()-start_time:.1f}s!")

if __name__ == "__main__":
    main()
