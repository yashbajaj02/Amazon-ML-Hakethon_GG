import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

NUM_REGEX = re.compile(r"\b\d+\b")
BRANCH_KEYWORDS = {
    "central", "west", "east", "north", "south", "branch",
    "services", "capital", "holdings", "group", "international", "global"
}


def compute_pair_features(
    name1: str,
    addr1: str,
    country1: str,
    name2: str,
    addr2: str,
    country2: str,
) -> Dict[str, float]:
    """
    Computes a vector of similarity metrics between two business records.
    """
    # Name features
    name_ratio = fuzz.ratio(name1, name2) / 100.0
    name_partial = fuzz.partial_ratio(name1, name2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
    name_token_set = fuzz.token_set_ratio(name1, name2) / 100.0

    # Address features
    addr_ratio = fuzz.ratio(addr1, addr2) / 100.0
    addr_partial = fuzz.partial_ratio(addr1, addr2) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    addr_token_set = fuzz.token_set_ratio(addr1, addr2) / 100.0

    # Token overlap (Jaccard)
    tokens1 = set(name1.split())
    tokens2 = set(name2.split())
    name_jaccard = (
        len(tokens1 & tokens2) / len(tokens1 | tokens2)
        if (tokens1 or tokens2)
        else 0.0
    )

    addr_tok1 = set(addr1.split())
    addr_tok2 = set(addr2.split())
    addr_jaccard = (
        len(addr_tok1 & addr_tok2) / len(addr_tok1 | addr_tok2)
        if (addr_tok1 or addr_tok2)
        else 0.0
    )

    # Length features
    len_diff_name = abs(len(name1) - len(name2)) / max(len(name1) + len(name2), 1)
    len_diff_addr = abs(len(addr1) - len(addr2)) / max(len(addr1) + len(addr2), 1)

    # Exact matches
    exact_name = 1.0 if name1 and (name1 == name2) else 0.0
    exact_addr = 1.0 if addr1 and (addr1 == addr2) else 0.0
    same_country = 1.0 if country1 == country2 else 0.0

    # Numerical building/street numbers
    nums1 = set(NUM_REGEX.findall(addr1))
    nums2 = set(NUM_REGEX.findall(addr2))
    num_conflict = 1.0 if (nums1 and nums2 and not (nums1 & nums2)) else 0.0
    num_exact = 1.0 if (nums1 and nums2 and nums1 == nums2) else 0.0
    num_overlap = 1.0 if (nums1 and nums2 and (nums1 & nums2)) else 0.0

    # Branch / corporate division mismatch
    b1 = tokens1 & BRANCH_KEYWORDS
    b2 = tokens2 & BRANCH_KEYWORDS
    branch_mismatch = 1.0 if (b1 ^ b2) else 0.0

    return {
        "name_ratio": name_ratio,
        "name_partial": name_partial,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_jaccard": name_jaccard,
        "addr_ratio": addr_ratio,
        "addr_partial": addr_partial,
        "addr_token_sort": addr_token_sort,
        "addr_token_set": addr_token_set,
        "addr_jaccard": addr_jaccard,
        "len_diff_name": len_diff_name,
        "len_diff_addr": len_diff_addr,
        "exact_name": exact_name,
        "exact_addr": exact_addr,
        "same_country": same_country,
        "num_conflict": num_conflict,
        "num_exact": num_exact,
        "num_overlap": num_overlap,
        "branch_mismatch": branch_mismatch,
    }


def extract_features_for_pairs(
    candidate_pairs: List[Tuple[str, str]],
    records_s1: Dict[str, Tuple[str, str, str]],
    records_target: Dict[str, Tuple[str, str, str]],
) -> pd.DataFrame:
    """
    Extracts features for all candidate pairs (s1_id, target_id).
    """
    rows = []
    for s1_id, target_id in candidate_pairs:
        name1, addr1, c1 = records_s1.get(s1_id, ("", "", ""))
        name2, addr2, c2 = records_target.get(target_id, ("", "", ""))

        feats = compute_pair_features(name1, addr1, c1, name2, addr2, c2)
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = target_id
        rows.append(feats)

    return pd.DataFrame(rows)
