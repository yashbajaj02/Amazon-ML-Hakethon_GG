from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


def load_source_tsv(filepath: Path) -> pd.DataFrame:
    """
    Loads a source TSV file.
    Columns: entity_id, business_name, business_address, country
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype={
            "entity_id": str,
            "business_name": str,
            "business_address": str,
            "country": str,
        },
        keep_default_na=False,
    )
    return df


def load_ground_truth(filepath: Path) -> Dict[str, List[str]]:
    """
    Loads train_ground_truth.tsv.
    Columns: source1_entity_id, matched_entity_ids
    Returns a dictionary mapping S1 ID to list of matched S2/S3 IDs.
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype={"source1_entity_id": str, "matched_entity_ids": str},
        keep_default_na=False,
    )
    gt_dict = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        matches_raw = str(row["matched_entity_ids"]).strip()
        if matches_raw:
            gt_dict[s1_id] = [m.strip() for m in matches_raw.split(",") if m.strip()]
        else:
            gt_dict[s1_id] = []
    return gt_dict


def load_dataset_split(
    data_dir: Path, split_prefix: str = "train"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads source1, source2, and source3 dataframes for a given split (train or test).
    """
    s1_path = data_dir / f"{split_prefix}_source1.tsv"
    s2_path = data_dir / f"{split_prefix}_source2.tsv"
    s3_path = data_dir / f"{split_prefix}_source3.tsv"

    s1_df = load_source_tsv(s1_path)
    s2_df = load_source_tsv(s2_path)
    s3_df = load_source_tsv(s3_path)

    return s1_df, s2_df, s3_df
