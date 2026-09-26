from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..evaluation.metrics import compute_macro_f05

FEATURE_COLUMNS = [
    "name_ratio",
    "name_partial",
    "name_token_sort",
    "name_token_set",
    "name_jaccard",
    "addr_ratio",
    "addr_partial",
    "addr_token_sort",
    "addr_token_set",
    "addr_jaccard",
    "len_diff_name",
    "len_diff_addr",
    "exact_name",
    "exact_addr",
    "same_country",
    "num_conflict",
    "num_exact",
    "num_overlap",
    "branch_mismatch",
]


class EntityMatcher:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train LightGBM binary classifier on pairwise features."""
        features = X[FEATURE_COLUMNS]
        self.model.fit(features, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns positive class match probability."""
        features = X[FEATURE_COLUMNS]
        return self.model.predict_proba(features)[:, 1]

    def optimize_threshold(
        self,
        val_df: pd.DataFrame,
        val_probs: np.ndarray,
        ground_truth: Dict[str, List[str]],
        thresholds: Optional[List[float]] = None,
    ) -> float:
        """
        Finds the probability threshold that maximizes Macro F_0.5 score on validation data.
        """
        if thresholds is None:
            thresholds = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

        best_score = -1.0
        best_thresh = self.threshold

        val_df = val_df.copy()
        val_df["prob"] = val_probs

        for thresh in thresholds:
            matched = val_df[val_df["prob"] >= thresh]
            preds: Dict[str, List[str]] = {s1: [] for s1 in ground_truth}
            for _, row in matched.iterrows():
                s1_id = row["source1_entity_id"]
                cand_id = row["candidate_entity_id"]
                if s1_id in preds:
                    preds[s1_id].append(cand_id)

            # Ensure uniqueness
            for s1 in preds:
                preds[s1] = list(dict.fromkeys(preds[s1]))

            score = compute_macro_f05(preds, ground_truth)
            if score > best_score:
                best_score = score
                best_thresh = thresh

        self.threshold = best_thresh
        return best_thresh
