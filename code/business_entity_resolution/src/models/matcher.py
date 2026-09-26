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
    "core_name_ratio",
    "core_exact",
    "first_word_match",
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
    "pin_conflict",
    "pin_match",
    "branch_mismatch",
]


class EntityMatcher:
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold
        self.model = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=63,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=25,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train high-capacity LightGBM binary classifier on pairwise features."""
        features = X[FEATURE_COLUMNS]
        self.model.fit(features, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts positive class matching probability."""
        features = X[FEATURE_COLUMNS]
        return self.model.predict_proba(features)[:, 1]

    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        probs: np.ndarray,
        ground_truth: Dict[str, List[str]],
        thresholds: List[float] = [0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85],
    ) -> float:
        """
        Grid searches optimal probability decision threshold directly targeting Macro F0.5.
        """
        best_f05 = -1.0
        best_thresh = self.threshold

        s1_entities = list(ground_truth.keys())
        X_eval = X_val.copy()
        X_eval["prob"] = probs

        for thresh in thresholds:
            matched_pairs = X_eval[X_eval["prob"] >= thresh]
            preds = {s1_id: [] for s1_id in s1_entities}
            for _, row in matched_pairs.iterrows():
                preds[row["source1_entity_id"]].append(row["candidate_entity_id"])

            for s1_id in preds:
                preds[s1_id] = list(dict.fromkeys(preds[s1_id]))

            f05 = compute_macro_f05(preds, ground_truth)
            empty_count = sum(1 for v in preds.values() if len(v) == 0)
            print(f"  [Threshold Search] Thresh={thresh:.2f} -> Macro F0.5={f05:.4f} (Singletons: {empty_count:,}/{len(s1_entities):,})")

            if f05 > best_f05:
                best_f05 = f05
                best_thresh = thresh

        self.threshold = best_thresh
        return best_thresh
