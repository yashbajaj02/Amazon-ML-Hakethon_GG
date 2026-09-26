import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution"))

from src.models.matcher import EntityMatcher, FEATURE_COLUMNS
from src.evaluation.metrics import compute_macro_f05

class TestPhase4Matcher(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create synthetic training set with 200 pairs
        n = 200
        # Positive pairs: high similarity
        pos_data = {col: np.random.uniform(0.7, 1.0, size=n // 2) for col in FEATURE_COLUMNS}
        pos_df = pd.DataFrame(pos_data)
        pos_df["label"] = 1
        pos_df["source1_entity_id"] = [f"S1-{i:03d}" for i in range(n // 2)]
        pos_df["candidate_entity_id"] = [f"S2-{i:03d}" for i in range(n // 2)]

        # Negative pairs: low similarity
        neg_data = {col: np.random.uniform(0.0, 0.4, size=n // 2) for col in FEATURE_COLUMNS}
        neg_df = pd.DataFrame(neg_data)
        neg_df["label"] = 0
        neg_df["source1_entity_id"] = [f"S1-{i:03d}" for i in range(n // 2, n)]
        neg_df["candidate_entity_id"] = [f"S2-{i:03d}" for i in range(n // 2, n)]

        self.df = pd.concat([pos_df, neg_df], ignore_index=True)
        self.matcher = EntityMatcher()
        self.matcher.fit(self.df, self.df["label"].values)

    def test_predict_proba_bounds(self):
        """Probabilities must be strictly bounded in [0.0, 1.0]."""
        probs = self.matcher.predict_proba(self.df)
        self.assertEqual(len(probs), len(self.df))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_model_discrimination(self):
        """High similarity pairs should have significantly higher probabilities than low similarity pairs."""
        probs = self.matcher.predict_proba(self.df)
        pos_probs = probs[: len(probs) // 2]
        neg_probs = probs[len(probs) // 2 :]
        self.assertGreater(pos_probs.mean(), 0.70)
        self.assertLess(neg_probs.mean(), 0.30)

    def test_threshold_optimizer(self):
        """Threshold optimizer should select a threshold that achieves high F_0.5."""
        probs = self.matcher.predict_proba(self.df)
        ground_truth = {}
        for _, row in self.df.iterrows():
            s1 = row["source1_entity_id"]
            c = row["candidate_entity_id"]
            if row["label"] == 1:
                ground_truth.setdefault(s1, []).append(c)
            else:
                ground_truth.setdefault(s1, [])

        best_thresh = self.matcher.optimize_threshold(self.df, probs, ground_truth)
        self.assertTrue(0.0 < best_thresh < 1.0)
        self.assertEqual(self.matcher.threshold, best_thresh)

if __name__ == "__main__":
    unittest.main()
