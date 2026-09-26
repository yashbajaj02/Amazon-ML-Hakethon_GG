import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution"))

from src.features.feature_extractor import compute_pair_features, extract_features_for_pairs

class TestPhase3Features(unittest.TestCase):
    def test_identical_records(self):
        """Identical records should have 1.0 on all similarity metrics."""
        name = "Google India Pvt Ltd"
        addr = "RMZ Infinity Old Madras Road Bangalore"
        country = "INDIA"
        
        feats = compute_pair_features(name, addr, country, name, addr, country)
        self.assertEqual(feats["name_ratio"], 1.0)
        self.assertEqual(feats["addr_ratio"], 1.0)
        self.assertEqual(feats["exact_name"], 1.0)
        self.assertEqual(feats["exact_addr"], 1.0)
        self.assertEqual(feats["same_country"], 1.0)
        self.assertEqual(feats["name_jaccard"], 1.0)
        self.assertEqual(feats["addr_jaccard"], 1.0)
        self.assertEqual(feats["len_diff_name"], 0.0)
        self.assertEqual(feats["len_diff_addr"], 0.0)

    def test_completely_different_records(self):
        """Completely unrelated records should have low similarities and zero exact matches."""
        feats = compute_pair_features(
            "Apple Inc", "One Infinite Loop Cupertino", "US",
            "State Bank of India", "Nariman Point Mumbai", "INDIA"
        )
        self.assertEqual(feats["exact_name"], 0.0)
        self.assertEqual(feats["exact_addr"], 0.0)
        self.assertEqual(feats["same_country"], 0.0)
        self.assertLess(feats["name_ratio"], 0.40)
        self.assertLess(feats["addr_ratio"], 0.45)
        self.assertEqual(feats["name_jaccard"], 0.0)
        self.assertEqual(feats["addr_jaccard"], 0.0)

    def test_empty_string_handling(self):
        """Empty strings should not cause DivisionByZero or exceptions."""
        feats = compute_pair_features("", "", "US", "", "", "US")
        for k, v in feats.items():
            self.assertFalse(np.isnan(v), f"Feature {k} returned NaN")
            self.assertFalse(np.isinf(v), f"Feature {k} returned Inf")

    def test_feature_bounds(self):
        """All features should be strictly within [0.0, 1.0]."""
        feats = compute_pair_features("Test A", "Address 1", "US", "Test B", "Address 2", "US")
        for k, v in feats.items():
            self.assertTrue(0.0 <= v <= 1.0, f"Feature {k} out of range [0.0, 1.0]: {v}")

    def test_batch_feature_extractor(self):
        """extract_features_for_pairs should produce properly formatted DataFrame."""
        pairs = [("S1-1", "S2-1"), ("S1-1", "S3-2")]
        records_s1 = {"S1-1": ("Shop A", "Street 1", "US")}
        records_target = {
            "S2-1": ("Shop A LLC", "Street 1", "US"),
            "S3-2": ("Shop B", "Street 2", "US"),
        }
        df = extract_features_for_pairs(pairs, records_s1, records_target)
        self.assertEqual(len(df), 2)
        self.assertIn("source1_entity_id", df.columns)
        self.assertIn("candidate_entity_id", df.columns)
        self.assertIn("name_token_set", df.columns)

if __name__ == "__main__":
    unittest.main()
