import unittest
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution"))

from src.evaluation.metrics import compute_entity_f05, compute_macro_f05
from src.data.preprocessor import clean_business_name, clean_address, normalize_text, preprocess_dataframe

class TestPhase1Core(unittest.TestCase):
    def test_singleton_correct(self):
        """Singletons with empty predictions should score 1.0."""
        score = compute_entity_f05([], [])
        self.assertEqual(score, 1.0)

    def test_singleton_false_positive(self):
        """Singletons with any predicted match should score 0.0."""
        score = compute_entity_f05(["S2-001"], [])
        self.assertEqual(score, 0.0)

    def test_non_singleton_missed(self):
        """Non-singletons with no predicted matches should score 0.0."""
        score = compute_entity_f05([], ["S2-001"])
        self.assertEqual(score, 0.0)

    def test_exact_match(self):
        """Perfect prediction should score 1.0."""
        score = compute_entity_f05(["S2-001", "S3-002"], ["S2-001", "S3-002"])
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_official_example(self):
        """Test against the official competition example in README."""
        # Pred: [S2-00047, S2-00193, S3-00812], GT: [S2-00047, S3-00812]
        # P = 2/3, R = 1.0 -> F_0.5 = (1.25 * 2/3 * 1.0) / (0.25 * 2/3 + 1.0) = 0.7142857...
        score = compute_entity_f05(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
        self.assertAlmostEqual(score, 5.0 / 7.0, places=5)

    def test_macro_average(self):
        """Macro average across 3 entities (1 singleton, 1 perfect, 1 partial)."""
        gt = {
            "S1-1": [],
            "S1-2": ["S2-1"],
            "S1-3": ["S2-2", "S3-3"],
        }
        pred = {
            "S1-1": [],               # 1.0
            "S1-2": ["S2-1"],          # 1.0
            "S1-3": ["S2-2"],          # P=1.0, R=0.5 -> F0.5 = (1.25*1*0.5)/(0.25*1+0.5) = 0.625 / 0.75 = 0.83333
        }
        expected_macro = (1.0 + 1.0 + (5.0 / 6.0)) / 3.0
        score = compute_macro_f05(pred, gt)
        self.assertAlmostEqual(score, expected_macro, places=5)

    def test_text_normalization(self):
        """Verify business name and address cleaning rules."""
        raw_name = "Starbucks Corp. & Co."
        cleaned = clean_business_name(raw_name)
        self.assertIn("corporation", cleaned)
        self.assertIn("and", cleaned)
        self.assertIn("company", cleaned)

        raw_addr = "123 Main St., Suite 400"
        cleaned_addr = clean_address(raw_addr)
        self.assertIn("street", cleaned_addr)
        self.assertIn("suite", cleaned_addr)

    def test_rule_compliance_disallowed_imports(self):
        """Ensure no external lookup or geo libraries are referenced in code."""
        code_dir = PROJECT_ROOT / "code" / "business_entity_resolution"
        disallowed = ["geopy", "libpostal", "requests", "urllib", "googlemaps", "openai"]
        for py_file in code_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for term in disallowed:
                self.assertNotIn(f"import {term}", content, f"Disallowed package {term} found in {py_file}")

if __name__ == "__main__":
    unittest.main()
