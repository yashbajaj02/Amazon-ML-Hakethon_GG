import unittest
import sys
import subprocess
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution"))

from src.pipeline.run_pipeline import save_candidates_tsv, save_matches_tsv

class TestPhase5SubmissionValidation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = PROJECT_ROOT / "tests" / "mock_test_dir"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = PROJECT_ROOT / "tests" / "mock_output"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Create mock test source files required by validate_submission.py
        s1_df = pd.DataFrame([
            {"entity_id": "S1-0001", "business_name": "Acme Tools Inc", "business_address": "100 Industrial Pkwy", "country": "US"},
            {"entity_id": "S1-0002", "business_name": "Tata Motors Ltd", "business_address": "Bombay House Mumbai", "country": "India"},
            {"entity_id": "S1-0003", "business_name": "Paul Boulangerie", "business_address": "12 Rue Rivoli", "country": "France"},
        ])
        s1_df.to_csv(self.tmp_dir / "test_source1.tsv", sep="\t", index=False)

        s2_df = pd.DataFrame([
            {"entity_id": "S2-0001", "business_name": "Acme Tools Incorporated", "business_address": "100 Industrial Pkwy", "country": "US"},
            {"entity_id": "S2-0002", "business_name": "Tata Motors Limited", "business_address": "Bombay House Mumbai", "country": "India"},
        ])
        s2_df.to_csv(self.tmp_dir / "test_source2.tsv", sep="\t", index=False)

        s3_df = pd.DataFrame([
            {"entity_id": "S3-0003", "business_name": "Paul Boulangerie SAS", "business_address": "12 Rue Rivoli Paris", "country": "France"},
        ])
        s3_df.to_csv(self.tmp_dir / "test_source3.tsv", sep="\t", index=False)

    def test_submission_validator_pass(self):
        """Generated candidate and matching TSVs must strictly pass validate_submission.py (exit code 0)."""
        candidates = {
            "S1-0001": ["S2-0001"],
            "S1-0002": ["S2-0002"],
            "S1-0003": ["S3-0003"],
        }
        matches = {
            "S1-0001": ["S2-0001"],
            "S1-0002": ["S2-0002"],
            "S1-0003": [],  # Singleton prediction
        }

        cand_path = self.out_dir / "candidate_pairs.tsv"
        match_path = self.out_dir / "matching_results.tsv"

        save_candidates_tsv(candidates, cand_path)
        save_matches_tsv(matches, match_path)

        # Run official validator
        validator_script = PROJECT_ROOT / "utils" / "validate_submission.py"
        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(match_path),
            "--candidate", str(cand_path),
            "--test-dir", str(self.tmp_dir),
            "--check-ids",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode, 0,
            f"Validator failed with exit code {result.returncode}:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        self.assertIn("PASS", result.stdout)

    def tearDown(self):
        # Cleanup mock dirs
        import shutil
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)

if __name__ == "__main__":
    unittest.main()
