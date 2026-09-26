import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution"))

from src.blocking.candidate_generator import CandidateGenerator
from src.data.preprocessor import preprocess_dataframe

class TestPhase2Blocking(unittest.TestCase):
    def setUp(self):
        # Create synthetic test dataset covering multi-country, singletons, and noise
        self.s1_data = pd.DataFrame([
            {"entity_id": "S1-001", "business_name": "Acme Tools Inc", "business_address": "100 Industrial Pkwy", "country": "US"},
            {"entity_id": "S1-002", "business_name": "Tata Motors Ltd", "business_address": "Bombay House 24 Homi Mody St", "country": "India"},
            {"entity_id": "S1-003", "business_name": "Boulangerie Paul", "business_address": "12 Rue de Rivoli Paris", "country": "France"}, # Unseen country
            {"entity_id": "S1-004", "business_name": "Isolated Singleton Entity", "business_address": "Nowhere Road 99", "country": "US"},   # Singleton
        ])

        self.targets_data = pd.DataFrame([
            {"entity_id": "S2-001", "business_name": "Acme Tools Incorporated", "business_address": "100 Industrial Parkway", "country": "US"},
            {"entity_id": "S3-001", "business_name": "Acme Tooling", "business_address": "100 Industrial Pkwy", "country": "US"},
            {"entity_id": "S2-002", "business_name": "Tata Motors Limited", "business_address": "Bombay House Homi Mody Street Mumbai", "country": "India"},
            {"entity_id": "S3-003", "business_name": "Paul Boulangerie SAS", "business_address": "12 Rue de Rivoli Paris France", "country": "France"},
            {"entity_id": "S2-999", "business_name": "Completely Unrelated Shop", "business_address": "Random Street 1", "country": "US"},
        ])

        self.s1_prep = preprocess_dataframe(self.s1_data)
        self.targets_prep = preprocess_dataframe(self.targets_data)

    def test_unseen_country_handling(self):
        """Ensure France (test country not in train) is blocked and candidates generated properly."""
        generator = CandidateGenerator(top_k=5, min_sim=0.10)
        candidates = generator.generate_candidates_by_country(self.s1_prep, self.targets_prep)
        
        self.assertIn("S1-003", candidates)
        self.assertIn("S3-003", candidates["S1-003"])

    def test_singleton_handling(self):
        """Entities with no similar targets should produce an empty list, not fail."""
        generator = CandidateGenerator(top_k=5, min_sim=0.30)
        candidates = generator.generate_candidates_by_country(self.s1_prep, self.targets_prep)
        
        self.assertIn("S1-004", candidates)
        self.assertEqual(candidates["S1-004"], [])

    def test_no_duplicate_candidate_ids(self):
        """Candidate IDs for any S1 entity must be strictly unique."""
        generator = CandidateGenerator(top_k=10, min_sim=0.05)
        candidates = generator.generate_candidates_by_country(self.s1_prep, self.targets_prep)
        
        for s1_id, cands in candidates.items():
            self.assertEqual(len(cands), len(set(cands)), f"Duplicates found for {s1_id}: {cands}")

    def test_valid_target_prefixes_only(self):
        """Candidate IDs must only come from S2- or S3-, never S1-."""
        generator = CandidateGenerator(top_k=5, min_sim=0.10)
        candidates = generator.generate_candidates_by_country(self.s1_prep, self.targets_prep)
        
        for s1_id, cands in candidates.items():
            for c in cands:
                self.assertTrue(c.startswith("S2-") or c.startswith("S3-"), f"Invalid candidate ID: {c}")

    def test_all_s1_entities_present(self):
        """Every S1 entity must be present as a key in the candidates dictionary."""
        generator = CandidateGenerator(top_k=5, min_sim=0.10)
        candidates = generator.generate_candidates_by_country(self.s1_prep, self.targets_prep)
        
        for s1_id in self.s1_prep["entity_id"]:
            self.assertIn(s1_id, candidates)

if __name__ == "__main__":
    unittest.main()
