from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CODE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = CODE_DIR / "src"

# Data Paths
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / "output"
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Model Checkpoints
CHECKPOINT_DIR = CODE_DIR / "checkpoints"

# Config Parameters
RANDOM_SEED = 42
BLOCKING_TOP_K = 12  # Number of candidates per S1 entity
MODEL_THRESHOLD = 0.65  # Tuned for F_0.5 (precision-heavy)
