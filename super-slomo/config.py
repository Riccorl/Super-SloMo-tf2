from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VALID_DIR = DATA_DIR / "val"
LOG_DIR = Path(__file__).resolve().parent.parent / "log"

REC_LOSS = 0.1
PERCEP_LOSS = 1.0
WRAP_LOSS = 1.0
SMOOTH_LOSS = 50.0
