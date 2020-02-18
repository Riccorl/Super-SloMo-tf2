from pathlib import Path

TRAIN_DIR = (
    Path(__file__).resolve().parent.parent / "data/preprocessed/train"
)
LOG_DIR = Path(__file__).resolve().parent.parent / "log"

REC_LOSS = 0.1
PERCEP_LOSS = 1.0
WRAP_LOSS = 1.0
SMOOTH_LOSS = 50.0
