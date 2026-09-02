"""Central configuration for the BEED robustness experiment."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(
    os.environ.get("BEED_DATA_PATH", PROJECT_ROOT / "data" / "BEED_Data_clean.csv")
)
OUTPUT_DIR = Path(os.environ.get("BEED_OUTPUT_DIR", PROJECT_ROOT / "results"))

RANDOM_STATE = 42
TEST_SIZE = 0.20

SCALER_TYPE = "robust"
USE_FEATURE_CLIPPING = True
CLIP_VALUE = 4.0

SNR_LEVELS = [30, 20, 10]
N_NOISE_REPEATS = 5
ROBUST_TRAIN_SNR_LEVELS = [30, 20, 10]
ROBUST_REPEAT_MAP = {30: 1, 20: 2, 10: 3}

MLP_EPOCHS = 150
MLP_BATCH_SIZE = 32
MLP_VALIDATION_SPLIT = 0.15

