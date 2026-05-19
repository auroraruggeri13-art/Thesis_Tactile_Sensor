"""
config.py — Centralized configuration for the Tactile Sensor project.

To change the data root for a new machine, set the environment variable:
    Windows:  set TACTILE_DATA_DIR=C:\\path\\to\\your\\data
    Linux:    export TACTILE_DATA_DIR=/path/to/your/data

All scripts should import from here instead of hard-coding paths.
"""

import os
from pathlib import Path

# ============================================================
# Base Paths — override via environment variable
# ============================================================
_DEFAULT_DATA_ROOT = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab")

BASE_DATA_DIR = Path(os.environ.get("TACTILE_DATA_DIR", _DEFAULT_DATA_ROOT))

RAW_DATA_DIR        = BASE_DATA_DIR / "test data"
TRAIN_VAL_TEST_DIR  = BASE_DATA_DIR / "train_validation_test_data"
MODEL_DIR           = BASE_DATA_DIR / "models parameters" / "averaged models"
COMSOL_DIR          = BASE_DATA_DIR / "COMSOL_plots"
HYSTERESIS_DIR      = BASE_DATA_DIR / "hysteresys"

# ============================================================
# Sensor & Data Versioning
# ============================================================
SENSOR_VERSION = "5.200"   # used in filenames: train_data_v{SENSOR_VERSION}.csv

# ============================================================
# Barometer Processing
# ============================================================
WARMUP_SECONDS      = 2.0      # seconds to discard at recording start
EMA_ALPHA           = 0.0001   # smaller = slower/smoother drift removal
ASOF_TOLERANCE_S    = 0.05     # merge_asof time tolerance in seconds
MAD_THRESHOLD       = 10.0     # median absolute deviation outlier threshold
DENOISE_WINDOW      = 5        # rolling-mean window for optional denoising

# Drift removal: "ema" | "temperature" | "both" | "none"
DRIFT_REMOVAL_METHOD = "ema"

# ============================================================
# Sliding-Window Feature Engineering
# ============================================================
WINDOW_SIZE           = 10
USE_FIRST_DERIVATIVE  = True
USE_SECOND_DERIVATIVE = True

# ============================================================
# Train / Validation / Test Split
# ============================================================
TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15
TEST_FRACTION  = 0.15
RANDOM_STATE   = 42

# ============================================================
# Real-Time Inference
# ============================================================
SERIAL_BAUD_RATE  = 115200
WARMUP_SAMPLES    = 100      # discard first N samples to stabilise
CONTACT_THRESHOLD = 0.05     # Fz threshold (N) for contact detection

# ============================================================
# Spatial Filter (keep only contacts inside sensor area, mm)
# ============================================================
SPATIAL_FILTER = {
    "x_range": (-25.0, 25.0),
    "y_range": (-25.0, 25.0),
}
