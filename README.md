# Tactile Sensor — Master Thesis

A research project on **barometer-based and vision-based tactile sensing** for robotic contact-force estimation. The sensor embeds 6 DPS310 barometric pressure sensors in a silicone pad; ML models map barometer readings (or visual deformation) to 3-axis contact forces validated against an ATI F/T sensor.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Pipeline 1 — Barometer-Based Force Estimation](#pipeline-1--barometer-based-force-estimation)
- [Pipeline 2 — Vision-Based Force Estimation](#pipeline-2--vision-based-force-estimation)
- [Configuration & Hard-Coded Paths](#configuration--hard-coded-paths)
- [Data Organization](#data-organization)
- [Hardware Setup](#hardware-setup)
- [Model Comparison](#model-comparison)
- [Troubleshooting](#troubleshooting)
- [Utilities Reference](#utilities-reference)

---

## Project Overview

| Aspect | Detail |
|--------|--------|
| **Sensor** | 6× DPS310 barometers embedded in silicone pad |
| **Sampling rate** | 125 Hz (Arduino MKR Zero) |
| **Prediction targets** | Contact position (x, y) + 3-axis forces (Fx, Fy, Fz) |
| **Best model** | LightGBM with sliding-window features (200–500+ Hz inference) |
| **Ground truth** | ATI Industrial F/T sensor |
| **Tracking** | AprilTag or Atracsys optical tracking |

---

## Prerequisites

### Software

- **Python 3.10 or 3.11** — Python 3.12+ has TensorFlow compatibility issues
- **pip** — `python -m pip install --upgrade pip`
- *(Optional)* Arduino IDE 2.x — for flashing firmware

### Hardware *(only needed for data collection)*

- Arduino MKR Zero + 6× DPS310 pressure sensors (I2C)
- ATI F/T sensor (ground truth)
- AprilTag camera rig or Atracsys optical tracker

### For the Real-Time Demo

No ROS required. `realtime_demo.py` works standalone with just the Arduino connected.

---

## Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd "Thesis - Tactile Sensor"

# 2. Create a virtual environment
python -m venv TactileSensingMasterThesis

# 3. Activate it
#    Windows (PowerShell):
TactileSensingMasterThesis\Scripts\Activate.ps1
#    macOS / Linux:
source TactileSensingMasterThesis/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify
python -c "import pandas, numpy, lightgbm, torch; print('All core imports OK')"
```

> **GPU acceleration:** Install CUDA 11.8+ and cuDNN, then reinstall `torch` and `tensorflow` with their GPU-enabled variants (see official docs). Without GPU, all scripts fall back to CPU.

---

## Project Structure

```
Thesis - Tactile Sensor/
│
├── config.py                                         <- Centralized paths & constants
├── realtime_demo.py                                  <- EASIEST ENTRY POINT (no ROS)
├── requirements.txt
├── README.md
├── PIPELINE.md                                       <- Step-by-step execution guide
│
├── Barometers_Based_Tactile_Sensor/                  <- Part 1: barometer pipeline
│   ├── utils/
│   │   ├── barometer_processing.py                   # Signal processing
│   │   ├── sensor_io.py                              # Load ATI / tracking logs
│   │   ├── signal_utils.py                           # Derivatives, denoising
│   │   ├── metrics_utils.py                          # RMSE, grouped metrics
│   │   ├── plot_utils.py                             # Visualization helpers
│   │   └── io_utils.py                               # File I/O helpers
│   │
│   ├── Pose_Forces_Synchronized_with_Baro_Processing.py  # Steps 2-3
│   ├── train_validation_test_dataset_generation.py        # Step 4
│   ├── LightGBM_sliding_window_predictions.py             # Step 5 (recommended)
│   ├── Linear_Regression_force_predictions.py
│   ├── random_forest_force_prediction.py
│   ├── NN_force_prediction.py
│   ├── LSTM.py
│   ├── 1D_CNN_prediction.py
│   ├── transformer_sliding_window_predictions.py
│   └── realtime_tactile_prediction.py                     # Advanced real-time module
│
├── Vision_vs_Tactile/                                <- Part 2: vision pipeline
│   ├── direct_jpeg_extractor.py                      # Extract frames from ROS bag
│   ├── plot_models_predictions_from_csv.py
│   └── Codes_for_cluster/                            # HPC versions
│       ├── code/forces_for_free.py
│       └── Tactile_sensor_model/train_lightgbm.py
│
├── Arduino Codes & Barometers Logging & AprilTags Scripts/
│   ├── TactileSensorArduinoCode/TactileSensorCode.ino
│   ├── baro_serial_node.py
│   └── tagposes_from_detections.py
│
├── predictions_with_temperature/                     <- Temperature-aware variant
├── CAD_files/                                        <- Fusion360 + STL files
└── TactileSensingMasterThesis/                       <- Local venv (git-ignored)
```

---

## Pipeline 1 — Barometer-Based Force Estimation

> See [PIPELINE.md](PIPELINE.md) for a concise numbered execution checklist.

### Step 1 — Collect Raw Data

Connect the Arduino MKR Zero and start logging. See [Hardware Setup](#hardware-setup).

**Files produced per test session:**
```
test data/test <N> - sensor v<V>/
├── apriltag_detections_trial<N>.txt    # 3D tip pose from AprilTag
├── ati_middle_trial<N>.txt             # ATI force/torque log
└── barometers_trial<N>.txt             # Arduino 6-channel barometer CSV
```

### Step 2–3 — Synchronize Barometers with Forces

**Script:** `Barometers_Based_Tactile_Sensor/Pose_Forces_Synchronized_with_Baro_Processing.py`

Edit the `TEST_NUMS` list at the top, then run:

```bash
python Barometers_Based_Tactile_Sensor/Pose_Forces_Synchronized_with_Baro_Processing.py
```

What it does:
- Removes 2 s warmup; applies EMA drift correction
- Transforms AprilTag detections to tip position in sensor frame
- Time-aligns barometers + ATI via `merge_asof` (50 ms tolerance)
- Applies spatial mask (contacts inside sensor area only)
- MAD-based outlier removal

**Output:** `synchronized_events_<N>.csv` in each test folder + diagnostic plots

### Step 4 — Build Train / Validation / Test Splits

**Script:** `Barometers_Based_Tactile_Sensor/train_validation_test_dataset_generation.py`

Edit `CSV_FILENAMES` to list your synchronized CSVs, then:

```bash
python Barometers_Based_Tactile_Sensor/train_validation_test_dataset_generation.py
```

**Output:**
```
train_validation_test_data/
├── train_data_v<VERSION>.csv
├── validation_data_v<VERSION>.csv
└── test_data_v<VERSION>.csv
```

### Step 5 — Train a Model

All training scripts read from `train_validation_test_data/` and save to `models parameters/`.

**Recommended — LightGBM:**

```bash
python Barometers_Based_Tactile_Sensor/LightGBM_sliding_window_predictions.py
```

Configure at the top of the script: `WINDOW_SIZE`, `USE_SECOND_DERIVATIVE`, `SENSOR_VERSION`.

**Output:**
```
models parameters/averaged models/
├── lightgbm_sliding_window_model_v<VERSION>.pkl
└── scaler_sliding_window_v<VERSION>.pkl
```

### Step 6 — Real-Time Inference

**Simplest option (no ROS):**

```bash
python realtime_demo.py
# Or specify port explicitly:
python realtime_demo.py --port COM3
```

Update `_DEFAULT_DATA_ROOT` and `SENSOR_VERSION` in [config.py](config.py) before running.

**Expected output:**

```
========== Real-Time Tactile Sensor Demo ==========
Sample        X(mm)      Y(mm)     Fx(N)      Fy(N)      Fz(N)    Contact
   1234      -5.23       2.14      0.120      0.050      0.340       YES
```

---

## Pipeline 2 — Vision-Based Force Estimation

### Step 1 — Extract Frames from ROS Bag

```bash
python Vision_vs_Tactile/direct_jpeg_extractor.py
```

An interactive HSV color picker opens — tune the segmentation mask for the deformed region. The script then extracts JPEG frames and a synchronized CSV.

### Step 2 — Estimate Forces from Deformation

```bash
python Vision_vs_Tactile/Codes_for_cluster/code/forces_for_free.py
```

*(For HPC clusters: use the `.sh` scripts in `Vision_vs_Tactile/Codes_for_cluster/`)*

---

## Configuration & Hard-Coded Paths

**All scripts contain absolute paths that must be updated for a new machine.**

### Preferred approach — edit `config.py`

Open [config.py](config.py) and change `_DEFAULT_DATA_ROOT`:

```python
_DEFAULT_DATA_ROOT = Path(r"C:\Users\yourname\data\Thesis- Biorobotics Lab")
```

Or set an environment variable so you never touch the file:

```bash
# Windows
set TACTILE_DATA_DIR=C:\Users\yourname\data\Thesis- Biorobotics Lab

# Linux / macOS
export TACTILE_DATA_DIR=/home/yourname/data/Thesis-Biorobotics-Lab
```

### Key variables in config.py

| Variable | Description | Default |
|----------|-------------|---------|
| `BASE_DATA_DIR` | Root of all data | OneDrive path |
| `SENSOR_VERSION` | Model/data version tag | `"5.200"` |
| `WINDOW_SIZE` | Sliding-window length | `10` |
| `DRIFT_REMOVAL_METHOD` | `"ema"` / `"temperature"` / `"both"` / `"none"` | `"ema"` |
| `EMA_ALPHA` | EMA smoothing factor | `0.0001` |

> Individual scripts may still override these locally while migration is in progress. Script-level values take precedence until you update each script to `from config import ...`.

---

## Data Organization

```
BASE_DATA_DIR/
├── test data/
│   ├── test 52000 - sensor v5/
│   │   ├── apriltag_detections_trial52000.txt
│   │   ├── ati_middle_trial52000.txt
│   │   ├── barometers_trial52000.txt
│   │   ├── processing_test_52000.csv         <- Step 2 output
│   │   └── synchronized_events_52000.csv      <- Step 3 output
│   └── test 52001 - sensor v5/  ...
│
├── train_validation_test_data/
│   ├── train_data_v5.20.csv
│   ├── validation_data_v5.20.csv
│   └── test_data_v5.20.csv
│
└── models parameters/
    └── averaged models/
        ├── lightgbm_sliding_window_model_v5.200.pkl
        └── scaler_sliding_window_v5.200.pkl
```

### File formats

| File | Key Columns |
|------|-------------|
| `barometers_trial<N>.txt` | `Epoch_s, b1_P, b1_T, b2_P, b2_T, ..., b6_P, b6_T` |
| `ati_middle_trial<N>.txt` | `timestamp, Fx, Fy, Fz, Tx, Ty, Tz` |
| `synchronized_events_<N>.csv` | `t, b1..b6, x, y, z, fx, fy, fz` |
| `train_data_v<V>.csv` | `t, b1..b6, x, y, fx, fy, fz` |

---

## Hardware Setup

### Flash Arduino Firmware

1. Open **Arduino IDE 2.x**
2. Board: **Arduino MKR Zero** (Tools > Board > MKR Zero)
3. Open `Arduino Codes & Barometers Logging & AprilTags Scripts/TactileSensorArduinoCode/TactileSensorCode.ino`
4. Click **Upload** (Ctrl+U)

**Serial output at 115200 baud:**
```
Time(ms), B1_P, B1_T, B2_P, B2_T, B3_P, B3_T, B4_P, B4_T, B5_P, B5_T, B6_P, B6_T
0, 101325.2, 22.5, ...
```

**I2C wiring:** SDA -> Pin 11, SCL -> Pin 12, 3.3 V, GND

### Sensor Specs

| Sensor | Part | Range | Rate |
|--------|------|-------|------|
| Barometer | DPS310 | 300-1200 hPa | 125 Hz |
| F/T (ground truth) | ATI Nano17 | +/-70 N, +/-2 N*m | 1 kHz |

---

## Model Comparison

| Model | Script | Approx. R² (Fz) | Inference Hz | Notes |
|-------|--------|----------------|-------------|-------|
| Linear Regression | `Linear_Regression_force_predictions.py` | ~0.85 | 1000+ | Baseline |
| Random Forest | `random_forest_force_prediction.py` | ~0.94 | ~50 | Per-target |
| XGBoost | `random_forest_force_prediction.py` | ~0.95 | ~100 | Per-target |
| **LightGBM** | `LightGBM_sliding_window_predictions.py` | **~0.97** | **200-500** | **Recommended** |
| MLP | `NN_force_prediction.py` | ~0.96 | ~200 | GPU-friendly |
| LSTM | `LSTM.py` | ~0.97 | ~150 | Best temporal |
| 1D CNN | `1D_CNN_prediction.py` | ~0.97 | ~150 | Compact |
| Transformer | `transformer_sliding_window_predictions.py` | ~0.97 | ~100 | Experimental |

*Numbers are approximate; actual results depend on dataset version and hyperparameters.*

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'lightgbm'`**
```bash
pip install lightgbm==4.3.0
```

**`ModuleNotFoundError: No module named 'tensorflow'`**
```bash
pip install tensorflow==2.14.1
# CPU-only alternative:
pip install tensorflow-cpu==2.14.1
```

**`RuntimeError: No serial ports found`**
```bash
python -m serial.tools.list_ports        # list available ports
python realtime_demo.py --port COM3      # Windows
python realtime_demo.py --port /dev/ttyACM0  # Linux
```

**`FileNotFoundError: synchronized_events_<N>.csv`**
- Check that `TEST_NUMS` in the script matches your actual test folder names
- Verify `BASE_DATA_DIR` in `config.py` points to the correct location

**`KeyError: 'x_position_mm'` when loading CSVs**
- Column names vary between test batches. Check your CSV header and update the column-name mapping in `Pose_Forces_Synchronized_with_Baro_Processing.py`.

**Empty output CSV after synchronization**
- Increase `ASOF_TOLERANCE_S` (e.g. `0.1`) for wider time matching
- Verify both input files cover an overlapping time range
- Check that time columns are in seconds, not milliseconds

**`CUDA out of memory` during deep learning training**
```python
# Add at top of script before any TF/Torch imports:
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

**Model predictions are all zeros or NaN**
- Verify the scaler was saved and loaded from the same training run
- Check for `-999` sentinel values in training data (`signal_utils.convert_sentinel_to_nan`)
- Ensure `SENSOR_VERSION` matches between training and inference

---

## Utilities Reference

### `Barometers_Based_Tactile_Sensor/utils/`

| Module | Key Functions |
|--------|---------------|
| `barometer_processing.py` | `load_barometer_data()`, `process_barometers()`, `rezero_barometers_when_fz_zero()` |
| `sensor_io.py` | `load_ati_data()`, `load_atracsys_data()`, `asof_join()` |
| `signal_utils.py` | `maybe_denoise()`, `convert_sentinel_to_nan()` |
| `metrics_utils.py` | `calculate_grouped_rmse()`, `evaluate_constrained_region()` |
| `plot_utils.py` | `plot_pred_vs_actual()`, `plot_error_distributions()` |
| `io_utils.py` | `load_tabular_csv()` |

### Sliding-Window Features

For each time step `t`, the feature vector stacks the past `W` samples:

```
[b1..b6 at t-W..t]  +  [delta_b1..delta_b6 at t-W..t]  +  (optional) [delta2_b1..delta2_b6 at t-W..t]
```

No future data is used. Window size and derivative order are set in `config.py`.

---

## Notes

- Always match `SENSOR_VERSION` across training, evaluation, and inference scripts.
- Train/val/test CSVs must not overlap in time (no data leakage).
- The venv folder (`TactileSensingMasterThesis/`) is git-ignored; recreate with `pip install -r requirements.txt`.
- Clear Jupyter notebook outputs before committing:
  ```bash
  jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True sensor_characterization_plots.ipynb
  ```

---

**Author:** Aurora Ruggeri — Biorobotics Lab, EPFL
**Last Updated:** March 2026
