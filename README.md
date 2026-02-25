# Tactile Sensor — Master Thesis

This repository contains all code for a master thesis on **barometer-based tactile sensing**. The project is split into two main parts, both aimed at predicting contact forces — with the goal of comparing the two approaches against a ground-truth force/torque sensor (ATI).

---

## Project Overview

### Part 1 — Barometer-Based Tactile Sensor (`Barometers_Based_Tactile_Sensor/`)

A soft tactile sensor embedding **6 barometric pressure sensors (DPS310)** in a silicone pad. When a contact force is applied, the silicone deforms and the internal pressure changes. Machine learning models are trained to map those barometer readings to the 3-axis contact forces (Fx, Fy, Fz).

**Pipeline:**
1. Collect raw data: AprilTag-based 3D pose tracking + ATI force/torque sensor + Arduino barometer array
2. Extract and align tip pose and forces from AprilTag detections and ATI logs
3. Synchronize barometer readings with force/pose into a clean per-test dataset
4. Aggregate multiple tests into train/validation/test splits
5. Train and evaluate multiple ML models (linear regression, Random Forest, XGBoost, LightGBM, MLP, LSTM, 1D CNN)
6. Deploy the best model for real-time inference at 200–500+ Hz

### Part 2 — Vision-Based Force Estimation (`Vision_vs_Tactile/`)

An alternative approach that estimates contact forces **from visual deformation** of the sensor surface. A camera observes the silicone pad; the displacement/deformation of the surface (tracked via color or marker features) is used to predict applied forces.

**Pipeline:**
1. Extract image frames and barometer readings from a ROS bag file
2. Detect and track the deformation region using HSV color segmentation
3. Estimate forces from the deformation (color blob area, centroid shift, or displacement magnitude)

### Comparison Goal

Both approaches — **barometer-based** (Part 1) and **vision-based** (Part 2) — produce force estimates that are compared against the **ATI force/torque sensor** ground truth. This enables a quantitative evaluation of:
- Prediction accuracy (RMSE per axis)
- Temporal response and latency
- Robustness and practical deployment constraints

---

## Repository Structure

```
Thesis - Tactile Sensor/
│
├── Barometers_Based_Tactile_Sensor/       # Part 1: barometer-based pipeline
│   ├── utils/                             # Shared utility modules
│   ├── check_weight_location_from_wrench_and_tracking.py
│   ├── data_organization - New.py
│   ├── train_validation_test_dataset_generation.py
│   ├── barometers_plot.py
│   ├── hysteresis_material_testing.py
│   ├── COMSOL_pressure_analysis.py
│   ├── Sensor1_Repeteability.py
│   ├── Linear_Regression_force_predictions.py
│   ├── random_forest_force_prediction.py
│   ├── XGBoost_force_prediction.py
│   ├── NN_force_prediction.py
│   ├── LSTM.py
│   ├── 1D_CNN_prediction.py
│   ├── RF_or_XGB_sliding_window_predictions.py
│   ├── realtime_tactile_prediction.py
│   └── sensor_characterization_plots.ipynb
│
├── Vision_vs_Tactile/                     # Part 2: vision-based force estimation
│   ├── utils/
│   ├── direct_jpeg_extractor.py           # Extract frames from ROS bag + sync barometers
│   ├── forces_for_free.py                 # Estimate forces from visual deformation
│   └── hsv_color_picker.py               # Interactive HSV color range selector
│
├── Arduino Codes & Barometers Logging & AprilTags Scripts/
│   ├── TactileSensorArduinoCode/
│   │   └── TactileSensorCode.ino          # Firmware: 6x DPS310 barometers at 125 Hz
│   ├── single_sensor_hysteresis_testing_Arduino/
│   │   └── adafruit_single_sensor.ino
│   ├── baro_serial_node.py                # ROS node for Arduino barometer logging
│   └── tagposes_from_detections.py        # Compute tip poses from AprilTag detections
│
├── realtime_demo.py                       # Minimal real-time demo (no ROS required)
├── sensor_characterization.ipynb
├── requirements.txt
└── README.md
```

---

## Getting Started on a New Machine

Follow these steps in order to get the project running on a fresh laptop.

### 1. Clone the repository

```bash
git clone <repo-url>
cd "Thesis - Tactile Sensor"
```

### 2. Install Python

Make sure you have **Python 3.10 or 3.11** installed (3.12+ may have compatibility issues with TensorFlow).
Download from [python.org](https://www.python.org/downloads/) and check "Add Python to PATH" during installation.

Verify:
```
python --version
```

### 3. Create and activate a virtual environment

```powershell
python -m venv TactileSensingMasterThesis
TactileSensingMasterThesis\Scripts\Activate.ps1
```

If PowerShell blocks the activation script:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

### 4. Install dependencies

```
pip install -r requirements.txt
```

This installs all required packages (pandas, numpy, scikit-learn, XGBoost, TensorFlow, PyTorch, OpenCV, rosbags, etc.). It may take several minutes.

> **GPU (optional):** If you have an NVIDIA GPU and want to accelerate TensorFlow/PyTorch training, install the appropriate CUDA toolkit and cuDNN separately. The `requirements.txt` installs CPU-only builds by default.

### 5. Set up the data folder

The scripts expect raw data to live in a specific folder on your machine. Open each script you plan to run and update the hard-coded base path near the top of the file.

Look for a line like:
```python
BASE_DIR = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\"
```

Change it to the path where your data lives, for example:
```python
BASE_DIR = r"C:\Users\yourname\data\Thesis- Biorobotics Lab\"
```

Then create the expected folder structure under that base path (see the **Data Organization** section below). Copy your raw test data files into the corresponding folders.

### 6. (Optional) Flash the Arduino

If you are collecting new data:
- Open `Arduino Codes & Barometers Logging & AprilTags Scripts/TactileSensorArduinoCode/TactileSensorCode.ino` in the Arduino IDE
- Select board: **Arduino MKR Zero**
- Upload to the board

The Arduino will output barometer readings over USB serial at 115200 baud.

### 7. (Optional) Set up VS Code

For the best development experience:
1. Open the repo folder in VS Code
2. Press `Ctrl+Shift+P` → **Python: Select Interpreter**
3. Choose the interpreter at `TactileSensingMasterThesis\Scripts\python.exe` (inside the repo folder)

---

## Environment Setup

Python virtual environment created for this project:

```
C:\venvs\TactileSensingMasterThesis
```

Activate (PowerShell):
```powershell
C:\venvs\TactileSensingMasterThesis\Scripts\Activate.ps1
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Run a script directly without activating:
```
C:\venvs\TactileSensingMasterThesis\Scripts\python.exe path\to\script.py
```

Recreate from scratch:
```
python -m venv TactileSensingMasterThesis
TactileSensingMasterThesis\Scripts\python.exe -m pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `tensorflow`, `torch`, `opencv-python`, `rosbags`, `pyserial`, `matplotlib`, `seaborn`, `tqdm`, `numba`, `joblib`

---

## Data Organization

Most scripts hard-code this base folder:

```
C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\
```

Expected structure under that base:

```
test data\
  test <test_num> - sensor v<version_num>\
    apriltag_detections_trial<test_num>.txt  # AprilTag pose detections log
    ati_middle_trial<test_num>.txt          # ATI force/torque log
    barometers_trial<test_num>.txt          # Arduino barometer log
    processing_test_<test_num>.csv          # Output of step 1
    synchronized_events_<test_num>.csv      # Output of step 2
    lin<test_num>\                          # Tip-path plots
    plots synchronized\                     # Diagnostic plots

train_validation_test_data\
  train_data_v<version>.csv
  validation_data_v<version>.csv
  test_data_v<version>.csv
  sensor_v<version>\

models parameters\
  averaged models\     # LightGBM sliding-window models + scalers
  cnn_models\
  random forest\
  regression\
  NN\
  XGBoost\

COMSOL_plots\
hysterisis\
hysteresys material testing\
Sensor-Logs\
ATI_calibration_files\
```

> **Note:** Update the hard-coded base path at the top of each script if running on a different machine.

---

## Part 1 — Barometer Pipeline (Typical Order)

### Step 1: Extract Tip Pose and Forces

**Script:** `AprilTags_check_weight_location_from_wrench_and_tracking copy.py`

- Inputs: `apriltag_detections_trial<num>.txt`, `ati_middle_trial<num>.txt`
- Outputs: `processing_test_<num>.csv`, tip-path plots in `lin<num>\`
- What it does: detects tip pose from AprilTag markers, transforms pose to top frame, aligns ATI wrenches, and exports tip-path plots.

### Step 2: Synchronize Barometers with Forces

**Script:** `data_organization - New.py`

- Inputs: `processing_test_<num>.csv`, `barometers_trial<num>.txt`
- Outputs: `synchronized_events_<num>.csv`, diagnostic plots in `plots synchronized\`
- What it does: removes 2 s warmup, applies drift removal (EMA or linear temperature compensation), dynamically re-zeros barometers when Fz = 0, synchronizes via `merge_asof` (50 ms tolerance), applies spatial masking and MAD-based outlier removal.

### Step 3: Build Train/Validation/Test Splits

**Script:** `train_validation_test_dataset_generation.py`

- Inputs: list of `synchronized_events_<num>.csv` files
- Outputs: `train_data_v<version>.csv`, `validation_data_v<version>.csv`, `test_data_v<version>.csv`

### Step 4: Train ML Models

All model scripts read from `train_validation_test_data\` and save to `models parameters\`.

| Script | Model | Notes |
|---|---|---|
| `Linear_Regression_force_predictions.py` | Linear Regression | Baseline |
| `random_forest_force_prediction.py` | Random Forest | Per-target models |
| `XGBoost_force_prediction.py` | XGBoost | Per-target models |
| `NN_force_prediction.py` | MLP (Keras) | GPU-optimized |
| `LSTM.py` | LSTM (Keras) | Temporal patterns |
| `1D_CNN_prediction.py` | 1D CNN (Keras) | Sliding-window input |
| `RF_or_XGB_sliding_window_predictions.py` | RF / XGBoost / LightGBM | Sliding window + derivatives |

### Step 5: Real-Time Deployment

**Script:** `realtime_tactile_prediction.py`
**Demo:** `realtime_demo.py`

- Connects to the Arduino over serial (USB)
- Loads a saved LightGBM model + scaler
- Predicts Fx, Fy, Fz in a loop at 200–500+ Hz (Numba JIT-compiled)
- Demo usage (no ROS needed): `python realtime_demo.py --port COM3`

---

## Part 2 — Vision-Based Force Estimation

### Step 1: Extract Frames and Sync Barometers

**Script:** `Vision_vs_Tactile/direct_jpeg_extractor.py`

- Inputs: ROS bag file containing camera images and barometer topics
- Outputs: extracted JPEG frames with timestamps, synchronized barometer readings
- Features an interactive HSV color picker (Tab 1: include colors, Tab 2: exclude false positives) for tuning the deformation segmentation

### Step 2: Estimate Forces from Deformation

**Script:** `Vision_vs_Tactile/forces_for_free.py`

- Inputs: extracted frames + synchronized barometer/force data
- Process: color-based segmentation of the deformed sensor surface → geometric features (blob area, centroid shift) → force estimate
- Output: time series of vision-estimated forces, compared against ATI ground truth

### Utility

**Script:** `Vision_vs_Tactile/hsv_color_picker.py`

- Interactive tool for selecting HSV color ranges on a sample image
- Used to calibrate the color segmentation step

---

## Modeling Approach and Time-Series Features

Barometer signals are time-dependent: the array captures pressure dynamics, hysteresis, and transient effects not visible in a single sample. Models are trained on time-series features.

**Why sliding windows:**
- Contact events evolve over tens of milliseconds; a short window captures loading/unloading and sensor settling
- Local context reduces noise sensitivity compared to single-frame input
- Models can implicitly infer velocity/acceleration trends from history

**Sliding window concept:**
- For each prediction time `t`, a window of past samples (e.g., 10–31 samples) is stacked as features
- Window size configured per script (`WINDOW_SIZE`, `WINDOW_RADIUS`)
- Inputs remain aligned to the prediction target at time `t` (no future data leakage)

**First and second derivatives:**
- `d1`: rate of change in pressure — correlates with dynamic force changes
- `d2`: curvature — captures transient behavior (rapid loading/unloading)
- Computed after optional denoising (rolling mean), then appended as additional features
- Controlled by flags in each script (`USE_DERIVATIVES`, `USE_SECOND_DERIVATIVE`)

---

## Arduino Firmware

**File:** `Arduino Codes & Barometers Logging & AprilTags Scripts/TactileSensorArduinoCode/TactileSensorCode.ino`

- Hardware: Arduino MKR Zero + 6× DPS310 barometric pressure sensors
- Reads pressure (Pa) and temperature (°C) from all 6 sensors
- Serial output at 115200 baud in CSV format:
  ```
  Time(ms), B1_P, B1_T, B2_P, B2_T, ..., B6_P, B6_T
  ```
- ROS logging wrapper: `baro_serial_node.py` (publishes on `/baro6_raw`)

---

## Utility Modules (`Barometers_Based_Tactile_Sensor/utils/`)

| Module | Description |
|---|---|
| `barometer_processing.py` | Load barometer data, drift removal, re-zeroing, outlier detection, plotting |
| `sensor_io.py` | Load AprilTag pose and ATI logs; `asof_join()` for time-series alignment |
| `signal_utils.py` | Denoising (rolling mean, Savitzky-Golay), first/second derivatives |
| `metrics_utils.py` | RMSE per target, accuracy in constrained spatial regions |
| `plot_utils.py` | Scatter plots, error distributions, Keras training history plots |
| `io_utils.py` | File path helpers, data loading/saving |

---

## Notes

- All scripts contain hard-coded absolute paths and version numbers — update them at the top of each file before running.
- The XGBoost script has a typo in its output path (`models paramters` instead of `models parameters`); this is noted in the data structure section above.
- The `TactileSensingMasterThesis/` folder at the repo root is a local Python venv — it is not part of the source code.
