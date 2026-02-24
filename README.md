# Tactile Sensor Thesis Project

This repository collects data-processing and modeling scripts for a master thesis on tactile sensing. The code base is organized around a repeatable pipeline: raw tracking and force data are converted into synchronized datasets, which are then used for exploratory analysis and machine-learning models.

## Overview

Primary goals:

- Extract indenter tip pose and top-frame forces from Atracsys + ATI logs.
- Synchronize force/pose with barometer readings into clean per-test datasets.
- Aggregate multiple tests into train/validation/test splits.
- Train and evaluate predictive models for contact location and force.

## Environment

Python environment (created for this project):

```
C:\venvs\TactileSensingMasterThesis
```

Activate (PowerShell):

```
C:\venvs\TactileSensingMasterThesis\Scripts\Activate.ps1
```

Run a script directly:

```
C:\venvs\TactileSensingMasterThesis\Scripts\python.exe path\to\script.py
```

<!-- VS Code interpreter:

1) Ctrl+Shift+P
2) Python: Select Interpreter
3) C:\venvs\TactileSensingMasterThesis\Scripts\python.exe -->

## Requirements

The environment snapshot is stored in `requirements.txt`.

To recreate:

```
python -m venv TactileSensingMasterThesis
TactileSensingMasterThesis\Scripts\python.exe -m pip install -r requirements.txt
```

## Data Organization (expected by scripts)

Most scripts hard-code this base folder:

```
C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\
```

Recommended structure under that base:

```
test data\
  test <test_num> - sensor v<version_num>\
    atracsys_trial<test_num>.txt
    ati_middle_trial<test_num>.txt
    barometers_trial<test_num>.txt
    processing_test_<test_num>.csv
    synchronized_events_<test_num>.csv
    lin<test_num>\
      tip_path_top_frame_trial<test_num>.csv
      (plots: 2D/3D tip path, coordinates, ATI wrench plots)
    plots synchronized\

train_validation_test_data\
  train_data_v<version>.csv
  validation_data_v<version>.csv
  test_data_v<version>.csv
  sensor_v<version>\

models parameters\
  averaged models\
  cnn_models\
  random forest\
  regression\
  NN\
  XGBoost    (note: XGBoost_force_prediction.py uses "models paramters")

COMSOL_plots\
hysterisis\
hysteresys material testing\
Sensor-Logs\

ATI_calibration_files\   (used by plot_ati_from_txt.py)
```

## Processing Pipeline (typical order)

1) Raw Atracsys + ATI -> `check_weight_location_from_wrench_and_tracking.py`
   - Outputs: `processing_test_<test>.csv` and tip-path plots.
2) `processing_test_<test>.csv` + barometers -> `data_organization - New.py`
   - Outputs: `synchronized_events_<test>.csv` + diagnostic plots.
3) Multiple synchronized tests -> `train_validation_test_dataset_generation.py`
   - Outputs: train/validation/test CSVs.
4) Train/evaluate models using the ML scripts in this repository.

## Modeling Approach and Time-Series Features

The tactile sensor signals are time-dependent: the barometer array captures pressure dynamics, hysteresis, and transient effects that are not visible in a single sample. For this reason, the ML models are trained on time-series features rather than single-frame inputs.

Methods used in this project:

- Linear models (baseline): Linear regression for fast interpretability and a sanity check on performance.
- Tree-based ensembles: Random Forest, XGBoost, and LightGBM for non-linear relationships and robust performance on tabular features.
- Neural networks (MLP): Fully connected models to learn non-linear mappings from engineered features.
- Sequence models: LSTM and 1D CNN to model temporal patterns directly.

Why time windows:

- Contact events evolve over tens of milliseconds; a short temporal window captures loading/unloading and sensor settling.
- Time windows reduce noise sensitivity by providing local context rather than a single noisy reading.
- Models can infer velocity/acceleration trends implicitly when given history.

Sliding-window concept:

- For each prediction time t, a window of past samples (e.g., 10 to 31 samples) is stacked as features.
- The window size is configured per script (e.g., WINDOW_SIZE, WINDOW_RADIUS).
- Inputs remain aligned to the prediction target at time t (no future leakage).

First and second derivatives:

- First derivative (d1) approximates rate of change in pressure, which correlates with dynamic force changes.
- Second derivative (d2) captures curvature and transient behavior (e.g., rapid loading/unloading).
- These derivatives are computed after optional denoising (rolling mean), then appended as additional features.
- Enabling d1/d2 is controlled by flags in the scripts (e.g., USE_DERIVATIVES, USE_SECOND_DERIVATIVE).

## Script Reference

Each script lists its own hard-coded paths near the top. The entries below summarize required inputs and produced outputs.

### check_weight_location_from_wrench_and_tracking.py

- Inputs:
  - `test data\test <num> - sensor v<version>\atracsys_trial<num>.txt`
  - `test data\test <num> - sensor v<version>\ati_middle_trial<num>.txt`
- Outputs:
  - `test data\test <num> - sensor v<version>\processing_test_<num>.csv`
  - `test data\test <num> - sensor v<version>\lin<num>\tip_path_top_frame_trial<num>.csv`
  - Plots in `lin<num>\`
- Purpose: compute tip pose in the top frame, align ATI wrenches, and export plots.

### data_organization - New.py

- Inputs:
  - `processing_test_<num>.csv`
  - `barometers_trial<num>.txt`
- Outputs:
  - `synchronized_events_<num>.csv`
  - `plots synchronized\` (diagnostic PNGs)
- Purpose: synchronize barometers and force/pose, re-zero, filter, and export a clean dataset.

### train_validation_test_dataset_generation.py

- Inputs:
  - A list of `synchronized_events_<num>.csv` files (configured in the script).
- Outputs:
  - `train_validation_test_data\train_data_v<version>.csv`
  - `train_validation_test_data\validation_data_v<version>.csv`
  - `train_validation_test_data\test_data_v<version>.csv`
  - Distribution plots in `train_validation_test_data\sensor_v<version>\`
- Purpose: combine multiple tests into train/validation/test splits.

### check_data.py

- Input: `test data\test <num> - sensor v<version>\synchronized_events_<num>.csv`
- Output: `test data\relationship_analysis\` (plots + summary text)
- Purpose: correlation analysis, feature importance, and exploratory plots.

### barometers_plot.py

- Input: `test data\test <num> - sensor v<version>\barometers_trial<num>.txt`
- Output: PNG saved next to the barometer file
- Purpose: visualize barometer pressures (and optional temperature channels).

### Hysteresis_Analysis.py

- Input: `test data\test <num> - sensor v<version>\synchronized_events_<num>.csv`
- Output: `hysterisis\hysteresis_Fz_vs_Barometers_test<num>.png`
- Purpose: plot hysteresis curves (barometers vs Fz).

### hysteresis_material_testing.py

- Inputs:
  - `Sensor-Logs\test_<num>.csv`
  - `Sensor-Logs\ati_middle_trial<num>.txt`
- Outputs: `hysteresys material testing\` (merged CSV + plots)
- Purpose: align barometer and ATI logs with manual time shift; compute hysteresis.

### plot_ati_from_txt.py

- Input: `ATI_calibration_files\ati_middle_trial*.txt`
- Output: summary CSV + plots in the same folder
- Purpose: inspect ATI calibration trials.
- Note: expects `plot_style.py` to be importable (not present in this repo).

### COMSOL_pressure_analysis.py

- Input: `COMSOL_simulation_data.csv` (or any CSV in base folder with required columns)
- Output: `COMSOL_plots\plots_*` PNGs
- Purpose: analyze COMSOL simulation sweeps.

### predictions.py

- Inputs:
  - `synchronized_events_<num>.csv`
  - `models parameters\averaged models\lightgbm_sliding_window_model_v<version>.pkl`
  - `models parameters\averaged models\scaler_sliding_window_v<version>.pkl`
- Output: interactive plots (no files by default)
- Purpose: run the sliding-window model on a single dataset and visualize errors.

### new_prediction.py

- Inputs:
  - `models parameters\random forest\specialized_rf_models_v<version>.pkl`
  - `synchronized_events_<num>.csv`
- Output: predictions CSV saved in the same models folder
- Purpose: generate predictions using per-target RF models.

### random_forest_force_prediction.py

- Inputs:
  - `train_validation_test_data\train_data_v<version>.csv`
  - `train_validation_test_data\test_data_v<version>.csv`
- Outputs: `models parameters\random forest\` (models, scalers, plots)
- Purpose: train per-target Random Forest models.

### XGBoost_force_prediction.py

- Inputs:
  - `train_validation_test_data\train_data_v<version>.csv`
  - `train_validation_test_data\test_data_v<version>.csv`
- Outputs: `models paramters\XGBoost\` (models, scalers, plots)
- Purpose: train per-target XGBoost models.

### Linear_Regression_force_predictions.py

- Inputs:
  - `train_validation_test_data\train/validation/test_data_v<version>.csv`
- Outputs:
  - `linear_regression_predictions_v<version>.png` (saved next to the script)
  - `models parameters\regression\` (models + scalers)
- Purpose: baseline linear regression models.

### NN_force_prediction.py

- Inputs:
  - `train_validation_test_data\train/validation/test_data_v<version>.csv`
- Outputs: `models parameters\NN\` (weights, scalers, plots)
- Purpose: feedforward neural network regression.

### RF_or_XGB_sliding_window_predictions.py

- Inputs:
  - `train_validation_test_data\train_data_v<version>.csv`
  - `train_validation_test_data\test_data_v<version>.csv`
- Outputs: `models parameters\averaged models\` (models, scalers, plots)
- Purpose: sliding-window features + RF/XGBoost/LightGBM models.

### LSTM.py

- Inputs:
  - `train_validation_test_data\train/validation/test_data_v<version>.csv`
- Outputs: `models parameters\averaged models\` (model, scalers, plots)
- Purpose: LSTM sequence model with optional derivatives.

### 1D_CNN_prediction.py

- Inputs:
  - `train_validation_test_data\train_data_v<version>.csv`
  - `train_validation_test_data\test_data_v<version>.csv`
- Outputs: `models parameters\cnn_models\` (model, scalers, plots)
- Purpose: Conv1D model on sliding-window barometer data.

## Notes

- Some scripts contain hard-coded absolute paths and version numbers; update them at the top of each file.
- If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
