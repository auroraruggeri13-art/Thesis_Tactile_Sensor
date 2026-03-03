# Thesis Project Context — Tactile Sensor (Biorobotics Lab)

## Master's Thesis Overview

**Institution**: EPFL — Biorobotics Laboratory
**Topic**: Design, fabrication and characterisation of a barometer-based tactile
sensor for 3-axis force estimation (Fx, Fy, Fz) using machine learning.
**Goal**: Build a low-cost, compliant tactile sensor that can be mounted on a
robot end-effector and estimate contact forces in real time from 6 barometric
pressure readings, replacing expensive 6-axis force/torque sensors for tasks
such as surface exploration and contact-force feedback.

### Sensor Hardware (v5)
- **Sensing elements**: 6 MEMS barometers embedded in a compliant silicone pad
- **Mechanical design**: the silicone deforms under contact, transmitting pressure
  differentially to the barometers depending on the force direction and magnitude
- **Sampling rate**: ~60 Hz (Raspberry Pi-based DAQ)
- **Force range characterised**: 0–12 N normal (Fz), ±2 N lateral (Fx, Fy)
- **Ground truth**: ATI Nano17 6-axis force/torque sensor (rigidly coupled)
- **Pose tracking**: Atracsys optical tracker + AprilTag markers on box and probe

### ML Pipeline
- **Input**: sliding window of 10 past barometer samples + 1st and 2nd derivatives
  → 198 features total
- **Model**: LightGBM ensemble (5 independent regressors: x, y, Fx, Fy, Fz)
- **Preprocessing**: rolling-mean denoising (window=5) on barometer channels
- **Training/test split**: stratified by trial; test set is held-out dynamic
  indenter trajectory (test_data_v5.18.csv)
- Other models compared: Linear Regression, Random Forest, XGBoost, LSTM, NN

### Thesis Sections (work in progress)
- Sensor design and fabrication
- Data collection protocol (Atracsys + ATI synchronisation)
- Machine learning model comparison
- **Sensor characterisation** ← active focus:
  - Repeatability (Sensor1_Repeteability.py)
  - Force resolution (force_resolution_static_analysis.py) ← completed this session
  - Spatial resolution / contact location (AprilTags pipeline)
- Vision vs Tactile comparison (Vision_vs_Tactile/)

---

## Work Done in the Previous Session (2026-03-01)

### 1. Diagnosed diagonal line artefacts in ATI wrench plots
- **Problem**: Top Frame force/torque plots showed straight diagonal lines
  connecting data points across time gaps; ATI Frame plots were clean.
- **Root cause**: `ati_top_df.dropna()` silently removes NaN rows (created when
  Atracsys tracker loses the probe AprilTag), leaving large time jumps between
  consecutive rows. matplotlib then draws a straight line across each gap.
- **Fix identified**: insert NaN rows wherever `dt > threshold` before calling
  `ax.plot()` — not yet implemented in code.
- **Key point**: this is a plotting-only artefact; it does NOT affect any σ or
  RMSE calculations.

### 2. Investigated force resolution QC plot artefacts
- The green "Pred Fz" line in the QC plot also showed diagonal artefacts.
- Confirmed: `LGBM_MAX_TIME_GAP=0.05 s` discards ~47% of windows from
  train_data_v5.1893 due to recording gaps in the raw barometer data.
- These gaps create time jumps in the prediction array → matplotlib artefacts.
- **Conclusion**: artefacts do not affect σ (computed on dense array indices,
  not time); quasi-static segments are on stable plateaux where gaps are benign.

### 3. Fixed quasi-static segment detection (7 → 5 segments)
- **Problem**: algorithm detected 7 windows but only 5 visual force steps exist.
- **Root cause**: force steps 4 (~6 N) and 5 (~9 N) each had brief rolling-std
  spikes that split one plateau into two fragments separated by <0.5 s.
- **Fix**: raised `MIN_SEGMENT_SAMPLES` from 10 → 190, which drops the tiny
  first fragments (112 samples) while keeping the settled second fragments.
- Result: exactly 5 segments, one per visual force step.

### 4. Confirmed correct dataset assignment for each method
- Tested Method B (RMSE) on train_data_v5.1893 → gave 3–7× worse RMSE
  (dominated by step-loading transitions the model can't track).
- **Decision**: keep Method A on train_data_v5.1893 and Method B on test_data_v5.18.

### 5. Wrote LaTeX thesis section on Force Resolution
- Full `\subsubsection{Force Resolution}` written with:
  - Mathematical definition (3σ criterion)
  - Method A results table (per-segment breakdown)
  - Method B results table
  - Combined summary table
  - Discussion of force-range dependence and training-data-distribution hypothesis
  - Figure captions for both QC plot and scatter plot
- Uses `siunitx` and `booktabs` packages.

---

## Project Overview
Barometer-based tactile sensor for force estimation (Fx, Fy, Fz) using LightGBM
sliding-window predictions. Ground truth is ATI Nano17 force/torque sensor.
Sensor version in active use: **v5.18** (current model: v5.180).

---

## Repository Structure (key files)

```
Barometers_Based_Tactile_Sensor/
  force_resolution_static_analysis.py   ← force resolution characterisation script
  AprilTags_check_weight_location_from_wrench_and_tracking copy.py  ← tip path + wrench transform
  train_validation_test_dataset_generation.py  ← dataset splits
  LSTM.py, NN_force_prediction.py, random_forest_force_prediction.py
  RF_or_XGB_sliding_window_predictions.py
  Linear_Regression_force_predictions.py
  utils/
    io_utils.py        ← load_tabular_csv
    sensor_io.py       ← load_ati_data, load_atracsys_data, asof_join
    plot_utils.py      ← plot_pred_vs_actual
sensor_characterization.ipynb
```

---

## Data Paths

```
Base data dir:
  C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data\

Key files:
  train_data_v5.1893.csv   ← quasi-static step-loading experiment (5 force levels, 2–9 N)
  test_data_v5.18.csv      ← held-out dynamic test set (5638 samples)
  test_data_v5.19.csv      ← v5.19 test set (also available)
  train_data_v5.19.csv, validation_data_v5.19.csv

Model dir:
  C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models\

Active models:
  lightgbm_sliding_window_model_v5.180.pkl  ← 5 sub-models [x, y, fx, fy, fz]
  scaler_sliding_window_v5.180.pkl
  lightgbm_sliding_window_model_v5.190.pkl  ← v5.19 model (198 features, same architecture)
  scaler_sliding_window_v5.190.pkl

Output dir:
  C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\Sensor Characterization\force_resolution\
```

---

## LightGBM Model Hyperparameters (v5.180)

| Parameter             | Value  | Notes                          |
|-----------------------|--------|--------------------------------|
| LGBM_WINDOW_SIZE      | 10     | past samples per window        |
| LGBM_USE_2ND_DERIV    | True   | → 198 features total           |
| LGBM_APPLY_DENOISING  | True   | rolling-mean on barometers     |
| LGBM_DENOISE_WINDOW   | 5      | samples                        |
| LGBM_MAX_TIME_GAP     | 0.05 s | skip windows spanning gaps     |
| LGBM_TARGETS          | [x, y, fx, fy, fz] | model list index order |
| Feature count         | 198    | 6 baros × 11 steps × 3 derivs |

---

## Force Resolution Results (current best — sensor v5.18)

### Method A — Quasi-static repeatability (train_data_v5.1893, 5 segments, median σ)

| Axis | σ [N]  | ΔF_min = 3σ [N] |
|------|--------|-----------------|
| Fx   | 0.084  | **0.251**       |
| Fy   | 0.035  | **0.104**       |
| Fz   | 0.270  | **0.811**       |

Detection criterion: rolling-std(GT Fz) < 0.25 N over 1.0 s window,
MIN_SEGMENT_SAMPLES = 190.

Per-segment breakdown (sorted by load):
| Seg | |Fz| [N] | n    | σ_Fz [N] | ΔFz [N] |
|-----|-----------|------|-----------|---------|
| 1   | 2.02      | 527  | 0.251     | 0.752   |
| 2   | 3.67      | 473  | 0.159     | 0.477   |
| 3   | 5.03      | 684  | **0.270** | **0.811** ← median |
| 4   | 5.80      | 234  | 0.517     | 1.550   |
| 5   | 8.80      | 199  | 0.424     | 1.273   |

Segments 4 & 5 are fragmented (shorter hold times at higher loads).
Best resolution at 3.67 N — consistent with training data distribution peak.

### Method B — RMSE on held-out test split (test_data_v5.18, 5338 samples)

| Axis | MAE [N] | RMSE [N] | ΔF_min = 3σ [N] | Bias [N]  | R²    |
|------|---------|----------|-----------------|-----------|-------|
| Fx   | 0.165   | 0.240    | **0.718**       | −0.008    | 0.617 |
| Fy   | 0.099   | 0.131    | **0.393**       | +0.012    | 0.610 |
| Fz   | 0.478   | 0.655    | **1.966**       | −0.091    | 0.592 |

Method A is ~2.4–3.8× lower than Method B (static floor vs dynamic conditions).

---

## Key Design Decisions

- **Method A** needs the dedicated step-loading dataset (train_data_v5.1893) —
  the test set has no long quasi-static windows (max blob = 113 samples).
- **Method B** must use test_data_v5.18 — using train_data_v5.1893 gives
  misleading RMSE (3–7× worse) because step-loading transitions dominate.
- Using MEDIAN aggregation for Method A is robust against the short/fragmented
  segments at higher force levels.

---

## Frame Definitions (AprilTags script)

```
C  : Camera frame      — Atracsys world frame (fixed)
B  : Box frame         — tag1, rigidly fixed on the box
T  : Top frame         — box surface; origin = B + d_T_in_B, axes parallel to B
P  : Probe frame       — tag2, moving with the tool
A  : ATI frame         — fixed offset+rotation relative to P
```

Active trial: **51995**, sensor v5
Current offsets (trial 51995):
  X_SHIFT = 9 mm, Y_SHIFT = 64 mm, Z_SHIFT = 120 mm

R_P_A = Rz(48°) @ Rx(180°)  — ATI mounted 48° rotated, Z flipped
TIP_OFFSET_IN_PROBE_M = [0.0, 0.01, −0.065] m
ATI_ABOVE_TIP_M = 0.025 m

---

## Known Issues / Gotchas

1. **Diagonal line artefacts in plots**: matplotlib connects across time gaps
   (NaN rows dropped from ati_top_df). Fix: insert NaN rows at gaps > threshold
   before plotting. Does NOT affect σ calculations.

2. **~47% of windows discarded** in train_data_v5.1893 due to LGBM_MAX_TIME_GAP=0.05s.
   The raw data has many recording gaps >50 ms. Gaps at force transitions don't
   affect σ (quasi-static windows are on stable plateaux).

3. **Rolling-std is index-based, not time-based** — gaps within a plateau are
   treated as consecutive samples. Benign if force is stable across the gap.

4. **Segments 4 & 5 fragmented**: visual step 4 (~6 N) split into two windows
   (0.47 s gap between them) by brief noise spike. MIN_SEGMENT_SAMPLES=190
   correctly drops the tiny first fragment (112 samples).

5. **ATI Top Frame torque artefacts**: Atracsys tracker dropouts → NaN rows →
   dropna() → time gaps → diagonal matplotlib lines in Top Frame torque plot.
   ATI Frame plot is unaffected (raw sensor, no transformation needed).

---

## Current Git Branch
`aurora`

## Active Script Config (force_resolution_static_analysis.py)
```python
SENSOR_VERSION       = 5.18
RAW_CSV_FILES        = ["train_data_v5.1893.csv"]   # Method A
TEST_SPLIT_CSV       = "test_data_v5.18.csv"        # Method B
LGBM_MODEL_FILE      = "lightgbm_sliding_window_model_v5.180.pkl"
MIN_SEGMENT_SAMPLES  = 190
GT_ROLLING_STD_MAX_N = 0.25
ROLLING_WINDOW_S     = 1.0
AGGREGATION          = "median"
```
