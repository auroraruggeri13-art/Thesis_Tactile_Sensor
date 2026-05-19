#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-series prediction plot for model v5.200 on test set v5.200.
Shows 5 rows (x, y, fx, fy, fz) over a 10-second sliding window.
"""

import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils.signal_utils import maybe_denoise, convert_sentinel_to_nan

# ============================================================
# CONFIG
# ============================================================
DATA_DIRECTORY  = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data"
MODEL_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"

sensor_version  = 5.2
VERSION_STR     = f"v{sensor_version:.3f}"   # e.g. "v5.200"
TEST_FILENAME   = f"test_data_v{sensor_version}.csv"

TIME_COL   = "t"
BARO_COLS  = ["b1", "b2", "b3", "b4", "b5", "b6"]
TARGET_COLS = ["x", "y", "fx", "fy", "fz"]

WINDOW_SIZE           = 10
APPLY_DENOISING       = True
DENOISE_WINDOW        = 5
CONVERT_SENTINEL_TO_NAN = True
NO_CONTACT_SENTINEL   = -999.0
USE_SECOND_DERIVATIVE = True

# How many seconds to show per plot window
SEGMENT_DURATION = 11   # seconds
T_START_OFFSET   = 1976    # seconds from the start of the test data
MA_WINDOW        = 3       # moving-average smoothing on predictions

# ============================================================
# FEATURE ENGINEERING (mirrors the training script exactly)
# ============================================================

def build_window_features(df, baro_cols, time_col, target_cols, window_size,
                           max_time_gap=0.05, use_second_derivative=False):
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df = maybe_denoise(df, baro_cols, apply_denoising=APPLY_DENOISING,
                       denoise_window=DENOISE_WINDOW)

    for col in baro_cols:
        d1 = df[col].diff()
        df[f"{col}_d1"] = d1.fillna(0.0)
        if use_second_derivative:
            df[f"{col}_d2"] = d1.diff().fillna(0.0)

    N = len(df)
    W = window_size

    time_vals = df[time_col].values
    baro_data = df[baro_cols].values
    d1_cols   = [f"{col}_d1" for col in baro_cols]
    d1_data   = df[d1_cols].values
    if use_second_derivative:
        d2_cols = [f"{col}_d2" for col in baro_cols]
        d2_data = df[d2_cols].values

    X_list, y_list, valid_indices = [], [], []

    for current_idx in range(W, N):
        start = current_idx - W
        end   = current_idx + 1
        time_diffs = np.diff(time_vals[start:end])
        if np.max(time_diffs) > max_time_gap:
            continue
        baro_win = baro_data[start:end, :].flatten()
        d1_win   = d1_data[start:end, :].flatten()
        if use_second_derivative:
            d2_win = d2_data[start:end, :].flatten()
            X_list.append(np.concatenate([baro_win, d1_win, d2_win]))
        else:
            X_list.append(np.concatenate([baro_win, d1_win]))
        y_list.append(df.loc[current_idx, target_cols].values)
        valid_indices.append(current_idx)

    X         = np.array(X_list)
    y         = np.array(y_list)
    center_df = df.iloc[valid_indices].reset_index(drop=True)
    return X, y, center_df


# ============================================================
# MAIN
# ============================================================

def main():
    # ---- Load test data ----
    test_path = os.path.join(DATA_DIRECTORY, TEST_FILENAME)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    test_df = pd.read_csv(test_path, skipinitialspace=True)
    test_df.columns = test_df.columns.str.strip()
    print(f"Loaded test data: {len(test_df)} rows")

    if CONVERT_SENTINEL_TO_NAN:
        test_df = convert_sentinel_to_nan(test_df, TARGET_COLS, NO_CONTACT_SENTINEL)

    needed_cols = [TIME_COL] + BARO_COLS + TARGET_COLS
    test_df = test_df.dropna(subset=needed_cols).reset_index(drop=True)
    print(f"After dropping NaN: {len(test_df)} rows")

    # ---- Load scaler and model ----
    scaler_path = os.path.join(MODEL_DIRECTORY, f"scaler_sliding_window_{VERSION_STR}.pkl")
    model_path  = os.path.join(MODEL_DIRECTORY, f"lightgbm_sliding_window_model_{VERSION_STR}.pkl")

    for p in (scaler_path, model_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(model_path, "rb") as f:
        lgbm_models = pickle.load(f)

    print(f"Loaded scaler: {scaler_path}")
    print(f"Loaded model:  {model_path}  ({len(lgbm_models)} sub-models)")

    # ---- Build features ----
    print("\nBuilding windowed features ...")
    X_test, y_test, center_df = build_window_features(
        test_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
        use_second_derivative=USE_SECOND_DERIVATIVE
    )
    print(f"Feature matrix: {X_test.shape}")

    X_test_scaled = scaler.transform(X_test)

    # ---- Predict ----
    print("Predicting ...")
    y_pred = np.column_stack([m.predict(X_test_scaled) for m in lgbm_models])

    time_vals = center_df[TIME_COL].values

    # ---- Pick first clean 10-second segment ----
    t_start = time_vals[0] + T_START_OFFSET
    t_end   = t_start + SEGMENT_DURATION
    mask    = (time_vals >= t_start) & (time_vals < t_end)
    t_seg   = time_vals[mask] - time_vals[mask][0]  # relative time from plot start
    gt_seg  = y_test[mask]
    pr_seg  = y_pred[mask]

    if len(t_seg) == 0:
        raise RuntimeError("No samples found in the selected 10-second window.")

    print(f"Plotting {len(t_seg)} samples  [{t_seg[0]:.3f} s -> {t_seg[-1]:.3f} s]")

    # Moving-average smoother for predictions
    def moving_average(arr, w=MA_WINDOW):
        return np.convolve(arr, np.ones(w) / w, mode="same")

    # ---- 5-row subplot ----
    ROW_META = {
        "x":  {"ylabel": "x [mm]"},
        "y":  {"ylabel": "y [mm]"},
        "fx": {"ylabel": "fx [N]"},
        "fy": {"ylabel": "fy [N]"},
        "fz": {"ylabel": "fz [N]"},
    }
    COLOR_GT   = "#d4a017"   # golden yellow
    COLOR_PRED = "#292f56"   # dark navy

    fig, axes = plt.subplots(5, 1, figsize=(26, 22), sharex=True)
    fig.suptitle(
        f"Time Series: Predicted vs Ground Truth",
        fontsize=26, fontweight="bold", y=1.01
    )

    for row_idx, target in enumerate(TARGET_COLS):
        ax  = axes[row_idx]
        ci  = TARGET_COLS.index(target)
        meta = ROW_META[target]
        ax.tick_params(axis="both", labelsize=17)

        gt      = moving_average(gt_seg[:, ci], 5)
        pred_ma = moving_average(pr_seg[:, ci], MA_WINDOW)

        ax.scatter(t_seg, gt,      color=COLOR_GT,   s=15, label="Ground Truth")
        ax.scatter(t_seg, pred_ma, color=COLOR_PRED,  s=15, label=f"Predicted")

        ax.set_ylabel(meta["ylabel"], fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=17, framealpha=0.75)

    axes[-1].set_xlabel("Time [s]", fontsize=21)
    # Use offset notation so the shared x-axis is readable
    axes[-1].xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    axes[-1].ticklabel_format(style="plain", axis="x")

    plt.tight_layout(pad=2.5, h_pad=3.0)

    save_path = os.path.join(
        MODEL_DIRECTORY,
        f"timeseries_10s_{VERSION_STR}.png"
    )
    fig.savefig(save_path, dpi=500, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
