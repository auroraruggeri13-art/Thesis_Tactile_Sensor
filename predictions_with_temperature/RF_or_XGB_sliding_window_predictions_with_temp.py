#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temperature-Aware LightGBM Sliding Window Predictions — v5.20

This is the temperature-aware variant of RF_or_XGB_sliding_window_predictions.py.
It extends the baseline feature vector with barometer temperature readings (t1..t6),
enabling the model to correct for thermal pressure drift at inference time rather
than relying solely on EMA post-processing.

Key differences from the baseline:
  - TEMP_COLS = ["t1", "t2", "t3", "t4", "t5", "t6"]
  - INCLUDE_TEMPERATURE flag (set False to ablate temperature as a feature)
  - build_window_features_with_temp: adds temperature windows + d1 to the feature vector
  - Graceful fallback when temperature columns are absent (INCLUDE_TEMPERATURE auto-disables)
  - All calls to build features go through the unified builder
  - Scaler (StandardScaler) covers temperature columns automatically

Feature vector layout per sample (window size W, 6 baro, 6 temp):
  [b1..b6 raw window (6*(W+1))]
  [b1..b6 d1  window (6*(W+1))]
  [b1..b6 d2  window (6*(W+1))]  ← only when USE_SECOND_DERIVATIVE=True
  [t1..t6 raw window (6*(W+1))]  ← only when INCLUDE_TEMPERATURE=True
  [t1..t6 d1  window (6*(W+1))]  ← only when INCLUDE_TEMPERATURE=True

sensor_version = 5.20
"""

import os
import sys
from pathlib import Path

# ── Path setup: shared utilities from source directory ────────────────────────
_THIS_DIR = Path(__file__).parent
_SRC_DIR  = _THIS_DIR.parent / "Barometers_Based_Tactile_Sensor"
sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from utils.metrics_utils import calculate_grouped_rmse, evaluate_constrained_region
from utils.plot_utils import plot_pred_vs_actual, plot_error_distributions
from utils.signal_utils import maybe_denoise, convert_sentinel_to_nan

# Optional libraries
try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
    HAVE_LIGHTGBM = True
except ImportError:
    HAVE_LIGHTGBM = False

try:
    from xgboost import XGBRegressor
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False

from sklearn.ensemble import RandomForestRegressor

# ============================================================
# ======================= CONFIG =============================
# ============================================================

DATA_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data"
sensor_version = 5.20
TRAIN_FILENAME      = f"train_data_v{sensor_version}.csv"
VALIDATION_FILENAME = f"validation_data_v{sensor_version}.csv"
TEST_FILENAME       = f"test_data_v{sensor_version}.csv"

TIME_COL    = "t"
BARO_COLS   = ["b1", "b2", "b3", "b4", "b5", "b6"]
TEMP_COLS   = ["t1", "t2", "t3", "t4", "t5", "t6"]
TARGET_COLS = ["x", "y", "fx", "fy", "fz"]

# ── Temperature feature flag ──────────────────────────────────────────────────
# Set False to run an ablation study without temperature features.
# Will auto-disable at runtime if t1..t6 are absent / all-NaN in the loaded data.
INCLUDE_TEMPERATURE = True

RANDOM_STATE = 42

# Window size study configuration
RUN_WINDOW_SIZE_STUDY    = False
WINDOW_SIZES             = [1, 5, 10, 15, 20]
WINDOW_SIZE              = 10   # ~0.1 s at 100 Hz

# Learning rate / estimator tuning
RUN_LEARNING_RATE_STUDY  = False
LEARNING_RATE_CONFIGS = [
    {'learning_rate': 0.01,  'n_estimators': 500},
    {'learning_rate': 0.03,  'n_estimators': 450},
    {'learning_rate': 0.05,  'n_estimators': 400},
    {'learning_rate': 0.075, 'n_estimators': 350},
    {'learning_rate': 0.1,   'n_estimators': 200},
    {'learning_rate': 0.3,   'n_estimators': 150},
    {'learning_rate': 0.5,   'n_estimators': 100},
    {'learning_rate': 0.8,   'n_estimators': 50},
]

# Default hyperparameters
LR_POSITION         = 0.03
N_EST_POSITION      = 450
LR_FORCE            = 0.075
N_EST_FORCE         = 350

# num_leaves tuning
RUN_NUM_LEAVES_STUDY    = False
NUM_LEAVES_CONFIGS      = [10, 100, 150, 200, 250, 300]
NUM_LEAVES_POSITION     = 200
NUM_LEAVES_FORCE        = 200

# Sentinel conversion
CONVERT_SENTINEL_TO_NAN = True
NO_CONTACT_SENTINEL     = -999.0

# Optional denoising (on baro channels only, before derivatives)
APPLY_DENOISING  = True
DENOISE_WINDOW   = 5

# Which models to train
USE_RANDOM_FOREST = False
USE_XGBOOST       = False
USE_LIGHTGBM      = True

# Include second derivatives for barometer features
USE_SECOND_DERIVATIVE = True


# ============================================================
# =============== FEATURE ENGINEERING ========================
# ============================================================

def _resolve_temp_cols(df: pd.DataFrame, requested_temp_cols: list) -> list:
    """Return temp columns that are actually present and non-all-NaN in df."""
    available = []
    for c in requested_temp_cols:
        if c in df.columns and df[c].notna().any():
            available.append(c)
    return available


def build_window_features(df: pd.DataFrame,
                          baro_cols: list,
                          time_col: str,
                          target_cols: list,
                          window_size: int,
                          temp_cols: list = None,
                          max_time_gap: float = 0.05,
                          use_second_derivative: bool = False):
    """
    Build sliding-window features from barometer pressures and (optionally)
    barometer temperatures.

    For each sample i in [window_size, N]:
      - Check that the window [i-W .. i] has no large time gaps
      - Features = windowed raw baro  (6 sensors × (W+1) steps)
                 + windowed baro d1   (6 sensors × (W+1) steps)
                 + windowed baro d2   (if use_second_derivative)
                 + windowed raw temp  (n_temp × (W+1) steps, if temp_cols non-empty)
                 + windowed temp d1   (n_temp × (W+1) steps, if temp_cols non-empty)
      - Target = target values at index i

    Parameters
    ----------
    df : pd.DataFrame
        Sorted timeseries with baro + optional temp + target columns.
    baro_cols : list[str]
        Barometer pressure column names (e.g. ["b1".."b6"]).
    time_col : str
        Column name for timestamps.
    target_cols : list[str]
        Target column names.
    window_size : int
        Number of *past* samples to include (window spans [i-W .. i]).
    temp_cols : list[str] or None
        Temperature column names. Pass [] or None to disable temperature features.
    max_time_gap : float
        Windows that span this time gap (s) are skipped (file boundary guard).
    use_second_derivative : bool
        If True, add second baro derivatives to the feature vector.

    Returns
    -------
    X            : np.ndarray  (M, n_features)
    y            : np.ndarray  (M, n_targets)
    center_df    : pd.DataFrame  rows corresponding to the current (latest) sample
    feature_names: list[str]
    """
    if temp_cols is None:
        temp_cols = []

    df = df.sort_values(time_col).reset_index(drop=True).copy()

    # Denoise barometer channels only
    df = maybe_denoise(df, baro_cols,
                       apply_denoising=APPLY_DENOISING,
                       denoise_window=DENOISE_WINDOW)

    # Compute baro derivatives
    for col in baro_cols:
        d1 = df[col].diff()
        df[f"{col}_d1"] = d1.fillna(0.0)
        if use_second_derivative:
            df[f"{col}_d2"] = d1.diff().fillna(0.0)

    # Compute temperature first derivative (temperature changes slowly,
    # but its rate captures thermal ramp-up which correlates with pressure drift)
    for col in temp_cols:
        df[f"{col}_d1"] = df[col].diff().fillna(0.0)

    N = len(df)
    W = window_size
    if N <= W:
        raise ValueError(f"Not enough samples ({N}) for window size {W}")

    # Build feature name list (for diagnostics)
    feature_names = []
    for col in baro_cols:
        for offset in range(-W, 1):
            feature_names.append(f"{col}@{offset}")
        for offset in range(-W, 1):
            feature_names.append(f"{col}_d1@{offset}")
        if use_second_derivative:
            for offset in range(-W, 1):
                feature_names.append(f"{col}_d2@{offset}")
    for col in temp_cols:
        for offset in range(-W, 1):
            feature_names.append(f"{col}@{offset}")
        for offset in range(-W, 1):
            feature_names.append(f"{col}_d1@{offset}")

    # Pre-extract numpy arrays for speed
    time_vals  = df[time_col].values
    baro_data  = df[baro_cols].values
    d1_data    = df[[f"{c}_d1" for c in baro_cols]].values
    d2_data    = df[[f"{c}_d2" for c in baro_cols]].values if use_second_derivative else None
    temp_data  = df[temp_cols].values           if temp_cols else None
    temp_d1    = df[[f"{c}_d1" for c in temp_cols]].values if temp_cols else None

    X_list       = []
    y_list       = []
    valid_indices = []
    skipped      = 0

    for current_idx in range(W, N):
        start = current_idx - W
        end   = current_idx + 1

        # Skip windows that span a file boundary
        window_times = time_vals[start:end]
        if np.max(np.diff(window_times)) > max_time_gap:
            skipped += 1
            continue

        parts = [
            baro_data[start:end, :].flatten(),
            d1_data[start:end,   :].flatten(),
        ]
        if use_second_derivative:
            parts.append(d2_data[start:end, :].flatten())
        if temp_cols:
            parts.append(temp_data[start:end, :].flatten())
            parts.append(temp_d1[start:end,   :].flatten())

        X_list.append(np.concatenate(parts))
        y_list.append(df.loc[current_idx, target_cols].values)
        valid_indices.append(current_idx)

    print(f"  Built {len(X_list)} valid windows, skipped {skipped} boundary windows")

    X         = np.array(X_list)
    y         = np.array(y_list)
    center_df = df.iloc[valid_indices].reset_index(drop=True)

    return X, y, center_df, feature_names


# ============================================================
# ======================= PLOTTING ===========================
# ============================================================

def plot_barometer_with_derivatives(df, baro_col, n_samples=800, title_suffix=""):
    df = maybe_denoise(df, [baro_col],
                       apply_denoising=APPLY_DENOISING,
                       denoise_window=DENOISE_WINDOW).copy()
    df[f"{baro_col}_d1"] = df[baro_col].diff().fillna(0.0)
    df[f"{baro_col}_d2"] = df[f"{baro_col}_d1"].diff().fillna(0.0)
    df_sub = df.iloc[:n_samples].copy()

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df_sub[TIME_COL], df_sub[baro_col], label=f"{baro_col} raw")
    ax1.set_xlabel("Time"); ax1.set_ylabel("Pressure"); ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(df_sub[TIME_COL], df_sub[f"{baro_col}_d1"], alpha=0.6, label=f"{baro_col}_d1")
    ax2.plot(df_sub[TIME_COL], df_sub[f"{baro_col}_d2"], alpha=0.4, label=f"{baro_col}_d2")
    ax2.set_ylabel("Derivatives")
    lines  = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right")
    plt.title(f"{baro_col} with first & second derivatives {title_suffix}")
    plt.tight_layout()
    return fig


def plot_loss_curves(loss_history, model_name, target_cols=None):
    if isinstance(loss_history, dict) and 'train' not in loss_history:
        n_targets = len(loss_history)
        fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))
        if n_targets == 1:
            axes = [axes]
        for i, (target, losses) in enumerate(loss_history.items()):
            ax = axes[i]
            ax.plot(losses['train'], label='Training Loss',   linewidth=2, color='#005c7f')
            if 'valid' in losses:
                ax.plot(losses['valid'], label='Validation Loss', linewidth=2, color='#44b155')
            ax.set_xlabel('Iteration'); ax.set_ylabel('MSE Loss')
            ax.set_title(f'{model_name} - {target}'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.suptitle(f'{model_name} Training Loss Curves', fontsize=13, y=1.02)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        if 'train' in loss_history:
            ax.plot(loss_history['train'], label='Training Loss',   linewidth=2, color='#005c7f')
        if 'valid' in loss_history:
            ax.plot(loss_history['valid'], label='Validation Loss', linewidth=2, color='#44b155')
        ax.set_xlabel('Iteration'); ax.set_ylabel('MSE Loss')
        ax.set_title(f'{model_name} Training Loss Curve'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_window_size_comparison(window_sizes, results, target_cols):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    contact_rmse = [results[ws]['contact_euclidean_rmse'] for ws in window_sizes]
    force_rmse   = [results[ws]['force_euclidean_rmse']   for ws in window_sizes]

    color1 = '#005c7f'
    ax1.set_xlabel('Window Size (samples)', fontsize=12)
    ax1.set_ylabel('Contact Position RMSE (mm)', fontsize=12, color=color1)
    line1 = ax1.plot(window_sizes, contact_rmse, marker='o', linewidth=2.5,
                     markersize=9, color=color1, label='Contact Position Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3); ax1.set_xticks(window_sizes)
    best_c = np.argmin(contact_rmse)
    ax1.scatter([window_sizes[best_c]], [contact_rmse[best_c]],
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)

    ax2 = ax1.twinx()
    color2 = '#44b155'
    ax2.set_ylabel('3-DOF Force RMSE (N)', fontsize=12, color=color2)
    line2 = ax2.plot(window_sizes, force_rmse, marker='s', linewidth=2.5,
                     markersize=9, color=color2, label='Force Error')
    ax2.tick_params(axis='y', labelcolor=color2)
    best_f = np.argmin(force_rmse)
    ax2.scatter([window_sizes[best_f]], [force_rmse[best_f]],
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)

    lines  = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=2, fontsize=11, frameon=True, shadow=True)
    plt.title('Window Size Hyperparameter Study (with temperature)', fontsize=14,
              fontweight='bold', pad=15)
    fig.tight_layout()
    return fig


def plot_learning_rate_comparison(configs, results_position, results_force):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    config_labels = [f"LR={c['learning_rate']}\nN={c['n_estimators']}" for c in configs]
    x_pos         = np.arange(len(configs))
    contact_rmse  = [results_position[i]['contact_euclidean_rmse'] for i in range(len(configs))]
    force_rmse    = [results_force[i]['force_euclidean_rmse']      for i in range(len(configs))]

    color1 = '#005c7f'
    ax1.set_xlabel('Learning Rate Configuration', fontsize=12)
    ax1.set_ylabel('Contact Position RMSE (mm)', fontsize=12, color=color1)
    line1 = ax1.plot(x_pos, contact_rmse, marker='o', linewidth=2.5,
                     markersize=9, color=color1, label='Contact Position Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(x_pos); ax1.set_xticklabels(config_labels, fontsize=9)
    best_c = np.argmin(contact_rmse)
    ax1.scatter([x_pos[best_c]], [contact_rmse[best_c]],
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)

    ax2    = ax1.twinx()
    color2 = '#44b155'
    ax2.set_ylabel('3-DOF Force RMSE (N)', fontsize=12, color=color2)
    line2 = ax2.plot(x_pos, force_rmse, marker='s', linewidth=2.5,
                     markersize=9, color=color2, label='Force Error')
    ax2.tick_params(axis='y', labelcolor=color2)
    best_f = np.argmin(force_rmse)
    ax2.scatter([x_pos[best_f]], [force_rmse[best_f]],
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)

    lines  = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, fontsize=11, frameon=True, shadow=True)
    plt.title('Learning Rate & N_Estimators Hyperparameter Study',
              fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    return fig


def plot_num_leaves_comparison(num_leaves_list, results_position, results_force):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x_pos        = np.arange(len(num_leaves_list))
    contact_rmse = [results_position[i]['contact_euclidean_rmse'] for i in range(len(num_leaves_list))]
    force_rmse   = [results_force[i]['force_euclidean_rmse']      for i in range(len(num_leaves_list))]

    color1 = '#005c7f'
    ax1.set_xlabel('num_leaves', fontsize=12)
    ax1.set_ylabel('Contact Position RMSE (mm)', fontsize=12, color=color1)
    line1 = ax1.plot(x_pos, contact_rmse, marker='o', linewidth=2.5,
                     markersize=9, color=color1, label='Contact Position Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(nl) for nl in num_leaves_list], fontsize=10)
    best_c = np.argmin(contact_rmse)
    ax1.scatter([x_pos[best_c]], [contact_rmse[best_c]],
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)

    ax2    = ax1.twinx()
    color2 = '#44b155'
    ax2.set_ylabel('3-DOF Force RMSE (N)', fontsize=12, color=color2)
    line2 = ax2.plot(x_pos, force_rmse, marker='s', linewidth=2.5,
                     markersize=9, color=color2, label='Force Error')
    ax2.tick_params(axis='y', labelcolor=color2)
    best_f = np.argmin(force_rmse)
    ax2.scatter([x_pos[best_f]], [force_rmse[best_f]],
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)

    lines  = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=2, fontsize=11, frameon=True, shadow=True)
    plt.title('num_leaves Hyperparameter Study', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    return fig


# ============================================================
# ======================== HELPERS ===========================
# ============================================================

def _build_features(df, active_temp_cols, window_size, use_second_derivative=None):
    """Thin wrapper so all call-sites stay consistent."""
    if use_second_derivative is None:
        use_second_derivative = USE_SECOND_DERIVATIVE
    return build_window_features(
        df,
        baro_cols=BARO_COLS,
        time_col=TIME_COL,
        target_cols=TARGET_COLS,
        window_size=window_size,
        temp_cols=active_temp_cols,
        use_second_derivative=use_second_derivative,
    )


def _train_lgbm(X_train, y_train, X_val, y_val, lr, n_est, num_leaves,
                target_name, target_type_label):
    """Train a single LGBMRegressor and return (model, loss_dict)."""
    print(f"    {target_name} ({target_type_label})  LR={lr}, N={n_est}, leaves={num_leaves}")
    model = LGBMRegressor(
        n_estimators=n_est,
        learning_rate=lr,
        num_leaves=num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        metric='mse',
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    loss = {
        'train': model.evals_result_['train']['l2'],
        'valid': model.evals_result_['valid']['l2'],
    }
    print(f"      train MSE {loss['train'][-1]:.4f}  |  val MSE {loss['valid'][-1]:.4f}")
    return model, loss


# ============================================================
# ======================== MAIN ==============================
# ============================================================

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    train_path = os.path.join(DATA_DIRECTORY, TRAIN_FILENAME)
    val_path   = os.path.join(DATA_DIRECTORY, VALIDATION_FILENAME)
    test_path  = os.path.join(DATA_DIRECTORY, TEST_FILENAME)

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Data file not found: {p}")

    train_df = pd.read_csv(train_path,      skipinitialspace=True)
    val_df   = pd.read_csv(val_path,        skipinitialspace=True)
    test_df  = pd.read_csv(test_path,       skipinitialspace=True)

    for df in [train_df, val_df, test_df]:
        df.columns = df.columns.str.strip()

    print("Train  columns:", train_df.columns.tolist())
    print("Val    columns:", val_df.columns.tolist())
    print("Test   columns:", test_df.columns.tolist())
    print(f"Rows — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ── Resolve active temperature columns ────────────────────────────────────
    active_temp_cols: list = []
    if INCLUDE_TEMPERATURE:
        active_temp_cols = _resolve_temp_cols(train_df, TEMP_COLS)
        if not active_temp_cols:
            print("\n[WARN] INCLUDE_TEMPERATURE=True but no valid temperature columns "
                  "found in training data. Falling back to pressure-only mode.")
        else:
            print(f"\n[OK]  Temperature features active: {active_temp_cols}")
            # Validate presence in val/test too (warn if mismatch)
            for split_name, split_df in [("val", val_df), ("test", test_df)]:
                missing_in = [c for c in active_temp_cols
                              if c not in split_df.columns or split_df[c].isna().all()]
                if missing_in:
                    print(f"[WARN] Temperature columns {missing_in} missing/all-NaN "
                          f"in {split_name} split — those samples will be dropped.")
    else:
        print("\n[INFO] INCLUDE_TEMPERATURE=False — running pressure-only baseline.")

    # ── Sentinel → NaN ────────────────────────────────────────────────────────
    if CONVERT_SENTINEL_TO_NAN:
        train_df = convert_sentinel_to_nan(train_df, TARGET_COLS, NO_CONTACT_SENTINEL)
        val_df   = convert_sentinel_to_nan(val_df,   TARGET_COLS, NO_CONTACT_SENTINEL)
        test_df  = convert_sentinel_to_nan(test_df,  TARGET_COLS, NO_CONTACT_SENTINEL)

    # ── Drop rows with NaN in required columns ─────────────────────────────────
    needed_cols = [TIME_COL] + BARO_COLS + TARGET_COLS + active_temp_cols
    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_before = len(df)
        df.dropna(subset=needed_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
        dropped = n_before - len(df)
        if dropped:
            print(f"  Dropped {dropped} rows with NaN in {df_name}")

    # ── Feature description summary ───────────────────────────────────────────
    n_baro_feat = len(BARO_COLS) * (WINDOW_SIZE + 1) * (3 if USE_SECOND_DERIVATIVE else 2)
    n_temp_feat = len(active_temp_cols) * (WINDOW_SIZE + 1) * 2 if active_temp_cols else 0
    print(f"\nExpected features: {n_baro_feat} (baro) + {n_temp_feat} (temp) "
          f"= {n_baro_feat + n_temp_feat} total")

    # ── Window Size Study ─────────────────────────────────────────────────────
    if RUN_WINDOW_SIZE_STUDY:
        print("\n" + "="*70)
        print("RUNNING WINDOW SIZE HYPERPARAMETER STUDY  (with temperature)")
        print(f"Testing window sizes: {WINDOW_SIZES}")
        print("="*70)

        window_size_results = {}

        for ws in WINDOW_SIZES:
            print(f"\n{'='*70}\nWindow Size: {ws}\n{'='*70}")

            X_tr, y_tr, _, _ = _build_features(train_df, active_temp_cols, ws)
            X_vl, y_vl, _, _ = _build_features(val_df,   active_temp_cols, ws)
            X_te, y_te, _, _ = _build_features(test_df,  active_temp_cols, ws)

            scaler_ws = StandardScaler()
            X_tr = scaler_ws.fit_transform(X_tr)
            X_vl = scaler_ws.transform(X_vl)
            X_te = scaler_ws.transform(X_te)

            if USE_LIGHTGBM and HAVE_LIGHTGBM:
                preds = []
                for i, target in enumerate(TARGET_COLS):
                    lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=31,
                                         subsample=0.8, colsample_bytree=0.8,
                                         random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
                                         metric='mse')
                    lgbm.fit(X_tr, y_tr[:, i],
                             eval_set=[(X_vl, y_vl[:, i])], eval_names=['valid'])
                    preds.append(lgbm.predict(X_vl))
                y_pred_ws = np.column_stack(preds)
            elif USE_XGBOOST and HAVE_XGBOOST:
                preds = []
                for i in range(len(TARGET_COLS)):
                    xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                                       subsample=0.8, colsample_bytree=0.8,
                                       random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
                                       eval_metric='rmse')
                    xgb.fit(X_tr, y_tr[:, i], eval_set=[(X_vl, y_vl[:, i])], verbose=False)
                    preds.append(xgb.predict(X_vl))
                y_pred_ws = np.column_stack(preds)
            elif USE_RANDOM_FOREST:
                preds = []
                for i in range(len(TARGET_COLS)):
                    rf = RandomForestRegressor(n_estimators=100, max_depth=20,
                                               min_samples_split=5, min_samples_leaf=2,
                                               random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
                    rf.fit(X_tr, y_tr[:, i])
                    preds.append(rf.predict(X_vl))
                y_pred_ws = np.column_stack(preds)
            else:
                raise ValueError("No model enabled for window size study.")

            metrics = calculate_grouped_rmse(y_vl, y_pred_ws, TARGET_COLS,
                                             title_suffix=f"Window Size = {ws}",
                                             return_metrics=True)
            window_size_results[ws] = metrics

        fig_comparison = plot_window_size_comparison(WINDOW_SIZES, window_size_results, TARGET_COLS)
        save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
        os.makedirs(save_dir, exist_ok=True)
        version = f'v{sensor_version:.2f}'
        fig_comparison.savefig(
            os.path.join(save_dir, f'window_size_comparison_with_temp_{version}.png'),
            bbox_inches='tight', dpi=300)
        plt.show()
        print("\nWindow size study complete. Exiting.")
        return

    # ── Learning Rate Study ───────────────────────────────────────────────────
    if RUN_LEARNING_RATE_STUDY:
        print("\n" + "="*70)
        print("RUNNING LEARNING RATE & N_ESTIMATORS STUDY  (with temperature)")
        print("="*70)

        X_tr, y_tr, _, _ = _build_features(train_df, active_temp_cols, WINDOW_SIZE)
        X_vl, y_vl, _, _ = _build_features(val_df,   active_temp_cols, WINDOW_SIZE)
        X_te, y_te, _, _ = _build_features(test_df,  active_temp_cols, WINDOW_SIZE)
        scaler_lr = StandardScaler()
        X_tr = scaler_lr.fit_transform(X_tr)
        X_vl = scaler_lr.transform(X_vl)
        X_te = scaler_lr.transform(X_te)

        lr_results_position = {}
        lr_results_force    = {}

        for config_idx, config in enumerate(LEARNING_RATE_CONFIGS):
            lr   = config['learning_rate']
            n_est = config['n_estimators']
            print(f"\nConfig {config_idx + 1}/{len(LEARNING_RATE_CONFIGS)}: LR={lr}, N={n_est}")

            if not (USE_LIGHTGBM and HAVE_LIGHTGBM):
                raise ValueError("LightGBM must be enabled for LR study.")

            pos_preds = []
            for t in ['x', 'y']:
                ti = TARGET_COLS.index(t)
                m = LGBMRegressor(n_estimators=n_est, learning_rate=lr, num_leaves=31,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, metric='mse')
                m.fit(X_tr, y_tr[:, ti], eval_set=[(X_vl, y_vl[:, ti])], eval_names=['valid'])
                pos_preds.append(m.predict(X_vl))
            y_pred_pos = np.column_stack(pos_preds)
            y_val_pos  = y_vl[:, [TARGET_COLS.index('x'), TARGET_COLS.index('y')]]
            lr_results_position[config_idx] = calculate_grouped_rmse(
                y_val_pos, y_pred_pos, ['x', 'y'],
                title_suffix=f"Pos LR={lr}", return_metrics=True)

            frc_preds = []
            for t in ['fx', 'fy', 'fz']:
                ti = TARGET_COLS.index(t)
                m = LGBMRegressor(n_estimators=n_est, learning_rate=lr, num_leaves=31,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, metric='mse')
                m.fit(X_tr, y_tr[:, ti], eval_set=[(X_vl, y_vl[:, ti])], eval_names=['valid'])
                frc_preds.append(m.predict(X_vl))
            y_pred_frc = np.column_stack(frc_preds)
            y_val_frc  = y_vl[:, [TARGET_COLS.index(c) for c in ['fx', 'fy', 'fz']]]
            lr_results_force[config_idx] = calculate_grouped_rmse(
                y_val_frc, y_pred_frc, ['fx', 'fy', 'fz'],
                title_suffix=f"Force LR={lr}", return_metrics=True)

        fig_lr = plot_learning_rate_comparison(LEARNING_RATE_CONFIGS,
                                               lr_results_position, lr_results_force)
        save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
        os.makedirs(save_dir, exist_ok=True)
        version = f'v{sensor_version:.2f}'
        fig_lr.savefig(
            os.path.join(save_dir, f'learning_rate_comparison_with_temp_{version}.png'),
            bbox_inches='tight', dpi=300)
        plt.show()
        print("\nLearning rate study complete. Exiting.")
        return

    # ── num_leaves Study ─────────────────────────────────────────────────────
    if RUN_NUM_LEAVES_STUDY:
        print("\n" + "="*70)
        print("RUNNING NUM_LEAVES STUDY  (with temperature)")
        print("="*70)

        X_tr, y_tr, _, _ = _build_features(train_df, active_temp_cols, WINDOW_SIZE)
        X_vl, y_vl, _, _ = _build_features(val_df,   active_temp_cols, WINDOW_SIZE)
        X_te, y_te, _, _ = _build_features(test_df,  active_temp_cols, WINDOW_SIZE)
        scaler_nl = StandardScaler()
        X_tr = scaler_nl.fit_transform(X_tr)
        X_vl = scaler_nl.transform(X_vl)
        X_te = scaler_nl.transform(X_te)

        nl_results_position = {}
        nl_results_force    = {}

        for nl_idx, num_leaves_val in enumerate(NUM_LEAVES_CONFIGS):
            print(f"\nnum_leaves = {num_leaves_val}")

            if not (USE_LIGHTGBM and HAVE_LIGHTGBM):
                raise ValueError("LightGBM must be enabled for num_leaves study.")

            pos_preds = []
            for t in ['x', 'y']:
                ti = TARGET_COLS.index(t)
                m = LGBMRegressor(n_estimators=N_EST_POSITION, learning_rate=LR_POSITION,
                                  num_leaves=num_leaves_val, subsample=0.8, colsample_bytree=0.8,
                                  random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, metric='mse')
                m.fit(X_tr, y_tr[:, ti], eval_set=[(X_vl, y_vl[:, ti])], eval_names=['valid'])
                pos_preds.append(m.predict(X_vl))
            y_pred_pos = np.column_stack(pos_preds)
            y_val_pos  = y_vl[:, [TARGET_COLS.index('x'), TARGET_COLS.index('y')]]
            nl_results_position[nl_idx] = calculate_grouped_rmse(
                y_val_pos, y_pred_pos, ['x', 'y'],
                title_suffix=f"Pos leaves={num_leaves_val}", return_metrics=True)

            frc_preds = []
            for t in ['fx', 'fy', 'fz']:
                ti = TARGET_COLS.index(t)
                m = LGBMRegressor(n_estimators=N_EST_FORCE, learning_rate=LR_FORCE,
                                  num_leaves=num_leaves_val, subsample=0.8, colsample_bytree=0.8,
                                  random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, metric='mse')
                m.fit(X_tr, y_tr[:, ti], eval_set=[(X_vl, y_vl[:, ti])], eval_names=['valid'])
                frc_preds.append(m.predict(X_vl))
            y_pred_frc = np.column_stack(frc_preds)
            y_val_frc  = y_vl[:, [TARGET_COLS.index(c) for c in ['fx', 'fy', 'fz']]]
            nl_results_force[nl_idx] = calculate_grouped_rmse(
                y_val_frc, y_pred_frc, ['fx', 'fy', 'fz'],
                title_suffix=f"Force leaves={num_leaves_val}", return_metrics=True)

        best_nl_c = NUM_LEAVES_CONFIGS[min(range(len(NUM_LEAVES_CONFIGS)),
                                           key=lambda i: nl_results_position[i]['contact_euclidean_rmse'])]
        best_nl_f = NUM_LEAVES_CONFIGS[min(range(len(NUM_LEAVES_CONFIGS)),
                                           key=lambda i: nl_results_force[i]['force_euclidean_rmse'])]
        print(f"\nBest num_leaves for position: {best_nl_c}")
        print(f"Best num_leaves for force:    {best_nl_f}")

        fig_nl = plot_num_leaves_comparison(NUM_LEAVES_CONFIGS, nl_results_position, nl_results_force)
        save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
        os.makedirs(save_dir, exist_ok=True)
        version = f'v{sensor_version:.2f}'
        fig_nl.savefig(
            os.path.join(save_dir, f'num_leaves_comparison_with_temp_{version}.png'),
            bbox_inches='tight', dpi=300)
        plt.show()
        print("\nnum_leaves study complete. Exiting.")
        return

    # ── Single Window Size Run ────────────────────────────────────────────────
    print(f"\nUsing window size: {WINDOW_SIZE}")
    print("Building windowed feature matrices...")

    X_train, y_train, train_center, feat_names = _build_features(
        train_df, active_temp_cols, WINDOW_SIZE)
    X_val,   y_val,   val_center,   _          = _build_features(
        val_df,   active_temp_cols, WINDOW_SIZE)
    X_test,  y_test,  test_center,  _          = _build_features(
        test_df,  active_temp_cols, WINDOW_SIZE)

    print(f"Feature count: {X_train.shape[1]}  "
          f"(window={WINDOW_SIZE}, baro+d1{'d2' if USE_SECOND_DERIVATIVE else ''}"
          f"{', temp+d1' if active_temp_cols else ''})")
    print(f"Feature matrix shapes: Train={X_train.shape}, Val={X_val.shape}, "
          f"Test={X_test.shape}")

    # ── Normalize ─────────────────────────────────────────────────────────────
    print("\nNormalizing with StandardScaler (covers baro + temp uniformly)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    print("Normalization complete (mean=0, std=1 per feature).")

    # ── Train models ──────────────────────────────────────────────────────────
    models         = {}
    y_preds        = {}
    loss_histories = {}

    if USE_RANDOM_FOREST:
        print("\nTraining Random Forest...")
        rf_models = []
        rf_preds  = []
        for i, target in enumerate(TARGET_COLS):
            rf = RandomForestRegressor(n_estimators=100, max_depth=20,
                                       min_samples_split=5, min_samples_leaf=2,
                                       random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
            rf.fit(X_train, y_train[:, i])
            rf_models.append(rf)
            rf_preds.append(rf.predict(X_test))
        models["RandomForest"]  = rf_models
        y_preds["RandomForest"] = np.column_stack(rf_preds)

    if USE_XGBOOST and HAVE_XGBOOST:
        print("\nTraining XGBoost...")
        xgb_models = []
        xgb_preds  = []
        xgb_loss   = {}
        for i, target in enumerate(TARGET_COLS):
            xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
                               eval_metric='rmse')
            xgb.fit(X_train, y_train[:, i],
                    eval_set=[(X_train, y_train[:, i]), (X_val, y_val[:, i])],
                    verbose=False, early_stopping_rounds=30)
            xgb_models.append(xgb)
            xgb_preds.append(xgb.predict(X_test))
            xgb_loss[target] = {
                'train': xgb.evals_result()['validation_0']['rmse'],
                'valid': xgb.evals_result()['validation_1']['rmse'],
            }
        models["XGBoost"]       = xgb_models
        y_preds["XGBoost"]      = np.column_stack(xgb_preds)
        loss_histories["XGBoost"] = xgb_loss

    if USE_LIGHTGBM and HAVE_LIGHTGBM:
        print("\nTraining LightGBM (temperature-aware)...")
        print(f"  Position (x,y): LR={LR_POSITION}, N={N_EST_POSITION}, "
              f"leaves={NUM_LEAVES_POSITION}")
        print(f"  Force (fx,fy,fz): LR={LR_FORCE}, N={N_EST_FORCE}, "
              f"leaves={NUM_LEAVES_FORCE}")

        lgbm_models = []
        lgbm_preds  = []
        lgbm_loss   = {}

        for i, target in enumerate(TARGET_COLS):
            if target in ['x', 'y']:
                lr, n_est, num_leaves_val, ttype = LR_POSITION, N_EST_POSITION, NUM_LEAVES_POSITION, "position"
            else:
                lr, n_est, num_leaves_val, ttype = LR_FORCE, N_EST_FORCE, NUM_LEAVES_FORCE, "force"

            model, loss = _train_lgbm(
                X_train, y_train[:, i],
                X_val,   y_val[:, i],
                lr, n_est, num_leaves_val, target, ttype,
            )
            lgbm_models.append(model)
            lgbm_preds.append(model.predict(X_test))
            lgbm_loss[target] = loss

        models["LightGBM"]         = lgbm_models
        y_preds["LightGBM"]        = np.column_stack(lgbm_preds)
        loss_histories["LightGBM"] = lgbm_loss

    # ── Metrics ───────────────────────────────────────────────────────────────
    for name, y_hat in y_preds.items():
        d2_suffix   = " + d2"  if USE_SECOND_DERIVATIVE else ""
        temp_suffix = " + temp" if active_temp_cols     else ""
        print(f"\n=== Performance ({name}) with window + gradients{d2_suffix}{temp_suffix} ===")
        for i, col in enumerate(TARGET_COLS):
            mae = mean_absolute_error(y_test[:, i], y_hat[:, i])
            r2  = r2_score(y_test[:, i], y_hat[:, i])
            print(f"  {col:>3} | MAE: {mae:6.3f} | R²: {r2:6.3f}")

    # ── Save models and plots ─────────────────────────────────────────────────
    print("\n" + "="*70 + "\nSAVING MODELS AND PLOTS\n" + "="*70)

    save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
    save_dir_comparison = os.path.join(save_dir, "comparison_plots")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_comparison, exist_ok=True)

    version = f'v{sensor_version:.3f}_with_temp'

    # Save scaler
    scaler_path = os.path.join(save_dir, f'scaler_sliding_window_{version}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved: {scaler_path}")

    # Save active_temp_cols alongside models (needed for inference)
    meta_path = os.path.join(save_dir, f'feature_meta_{version}.pkl')
    meta = {
        'baro_cols':         BARO_COLS,
        'temp_cols':         active_temp_cols,
        'window_size':       WINDOW_SIZE,
        'use_second_deriv':  USE_SECOND_DERIVATIVE,
        'sensor_version':    sensor_version,
        'feature_names':     feat_names,
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Feature metadata saved: {meta_path}")

    # Save models
    for name, model in models.items():
        model_path = os.path.join(
            save_dir, f'{name.lower().replace(" ","_")}_sliding_window_model_{version}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"{name} model saved: {model_path}")

    # 1) Barometer + derivatives plot
    fig_baro = plot_barometer_with_derivatives(train_df, BARO_COLS[0], title_suffix="(train)")
    fig_baro.savefig(os.path.join(save_dir, f'sliding_window_barometer_{version}.png'),
                     bbox_inches='tight', dpi=300)
    plt.close()

    # 2) Loss curves
    for model_name, loss_hist in loss_histories.items():
        fig_loss = plot_loss_curves(loss_hist, model_name, TARGET_COLS)
        fig_loss.savefig(
            os.path.join(save_dir, f'loss_curve_{model_name.lower()}_{version}.png'),
            bbox_inches='tight', dpi=300)
        plt.show()
        print(f"{model_name} loss curve saved.")

    # 3) Pred vs actual + error distribution
    for model_name, y_hat in y_preds.items():
        fig = plot_pred_vs_actual(y_test, y_hat, TARGET_COLS,
                                  title_suffix=model_name, scatter_color='#292f56')
        fig.savefig(
            os.path.join(save_dir,
                         f'sliding_window_predictions_{model_name.lower()}_{version}.png'),
            bbox_inches='tight', dpi=300)
        plt.show()
        fig.savefig(
            os.path.join(save_dir_comparison,
                         f'actual_vs_prediction_{model_name.lower()}_{version}.png'),
            bbox_inches='tight', dpi=300)
        print(f"{model_name} prediction plot saved.")

        # Per-target RMSE summary
        sep = "=" * 50
        print(f"\n{sep}\n{model_name} — Test Set RMSE\n{sep}")
        for j, col in enumerate(TARGET_COLS):
            rmse = np.sqrt(np.mean((y_test[:, j] - y_hat[:, j]) ** 2))
            unit = "mm" if col in ["x", "y"] else "N"
            print(f"  {col.ljust(6)} RMSE: {rmse:.4f} {unit}")
        pos_idx = [i for i, c in enumerate(TARGET_COLS) if c in ["x", "y"]]
        frc_idx = [i for i, c in enumerate(TARGET_COLS) if c in ["fx", "fy", "fz"]]
        if pos_idx:
            euc_pos = np.sqrt(sum(np.mean((y_test[:, i] - y_hat[:, i])**2) for i in pos_idx))
            print(f"  Position Euclidean RMSE : {euc_pos:.4f} mm")
        if frc_idx:
            euc_frc = np.sqrt(sum(np.mean((y_test[:, i] - y_hat[:, i])**2) for i in frc_idx))
            print(f"  Force    Euclidean RMSE : {euc_frc:.4f} N")
        print(sep + "\n")

        fig_err = plot_error_distributions(y_test, y_hat, TARGET_COLS,
                                           title_suffix=model_name)
        fig_err.savefig(
            os.path.join(save_dir, f'error_distribution_{model_name.lower()}_{version}.png'),
            bbox_inches='tight', dpi=300)
        plt.close()

        calculate_grouped_rmse(y_test, y_hat, TARGET_COLS,
                               title_suffix=f"({model_name} — All Data)")

    # 4) LightGBM tree visualization
    if "LightGBM" in models:
        try:
            import graphviz  # noqa: F401
            print("\nGenerating LightGBM tree visualizations...")
            for i, target in enumerate(TARGET_COLS):
                fig, ax = plt.subplots(figsize=(20, 10))
                lgb.plot_tree(models["LightGBM"][i].booster_, tree_index=0, ax=ax,
                              show_info=['split_gain'])
                plt.title(f'LightGBM Tree — {target} (with temp) — tree 0', fontsize=14)
                plt.tight_layout()
                tree_path = os.path.join(save_dir, f'lightgbm_tree_{target}_{version}.png')
                fig.savefig(tree_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
                print(f"  Tree saved: {tree_path}")
        except ImportError:
            print("graphviz not installed — skipping tree visualization.")
        except Exception as e:
            print(f"Tree visualization skipped: {e}")

    print(f"\nAll outputs saved to: {save_dir}")
    temp_info = f"active temp cols: {active_temp_cols}" if active_temp_cols else "pressure-only"
    print(f"Pipeline config: sensor_version={sensor_version}, {temp_info}, "
          f"window={WINDOW_SIZE}, d2={USE_SECOND_DERIVATIVE}")
    print("Done.")


if __name__ == "__main__":
    main()
