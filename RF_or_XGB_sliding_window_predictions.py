#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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
sensor_version = 5.103
TRAIN_FILENAME = f"train_data_v{sensor_version}.csv"
VALIDATION_FILENAME = f"validation_data_v{sensor_version}.csv"
TEST_FILENAME  = f"test_data_v{sensor_version}.csv"

TIME_COL   = "t"
BARO_COLS  = ["b1", "b2", "b3", "b4", "b5", "b6"]
TARGET_COLS = ["x", "y", "fz","tz"]

RANDOM_STATE = 42

# Window size study configuration
RUN_WINDOW_SIZE_STUDY = False  # Set to True to test multiple window sizes and generate comparison plots
WINDOW_SIZES = [1, 5, 10, 15, 20]  # Window sizes to test during hyperparameter search

# Sliding window size (in samples). Looks at past WINDOW_SIZE samples.
# Used when RUN_WINDOW_SIZE_STUDY = False
WINDOW_SIZE = 10   # 10 past samples (~0.1s at 100 Hz)

# Learning rate and n_estimators tuning configuration
RUN_LEARNING_RATE_STUDY = False  # Set to True to tune learning rate and n_estimators
LEARNING_RATE_CONFIGS = [
    {'learning_rate': 0.01, 'n_estimators': 500},
    {'learning_rate': 0.03, 'n_estimators': 450},
    {'learning_rate': 0.05, 'n_estimators': 400},
    {'learning_rate': 0.075, 'n_estimators': 350},
    {'learning_rate': 0.1, 'n_estimators': 300},
    {'learning_rate': 0.15, 'n_estimators': 250},
    {'learning_rate': 0.2, 'n_estimators': 200},
    {'learning_rate': 0.25, 'n_estimators': 180},
    {'learning_rate': 0.3, 'n_estimators': 150},
]

# Default learning rate settings (used when RUN_LEARNING_RATE_STUDY = False)
# Separate for position and force predictions
LR_POSITION = 0.03  # Learning rate for x, y
N_EST_POSITION = 450  # Number of estimators for x, y
LR_FORCE = 0.075  # Learning rate for fx, fy, fz
N_EST_FORCE = 350  # Number of estimators for fx, fy, fz

# num_leaves tuning configuration
RUN_NUM_LEAVES_STUDY = False  # Set to True to tune num_leaves
NUM_LEAVES_CONFIGS = [10, 100, 150, 200, 250, 300]  # Values to test

# Default num_leaves settings (used when RUN_NUM_LEAVES_STUDY = False)
# Separate for position and force predictions
NUM_LEAVES_POSITION = 200  # num_leaves for x, y
NUM_LEAVES_FORCE = 200  # num_leaves for fx, fy, fz

# No-contact sentinel conversion: Convert sentinel values to NaN for training
CONVERT_SENTINEL_TO_NAN = True  # Set to True to enable
NO_CONTACT_SENTINEL = -999.0  # Sentinel value used in data_organization script

# Optional denoising (rolling mean on barometer channels BEFORE diffs)
APPLY_DENOISING = True
DENOISE_WINDOW  = 5       # odd number: 3,5,7,...

# Which models to train
USE_RANDOM_FOREST = False
USE_XGBOOST       = False
USE_LIGHTGBM      = True   # main model

# Feature engineering options
USE_SECOND_DERIVATIVE = False  # Set to True to include second derivatives in features


# ============================================================
# =============== FEATURE ENGINEERING ========================
# ============================================================

def convert_sentinel_to_nan(df, target_cols, sentinel=-999.0):
    """
    Convert sentinel values to NaN in target columns.
    
    This enables the model to learn 'no prediction' patterns from training data.
    When |fz| was ~0 during data collection, data_organization script set
    x, y, fx, fy to sentinel value (-999). We convert these to NaN so the
    model learns to associate certain barometer patterns with 'no contact'.
    
    Args:
        df: DataFrame with target columns
        target_cols: List of target column names
        sentinel: Sentinel value to replace with NaN
    
    Returns:
        DataFrame with sentinel values converted to NaN
    """
    df = df.copy()
    n_converted = 0
    
    for col in target_cols:
        if col in df.columns:
            sentinel_mask = df[col] == sentinel
            n_col_converted = sentinel_mask.sum()
            if n_col_converted > 0:
                df.loc[sentinel_mask, col] = np.nan
                n_converted += n_col_converted
    
    if n_converted > 0:
        pct = 100 * n_converted / (len(df) * len(target_cols))
        print(f"\nConverted {n_converted} sentinel values ({sentinel}) to NaN ({pct:.2f}% of target values)")
        print(f"  This teaches the model to recognize no-contact barometer patterns")
    
    return df


def maybe_denoise(df, baro_cols):
    """
    Optionally apply a simple rolling-mean denoising on barometer channels.
    Operates in-place on a copy.
    """
    if not APPLY_DENOISING:
        return df

    df = df.copy()
    win = DENOISE_WINDOW
    for col in baro_cols:
        df[col] = df[col].rolling(win, center=True).mean().bfill().ffill()
    return df


def build_window_features(df, baro_cols, time_col, target_cols, window_size, max_time_gap=0.05, use_second_derivative=False):
    """
    Build features using a past-only temporal sliding window, skipping windows that span file boundaries.

    For each index i in [window_size, N]:
        - Check if window has large time gaps (file boundaries)
        - If valid: features = all barometer values in past window [i-window_size .. i]
                             + all first derivatives in the same window
                             + all second derivatives in the same window (if use_second_derivative=True)
        - If invalid: skip this window
        - target  = targets at index i

    Args:
        window_size: Number of past samples to include in the window.
        max_time_gap: Maximum allowed time gap within a window (seconds).
                      If time jump > this, skip that window to avoid boundary issues.
        use_second_derivative: If True, include second derivatives in features.

    Returns:
        X: (M, num_features) - only valid windows
        y: (M, len(target_cols)) - corresponding targets
        center_df: dataframe of the current timestep rows (for plotting, time alignment)
        feature_names: list of feature names (strings)
    """
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df = maybe_denoise(df, baro_cols)

    # First and second derivatives
    for col in baro_cols:
        d1 = df[col].diff()
        df[f"{col}_d1"] = d1.fillna(0.0)
        if use_second_derivative:
            d2 = d1.diff()
            df[f"{col}_d2"] = d2.fillna(0.0)

    N = len(df)
    W = window_size
    if N <= W:
        raise ValueError(f"Not enough samples ({N}) for window size {W}")

    # Feature name template (for information)
    # Offset 0 = current, -1 = 1 step back, -2 = 2 steps back, etc.
    feature_names = []
    for col in baro_cols:
        for offset in range(-W, 1):
            feature_names.append(f"{col}@{offset}")
        for offset in range(-W, 1):
            feature_names.append(f"{col}_d1@{offset}")
        if use_second_derivative:
            for offset in range(-W, 1):
                feature_names.append(f"{col}_d2@{offset}")

    print("Building windowed features (skipping boundary windows)...")

    # Arrays
    time_vals = df[time_col].values
    baro_data = df[baro_cols].values
    d1_cols = [f"{col}_d1" for col in baro_cols]
    d1_data = df[d1_cols].values
    if use_second_derivative:
        d2_cols = [f"{col}_d2" for col in baro_cols]
        d2_data = df[d2_cols].values

    X_list = []
    y_list = []
    valid_indices = []
    skipped = 0

    for current_idx in range(W, N):
        start = current_idx - W
        end = current_idx + 1  # Include current sample

        window_times = time_vals[start:end]
        time_diffs = np.diff(window_times)
        max_gap = np.max(time_diffs)

        if max_gap > max_time_gap:
            skipped += 1
            continue

        baro_window = baro_data[start:end, :].flatten()
        d1_window   = d1_data[start:end, :].flatten()
        
        if use_second_derivative:
            d2_window   = d2_data[start:end, :].flatten()
            X_list.append(np.concatenate([baro_window, d1_window, d2_window]))
        else:
            X_list.append(np.concatenate([baro_window, d1_window]))
        y_list.append(df.loc[current_idx, target_cols].values)
        valid_indices.append(current_idx)

    print(f"Built {len(X_list)} valid windows, skipped {skipped} boundary windows")

    X = np.array(X_list)
    y = np.array(y_list)
    center_df = df.iloc[valid_indices].reset_index(drop=True)

    return X, y, center_df, feature_names


# ============================================================
# ======================= PLOTTING ===========================
# ============================================================

def plot_barometer_with_derivatives(df, baro_col, n_samples=800, title_suffix=""):
    df = maybe_denoise(df, [baro_col]).copy()
    df[f"{baro_col}_d1"] = df[baro_col].diff().fillna(0.0)
    df[f"{baro_col}_d2"] = df[f"{baro_col}_d1"].diff().fillna(0.0)

    df_sub = df.iloc[:n_samples].copy()

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df_sub[TIME_COL], df_sub[baro_col], label=f"{baro_col} raw")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Pressure")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(df_sub[TIME_COL], df_sub[f"{baro_col}_d1"], alpha=0.6, label=f"{baro_col}_d1")
    ax2.plot(df_sub[TIME_COL], df_sub[f"{baro_col}_d2"], alpha=0.4, label=f"{baro_col}_d2")
    ax2.set_ylabel("Derivatives")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title(f"{baro_col} with first & second derivatives {title_suffix}")
    plt.tight_layout()
    return fig


def plot_loss_curves(loss_history, model_name, target_cols=None):
    # Loss history is per-target dict for LightGBM
    if isinstance(loss_history, dict) and 'train' not in loss_history:
        n_targets = len(loss_history)
        fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))
        if n_targets == 1:
            axes = [axes]
        for i, (target, losses) in enumerate(loss_history.items()):
            ax = axes[i]
            ax.plot(losses['train'], label='Training Loss', linewidth=2)
            if 'valid' in losses:
                ax.plot(losses['valid'], label='Validation Loss', linewidth=2)
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('MSE Loss', fontsize=10)
            ax.set_title(f'{model_name} - {target}', fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle(f'{model_name} Training Loss Curves', fontsize=13, y=1.02)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        if 'train' in loss_history:
            ax.plot(loss_history['train'], label='Training Loss', linewidth=2)
        if 'valid' in loss_history:
            ax.plot(loss_history['valid'], label='Validation Loss', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('MSE Loss', fontsize=11)
        ax.set_title(f'{model_name} Training Loss Curve', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pred_vs_actual(y_test, y_pred, target_cols, title_suffix=""):
    n_targets = len(target_cols)
    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4))
    if n_targets == 1:
        axes = [axes]

    for i, col in enumerate(target_cols):
        ax = axes[i]
        true_vals = y_test[:, i]
        pred_vals = y_pred[:, i]

        ax.scatter(true_vals, pred_vals, alpha=0.5, s=20)

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        lims = [min_val - (max_val - min_val)*0.05,
                max_val + (max_val - min_val)*0.05]

        ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=1.5)

        mae = mean_absolute_error(true_vals, pred_vals)
        r2  = r2_score(true_vals, pred_vals)

        ax.set_title(f'{col}\nMAE: {mae:.2f} | R²: {r2:.3f}', fontsize=10)
        ax.set_xlabel('Actual', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    if title_suffix:
        plt.suptitle(f"Predicted vs Actual ({title_suffix})")
    plt.tight_layout()
    return fig


def plot_error_distributions(y_test, y_pred, target_cols, title_suffix=""):
    """
    Plot histograms and statistics of prediction errors for each target.
    Shows distribution, mean, std, and percentiles.
    """
    n_targets = len(target_cols)
    fig, axes = plt.subplots(2, n_targets, figsize=(4 * n_targets, 8))
    if n_targets == 1:
        axes = axes.reshape(-1, 1)
    
    for i, col in enumerate(target_cols):
        errors = y_test[:, i] - y_pred[:, i]
        
        # Top row: Histogram with statistics
        ax_hist = axes[0, i]
        n, bins, patches = ax_hist.hist(errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
        
        # Add vertical line for mean
        mean_err = np.mean(errors)
        ax_hist.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.3f}')
        ax_hist.axvline(0, color='green', linestyle='-', linewidth=1.5, alpha=0.7, label='Zero')
        
        # Statistics
        std_err = np.std(errors)
        mae = np.mean(np.abs(errors))
        
        unit = "mm" if col in ['x', 'y'] else "N"
        ax_hist.set_xlabel(f'Error ({unit})', fontsize=10)
        ax_hist.set_ylabel('Frequency', fontsize=10)
        ax_hist.set_title(f'{col} Error Distribution\nStd: {std_err:.3f} | MAE: {mae:.3f}', fontsize=11)
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)
        
        # Bottom row: Box plot with percentiles
        ax_box = axes[1, i]
        bp = ax_box.boxplot([errors], vert=True, widths=0.6, patch_artist=True,
                             boxprops=dict(facecolor='lightblue', alpha=0.7),
                             medianprops=dict(color='red', linewidth=2),
                             whiskerprops=dict(linewidth=1.5),
                             capprops=dict(linewidth=1.5))
        
        ax_box.axhline(0, color='green', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Add percentile annotations
        p25, p50, p75 = np.percentile(errors, [25, 50, 75])
        p95 = np.percentile(errors, 95)
        p5 = np.percentile(errors, 5)
        
        ax_box.set_ylabel(f'Error ({unit})', fontsize=10)
        ax_box.set_title(f'{col} Error Box Plot\n5th: {p5:.3f} | 95th: {p95:.3f}', fontsize=11)
        ax_box.grid(True, alpha=0.3, axis='y')
        ax_box.set_xticklabels([col])
    
    if title_suffix:
        plt.suptitle(f"Error Distribution Analysis ({title_suffix})", fontsize=13, y=0.995)
    plt.tight_layout()
    return fig


def calculate_grouped_rmse(y_true, y_pred, target_names, title_suffix="", return_metrics=False):
    print("\n" + "="*70 + f"\nGROUPED RMSE METRICS {title_suffix}\n" + "="*70)
    
    metrics = {}

    # Contact location
    contact_indices = [i for i, name in enumerate(target_names) if name in ['x', 'y']]
    if len(contact_indices) >= 2:
        contact_location_true = y_true[:, contact_indices]
        contact_location_pred = y_pred[:, contact_indices]
        contact_location_errors = contact_location_true - contact_location_pred
        contact_location_rmse = np.sqrt(np.mean(contact_location_errors ** 2))

        contact_euclidean_errors = np.sqrt(np.sum(contact_location_errors ** 2, axis=1))
        contact_euclidean_rmse = np.sqrt(np.mean(contact_euclidean_errors ** 2))

        print(f"\nContact Location (x, y):")
        print(f"  - Component-wise RMSE: {contact_location_rmse:.4f} mm")
        print(f"  - Euclidean RMSE:      {contact_euclidean_rmse:.4f} mm")
        print(f"  - Mean error distance: {np.mean(contact_euclidean_errors):.4f} mm")
        
        metrics['contact_location_rmse'] = contact_location_rmse
        metrics['contact_euclidean_rmse'] = contact_euclidean_rmse

    # Force vector
    force_indices = [i for i, name in enumerate(target_names) if name in ['fx', 'fy', 'fz']]
    if force_indices:
        force_vector_true = y_true[:, force_indices]
        force_vector_pred = y_pred[:, force_indices]
        force_vector_errors = force_vector_true - force_vector_pred
        force_vector_rmse = np.sqrt(np.mean(force_vector_errors ** 2))

        force_euclidean_errors = np.sqrt(np.sum(force_vector_errors ** 2, axis=1))
        force_euclidean_rmse = np.sqrt(np.mean(force_euclidean_errors ** 2))

        force_names = [target_names[i] for i in force_indices]
        print(f"\n{len(force_indices)}-DOF Force Vector ({', '.join(force_names)}):")
        print(f"  - Component-wise RMSE: {force_vector_rmse:.4f} N")
        print(f"  - Euclidean RMSE:      {force_euclidean_rmse:.4f} N")
        print(f"  - Mean error magnitude: {np.mean(force_euclidean_errors):.4f} N")
        
        metrics['force_vector_rmse'] = force_vector_rmse
        metrics['force_euclidean_rmse'] = force_euclidean_rmse
    
    if return_metrics:
        return metrics


def evaluate_constrained_region(y_test, y_pred, target_cols, x_range=10, y_range=8):
    """
    Evaluate predictions for the constrained region:
    |x| <= x_range, |y| <= y_range
    """
    try:
        x_idx = target_cols.index('x')
        y_idx = target_cols.index('y')
    except ValueError:
        print("Warning: 'x' or 'y' not found in target columns. Skipping constrained region analysis.")
        return None

    x_actual = y_test[:, x_idx]
    y_actual = y_test[:, y_idx]

    mask = (np.abs(x_actual) <= x_range) & (np.abs(y_actual) <= y_range)
    n_samples = np.sum(mask)

    if n_samples == 0:
        print(f"\nNo samples in constrained region (x: ±{x_range}, y: ±{y_range})")
        return None

    print("\n" + "="*70)
    print(f"CONSTRAINED REGION ANALYSIS (Rectangle: {2*x_range} x {2*y_range} mm)")
    print(f"  x: ±{x_range} mm, y: ±{y_range} mm")
    print("="*70)
    print(f"Samples in region: {n_samples}/{len(y_test)} ({100*n_samples/len(y_test):.1f}%)\n")

    y_test_c = y_test[mask]
    y_pred_c = y_pred[mask]

    print("Individual Target Metrics (Constrained Region):")
    print("-" * 70)
    for i, col in enumerate(target_cols):
        mae = mean_absolute_error(y_test_c[:, i], y_pred_c[:, i])
        r2  = r2_score(y_test_c[:, i], y_pred_c[:, i])
        rmse = np.sqrt(np.mean((y_test_c[:, i] - y_pred_c[:, i])**2))
        corr = np.corrcoef(y_test_c[:, i], y_pred_c[:, i])[0, 1]

        unit = "mm" if col in ['x', 'y'] else "N"
        print(f"{col:>3} | MAE: {mae:6.3f} {unit} | RMSE: {rmse:6.3f} {unit} | R²: {r2:6.3f} | Corr: {corr:6.3f}")

    calculate_grouped_rmse(y_test_c, y_pred_c, target_cols, title_suffix="(Constrained Region)")
    return mask


def plot_window_size_comparison(window_sizes, results, target_cols):
    """
    Plot comparison of error metrics across different window sizes.
    Both curves on the same plot with dual y-axes:
    - Left y-axis: Contact position error (x, y) in mm
    - Right y-axis: 3-DOF force error (fx, fy, fz) in N
    
    Args:
        window_sizes: List of window sizes tested
        results: Dict mapping window_size -> {'contact_rmse': float, 'force_rmse': float}
        target_cols: List of target column names
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Extract metrics
    contact_rmse = [results[ws]['contact_euclidean_rmse'] for ws in window_sizes]
    force_rmse = [results[ws]['force_euclidean_rmse'] for ws in window_sizes]
    
    # Plot contact position error on left y-axis
    color1 = 'steelblue'
    ax1.set_xlabel('Window Size (samples)', fontsize=12)
    ax1.set_ylabel('Contact Position RMSE (mm)', fontsize=12, color=color1)
    line1 = ax1.plot(window_sizes, contact_rmse, marker='o', linewidth=2.5, markersize=9, 
                     color=color1, label='Contact Position Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(window_sizes)
    
    # Annotate best value for contact position
    best_idx_contact = np.argmin(contact_rmse)
    ax1.scatter([window_sizes[best_idx_contact]], [contact_rmse[best_idx_contact]], 
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)
    
    # Create second y-axis for force error
    ax2 = ax1.twinx()
    color2 = 'darkorange'
    ax2.set_ylabel('3-DOF Force RMSE (N)', fontsize=12, color=color2)
    line2 = ax2.plot(window_sizes, force_rmse, marker='s', linewidth=2.5, markersize=9, 
                     color=color2, label='Force Error')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Annotate best value for force
    best_idx_force = np.argmin(force_rmse)
    ax2.scatter([window_sizes[best_idx_force]], [force_rmse[best_idx_force]], 
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
               ncol=2, fontsize=11, frameon=True, shadow=True)
    
    plt.title('Window Size Hyperparameter Study', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    return fig


def plot_learning_rate_comparison(configs, results_position, results_force):
    """
    Plot comparison of error metrics across different learning rate configurations.
    Both curves on the same plot with dual y-axes:
    - Left y-axis: Contact position error (x, y) in mm
    - Right y-axis: 3-DOF force error (fx, fy, fz) in N
    
    Args:
        configs: List of dicts with 'learning_rate' and 'n_estimators'
        results_position: Dict mapping config_idx -> {'contact_euclidean_rmse': float}
        results_force: Dict mapping config_idx -> {'force_euclidean_rmse': float}
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Create labels for x-axis
    config_labels = [f"LR={c['learning_rate']}\nN={c['n_estimators']}" for c in configs]
    x_pos = np.arange(len(configs))
    
    # Extract metrics
    contact_rmse = [results_position[i]['contact_euclidean_rmse'] for i in range(len(configs))]
    force_rmse = [results_force[i]['force_euclidean_rmse'] for i in range(len(configs))]
    
    # Plot contact position error on left y-axis
    color1 = 'steelblue'
    ax1.set_xlabel('Learning Rate Configuration', fontsize=12)
    ax1.set_ylabel('Contact Position RMSE (mm)', fontsize=12, color=color1)
    line1 = ax1.plot(x_pos, contact_rmse, marker='o', linewidth=2.5, markersize=9, 
                     color=color1, label='Contact Position Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(config_labels, fontsize=9)
    
    # Annotate best value for contact position
    best_idx_contact = np.argmin(contact_rmse)
    ax1.scatter([x_pos[best_idx_contact]], [contact_rmse[best_idx_contact]], 
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)
    
    # Create second y-axis for force error
    ax2 = ax1.twinx()
    color2 = 'darkorange'
    ax2.set_ylabel('3-DOF Force RMSE (N)', fontsize=12, color=color2)
    line2 = ax2.plot(x_pos, force_rmse, marker='s', linewidth=2.5, markersize=9, 
                     color=color2, label='Force Error')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Annotate best value for force
    best_idx_force = np.argmin(force_rmse)
    ax2.scatter([x_pos[best_idx_force]], [force_rmse[best_idx_force]], 
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=2, fontsize=11, frameon=True, shadow=True)
    
    plt.title('Learning Rate & N_Estimators Hyperparameter Study', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    return fig


def plot_num_leaves_comparison(num_leaves_list, results_position, results_force):
    """
    Plot comparison of error metrics across different num_leaves values.
    Both curves on the same plot with dual y-axes:
    - Left y-axis: Contact position error (x, y) in mm
    - Right y-axis: 3-DOF force error (fx, fy, fz) in N
    
    Args:
        num_leaves_list: List of num_leaves values tested
        results_position: Dict mapping idx -> {'contact_euclidean_rmse': float}
        results_force: Dict mapping idx -> {'force_euclidean_rmse': float}
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(num_leaves_list))
    
    # Extract metrics
    contact_rmse = [results_position[i]['contact_euclidean_rmse'] for i in range(len(num_leaves_list))]
    force_rmse = [results_force[i]['force_euclidean_rmse'] for i in range(len(num_leaves_list))]
    
    # Plot contact position error on left y-axis
    color1 = 'steelblue'
    ax1.set_xlabel('num_leaves', fontsize=12)
    ax1.set_ylabel('Contact Position RMSE (mm)', fontsize=12, color=color1)
    line1 = ax1.plot(x_pos, contact_rmse, marker='o', linewidth=2.5, markersize=9, 
                     color=color1, label='Contact Position Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(nl) for nl in num_leaves_list], fontsize=10)
    
    # Annotate best value for contact position
    best_idx_contact = np.argmin(contact_rmse)
    ax1.scatter([x_pos[best_idx_contact]], [contact_rmse[best_idx_contact]], 
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)
    
    # Create second y-axis for force error
    ax2 = ax1.twinx()
    color2 = 'darkorange'
    ax2.set_ylabel('3-DOF Force RMSE (N)', fontsize=12, color=color2)
    line2 = ax2.plot(x_pos, force_rmse, marker='s', linewidth=2.5, markersize=9, 
                     color=color2, label='Force Error')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Annotate best value for force
    best_idx_force = np.argmin(force_rmse)
    ax2.scatter([x_pos[best_idx_force]], [force_rmse[best_idx_force]], 
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
               ncol=2, fontsize=11, frameon=True, shadow=True)
    
    plt.title('num_leaves Hyperparameter Study', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    return fig


# ============================================================
# ======================== MAIN ==============================
# ============================================================

def main():
    # ---------- Load data ----------
    train_path = os.path.join(DATA_DIRECTORY, TRAIN_FILENAME)
    val_path   = os.path.join(DATA_DIRECTORY, VALIDATION_FILENAME)
    test_path  = os.path.join(DATA_DIRECTORY, TEST_FILENAME)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data file not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data file not found: {val_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data file not found: {test_path}")

    train_df = pd.read_csv(train_path, skipinitialspace=True)
    train_df.columns = train_df.columns.str.strip()

    val_df = pd.read_csv(val_path, skipinitialspace=True)
    val_df.columns = val_df.columns.str.strip()

    test_df = pd.read_csv(test_path, skipinitialspace=True)
    test_df.columns = test_df.columns.str.strip()

    print("Loaded train data with columns:", train_df.columns.tolist())
    print("Loaded validation data with columns:", val_df.columns.tolist())
    print("Loaded test data with columns:", test_df.columns.tolist())
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

    # Convert sentinel values to NaN if enabled
    if CONVERT_SENTINEL_TO_NAN:
        print("\nConverting sentinel values to NaN in target columns...")
        train_df = convert_sentinel_to_nan(train_df, TARGET_COLS, NO_CONTACT_SENTINEL)
        val_df = convert_sentinel_to_nan(val_df, TARGET_COLS, NO_CONTACT_SENTINEL)
        test_df = convert_sentinel_to_nan(test_df, TARGET_COLS, NO_CONTACT_SENTINEL)

    needed_cols = [TIME_COL] + BARO_COLS + TARGET_COLS
    train_df = train_df.dropna(subset=needed_cols).reset_index(drop=True)
    val_df   = val_df.dropna(subset=needed_cols).reset_index(drop=True)
    test_df  = test_df.dropna(subset=needed_cols).reset_index(drop=True)

    # ---------- Window Size Study or Single Run ----------
    if RUN_WINDOW_SIZE_STUDY:
        print("\n" + "="*70)
        print("RUNNING WINDOW SIZE HYPERPARAMETER STUDY")
        print(f"Testing window sizes: {WINDOW_SIZES}")
        print("="*70)
        
        window_size_results = {}
        
        for ws in WINDOW_SIZES:
            print(f"\n{'='*70}")
            print(f"Testing Window Size: {ws}")
            print(f"{'='*70}")
            
            # Build features for this window size
            X_train_ws, y_train_ws, _, _ = build_window_features(
                train_df, BARO_COLS, TIME_COL, TARGET_COLS, ws,
                use_second_derivative=USE_SECOND_DERIVATIVE
            )
            X_val_ws, y_val_ws, _, _ = build_window_features(
                val_df, BARO_COLS, TIME_COL, TARGET_COLS, ws,
                use_second_derivative=USE_SECOND_DERIVATIVE
            )
            X_test_ws, y_test_ws, _, _ = build_window_features(
                test_df, BARO_COLS, TIME_COL, TARGET_COLS, ws,
                use_second_derivative=USE_SECOND_DERIVATIVE
            )
            
            # Normalize
            scaler_ws = StandardScaler()
            X_train_ws = scaler_ws.fit_transform(X_train_ws)
            X_val_ws = scaler_ws.transform(X_val_ws)
            X_test_ws = scaler_ws.transform(X_test_ws)
            
            # Train model (use the enabled model)
            if USE_LIGHTGBM and HAVE_LIGHTGBM:
                lgbm_models_ws = []
                lgbm_preds_ws = []
                
                for i, target in enumerate(TARGET_COLS):
                    lgbm = LGBMRegressor(
                        n_estimators=300,
                        learning_rate=0.1,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                        metric='mse',
                    )
                    lgbm.fit(X_train_ws, y_train_ws[:, i],
                            eval_set=[(X_val_ws, y_val_ws[:, i])],
                            eval_names=['valid'])
                    lgbm_models_ws.append(lgbm)
                    lgbm_preds_ws.append(lgbm.predict(X_test_ws))
                
                y_pred_ws = np.column_stack(lgbm_preds_ws)
                
            elif USE_XGBOOST and HAVE_XGBOOST:
                xgb_models_ws = []
                xgb_preds_ws = []
                
                for i, target in enumerate(TARGET_COLS):
                    xgb = XGBRegressor(
                        n_estimators=300,
                        learning_rate=0.1,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=0,
                        eval_metric='rmse'
                    )
                    xgb.fit(X_train_ws, y_train_ws[:, i],
                           eval_set=[(X_val_ws, y_val_ws[:, i])],
                           verbose=False)
                    xgb_models_ws.append(xgb)
                    xgb_preds_ws.append(xgb.predict(X_test_ws))
                
                y_pred_ws = np.column_stack(xgb_preds_ws)
                
            elif USE_RANDOM_FOREST:
                rf_models_ws = []
                rf_preds_ws = []
                
                for i, target in enumerate(TARGET_COLS):
                    rf = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=20,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=0
                    )
                    rf.fit(X_train_ws, y_train_ws[:, i])
                    rf_models_ws.append(rf)
                    rf_preds_ws.append(rf.predict(X_test_ws))
                
                y_pred_ws = np.column_stack(rf_preds_ws)
            else:
                raise ValueError("No model enabled for window size study. Enable at least one model.")
            
            # Evaluate and store results
            metrics = calculate_grouped_rmse(y_test_ws, y_pred_ws, TARGET_COLS, 
                                            title_suffix=f"Window Size = {ws}",
                                            return_metrics=True)
            window_size_results[ws] = metrics
        
        # Plot comparison
        print("\n" + "="*70)
        print("GENERATING WINDOW SIZE COMPARISON PLOT")
        print("="*70)
        
        fig_comparison = plot_window_size_comparison(WINDOW_SIZES, window_size_results, TARGET_COLS)
        
        save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
        os.makedirs(save_dir, exist_ok=True)
        version = f'v{sensor_version:.2f}'
        
        comparison_path = os.path.join(save_dir, f'window_size_comparison_{version}.png')
        fig_comparison.savefig(comparison_path, bbox_inches='tight', dpi=300)
        plt.show()
        print(f"Window size comparison plot saved to: {comparison_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("WINDOW SIZE STUDY SUMMARY")
        print("="*70)
        print(f"{'Window Size':<15} {'Contact RMSE (mm)':<20} {'Force RMSE (N)':<20}")
        print("-" * 70)
        for ws in WINDOW_SIZES:
            contact_rmse = window_size_results[ws]['contact_euclidean_rmse']
            force_rmse = window_size_results[ws]['force_euclidean_rmse']
            print(f"{ws:<15} {contact_rmse:<20.4f} {force_rmse:<20.4f}")
        
        # Find best window sizes
        best_ws_contact = min(WINDOW_SIZES, key=lambda ws: window_size_results[ws]['contact_euclidean_rmse'])
        best_ws_force = min(WINDOW_SIZES, key=lambda ws: window_size_results[ws]['force_euclidean_rmse'])
        
        print("\n" + "="*70)
        print(f"Best window size for contact position: {best_ws_contact}")
        print(f"Best window size for force: {best_ws_force}")
        print("="*70)
        
        print("\nWindow size study complete. Exiting.")
        return
    
    # ---------- Learning Rate & N_Estimators Study ----------
    if RUN_LEARNING_RATE_STUDY:
        print("\n" + "="*70)
        print("RUNNING LEARNING RATE & N_ESTIMATORS HYPERPARAMETER STUDY")
        print(f"Testing {len(LEARNING_RATE_CONFIGS)} configurations")
        print("="*70)
        
        # Build features once with selected window size
        X_train_lr, y_train_lr, _, _ = build_window_features(
            train_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
            use_second_derivative=USE_SECOND_DERIVATIVE
        )
        X_val_lr, y_val_lr, _, _ = build_window_features(
            val_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
            use_second_derivative=USE_SECOND_DERIVATIVE
        )
        X_test_lr, y_test_lr, _, _ = build_window_features(
            test_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
            use_second_derivative=USE_SECOND_DERIVATIVE
        )
        
        # Normalize
        scaler_lr = StandardScaler()
        X_train_lr = scaler_lr.fit_transform(X_train_lr)
        X_val_lr = scaler_lr.transform(X_val_lr)
        X_test_lr = scaler_lr.transform(X_test_lr)
        
        lr_results_position = {}
        lr_results_force = {}
        
        for config_idx, config in enumerate(LEARNING_RATE_CONFIGS):
            lr = config['learning_rate']
            n_est = config['n_estimators']
            
            print(f"\n{'='*70}")
            print(f"Testing Config {config_idx + 1}/{len(LEARNING_RATE_CONFIGS)}: LR={lr}, N_estimators={n_est}")
            print(f"{'='*70}")
            
            if USE_LIGHTGBM and HAVE_LIGHTGBM:
                # Train position models (x, y)
                position_preds = []
                for i, target in enumerate(['x', 'y']):
                    target_idx = TARGET_COLS.index(target)
                    lgbm = LGBMRegressor(
                        n_estimators=n_est,
                        learning_rate=lr,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                        metric='mse',
                    )
                    lgbm.fit(X_train_lr, y_train_lr[:, target_idx],
                            eval_set=[(X_val_lr, y_val_lr[:, target_idx])],
                            eval_names=['valid'])
                    position_preds.append(lgbm.predict(X_test_lr))
                
                y_pred_position = np.column_stack(position_preds)
                y_test_position = y_test_lr[:, [TARGET_COLS.index('x'), TARGET_COLS.index('y')]]
                
                # Calculate position metrics
                metrics_pos = calculate_grouped_rmse(
                    y_test_position, y_pred_position, ['x', 'y'],
                    title_suffix=f"Position - LR={lr}, N={n_est}",
                    return_metrics=True
                )
                lr_results_position[config_idx] = metrics_pos
                
                # Train force models (fx, fy, fz)
                force_preds = []
                for i, target in enumerate(['fx', 'fy', 'fz']):
                    target_idx = TARGET_COLS.index(target)
                    lgbm = LGBMRegressor(
                        n_estimators=n_est,
                        learning_rate=lr,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                        metric='mse',
                    )
                    lgbm.fit(X_train_lr, y_train_lr[:, target_idx],
                            eval_set=[(X_val_lr, y_val_lr[:, target_idx])],
                            eval_names=['valid'])
                    force_preds.append(lgbm.predict(X_test_lr))
                
                y_pred_force = np.column_stack(force_preds)
                y_test_force = y_test_lr[:, [TARGET_COLS.index('fx'), TARGET_COLS.index('fy'), TARGET_COLS.index('fz')]]
                
                # Calculate force metrics
                metrics_force = calculate_grouped_rmse(
                    y_test_force, y_pred_force, ['fx', 'fy', 'fz'],
                    title_suffix=f"Force - LR={lr}, N={n_est}",
                    return_metrics=True
                )
                lr_results_force[config_idx] = metrics_force
                
            else:
                raise ValueError("LightGBM must be enabled for learning rate study.")
        
        # Plot comparison
        print("\n" + "="*70)
        print("GENERATING LEARNING RATE COMPARISON PLOT")
        print("="*70)
        
        fig_lr_comparison = plot_learning_rate_comparison(LEARNING_RATE_CONFIGS, lr_results_position, lr_results_force)
        
        save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
        os.makedirs(save_dir, exist_ok=True)
        version = f'v{sensor_version:.2f}'
        
        lr_comparison_path = os.path.join(save_dir, f'learning_rate_comparison_{version}.png')
        fig_lr_comparison.savefig(lr_comparison_path, bbox_inches='tight', dpi=300)
        plt.show()
        print(f"Learning rate comparison plot saved to: {lr_comparison_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("LEARNING RATE STUDY SUMMARY")
        print("="*70)
        print(f"{'Config':<25} {'Contact RMSE (mm)':<20} {'Force RMSE (N)':<20}")
        print("-" * 70)
        for idx, config in enumerate(LEARNING_RATE_CONFIGS):
            lr = config['learning_rate']
            n_est = config['n_estimators']
            contact_rmse = lr_results_position[idx]['contact_euclidean_rmse']
            force_rmse = lr_results_force[idx]['force_euclidean_rmse']
            print(f"LR={lr}, N={n_est:<7} {contact_rmse:<20.4f} {force_rmse:<20.4f}")
        
        # Find best configurations
        best_config_contact = min(range(len(LEARNING_RATE_CONFIGS)), 
                                 key=lambda i: lr_results_position[i]['contact_euclidean_rmse'])
        best_config_force = min(range(len(LEARNING_RATE_CONFIGS)), 
                               key=lambda i: lr_results_force[i]['force_euclidean_rmse'])
        
        print("\n" + "="*70)
        print(f"Best config for contact position: LR={LEARNING_RATE_CONFIGS[best_config_contact]['learning_rate']}, "
              f"N={LEARNING_RATE_CONFIGS[best_config_contact]['n_estimators']}")
        print(f"Best config for force: LR={LEARNING_RATE_CONFIGS[best_config_force]['learning_rate']}, "
              f"N={LEARNING_RATE_CONFIGS[best_config_force]['n_estimators']}")
        print("="*70)
        
        print("\nLearning rate study complete. Exiting.")
        return
    
    # ---------- num_leaves Study ----------
    if RUN_NUM_LEAVES_STUDY:
        print("\n" + "="*70)
        print("RUNNING NUM_LEAVES HYPERPARAMETER STUDY")
        print("="*70)
        print(f"Testing {len(NUM_LEAVES_CONFIGS)} different num_leaves values: {NUM_LEAVES_CONFIGS}")
        print(f"Using window size: {WINDOW_SIZE}")
        print(f"Using optimized learning rates:")
        print(f"  Position: LR={LR_POSITION}, N={N_EST_POSITION}")
        print(f"  Force: LR={LR_FORCE}, N={N_EST_FORCE}")
        
        # Build features once with the default window size
        X_train_nl, y_train_nl, _, _ = build_window_features(
            train_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
            use_second_derivative=USE_SECOND_DERIVATIVE
        )
        X_val_nl, y_val_nl, _, _ = build_window_features(
            val_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
            use_second_derivative=USE_SECOND_DERIVATIVE
        )
        X_test_nl, y_test_nl, _, _ = build_window_features(
            test_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
            use_second_derivative=USE_SECOND_DERIVATIVE
        )
        
        # Normalize
        scaler_nl = StandardScaler()
        X_train_nl = scaler_nl.fit_transform(X_train_nl)
        X_val_nl = scaler_nl.transform(X_val_nl)
        X_test_nl = scaler_nl.transform(X_test_nl)
        
        nl_results_position = {}
        nl_results_force = {}
        
        for nl_idx, num_leaves_val in enumerate(NUM_LEAVES_CONFIGS):
            print(f"\n{'='*70}")
            print(f"Testing Config {nl_idx + 1}/{len(NUM_LEAVES_CONFIGS)}: num_leaves={num_leaves_val}")
            print(f"{'='*70}")
            
            if USE_LIGHTGBM and HAVE_LIGHTGBM:
                # Train position models (x, y)
                position_preds = []
                for i, target in enumerate(['x', 'y']):
                    target_idx = TARGET_COLS.index(target)
                    lgbm = LGBMRegressor(
                        n_estimators=N_EST_POSITION,
                        learning_rate=LR_POSITION,
                        num_leaves=num_leaves_val,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                        metric='mse',
                    )
                    lgbm.fit(X_train_nl, y_train_nl[:, target_idx],
                            eval_set=[(X_val_nl, y_val_nl[:, target_idx])],
                            eval_names=['valid'])
                    position_preds.append(lgbm.predict(X_test_nl))
                
                y_pred_position = np.column_stack(position_preds)
                y_test_position = y_test_nl[:, [TARGET_COLS.index('x'), TARGET_COLS.index('y')]]
                
                # Calculate position metrics
                metrics_pos = calculate_grouped_rmse(
                    y_test_position, y_pred_position, ['x', 'y'],
                    title_suffix=f"Position - num_leaves={num_leaves_val}",
                    return_metrics=True
                )
                nl_results_position[nl_idx] = metrics_pos
                
                # Train force models (fx, fy, fz)
                force_preds = []
                for i, target in enumerate(['fx', 'fy', 'fz']):
                    target_idx = TARGET_COLS.index(target)
                    lgbm = LGBMRegressor(
                        n_estimators=N_EST_FORCE,
                        learning_rate=LR_FORCE,
                        num_leaves=num_leaves_val,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                        metric='mse',
                    )
                    lgbm.fit(X_train_nl, y_train_nl[:, target_idx],
                            eval_set=[(X_val_nl, y_val_nl[:, target_idx])],
                            eval_names=['valid'])
                    force_preds.append(lgbm.predict(X_test_nl))
                
                y_pred_force = np.column_stack(force_preds)
                y_test_force = y_test_nl[:, [TARGET_COLS.index('fx'), TARGET_COLS.index('fy'), TARGET_COLS.index('fz')]]
                
                # Calculate force metrics
                metrics_force = calculate_grouped_rmse(
                    y_test_force, y_pred_force, ['fx', 'fy', 'fz'],
                    title_suffix=f"Force - num_leaves={num_leaves_val}",
                    return_metrics=True
                )
                nl_results_force[nl_idx] = metrics_force
                
            else:
                raise ValueError("LightGBM must be enabled for num_leaves study.")
        
        # Plot comparison
        print("\n" + "="*70)
        print("GENERATING NUM_LEAVES COMPARISON PLOT")
        print("="*70)
        
        fig_nl_comparison = plot_num_leaves_comparison(NUM_LEAVES_CONFIGS, nl_results_position, nl_results_force)
        
        save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
        os.makedirs(save_dir, exist_ok=True)
        version = f'v{sensor_version:.2f}'
        
        nl_comparison_path = os.path.join(save_dir, f'num_leaves_comparison_{version}.png')
        fig_nl_comparison.savefig(nl_comparison_path, bbox_inches='tight', dpi=300)
        plt.show()
        print(f"num_leaves comparison plot saved to: {nl_comparison_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("NUM_LEAVES STUDY SUMMARY")
        print("="*70)
        print(f"{'num_leaves':<15} {'Contact RMSE (mm)':<20} {'Force RMSE (N)':<20}")
        print("-" * 70)
        for idx, num_leaves_val in enumerate(NUM_LEAVES_CONFIGS):
            contact_rmse = nl_results_position[idx]['contact_euclidean_rmse']
            force_rmse = nl_results_force[idx]['force_euclidean_rmse']
            print(f"{num_leaves_val:<15} {contact_rmse:<20.4f} {force_rmse:<20.4f}")
        
        # Find best configurations
        best_nl_contact = NUM_LEAVES_CONFIGS[min(range(len(NUM_LEAVES_CONFIGS)), 
                                 key=lambda i: nl_results_position[i]['contact_euclidean_rmse'])]
        best_nl_force = NUM_LEAVES_CONFIGS[min(range(len(NUM_LEAVES_CONFIGS)), 
                               key=lambda i: nl_results_force[i]['force_euclidean_rmse'])]
        
        print("\n" + "="*70)
        print(f"Best num_leaves for contact position: {best_nl_contact}")
        print(f"Best num_leaves for force: {best_nl_force}")
        print("="*70)
        
        print("\nnum_leaves study complete. Exiting.")
        return
    
    # ---------- Single Window Size Run ----------
    print(f"\nUsing single window size: {WINDOW_SIZE}")
    
    # Build windowed feature matrices
    X_train, y_train, train_center, feat_names = build_window_features(
        train_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE, 
        use_second_derivative=USE_SECOND_DERIVATIVE
    )
    X_val, y_val, val_center, _ = build_window_features(
        val_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
        use_second_derivative=USE_SECOND_DERIVATIVE
    )
    X_test, y_test, test_center, _ = build_window_features(
        test_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
        use_second_derivative=USE_SECOND_DERIVATIVE
    )

    print(f"Engineered {X_train.shape[1]} features "
          f"(window size = {WINDOW_SIZE}, denoising = {APPLY_DENOISING}).")
    print(f"Feature matrix shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # ---------- Normalize features ----------
    print("\nNormalizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    print("Features normalized (mean=0, std=1)")

    # ---------- Train models ----------
    models = {}
    y_preds = {}
    loss_histories = {}

    if USE_RANDOM_FOREST:
        print("\nTraining Random Forest (multi-output: one model per target)...")
        rf_models = []
        rf_preds = []

        for i, target in enumerate(TARGET_COLS):
            print(f"  Training Random Forest for target: {target}")
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0
            )
            rf.fit(X_train, y_train[:, i])
            rf_models.append(rf)
            rf_preds.append(rf.predict(X_test))
            print(f"    Random Forest trained for {target}")

        models["RandomForest"] = rf_models
        y_preds["RandomForest"] = np.column_stack(rf_preds)
    else:
        print("Random Forest disabled.")

    if USE_XGBOOST and HAVE_XGBOOST:
        print("\nTraining XGBoost (multi-output: one model per target)...")
        xgb_models = []
        xgb_preds = []
        xgb_loss_history = {}

        for i, target in enumerate(TARGET_COLS):
            print(f"  Training XGBoost for target: {target}")
            xgb = XGBRegressor(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
                eval_metric='rmse'
            )

            xgb.fit(
                X_train, y_train[:, i],
                eval_set=[(X_train, y_train[:, i]), (X_val, y_val[:, i])],
                verbose=False
            )

            xgb_models.append(xgb)
            xgb_preds.append(xgb.predict(X_test))

            xgb_loss_history[target] = {
                'train': xgb.evals_result()['validation_0']['rmse'],
                'valid': xgb.evals_result()['validation_1']['rmse'],
            }
            print(f"    Final training RMSE: {xgb_loss_history[target]['train'][-1]:.4f}")
            print(f"    Final validation RMSE: {xgb_loss_history[target]['valid'][-1]:.4f}")

        models["XGBoost"] = xgb_models
        y_preds["XGBoost"] = np.column_stack(xgb_preds)
        loss_histories["XGBoost"] = xgb_loss_history
    else:
        print("XGBoost not available or disabled.")

    if USE_LIGHTGBM and HAVE_LIGHTGBM:
        print("\nTraining LightGBM (multi-output: one model per target)...")
        print(f"Using separate hyperparameters for position and force:")
        print(f"  Position (x, y): LR={LR_POSITION}, N_estimators={N_EST_POSITION}, num_leaves={NUM_LEAVES_POSITION}")
        print(f"  Force (fx, fy, fz): LR={LR_FORCE}, N_estimators={N_EST_FORCE}, num_leaves={NUM_LEAVES_FORCE}")
        
        lgbm_models = []
        lgbm_preds  = []
        lgbm_loss_history = {}

        for i, target in enumerate(TARGET_COLS):
            # Select hyperparameters based on target type
            if target in ['x', 'y']:
                lr = LR_POSITION
                n_est = N_EST_POSITION
                num_leaves_val = NUM_LEAVES_POSITION
                target_type = "position"
            else:  # fx, fy, fz
                lr = LR_FORCE
                n_est = N_EST_FORCE
                num_leaves_val = NUM_LEAVES_FORCE
                target_type = "force"
            
            print(f"  Training model for target: {target} ({target_type}) - LR={lr}, N={n_est}, num_leaves={num_leaves_val}")
            lgbm = LGBMRegressor(
                n_estimators=n_est,
                learning_rate=lr,
                num_leaves=num_leaves_val,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
                metric='mse',   # 'l2' equivalent
            )

            lgbm.fit(
                X_train, y_train[:, i],
                eval_set=[(X_train, y_train[:, i]), (X_val, y_val[:, i])],
                eval_names=['train', 'valid'],
            )

            lgbm_models.append(lgbm)
            lgbm_preds.append(lgbm.predict(X_test))

            lgbm_loss_history[target] = {
                'train': lgbm.evals_result_['train']['l2'],
                'valid': lgbm.evals_result_['valid']['l2'],
            }
            print(f"    Final training MSE: {lgbm_loss_history[target]['train'][-1]:.4f}")
            print(f"    Final validation MSE: {lgbm_loss_history[target]['valid'][-1]:.4f}")

        models["LightGBM"] = lgbm_models
        y_preds["LightGBM"] = np.column_stack(lgbm_preds)
        loss_histories["LightGBM"] = lgbm_loss_history
    else:
        print("LightGBM not available or disabled.")

    # ---------- Metrics ----------
    for name, y_hat in y_preds.items():
        d2_suffix = " + d2" if USE_SECOND_DERIVATIVE else ""
        print(f"\n=== Performance ({name}) with window + gradients{d2_suffix} ===")
        for i, col in enumerate(TARGET_COLS):
            mae = mean_absolute_error(y_test[:, i], y_hat[:, i])
            r2  = r2_score(y_test[:, i], y_hat[:, i])
            print(f"{col:>3} | MAE: {mae:6.3f} | R²: {r2:6.3f}")

    # ---------- Save Models and Plots ----------
    print("\n" + "="*70 + "\nSAVING MODELS AND PLOTS\n" + "="*70)

    save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
    save_dir_comparison = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models\comparison_plots"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_comparison, exist_ok=True)

    version = f'v{sensor_version:.3f}'
    
    # Save scaler
    scaler_path = os.path.join(save_dir, f'scaler_sliding_window_{version}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # Save models
    for name, model in models.items():
        model_name = name.lower().replace(" ", "_")
        model_path = os.path.join(save_dir, f'{model_name}_sliding_window_model_{version}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"{name} model saved to: {model_path}")

    # 1) Barometer + derivatives plot
    fig_baro = plot_barometer_with_derivatives(train_df, BARO_COLS[0], title_suffix="(train set)")
    fig_baro.savefig(os.path.join(save_dir, f'sliding_window_barometer_{version}.png'),
                     bbox_inches='tight', dpi=300)
    plt.close()

    # 2) Loss curves
    for model_name, loss_hist in loss_histories.items():
        fig_loss = plot_loss_curves(loss_hist, model_name, TARGET_COLS)
        fig_loss.savefig(os.path.join(save_dir, f'loss_curve_{model_name.lower()}_{version}.png'),
                         bbox_inches='tight', dpi=300)
        plt.show()
        print(f"{model_name} loss curve saved.")

    # 3) Pred vs actual for all models
    for model_name, y_hat in y_preds.items():
        fig = plot_pred_vs_actual(y_test, y_hat, TARGET_COLS, title_suffix=model_name)
        fig.savefig(os.path.join(save_dir, f'sliding_window_predictions_{model_name.lower()}_{version}.png'), bbox_inches='tight', dpi=300)
        plt.show()
        fig.savefig(os.path.join(save_dir_comparison, f'actual_vs_prediction_{model_name.lower()}_{version}.png'), bbox_inches='tight', dpi=300)
        print(f"{model_name} prediction plot saved.")

        # 4) Error distribution plots
        fig_err = plot_error_distributions(y_test, y_hat, TARGET_COLS, title_suffix=model_name)
        fig_err.savefig(os.path.join(save_dir, f'error_distribution_{model_name.lower()}_{version}.png'),
                        bbox_inches='tight', dpi=300)
        plt.close()
        print(f"{model_name} error distribution plot saved.")

        calculate_grouped_rmse(y_test, y_hat, TARGET_COLS, title_suffix=f"({model_name} - All Data)")

    # 5) LightGBM tree visualization (first tree for each target)
    if "LightGBM" in models:
        lgbm_models = models["LightGBM"]
        try:
            import graphviz
            print("\nGenerating LightGBM tree visualizations...")
            for i, target in enumerate(TARGET_COLS):
                fig, ax = plt.subplots(figsize=(20, 10))
                lgb.plot_tree(lgbm_models[i].booster_, tree_index=0, ax=ax, show_info=['split_gain'])
                plt.title(f'LightGBM Decision Tree (Target: {target}, Tree Index: 0)', fontsize=14)
                plt.tight_layout()
                tree_path = os.path.join(save_dir, f'lightgbm_tree_{target}_{version}.png')
                fig.savefig(tree_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
                print(f"  Saved tree for {target}: {tree_path}")
            print("LightGBM tree visualizations saved.")
        except ImportError as ie:
            print(f"Warning: graphviz not installed. Skipping tree visualization.")
            print(f"  Error details: {ie}")
            print("Tip: Install graphviz to enable tree visualization: conda install python-graphviz")
        except Exception as e:
            print(f"Warning: Could not generate LightGBM tree plots. Error: {e}")
            import traceback
            traceback.print_exc()
            print("Tree visualization skipped.")

        # # Constrained region: 20 x 16 mm centered at origin
        # mask = evaluate_constrained_region(y_test, y_hat, TARGET_COLS, x_range=10, y_range=8)
        # if mask is not None and np.sum(mask) > 0:
        #     y_test_c = y_test[mask]
        #     y_pred_c = y_hat[mask]
        #     fig_c = plot_pred_vs_actual(
        #         y_test_c, y_pred_c, TARGET_COLS,
        #         title_suffix=f"{chosen_model} (Rectangle: 20x16 mm)"
        #     )
        #     fig_c.savefig(os.path.join(save_dir, f'sliding_window_predictions_{chosen_model.lower()}_constrained_{version}.png'),
        #                   bbox_inches='tight', dpi=300)
        #     plt.close(fig_c)
        #     print("Constrained region plot saved.")

    print(f"\nAll plots saved to: {save_dir}")
    print("  - Barometer with derivatives plot")
    if loss_histories:
        print("  - Loss curve plots (training/validation)")
    print("  - Prediction vs actual plot (all data)")
    print("  - Error distribution plots (histograms and box plots)")
    if "LightGBM" in models:
        print("  - LightGBM tree visualizations (first tree for each target)")
    print("  - Prediction vs actual plot (constrained region)")
    print("\nProcess complete!")


if __name__ == "__main__":
    main()
