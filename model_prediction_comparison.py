#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Prediction Comparison Script

Load a trained model and scaler, make predictions on test data,
and plot predicted vs actual values over time.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse

from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# ======================= CONFIG =============================
# ============================================================

# Default paths
DEFAULT_MODEL_DIR = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
DEFAULT_TEST_DATA = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test 2 - sensor v5\synchronized_events_2.csv"

# Data columns
TIME_COL = "t"
BARO_COLS = ["b1", "b2", "b3", "b4", "b5", "b6"]
TARGET_COLS = ["x", "y", "fx", "fy", "fz"]

# Feature engineering parameters (must match training)
WINDOW_SIZE = 10
APPLY_DENOISING = True
DENOISE_WINDOW = 5
USE_SECOND_DERIVATIVE = False


# ============================================================
# =============== FEATURE ENGINEERING ========================
# ============================================================

def maybe_denoise(df, baro_cols):
    """Apply rolling-mean denoising on barometer channels."""
    if not APPLY_DENOISING:
        return df

    df = df.copy()
    win = DENOISE_WINDOW
    for col in baro_cols:
        df[col] = df[col].rolling(win, center=True).mean().bfill().ffill()
    return df


def build_window_features(df, baro_cols, time_col, target_cols, window_size,
                          max_time_gap=0.05, use_second_derivative=False):
    """
    Build features using a past-only temporal sliding window.
    Must match the feature engineering used during training.
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

    print("Building windowed features...")

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
        end = current_idx + 1

        window_times = time_vals[start:end]
        time_diffs = np.diff(window_times)
        max_gap = np.max(time_diffs)

        if max_gap > max_time_gap:
            skipped += 1
            continue

        baro_window = baro_data[start:end, :].flatten()
        d1_window = d1_data[start:end, :].flatten()

        # Rolling statistics over the window
        baro_window_2d = baro_data[start:end, :]
        d1_window_2d = d1_data[start:end, :]

        baro_stats = np.concatenate([
            np.mean(baro_window_2d, axis=0),
            np.std(baro_window_2d, axis=0),
            np.min(baro_window_2d, axis=0),
            np.max(baro_window_2d, axis=0)
        ])

        d1_stats = np.concatenate([
            np.mean(d1_window_2d, axis=0),
            np.std(d1_window_2d, axis=0)
        ])

        if use_second_derivative:
            d2_window = d2_data[start:end, :].flatten()
            d2_window_2d = d2_data[start:end, :]
            d2_stats = np.concatenate([
                np.mean(d2_window_2d, axis=0),
                np.std(d2_window_2d, axis=0)
            ])
            X_list.append(np.concatenate([baro_window, d1_window, d2_window, baro_stats, d1_stats, d2_stats]))
        else:
            X_list.append(np.concatenate([baro_window, d1_window, baro_stats, d1_stats]))

        y_list.append(df.loc[current_idx, target_cols].values)
        valid_indices.append(current_idx)

    print(f"Built {len(X_list)} valid windows, skipped {skipped} boundary windows")

    X = np.array(X_list)
    y = np.array(y_list)
    center_df = df.iloc[valid_indices].reset_index(drop=True)

    return X, y, center_df


def load_model_and_scaler(model_dir, model_name):
    """
    Load model and scaler from the specified directory.

    Args:
        model_dir: Directory containing the model files
        model_name: Name of the model file (with or without extension)

    Returns:
        model: Loaded model (list of models for multi-output)
        scaler: Loaded StandardScaler
    """
    # Handle model name with or without extension
    if not model_name.endswith('.pkl'):
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
    else:
        model_path = os.path.join(model_dir, model_name)

    # Try to find matching scaler
    # Extract version from model name
    model_basename = os.path.basename(model_path)

    # Try different scaler naming patterns
    scaler_patterns = []
    if 'lightgbm' in model_basename.lower():
        version = model_basename.replace('lightgbm_sliding_window_model_', '').replace('.pkl', '')
        scaler_patterns.append(f"scaler_sliding_window_{version}.pkl")
    elif 'xgboost' in model_basename.lower():
        version = model_basename.replace('xgboost_sliding_window_model_', '').replace('.pkl', '')
        scaler_patterns.append(f"scaler_sliding_window_{version}.pkl")
    elif 'randomforest' in model_basename.lower():
        version = model_basename.replace('randomforest_sliding_window_model_', '').replace('.pkl', '')
        scaler_patterns.append(f"scaler_sliding_window_{version}.pkl")

    # Also try generic pattern
    scaler_patterns.append("scaler_sliding_window_v5.11.pkl")  # fallback

    scaler = None
    for pattern in scaler_patterns:
        scaler_path = os.path.join(model_dir, pattern)
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Loaded scaler from: {scaler_path}")
            break

    if scaler is None:
        raise FileNotFoundError(f"Could not find scaler file. Tried: {scaler_patterns}")

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Loaded model from: {model_path}")

    return model, scaler


def make_predictions(model, X):
    """
    Make predictions using the loaded model.
    Handles both single model and list of models (multi-output).
    """
    if isinstance(model, list):
        # Multi-output: one model per target
        predictions = []
        for m in model:
            predictions.append(m.predict(X))
        return np.column_stack(predictions)
    else:
        # Single multi-output model
        return model.predict(X)


def plot_predictions_vs_time(time_vals, y_actual, y_pred, target_cols, title=""):
    """
    Plot predicted and actual values over time for each target.
    """
    n_targets = len(target_cols)
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 3 * n_targets), sharex=True)

    if n_targets == 1:
        axes = [axes]

    # Normalize time to start from 0
    time_normalized = time_vals - time_vals[0]

    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        ax.plot(time_normalized, y_actual[:, i], label='Actual', alpha=0.8, linewidth=1)
        ax.plot(time_normalized, y_pred[:, i], label='Predicted', alpha=0.8, linewidth=1)

        # Calculate metrics
        mae = mean_absolute_error(y_actual[:, i], y_pred[:, i])
        r2 = r2_score(y_actual[:, i], y_pred[:, i])

        unit = "mm" if col in ['x', 'y'] else "N"
        ax.set_ylabel(f"{col} ({unit})", fontsize=11)
        ax.set_title(f"{col}: MAE = {mae:.3f} {unit}, R² = {r2:.3f}", fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()
    return fig


def plot_error_over_time(time_vals, y_actual, y_pred, target_cols, title=""):
    """
    Plot prediction error over time for each target.
    """
    n_targets = len(target_cols)
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 2.5 * n_targets), sharex=True)

    if n_targets == 1:
        axes = [axes]

    # Normalize time to start from 0
    time_normalized = time_vals - time_vals[0]

    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        error = y_actual[:, i] - y_pred[:, i]

        ax.plot(time_normalized, error, alpha=0.7, linewidth=0.8, color='red')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        # Add rolling mean of error
        window = min(50, len(error) // 10)
        if window > 1:
            rolling_mean = pd.Series(error).rolling(window, center=True).mean()
            ax.plot(time_normalized, rolling_mean, color='blue', linewidth=1.5,
                   label=f'Rolling mean (w={window})', alpha=0.8)

        unit = "mm" if col in ['x', 'y'] else "N"
        ax.set_ylabel(f"Error ({unit})", fontsize=10)
        ax.set_title(f"{col} Prediction Error (Mean: {np.mean(error):.3f}, Std: {np.std(error):.3f})", fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()
    return fig


def print_metrics(y_actual, y_pred, target_cols):
    """Print detailed metrics for each target."""
    print("\n" + "=" * 70)
    print("PREDICTION METRICS")
    print("=" * 70)

    for i, col in enumerate(target_cols):
        actual = y_actual[:, i]
        pred = y_pred[:, i]
        error = actual - pred

        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(np.mean(error ** 2))
        r2 = r2_score(actual, pred)
        corr = np.corrcoef(actual, pred)[0, 1]

        unit = "mm" if col in ['x', 'y'] else "N"
        print(f"\n{col}:")
        print(f"  MAE:  {mae:8.4f} {unit}")
        print(f"  RMSE: {rmse:8.4f} {unit}")
        print(f"  R²:   {r2:8.4f}")
        print(f"  Corr: {corr:8.4f}")
        print(f"  Error range: [{error.min():.4f}, {error.max():.4f}] {unit}")


def main():
    parser = argparse.ArgumentParser(description="Load model and compare predictions with actual values")
    parser.add_argument("--model", type=str, default="lightgbm_sliding_window_model_v5.11",
                       help="Model filename (with or without .pkl extension)")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                       help="Directory containing model and scaler files")
    parser.add_argument("--data", type=str, default=DEFAULT_TEST_DATA,
                       help="Path to test data CSV file")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save plots (optional)")

    args = parser.parse_args()

    # Load test data
    print(f"\nLoading test data from: {args.data}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Test data file not found: {args.data}")

    df = pd.read_csv(args.data, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} samples with columns: {df.columns.tolist()}")

    # Check required columns
    needed_cols = [TIME_COL] + BARO_COLS + TARGET_COLS
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    # Drop NaN values
    df = df.dropna(subset=needed_cols).reset_index(drop=True)
    print(f"After dropping NaN: {len(df)} samples")

    # Load model and scaler
    print(f"\nLoading model: {args.model}")
    model, scaler = load_model_and_scaler(args.model_dir, args.model)

    # Build features
    print("\nBuilding features...")
    X, y_actual, center_df = build_window_features(
        df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
        use_second_derivative=USE_SECOND_DERIVATIVE
    )
    print(f"Feature matrix shape: {X.shape}")

    # Scale features
    print("Scaling features...")
    X_scaled = scaler.transform(X)

    # Make predictions
    print("Making predictions...")
    y_pred = make_predictions(model, X_scaled)

    # Get time values for plotting
    time_vals = center_df[TIME_COL].values

    # Print metrics
    print_metrics(y_actual, y_pred, TARGET_COLS)

    # Create plots
    print("\nGenerating plots...")

    # Extract model name for title
    model_name = os.path.basename(args.model).replace('.pkl', '')
    data_name = os.path.basename(os.path.dirname(args.data))

    # Plot predictions vs time
    fig1 = plot_predictions_vs_time(
        time_vals, y_actual, y_pred, TARGET_COLS,
        title=f"Predictions vs Actual - {model_name}\nData: {data_name}"
    )

    # Plot error over time
    fig2 = plot_error_over_time(
        time_vals, y_actual, y_pred, TARGET_COLS,
        title=f"Prediction Error over Time - {model_name}"
    )

    # Save plots if directory specified
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

        fig1.savefig(os.path.join(args.save_dir, f"predictions_vs_time_{model_name}.png"),
                    bbox_inches='tight', dpi=300)
        fig2.savefig(os.path.join(args.save_dir, f"error_over_time_{model_name}.png"),
                    bbox_inches='tight', dpi=300)
        print(f"\nPlots saved to: {args.save_dir}")

    plt.show()
    print("\nDone!")


if __name__ == "__main__":
    main()
