#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D CNN for predicting x, y, fx, fy, fz from 6 barometers
using sliding-window time series input.

- Input per sample: window of raw barometer values
  shape: (window_length, 6)
- Model: Conv1D -> Conv1D -> GlobalAveragePooling1D -> Dense -> Dense(5)
- Uses same CSV format as previous RF/LightGBM scripts.
"""

import os
import sys
from pathlib import Path

# Make utils importable when running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.metrics_utils import calculate_grouped_rmse
from utils.plot_utils import plot_pred_vs_actual
from utils.signal_utils import maybe_denoise

# ============================================================
# ==================== GPU/CUDA SETUP ========================
# ============================================================

# Configure TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n{'='*70}")
        print(f"GPU DETECTED: {len(gpus)} GPU(s) available")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        print(f"{'='*70}\n")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("\n" + "="*70)
    print("NO GPU DETECTED - Running on CPU")
    print("To use GPU, install: pip install tensorflow[and-cuda]")
    print("="*70 + "\n")

# ============================================================
# ======================= CONFIG =============================
# ============================================================

DATA_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data"
sensor_version = 4
TRAIN_FILENAME = f"train_data_v{sensor_version}.csv"
TEST_FILENAME  = f"test_data_v{sensor_version}.csv"

TIME_COL   = "t"
BARO_COLS  = ["b1", "b2", "b3", "b4", "b5", "b6"]
TARGET_COLS = ["x", "y", "fx", "fy", "fz"]

RANDOM_STATE = 42

WINDOW_RADIUS   = 15   # 31-sample window (increased for more context)
APPLY_DENOISING = True
DENOISE_WINDOW  = 5

VAL_SIZE   = 0.2
BATCH_SIZE = 128
EPOCHS     = 80

SAVE_DIR = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\cnn_models"


# ============================================================
# ==================== PREPROCESS HELPERS ====================
# ============================================================

def set_seeds(seed=RANDOM_STATE):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_window_sequences(df, baro_cols, time_col, target_cols, window_radius):
    """
    Build:
      X: (num_windows, window_length, 18)  # raw + d1 + d2 for 6 barometers = 18 channels
      y: (num_windows, 5)

    Same as RF/XGB script: includes first and second derivatives for dynamic information.
    """
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df = maybe_denoise(df, baro_cols, apply_denoising=APPLY_DENOISING, denoise_window=DENOISE_WINDOW)

    N = len(df)
    R = window_radius
    if N <= 2 * R:
        raise ValueError(f"Not enough samples ({N}) for window radius {R}")

    # Build multi-channel data: raw + d1 + d2 (same as RF/XGB feature engineering)
    series_list = []
    for col in baro_cols:
        raw = df[col].values
        d1  = df[col].diff().fillna(0.0).values
        d2  = df[col].diff().diff().fillna(0.0).values

        series_list.append(raw)
        series_list.append(d1)
        series_list.append(d2)

    # Shape: (N, num_channels) where num_channels = 18 for raw+d1+d2
    multi_data = np.stack(series_list, axis=1).astype(np.float32)

    targets = df[target_cols].values
    window_len = 2 * R + 1
    M = N - 2 * R
    num_channels = multi_data.shape[1]

    X = np.zeros((M, window_len, num_channels), dtype=np.float32)
    y = np.zeros((M, len(target_cols)), dtype=np.float32)

    for idx, center in enumerate(range(R, N - R)):
        start = center - R
        end   = center + R + 1
        X[idx, :, :] = multi_data[start:end, :]
        y[idx, :]    = targets[center, :]

    print(f"Built sequences with {num_channels} channels (6 baro \u00d7 3 [raw+d1+d2])")
    return X, y


def standardize_features(X_train, X_val, X_test):
    """
    Standardize features based on TRAIN ONLY, then apply to val+test.
    """
    orig_train_shape = X_train.shape
    orig_val_shape   = X_val.shape
    orig_test_shape  = X_test.shape

    # Flatten time dim for scaling
    X_train_flat = X_train.reshape(-1, orig_train_shape[-1])
    X_val_flat   = X_val.reshape(-1,   orig_val_shape[-1])
    X_test_flat  = X_test.reshape(-1,  orig_test_shape[-1])

    mean = X_train_flat.mean(axis=0, keepdims=True)
    std  = X_train_flat.std(axis=0, keepdims=True) + 1e-8

    X_train_flat_std = (X_train_flat - mean) / std
    X_val_flat_std   = (X_val_flat   - mean) / std
    X_test_flat_std  = (X_test_flat  - mean) / std

    X_train_std = X_train_flat_std.reshape(orig_train_shape)
    X_val_std   = X_val_flat_std.reshape(orig_val_shape)
    X_test_std  = X_test_flat_std.reshape(orig_test_shape)

    scaler = {"mean": mean, "std": std}
    return X_train_std, X_val_std, X_test_std, scaler


def standardize_targets(y_train, y_val):
    """
    Standardize targets based on TRAIN ONLY, apply to val.
    (Test targets stay in physical units; we only standardize predictions.)
    """
    mean = y_train.mean(axis=0, keepdims=True)
    std  = y_train.std(axis=0, keepdims=True) + 1e-8

    y_train_std = (y_train - mean) / std
    y_val_std   = (y_val   - mean) / std

    scaler = {"mean": mean, "std": std}
    return y_train_std, y_val_std, scaler


def inverse_standardize(y_std, scaler):
    return y_std * scaler["std"] + scaler["mean"]


# ============================================================
# ======================= MODEL BUILD ========================
# ============================================================

def build_cnn_model(window_len, n_channels, n_targets):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, padding="same",
                     activation="relu", input_shape=(window_len, n_channels)))
    model.add(BatchNormalization())

    model.add(Conv1D(64, kernel_size=5, padding="same", activation="relu"))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(n_targets, activation="linear"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ============================================================
# ======================== MAIN =============================
# ============================================================

def main():
    set_seeds()

    # ---------- Load data ----------
    train_path = os.path.join(DATA_DIRECTORY, TRAIN_FILENAME)
    test_path  = os.path.join(DATA_DIRECTORY, TEST_FILENAME)

    train_df = pd.read_csv(train_path, skipinitialspace=True)
    train_df.columns = train_df.columns.str.strip()
    test_df  = pd.read_csv(test_path, skipinitialspace=True)
    test_df.columns = test_df.columns.str.strip()

    print("Loaded train data with columns:", train_df.columns.tolist())
    print("Loaded test data with columns:", test_df.columns.tolist())
    print(f"Train samples (rows): {len(train_df)}, Test samples (rows): {len(test_df)}")

    needed_cols = [TIME_COL] + BARO_COLS + TARGET_COLS
    train_df = train_df.dropna(subset=needed_cols).reset_index(drop=True)
    test_df  = test_df.dropna(subset=needed_cols).reset_index(drop=True)

    # ---------- Build sequences ----------
    X_train_full, y_train_full = build_window_sequences(
        train_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_RADIUS
    )
    X_test, y_test = build_window_sequences(
        test_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_RADIUS
    )

    print(f"Window length: {X_train_full.shape[1]}, Channels: {X_train_full.shape[2]}")
    print(f"Train windows: {X_train_full.shape[0]}, Test windows: {X_test.shape[0]}")

    # Train/val split on windows
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # ---------- Standardize features & targets ----------
    X_train_std, X_val_std, X_test_std, feat_scaler = standardize_features(
        X_train, X_val, X_test
    )
    y_train_std, y_val_std, target_scaler = standardize_targets(y_train, y_val)

    # ---------- Build & train model ----------
    window_len = X_train_std.shape[1]
    n_channels = X_train_std.shape[2]
    n_targets  = y_train_std.shape[1]

    model = build_cnn_model(window_len, n_channels, n_targets)
    model.summary()

    os.makedirs(SAVE_DIR, exist_ok=True)
    version = TRAIN_FILENAME.split("train_data_")[1].split(".")[0]
    best_model_path = os.path.join(SAVE_DIR, f"cnn_1d_best_{version}.h5")

    cbs = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(best_model_path, monitor="val_loss",
                        save_best_only=True, save_weights_only=False),
    ]

    history = model.fit(
        X_train_std, y_train_std,
        validation_data=(X_val_std, y_val_std),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=2,
    )

    print(f"\nBest model saved to: {best_model_path}")

    # ---------- Evaluate on test set ----------
    y_pred_std = model.predict(X_test_std, batch_size=BATCH_SIZE)
    y_pred = inverse_standardize(y_pred_std, target_scaler)

    print("\n=== Test performance (CNN 1D) ===")
    for i, col in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2  = r2_score(y_test[:, i], y_pred[:, i])
        print(f"{col:>3} | MAE: {mae:6.3f} | R\u00b2: {r2:6.3f}")

    # ---------- Plots ----------
    # alpha=0.4, s=10, title_fontsize=9 matches the original CNN aesthetics
    fig = plot_pred_vs_actual(
        y_test, y_pred, TARGET_COLS, title_suffix="CNN1D",
        alpha=0.4, s=10, title_fontsize=9,
    )
    fig_path = os.path.join(SAVE_DIR, f"cnn1d_predictions_{version}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nPrediction plot saved to: {fig_path}")

    calculate_grouped_rmse(y_test, y_pred, TARGET_COLS)

    # Save scalers
    scaler_path = os.path.join(SAVE_DIR, f"cnn1d_scalers_{version}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(
            {
                "feature_scaler": feat_scaler,
                "target_scaler": target_scaler,
                "config": {
                    "window_radius": WINDOW_RADIUS,
                    "baro_cols": BARO_COLS,
                    "target_cols": TARGET_COLS,
                },
            },
            f,
        )
    print(f"Scalers saved to: {scaler_path}")


if __name__ == "__main__":
    main()
