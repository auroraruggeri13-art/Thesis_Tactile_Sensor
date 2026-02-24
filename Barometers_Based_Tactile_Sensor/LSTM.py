#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

# Make utils importable when running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam

from utils.metrics_utils import calculate_grouped_rmse, evaluate_constrained_region
from utils.plot_utils import (
    plot_pred_vs_actual,
    plot_error_distributions,
    plot_training_history_keras,
)
from utils.signal_utils import maybe_denoise

# ============================================================
# ==================== GPU CONFIGURATION =====================
# ============================================================

# Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU(s) detected: {len(physical_devices)}")
        print(f"GPU name(s): {[device.name for device in physical_devices]}")
        # Set mixed precision for faster training on compatible GPUs
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision (FP16) enabled for faster GPU training")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected. Running on CPU.")

# ============================================================
# ======================= CONFIG =============================
# ============================================================

DATA_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data"
sensor_version = 5.12
TRAIN_FILENAME = f"train_data_v{sensor_version}.csv"
VALIDATION_FILENAME = f"validation_data_v{sensor_version}.csv"
TEST_FILENAME  = f"test_data_v{sensor_version}.csv"

TIME_COL   = "t"
BARO_COLS  = ["b1", "b2", "b3", "b4", "b5", "b6"]
TARGET_COLS = ["x", "y", "fx", "fy", "fz"]

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Sliding window size (in samples). Looks at past WINDOW_SIZE samples.
WINDOW_SIZE = 10   # 10 past samples (~0.10s at 100 Hz)

# Optional denoising (rolling mean on barometer channels BEFORE diffs)
APPLY_DENOISING = True
DENOISE_WINDOW  = 5       # odd number: 3,5,7,...

# Feature engineering options
USE_DERIVATIVES = True     # Include first derivatives
USE_SECOND_DERIVATIVE = False  # Include second derivatives

# LSTM Model Configuration
LSTM_UNITS = [64, 32]      # List of LSTM layer sizes
DENSE_UNITS = [32, 16]     # Dense layers after LSTM
DROPOUT_RATE = 0.2         # Dropout rate for regularization
BATCH_SIZE = 128           # Increased for better GPU utilization
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2     # Split from training data for validation

# Early stopping
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7


# ============================================================
# =============== FEATURE ENGINEERING ========================
# ============================================================

def build_lstm_sequences(df, baro_cols, time_col, target_cols, window_size,
                         max_time_gap=0.05, use_derivatives=True, use_second_derivative=False):
    """
    Build 3D sequences for LSTM: (samples, timesteps, features)

    For each valid window:
        - X shape: (samples, window_size+1, n_features_per_timestep)
        - y shape: (samples, n_targets)

    Features per timestep include:
        - Raw barometer values (6 channels)
        - First derivatives (6 channels) if use_derivatives=True
        - Second derivatives (6 channels) if use_second_derivative=True

    Args:
        window_size: Number of past samples to include
        max_time_gap: Maximum allowed time gap within window (seconds)
        use_derivatives: Include first derivatives
        use_second_derivative: Include second derivatives

    Returns:
        X: (samples, timesteps, features) - 3D array for LSTM
        y: (samples, n_targets) - target values
        center_df: dataframe of current timestep rows
        feature_names: list of feature names per timestep
    """
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df = maybe_denoise(df, baro_cols, apply_denoising=APPLY_DENOISING, denoise_window=DENOISE_WINDOW)

    # Calculate derivatives
    feature_cols = baro_cols.copy()

    if use_derivatives:
        for col in baro_cols:
            df[f"{col}_d1"] = df[col].diff().fillna(0.0)
        d1_cols = [f"{col}_d1" for col in baro_cols]
        feature_cols.extend(d1_cols)

    if use_second_derivative:
        for col in baro_cols:
            d1 = df[col].diff()
            df[f"{col}_d2"] = d1.diff().fillna(0.0)
        d2_cols = [f"{col}_d2" for col in baro_cols]
        feature_cols.extend(d2_cols)

    N = len(df)
    W = window_size
    if N <= W:
        raise ValueError(f"Not enough samples ({N}) for window size {W}")

    print(f"Building LSTM sequences with {len(feature_cols)} features per timestep...")
    print(f"Features per timestep: {feature_cols}")

    time_vals = df[time_col].values
    feature_data = df[feature_cols].values

    X_list = []
    y_list = []
    valid_indices = []
    skipped = 0

    for current_idx in range(W, N):
        start = current_idx - W
        end = current_idx + 1  # Include current sample

        # Check for time gaps (file boundaries)
        window_times = time_vals[start:end]
        time_diffs = np.diff(window_times)
        max_gap = np.max(time_diffs)

        if max_gap > max_time_gap:
            skipped += 1
            continue

        # Extract sequence: shape (window_size+1, n_features)
        sequence = feature_data[start:end, :]

        X_list.append(sequence)
        y_list.append(df.loc[current_idx, target_cols].values)
        valid_indices.append(current_idx)

    print(f"Built {len(X_list)} valid sequences, skipped {skipped} boundary windows")

    # Convert to 3D numpy array: (samples, timesteps, features)
    X = np.array(X_list, dtype=np.float32)  # Use float32 for GPU efficiency
    y = np.array(y_list, dtype=np.float32)
    center_df = df.iloc[valid_indices].reset_index(drop=True)

    print(f"X shape: {X.shape} (samples, timesteps, features)")
    print(f"y shape: {y.shape} (samples, targets)")

    return X, y, center_df, feature_cols


# ============================================================
# ==================== LSTM MODEL ============================
# ============================================================

def build_lstm_model(input_shape, n_targets, lstm_units, dense_units, dropout_rate, learning_rate):
    """
    Build a multi-output LSTM regression model optimized for GPU.

    Args:
        input_shape: (timesteps, features) tuple
        n_targets: Number of output targets
        lstm_units: List of LSTM layer sizes
        dense_units: List of dense layer sizes
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='LSTM_Regressor')

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # LSTM layers - using CuDNN-compatible LSTM for GPU acceleration
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)  # Return sequences for all but last LSTM
        model.add(layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=0.0,  # Set to 0 for CuDNN compatibility (faster on GPU)
            name=f'lstm_{i+1}'
        ))

    # Dense layers
    for i, units in enumerate(dense_units):
        model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))

    # Output layer - use float32 for mixed precision
    model.add(layers.Dense(n_targets, activation='linear', dtype='float32', name='output'))

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


# ============================================================
# ======================= PLOTTING ===========================
# ============================================================

def plot_barometer_with_derivatives(df, baro_col, n_samples=800, title_suffix=""):
    df = maybe_denoise(df, [baro_col], apply_denoising=APPLY_DENOISING, denoise_window=DENOISE_WINDOW).copy()
    df[f"{baro_col}_d1"] = df[baro_col].diff().fillna(0.0)
    df[f"{baro_col}_d2"] = df[f"{baro_col}_d1"].diff().fillna(0.0)

    df_sub = df.iloc[:n_samples].copy()

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df_sub[TIME_COL], df_sub[baro_col], label=f"{baro_col} raw", linewidth=2)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Pressure", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df_sub[TIME_COL], df_sub[f"{baro_col}_d1"], alpha=0.6, label=f"{baro_col}_d1", linewidth=1.5)
    ax2.plot(df_sub[TIME_COL], df_sub[f"{baro_col}_d2"], alpha=0.4, label=f"{baro_col}_d2", linewidth=1.5)
    ax2.set_ylabel("Derivatives", fontsize=11)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title(f"{baro_col} with derivatives {title_suffix}", fontsize=12)
    plt.tight_layout()
    return fig


def plot_prediction_timeline(y_test, y_pred, target_cols, test_center_df, n_samples=1000, title_suffix=""):
    """
    Plot predictions vs actual values over time for a subset of samples.
    """
    n_targets = len(target_cols)
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 3 * n_targets))
    if n_targets == 1:
        axes = [axes]

    n_plot = min(n_samples, len(y_test))
    time_vals = test_center_df[TIME_COL].values[:n_plot]

    for i, col in enumerate(target_cols):
        ax = axes[i]
        true_vals = y_test[:n_plot, i]
        pred_vals = y_pred[:n_plot, i]

        ax.plot(time_vals, true_vals, label='Actual', linewidth=1.5, alpha=0.8)
        ax.plot(time_vals, pred_vals, label='Predicted', linewidth=1.5, alpha=0.8, linestyle='--')

        mae = mean_absolute_error(true_vals, pred_vals)
        r2 = r2_score(true_vals, pred_vals)

        unit = "mm" if col in ['x', 'y'] else "N"
        ax.set_ylabel(f'{col} ({unit})', fontsize=10)
        ax.set_title(f'{col} - MAE: {mae:.3f} | R\u00b2: {r2:.3f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if i == n_targets - 1:
            ax.set_xlabel('Time (s)', fontsize=10)

    if title_suffix:
        plt.suptitle(f"Prediction Timeline ({title_suffix})", fontsize=12, y=0.995)
    plt.tight_layout()
    return fig


# ============================================================
# ======================== MAIN ==============================
# ============================================================

def main():
    print("="*70)
    print("LSTM-Based Barometer Sensor Regression (GPU-Accelerated)")
    print("="*70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Random seed: {RANDOM_STATE}")

    # ---------- Load data ----------
    train_path = os.path.join(DATA_DIRECTORY, TRAIN_FILENAME)
    test_path  = os.path.join(DATA_DIRECTORY, TEST_FILENAME)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data file not found: {test_path}")

    train_df = pd.read_csv(train_path, skipinitialspace=True)
    train_df.columns = train_df.columns.str.strip()

    test_df = pd.read_csv(test_path, skipinitialspace=True)
    test_df.columns = test_df.columns.str.strip()

    print("\nLoaded train data with columns:", train_df.columns.tolist())
    print("Loaded test data with columns:", test_df.columns.tolist())
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    needed_cols = [TIME_COL] + BARO_COLS + TARGET_COLS
    train_df = train_df.dropna(subset=needed_cols).reset_index(drop=True)
    test_df  = test_df.dropna(subset=needed_cols).reset_index(drop=True)

    # ---------- Build LSTM sequences ----------
    print("\n" + "="*70)
    print("BUILDING LSTM SEQUENCES")
    print("="*70)

    X_train, y_train, train_center, feat_names = build_lstm_sequences(
        train_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
        use_derivatives=USE_DERIVATIVES,
        use_second_derivative=USE_SECOND_DERIVATIVE
    )

    X_test, y_test, test_center, _ = build_lstm_sequences(
        test_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
        use_derivatives=USE_DERIVATIVES,
        use_second_derivative=USE_SECOND_DERIVATIVE
    )

    print(f"\nSequence shape: (samples, timesteps, features)")
    print(f"Training:   {X_train.shape}")
    print(f"Testing:    {X_test.shape}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Denoising: {APPLY_DENOISING}")
    print(f"Use derivatives: {USE_DERIVATIVES}")
    print(f"Use second derivatives: {USE_SECOND_DERIVATIVE}")

    # ---------- Normalize features ----------
    print("\n" + "="*70)
    print("NORMALIZING FEATURES")
    print("="*70)

    # Reshape for scaling: (samples * timesteps, features)
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_test_2d = scaler.transform(X_test_2d)

    # Reshape back to 3D
    X_train = X_train_2d.reshape(n_samples_train, n_timesteps, n_features).astype(np.float32)
    X_test = X_test_2d.reshape(n_samples_test, n_timesteps, n_features).astype(np.float32)

    print("Features normalized (mean=0, std=1)")
    print(f"Normalized train shape: {X_train.shape}")
    print(f"Normalized test shape: {X_test.shape}")

    # ---------- Build and train LSTM model ----------
    print("\n" + "="*70)
    print("BUILDING LSTM MODEL")
    print("="*70)

    input_shape = (n_timesteps, n_features)
    n_targets = len(TARGET_COLS)

    model = build_lstm_model(
        input_shape=input_shape,
        n_targets=n_targets,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE
    )

    print("\nModel Architecture:")
    model.summary()

    print("\n" + "="*70)
    print("TRAINING LSTM MODEL")
    print("="*70)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Validation split: {VALIDATION_SPLIT}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Reduce LR patience: {REDUCE_LR_PATIENCE}")

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-6,
        verbose=1
    )

    # Train model with GPU
    print("\nTraining on GPU...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("\nTraining complete!")
    print(f"Best epoch: {np.argmin(history.history['val_loss']) + 1}")
    print(f"Best validation loss: {np.min(history.history['val_loss']):.6f}")

    # ---------- Evaluate model ----------
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)

    # Predictions
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

    # Per-target metrics
    print("\nIndividual Target Metrics:")
    print("-" * 70)
    for i, col in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(np.mean((y_test[:, i] - y_pred[:, i])**2))
        corr = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]

        unit = "mm" if col in ['x', 'y'] else "N"
        print(f"{col:>3} | MAE: {mae:6.3f} {unit} | RMSE: {rmse:6.3f} {unit} | R\u00b2: {r2:6.3f} | Corr: {corr:6.3f}")

    # Grouped metrics
    calculate_grouped_rmse(y_test, y_pred, TARGET_COLS, title_suffix="(All Data)")

    # Constrained region analysis
    mask = evaluate_constrained_region(y_test, y_pred, TARGET_COLS, x_range=10, y_range=8)

    # ---------- Save model and artifacts ----------
    print("\n" + "="*70)
    print("SAVING MODEL AND PLOTS")
    print("="*70)

    save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
    os.makedirs(save_dir, exist_ok=True)

    version = f'v{sensor_version:.1f}'

    # Save scaler
    scaler_path = os.path.join(save_dir, f'scaler_lstm_{version}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # Save model
    model_path = os.path.join(save_dir, f'lstm_model_{version}.h5')
    model.save(model_path)
    print(f"LSTM model saved to: {model_path}")

    # Save model architecture as JSON
    model_json = model.to_json()
    json_path = os.path.join(save_dir, f'lstm_model_architecture_{version}.json')
    with open(json_path, 'w') as f:
        f.write(model_json)
    print(f"Model architecture saved to: {json_path}")

    # Save training history
    history_path = os.path.join(save_dir, f'lstm_training_history_{version}.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to: {history_path}")

    # Generate and save plots
    print("\nGenerating plots...")

    fig_baro = plot_barometer_with_derivatives(train_df, BARO_COLS[0], title_suffix="(train set)")
    fig_baro.savefig(os.path.join(save_dir, f'lstm_barometer_{version}.png'),
                     bbox_inches='tight', dpi=300)
    plt.close(fig_baro)
    print("  \u2713 Barometer plot saved")

    fig_history = plot_training_history_keras(history, TARGET_COLS)
    fig_history.savefig(os.path.join(save_dir, f'lstm_training_history_{version}.png'),
                        bbox_inches='tight', dpi=300)
    plt.close(fig_history)
    print("  \u2713 Training history plot saved")

    # LSTM shows units in subplot titles (show_units=True preserves original behaviour)
    fig_pred = plot_pred_vs_actual(y_test, y_pred, TARGET_COLS, title_suffix="LSTM", show_units=True)
    fig_pred.savefig(os.path.join(save_dir, f'lstm_predictions_{version}.png'),
                     bbox_inches='tight', dpi=300)
    plt.close(fig_pred)
    print("  \u2713 Prediction scatter plots saved")

    fig_err = plot_error_distributions(y_test, y_pred, TARGET_COLS, title_suffix="LSTM")
    fig_err.savefig(os.path.join(save_dir, f'lstm_error_distribution_{version}.png'),
                    bbox_inches='tight', dpi=300)
    plt.close(fig_err)
    print("  \u2713 Error distribution plots saved")

    fig_timeline = plot_prediction_timeline(y_test, y_pred, TARGET_COLS, test_center,
                                           n_samples=1000, title_suffix="LSTM")
    fig_timeline.savefig(os.path.join(save_dir, f'lstm_timeline_{version}.png'),
                         bbox_inches='tight', dpi=300)
    plt.close(fig_timeline)
    print("  \u2713 Prediction timeline plot saved")

    if mask is not None and np.sum(mask) > 0:
        y_test_c = y_test[mask]
        y_pred_c = y_pred[mask]
        fig_c = plot_pred_vs_actual(
            y_test_c, y_pred_c, TARGET_COLS,
            title_suffix="LSTM (Rectangle: 20x16 mm)",
            show_units=True,
        )
        fig_c.savefig(os.path.join(save_dir, f'lstm_predictions_constrained_{version}.png'),
                      bbox_inches='tight', dpi=300)
        plt.close(fig_c)
        print("  \u2713 Constrained region plot saved")

    print(f"\nAll files saved to: {save_dir}")

    print("\n" + "="*70)
    print("PROCESS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
