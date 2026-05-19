#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based sliding window predictions for tactile sensor.
Mirrors the structure of RF_or_XGB_sliding_window_predictions.py,
but replaces LightGBM with a Transformer encoder (PyTorch).

Input:  6 barometer channels + derivatives in a sliding window -> 3D sequence
Model:  TransformerEncoder -> last-token regression head
Output: x, y, fx, fy, fz
"""
import os
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import mean_absolute_error, r2_score

from utils.metrics_utils import calculate_grouped_rmse
from utils.plot_utils import plot_pred_vs_actual, plot_error_distributions
from utils.signal_utils import maybe_denoise, convert_sentinel_to_nan


# ============================================================
# ======================= CONFIG =============================
# ============================================================

DATA_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data"
sensor_version = 5.01
TRAIN_FILENAME      = f"train_data_v{sensor_version}.csv"
VALIDATION_FILENAME = f"validation_data_v{sensor_version}.csv"
TEST_FILENAME       = f"test_data_v{sensor_version}.csv"

TIME_COL    = "t"
BARO_COLS   = ["b1", "b2", "b3", "b4", "b5", "b6"]
TARGET_COLS = ["x", "y", "fx", "fy", "fz"]

RANDOM_STATE = 42

# Sliding window size (in samples). Looks at past WINDOW_SIZE samples.
# Window of size W -> sequence length W+1 (includes current sample)
WINDOW_SIZE = 10   # ~0.1 s at 100 Hz

# Window size study
RUN_WINDOW_SIZE_STUDY = False
WINDOW_SIZES = [1, 5, 10, 15, 20]

# No-contact sentinel
CONVERT_SENTINEL_TO_NAN = True
NO_CONTACT_SENTINEL = -999.0

# Optional denoising (rolling mean on barometer channels BEFORE diffs)
APPLY_DENOISING = True
DENOISE_WINDOW  = 5

# Feature engineering: include first derivatives as extra channels.
# Set True to also add second derivatives.
USE_SECOND_DERIVATIVE = True

# ============================================================
# ================ TRANSFORMER HYPERPARAMETERS ===============
# ============================================================

D_MODEL    = 64    # Embedding dimension (must be divisible by N_HEADS)
N_HEADS    = 4     # Number of attention heads
N_LAYERS   = 3     # Number of TransformerEncoder layers
D_FFN      = 128   # Feed-forward network hidden dimension
DROPOUT    = 0.1   # Dropout rate

# Training
BATCH_SIZE   = 512
MAX_EPOCHS   = 200
LEARNING_RATE = 1e-3
PATIENCE     = 20   # Early stopping: epochs without val improvement
LR_PATIENCE  = 10   # ReduceLROnPlateau: epochs before halving LR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# =============== FEATURE ENGINEERING ========================
# ============================================================

def build_window_sequences(df, baro_cols, time_col, target_cols, window_size,
                            max_time_gap=0.05, use_second_derivative=False):
    """
    Build 3D windowed sequences for the Transformer.

    For each index i in [window_size, N]:
        - Check if window has large time gaps (file boundaries)
        - If valid: sequence = (W+1, C)
            C = n_baro [+ n_baro (d1)] [+ n_baro (d2)]
        - target = targets at index i

    This is the Transformer equivalent of build_window_features() in the
    LightGBM script. Instead of flattening the window to 1D, we keep the
    time dimension so the Transformer can attend across timesteps.

    Args:
        window_size:          Number of past samples (W). Sequence length = W+1.
        max_time_gap:         Max allowed gap within a window (seconds).
        use_second_derivative: If True, include second derivatives as channels.

    Returns:
        X:          (M, W+1, C)  – sequences
        y:          (M, n_targets)
        center_df:  DataFrame of the current-timestep rows (for diagnostics)
        n_channels: C (input feature dimension per timestep)
    """
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df = maybe_denoise(df, baro_cols, apply_denoising=APPLY_DENOISING,
                       denoise_window=DENOISE_WINDOW)

    # Build channel list: raw + d1 [+ d2]
    channel_cols = list(baro_cols)
    for col in baro_cols:
        d1 = df[col].diff()
        df[f"{col}_d1"] = d1.fillna(0.0)
        channel_cols.append(f"{col}_d1")
        if use_second_derivative:
            d2 = d1.diff()
            df[f"{col}_d2"] = d2.fillna(0.0)
            channel_cols.append(f"{col}_d2")

    N = len(df)
    W = window_size
    C = len(channel_cols)

    if N <= W:
        raise ValueError(f"Not enough samples ({N}) for window size {W}")

    print(f"Building windowed sequences: seq_len={W+1}, channels={C} ...")

    time_vals = df[time_col].values
    data_arr  = df[channel_cols].values  # (N, C)

    X_list       = []
    y_list       = []
    valid_indices = []
    skipped      = 0

    for current_idx in range(W, N):
        start = current_idx - W
        end   = current_idx + 1  # inclusive of current sample

        window_times = time_vals[start:end]
        if np.max(np.diff(window_times)) > max_time_gap:
            skipped += 1
            continue

        X_list.append(data_arr[start:end, :])          # (W+1, C)
        y_list.append(df.loc[current_idx, target_cols].values.astype(float))
        valid_indices.append(current_idx)

    print(f"Built {len(X_list)} valid sequences, skipped {skipped} boundary windows")

    X         = np.stack(X_list, axis=0)               # (M, W+1, C)
    y         = np.array(y_list)                        # (M, n_targets)
    center_df = df.iloc[valid_indices].reset_index(drop=True)

    return X, y, center_df, C


# ============================================================
# ====================== DATASET =============================
# ============================================================

class SensorDataset(Dataset):
    """Simple PyTorch Dataset wrapping numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, seq_len, C),  y: (N, n_targets)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# =================== TRANSFORMER MODEL ======================
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (added to projected input)."""

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position  = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term  = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TactileSensorTransformer(nn.Module):
    """
    Transformer encoder for multi-output regression of tactile sensor data.

    Architecture:
        1. Linear projection: (batch, seq_len, C) -> (batch, seq_len, d_model)
        2. Sinusoidal positional encoding
        3. Stack of TransformerEncoderLayer (pre-norm, multi-head attention + FFN)
        4. Take the last-timestep token (current sample) as the summary vector
        5. Linear regression head: d_model -> n_targets

    The last timestep corresponds to the current measurement, and the earlier
    timesteps provide temporal context via self-attention.
    """

    def __init__(self, n_channels: int, n_targets: int,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, d_ffn: int = 128,
                 dropout: float = 0.1, max_seq_len: int = 256):
        super().__init__()

        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-norm (more stable training than post-norm)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)
        self.norm        = nn.LayerNorm(d_model)
        self.head        = nn.Linear(d_model, n_targets)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, C)
        x = self.input_proj(x)      # (batch, seq_len, d_model)
        x = self.pos_enc(x)         # add positional encoding
        x = self.transformer(x)     # (batch, seq_len, d_model)
        x = self.norm(x[:, -1, :])  # last timestep = current sample
        return self.head(x)         # (batch, n_targets)


# ============================================================
# ===================== TRAINING =============================
# ============================================================

def train_transformer(model, train_loader, val_loader,
                      max_epochs, lr, patience, lr_patience, device):
    """
    Train the Transformer with Adam, ReduceLROnPlateau, and early stopping.

    Returns:
        loss_history: dict with 'train' and 'valid' MSE lists (one per epoch)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=lr_patience,
                                  factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss     = float('inf')
    best_state        = None
    epochs_no_improve = 0
    train_losses      = []
    val_losses        = []

    model.to(device)

    for epoch in range(1, max_epochs + 1):
        # --- Training pass ---
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(X_batch)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # --- Validation pass ---
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                running_val += criterion(model(X_batch), y_batch).item() * len(X_batch)
        val_loss = running_val / len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    print(f"  Best validation MSE: {best_val_loss:.5f}")

    return {'train': train_losses, 'valid': val_losses}


def predict(model, X: np.ndarray, device, batch_size: int = 2048) -> np.ndarray:
    """Inference on a numpy array X. Returns numpy predictions."""
    model.eval()
    model.to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = X_tensor[start:start + batch_size].to(device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ============================================================
# ======================= PLOTTING ===========================
# ============================================================

def plot_loss_curves(loss_history: dict, model_name: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(loss_history['train'], label='Training Loss',   linewidth=2, color='#005c7f')
    ax.plot(loss_history['valid'], label='Validation Loss', linewidth=2, color='#44b155')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE Loss', fontsize=11)
    ax.set_title(f'{model_name} Training Loss Curve', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_window_size_comparison(window_sizes, results):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    contact_rmse = [results[ws]['contact_euclidean_rmse'] for ws in window_sizes]
    force_rmse   = [results[ws]['force_euclidean_rmse']   for ws in window_sizes]

    color1 = '#005c7f'
    ax1.set_xlabel('Window Size (samples)', fontsize=12)
    ax1.set_ylabel('Contact Position RMSE (mm)', fontsize=12, color=color1)
    line1 = ax1.plot(window_sizes, contact_rmse, marker='o', linewidth=2.5, markersize=9,
                     color=color1, label='Contact Position Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(window_sizes)
    best_c = np.argmin(contact_rmse)
    ax1.scatter([window_sizes[best_c]], [contact_rmse[best_c]],
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)

    ax2 = ax1.twinx()
    color2 = '#44b155'
    ax2.set_ylabel('3-DOF Force RMSE (N)', fontsize=12, color=color2)
    line2 = ax2.plot(window_sizes, force_rmse, marker='s', linewidth=2.5, markersize=9,
                     color=color2, label='Force Error')
    ax2.tick_params(axis='y', labelcolor=color2)
    best_f = np.argmin(force_rmse)
    ax2.scatter([window_sizes[best_f]], [force_rmse[best_f]],
                color='red', s=200, zorder=5, marker='*', edgecolors='darkred', linewidths=1.5)

    lines  = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=2, fontsize=11, frameon=True, shadow=True)
    plt.title('Window Size Hyperparameter Study (Transformer)', fontsize=14,
              fontweight='bold', pad=15)
    fig.tight_layout()
    return fig


# ============================================================
# ======================== MAIN ==============================
# ============================================================

def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print(f"Using device: {DEVICE}")

    # ---------- Load data ----------
    train_path = os.path.join(DATA_DIRECTORY, TRAIN_FILENAME)
    val_path   = os.path.join(DATA_DIRECTORY, VALIDATION_FILENAME)
    test_path  = os.path.join(DATA_DIRECTORY, TEST_FILENAME)

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

    train_df = pd.read_csv(train_path, skipinitialspace=True)
    train_df.columns = train_df.columns.str.strip()
    val_df = pd.read_csv(val_path, skipinitialspace=True)
    val_df.columns = val_df.columns.str.strip()
    test_df = pd.read_csv(test_path, skipinitialspace=True)
    test_df.columns = test_df.columns.str.strip()

    print("Loaded train data with columns:", train_df.columns.tolist())
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if CONVERT_SENTINEL_TO_NAN:
        print("\nConverting sentinel values to NaN...")
        train_df = convert_sentinel_to_nan(train_df, TARGET_COLS, NO_CONTACT_SENTINEL)
        val_df   = convert_sentinel_to_nan(val_df,   TARGET_COLS, NO_CONTACT_SENTINEL)
        test_df  = convert_sentinel_to_nan(test_df,  TARGET_COLS, NO_CONTACT_SENTINEL)

    needed_cols = [TIME_COL] + BARO_COLS + TARGET_COLS
    train_df = train_df.dropna(subset=needed_cols).reset_index(drop=True)
    val_df   = val_df.dropna(subset=needed_cols).reset_index(drop=True)
    test_df  = test_df.dropna(subset=needed_cols).reset_index(drop=True)

    # ---------- Window Size Study ----------
    if RUN_WINDOW_SIZE_STUDY:
        print("\n" + "="*70)
        print("RUNNING WINDOW SIZE HYPERPARAMETER STUDY (Transformer)")
        print(f"Testing window sizes: {WINDOW_SIZES}")
        print("="*70)

        window_size_results = {}

        for ws in WINDOW_SIZES:
            print(f"\n{'='*70}\nTesting Window Size: {ws}\n{'='*70}")

            X_tr, y_tr, _, C = build_window_sequences(
                train_df, BARO_COLS, TIME_COL, TARGET_COLS, ws,
                use_second_derivative=USE_SECOND_DERIVATIVE)
            X_vl, y_vl, _, _ = build_window_sequences(
                val_df, BARO_COLS, TIME_COL, TARGET_COLS, ws,
                use_second_derivative=USE_SECOND_DERIVATIVE)

            # Normalize per-channel using train statistics
            X_mean = X_tr.mean(axis=(0, 1), keepdims=True)
            X_std  = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
            X_tr = (X_tr - X_mean) / X_std
            X_vl = (X_vl - X_mean) / X_std

            train_loader = DataLoader(SensorDataset(X_tr, y_tr),
                                      batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
            val_loader   = DataLoader(SensorDataset(X_vl, y_vl),
                                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            model = TactileSensorTransformer(
                n_channels=C, n_targets=len(TARGET_COLS),
                d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                d_ffn=D_FFN, dropout=DROPOUT, max_seq_len=ws + 2,
            )
            train_transformer(model, train_loader, val_loader,
                              max_epochs=MAX_EPOCHS, lr=LEARNING_RATE,
                              patience=PATIENCE, lr_patience=LR_PATIENCE, device=DEVICE)

            y_pred_vl = predict(model, X_vl, DEVICE)
            metrics = calculate_grouped_rmse(y_vl, y_pred_vl, TARGET_COLS,
                                             title_suffix=f"Window Size = {ws}",
                                             return_metrics=True)
            window_size_results[ws] = metrics

        fig_comp = plot_window_size_comparison(WINDOW_SIZES, window_size_results)
        save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
        os.makedirs(save_dir, exist_ok=True)
        version = f'v{sensor_version:.2f}'
        fig_comp.savefig(
            os.path.join(save_dir, f'transformer_window_size_comparison_{version}.png'),
            bbox_inches='tight', dpi=300)
        plt.show()

        print("\n" + "="*70 + "\nWINDOW SIZE STUDY SUMMARY\n" + "="*70)
        print(f"{'Window Size':<15} {'Contact RMSE (mm)':<20} {'Force RMSE (N)':<20}")
        print("-" * 70)
        for ws in WINDOW_SIZES:
            cr = window_size_results[ws]['contact_euclidean_rmse']
            fr = window_size_results[ws]['force_euclidean_rmse']
            print(f"{ws:<15} {cr:<20.4f} {fr:<20.4f}")

        best_ws_contact = min(WINDOW_SIZES, key=lambda ws: window_size_results[ws]['contact_euclidean_rmse'])
        best_ws_force   = min(WINDOW_SIZES, key=lambda ws: window_size_results[ws]['force_euclidean_rmse'])
        print(f"\nBest window size for contact position: {best_ws_contact}")
        print(f"Best window size for force:            {best_ws_force}")
        print("\nWindow size study complete. Exiting.")
        return

    # ---------- Single Window Size Run ----------
    print(f"\nUsing window size: {WINDOW_SIZE}")

    X_train, y_train, train_center, C = build_window_sequences(
        train_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
        use_second_derivative=USE_SECOND_DERIVATIVE)
    X_val, y_val, val_center, _ = build_window_sequences(
        val_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
        use_second_derivative=USE_SECOND_DERIVATIVE)
    X_test, y_test, test_center, _ = build_window_sequences(
        test_df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
        use_second_derivative=USE_SECOND_DERIVATIVE)

    print(f"Sequence shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"  (samples, seq_len={WINDOW_SIZE+1}, channels={C})")

    # ---------- Normalize per-channel (train statistics) ----------
    print("\nNormalizing sequences (per-channel, train statistics)...")
    X_mean = X_train.mean(axis=(0, 1), keepdims=True)   # (1, 1, C)
    X_std  = X_train.std(axis=(0, 1),  keepdims=True) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_val   = (X_val   - X_mean) / X_std
    X_test  = (X_test  - X_mean) / X_std

    # ---------- Datasets and dataloaders ----------
    train_loader = DataLoader(SensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(SensorDataset(X_val,   y_val),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ---------- Build model ----------
    print(f"\nBuilding Transformer model:")
    print(f"  d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, d_ffn={D_FFN}, dropout={DROPOUT}")
    print(f"  input channels={C}, output targets={len(TARGET_COLS)}")

    model = TactileSensorTransformer(
        n_channels=C,
        n_targets=len(TARGET_COLS),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ffn=D_FFN,
        dropout=DROPOUT,
        max_seq_len=WINDOW_SIZE + 2,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")

    # ---------- Train ----------
    print(f"\nTraining Transformer (max_epochs={MAX_EPOCHS}, patience={PATIENCE}) ...")
    loss_history = train_transformer(
        model, train_loader, val_loader,
        max_epochs=MAX_EPOCHS, lr=LEARNING_RATE,
        patience=PATIENCE, lr_patience=LR_PATIENCE, device=DEVICE,
    )

    # ---------- Evaluate ----------
    y_hat = predict(model, X_test, DEVICE)

    d2_suffix = " + d2" if USE_SECOND_DERIVATIVE else ""
    print(f"\n=== Performance (Transformer) with window + gradients{d2_suffix} ===")
    for i, col in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test[:, i], y_hat[:, i])
        r2  = r2_score(y_test[:, i], y_hat[:, i])
        print(f"{col:>3} | MAE: {mae:6.3f} | R²: {r2:6.3f}")

    calculate_grouped_rmse(y_test, y_hat, TARGET_COLS,
                           title_suffix="(Transformer - All Data)")

    # ---------- Save ----------
    print("\n" + "="*70 + "\nSAVING MODEL AND PLOTS\n" + "="*70)

    save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models"
    save_dir_comparison = os.path.join(save_dir, "comparison_plots")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_comparison, exist_ok=True)

    version    = f'v{sensor_version:.3f}'
    model_name = "Transformer"

    # Save model weights + normalization stats (everything needed for inference)
    model_path = os.path.join(save_dir, f'transformer_model_{version}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_channels': C,
            'n_targets':  len(TARGET_COLS),
            'd_model':    D_MODEL,
            'n_heads':    N_HEADS,
            'n_layers':   N_LAYERS,
            'd_ffn':      D_FFN,
            'dropout':    DROPOUT,
            'max_seq_len': WINDOW_SIZE + 2,
        },
        'X_mean':               X_mean,
        'X_std':                X_std,
        'window_size':          WINDOW_SIZE,
        'baro_cols':            BARO_COLS,
        'target_cols':          TARGET_COLS,
        'use_second_derivative': USE_SECOND_DERIVATIVE,
    }, model_path)
    print(f"Model saved to: {model_path}")

    # Loss curve
    fig_loss = plot_loss_curves(loss_history, model_name)
    fig_loss.savefig(os.path.join(save_dir, f'loss_curve_transformer_{version}.png'),
                     bbox_inches='tight', dpi=300)
    plt.show()
    print("Loss curve saved.")

    # Pred vs actual
    fig_pred = plot_pred_vs_actual(y_test, y_hat, TARGET_COLS,
                                   title_suffix=model_name, scatter_color='#292f56')
    fig_pred.savefig(os.path.join(save_dir, f'sliding_window_predictions_transformer_{version}.png'),
                     bbox_inches='tight', dpi=300)
    fig_pred.savefig(
        os.path.join(save_dir_comparison, f'actual_vs_prediction_transformer_{version}.png'),
        bbox_inches='tight', dpi=300)
    plt.show()
    print("Prediction plot saved.")

    # Error distribution
    fig_err = plot_error_distributions(y_test, y_hat, TARGET_COLS, title_suffix=model_name)
    fig_err.savefig(os.path.join(save_dir, f'error_distribution_transformer_{version}.png'),
                    bbox_inches='tight', dpi=300)
    plt.close()
    print("Error distribution plot saved.")

    print(f"\nAll outputs saved to: {save_dir}")
    print("Process complete!")


if __name__ == "__main__":
    main()
