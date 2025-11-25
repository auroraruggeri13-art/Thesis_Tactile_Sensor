#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCN for tactile time series with:
- X scaler (fit on train), Y scaler (fit on train)
- inverse-variance weighted MSE over targets
- ReduceLROnPlateau scheduler, early stopping
- optional derivative channels
"""

import os, math, pickle
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# ======  CONFIG  =========
# =========================
@dataclass
class Config:
    # Data
    DATA_DIRECTORY: str = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data"
    CSV_FILENAMES: List[str] = None
    BARO_COLS: List[str] = None
    TARGETS: List[str] = None
    TIME_COL: str = None  # optional

    # Features
    ADD_DERIVATIVES: bool = True  # add Δb/Δt as extra channels

    # Windowing
    SEQ_LEN: int = 256
    PRED_OFFSET: int = 0
    STRIDE: int = 4

    # Splits
    TRAIN_FRAC: float = 0.8
    VAL_FRAC: float = 0.1  # test = rest

    # TCN
    IN_CHANNELS: int = 6   # will be updated if derivatives added
    HIDDEN_CHANNELS: int = 256
    N_BLOCKS: int = 6
    KERNEL_SIZE: int = 3
    DROPOUT: float = 0.10

    # Optimization
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    BATCH_SIZE: int = 256
    EPOCHS: int = 100
    EARLY_STOP_PATIENCE: int = 15
    USE_WEIGHTED_LOSS: bool = True

    # Saving
    SAVE_DIR: str = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\tcn"
    MODEL_NAME: str = "tcn_v4.pt"
    X_SCALER_NAME: str = "tcn_x_scaler_v4.pkl"
    Y_SCALER_NAME: str = "tcn_y_scaler_v4.pkl"
    META_NAME: str = "tcn_meta_v4.pkl"

def default_config() -> Config:
    cfg = Config()
    cfg.CSV_FILENAMES = [
        r"test 4101 - sensor v4\synchronized_events_4101.csv",
        # add more if needed
    ]
    cfg.BARO_COLS = ['b1','b2','b3','b4','b5','b6']
    cfg.TARGETS = ['x','y','fx','fy','fz']
    return cfg

# =========================
# ======  DATASET  ========
# =========================
class SlidingWindowDataset(Dataset):
    def __init__(self, X_2d: np.ndarray, Y_2d: np.ndarray, seq_len: int, pred_offset: int = 0, stride: int = 1):
        assert len(X_2d) == len(Y_2d)
        self.X = X_2d
        self.Y = Y_2d
        self.seq_len = seq_len
        self.pred_offset = pred_offset
        self.stride = stride

        T = len(X_2d)
        self.indices = []
        end = T - pred_offset
        i = 0
        while i + seq_len <= end:
            self.indices.append(i)
            i += stride

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        s = self.indices[idx]
        x_seq = self.X[s:s + self.seq_len]                       # (L, F)
        y_t   = self.Y[s + self.seq_len - 1 - self.pred_offset]  # (O,)
        return torch.from_numpy(x_seq).float(), torch.from_numpy(y_t).float()

def collate_seq(batch):
    xs, ys = zip(*batch)
    X = torch.stack(xs, dim=0).permute(0, 2, 1)  # (B, F, L)
    y = torch.stack(ys, dim=0)                   # (B, O)
    return X, y

# =========================
# ======  MODEL  ==========
# =========================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)
            if self.downsample.bias is not None: nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=6, hidden_channels=256, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        ch_in = in_channels
        dil = 1
        for _ in range(n_blocks):
            layers.append(TemporalBlock(ch_in, hidden_channels, kernel_size, dil, dropout))
            ch_in = hidden_channels
            dil *= 2
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, 1),
        )
    def forward(self, x):         # x: (B,C,L)
        h = self.tcn(x)           # (B,H,L)
        h_last = h[:, :, -1:]     # (B,H,1)
        out = self.head(h_last)   # (B,O,1)
        return out.squeeze(-1)    # (B,O)

def tcn_receptive_field(n_blocks: int, kernel_size: int) -> int:
    return 1 + 2*(kernel_size - 1)*(2**n_blocks - 1)

# =========================
# ======  UTILS  ==========
# =========================
def print_metrics(y_true, y_pred, targets: List[str]):
    print("\nPER-TARGET METRICS")
    for i, t in enumerate(targets):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = mean_squared_error(y_true[:, i], y_pred[:, i], squared=False)
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        unit = "mm" if t in ['x','y'] else "N"
        print(f"  {t:>2s} | MAE: {mae:.4f} {unit} | RMSE: {rmse:.4f} {unit} | R²: {r2:.4f}")

    if set(['x','y']).issubset(targets):
        ix = [targets.index('x'), targets.index('y')]
        err = y_true[:, ix] - y_pred[:, ix]
        comp_rmse = np.sqrt((err**2).mean())
        eucl = np.sqrt((err**2).sum(axis=1))
        eucl_rmse = np.sqrt((eucl**2).mean())
        print("\nContact (x,y):")
        print(f"  Component RMSE: {comp_rmse:.4f} mm | Euclidean RMSE: {eucl_rmse:.4f} mm | Mean dist: {eucl.mean():.4f} mm")

    f_names = [t for t in ['fx','fy','fz'] if t in targets]
    if f_names:
        ix = [targets.index(t) for t in f_names]
        err = y_true[:, ix] - y_pred[:, ix]
        comp_rmse = np.sqrt((err**2).mean())
        eucl = np.sqrt((err**2).sum(axis=1))
        eucl_rmse = np.sqrt((eucl**2).mean())
        print(f"\nForce ({', '.join(f_names)}):")
        print(f"  Component RMSE: {comp_rmse:.4f} N | Euclidean RMSE: {eucl_rmse:.4f} N | Mean magnitude: {eucl.mean():.4f} N")

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total = 0.0; n = 0
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y); n += len(y)
    return total / max(1, n)

@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total = 0.0; n = 0
    ys, ps = [], []
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        total += loss.item() * len(y); n += len(y)
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    y_full = np.vstack(ys) if ys else np.zeros((0,1))
    p_full = np.vstack(ps) if ps else np.zeros((0,1))
    return total / max(1, n), y_full, p_full

# =========================
# ======  MAIN  ===========
# =========================
def main():
    cfg = default_config()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    # ---- Load & concat ----
    dfs = []
    for rel in cfg.CSV_FILENAMES:
        fp = os.path.join(cfg.DATA_DIRECTORY, rel)
        if not os.path.exists(fp):
            print(f"[WARN] Missing: {fp} (skipped)")
            continue
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip().str.lower()
        dfs.append(df)
    if not dfs: raise FileNotFoundError("No valid CSVs found.")
    df = pd.concat(dfs, ignore_index=True)

    need_cols = [c for c in cfg.BARO_COLS + cfg.TARGETS if c in df.columns]
    df = df[need_cols].copy()

    # ---- Build X (optionally add derivatives) ----
    X_raw = df[cfg.BARO_COLS].to_numpy(dtype=np.float32)
    if cfg.ADD_DERIVATIVES:
        dX = np.vstack([np.zeros((1, X_raw.shape[1]), dtype=X_raw.dtype), np.diff(X_raw, axis=0)])
        X_raw = np.hstack([X_raw, dX])   # doubles feature count
    cfg.IN_CHANNELS = X_raw.shape[1]
    Y_raw = df[cfg.TARGETS].to_numpy(dtype=np.float32)

    # ---- Time-based split on raw ----
    T = len(df)
    i_train_end = int(T * cfg.TRAIN_FRAC)
    i_val_end   = int(T * (cfg.TRAIN_FRAC + cfg.VAL_FRAC))
    X_train_raw, X_val_raw, X_test_raw = X_raw[:i_train_end], X_raw[i_train_end:i_val_end], X_raw[i_val_end:]
    Y_train_raw, Y_val_raw, Y_test_raw = Y_raw[:i_train_end], Y_raw[i_train_end:i_val_end], Y_raw[i_val_end:]

    # ---- Scale X (fit on train) ----
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_val   = x_scaler.transform(X_val_raw)
    X_test  = x_scaler.transform(X_test_raw)

    # ---- Scale Y (fit on train) ----
    y_scaler = StandardScaler()
    Y_train = y_scaler.fit_transform(Y_train_raw)
    Y_val   = y_scaler.transform(Y_val_raw)
    Y_test  = y_scaler.transform(Y_test_raw)

    # ---- Weighted loss (inverse variance from ORIGINAL train Y) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.USE_WEIGHTED_LOSS:
        y_std_orig = Y_train_raw.std(axis=0) + 1e-8
        loss_w = torch.tensor(1.0 / (y_std_orig**2), dtype=torch.float32, device=device)
        def weighted_mse(pred, target):
            return (loss_w * (pred - target)**2).mean()
        criterion = weighted_mse
    else:
        criterion = nn.MSELoss()

    # ---- Datasets / Loaders ----
    train_ds = SlidingWindowDataset(X_train, Y_train, seq_len=cfg.SEQ_LEN, pred_offset=cfg.PRED_OFFSET, stride=cfg.STRIDE)
    val_ds   = SlidingWindowDataset(X_val,   Y_val,   seq_len=cfg.SEQ_LEN, pred_offset=cfg.PRED_OFFSET, stride=cfg.STRIDE)
    test_ds  = SlidingWindowDataset(X_test,  Y_test,  seq_len=cfg.SEQ_LEN, pred_offset=cfg.PRED_OFFSET, stride=cfg.STRIDE)

    train_ld = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  drop_last=True,  collate_fn=collate_seq)
    val_ld   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_seq)
    test_ld  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_seq)

    # ---- Model ----
    model = TCN(
        in_channels=cfg.IN_CHANNELS,
        out_channels=len(cfg.TARGETS),
        n_blocks=cfg.N_BLOCKS,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        kernel_size=cfg.KERNEL_SIZE,
        dropout=cfg.DROPOUT,
    ).to(device)

    rf = tcn_receptive_field(cfg.N_BLOCKS, cfg.KERNEL_SIZE)
    print(f"\nTCN receptive field (timesteps): {rf} => must be ≤ SEQ_LEN={cfg.SEQ_LEN}")
    if rf > cfg.SEQ_LEN:
        print("[NOTE] RF exceeds window; increase SEQ_LEN or reduce blocks/kernel.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    # ---- Train with early stopping ----
    best_val = float("inf"); best_state = None; patience = cfg.EARLY_STOP_PATIENCE
    for epoch in range(1, cfg.EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_ld, device, criterion, optimizer)
        vl_loss, _, _ = eval_epoch(model, val_ld, device, criterion)
        scheduler.step(vl_loss)
        print(f"Epoch {epoch:03d} | train MSE: {tr_loss:.6f} | val MSE: {vl_loss:.6f}")

        if vl_loss < best_val - 1e-8:
            best_val = vl_loss
            best_state = {"epoch": epoch, "model": model.state_dict()}
            patience = cfg.EARLY_STOP_PATIENCE
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping."); break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        print(f"Loaded best epoch: {best_state['epoch']} (val MSE={best_val:.6f})")

    # ---- Evaluate on test (invert scaling) ----
    test_mse, y_true_s, y_pred_s = eval_epoch(model, test_ld, device, criterion)
    y_true = y_scaler.inverse_transform(y_true_s)
    y_pred = y_scaler.inverse_transform(y_pred_s)
    print(f"\nTest MSE (in scaled space): {test_mse:.6f}")
    print_metrics(y_true, y_pred, cfg.TARGETS)

    # ---- Save ----
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    model_path  = os.path.join(cfg.SAVE_DIR, cfg.MODEL_NAME)
    xsc_path    = os.path.join(cfg.SAVE_DIR, cfg.X_SCALER_NAME)
    ysc_path    = os.path.join(cfg.SAVE_DIR, cfg.Y_SCALER_NAME)
    meta_path   = os.path.join(cfg.SAVE_DIR, cfg.META_NAME)

    torch.save(model.state_dict(), model_path)
    with open(xsc_path, 'wb') as f: pickle.dump(x_scaler, f)
    with open(ysc_path, 'wb') as f: pickle.dump(y_scaler, f)
    with open(meta_path, 'wb') as f:
        pickle.dump({
            "targets": cfg.TARGETS,
            "baro_cols": cfg.BARO_COLS,
            "seq_len": cfg.SEQ_LEN,
            "kernel_size": cfg.KERNEL_SIZE,
            "n_blocks": cfg.N_BLOCKS,
            "hidden_channels": cfg.HIDDEN_CHANNELS,
            "dropout": cfg.DROPOUT,
            "receptive_field": rf,
            "stride": cfg.STRIDE,
            "pred_offset": cfg.PRED_OFFSET,
            "add_derivatives": cfg.ADD_DERIVATIVES,
            "in_channels": cfg.IN_CHANNELS,
            "use_weighted_loss": cfg.USE_WEIGHTED_LOSS,
        }, f)

    print(f"\nSaved model: {model_path}")
    print(f"Saved X scaler: {xsc_path}")
    print(f"Saved Y scaler: {ysc_path}")
    print(f"Saved meta: {meta_path}")

if __name__ == "__main__":
    main()
