#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barometer–force synchronization with manual time shift + masking only.

Steps:
1) Load processing_test CSV (epoch seconds in 'time', positions, forces)
2) Load barometer CSV (uses 'Epoch_s' if present)
3) Apply manual barometer time shift (seconds)
4) Synchronize with merge_asof (nearest, tolerance)
5) Apply spatial mask (keep only rows within bounds)
6) Save synchronized CSV

Author: Aurora Ruggeri
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================== CONFIGURATION ===========================

TEST_NUM = 4105
VERSION_NUM = 4

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data")
DATA_DIR = BASE_DIR / f"test {TEST_NUM} - sensor v{VERSION_NUM}"

# Input files
PROCESSING_FILE = DATA_DIR / f"processing_test_{TEST_NUM}.csv"
BAROMETER_FILE  = None  # if None, picks most recent datalog_*.csv in DATA_DIR

# Output
OUTPUT_FILE = DATA_DIR / f"synchronized_events_{TEST_NUM}.csv"
save_plots_dir = DATA_DIR / "plots synchronized"
save_plots_dir.mkdir(exist_ok=True, parents=True)

# --- Synchronization params (same style as your short script) ---
ASOF_DIRECTION   = "nearest"
ASOF_TOLERANCE_S = 0.05   # seconds

# --- Manual time shift (APPLIED TO BAROMETER 't') ---
# Positive -> shift barometer FORWARD in time; Negative -> backward
BARO_TIME_SHIFT_S = 0.00000234

# --- Spatial masking (keep this) ---
SPATIAL_FILTER = {
    'x_range': (-20, 20),  # mm
    'y_range': (-8, 8),    # mm
    'z_range': (-1.0, 1.0) # mm
}

# =============================== HELPERS ===============================

def ensure_seconds(series: pd.Series) -> pd.Series:
    """Coerce a time-like numeric column to seconds."""
    s = pd.to_numeric(series, errors='coerce')
    med = s.dropna().abs().median()
    if med > 1e12:   s *= 1e-9  # ns
    elif med > 1e9:  s *= 1e-6  # us
    elif med > 1e6:  s *= 1e-3  # ms
    return s

def find_barometer_file(directory: Path) -> Path:
    """Pick most-recent datalog_*.csv if BAROMETER_FILE is not set."""
    candidates = list(directory.glob("datalog_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No 'datalog_*.csv' in {directory}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

# =============================== LOADERS ===============================

def load_processing_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ['time', 'x_position_mm', 'y_position_mm', 'z_position_mm']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in processing CSV: {missing}")

    # forces (upper/lower accepted)
    force_names = ['Fx','Fy','Fz','Tx','Ty','Tz']
    present_forces = []
    for c in force_names:
        if c in df.columns:
            present_forces.append(c)
        elif c.lower() in df.columns:
            present_forces.append(c.lower())
    if not present_forces:
        raise ValueError("No force/torque columns found (Fx/Fy/Fz/Tx/Ty/Tz).")

    df['time'] = ensure_seconds(df['time'])
    keep = ['time','x_position_mm','y_position_mm','z_position_mm'] + present_forces
    return df[keep].sort_values('time', ignore_index=True)

def load_barometer_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if 'Epoch_s' in df.columns:
        t = ensure_seconds(pd.to_numeric(df['Epoch_s'], errors='coerce'))
    else:
        # Fallback: parse from filename date + 'Timestamp' column (legacy format)
        ts_col = next((c for c in df.columns if 'timestamp' in c.lower()), None)
        if ts_col is None:
            raise ValueError("Barometer CSV needs 'Epoch_s' or a 'Timestamp' column.")
        m = re.search(r'datalog_(\d{4}-\d{2}-\d{2})', os.path.basename(str(path)))
        if not m:
            raise ValueError("Filename must look like 'datalog_YYYY-MM-DD_...csv' for legacy mode.")
        date_str = m.group(1)
        dt_strings = date_str + ' ' + df[ts_col].astype(str)
        dt = pd.to_datetime(dt_strings, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce', utc=True)
        t = dt.view('int64') / 1e9  # seconds

    # barometer columns
    baro_cols = [f'barometer {i}' for i in range(1,7) if f'barometer {i}' in df.columns]
    if len(baro_cols) != 6:
        raise ValueError(f"Expected 6 barometer columns, found {len(baro_cols)}: {baro_cols}")

    out = pd.DataFrame({'time': t})
    for i, c in enumerate(baro_cols, start=1):
        out[f'b{i}'] = pd.to_numeric(df[c], errors='coerce')

    # apply manual shift (same logic as your short script)
    out['time'] = out['time'] + BARO_TIME_SHIFT_S

    return out.dropna(subset=['time']).sort_values('time', ignore_index=True)

# =========================== CORE OPERATIONS ===========================

def synchronize_data(proc_df: pd.DataFrame, baro_df: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbor time merge with tolerance (epoch seconds)."""
    merged = pd.merge_asof(
        proc_df.sort_values('time'),
        baro_df.sort_values('time'),
        on='time',
        direction=ASOF_DIRECTION,
        tolerance=ASOF_TOLERANCE_S
    )
    # keep only rows where barometers matched
    merged = merged.dropna(subset=['b1','b2','b3','b4','b5','b6'])
    return merged

def apply_spatial_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows in the specified spatial bounds."""
    if SPATIAL_FILTER is None:
        return df

    x_min, x_max = SPATIAL_FILTER['x_range']
    y_min, y_max = SPATIAL_FILTER['y_range']
    z_min, z_max = SPATIAL_FILTER['z_range']

    mask = (
        df['x_position_mm'].between(x_min, x_max) &
        df['y_position_mm'].between(y_min, y_max) &
        df['z_position_mm'].between(z_min, z_max)
    )
    return df.loc[mask].copy()

def zero_barometers(df, baseline_duration=1.0):
    """
    Subtract the mean of the first `baseline_duration` seconds from each barometer.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'time' and barometer columns (b1..b6 or barometer 1..6).
    baseline_duration : float
        Duration in seconds from the start used to compute the baseline mean.

    Returns
    -------
    df_zeroed : pd.DataFrame
        Copy of df with barometer data baseline-corrected.
    """
    df = df.copy()
    t0 = df["time"].min()
    mask = df["time"] <= t0 + baseline_duration

    # Find barometer columns automatically
    baro_cols = [c for c in df.columns if c.lower().startswith("b") or "barometer" in c.lower()]
    if not baro_cols:
        print("No barometer columns found.")
        return df

    for c in baro_cols:
        baseline = df.loc[mask, c].mean()
        df[c] = df[c] - baseline
        print(f"{c}: baseline {baseline:.2f} hPa subtracted")

    return df

# ==============================PLOTTING FUNCTIONS=======================

def _force_name(df, base):
    return base if base in df.columns else base.lower() if base.lower() in df.columns else None

def plot_barometers_subplots(df, save_path=None, suptitle="Barometers"):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for i in range(6):
        col = f"b{i+1}"
        if col not in df.columns: continue
        axes[i].plot(df["time"], df[col], linewidth=1.2)
        axes[i].set_ylabel(f"{col} [hPa]")
        axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0,0,1,0.97])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

def plot_forces_torques_subplots(df, save_path=None, suptitle="ATI Force/Torque"):
    cols = []
    for base in ["Fx","Fy","Fz","Tx","Ty","Tz"]:
        c = _force_name(df, base)
        if c: cols.append((c, f"{base}"))
    if not cols:
        print("No ATI columns found."); return None

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for ax, (c, label) in zip(axes, cols):
        ax.plot(df["time"], df[c], linewidth=1.2)
        unit = "N" if label.startswith("F") else "N·m"
        ax.set_ylabel(f"{label} [{unit}]")
        ax.grid(alpha=0.3)
    axes[min(len(cols),6)-1].set_xlabel("Time [s]")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0,0,1,0.97])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

def plot_barometers_vs_force(df, force_col="Fz", save_path=None, suptitle=None):
    fcol = _force_name(df, force_col)
    if not fcol:
        print(f"No force column matching {force_col} found."); return None

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    for i in range(6):
        ax1 = axes[i]
        bcol = f"b{i+1}"
        if bcol not in df.columns: continue
        # left: barometer
        line1 = ax1.plot(df["time"], df[bcol], label=bcol, linewidth=1.2)
        ax1.set_ylabel(f"{bcol} [hPa]")
        ax1.grid(alpha=0.3)
        # right: force (scaled via twin axis)
        ax2 = ax1.twinx()
        line2 = ax2.plot(df["time"], df[fcol], linestyle="--", alpha=0.8, label=fcol, linewidth=1.2, color="C3")
        unit = "N" if fcol.lower().startswith("f") else "N·m"
        ax2.set_ylabel(f"{fcol} [{unit}]")
        # combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(suptitle or f"Barometers vs {force_col}")
    fig.tight_layout(rect=[0,0,1,0.97])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

def plot_barometers_vs_fz(df, save_path=None):
    return plot_barometers_vs_force(df, "Fz", save_path, suptitle="Barometers vs Fz")

def plot_barometers_vs_fx(df, save_path=None):
    return plot_barometers_vs_force(df, "Fx", save_path, suptitle="Barometers vs Fx")

def plot_barometers_vs_fy(df, save_path=None):
    return plot_barometers_vs_force(df, "Fy", save_path, suptitle="Barometers vs Fy")

def plot_path_xy_filtered(out_dir: str,
                          X_mm: np.ndarray, Y_mm: np.ndarray, t_s: np.ndarray,
                          test_num: int,
                          break_on_time_gaps: bool = True,  gap_factor_time: float = 5.0,
                          break_on_xy_jumps: bool = True,  jump_factor: float = 5.0,
                          rect_size_mm=(40.0, 16.0),
                          xlim=(-25, 25), ylim=(-10, 10)):

    plt.figure(figsize=(8, 6))

    n = len(X_mm)
    breaks = []

    if break_on_time_gaps and n > 1:
        dt = np.diff(t_s)
        med_dt = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.0
        if med_dt > 0:
            b = np.where(dt > gap_factor_time * med_dt)[0]
            breaks.extend(b.tolist())

    if break_on_xy_jumps and n > 1:
        steps = np.hypot(np.diff(X_mm), np.diff(Y_mm))
        med_step = np.median(steps[steps > 0]) if np.any(steps > 0) else 0.0
        if med_step > 0:
            b = np.where(steps > jump_factor * med_step)[0]
            breaks.extend(b.tolist())

    breaks = sorted(set(breaks))

    # Build arrays with NaNs inserted after each break
    if breaks:
        xs, ys = [X_mm[0]], [Y_mm[0]]
        for i in range(n - 1):
            xs.append(X_mm[i+1]); ys.append(Y_mm[i+1])
            if i in breaks:
                xs.append(np.nan); ys.append(np.nan)
        x_plot = np.array(xs); y_plot = np.array(ys)
    else:
        x_plot, y_plot = X_mm, Y_mm

    plt.plot(x_plot, y_plot, linewidth=1.2)

    # Sensor rectangle (centered)
    try:
        ax = plt.gca()
        rw, rh = rect_size_mm
        rect = Rectangle((-rw/2, -rh/2), rw, rh, fill=False,
                         edgecolor='k', linewidth=1.0, linestyle='--', alpha=0.9)
        ax.add_patch(rect)
    except Exception:
        pass

    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim(*xlim); plt.ylim(*ylim)
    plt.xlabel("Top X [mm]"); plt.ylabel("Top Y [mm]")
    plt.title(f"Tip path on top surface — trial {test_num}")

    out_path = os.path.join(out_dir, f"tip_path_top_xy_trial{test_num}.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {out_path}")

# ================================ MAIN =================================

def main():
    print("#"*60)
    print(f"# BARO–FORCE SYNC | Test {TEST_NUM} v{VERSION_NUM}")
    print("#"*60)
    print(f"- Baro manual time shift: {BARO_TIME_SHIFT_S:+.6f} s")
    print(f"- merge_asof: direction={ASOF_DIRECTION}, tolerance={ASOF_TOLERANCE_S}s")

    if not PROCESSING_FILE.exists():
        raise FileNotFoundError(PROCESSING_FILE)

    baro_path = BAROMETER_FILE if BAROMETER_FILE else find_barometer_file(DATA_DIR)
    if not Path(baro_path).exists():
        raise FileNotFoundError(baro_path)

    proc_df = load_processing_data(PROCESSING_FILE)
    baro_df = load_barometer_data(baro_path)

    merged = synchronize_data(proc_df, baro_df)
    
    # Zero the barometer data before plotting
    merged = zero_barometers(merged, baseline_duration=0.000002)
    
    df = merged  # or pd.read_csv("synchronized_events_4105.csv")

    plot_barometers_subplots(df, save_path=save_plots_dir / "barometers_subplots.png")
    plot_forces_torques_subplots(df, save_path=save_plots_dir / "ati_forces_torques.png")

    plot_barometers_vs_fz(df, save_path=save_plots_dir / "barometers_vs_Fz.png")
    plot_barometers_vs_fx(df, save_path=save_plots_dir / "barometers_vs_Fx.png")
    plot_barometers_vs_fy(df, save_path=save_plots_dir / "barometers_vs_Fy.png")

    plt.show()  # if you want to display them immediately
    
    masked = apply_spatial_filter(merged)
    
    plot_path_xy_filtered(
        out_dir=str(save_plots_dir),
        X_mm=masked["x_position_mm"].to_numpy(),
        Y_mm=masked["y_position_mm"].to_numpy(),
        t_s=masked["time"].to_numpy(),
        test_num=TEST_NUM
    )


    if masked.empty:
        print("WARNING: no rows after spatial filter — saving UNSFILTERED synchronized data instead.")
        masked = merged

    # --- rename columns for export ---
    masked = masked.rename(columns={
        'x_position_mm': 'x', 'y_position_mm': 'y', 'z_position_mm': 'z',
        'Fx': 'fx', 'Fy': 'fy', 'Fz': 'fz', 'fx': 'fx', 'fy': 'fy', 'fz': 'fz',
        'barometer 1': 'b1', 'barometer 2': 'b2', 'barometer 3': 'b3',
        'barometer 4': 'b4', 'barometer 5': 'b5', 'barometer 6': 'b6',
    })

    # --- select and save only desired columns ---
    masked = masked[['x','y','z','fx','fy','fz','b1','b2','b3','b4','b5','b6']]

    masked.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Rows: {len(masked)} | Columns: {list(masked.columns)}")


if __name__ == "__main__":
    main()