#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barometer-Force Data Synchronization and Processing - UPDATED VERSION

NEW FEATURES:
- Two drift removal methods: EMA and Linear Temperature Compensation
- Fixed zeroing logic (works correctly after warmup removal)
- Comparison plots to evaluate methods
- Real-time compatible algorithms

Steps:
1) Load processing_test CSV (ROS/epoch seconds in 'time', positions, forces)
2) Load barometer CSV/TXT (new format: Epoch_s + b1..b6, t1..t6)
3) Remove initial warm-up period to eliminate sensor startup shift
4) Apply drift removal (choose method in config)
5) Synchronize with merge_asof (nearest, tolerance)
6) Zero barometers at baseline (fixed implementation)
7) Apply spatial mask (keep only rows within bounds)
8) Generate plots and save synchronized CSV

Author: Aurora Ruggeri (updated with Claude's improvements)
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("WARNING: ruptures library not found. Install with: pip install ruptures")
    print("Step removal will be disabled.")

# Import new drift removal functions
from barometer_drift_removal import (
    remove_drift_ema,
    remove_drift_temperature_linear,
    zero_barometers_fixed,
    plot_drift_comparison,
    plot_temperature_vs_pressure,
    remove_steps_robust
)

# =========================== CONFIGURATION ===========================

# Process multiple test numbers (single value or list)
TEST_NUMS = [51036]  
# [51000, 51001, 51002, 51003, 51004, 51005, 51006, 51007, 51008, 51100, 51101, 51102, 51103, 51104, 51105, 51106, 51200, 51201, 51202, 51203, 51204, 51205]
# [51300, 51301, 51302, 51303, 51304, 51400, 51401, 51402, 51403, 51404, 51405, 51500, 51501, 51502, 51503]
# [2, 3, 4, 5]  
VERSION_NUM = 5

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data")

# --- Synchronization params ---
ASOF_DIRECTION = "nearest"
ASOF_TOLERANCE_S = 0.05  # seconds

# --- Remove initial warm-up data ---
REMOVE_INITIAL_WARMUP_DURATION = 0.1  # seconds to remove from the beginning (0 = disabled)

# --- Step detection & leveling (removes hardware jumps before drift removal) ---
ENABLE_STEP_LEVELING = True             # Enable step detection and leveling
STEP_THRESHOLD_HPA = 15.0                 # Jump size (hPa) to consider a step
STEP_WINDOW_SIZE = 10                    # Window size for median calculation

# --- Drift removal method selection ---
DRIFT_REMOVAL_METHOD = "ema"  # Options: "ema", "temperature", "both", "none"

# EMA parameters (for method='ema' or 'both')
EMA_ALPHA = 0.0001  # Smoothing factor (smaller = removes more drift)
                   # Time constant ≈ 1/alpha samples
                   # At 100Hz: 0.001 → ~10s, 0.0001 → ~100s

# Sensor-specific alpha overrides (for sensors with different drift characteristics)
EMA_ALPHA_OVERRIDE = {
    # 'b6': 0.005,  # b6 has faster drift, needs larger alpha
    # 'b3': 0.002,  # Example: uncomment if b3 also needs adjustment
}

# Temperature compensation parameters (for method='temperature' or 'both')
TEMP_SKIP_INITIAL_FRACTION = 0.05  # Fraction of data to skip when fitting

# Zeroing parameters
ZERO_AT_START = True  # Zero the data after drift removal

# --- Dynamic re-zero based on Fz ≈ 0 ---
ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ = True  # Set True to enable
FZ_ZERO_THRESHOLD = 0.2                   # |Fz| [N] considered "zero"
MIN_ZERO_DURATION_S = 0.01                # Minimum duration of zero-force interval
MIN_ZERO_SAMPLES = 5                      # Minimum number of samples in that interval

# --- No-contact masking for training data (NEW) ---
ENABLE_NO_CONTACT_MASKING = True          # Teach model to recognize no-contact states
NO_CONTACT_FZ_THRESHOLD = 0.1             # |fz| < this value triggers masking
NO_CONTACT_SENTINEL = -999.0              # Sentinel value for "no prediction" (converted to NaN in training)

# --- Spatial masking ---
SPATIAL_FILTER = {
    'x_range': (-20, 20),    # mm
    'y_range': (-8, 8),      # mm
    'z_range': (-10.0, 10.0) # mm
}

# --- Path plotting ---
PATH_PLOT_RECT_SIZE_MM = (40.0, 16.0)  # Rectangle size for sensor outline
PATH_PLOT_XLIM = (-25, 25)
PATH_PLOT_YLIM = (-10, 10)
PATH_GAP_FACTOR_TIME = 20.0            # Time gap factor for trajectory segmentation
PATH_BREAK_ON_XY_JUMPS = False         # Enable/disable XY jump detection
PATH_JUMP_FACTOR = 20.0                # XY jump factor for trajectory segmentation

# =============================== HELPERS ===============================

def mask_no_contact_samples(df: pd.DataFrame, fz_threshold: float = 0.1, 
                            sentinel: float = -999.0) -> pd.DataFrame:
    """
    Teach the model to recognize no-contact states by masking targets.
    
    When |fz| < threshold, set x, y, fx, fy to sentinel value (-999).
    This teaches the model to predict 'no contact' based on barometer patterns.
    
    The sentinel value will be converted to NaN in training scripts for proper handling.
    
    Args:
        df: DataFrame with columns 'fz', 'x', 'y', 'fx', 'fy'
        fz_threshold: Threshold for considering force as zero (N)
        sentinel: Sentinel value to use for masked outputs
    
    Returns:
        DataFrame with masked values
    """
    df = df.copy()
    
    # Find no-contact samples
    if 'fz' in df.columns:
        no_contact_mask = df['fz'].abs() < fz_threshold
        n_masked = no_contact_mask.sum()
        
        if n_masked > 0:
            # Set positions and lateral forces to sentinel value
            if 'x' in df.columns:
                df.loc[no_contact_mask, 'x'] = sentinel
            if 'y' in df.columns:
                df.loc[no_contact_mask, 'y'] = sentinel
            if 'fx' in df.columns:
                df.loc[no_contact_mask, 'fx'] = sentinel
            if 'fy' in df.columns:
                df.loc[no_contact_mask, 'fy'] = sentinel
            
            pct_masked = 100 * n_masked / len(df)
            print(f"\nNo-contact masking applied:")
            print(f"  Threshold: |fz| < {fz_threshold} N")
            print(f"  Masked samples: {n_masked}/{len(df)} ({pct_masked:.1f}%)")
            print(f"  Set x, y, fx, fy to {sentinel} (will be NaN in training)")
            print(f"  This teaches the model to recognize no-contact barometer patterns")
    
    return df


def ensure_seconds(series: pd.Series) -> pd.Series:
    """
    Convert time-like numeric column to seconds.

    Handles:
    - ROS/Unix epoch seconds (1e9-ish) WITHOUT rescaling
    - Nanoseconds (> 1e14) -> scale by 1e-9
    - Microseconds (> 1e11) -> scale by 1e-6
    - Milliseconds (1e5 to 1e9) -> scale by 1e-3
    """
    s = pd.to_numeric(series, errors='coerce')
    med = s.dropna().abs().median()

    if med > 1e14:
        s *= 1e-9  # ns -> s
    elif med > 1e11:
        s *= 1e-6  # µs -> s
    elif med > 1e5 and med < 1e9:
        s *= 1e-3  # ms -> s
    # else: already in seconds (includes Unix epoch ~1.7e9, ROS time ~1e9)

    return s


def _force_name(df, base):
    """Helper to find force/torque column name (case-insensitive)."""
    return base if base in df.columns else base.lower() if base.lower() in df.columns else None


# =============================== LOADERS ===============================

def load_processing_data(path: Path) -> pd.DataFrame:
    """Load processing CSV with positions and forces/torques."""
    df = pd.read_csv(path)

    # Find time column
    time_col = None
    for cand in ['time', 'ros_time', 'ros_time_s', 'Epoch_s', 'time_ros', 't']:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None and 'time_ns' in df.columns:
        time_col = 'time_ns'
    if time_col is None:
        raise ValueError("No suitable time column found in processing CSV "
                         "(expected one of: time, ros_time, Epoch_s, t, time_ns).")

    df['time'] = ensure_seconds(df[time_col])

    # Check for position columns
    required_pos = ['x_position_mm', 'y_position_mm', 'z_position_mm']
    missing = [c for c in required_pos if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in processing CSV: {missing}")

    # Find force/torque columns (case-insensitive)
    force_names = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    present_forces = []
    for c in force_names:
        if c in df.columns:
            present_forces.append(c)
        elif c.lower() in df.columns:
            present_forces.append(c.lower())
    if not present_forces:
        raise ValueError("No force/torque columns found (Fx/Fy/Fz/Tx/Ty/Tz).")

    keep = ['time', 'x_position_mm', 'y_position_mm', 'z_position_mm'] + present_forces
    return df[keep].sort_values('time', ignore_index=True)


def load_barometer_data(path: Path) -> pd.DataFrame:
    """
    Load barometer file in multiple formats:
    - NEW: PcTime, Epoch_s, Time_ms, b1_P..b6_P, b1_T..b6_T (temperature columns optional)
    - OLD: 'Epoch_s' + 'b1..b6' directly
    - LEGACY: 'Epoch_s' + 'barometer 1..6'
    - VERY OLD: 'Timestamp' + datalog_YYYY-MM-DD_*.csv format
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Case 1: New ROS/epoch format with b1_P..b6_P
    if 'Epoch_s' in df.columns and 'b1_P' in df.columns:
        t = ensure_seconds(df['Epoch_s'])
        out = pd.DataFrame({'time': t})

        for i in range(1, 7):
            pcol = f"b{i}_P"
            if pcol not in df.columns:
                raise ValueError(f"Missing expected barometer column '{pcol}' in {path}")
            out[f"b{i}"] = pd.to_numeric(df[pcol], errors='coerce')

            # Temperature column (optional but used if present)
            tcol = f"b{i}_T"
            if tcol in df.columns:
                out[f"t{i}"] = pd.to_numeric(df[tcol], errors='coerce')

    # Case 2: Old format with b1..b6 directly
    elif 'Epoch_s' in df.columns and 'b1' in df.columns:
        t = ensure_seconds(df['Epoch_s'])
        out = pd.DataFrame({'time': t})

        for i in range(1, 7):
            col = f"b{i}"
            if col not in df.columns:
                raise ValueError(f"Missing expected barometer column '{col}' in {path}")
            out[col] = pd.to_numeric(df[col], errors='coerce')

    # Case 3: Legacy format with 'barometer 1..6' + Epoch_s
    elif 'Epoch_s' in df.columns:
        t = ensure_seconds(pd.to_numeric(df['Epoch_s'], errors='coerce'))
        baro_cols = [f'barometer {i}' for i in range(1, 7) if f'barometer {i}' in df.columns]
        if len(baro_cols) != 6:
            raise ValueError(f"Expected 6 legacy barometer columns, found {len(baro_cols)}: {baro_cols}")
        out = pd.DataFrame({'time': t})
        for i, c in enumerate(baro_cols, start=1):
            out[f'b{i}'] = pd.to_numeric(df[c], errors='coerce')

    # Case 4: Very old 'Timestamp' + datalog_YYYY-MM-DD_*.csv
    else:
        ts_col = next((c for c in df.columns if 'timestamp' in c.lower()), None)
        if ts_col is None:
            raise ValueError("Barometer file needs either 'Epoch_s' or a 'Timestamp' column.")
        m = re.search(r'datalog_(\d{4}-\d{2}-\d{2})', os.path.basename(str(path)))
        if not m:
            raise ValueError("Filename must look like 'datalog_YYYY-MM-DD_...csv' for legacy mode.")
        date_str = m.group(1)
        dt_strings = date_str + ' ' + df[ts_col].astype(str)
        dt = pd.to_datetime(dt_strings, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce', utc=True)
        t = dt.view('int64') / 1e9  # seconds
        baro_cols = [f'barometer {i}' for i in range(1, 7) if f'barometer {i}' in df.columns]
        if len(baro_cols) != 6:
            raise ValueError(f"Expected 6 legacy barometer columns, found {len(baro_cols)}: {baro_cols}")
        out = pd.DataFrame({'time': t})
        for i, c in enumerate(baro_cols, start=1):
            out[f'b{i}'] = pd.to_numeric(df[c], errors='coerce')

    return out.dropna(subset=['time']).sort_values('time', ignore_index=True)


# =========================== CORE OPERATIONS ===========================

def synchronize_data(proc_df: pd.DataFrame, baro_df: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbor time merge with tolerance."""
    merged = pd.merge_asof(
        proc_df.sort_values('time'),
        baro_df.sort_values('time'),
        on='time',
        direction=ASOF_DIRECTION,
        tolerance=ASOF_TOLERANCE_S
    )

    # Keep only rows where barometers matched
    baro_cols_to_check = []
    for i in range(1, 7):
        col = f'b{i}'
        if col in merged.columns and merged[col].notna().any():
            baro_cols_to_check.append(col)

    if baro_cols_to_check:
        print(f"Checking for NaN in barometers: {baro_cols_to_check}")
        merged = merged.dropna(subset=baro_cols_to_check)
    else:
        print("WARNING: No barometer data found after merge!")

    return merged


def apply_spatial_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows within specified spatial bounds."""
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


def rezero_barometers_when_fz_zero(df: pd.DataFrame, fz_threshold: float = 0.05, min_zero_duration: float = 0.2, min_samples: int = 20) -> pd.DataFrame:
    """
    Adaptive drift correction: re-zero barometers when Fz ≈ 0.

    Finds time intervals where |Fz| < fz_threshold and re-zeros barometers
    at those points to counter slow drift.
    """
    df = df.copy()

    if "time" not in df.columns:
        raise ValueError("rezero_barometers_when_fz_zero: dataframe must contain 'time' column.")

    # Find Fz column name (handles 'Fz' or 'fz')
    fz_col = _force_name(df, "Fz")
    if not fz_col:
        print("rezero_barometers_when_fz_zero: no Fz/fz column found, skipping.")
        return df

    t = df["time"].to_numpy()
    fz = pd.to_numeric(df[fz_col], errors="coerce").to_numpy()

    # Barometer columns
    baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    if not baro_cols:
        print("rezero_barometers_when_fz_zero: no barometer columns found, skipping.")
        return df

    # Mask of "near-zero force"
    mask_low = np.isfinite(fz) & (np.abs(fz) <= fz_threshold)

    # Find contiguous segments of mask_low == True
    segments = []
    in_seg = False
    start = None
    n = len(mask_low)

    for i, is_low in enumerate(mask_low):
        if is_low and not in_seg:
            in_seg = True
            start = i
        elif not is_low and in_seg:
            # segment ended at i-1
            end = i - 1
            segments.append((start, end))
            in_seg = False

    # If we ended inside a segment, close it at the end
    if in_seg:
        segments.append((start, n - 1))

    if not segments:
        print("\nDynamic re-zeroing: no zero-force segments found.")
        return df

    print(f"\nDynamic re-zeroing (Fz_threshold={fz_threshold}N, min_duration={min_zero_duration}s):")
    idx_all = np.arange(n)
    n_rezerod = 0

    for (s, e) in segments:
        seg_len = e - s + 1
        if seg_len < min_samples:
            continue
        if t[e] - t[s] < min_zero_duration:
            continue

        n_rezerod += 1
        seg_slice = slice(s, e + 1)

        print(f"  Zero-force segment #{n_rezerod}: idx {s}-{e}, t={t[s]:.3f}-{t[e]:.3f}s, n={seg_len}")

        for c in baro_cols:
            baseline = df[c].iloc[seg_slice].mean()
            # subtract baseline from this point onward
            df.loc[idx_all >= s, c] = df.loc[idx_all >= s, c] - baseline

    print(f"  Total re-zero events: {n_rezerod}")

    return df


# ============================== PLOTTING ===============================

def plot_barometers_subplots(df, save_path=None, suptitle="Barometers"):
    """Plot all 6 barometers in subplots."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for i in range(6):
        col = f"b{i+1}"
        if col not in df.columns:
            continue
        axes[i].plot(df["time"], df[col], linewidth=1.2)
        axes[i].set_ylabel(f"{col} [hPa]")
        axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel("Experiment Time [s]")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    return fig


def plot_forces_torques_subplots(df, save_path=None, suptitle="ATI Force/Torque"):
    """Plot all 6 force/torque channels in subplots."""
    cols = []
    for base in ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]:
        c = _force_name(df, base)
        if c:
            cols.append((c, f"{base}"))
    if not cols:
        print("No ATI columns found.")
        return None

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for ax, (c, label) in zip(axes, cols):
        ax.plot(df["time"], df[c], linewidth=1.2)
        unit = "N" if label.startswith("F") else "N·m"
        ax.set_ylabel(f"{label} [{unit}]")
        ax.grid(alpha=0.3)
    axes[min(len(cols), 6)-1].set_xlabel("Experiment Time [s]")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    return fig


def plot_barometers_vs_force(df, force_col="Fz", save_path=None, suptitle=None):
    """Plot barometers alongside a force/torque channel (dual y-axis)."""
    fcol = _force_name(df, force_col)
    if not fcol:
        print(f"No force column matching {force_col} found.")
        return None

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    for i in range(6):
        ax1 = axes[i]
        bcol = f"b{i+1}"
        if bcol not in df.columns:
            continue
        # left: barometer
        line1 = ax1.plot(df["time"], df[bcol], label=bcol, linewidth=1.2)
        ax1.set_ylabel(f"{bcol} [hPa]")
        ax1.grid(alpha=0.3)
        # right: force (scaled via twin axis)
        ax2 = ax1.twinx()
        line2 = ax2.plot(df["time"], df[fcol], linestyle="--", alpha=0.8,
                         label=fcol, linewidth=1.2, color="C3")
        unit = "N" if fcol.lower().startswith("f") else "N·m"
        ax2.set_ylabel(f"{fcol} [{unit}]")
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Experiment Time [s]")
    fig.suptitle(suptitle or f"Barometers vs {force_col}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    return fig


def plot_barometers_vs_fz(df, save_path=None):
    """Plot barometers vs Fz."""
    return plot_barometers_vs_force(df, "Fz", save_path, suptitle="Barometers vs Fz")


def plot_barometers_vs_fx(df, save_path=None):
    """Plot barometers vs Fx."""
    return plot_barometers_vs_force(df, "Fx", save_path, suptitle="Barometers vs Fx")


def plot_barometers_vs_fy(df, save_path=None):
    """Plot barometers vs Fy."""
    return plot_barometers_vs_force(df, "Fy", save_path, suptitle="Barometers vs Fy")


def plot_path_xy_filtered(out_dir: str,
                          X_mm: np.ndarray, Y_mm: np.ndarray, t_s: np.ndarray,
                          test_num: int,
                          break_on_time_gaps: bool = True, gap_factor_time: float = 5.0,
                          break_on_xy_jumps: bool = True, jump_factor: float = 5.0,
                          rect_size_mm=(40.0, 16.0),
                          xlim=(-25, 25), ylim=(-10, 10)):
    """
    Plot XY path with optional trajectory segmentation.

    Parameters:
    -----------
    break_on_time_gaps : bool
        Insert breaks in trajectory when time gaps are detected
    gap_factor_time : float
        Multiplier for median time interval to detect gaps
    break_on_xy_jumps : bool
        Insert breaks when large XY jumps are detected
    jump_factor : float
        Multiplier for median step size to detect jumps
    rect_size_mm : tuple
        Size of sensor outline rectangle (width, height)
    """
    plt.figure(figsize=(8, 6))

    n = len(X_mm)
    breaks = []

    # Detect time gaps
    if break_on_time_gaps and n > 1:
        dt = np.diff(t_s)
        med_dt = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.0
        if med_dt > 0:
            b = np.where(dt > gap_factor_time * med_dt)[0]
            breaks.extend(b.tolist())

    # Detect XY jumps
    if break_on_xy_jumps and n > 1:
        steps = np.hypot(np.diff(X_mm), np.diff(Y_mm))
        med_step = np.median(steps[steps > 0]) if np.any(steps > 0) else 0.0
        if med_step > 0:
            b = np.where(steps > jump_factor * med_step)[0]
            breaks.extend(b.tolist())

    breaks = sorted(set(breaks))

    # Insert NaNs at breaks to segment trajectory
    if breaks:
        xs, ys = [X_mm[0]], [Y_mm[0]]
        for i in range(n - 1):
            xs.append(X_mm[i+1])
            ys.append(Y_mm[i+1])
            if i in breaks:
                xs.append(np.nan)
                ys.append(np.nan)
        x_plot = np.array(xs)
        y_plot = np.array(ys)
    else:
        x_plot, y_plot = X_mm, Y_mm

    plt.plot(x_plot, y_plot, linewidth=1.2)

    # Add sensor outline rectangle
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
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("Top X [mm]")
    plt.ylabel("Top Y [mm]")
    plt.title(f"Tip path on top surface — trial {test_num}")

    out_path = os.path.join(out_dir, f"tip_path_top_xy_trial{test_num}.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


# ================================ MAIN =================================

def process_single_test(TEST_NUM):
    """Process a single test number."""
    DATA_DIR = BASE_DIR / f"test {TEST_NUM} - sensor v{VERSION_NUM}"
    
    # Input files
    PROCESSING_FILE = DATA_DIR / f"processing_test_{TEST_NUM}.csv"
    BAROMETER_FILE = DATA_DIR / f"{TEST_NUM}barometers_trial.txt"
    
    # Output
    OUTPUT_FILE = DATA_DIR / f"synchronized_events_{TEST_NUM}.csv"
    save_plots_dir = DATA_DIR / "plots synchronized"
    save_plots_dir.mkdir(exist_ok=True, parents=True)
    
    print("#" * 60)
    print(f"# BAROMETER-FORCE SYNC | Test {TEST_NUM} v{VERSION_NUM}")
    print("#" * 60)
    print(f"Configuration:")
    print(f"  - Sync: direction={ASOF_DIRECTION}, tolerance={ASOF_TOLERANCE_S}s")
    print(f"  - Warmup removal: {REMOVE_INITIAL_WARMUP_DURATION}s")
    print(f"  - Drift removal method: {DRIFT_REMOVAL_METHOD}")
    if DRIFT_REMOVAL_METHOD in ['ema', 'both']:
        print(f"    - EMA alpha: {EMA_ALPHA}")
    if DRIFT_REMOVAL_METHOD in ['temperature', 'both']:
        print(f"    - Temperature skip fraction: {TEMP_SKIP_INITIAL_FRACTION}")
    print(f"  - Zero at start: {ZERO_AT_START}")
    print(f"  - Dynamic re-zeroing: {ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ}")
    print()

    # Check input files exist
    if not PROCESSING_FILE.exists():
        raise FileNotFoundError(f"Processing file not found: {PROCESSING_FILE}")
    if not BAROMETER_FILE.exists():
        raise FileNotFoundError(f"Barometer file not found: {BAROMETER_FILE}")

    print(f"Input files:")
    print(f"  - Processing: {PROCESSING_FILE}")
    print(f"  - Barometer:  {BAROMETER_FILE}")
    print()

    # Load data
    print("Loading data...")
    proc_df = load_processing_data(PROCESSING_FILE)
    baro_df = load_barometer_data(BAROMETER_FILE)
    print(f"  - Processing data: {len(proc_df)} rows")
    print(f"  - Barometer data:  {len(baro_df)} rows")

    # Keep original for comparison
    baro_df_original = baro_df.copy()

    # Plot raw barometers (PLOT 1)
    plot_barometers_subplots(baro_df,  
                            save_path=save_plots_dir / "1_barometers_raw.png",  
                            suptitle="Barometers (raw data)")

    # STEP 1: Remove initial warm-up data
    if REMOVE_INITIAL_WARMUP_DURATION > 0:
        t0 = baro_df["time"].min()
        initial_len = len(baro_df)
        baro_df = baro_df[baro_df["time"] > t0 + REMOVE_INITIAL_WARMUP_DURATION].copy()
        removed_rows = initial_len - len(baro_df)
        print(f"\nSTEP 1: Removed {removed_rows} rows from initial {REMOVE_INITIAL_WARMUP_DURATION}s warm-up period")
        baro_df = baro_df.reset_index(drop=True)

        plot_barometers_subplots(baro_df,
                                save_path=save_plots_dir / "2 _barometers_after_warmup_removal.png",
                                suptitle="Barometers (after warmup removal)")

    # STEP 1.5: Remove hardware steps/jumps (before drift removal to prevent EMA "tails")
    if ENABLE_STEP_LEVELING:
        print(f"\nSTEP 1.5: Applying robust step leveling...")
        baro_df = remove_steps_robust(baro_df, threshold=STEP_THRESHOLD_HPA, window_size=STEP_WINDOW_SIZE)
        plot_barometers_subplots(baro_df,
                                save_path=save_plots_dir / "2.5_barometers_after_step_leveling.png",
                                suptitle="Barometers (after step leveling)")

    # STEP 2: Apply drift removal
    ema_trends = {}
    temp_coeffs = {}
    
    if DRIFT_REMOVAL_METHOD == 'ema':
        print("\nSTEP 2: Applying EMA drift removal...")
        baro_df, ema_trends = remove_drift_ema(
            baro_df, 
            alpha=EMA_ALPHA,
            alpha_override=EMA_ALPHA_OVERRIDE,
            zero_at_start=ZERO_AT_START
        )
        plot_barometers_subplots(baro_df,
                                save_path=save_plots_dir / "3_barometers_after_ema.png",
                                suptitle="Barometers (after EMA drift removal)")
    
    elif DRIFT_REMOVAL_METHOD == 'temperature':
        print("\nSTEP 2: Applying temperature-based drift removal...")
        baro_df, temp_coeffs = remove_drift_temperature_linear(
            baro_df,
            skip_initial_fraction=TEMP_SKIP_INITIAL_FRACTION,
            zero_at_start=ZERO_AT_START
        )
        plot_barometers_subplots(baro_df,
                                save_path=save_plots_dir / "3_barometers_after_temp_correction.png",
                                suptitle="Barometers (after temperature correction)")
        
        # Plot temperature correlation
        plot_temperature_vs_pressure(baro_df, temp_coeffs=temp_coeffs,
                                    save_path=save_plots_dir / "temperature_vs_pressure_correlation.png")
    
    elif DRIFT_REMOVAL_METHOD == 'both':
        print("\nSTEP 2a: Applying temperature-based drift removal...")
        baro_df_temp, temp_coeffs = remove_drift_temperature_linear(
            baro_df.copy(),
            skip_initial_fraction=TEMP_SKIP_INITIAL_FRACTION,
            zero_at_start=ZERO_AT_START
        )
        
        print("\nSTEP 2b: Applying EMA drift removal...")
        baro_df_ema, ema_trends = remove_drift_ema(
            baro_df.copy(),
            alpha=EMA_ALPHA,
            alpha_override=EMA_ALPHA_OVERRIDE,
            zero_at_start=ZERO_AT_START
        )
        
        # Create comparison plot
        print("\nGenerating comparison plot...")
        plot_drift_comparison(
            baro_df, baro_df_ema, baro_df_temp,
            ema_trends=ema_trends,
            temp_coeffs=temp_coeffs,
            save_path=save_plots_dir / "3_drift_removal_comparison.png"
        )
        
        # Plot temperature correlation
        plot_temperature_vs_pressure(baro_df, temp_coeffs=temp_coeffs,
                                    save_path=save_plots_dir / "temperature_vs_pressure_correlation.png")
        
        # Choose which method to use for final output
        # You can change this to baro_df_temp if temperature correction works better
        print("\nUsing EMA method for final output (you can change this in code)")
        baro_df = baro_df_ema
    
    elif DRIFT_REMOVAL_METHOD == 'none':
        print("\nSTEP 2: Skipping drift removal (method='none')")
        # Still zero if requested
        if ZERO_AT_START:
            baro_df = zero_barometers_fixed(baro_df, reference_index='first_valid')
    
    else:
        raise ValueError(f"Unknown drift removal method: {DRIFT_REMOVAL_METHOD}")

    # Synchronize
    print("\nSynchronizing data...")
    merged = synchronize_data(proc_df, baro_df)
    print(f"  - Synchronized: {len(merged)} rows")

    # STEP 3: Dynamic re-zeroing when Fz ≈ 0 (optional)
    if ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ:
        merged = rezero_barometers_when_fz_zero(
            merged,
            fz_threshold=FZ_ZERO_THRESHOLD,
            min_zero_duration=MIN_ZERO_DURATION_S,
            min_samples=MIN_ZERO_SAMPLES
        )

    # Plot final processed barometers
    print("\nGenerating final plots...")
    plot_barometers_subplots(merged, 
                            save_path=save_plots_dir / "4_barometers_final.png", 
                            suptitle="Barometers (final processed)")

    plot_forces_torques_subplots(merged, save_path=save_plots_dir / "ati_forces_torques.png")
    plot_barometers_vs_fz(merged, save_path=save_plots_dir / "barometers_vs_Fz.png")
    plot_barometers_vs_fx(merged, save_path=save_plots_dir / "barometers_vs_Fx.png")
    plot_barometers_vs_fy(merged, save_path=save_plots_dir / "barometers_vs_Fy.png")
    plt.close('all')

    # Apply spatial filter
    print("\nApplying spatial filter...")
    masked = apply_spatial_filter(merged)
    print(f"  - After spatial filter: {len(masked)} rows")

    # Plot trajectory
    plot_path_xy_filtered(
        out_dir=str(save_plots_dir),
        X_mm=masked["x_position_mm"].to_numpy(),
        Y_mm=masked["y_position_mm"].to_numpy(),
        t_s=masked["time"].to_numpy(),
        test_num=TEST_NUM,
        break_on_time_gaps=True,
        gap_factor_time=PATH_GAP_FACTOR_TIME,
        break_on_xy_jumps=PATH_BREAK_ON_XY_JUMPS,
        jump_factor=PATH_JUMP_FACTOR,
        rect_size_mm=PATH_PLOT_RECT_SIZE_MM,
        xlim=PATH_PLOT_XLIM,
        ylim=PATH_PLOT_YLIM
    )

    if masked.empty:
        print("WARNING: No rows after spatial filter — saving unfiltered synchronized data instead.")
        masked = merged

    # Rename columns for export
    masked = masked.rename(columns={
        'time': 't',
        'x_position_mm': 'x', 'y_position_mm': 'y', 'z_position_mm': 'z',
        'Fx': 'fx', 'Fy': 'fy', 'Fz': 'fz', 'fx': 'fx', 'fy': 'fy', 'fz': 'fz',
        'Tx': 'tx', 'Ty': 'ty', 'Tz': 'tz', 'tx': 'tx', 'ty': 'ty', 'tz': 'tz',
        'barometer 1': 'b1', 'barometer 2': 'b2', 'barometer 3': 'b3',
        'barometer 4': 'b4', 'barometer 5': 'b5', 'barometer 6': 'b6',
    })

    # Ensure consistent column order
    desired_cols = ['t', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'x', 'y', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']
    masked = masked.reindex(columns=desired_cols)

    # STEP 4: Apply no-contact masking to teach model (optional)
    if ENABLE_NO_CONTACT_MASKING:
        print("\nApplying no-contact masking to training data...")
        masked = mask_no_contact_samples(
            masked, 
            fz_threshold=NO_CONTACT_FZ_THRESHOLD,
            sentinel=NO_CONTACT_SENTINEL
        )

    # Save output
    masked.to_csv(OUTPUT_FILE, index=False)
    print(f"\nOutput saved: {OUTPUT_FILE}")
    print(f"  - Rows: {len(masked)}")
    print(f"  - Columns: {list(masked.columns)}")
    print("\nProcessing complete!")
    print("\nBarometer processing pipeline:")
    print("  1. Remove initial samples (warmup removal)")
    if ENABLE_STEP_LEVELING:
        print(f"  2. Step detection & leveling (threshold: {STEP_THRESHOLD_HPA} hPa)")
    print(f"  3. Apply drift removal (method: {DRIFT_REMOVAL_METHOD})")
    if DRIFT_REMOVAL_METHOD in ['ema', 'both']:
        print(f"     - EMA alpha={EMA_ALPHA}, time constant ≈ {1/EMA_ALPHA:.0f} samples")
    if DRIFT_REMOVAL_METHOD in ['temperature', 'both']:
        print(f"     - Linear temperature compensation")
    if ZERO_AT_START:
        print("  4. Zero at start")
    if ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ:
        print(f"  5. Dynamic re-zeroing when Fz ≈ 0")
    if ENABLE_NO_CONTACT_MASKING:
        print(f"  6. No-contact masking (|fz| < {NO_CONTACT_FZ_THRESHOLD} → x,y,fx,fy = {NO_CONTACT_SENTINEL})")
    
    print(f"\nTest {TEST_NUM} processing complete!")
    print("=" * 60 + "\n")


def main():
    """Main entry point - process all test numbers."""
    # Ensure TEST_NUMS is a list
    test_list = TEST_NUMS if isinstance(TEST_NUMS, list) else [TEST_NUMS]
    
    print("\n" + "=" * 60)
    print(f"STARTING BATCH PROCESSING: {len(test_list)} test(s)")
    print(f"Test numbers: {test_list}")
    print("=" * 60 + "\n")
    
    for idx, test_num in enumerate(test_list, 1):
        print(f"\n{'#' * 60}")
        print(f"# PROCESSING TEST {idx}/{len(test_list)}: {test_num}")
        print(f"{'#' * 60}\n")
        
        try:
            process_single_test(test_num)
        except Exception as e:
            print(f"\n{'!' * 60}")
            print(f"ERROR processing test {test_num}: {e}")
            print(f"{'!' * 60}")
            print("Continuing with next test...\n")
            continue
    
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print(f"Processed {len(test_list)} test(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()