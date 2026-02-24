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
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))

# Import barometer processing pipeline
from utils.barometer_processing import (
    ensure_seconds,
    load_barometer_data,
    rezero_barometers_when_fz_zero,
    process_barometers,
    plot_barometers_subplots,
    plot_barometers_vs_fz,
    plot_barometers_vs_fx,
    plot_barometers_vs_fy,
)

# =========================== CONFIGURATION ===========================

# Process multiple test numbers (single value or list)
TEST_NUMS = [51092]  # Example: [51094, 51095, 51096]
# [51000, 51001, 51002, 51003, 51004, 51005, 51006, 51007, 51008, 51100, 51101, 51102, 51103, 51104, 51105, 51106, 51200, 51201, 51202, 51203, 51204, 51205]
# [51300, 51301, 51302, 51303, 51304, 51400, 51401, 51402, 51403, 51404, 51405, 51500, 51501, 51502, 51503]
# [51030, 51031, 51032, 51033, 51034, 51035, 51036, 51130, 51131]
# [2, 3, 4, 5]
VERSION_NUM = 5

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data")

# --- Synchronization params ---
ASOF_DIRECTION = "nearest"
ASOF_TOLERANCE_S = 0.05  # seconds

# --- Remove initial warm-up data ---
REMOVE_INITIAL_WARMUP_DURATION = 1  # seconds to remove from the beginning (0 = disabled)

# --- Step detection & leveling (removes hardware jumps before drift removal) ---
ENABLE_STEP_LEVELING = True
STEP_THRESHOLD_HPA = 5.0   # Catches even small deviations
STEP_WINDOW_SIZE = 200     # ~2 seconds at 100Hz

# Outliers removal parameters
ENABLE_OUTLIER_REMOVAL = True  # Enable/disable outlier removal
threshold_multiplier = 30.0  # Number of standard deviations for outlier detection

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
ENABLE_NO_CONTACT_MASKING = False          # Teach model to recognize no-contact states
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

            # Masking applied silently

    return df


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
        merged = merged.dropna(subset=baro_cols_to_check)

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


# ============================== PLOTTING ===============================

def plot_forces_torques_subplots(df, save_path=None, suptitle="ATI Force/Torque"):
    """Plot all 6 force/torque channels in subplots."""
    # Define colors: forces in green, torques in yellow
    force_color = '#44b155'
    torque_color = '#d6c52e'

    cols = []
    for base in ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]:
        c = _force_name(df, base)
        if c:
            cols.append((c, f"{base}"))
    if not cols:
        return None

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for idx, (ax, (c, label)) in enumerate(zip(axes, cols)):
        # Use green for forces (Fx, Fy, Fz), yellow for torques (Tx, Ty, Tz)
        color = force_color if label.startswith("F") else torque_color
        ax.plot(df["time"], df[c], linewidth=1.2, color=color)
        unit = "N" if label.startswith("F") else "N·m"
        ax.set_ylabel(f"{label} [{unit}]")
        ax.grid(alpha=0.3)
    axes[min(len(cols), 6)-1].set_xlabel("Experiment Time [s]")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


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

    plt.plot(x_plot, y_plot, linewidth=1.2, color='#44b155')

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

    print(f"Processing Test {TEST_NUM}...")

    # Check input files exist
    if not PROCESSING_FILE.exists():
        raise FileNotFoundError(f"Processing file not found: {PROCESSING_FILE}")
    if not BAROMETER_FILE.exists():
        raise FileNotFoundError(f"Barometer file not found: {BAROMETER_FILE}")

    # Load data
    proc_df = load_processing_data(PROCESSING_FILE)
    baro_df = load_barometer_data(BAROMETER_FILE)

    # Process barometers (warmup removal, step leveling, outlier removal, drift removal)
    baro_df = process_barometers(
        baro_df,
        save_plots_dir=save_plots_dir,
        warmup_duration=REMOVE_INITIAL_WARMUP_DURATION,
        enable_step_leveling=ENABLE_STEP_LEVELING,
        step_threshold_hpa=STEP_THRESHOLD_HPA,
        step_window_size=STEP_WINDOW_SIZE,
        enable_outlier_removal=ENABLE_OUTLIER_REMOVAL,
        outlier_threshold_multiplier=threshold_multiplier,
        drift_removal_method=DRIFT_REMOVAL_METHOD,
        ema_alpha=EMA_ALPHA,
        ema_alpha_override=EMA_ALPHA_OVERRIDE,
        temp_skip_initial_fraction=TEMP_SKIP_INITIAL_FRACTION,
        zero_at_start=ZERO_AT_START,
    )

    # Synchronize
    merged = synchronize_data(proc_df, baro_df)

    # Dynamic re-zeroing when Fz ≈ 0 (optional)
    if ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ:
        merged = rezero_barometers_when_fz_zero(
            merged,
            fz_threshold=FZ_ZERO_THRESHOLD,
            min_zero_duration=MIN_ZERO_DURATION_S,
            min_samples=MIN_ZERO_SAMPLES
        )

    # Plot final processed barometers
    plot_barometers_subplots(merged, save_path=save_plots_dir / "4_barometers_final.png", suptitle="Barometers (final processed)")

    plot_forces_torques_subplots(merged, save_path=save_plots_dir / "ati_forces_torques.png")
    plot_barometers_vs_fz(merged, save_path=save_plots_dir / "barometers_vs_Fz.png")
    plot_barometers_vs_fx(merged, save_path=save_plots_dir / "barometers_vs_Fx.png")
    plot_barometers_vs_fy(merged, save_path=save_plots_dir / "barometers_vs_Fy.png")
    plt.close('all')

    # Apply spatial filter
    masked = apply_spatial_filter(merged)

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
        masked = mask_no_contact_samples(
            masked,
            fz_threshold=NO_CONTACT_FZ_THRESHOLD,
            sentinel=NO_CONTACT_SENTINEL
        )

    # Save output
    masked.to_csv(OUTPUT_FILE, index=False)


def main():
    """Main entry point - process all test numbers."""
    # Ensure TEST_NUMS is a list
    test_list = TEST_NUMS if isinstance(TEST_NUMS, list) else [TEST_NUMS]

    for idx, test_num in enumerate(test_list, 1):
        try:
            process_single_test(test_num)
        except Exception as e:
            continue


if __name__ == "__main__":
    main()