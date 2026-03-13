#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barometer-Force Data Synchronization with Temperature - v5.20

Temperature-aware version of data_organization.
New feature: t1..t6 temperature columns (from barometer b{i}_T channels) are
preserved in the synchronized output CSV alongside the standard pressure
readings b1..b6.

Key differences from the baseline data_organization script:
- 'desired_cols' includes t1..t6 after b6
- Missing temperature packets are filled with NaN (not dropped)
- Output version tagged as sensor v5.20

Pipeline steps:
1) Load processing_test CSV (positions + forces)
2) Load barometer CSV/TXT (Epoch_s + b1..b6 + t1..t6 optional)
3) Remove initial warm-up, apply drift removal (EMA default)
4) Synchronize via merge_asof (nearest, tolerance)
5) Dynamic re-zeroing when Fz ~ 0 (optional)
6) Apply spatial mask
7) Save synchronized CSV with temperature columns

Author: Aurora Ruggeri
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Path setup: import shared utilities from source directory ──────────────
_THIS_DIR = Path(__file__).parent
_SRC_DIR = _THIS_DIR.parent / "Barometers_Based_Tactile_Sensor"
sys.path.insert(0, str(_SRC_DIR))

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
TEST_NUMS = [52000, 52001, 52002, 52003, 52004, 52005, 52006]
VERSION_NUM = 5

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data")

# --- Synchronization params ---
ASOF_DIRECTION = "nearest"
ASOF_TOLERANCE_S = 0.1  # seconds

# --- Remove initial warm-up data ---
REMOVE_INITIAL_WARMUP_DURATION = 0.2  # seconds to remove from the beginning (0 = disabled)

# --- Step detection & leveling ---
ENABLE_STEP_LEVELING = False
STEP_THRESHOLD_HPA = 5.0
STEP_WINDOW_SIZE = 200

# Outlier removal parameters
ENABLE_OUTLIER_REMOVAL = False
threshold_multiplier = 30.0

# --- Drift removal method selection ---
DRIFT_REMOVAL_METHOD = "ema"  # Options: "ema", "temperature", "both", "none"

# EMA parameters
EMA_ALPHA = 0.0001
EMA_ALPHA_OVERRIDE = {}

# Temperature compensation parameters
TEMP_SKIP_INITIAL_FRACTION = 0.05

# Zeroing parameters
ZERO_AT_START = True

# --- Dynamic re-zero based on Fz ≈ 0 ---
ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ = True
FZ_ZERO_THRESHOLD = 0.2
MIN_ZERO_DURATION_S = 0.01
MIN_ZERO_SAMPLES = 5

# --- No-contact masking ---
ENABLE_NO_CONTACT_MASKING = False
NO_CONTACT_FZ_THRESHOLD = 0.1
NO_CONTACT_SENTINEL = -999.0

# --- Spatial masking ---
SPATIAL_FILTER = {
    'x_range': (-20, 20),    # mm
    'y_range': (-8, 8),      # mm
    'z_range': (-10.0, 10.0) # mm
}

# --- Path plotting ---
PATH_PLOT_RECT_SIZE_MM = (40.0, 16.0)
PATH_PLOT_XLIM = (-25, 25)
PATH_PLOT_YLIM = (-10, 10)
PATH_GAP_FACTOR_TIME = 20.0
PATH_BREAK_ON_XY_JUMPS = False
PATH_JUMP_FACTOR = 20.0

# Temperature columns expected in barometer data
TEMP_COLS = [f"t{i}" for i in range(1, 7)]  # t1..t6

# =============================== HELPERS ===============================

def mask_no_contact_samples(df: pd.DataFrame, fz_threshold: float = 0.1,
                            sentinel: float = -999.0) -> pd.DataFrame:
    """Mask x, y, fx, fy to sentinel when |fz| < threshold."""
    df = df.copy()
    if 'fz' in df.columns:
        no_contact_mask = df['fz'].abs() < fz_threshold
        if no_contact_mask.sum() > 0:
            for col in ['x', 'y', 'fx', 'fy']:
                if col in df.columns:
                    df.loc[no_contact_mask, col] = sentinel
    return df


def _force_name(df, base):
    return base if base in df.columns else base.lower() if base.lower() in df.columns else None


# =============================== LOADERS ===============================

def load_processing_data(path: Path) -> pd.DataFrame:
    """Load processing CSV with positions and forces/torques."""
    df = pd.read_csv(path)

    time_col = None
    for cand in ['time', 'ros_time', 'ros_time_s', 'Epoch_s', 'time_ros', 't']:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None and 'time_ns' in df.columns:
        time_col = 'time_ns'
    if time_col is None:
        raise ValueError("No suitable time column found in processing CSV.")

    df['time'] = ensure_seconds(df[time_col])

    required_pos = ['x_position_mm', 'y_position_mm', 'z_position_mm']
    missing = [c for c in required_pos if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in processing CSV: {missing}")

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
    """Nearest-neighbor time merge with tolerance. Preserves temperature columns."""
    merged = pd.merge_asof(
        proc_df.sort_values('time'),
        baro_df.sort_values('time'),
        on='time',
        direction=ASOF_DIRECTION,
        tolerance=ASOF_TOLERANCE_S
    )

    # Drop rows where barometer pressure columns didn't match
    baro_cols_to_check = [f'b{i}' for i in range(1, 7) if f'b{i}' in merged.columns
                          and merged[f'b{i}'].notna().any()]
    if baro_cols_to_check:
        merged = merged.dropna(subset=baro_cols_to_check)

    # Temperature columns are optional: missing packets become NaN (not dropped)
    temp_present = [c for c in TEMP_COLS if c in merged.columns]
    if not temp_present:
        print("  [INFO] No temperature columns (t1..t6) found in barometer data. "
              "Running pressure-only mode.")

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
    axes_flat = axes.flatten(order='F')
    for ax, (c, label) in zip(axes_flat, cols):
        color = force_color if label.startswith("F") else torque_color
        ax.plot(df["time"], df[c], linewidth=1.2, color=color)
        unit = "N" if label.startswith("F") else "N·m"
        ax.set_ylabel(f"{label} [{unit}]")
        ax.grid(alpha=0.3)
    axes[2, 0].set_xlabel("Experiment Time [s]")
    axes[2, 1].set_xlabel("Experiment Time [s]")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_temperature_subplots(df, save_path=None, suptitle="Barometer Temperatures"):
    """Plot t1..t6 temperature channels, if present."""
    temp_cols_present = [c for c in TEMP_COLS if c in df.columns]
    if not temp_cols_present:
        return None

    n = len(temp_cols_present)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 2 * nrows + 2), sharex=True)
    axes = np.array(axes).flatten()
    for i, col in enumerate(temp_cols_present):
        axes[i].plot(df["time"], df[col], linewidth=1.2, color='#e07b39')
        axes[i].set_ylabel(f"{col} [°C]")
        axes[i].grid(alpha=0.3)
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    axes[min(i, nrows * 2 - 2)].set_xlabel("Experiment Time [s]")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_path_xy_filtered(out_dir, X_mm, Y_mm, t_s, test_num,
                          break_on_time_gaps=True, gap_factor_time=5.0,
                          break_on_xy_jumps=True, jump_factor=5.0,
                          rect_size_mm=(40.0, 16.0), xlim=(-25, 25), ylim=(-10, 10)):
    plt.figure(figsize=(8, 6))
    n = len(X_mm)
    breaks = []
    if break_on_time_gaps and n > 1:
        dt = np.diff(t_s)
        med_dt = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.0
        if med_dt > 0:
            breaks.extend(np.where(dt > gap_factor_time * med_dt)[0].tolist())
    if break_on_xy_jumps and n > 1:
        steps = np.hypot(np.diff(X_mm), np.diff(Y_mm))
        med_step = np.median(steps[steps > 0]) if np.any(steps > 0) else 0.0
        if med_step > 0:
            breaks.extend(np.where(steps > jump_factor * med_step)[0].tolist())
    breaks = sorted(set(breaks))
    if breaks:
        xs, ys = [X_mm[0]], [Y_mm[0]]
        for i in range(n - 1):
            xs.append(X_mm[i + 1]); ys.append(Y_mm[i + 1])
            if i in breaks:
                xs.append(np.nan); ys.append(np.nan)
        x_plot, y_plot = np.array(xs), np.array(ys)
    else:
        x_plot, y_plot = X_mm, Y_mm
    plt.plot(x_plot, y_plot, linewidth=1.2, color='#44b155')
    try:
        ax = plt.gca()
        rw, rh = rect_size_mm
        rect = Rectangle((-rw/2, -rh/2), rw, rh, fill=False,
                         edgecolor='k', linewidth=1.0, linestyle='--', alpha=0.9)
        ax.add_patch(rect)
    except Exception:
        pass
    plt.grid(True, alpha=0.3); plt.axis("equal")
    plt.xlim(*xlim); plt.ylim(*ylim)
    plt.xlabel("Top X [mm]"); plt.ylabel("Top Y [mm]")
    plt.title(f"Tip path on top surface — trial {test_num}")
    out_path = os.path.join(out_dir, f"tip_path_top_xy_trial{test_num}.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close()


# ================================ MAIN =================================

def process_single_test(TEST_NUM):
    """Process a single test: synchronize barometers (with temperature) + forces."""
    DATA_DIR = BASE_DIR / f"test {TEST_NUM} - sensor v{VERSION_NUM}"

    PROCESSING_FILE = DATA_DIR / f"processing_test_{TEST_NUM}.csv"
    BAROMETER_FILE  = DATA_DIR / f"{TEST_NUM}barometers_trial.txt"

    # ── Isolated output folder — keeps temperature-aware outputs separate from
    #    the original pipeline so synchronized_events_XXXXX.csv is never overwritten.
    OUTPUT_DIR     = DATA_DIR / "with_temperature"
    OUTPUT_FILE    = OUTPUT_DIR / f"synchronized_events_{TEST_NUM}.csv"
    save_plots_dir = OUTPUT_DIR / "plots synchronized"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    save_plots_dir.mkdir(exist_ok=True, parents=True)

    print(f"Processing Test {TEST_NUM} (temperature-aware)...")

    if not PROCESSING_FILE.exists():
        raise FileNotFoundError(f"Processing file not found: {PROCESSING_FILE}")
    if not BAROMETER_FILE.exists():
        raise FileNotFoundError(f"Barometer file not found: {BAROMETER_FILE}")

    # Load data
    proc_df = load_processing_data(PROCESSING_FILE)
    baro_df = load_barometer_data(BAROMETER_FILE)

    # Report whether temperature data is available
    temp_cols_in_baro = [c for c in TEMP_COLS if c in baro_df.columns]
    if temp_cols_in_baro:
        print(f"  [OK] Temperature channels found: {temp_cols_in_baro}")
    else:
        print("  [WARN] No temperature channels (t1..t6) in barometer file. "
              "Output will have NaN for temperature columns.")

    # Process barometers (warmup, step leveling, outlier removal, drift removal)
    # Note: temperature columns (t{i}) are NOT drift-corrected here — they represent
    # physical temperature and are passed through as-is.
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

    # Dynamic re-zeroing when Fz ≈ 0
    if ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ:
        merged = rezero_barometers_when_fz_zero(
            merged,
            fz_threshold=FZ_ZERO_THRESHOLD,
            min_zero_duration=MIN_ZERO_DURATION_S,
            min_samples=MIN_ZERO_SAMPLES
        )

    # Plots
    plot_barometers_subplots(merged,
                             save_path=save_plots_dir / "4_barometers_final.png",
                             suptitle="Barometers (final processed)")
    plot_forces_torques_subplots(merged,
                                 save_path=save_plots_dir / "ati_forces_torques.png")
    plot_barometers_vs_fz(merged, save_path=save_plots_dir / "barometers_vs_Fz.png")
    plot_barometers_vs_fx(merged, save_path=save_plots_dir / "barometers_vs_Fx.png")
    plot_barometers_vs_fy(merged, save_path=save_plots_dir / "barometers_vs_Fy.png")
    # Plot temperature channels if present
    plot_temperature_subplots(merged, save_path=save_plots_dir / "temperatures.png")
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
        'Fx': 'fx', 'Fy': 'fy', 'Fz': 'fz',
        'fx': 'fx', 'fy': 'fy', 'fz': 'fz',
        'Tx': 'tx', 'Ty': 'ty', 'Tz': 'tz',
        'tx': 'tx', 'ty': 'ty', 'tz': 'tz',
        'barometer 1': 'b1', 'barometer 2': 'b2', 'barometer 3': 'b3',
        'barometer 4': 'b4', 'barometer 5': 'b5', 'barometer 6': 'b6',
    })

    # ── Desired column order: standard columns + temperature channels ──────────
    desired_cols = [
        't',
        'b1', 'b2', 'b3', 'b4', 'b5', 'b6',
        # Temperature columns: present if sensor firmware exposes b{i}_T
        't1', 't2', 't3', 't4', 't5', 't6',
        'x', 'y',
        'fx', 'fy', 'fz',
        'tx', 'ty', 'tz',
    ]
    # Only keep columns that actually exist (temperature may be absent)
    existing_desired = [c for c in desired_cols if c in masked.columns]
    missing_desired  = [c for c in desired_cols if c not in masked.columns]

    # For absent temperature columns: insert NaN columns so downstream scripts
    # can detect their absence and handle gracefully.
    for c in missing_desired:
        if c in TEMP_COLS:
            masked[c] = np.nan

    # Re-check after NaN fill
    existing_desired = [c for c in desired_cols if c in masked.columns]
    masked = masked.reindex(columns=existing_desired)

    if missing_desired:
        non_temp_missing = [c for c in missing_desired if c not in TEMP_COLS]
        if non_temp_missing:
            print(f"  [WARN] Non-temperature columns absent in output: {non_temp_missing}")
        print(f"  [INFO] Temperature columns filled with NaN (not in raw data): "
              f"{[c for c in missing_desired if c in TEMP_COLS]}")

    # Optional: no-contact masking
    if ENABLE_NO_CONTACT_MASKING:
        masked = mask_no_contact_samples(
            masked,
            fz_threshold=NO_CONTACT_FZ_THRESHOLD,
            sentinel=NO_CONTACT_SENTINEL
        )

    masked.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE}  ({len(masked)} rows, "
          f"temp cols: {[c for c in TEMP_COLS if masked[c].notna().any()]})")


def main():
    test_list = TEST_NUMS if isinstance(TEST_NUMS, list) else [TEST_NUMS]
    for test_num in test_list:
        try:
            process_single_test(test_num)
        except Exception as e:
            print(f"  [ERROR] Test {test_num}: {e}")
            continue


if __name__ == "__main__":
    main()
