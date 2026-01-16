#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barometer-Force Data Synchronization and Processing

Steps:
1) Load processing_test CSV (ROS/epoch seconds in 'time', positions, forces)
2) Load barometer CSV/TXT (new format: Epoch_s + b1..b6, or legacy datalog_*.csv)
3) Remove initial warm-up period (optional)
4) Apply drift removal (time-based or temperature-based)
5) Synchronize with merge_asof (nearest, tolerance)
6) Zero barometers at baseline
7) Dynamic re-zeroing when Fz ≈ 0 (optional)
8) Remove outliers (optional)
9) Apply spatial mask (keep only rows within bounds)
10) Generate plots and save synchronized CSV

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

TEST_NUM = 4700
VERSION_NUM = 4

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data")
DATA_DIR = BASE_DIR / f"test {TEST_NUM} - sensor v{VERSION_NUM}"

# Input files
PROCESSING_FILE = DATA_DIR / f"processing_test_{TEST_NUM}.csv"
BAROMETER_FILE = DATA_DIR / f"{TEST_NUM}barometers_trial.txt"

# Output
OUTPUT_FILE = DATA_DIR / f"synchronized_events_{TEST_NUM}.csv"
save_plots_dir = DATA_DIR / "plots synchronized"
save_plots_dir.mkdir(exist_ok=True, parents=True)

# --- Synchronization params ---
ASOF_DIRECTION = "nearest"
ASOF_TOLERANCE_S = 0.05  # seconds

# --- Initial baseline zeroing ---
BASELINE_METHOD = 'auto'  # Options: 'auto', 'fixed', 'zero_fz'
                          # 'auto' = automatically find most stable period
                          # 'fixed' = use fixed time window (after warmup)
                          # 'zero_fz' = use periods where Fz ≈ 0
BASELINE_DURATION_S = 2  # seconds for baseline window
BASELINE_WARMUP_S = 2.0  # For 'fixed' method: skip first N seconds before baseline
BASELINE_FZ_THRESHOLD = 0.1  # For 'zero_fz' method: |Fz| threshold [N]

# --- Remove initial warm-up data ---
REMOVE_INITIAL_WARMUP_DURATION = 0.5  # seconds to remove from the beginning (0 = disabled)

# --- Drift removal method ---
# Options: 'time', 'temperature', or None
DRIFT_REMOVAL_METHOD = 'temperature'  # 'time' = polynomial detrending vs time
                                       # 'temperature' = temperature-based correction
                                       # None = no drift removal
DRIFT_POLY_ORDER = 2                   # Polynomial order for drift fitting (1=linear, 2=quadratic)
DRIFT_SKIP_INITIAL_FRACTION = 0.05     # Fraction of data to skip when fitting drift (avoid warm-up)

# --- Dynamic re-zero based on Fz ≈ 0 ---
ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ = True  # Set False to disable
FZ_ZERO_THRESHOLD = 0.2                   # |Fz| [N] considered "zero"
MIN_ZERO_DURATION_S = 0.01                # Minimum duration of zero-force interval
MIN_ZERO_SAMPLES = 5                      # Minimum number of samples in that interval

# --- Barometer outlier removal ---
REMOVE_BAROMETER_OUTLIERS = True  # Set False to disable outlier removal
OUTLIER_METHOD = 'iqr'            # 'iqr' or 'zscore'
OUTLIER_THRESHOLD = 10            # IQR multiplier (3.0) or Z-score threshold (3.0)

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


def zero_barometers(df, method='auto', baseline_duration=2.0, warmup_s=0.0,
                   fz_threshold=0.1, save_diagnostic_plot=None):
    """
    Subtract the mean of the baseline period from each barometer.

    Parameters:
    -----------
    method : str
        'auto' = automatically find most stable period (recommended for offline)
        'fixed' = use fixed time window after warmup (for real-time)
        'zero_fz' = use periods where |Fz| < fz_threshold
    baseline_duration : float
        Duration in seconds for baseline window
    warmup_s : float
        For 'fixed' method: skip first N seconds before baseline
    fz_threshold : float
        For 'zero_fz' method: |Fz| threshold [N] for zero-force periods
    save_diagnostic_plot : str or None
        If provided, save diagnostic plot to this path
    """
    df = df.copy()

    baro_cols = [c for c in df.columns if c.lower().startswith("b") and len(c) == 2 and c[1].isdigit()]
    if not baro_cols:
        print("zero_barometers: No barometer columns found.")
        return df

    t = df["time"].to_numpy()
    t0 = t[0]

    # === METHOD 1: AUTO - Find most stable period ===
    if method == 'auto':
        print(f"\n[Method: AUTO] Finding most stable {baseline_duration}s period:")

        # Calculate window size in samples
        total_duration = t[-1] - t[0]
        window_samples = int(baseline_duration * len(t) / total_duration)
        window_samples = max(window_samples, 10)  # At least 10 samples

        # Scan through data to find lowest variance window
        best_start_idx = 0
        min_total_variance = float('inf')
        variance_profile = []  # For diagnostic plotting

        step_size = max(1, window_samples // 4)
        scan_indices = []

        for start_idx in range(0, len(df) - window_samples, step_size):
            end_idx = start_idx + window_samples
            window_variance = sum(df[c].iloc[start_idx:end_idx].var() for c in baro_cols)

            variance_profile.append(window_variance)
            scan_indices.append(start_idx)

            if window_variance < min_total_variance:
                min_total_variance = window_variance
                best_start_idx = start_idx

        best_end_idx = best_start_idx + window_samples
        mask = df.index.isin(range(best_start_idx, best_end_idx))
        t_start = t[best_start_idx]
        t_end = t[best_end_idx - 1]

        print(f"  ✓ Most stable period: t={t_start:.3f}-{t_end:.3f}s (indices {best_start_idx}-{best_end_idx})")
        print(f"    Total variance: {min_total_variance:.4f}")

        # Create diagnostic plot if requested
        if save_diagnostic_plot:
            _plot_baseline_diagnostics(df, t, scan_indices, variance_profile,
                                      best_start_idx, window_samples,
                                      baro_cols, save_diagnostic_plot)

    # === METHOD 2: FIXED - Use time window after warmup ===
    elif method == 'fixed':
        t_start = t0 + warmup_s
        t_end = t_start + baseline_duration
        mask = (df["time"] >= t_start) & (df["time"] <= t_end)

        if warmup_s > 0:
            print(f"\n[Method: FIXED] Using t={warmup_s:.1f}-{warmup_s+baseline_duration:.1f}s")
            print(f"  (skipping first {warmup_s}s warmup)")
        else:
            print(f"\n[Method: FIXED] Using first {baseline_duration}s")

        n_samples = mask.sum()
        print(f"  Baseline samples: {n_samples}")

    # === METHOD 3: ZERO_FZ - Use zero-force periods ===
    elif method == 'zero_fz':
        fz_col = _force_name(df, "Fz")
        if not fz_col:
            print("\n[Method: ZERO_FZ] WARNING: No Fz column found. Falling back to 'fixed' method.")
            return zero_barometers(df, method='fixed', baseline_duration=baseline_duration,
                                  warmup_s=warmup_s, save_diagnostic_plot=save_diagnostic_plot)

        print(f"\n[Method: ZERO_FZ] Finding baseline from zero-force periods (|Fz| < {fz_threshold}N):")

        fz = pd.to_numeric(df[fz_col], errors="coerce").to_numpy()
        zero_fz_mask = np.isfinite(fz) & (np.abs(fz) <= fz_threshold)

        if not np.any(zero_fz_mask):
            print(f"  WARNING: No periods with |Fz| < {fz_threshold}N found. Falling back to 'fixed' method.")
            return zero_barometers(df, method='fixed', baseline_duration=baseline_duration,
                                  warmup_s=warmup_s, save_diagnostic_plot=save_diagnostic_plot)

        # Find contiguous zero-force segments
        segments = []
        in_seg = False
        start = None

        for i, is_zero in enumerate(zero_fz_mask):
            if is_zero and not in_seg:
                in_seg = True
                start = i
            elif not is_zero and in_seg:
                segments.append((start, i - 1))
                in_seg = False
        if in_seg:
            segments.append((start, len(zero_fz_mask) - 1))

        # Find the longest/most stable zero-force segment
        best_seg = None
        best_score = float('inf')

        for (s, e) in segments:
            duration = t[e] - t[s]
            if duration < baseline_duration:
                continue

            # Score = variance (lower is better)
            seg_variance = sum(df[c].iloc[s:e+1].var() for c in baro_cols)

            if seg_variance < best_score:
                best_score = seg_variance
                best_seg = (s, e)

        if best_seg is None:
            print(f"  WARNING: No zero-force segments ≥ {baseline_duration}s found.")
            print(f"  Using all {np.sum(zero_fz_mask)} scattered zero-force samples.")
            mask = zero_fz_mask
        else:
            s, e = best_seg
            mask = df.index.isin(range(s, e + 1))
            print(f"  ✓ Best zero-force period: t={t[s]:.3f}-{t[e]:.3f}s (indices {s}-{e})")
            print(f"    Variance: {best_score:.4f}, Duration: {t[e]-t[s]:.1f}s")

    else:
        raise ValueError(f"Unknown baseline method '{method}'. Use 'auto', 'fixed', or 'zero_fz'.")

    # === Apply baseline subtraction ===
    n_baseline = mask.sum()
    if n_baseline == 0:
        print("  ERROR: No baseline samples found!")
        return df

    print(f"\n  Subtracting baseline from {len(baro_cols)} barometers:")
    for c in baro_cols:
        baseline = df.loc[mask, c].mean()
        std = df.loc[mask, c].std()
        df[c] = df[c] - baseline

        # Quality indicator
        quality = "✓ excellent" if std < 0.5 else "✓ good" if std < 1.0 else "⚠ fair" if std < 2.0 else "✗ poor"
        print(f"    {c}: baseline={baseline:7.2f} hPa, std={std:5.3f} hPa  {quality}")

    return df


def _plot_baseline_diagnostics(df, t, scan_indices, variance_profile,
                               best_idx, window_samples, baro_cols, save_path):
    """Helper function to create diagnostic plot for baseline selection."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Variance profile across time
    ax = axes[0]
    scan_times = t[scan_indices]
    ax.plot(scan_times, variance_profile, 'b-', linewidth=1.5, label='Total variance')

    # Mark the selected baseline region
    best_time = t[best_idx]
    best_time_end = t[best_idx + window_samples - 1]
    ax.axvspan(best_time, best_time_end, alpha=0.3, color='green', label='Selected baseline')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Total Variance')
    ax.set_title('Baseline Selection: Variance Scan')
    ax.grid(alpha=0.3)
    ax.legend()

    # Plot 2: Barometer traces with baseline region highlighted
    ax = axes[1]
    for i, c in enumerate(baro_cols):
        ax.plot(t, df[c], linewidth=0.8, alpha=0.7, label=c)
    ax.axvspan(best_time, best_time_end, alpha=0.3, color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pressure [hPa]')
    ax.set_title('Barometer Data (before zeroing)')
    ax.grid(alpha=0.3)
    ax.legend(ncol=6, fontsize=8)

    # Plot 3: Zoomed view of baseline region
    ax = axes[2]
    baseline_mask = (t >= best_time) & (t <= best_time_end)
    t_zoom = t[baseline_mask]

    for c in baro_cols:
        y_zoom = df[c].to_numpy()[baseline_mask]
        ax.plot(t_zoom, y_zoom, linewidth=1.2, label=c)

        # Show mean and std
        mean_val = y_zoom.mean()
        std_val = y_zoom.std()
        ax.axhline(mean_val, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pressure [hPa]')
    ax.set_title(f'Baseline Region (zoomed) - Duration: {best_time_end - best_time:.2f}s')
    ax.grid(alpha=0.3)
    ax.legend(ncol=6, fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Diagnostic plot saved: {save_path}")


def remove_barometer_outliers(df: pd.DataFrame, method='iqr', threshold=3.0) -> pd.DataFrame:
    """
    Remove rows where any barometer has outlier values.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing barometer columns (b1..b6)
    method : str
        'iqr' for Interquartile Range method (default)
        'zscore' for Z-score method
    threshold : float
        For 'iqr': multiplier for IQR (default 3.0)
        For 'zscore': number of standard deviations (default 3.0)
    """
    df_filtered = df.copy()
    baro_cols = [f'b{i}' for i in range(1, 7) if f'b{i}' in df.columns]

    if not baro_cols:
        print("remove_barometer_outliers: No barometer columns found.")
        return df_filtered

    initial_rows = len(df_filtered)
    outlier_mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)

    print(f"\nRemoving barometer outliers (method={method}, threshold={threshold}):")
    for col in baro_cols:
        if method == 'iqr':
            Q1 = df_filtered[col].quantile(0.25)
            Q3 = df_filtered[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            col_outliers = (df_filtered[col] < lower_bound) | (df_filtered[col] > upper_bound)

        elif method == 'zscore':
            mean = df_filtered[col].mean()
            std = df_filtered[col].std()
            z_scores = np.abs((df_filtered[col] - mean) / std)
            col_outliers = z_scores > threshold

        else:
            raise ValueError(f"Unknown method '{method}'. Use 'iqr' or 'zscore'.")

        outlier_mask = outlier_mask | col_outliers
        n_outliers = col_outliers.sum()
        if n_outliers > 0:
            print(f"  {col}: {n_outliers} outliers detected")

    df_filtered = df_filtered[~outlier_mask].copy()
    removed_rows = initial_rows - len(df_filtered)

    print(f"Total removed: {removed_rows} rows ({removed_rows/initial_rows*100:.2f}%)")

    return df_filtered


def flatten_barometers_by_time(df, skip_initial_fraction=0.05, poly_order=2):
    """
    Remove time-based drift from barometers using polynomial detrending.

    This method fits a polynomial trend vs TIME (not temperature) and removes it.
    Useful when drift is temporal rather than temperature-dependent.
    """
    df = df.copy()

    if "time" not in df.columns:
        print("remove_barometer_drift_polynomial: no 'time' column found, skipping.")
        return df

    baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    if not baro_cols:
        print("remove_barometer_drift_polynomial: no barometer columns found, skipping.")
        return df

    t = df["time"].to_numpy()
    n_total = len(t)
    n_skip = int(n_total * skip_initial_fraction)

    print(f"\nTime-based drift removal: poly_order={poly_order}, skipping first {n_skip}/{n_total} samples ({skip_initial_fraction*100:.1f}%)")

    # Use relative time for numerical stability
    t_rel = t - t[0]

    for bcol in baro_cols:
        P = df[bcol].to_numpy()

        # Use only STABLE data for fitting (skip initial warm-up period)
        t_stable = t_rel[n_skip:]
        P_stable = P[n_skip:]

        mask = np.isfinite(t_stable) & np.isfinite(P_stable)
        if mask.sum() < poly_order + 1:
            print(f"  {bcol}: not enough valid stable data, skipping.")
            continue

        # Check if pressure has significant drift
        P_range = P_stable[mask].max() - P_stable[mask].min()
        if P_range < 0.1:  # Less than 0.1 hPa variation
            print(f"  {bcol}: pressure variation too small ({P_range:.2f} hPa), skipping.")
            continue

        # Fit polynomial: P ≈ polynomial(t)
        coeffs = np.polyfit(t_stable[mask], P_stable[mask], poly_order)

        # Compute trend for ALL data
        trend = np.polyval(coeffs, t_rel)

        # Reference to the trend value at the first stable point
        trend_ref = trend[n_skip]

        # Subtract drift (keep the initial stable value as reference)
        correction = trend - trend_ref
        df[bcol] = P - correction

        poly_name = "linear" if poly_order == 1 else f"order-{poly_order}"
        max_correction = np.abs(correction).max()
        print(f"  {bcol}: removed time-based drift ({poly_name}, max_correction={max_correction:.2f} hPa)")

    return df


def flatten_barometers_by_temperature(df, ref_mode="first", skip_initial_fraction=0.05, poly_order=2):
    """
    Remove temperature-driven drift from barometers using polynomial fitting.

    Expects columns:
        b1..b6  -> pressures
        t1..t6  -> temperatures (same sensor order)
    """
    df = df.copy()

    baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    temp_cols = [f"t{i}" for i in range(1, 7) if f"t{i}" in df.columns]

    if not baro_cols or len(baro_cols) != len(temp_cols):
        print("flatten_barometers_by_temperature: missing t1..t6 or b1..b6 — skipping.")
        return df

    # Calculate how many samples to skip at the beginning
    n_total = len(df)
    n_skip = int(n_total * skip_initial_fraction)

    print(f"\nTemperature-based drift removal: poly_order={poly_order}, skipping first {n_skip}/{n_total} samples ({skip_initial_fraction*100:.1f}%)")

    # Choose temperature reference from STABLE data (after warm-up)
    T_ref = {}
    for tcol in temp_cols:
        stable_temps = df[tcol].iloc[n_skip:]
        if ref_mode == "mean":
            T_ref[tcol] = stable_temps.mean()
        else:  # "first"
            T_ref[tcol] = stable_temps.iloc[0] if len(stable_temps) > 0 else df[tcol].iloc[0]

    for i in range(1, 7):
        bcol = f"b{i}"
        tcol = f"t{i}"
        if bcol not in df.columns or tcol not in df.columns:
            continue

        T = df[tcol].to_numpy()
        P = df[bcol].to_numpy()

        # Use only STABLE data for fitting (skip initial warm-up period)
        T_stable = T[n_skip:]
        P_stable = P[n_skip:]

        mask = np.isfinite(T_stable) & np.isfinite(P_stable)
        if mask.sum() < poly_order + 1:
            print(f"  {bcol}: not enough valid stable data, skipping.")
            continue

        # Check if temperature has sufficient variation
        T_range = T_stable[mask].max() - T_stable[mask].min()
        if T_range < 0.5:  # Less than 0.5°C variation
            print(f"  {bcol}: temperature variation too small ({T_range:.2f}°C), skipping.")
            continue

        # Fit polynomial: P ≈ polynomial(T)
        coeffs = np.polyfit(T_stable[mask], P_stable[mask], poly_order)

        # Compute trend for ALL data (including warm-up period)
        trend = np.polyval(coeffs, T)
        trend_ref = np.polyval(coeffs, np.full_like(T, T_ref[tcol]))

        # Calculate the correction
        correction = trend - trend_ref

        # Safety check: if correction is unreasonably large, skip it
        max_correction = np.abs(correction).max()
        P_range = P_stable[mask].max() - P_stable[mask].min()

        if max_correction > 2 * P_range:  # Correction larger than 2x the pressure range
            print(f"  {bcol}: correction too large ({max_correction:.2f} hPa vs range {P_range:.2f} hPa), skipping.")
            continue

        # Subtract temperature-dependent drift (referenced to T_ref)
        df[bcol] = P - correction

        poly_name = "linear" if poly_order == 1 else f"order-{poly_order}"
        print(f"  {bcol}: removed temperature drift ({poly_name}, T_ref={T_ref[tcol]:.2f}°C, max_correction={max_correction:.2f} hPa)")

    return df


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

def main():
    print("#" * 60)
    print(f"# BAROMETER-FORCE SYNC | Test {TEST_NUM} v{VERSION_NUM}")
    print("#" * 60)
    print(f"Configuration:")
    print(f"  - Sync: direction={ASOF_DIRECTION}, tolerance={ASOF_TOLERANCE_S}s")
    print(f"  - Drift removal: {DRIFT_REMOVAL_METHOD}")
    print(f"  - Dynamic re-zeroing: {ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ}")
    print(f"  - Outlier removal: {REMOVE_BAROMETER_OUTLIERS}")
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

    # Plot raw barometers
    plot_barometers_subplots(baro_df,  save_path=save_plots_dir / "1barometers_subplots_before_processing.png",  suptitle="Barometers (before processing)")

    # Remove initial warm-up data if enabled
    if REMOVE_INITIAL_WARMUP_DURATION > 0:
        t0 = baro_df["time"].min()
        initial_len = len(baro_df)
        baro_df = baro_df[baro_df["time"] > t0 + REMOVE_INITIAL_WARMUP_DURATION].copy()
        removed_rows = initial_len - len(baro_df)
        print(f"\nRemoved {removed_rows} rows from initial {REMOVE_INITIAL_WARMUP_DURATION}s warm-up period")
        baro_df = baro_df.reset_index(drop=True)

    # Apply drift removal
    if DRIFT_REMOVAL_METHOD == 'time':
        baro_df = flatten_barometers_by_time( baro_df, skip_initial_fraction=DRIFT_SKIP_INITIAL_FRACTION, poly_order=DRIFT_POLY_ORDER )
        plot_barometers_subplots(baro_df, save_path=save_plots_dir / "2barometers_subplots_drift_removed.png", suptitle="Barometers (after time-based drift removal)")
    elif DRIFT_REMOVAL_METHOD == 'temperature':
        baro_df = flatten_barometers_by_temperature( baro_df,  skip_initial_fraction=DRIFT_SKIP_INITIAL_FRACTION, poly_order=DRIFT_POLY_ORDER  )
        plot_barometers_subplots(baro_df, save_path=save_plots_dir / "2barometers_subplots_drift_removed.png", suptitle="Barometers (after temp drift flattening)")

    # Synchronize
    print("\nSynchronizing data...")
    merged = synchronize_data(proc_df, baro_df)
    print(f"  - Synchronized: {len(merged)} rows")

    # Zero barometers at baseline
    merged = zero_barometers(merged,
                            method=BASELINE_METHOD,
                            baseline_duration=BASELINE_DURATION_S,
                            warmup_s=BASELINE_WARMUP_S,
                            fz_threshold=BASELINE_FZ_THRESHOLD,
                            save_diagnostic_plot=save_plots_dir / "baseline_diagnostic.png")

    # Dynamic re-zeroing when Fz ≈ 0
    if ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ:
        merged = rezero_barometers_when_fz_zero( merged, fz_threshold=FZ_ZERO_THRESHOLD, min_zero_duration=MIN_ZERO_DURATION_S, min_samples=MIN_ZERO_SAMPLES, )

    # Remove barometer outliers
    if REMOVE_BAROMETER_OUTLIERS:
        merged = remove_barometer_outliers(merged, method=OUTLIER_METHOD, threshold=OUTLIER_THRESHOLD)
        print(f"  - After outlier removal: {len(merged)} rows")

    # Plot processed data
    print("\nGenerating plots...")
    plot_barometers_subplots(merged, save_path=save_plots_dir / "3barometers_subplots.png", suptitle="Barometers (after processing)")

    plot_forces_torques_subplots(merged, save_path=save_plots_dir / "ati_forces_torques.png")
    plot_barometers_vs_fz(merged, save_path=save_plots_dir / "barometers_vs_Fz.png")
    plot_barometers_vs_fx(merged, save_path=save_plots_dir / "barometers_vs_Fx.png")
    plot_barometers_vs_fy(merged, save_path=save_plots_dir / "barometers_vs_Fy.png")
    plt.close()

    # Apply spatial filter
    print("\nApplying spatial filter...")
    masked = apply_spatial_filter(merged)
    print(f"  - After spatial filter: {len(masked)} rows")

    # Plot trajectory
    plot_path_xy_filtered( out_dir=str(save_plots_dir), X_mm=masked["x_position_mm"].to_numpy(), Y_mm=masked["y_position_mm"].to_numpy(), t_s=masked["time"].to_numpy(), test_num=TEST_NUM, break_on_time_gaps=True, gap_factor_time=PATH_GAP_FACTOR_TIME, break_on_xy_jumps=PATH_BREAK_ON_XY_JUMPS, jump_factor=PATH_JUMP_FACTOR, rect_size_mm=PATH_PLOT_RECT_SIZE_MM, xlim=PATH_PLOT_XLIM, ylim=PATH_PLOT_YLIM )

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

    # Save output
    masked.to_csv(OUTPUT_FILE, index=False)
    print(f"\nOutput saved: {OUTPUT_FILE}")
    print(f"  - Rows: {len(masked)}")
    print(f"  - Columns: {list(masked.columns)}")
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()