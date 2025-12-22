#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Steps:
1) Load processing_test CSV (ROS/epoch seconds in 'time', positions, forces)
2) Load barometer CSV/TXT (new format: Epoch_s + b1..b6, or legacy datalog_*.csv)
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

TEST_NUM = 4651
VERSION_NUM = 4

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data")
DATA_DIR = BASE_DIR / f"test {TEST_NUM} - sensor v{VERSION_NUM}"

# Input files
PROCESSING_FILE = DATA_DIR / f"processing_test_{TEST_NUM}.csv"
BAROMETER_FILE  = DATA_DIR / f"barometers_trial{TEST_NUM}.txt"

# Output
OUTPUT_FILE = DATA_DIR / f"synchronized_events_{TEST_NUM}.csv"
save_plots_dir = DATA_DIR / "plots synchronized"
save_plots_dir.mkdir(exist_ok=True, parents=True)

# --- Synchronization params ---
ASOF_DIRECTION   = "nearest"
ASOF_TOLERANCE_S = 0.05   # seconds
bl_duration = 0.05  # seconds for barometer zeroing

# --- Dynamic re-zero based on Fz ≈ 0 ---
ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ = True   # <--- set False to disable
FZ_ZERO_THRESHOLD = 0.2                  # |Fz| [N] considered "zero"
MIN_ZERO_DURATION_S = 0.01                 # minimum duration of zero-force interval
MIN_ZERO_SAMPLES = 5                     # minimum number of samples in that interval

# --- Temperature-based drift flattening (before zeroing) ---
USE_TEMP_DRIFT_FLATTENING = True   # or False if you want to disable it

# --- Barometer outlier removal ---
REMOVE_BAROMETER_OUTLIERS = True         # <--- set False to disable outlier removal
OUTLIER_METHOD = 'iqr'                   # 'iqr' or 'zscore'
OUTLIER_THRESHOLD = 10                  # IQR multiplier (3.0) or Z-score threshold (3.0)

# --- Spatial masking (keep this) ---
SPATIAL_FILTER = {
    'x_range': (-20, 20),  # mm
    'y_range': (-8, 8),    # mm
    'z_range': (-2.0, 2.0) # mm
}

# =============================== HELPERS ===============================

def ensure_seconds(series: pd.Series) -> pd.Series:
    """
    Coerce a time-like numeric column to seconds.

    - Handles ROS/Unix epoch seconds (1e9-ish) WITHOUT rescaling.
    - Handles ns/us/ms (ROS stamps etc) by scaling down.
    """
    s = pd.to_numeric(series, errors='coerce')
    med = s.dropna().abs().median()

    # ultra-large -> nanoseconds (e.g., ROS nanosecond timestamps)
    if med > 1e14:
        s *= 1e-9   # ns -> s
    # large -> microseconds
    elif med > 1e11:
        s *= 1e-6   # µs -> s
    # medium-large (but not Unix epoch range) -> milliseconds
    # Unix epoch seconds are around 1.7e9, so only convert if < 1e9
    elif med > 1e5 and med < 1e9:
        s *= 1e-3   # ms -> s
    # else: assume already in seconds
    # (includes Unix epoch ~1.7e9, ROS time ~1e9, or small relative times)
    return s

# =============================== LOADERS ===============================

def load_processing_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # --- pick time column on ROS/epoch base ---
    time_col = None
    for cand in ['time', 'ros_time', 'ros_time_s', 'Epoch_s', 'time_ros', 't']:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        # as a last resort, try time_ns
        if 'time_ns' in df.columns:
            time_col = 'time_ns'
        else:
            raise ValueError("No suitable time column found in processing CSV "
                             "(expected one of: time, ros_time, Epoch_s, t, time_ns).")

    df['time'] = ensure_seconds(df[time_col])

    # positions
    required_pos = ['x_position_mm', 'y_position_mm', 'z_position_mm']
    missing = [c for c in required_pos if c not in df.columns]
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

    keep = ['time','x_position_mm','y_position_mm','z_position_mm'] + present_forces
    return df[keep].sort_values('time', ignore_index=True)

def load_barometer_data(path: Path) -> pd.DataFrame:
    """
    Load barometer file in either:
    - NEW format: PcTime, Epoch_s, Time_ms, b1..b6
    - LEGACY format: 'Epoch_s' + 'barometer 1..6' (datalog_*.csv)
    - LEGACY timestamp-only format (datalog_YYYY-MM-DD_*.csv + Timestamp column)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # ---- Case 1: new ROS/epoch format (your barometers_trial*.txt) ----
    if 'Epoch_s' in df.columns and 'b1_P' in df.columns:
        t = ensure_seconds(df['Epoch_s'])
        out = pd.DataFrame({'time': t})

        for i in range(1, 7):
            pcol = f"b{i}_P"
            if pcol not in df.columns:
                raise ValueError(f"Missing expected barometer column '{pcol}' in {path}")
            out[f"b{i}"] = pd.to_numeric(df[pcol], errors='coerce')

            # temperature column is optional but used if present
            tcol = f"b{i}_T"
            if tcol in df.columns:
                out[f"t{i}"] = pd.to_numeric(df[tcol], errors='coerce')


    # ---- Case 1b: old format with b1..b6 directly ----
    elif 'Epoch_s' in df.columns and 'b1' in df.columns:
        t = ensure_seconds(df['Epoch_s'])

        baro_cols = []
        for i in range(1, 7):
            col = f"b{i}"
            if col not in df.columns:
                raise ValueError(f"Missing expected barometer column '{col}' in {path}")
            baro_cols.append(col)

        out = pd.DataFrame({'time': t})
        # Only keep the 6 barometer columns we need (b1-b6), ignore any extra columns
        for col in baro_cols:
            out[col] = pd.to_numeric(df[col], errors='coerce')

    # ---- Case 2: legacy format with 'barometer 1..6' + Epoch_s ----
    elif 'Epoch_s' in df.columns:
        t = ensure_seconds(pd.to_numeric(df['Epoch_s'], errors='coerce'))
        baro_cols = [f'barometer {i}' for i in range(1,7) if f'barometer {i}' in df.columns]
        if len(baro_cols) != 6:
            raise ValueError(f"Expected 6 legacy barometer columns, found {len(baro_cols)}: {baro_cols}")
        out = pd.DataFrame({'time': t})
        for i, c in enumerate(baro_cols, start=1):
            out[f'b{i}'] = pd.to_numeric(df[c], errors='coerce')

    # ---- Case 3: very old 'Timestamp' + datalog_YYYY-MM-DD_*.csv ----
    else:
        ts_col = next((c for c in df.columns if 'timestamp' in c.lower()), None)
        if ts_col is None:
            raise ValueError("Barometer file needs either 'Epoch_s' or a 'Timestamp' column.")
        m = re.search(r'datalog_(\d{4}-\d{2}-\d{2})', os.path.basename(str(path)))
        if not m:
            raise ValueError("Filename must look like 'datalog_YYYY-MM-DD_...csv' for legacy mode.")
        date_str = m.group(1)
        dt_strings = date_str + ' ' + df[ts_col].astype(str)
        dt = pd.to_datetime(dt_strings, format='%Y-%m-%d %H:%M:%S.%f',
                            errors='coerce', utc=True)
        t = dt.view('int64') / 1e9  # seconds
        baro_cols = [f'barometer {i}' for i in range(1,7) if f'barometer {i}' in df.columns]
        if len(baro_cols) != 6:
            raise ValueError(f"Expected 6 legacy barometer columns, found {len(baro_cols)}: {baro_cols}")
        out = pd.DataFrame({'time': t})
        for i, c in enumerate(baro_cols, start=1):
            out[f'b{i}'] = pd.to_numeric(df[c], errors='coerce')


    return out.dropna(subset=['time']).sort_values('time', ignore_index=True)

# =========================== CORE OPERATIONS ===========================

def synchronize_data(proc_df: pd.DataFrame, baro_df: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbor time merge with tolerance (ROS/epoch seconds)."""
    merged = pd.merge_asof(
        proc_df.sort_values('time'),
        baro_df.sort_values('time'),
        on='time',
        direction=ASOF_DIRECTION,
        tolerance=ASOF_TOLERANCE_S
    )
    # keep only rows where barometers matched (but only check barometers that have data)
    # Check which barometers actually have non-NaN data in the merged result
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
        For 'iqr': multiplier for IQR (default 3.0, can use 1.5 for more aggressive filtering)
        For 'zscore': number of standard deviations (default 3.0)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with outlier rows removed
    """
    df_filtered = df.copy()
    baro_cols = [f'b{i}' for i in range(1, 7) if f'b{i}' in df.columns]
    
    if not baro_cols:
        print("remove_barometer_outliers: No barometer columns found.")
        return df_filtered
    
    initial_rows = len(df_filtered)
    outlier_mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
    
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
    
    print(f"remove_barometer_outliers ({method}, threshold={threshold}): "
          f"Removed {removed_rows} rows ({removed_rows/initial_rows*100:.2f}%)")
    
    return df_filtered

def zero_barometers(df, baseline_duration=1.0):
    """
    Subtract the mean of the first `baseline_duration` seconds from each barometer.

    df must contain 'time' and barometer columns (b1..b6 or barometer 1..6).
    """
    df = df.copy()
    t0 = df["time"].min()
    mask = df["time"] <= t0 + baseline_duration

    baro_cols = [c for c in df.columns if c.lower().startswith("b") or "barometer" in c.lower()]
    if not baro_cols:
        print("No barometer columns found.")
        return df

    for c in baro_cols:
        baseline = df.loc[mask, c].mean()
        df[c] = df[c] - baseline
        # print(f"{c}: baseline {baseline:.2f} hPa subtracted")

    return df

'''def remove_barometer_drift(df, poly_order: int = 1):
    """
    Remove slow drift from barometer channels by subtracting a polynomial
    trend (default: linear) fitted vs time.

    - df must contain 'time' and barometer columns (b1..b6 or 'barometer 1..6').
    - Returns a *copy* of the dataframe with de-drifted barometers.
    """
    df = df.copy()
    if "time" not in df.columns:
        raise ValueError("remove_barometer_drift: dataframe must contain 'time' column.")

    t = df["time"].to_numpy()
    if len(t) < poly_order + 1:
        print("remove_barometer_drift: not enough samples to fit trend.")
        return df

    # Center time to improve numerical stability
    t0 = t[0]
    t_rel = t - t0

    baro_cols = [c for c in df.columns if c.lower().startswith("b") or "barometer" in c.lower()]
    if not baro_cols:
        print("remove_barometer_drift: no barometer columns found.")
        return df

    for c in baro_cols:
        y = df[c].to_numpy()
        mask = np.isfinite(t_rel) & np.isfinite(y)
        if mask.sum() < poly_order + 1:
            print(f"remove_barometer_drift: not enough valid points for {c}.")
            continue

        # Fit polynomial trend y ≈ p(t_rel)
        coeffs = np.polyfit(t_rel[mask], y[mask], poly_order)
        trend = np.polyval(coeffs, t_rel)

        # Subtract trend
        df[c] = y - trend

    return df'''

def rezero_barometers_when_fz_zero(df: pd.DataFrame,
                                   fz_threshold: float = 0.05,
                                   min_zero_duration: float = 0.2,
                                   min_samples: int = 20) -> pd.DataFrame:
    """
    Adaptive drift correction:
    - Find time intervals where |Fz| is below `fz_threshold`
    - For each such interval (with enough samples and duration), compute the
      mean of each barometer in that interval and subtract it from all
      subsequent samples (including that interval).
      
    Effect: every time the probe is unloaded (Fz ≈ 0), barometers are
    're-zeroed' again to counter slow drift.
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
    baro_cols = [c for c in df.columns if c.lower().startswith("b") or "barometer" in c.lower()]
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
        print("rezero_barometers_when_fz_zero: no zero-force segments found.")
        return df

    idx_all = np.arange(n)

    for (s, e) in segments:
        seg_len = e - s + 1
        if seg_len < min_samples:
            continue
        if t[e] - t[s] < min_zero_duration:
            continue

        # print(f"[Re-zero] Zero-force segment from idx {s} to {e}, "
        #       f"t = {t[s]:.3f}–{t[e]:.3f} s, n = {seg_len}")

        seg_slice = slice(s, e + 1)
        for c in baro_cols:
            baseline = df[c].iloc[seg_slice].mean()
            # subtract baseline from this point onward
            df.loc[idx_all >= s, c] = df.loc[idx_all >= s, c] - baseline
            # print(f"  {c}: subtracting {baseline:.3f} from rows >= {s}")

    return df

def flatten_barometers_by_temperature(df, ref_mode="first"):
    """
    Remove temperature-driven drift from barometers using a simple linear model.

    Expects columns:
        b1..b6  -> pressures
        t1..t6  -> temperatures (same sensor order)

    For each i:
        Fit P_i ~ a_i * T_i + b_i
        Then subtract a_i * (T_i - T_ref) so that pressure is referenced to T_ref
        (T_ref = first temp or mean temp, depending on ref_mode).
    """
    df = df.copy()

    baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    temp_cols = [f"t{i}" for i in range(1, 7) if f"t{i}" in df.columns]

    if not baro_cols or len(baro_cols) != len(temp_cols):
        print("flatten_barometers_by_temperature: missing t1..t6 or b1..b6 — skipping.")
        return df

    # choose temperature reference
    T_ref = {}
    for tcol in temp_cols:
        if ref_mode == "mean":
            T_ref[tcol] = df[tcol].mean()
        else:  # "first"
            T_ref[tcol] = df[tcol].iloc[0]

    for i in range(1, 7):
        bcol = f"b{i}"
        tcol = f"t{i}"
        if bcol not in df.columns or tcol not in df.columns:
            continue

        T = df[tcol].to_numpy()
        P = df[bcol].to_numpy()

        mask = np.isfinite(T) & np.isfinite(P)
        if mask.sum() < 2:
            print(f"flatten_barometers_by_temperature: not enough valid data for {bcol}.")
            continue

        # Linear fit: P ≈ a*T + b
        a, b = np.polyfit(T[mask], P[mask], 1)

        # Subtract only the temperature-dependent part relative to T_ref
        df[bcol] = P - a * (T - T_ref[tcol])

        print(f"{bcol}: removed temperature trend (a={a:.4f} hPa/°C, T_ref={T_ref[tcol]:.2f} °C)")

    return df

# ==============================PLOTTING FUNCTIONS=======================

def _force_name(df, base):
    return base if base in df.columns else base.lower() if base.lower() in df.columns else None

def plot_barometers_subplots(df, save_path=None, suptitle="Barometers"):
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
    axes[min(len(cols),6)-1].set_xlabel("Experiment Time [s]")
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
        line2 = ax2.plot(df["time"], df[fcol], linestyle="--", alpha=0.8,
                         label=fcol, linewidth=1.2, color="C3")
        unit = "N" if fcol.lower().startswith("f") else "N·m"
        ax2.set_ylabel(f"{fcol} [{unit}]")
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Experiment Time [s]")
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
    plt.close()
    print(f"Saved plot: {out_path}")

# ================================ MAIN =================================

def main():
    print("#"*60)
    print(f"# BARO–FORCE SYNC | Test {TEST_NUM} v{VERSION_NUM}")
    print("#"*60)
    print(f"- merge_asof: direction={ASOF_DIRECTION}, tolerance={ASOF_TOLERANCE_S}s")

    if not PROCESSING_FILE.exists():
        raise FileNotFoundError(PROCESSING_FILE)

    baro_path = BAROMETER_FILE 
    if not Path(baro_path).exists():
        raise FileNotFoundError(baro_path)

    print(f"Using processing file: {PROCESSING_FILE}")
    print(f"Using barometer file:  {baro_path}")

    proc_df = load_processing_data(PROCESSING_FILE)
    baro_df = load_barometer_data(baro_path)    
    plot_barometers_subplots(baro_df, save_path=save_plots_dir / "barometers_subplots_before_processing.png", suptitle="Barometers (before processing)")
    
    if USE_TEMP_DRIFT_FLATTENING:
        baro_df = flatten_barometers_by_temperature(baro_df)
        plot_barometers_subplots(baro_df, save_path=save_plots_dir / "barometers_subplots_temp_flattened.png", suptitle="Barometers (after temp drift flattening)")
        
    merged = synchronize_data(proc_df, baro_df)

    # Zero the barometer data at the start
    merged = zero_barometers(merged, baseline_duration=bl_duration)

    # Optional: dynamic re-zero whenever |Fz| ≈ 0 to reduce drift
    if ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ:
        merged = rezero_barometers_when_fz_zero(
            merged,
            fz_threshold=FZ_ZERO_THRESHOLD,
            min_zero_duration=MIN_ZERO_DURATION_S,
            min_samples=MIN_ZERO_SAMPLES,
        )

    # Remove barometer outliers (if enabled)
    if REMOVE_BAROMETER_OUTLIERS:
        print("\nRemoving barometer outliers...")
        merged = remove_barometer_outliers(merged, method=OUTLIER_METHOD, threshold=OUTLIER_THRESHOLD)

    df = merged



    plot_barometers_subplots(df, save_path=save_plots_dir / "barometers_subplots.png", suptitle="Barometers (after processing)")
    plot_forces_torques_subplots(df, save_path=save_plots_dir / "ati_forces_torques.png")
    plot_barometers_vs_fz(df, save_path=save_plots_dir / "barometers_vs_Fz.png")
    plot_barometers_vs_fx(df, save_path=save_plots_dir / "barometers_vs_Fx.png")
    plot_barometers_vs_fy(df, save_path=save_plots_dir / "barometers_vs_Fy.png")

    plt.close()

    masked = apply_spatial_filter(merged)

    plot_path_xy_filtered(
        out_dir=str(save_plots_dir),
        X_mm=masked["x_position_mm"].to_numpy(),
        Y_mm=masked["y_position_mm"].to_numpy(),
        t_s=masked["time"].to_numpy(),
        test_num=TEST_NUM,
        break_on_time_gaps=True,
        gap_factor_time=20.0,  # Increase from 5.0 to be less sensitive
        break_on_xy_jumps=False,  # Disable XY jump detection to avoid segmentation
        jump_factor=20.0  # Increase from 5.0 if you re-enable it
    )

    if masked.empty:
        print("WARNING: no rows after spatial filter — saving UNSFILTERED synchronized data instead.")
        masked = merged

    # --- rename columns for export ---
    masked = masked.rename(columns={
        'time': 't',
        'x_position_mm': 'x', 'y_position_mm': 'y', 'z_position_mm': 'z',
        'Fx': 'fx', 'Fy': 'fy', 'Fz': 'fz', 'fx': 'fx', 'fy': 'fy', 'fz': 'fz',
        'Tx': 'tx', 'Ty': 'ty', 'Tz': 'tz', 'tx': 'tx', 'ty': 'ty', 'tz': 'tz',
        'barometer 1': 'b1', 'barometer 2': 'b2', 'barometer 3': 'b3',
        'barometer 4': 'b4', 'barometer 5': 'b5', 'barometer 6': 'b6',
    })

    # --- ensure the exported CSV has exactly the requested columns in order ---
    desired_cols = ['t', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'x', 'y', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']

    # Reindex will add any missing columns with NaN (so the CSV always contains the same columns)
    masked = masked.reindex(columns=desired_cols)

    masked.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Rows: {len(masked)} | Columns: {list(masked.columns)}")


if __name__ == "__main__":
    main()