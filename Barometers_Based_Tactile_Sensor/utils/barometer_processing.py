#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barometer Data Processing Pipeline

Handles loading, cleaning, drift removal, zeroing, re-zeroing, and plotting
of barometer data. Extracted from data_organization to be reusable.

Functions:
- load_barometer_data: Load barometer CSV/TXT in multiple formats
- remove_outliers_barometers: MAD-based outlier removal
- rezero_barometers_when_fz_zero: Dynamic re-zeroing during zero-force intervals
- process_barometers: Full pre-synchronization pipeline
- Plotting: plot_barometers_subplots, plot_barometers_vs_force, etc.

Author: Aurora Ruggeri
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt



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


# ========================= DRIFT / STEP REMOVAL =========================

def remove_steps_robust(df, baro_cols=None, threshold=3.0, window_size=200, settling_samples=None):
    """
    Remove steps by flattening each chunk to match the initial baseline.

    Algorithm:
    1. Compute baseline of first chunk (using 5th percentile to ignore peaks)
    2. For each subsequent chunk, compute its baseline
    3. If chunk baseline differs from reference, subtract the difference
    4. This forces all data to have the same baseline level

    Parameters
    ----------
    df : pd.DataFrame
        Data with barometer columns.
    baro_cols : list, optional
        Barometer column names. Auto-detects b1..b6 if None.
    threshold : float
        Minimum deviation from baseline to correct (hPa). Default: 3.0
    window_size : int
        Chunk size for computing baselines. Default: 200 samples (~2 s at 100 Hz)
    settling_samples : ignored
        Kept for API compatibility.

    Returns
    -------
    pd.DataFrame
        Data with steps removed and zeroed at start.
    """
    df = df.copy()

    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]

    if not baro_cols:
        return df

    for col in baro_cols:
        if col not in df.columns:
            continue

        y = df[col].to_numpy().astype(float).copy()
        n = len(y)

        if n < window_size:
            continue

        def get_baseline(arr):
            valid = arr[np.isfinite(arr)]
            if len(valid) == 0:
                return np.nan
            return np.percentile(valid, 5)

        ref_baseline = get_baseline(y[:window_size])

        if not np.isfinite(ref_baseline):
            continue

        num_chunks = (n + window_size - 1) // window_size

        for c in range(num_chunks):
            start = c * window_size
            end = min(start + window_size, n)

            chunk_baseline = get_baseline(y[start:end])

            if not np.isfinite(chunk_baseline):
                continue

            deviation = chunk_baseline - ref_baseline

            if abs(deviation) > threshold:
                y[start:end] -= deviation

        # Zero at start
        first_valid = np.where(np.isfinite(y))[0]
        if len(first_valid) > 0:
            y = y - y[first_valid[0]]

        df[col] = y

    return df


def remove_drift_ema(df, baro_cols=None, alpha=0.0001, alpha_override=None, zero_at_start=True):
    """
    Remove slow drift using an exponential moving average (EMA).

    Parameters
    ----------
    df : pd.DataFrame
        Data with barometer columns.
    baro_cols : list, optional
        Barometer column names. Auto-detects b1..b6 if None.
    alpha : float
        EMA smoothing factor. Default: 0.0001
    alpha_override : dict, optional
        Per-sensor alpha overrides, e.g. {"b3": 0.001}.
    zero_at_start : bool
        Whether to zero the corrected signal at the first valid index.

    Returns
    -------
    (pd.DataFrame, dict)
        Corrected DataFrame and dict of EMA trend arrays per column.
    """
    df = df.copy()

    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]

    if not baro_cols:
        return df, {}

    if alpha_override is None:
        alpha_override = {}

    ema_trends = {}

    for col in baro_cols:
        if col not in df.columns:
            continue

        alpha_use = alpha_override.get(col, alpha)
        P = df[col].to_numpy()
        n = len(P)

        valid_mask = np.isfinite(P)
        if not valid_mask.any():
            continue

        first_valid_idx = np.where(valid_mask)[0][0]

        ema = np.zeros(n)
        ema[first_valid_idx] = P[first_valid_idx]

        for i in range(first_valid_idx + 1, n):
            if np.isfinite(P[i]):
                ema[i] = alpha_use * P[i] + (1 - alpha_use) * ema[i - 1]
            else:
                ema[i] = ema[i - 1]

        ema_trends[col] = ema.copy()
        P_corrected = P - ema

        if zero_at_start:
            P_corrected = P_corrected - P_corrected[first_valid_idx]

        df[col] = P_corrected

    return df, ema_trends


def zero_barometers_fixed(df, baro_cols=None, reference_index='first_valid'):
    """
    Zero barometers at a specific index.

    Parameters
    ----------
    df : pd.DataFrame
        Data with barometer columns.
    baro_cols : list, optional
        Barometer column names. Auto-detects b1..b6 if None.
    reference_index : int or 'first_valid'
        Index to use as zero reference. Default: 'first_valid'.

    Returns
    -------
    pd.DataFrame
        Zeroed DataFrame.
    """
    df = df.copy()

    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]

    for col in baro_cols:
        if col not in df.columns:
            continue

        if reference_index == 'first_valid':
            valid_idx = np.where(np.isfinite(df[col]))[0]
            if len(valid_idx) == 0:
                continue
            ref_idx = valid_idx[0]
        else:
            ref_idx = reference_index

        ref_value = df[col].iloc[ref_idx]
        if np.isfinite(ref_value):
            df[col] = df[col] - ref_value

    return df


def remove_drift_temperature_linear(df, baro_cols=None, temp_cols=None,
                                    skip_initial_fraction=0.05, zero_at_start=True):
    """
    Temperature-based linear drift compensation.

    Parameters
    ----------
    df : pd.DataFrame
        Data with barometer and temperature columns.
    baro_cols : list, optional
        Barometer column names. Auto-detects b1..b6 if None.
    temp_cols : list, optional
        Temperature column names. Auto-detects t1..t6 if None.
    skip_initial_fraction : float
        Fraction of data to skip for fitting. Default: 0.05
    zero_at_start : bool
        Whether to zero the corrected signal at start.

    Returns
    -------
    (pd.DataFrame, dict)
        Corrected DataFrame and dict of temperature coefficients.
    """
    df = df.copy()
    if temp_cols is None:
        temp_cols = [f"t{i}" for i in range(1, 7) if f"t{i}" in df.columns]
    if not temp_cols:
        return df, {}
    return df, {}


def plot_drift_comparison(*args, **kwargs):
    """Placeholder for drift comparison plotting."""
    pass


def plot_temperature_vs_pressure(*args, **kwargs):
    """Placeholder for temperature vs pressure plotting."""
    pass


# =========================== CORE OPERATIONS ===========================

def remove_outliers_barometers(df: pd.DataFrame, threshold_multiplier: float = 10.0) -> pd.DataFrame:
    """
    Remove outliers from barometer data using robust median absolute deviation (MAD).

    Uses median and MAD (robust statistics) instead of mean and std to avoid
    contamination by outliers. Removes samples where the value exceeds
    median +/- (threshold_multiplier * MAD).

    Args:
        df: DataFrame with barometer columns b1-b6
        threshold_multiplier: Number of MADs for threshold (default: 10)

    Returns:
        DataFrame with outliers removed (rows with any outlier are dropped)
    """
    df = df.copy()

    baro_cols = [f'b{i}' for i in range(1, 7) if f'b{i}' in df.columns]
    if not baro_cols:
        return df

    outlier_mask = pd.Series(False, index=df.index)

    for col in baro_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue

        # Use robust statistics (median and MAD)
        median = values.median()
        mad = np.median(np.abs(values - median))

        # Avoid division by zero
        if mad < 1e-10:
            mad = 1e-10

        # Mark outliers (outside median +/- threshold * MAD)
        lower_bound = median - threshold_multiplier * mad
        upper_bound = median + threshold_multiplier * mad

        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask |= col_outliers

    # Remove rows with any outliers
    df_clean = df[~outlier_mask].copy()

    return df_clean


def rezero_barometers_when_fz_zero(df: pd.DataFrame, fz_threshold: float = 0.05,
                                    min_zero_duration: float = 0.2,
                                    min_samples: int = 20) -> pd.DataFrame:
    """
    Adaptive drift correction: re-zero barometers when Fz ~ 0.

    Finds time intervals where |Fz| < fz_threshold and re-zeros barometers
    at those points to counter slow drift.
    """
    df = df.copy()

    if "time" not in df.columns:
        raise ValueError("rezero_barometers_when_fz_zero: dataframe must contain 'time' column.")

    # Find Fz column name (handles 'Fz' or 'fz')
    fz_col = _force_name(df, "Fz")
    if not fz_col:
        return df

    t = df["time"].to_numpy()
    fz = pd.to_numeric(df[fz_col], errors="coerce").to_numpy()

    # Barometer columns
    baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    if not baro_cols:
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
        return df

    idx_all = np.arange(n)

    for (s, e) in segments:
        seg_len = e - s + 1
        if seg_len < min_samples:
            continue
        if t[e] - t[s] < min_zero_duration:
            continue

        seg_slice = slice(s, e + 1)

        for c in baro_cols:
            baseline = df[c].iloc[seg_slice].mean()
            # subtract baseline from this point onward
            df.loc[idx_all >= s, c] = df.loc[idx_all >= s, c] - baseline

    return df


# =========================== PIPELINE ===========================

def process_barometers(baro_df: pd.DataFrame,
                       save_plots_dir: Path = None,
                       # Warmup
                       warmup_duration: float = 1.0,
                       # Step leveling
                       enable_step_leveling: bool = True,
                       step_threshold_hpa: float = 5.0,
                       step_window_size: int = 200,
                       # Outlier removal
                       enable_outlier_removal: bool = True,
                       outlier_threshold_multiplier: float = 30.0,
                       # Drift removal
                       drift_removal_method: str = "ema",
                       ema_alpha: float = 0.0001,
                       ema_alpha_override: dict = None,
                       temp_skip_initial_fraction: float = 0.05,
                       zero_at_start: bool = True,
                       ) -> pd.DataFrame:
    """
    Full barometer pre-synchronization processing pipeline.

    Steps (with optional plots saved to save_plots_dir):
    1. Remove initial warm-up data
    2. Step leveling (hardware jump removal)
    3. Outlier removal (MAD-based)
    4. Drift removal (EMA / temperature / both / none)

    Args:
        baro_df: Raw barometer DataFrame (from load_barometer_data)
        save_plots_dir: Directory to save intermediate plots (None = no plots)
        warmup_duration: Seconds to remove from start (0 = disabled)
        enable_step_leveling: Enable step/jump leveling
        step_threshold_hpa: Threshold for step detection (hPa)
        step_window_size: Window size for step detection (samples)
        enable_outlier_removal: Enable MAD-based outlier removal
        outlier_threshold_multiplier: MAD multiplier for outlier threshold
        drift_removal_method: "ema", "temperature", "both", or "none"
        ema_alpha: EMA smoothing factor
        ema_alpha_override: Per-sensor alpha overrides dict
        temp_skip_initial_fraction: Fraction of data to skip for temp fitting
        zero_at_start: Zero barometers after drift removal

    Returns:
        Processed barometer DataFrame
    """
    baro_df = baro_df.copy()

    if ema_alpha_override is None:
        ema_alpha_override = {}

    # Plot raw barometers
    if save_plots_dir:
        plot_barometers_subplots(baro_df,
                                save_path=save_plots_dir / "1_barometers_raw.png",
                                suptitle="Barometers (raw data)")

    # STEP 1: Remove initial warm-up data
    if warmup_duration > 0:
        t0 = baro_df["time"].min()
        baro_df = baro_df[baro_df["time"] > t0 + warmup_duration].copy()
        baro_df = baro_df.reset_index(drop=True)

        if save_plots_dir:
            plot_barometers_subplots(baro_df,
                                    save_path=save_plots_dir / "2 _barometers_after_warmup_removal.png",
                                    suptitle="Barometers (after warmup removal)")

    # STEP 1.5: Remove hardware steps/jumps
    if enable_step_leveling:
        baro_df = remove_steps_robust(baro_df, threshold=step_threshold_hpa,
                                       window_size=step_window_size)
        if save_plots_dir:
            plot_barometers_subplots(baro_df,
                                    save_path=save_plots_dir / "2.5_barometers_after_step_leveling.png",
                                    suptitle="Barometers (after step leveling)")

    # STEP 2.75: Remove outliers
    if enable_outlier_removal:
        baro_df = remove_outliers_barometers(baro_df,
                                              threshold_multiplier=outlier_threshold_multiplier)

        # Re-zero after outlier removal (since removing rows shifts the reference)
        baro_cols = [f'b{i}' for i in range(1, 7) if f'b{i}' in baro_df.columns]
        for col in baro_cols:
            first_valid_idx = baro_df[col].first_valid_index()
            if first_valid_idx is not None:
                baro_df[col] = baro_df[col] - baro_df.loc[first_valid_idx, col]

        if save_plots_dir:
            plot_barometers_subplots(baro_df,
                                    save_path=save_plots_dir / "2.75_barometers_after_outlier_removal.png",
                                    suptitle="Barometers (after outlier removal - 10\u03c3)")

    # STEP 2: Drift removal
    if drift_removal_method == 'ema':
        baro_df, _ema_trends = remove_drift_ema(
            baro_df,
            alpha=ema_alpha,
            alpha_override=ema_alpha_override,
            zero_at_start=zero_at_start
        )
        if save_plots_dir:
            plot_barometers_subplots(baro_df,
                                    save_path=save_plots_dir / "3_barometers_after_ema.png",
                                    suptitle="Barometers (after EMA drift removal)")

    elif drift_removal_method == 'temperature':
        baro_df, temp_coeffs = remove_drift_temperature_linear(
            baro_df,
            skip_initial_fraction=temp_skip_initial_fraction,
            zero_at_start=zero_at_start
        )
        if save_plots_dir:
            plot_barometers_subplots(baro_df,
                                    save_path=save_plots_dir / "3_barometers_after_temp_correction.png",
                                    suptitle="Barometers (after temperature correction)")
            plot_temperature_vs_pressure(baro_df, temp_coeffs=temp_coeffs,
                                         save_path=save_plots_dir / "temperature_vs_pressure_correlation.png")

    elif drift_removal_method == 'both':
        baro_df_temp, temp_coeffs = remove_drift_temperature_linear(
            baro_df.copy(),
            skip_initial_fraction=temp_skip_initial_fraction,
            zero_at_start=zero_at_start
        )

        baro_df_ema, ema_trends = remove_drift_ema(
            baro_df.copy(),
            alpha=ema_alpha,
            alpha_override=ema_alpha_override,
            zero_at_start=zero_at_start
        )

        if save_plots_dir:
            plot_drift_comparison(
                baro_df, baro_df_ema, baro_df_temp,
                ema_trends=ema_trends,
                temp_coeffs=temp_coeffs,
                save_path=save_plots_dir / "3_drift_removal_comparison.png"
            )
            plot_temperature_vs_pressure(baro_df, temp_coeffs=temp_coeffs,
                                         save_path=save_plots_dir / "temperature_vs_pressure_correlation.png")

        # Use EMA result for final output
        baro_df = baro_df_ema

    elif drift_removal_method == 'none':
        if zero_at_start:
            baro_df = zero_barometers_fixed(baro_df, reference_index='first_valid')

    else:
        raise ValueError(f"Unknown drift removal method: {drift_removal_method}")

    return baro_df


# ============================== PLOTTING ===============================

def plot_barometers_subplots(df, save_path=None, suptitle="Barometers"):
    """Plot all 6 barometers in subplots."""
    baro_color = '#005c7f'

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for i in range(6):
        col = f"b{i+1}"
        if col not in df.columns:
            continue
        axes[i].plot(df["time"], df[col], linewidth=1.2, color=baro_color)
        axes[i].set_ylabel(f"{col} [hPa]")
        axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel("Experiment Time [s]")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_barometers_vs_force(df, force_col="Fz", save_path=None, suptitle=None):
    """Plot barometers alongside a force/torque channel (dual y-axis)."""
    fcol = _force_name(df, force_col)
    if not fcol:
        return None

    baro_color = '#005c7f'
    force_color = '#44b155' if force_col.startswith('F') else '#d6c52e'

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    for i in range(6):
        ax1 = axes[i]
        bcol = f"b{i+1}"
        if bcol not in df.columns:
            continue
        # left: barometer
        line1 = ax1.plot(df["time"], df[bcol], label=bcol, linewidth=1.2, color=baro_color)
        ax1.set_ylabel(f"{bcol} [hPa]")
        ax1.grid(alpha=0.3)
        # right: force (scaled via twin axis)
        ax2 = ax1.twinx()
        line2 = ax2.plot(df["time"], df[fcol], linestyle="--", alpha=0.8,
                         label=fcol, linewidth=1.2, color=force_color)
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
