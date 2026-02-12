#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Manual sync + hysteresis plots for material testing

- Loads barometer file (test_02.csv style: Time_ms,b1,t1,Time_s)
- Loads ATI file (ati_middle_trial02.txt: ROS-style wrench log)
- Applies a manual time shift to barometer timeline
- Synchronizes using merge_asof on time
- Plots:
    1) Pressure & Fz vs time on same plot (twin y-axes)
    2) Hysteresis: Fz (x) vs pressure (y)
- Saves plots and merged CSV in:
    C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\hysteresys material testing
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# =========================== CONFIGURATION ===========================

# --- Where to save results ---
BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab")
OUT_DIR = BASE_DIR / "hysteresys material testing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_NUM = '106'

# if TEST_NUM between '01' and '03', material = 'Dragon Skin 20', if '04' to '06' = 'Vytafle 20', if '07' to '09' = 'Ecoflex 00-30', if '10' to '12' = 'Mold Star 30'
MATERIAL_MAP = {
    '01': 'Dragon Skin 20',
    '02': 'Dragon Skin 20',
    '03': 'Dragon Skin 20',
    '103': 'Dragon Skin 20',
    '04': 'Vytaflex 20',
    '05': 'Vytaflex 20',
    '06': 'Vytaflex 20',
    '106': 'Vytaflex 20',
    '07': 'Ecoflex 00-30',
    '08': 'Ecoflex 00-30',
    '09': 'Ecoflex 00-30',
    '10': 'Mold Star 30',
    '11': 'Mold Star 30',
    '12': 'Mold Star 30',
}
MATERIAL = MATERIAL_MAP[TEST_NUM]

# --- Input files (EDIT THESE) ---
BARO_FILE = BASE_DIR / f"Sensor-Logs\\test_{TEST_NUM}.csv"
ATI_FILE  = BASE_DIR / f"Sensor-Logs\\ati_middle_trial{TEST_NUM}.txt"
# Manual time shift applied to BAROMETER data [seconds]
# Positive = barometer happens later in time
BARO_TIME_SHIFT = -1.12

# Max allowed difference between merged timestamps [seconds]
MERGE_TOLERANCE_S = 0.02

# Smoothing window size (number of points for moving average)
# Set to 1 to disable smoothing
SMOOTHING_WINDOW = 15

# Minimum force threshold [N] to include in hysteresis plot
# This removes noisy low-force data
FORCE_THRESHOLD_MIN = 0.05

# Peak detection parameters
PEAK_MIN_HEIGHT = 1.0  # Minimum force [N] to consider as a peak
PEAK_MIN_DISTANCE = 50  # Minimum distance between peaks [data points]
PEAK_MIN_PROMINENCE = 0.5  # Minimum prominence [N] to distinguish peaks

# Name tag for output files
TRIAL_TAG = f"trial{TEST_NUM}"




# ====================================================================


def load_barometer(path: Path, time_shift_s: float) -> pd.DataFrame:
    """
    Load barometer CSV with columns: Time_ms, b1, t1, Time_s
    Returns DataFrame with columns: ['t', 'pressure'] where t is in seconds.
    """
    df = pd.read_csv(path)
    if not {"Time_s", "b1"}.issubset(df.columns):
        raise ValueError(f"Barometer file missing required columns: {path}")

    df = df.copy()
    df["t"] = df["Time_s"] + float(time_shift_s)
    df.rename(columns={"b1": "pressure"}, inplace=True)
    return df[["t", "pressure"]]


def load_ati(path: Path) -> pd.DataFrame:
    """
    Load ATI wrench file with ROS-style columns, including:
    '%time', 'field.wrench.force.z', ...
    Returns DataFrame with columns: ['t', 'fz'] where t is in seconds from start.
    """
    df = pd.read_csv(path)

    # Check required columns
    if "%time" not in df.columns or "field.wrench.force.z" not in df.columns:
        raise ValueError(f"ATI file missing required columns: {path}")

    df = df.copy()
    # %time is in nanoseconds since epoch → convert to seconds-from-start
    df["t"] = (df["%time"] - df["%time"].iloc[0]) * 1e-9
    df.rename(columns={"field.wrench.force.z": "fz"}, inplace=True)

    return df[["t", "fz"]]


def sync_baro_ati(baro: pd.DataFrame, ati: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    """
    Merge barometer and ATI on 't' using nearest neighbor within tolerance_s.
    Returns DataFrame with columns: ['t', 'pressure', 'fz'].
    """
    baro_sorted = baro.sort_values("t")
    ati_sorted = ati.sort_values("t")

    merged = pd.merge_asof(
        baro_sorted,
        ati_sorted,
        on="t",
        direction="nearest",
        tolerance=tolerance_s,
    )

    # Drop rows where Fz was not matched
    merged = merged.dropna(subset=["fz"]).reset_index(drop=True)
    return merged


def smooth_data(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Apply moving average smoothing to pressure and fz columns.
    
    Args:
        df: DataFrame with 't', 'pressure', 'fz' columns
        window_size: Number of points for moving average window
    
    Returns:
        DataFrame with smoothed pressure and fz values
    """
    if window_size <= 1:
        return df
    
    df_smooth = df.copy()
    df_smooth['pressure'] = df['pressure'].rolling(window=window_size, center=True, min_periods=1).mean()
    df_smooth['fz'] = df['fz'].rolling(window=window_size, center=True, min_periods=1).mean()
    
    return df_smooth


def detect_cycles(df: pd.DataFrame, force_threshold: float = 0.0, 
                  peak_height: float = 1.0, peak_distance: int = 50, 
                  peak_prominence: float = 0.5):
    """
    Detect all force peaks and split data into individual loading-unloading cycles.
    
    Args:
        df: DataFrame with 't', 'pressure', 'fz' columns
        force_threshold: Minimum force to include
        peak_height: Minimum peak height [N]
        peak_distance: Minimum distance between peaks [data points]
        peak_prominence: Minimum prominence to distinguish peaks [N]
    
    Returns:
        List of DataFrames, each representing one cycle
    """
    df = df.copy()
    df['force'] = -df['fz']
    
    # Find all peaks in force data
    peaks, properties = find_peaks(df['force'].values, 
                                   height=peak_height,
                                   distance=peak_distance,
                                   prominence=peak_prominence)
    
    print(f"Found {len(peaks)} peaks in force data")
    
    if len(peaks) == 0:
        print("No peaks detected, returning empty list")
        return []
    
    # Split data into cycles around each peak
    cycles = []
    for i, peak_idx in enumerate(peaks):
        # Find local minima before and after peak
        # Search window: from previous peak to next peak (or start/end)
        start_search = peaks[i-1] if i > 0 else 0
        end_search = peaks[i+1] if i < len(peaks)-1 else len(df)
        
        # Find local minimum before peak
        before_data = df.iloc[start_search:peak_idx]
        if len(before_data) > 0:
            start_idx = before_data['force'].idxmin()
        else:
            start_idx = peak_idx
        
        # Find local minimum after peak
        after_data = df.iloc[peak_idx:end_search]
        if len(after_data) > 0:
            end_idx = after_data['force'].idxmin()
        else:
            end_idx = peak_idx
        
        # Extract cycle
        cycle = df.loc[start_idx:end_idx].copy()
        
        # Filter by threshold
        cycle = cycle[cycle['force'] >= force_threshold]
        
        if len(cycle) > 10:  # Ensure enough data points
            cycles.append(cycle)
            print(f"  Cycle {i+1}: {len(cycle)} points, peak force = {cycle['force'].max():.2f} N")
    
    return cycles


def calculate_hysteresis(df: pd.DataFrame, num_points: int = 100, force_threshold: float = 0.0):
    """
    Calculate pointwise hysteresis error as % of full scale for a single cycle.
    
    Args:
        df: DataFrame with 't', 'pressure', 'fz' columns (already a single cycle)
        num_points: Number of force levels for interpolation
        force_threshold: Minimum force to include in analysis [N]
    
    Returns:
        dict with hysteresis metrics
    """
    # Use -fz because plots use -fz (positive force)
    df = df.copy()
    if 'force' not in df.columns:
        df['force'] = -df['fz']
    
    # Filter by threshold if not already done
    df = df[df['force'] >= force_threshold]
    
    if len(df) < 10:
        return None
    
    # Find peak to split loading/unloading
    peak_idx = df['force'].idxmax()
    loading = df.loc[:peak_idx].copy()
    unloading = df.loc[peak_idx:].copy()

    
    # Sort by force for interpolation
    loading = loading.sort_values('force')
    unloading = unloading.sort_values('force')
    
    # Define common force levels for interpolation
    f_min = max(loading['force'].min(), unloading['force'].min())
    f_max = min(loading['force'].max(), unloading['force'].max())
    force_levels = np.linspace(f_min, f_max, num_points)
    
    # Interpolate pressure values at common force levels
    # Remove duplicates and ensure monotonic for interpolation
    loading_unique = loading.drop_duplicates(subset='force', keep='first')
    unloading_unique = unloading.drop_duplicates(subset='force', keep='first')
    
    if len(loading_unique) < 2 or len(unloading_unique) < 2:
        print("Warning: Not enough unique points for interpolation")
        return None
    
    try:
        interp_load = interp1d(loading_unique['force'], loading_unique['pressure'], 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_unload = interp1d(unloading_unique['force'], unloading_unique['pressure'], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        
        p_load = interp_load(force_levels)
        p_unload = interp_unload(force_levels)
        
        # Calculate hysteresis metrics
        p_min = df['pressure'].min()
        p_max = df['pressure'].max()
        full_scale = p_max - p_min
        
        if full_scale == 0:
            print("Warning: Full scale is zero, cannot calculate hysteresis")
            return None
        
        # Pointwise hysteresis as % of full scale
        H_pointwise = np.abs(p_load - p_unload) / full_scale * 100
        
        # Average and maximum hysteresis
        H_avg = np.mean(H_pointwise)
        H_max = np.max(H_pointwise)
        
        return {
            'H_avg': H_avg,
            'H_max': H_max,
            'force_levels': force_levels,
            'H_pointwise': H_pointwise,
            'p_load': p_load,
            'p_unload': p_unload,
            'loading_df': loading,
            'unloading_df': unloading,
            'p_min': p_min,
            'p_max': p_max
        }
    except Exception as e:
        print(f"Error during interpolation: {e}")
        return None


def plot_time_series(df: pd.DataFrame, out_path: Path):
    """
    Plot pressure and Fz vs time on twin y-axes and save to PNG.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Pressure [hPa]", fontsize=10)
    line1, = ax1.plot(df["t"], df["pressure"], label="Pressure", linewidth=2, color='#005c7f')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Fz [N]", fontsize=10)
    line2, = ax2.plot(df["t"], -df["fz"], label="Fz", linewidth=2.0, alpha=0.8, color='#44b155')

    # Build a combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=9)

    ax1.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.show()


def plot_hysteresis(df: pd.DataFrame, out_path: Path, material: str = MATERIAL, 
                   force_threshold: float = 0.0, TEST_NUM: str = TEST_NUM):
    """
    Plot all hysteresis cycles with individual and overall hysteresis metrics.
    """
    # Detect all cycles
    cycles = detect_cycles(df, force_threshold=force_threshold,
                          peak_height=PEAK_MIN_HEIGHT,
                          peak_distance=PEAK_MIN_DISTANCE,
                          peak_prominence=PEAK_MIN_PROMINENCE)
    
    if len(cycles) == 0:
        print("No cycles detected for hysteresis plot")
        return None
    
    # Calculate hysteresis for each cycle
    hyst_results = []
    for i, cycle in enumerate(cycles):
        hyst_data = calculate_hysteresis(cycle, force_threshold=force_threshold)
        if hyst_data is not None:
            hyst_results.append(hyst_data)
    
    if len(hyst_results) == 0:
        print("No valid hysteresis calculations")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Define colors for different cycles - one color per cycle
    palette_colors = ['#292f56', '#005c7f', '#008780', '#44b155', '#d6c52e']
    
    # Repeat colors if more cycles than colors
    cycle_colors = [palette_colors[i % len(palette_colors)] for i in range(len(hyst_results))]
    
    # Plot each cycle - use solid line for loading, dashed for unloading
    for i, hyst_data in enumerate(hyst_results):
        loading = hyst_data['loading_df']
        unloading = hyst_data['unloading_df']
        
        ax.plot(loading['force'], loading['pressure'], '-', 
                color=cycle_colors[i], linewidth=2, 
                label=f'Cycle {i+1} (loading)', alpha=0.8)
        ax.plot(unloading['force'], unloading['pressure'], '--', 
                color=cycle_colors[i], linewidth=2, 
                label=f'Cycle {i+1} (unloading)', alpha=0.8)
    
    # Calculate overall statistics
    H_avgs = [h['H_avg'] for h in hyst_results]
    H_avg_overall = np.mean(H_avgs)
    H_max_overall = np.max(H_avgs)
    
    # Create text box with individual and overall metrics
    textstr = f"Overall: $H_{{avg}}$ = {H_avg_overall:.2f}%FS, $H_{{max}}$ = {H_max_overall:.2f}%FS\n"
    textstr += "\n".join([f"Cycle {i+1}: {h['H_avg']:.2f}%FS" for i, h in enumerate(hyst_results)])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.set_xlabel("Fz [N]", fontsize=12)
    ax.set_ylabel("Pressure [hPa]", fontsize=12)
    ax.set_title(f"Hysteresis Cycles - {material} - {TEST_NUM}", fontsize=13)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.show()
    
    return {
        'cycles': hyst_results,
        'H_avg_overall': H_avg_overall,
        'H_max_overall': H_max_overall,
        'H_avgs': H_avgs
    }


def main():
    print("=== Loading data ===")
    print(f"Barometer file: {BARO_FILE}")
    print(f"ATI file      : {ATI_FILE}")

    baro = load_barometer(BARO_FILE, BARO_TIME_SHIFT)
    ati = load_ati(ATI_FILE)

    print("=== Synchronizing with manual time shift ===")
    merged = sync_baro_ati(baro, ati, MERGE_TOLERANCE_S)
    print(f"Merged rows: {len(merged)}")
    
    print(f"=== Smoothing data (window size: {SMOOTHING_WINDOW}) ===")
    merged = smooth_data(merged, SMOOTHING_WINDOW)
    print("Smoothing complete")

    # Save merged CSV
    merged_csv = OUT_DIR / f"merged_baro_ati_{TRIAL_TAG}.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"Saved merged data to: {merged_csv}")

    # Time series plot
    time_plot_path = OUT_DIR / f"time_pressure_fz_{TRIAL_TAG}_{MATERIAL_MAP[TEST_NUM]}.png"
    plot_time_series(merged, time_plot_path)
    print(f"Saved time series plot to: {time_plot_path}")

    # Hysteresis plot and calculation
    hyst_plot_path = OUT_DIR / f"hysteresis_pressure_vs_fz_{TRIAL_TAG}_{MATERIAL_MAP[TEST_NUM]}.png"
    hyst_data = plot_hysteresis(merged, hyst_plot_path, material=MATERIAL_MAP[TEST_NUM], 
                                force_threshold=FORCE_THRESHOLD_MIN, TEST_NUM=TEST_NUM)
    print(f"Saved hysteresis plot to: {hyst_plot_path}")
    
    # Print hysteresis metrics
    if hyst_data is not None:
        print("\n=== Hysteresis Analysis ===")
        print(f"Number of cycles detected: {len(hyst_data['cycles'])}")
        print(f"Overall average hysteresis: {hyst_data['H_avg_overall']:.2f}% FS")
        print(f"Maximum hysteresis (across all cycles): {hyst_data['H_max_overall']:.2f}% FS")
        print("\nIndividual cycles:")
        for i, h_avg in enumerate(hyst_data['H_avgs']):
            print(f"  Cycle {i+1}: {h_avg:.2f}% FS")
        if len(hyst_data['cycles']) > 0:
            print(f"\nFull scale range: {hyst_data['cycles'][0]['p_max'] - hyst_data['cycles'][0]['p_min']:.2f} hPa")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
