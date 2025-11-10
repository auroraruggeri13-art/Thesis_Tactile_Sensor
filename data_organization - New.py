#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add barometer channels to an already-synchronized CSV (time + x,y + Fx..Tz),
then export the merged CSV (original destination) and a copy to a new folder.
Plots: same outputs & filenames as the previous script.

Author: Aurora Ruggeri
"""

import os
import re
from typing import List
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# =========================
# ======  CONFIG  =========
# =========================
from pathlib import Path

test_num = 109
version_num = 1

# --- Input paths ---
# 1) Precombined CSV that already contains time + x,y + Fx..Tz
COMBINED_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data") / f"test {test_num} - sensor v{version_num}"
COMBINED_FILE = f"processing_test_{test_num}.csv"

# 2) Barometer CSV: automatically find csv file that starts with 'datalog_'
BARO_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data") / f"test {test_num} - sensor v{version_num}"
baro_files = list(BARO_DIR.glob("datalog_*.csv"))
if not baro_files:
    raise FileNotFoundError(f"No 'datalog_*.csv' found in: {BARO_DIR}")
BARO_FILENAME = baro_files[0].name

# --- Original output (keep as-is: same folder + same plot names) ---
OUTPUT_DIR  = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data") / f"test {test_num} - sensor v{version_num}"
OUTPUT_NAME = f"synchronized_events_{test_num}.csv"

# --- Path plot rectangle crop (mm), same as before ---
PATH_RECT_WIDTH_MM  = 45.0
PATH_RECT_HEIGHT_MM = 20.0
PATH_RECT_CENTER_X_MM = 0.0
PATH_RECT_CENTER_Y_MM = 0.0

# =========================
# ======  READERS  ========
# =========================

def read_barometer_csv(path: str) -> pd.DataFrame:
    """
    Read Arduino/MCU barometer CSV: 'Timestamp' plus 6 channels (e.g. 'barometer 1'..'barometer 6').
    The date is taken from the filename 'datalog_YYYY-MM-DD_HH-MM-SS.csv'.
    Returns DF indexed by epoch ns ('time_ns', UTC), columns: b1..b6 (hPa).
    """
    LOCAL_TZ = 'America/New_York'

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    ts_col = next((c for c in df.columns if c.lower().startswith('timestamp')), None)
    if ts_col is None:
        raise ValueError("Barometer CSV must have a 'Timestamp' column (HH:MM:SS.sss).")

    m = re.search(r'datalog_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}', os.path.basename(path))
    if not m:
        raise ValueError("Filename must contain date like 'datalog_YYYY-MM-DD_hh-mm-ss.csv'")
    date_str = m.group(1)

    dt_local = pd.to_datetime(
        date_str + ' ' + df[ts_col].astype(str),
        format='%Y-%m-%d %H:%M:%S.%f',
        errors='coerce'
    ).dt.tz_localize(LOCAL_TZ, nonexistent='NaT', ambiguous='NaT')
    dt_utc = dt_local.dt.tz_convert('UTC')

    df = df.loc[~dt_utc.isna()].copy()
    df['time_ns'] = dt_utc.astype(np.int64)
    df = df.drop_duplicates(subset=['time_ns']).sort_values('time_ns')

    # Map pressure columns to b1..b6
    bcols = []
    for i in range(1,7):
        if f'barometer {i}' in df.columns: bcols.append(f'barometer {i}')
        elif f'b{i}' in df.columns:        bcols.append(f'b{i}')
    if len(bcols) != 6:
        cand = [c for c in df.columns if 'baro' in c.lower()]
        if len(cand) >= 6: bcols = cand[:6]
        else: raise ValueError(f"Could not find 6 barometer columns, found: {bcols or cand}")

    out = df.set_index('time_ns')[[*bcols]].copy()
    out.index.name = 'time_ns'
    out.columns = [f"b{i}" for i in range(1,7)]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

# =========================
# ======  PLOTTING  =======
# =========================

def plot_barometers(df: pd.DataFrame, out_dir: str, filename: str = "plot_barometers.png"):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for i in range(6):
        ax = axes[i]
        col = f"b{i+1}"
        if col in df.columns:
            mask = df['t'].notna() & df[col].notna()
            if mask.any():
                ax.plot(df.loc[mask, 't'].values, df.loc[mask, col].values, linewidth=1.0)
                ax.set_ylabel(col + " (hPa)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Barometer pressures vs time")
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    p = Path(out_dir) / filename
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

def plot_wrench(df: pd.DataFrame, out_dir: str, filename: str = "plot_wrench.png"):
    force_cols = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
    labels = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (N⋅m)", "Ty (N⋅m)", "Tz (N⋅m)"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for i, col in enumerate(force_cols):
        ax = axes[i]
        if col in df.columns:
            mask = df['t'].notna() & df[col].notna()
            if mask.any():
                ax.plot(df.loc[mask, 't'].values, df.loc[mask, col].values, linewidth=1.0)
                ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
    fig.suptitle("Force/Torque vs time")
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    p = Path(out_dir) / filename
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

def plot_b1_and_fz(df: pd.DataFrame, out_dir: str, filename: str = "plot_b1_vs_fz.png"):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    if 'b1' in df.columns:
        mask1 = df['t'].notna() & df['b1'].notna()
        if mask1.any():
            ax1.plot(df.loc[mask1, 't'].values, df.loc[mask1, 'b1'].values, linewidth=1.0, label='b1 (hPa)')
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("b1 (hPa)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if 'fz' in df.columns:
        mask2 = df['t'].notna() & df['fz'].notna()
        if mask2.any():
            ax2.plot(df.loc[mask2, 't'].values, df.loc[mask2, 'fz'].values, linewidth=2.0, linestyle='-', label='fz (N)')
    ax2.set_ylabel("fz (N)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Barometer 1 vs Fz")
    fig.tight_layout()
    p = Path(out_dir) / filename
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

def plot_distributions(df: pd.DataFrame, out_dir: str):
    """
    Plot distribution histograms for each variable and save them in a distributions folder.
    """
    try:
        # Create distributions folder if it doesn't exist
        dist_dir = os.path.join(out_dir, 'distributions')
        os.makedirs(dist_dir, exist_ok=True)
        
        # Variables to plot
        barometer_cols = [f'b{i}' for i in range(1,7)]
        force_cols = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
        position_cols = ['-x (mm)', '-y (mm)']
    
            # Plot barometers distributions
        if any(col in df.columns for col in barometer_cols):
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(barometer_cols, 1):
                if col in df.columns and not df[col].isna().all():
                    plt.subplot(2, 3, i)
                    df[col].dropna().hist(bins=50, density=True)
                    plt.title(f'{col} Distribution')
                    plt.xlabel('Pressure (hPa)')
                    plt.ylabel('Density')
                    plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, f'barometers_distribution_{test_num}.png'))
            plt.close()
            print("Saved barometers distribution plot")
        
        # Plot force/torque distributions
        if any(col in df.columns for col in force_cols):
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(force_cols, 1):
                if col in df.columns and not df[col].isna().all():
                    plt.subplot(2, 3, i)
                    df[col].dropna().hist(bins=50, density=True)
                    plt.title(f'{col} Distribution')
                    plt.xlabel('Force (N)' if col in ['fx', 'fy', 'fz'] else 'Torque (N⋅m)')
                    plt.ylabel('Density')
                    plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, f'forces_distribution_{test_num}.png'))
            plt.close()
            print("Saved forces distribution plot")
        
        # Plot position distributions
        if any(col in df.columns for col in position_cols):
            plt.figure(figsize=(10, 5))
            for i, col in enumerate(position_cols, 1):
                if col in df.columns and not df[col].isna().all():
                    plt.subplot(1, 2, i)
                    df[col].dropna().hist(bins=50, density=True)
                    plt.title(f'{col} Distribution')
                    plt.xlabel('Position (mm)')
                    plt.ylabel('Density')
                    plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, f'positions_distribution_{test_num}.png'))
            plt.close()
            print("Saved positions distribution plot")
        
        print(f"\nAll distribution plots saved in: {dist_dir}")
        
    except Exception as e:
        print(f"Error generating distribution plots: {str(e)}")
        # Make sure to close any open figures in case of error
        plt.close('all')

def plot_path(df: pd.DataFrame, out_dir: str, filename: str = "plot_path.png"):
    """
    Plot the path using -x (mm) and -y (mm) columns
    """
    xcol = '-x (mm)'
    ycol = '-y (mm)'
    
    if xcol not in df.columns or ycol not in df.columns:
        print("Skipping path plot (missing -x (mm) or -y (mm) columns).")
        return

    x_vals_mm = df[xcol].values
    y_vals_mm = df[ycol].values
    xlabel = xcol
    ylabel = ycol

    mask = np.isfinite(x_vals_mm) & np.isfinite(y_vals_mm)
    if not mask.any():
        print("Skipping path plot (x/y all NaN).")
        return

    cx_center = PATH_RECT_CENTER_X_MM
    cy_center = PATH_RECT_CENTER_Y_MM
    half_w = PATH_RECT_WIDTH_MM / 2.0
    half_h = PATH_RECT_HEIGHT_MM / 2.0
    inside = (x_vals_mm >= (cx_center - half_w)) & (x_vals_mm <= (cx_center + half_w)) \
           & (y_vals_mm >= (cy_center - half_h)) & (y_vals_mm <= (cy_center + half_h))
    if not inside.any():
        print("Skipping path plot (no points inside rectangle).")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_vals_mm[inside], y_vals_mm[inside], linewidth=1.5)
    rect = Rectangle((cx_center-half_w, cy_center-half_h),
                     PATH_RECT_WIDTH_MM, PATH_RECT_HEIGHT_MM,
                     edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.plot(x_vals_mm[inside][0],  y_vals_mm[inside][0],  marker='o', color='green', label='start')
    ax.plot(x_vals_mm[inside][-1], y_vals_mm[inside][-1], marker='o', color='red',   label='end')

    ax.set_xlim(cx_center-half_w, cx_center+half_w)
    ax.set_ylim(cy_center-half_h, cy_center+half_h)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Contact XY path')
    ax.grid(True, alpha=0.3)
    p = Path(out_dir) / filename
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

# =========================
# ======  CORE  ===========
# =========================

def main():
    combined_path = os.path.join(COMBINED_DIR, COMBINED_FILE)
    baro_path     = os.path.join(BARO_DIR, BARO_FILENAME)

    # --- Read combined CSV (already synchronized positions & wrench) ---
    comb = pd.read_csv(combined_path)
    comb.columns = [c.strip() for c in comb.columns]
    
    # Print the first timestamp in YYYY-MM-DD-HH-MM-SS format from processing file (Boston time)
    if 'time' in comb.columns and len(comb) > 0:
        first_time = pd.to_datetime(comb['time'].iloc[0], unit='s', utc=True)
        boston_time = first_time.tz_convert('America/New_York')
        formatted_time = boston_time.strftime('%Y-%m-%d-%H-%M-%S')
        print(f"First timestamp in processing_test_{test_num}.csv: {formatted_time}")

    # sanity checks (unchanged)
    force_cols = {'time': 'time', 'x_position': 'x_position', 'y_position': 'y_position'}
    wrench_cols = [('Fx', 'fx'), ('Fy', 'fy'), ('Fz', 'fz'), ('Tx', 'tx'), ('Ty', 'ty'), ('Tz', 'tz')]
    missing = []
    for req, _ in force_cols.items():
        if req not in comb.columns:
            missing.append(req)
    for upper, lower in wrench_cols:
        if upper not in comb.columns and lower not in comb.columns:
            missing.append(f"{upper} or {lower}")
    if missing:
        raise ValueError(f"Combined CSV missing required columns: {missing}")

    # 'time' -> epoch ns index
    time_ns = (pd.to_numeric(comb['time'], errors='coerce') * 1e9).astype('int64')
    comb = comb.assign(time_ns=time_ns).set_index('time_ns').sort_index()

    # --- Read barometers (epoch ns index, b1..b6) ---
    baro = read_barometer_csv(baro_path)
    if baro.empty or comb.empty:
        raise ValueError("One of the streams is empty; cannot compute overlap.")

    # --- Compute strict overlap window in epoch ns ---
    t_start = max(int(comb.index.min()), int(baro.index.min()))
    t_end   = min(int(comb.index.max()), int(baro.index.max()))
    
    # Calculate and print overlap information
    overlap_duration = (t_end - t_start) / 1e9  # Convert nanoseconds to seconds
    if t_end <= t_start:
        print("No temporal overlap between processing and barometer data")
        raise ValueError(
            "No temporal overlap between combined and barometer data.\n"
            f"combined: [{int(comb.index.min())}, {int(comb.index.max())}], "
            f"baro: [{int(baro.index.min())}, {int(baro.index.max())}]"
        )
    else:
        print(f"Found overlap between processing and barometer data: {overlap_duration:.2f} seconds")
        # Convert start time to HH-MM-SS in Boston time
        start_time = pd.to_datetime(t_start, unit='ns', utc=True).tz_convert('America/New_York')
        end_time = pd.to_datetime(t_end, unit='ns', utc=True).tz_convert('America/New_York')
        start_time_str = start_time.strftime('%H-%M-%S')
        end_time_str = end_time.strftime('%H-%M-%S')
        print(f"Overlap period: {start_time_str} to {end_time_str}")

    # Restrict both streams to the overlap; this guarantees all saved rows overlap
    comb = comb.loc[t_start:t_end].copy()
    baro_seg = baro.loc[t_start:t_end].copy()

    # --- Interpolate barometers onto (overlap-restricted) combined timeline ---
    ti = comb.index.values.astype(np.int64)
    out = pd.DataFrame(index=comb.index)
    si = baro_seg.index.values.astype(np.int64)
    for i in range(1, 7):
        col = f"b{i}"
        out[col] = np.interp(ti, si, baro_seg[col].values.astype(float))

    # --- Build final DataFrame (same as before, but t0 from the overlapped window) ---
    t0 = int(comb.index.min())
    t_rel = (comb.index.values.astype(np.int64) - t0) / 1e9
    
    final = pd.DataFrame(index=comb.index)
    final['t'] = t_rel

    for i in range(1, 7):
        final[f'b{i}'] = out[f'b{i}'].values

    if 'x_position' in comb.columns and 'y_position' in comb.columns:
        final['-x (mm)'] = -comb['x_position'].values  # Values are already in mm
        final['-y (mm)'] = -comb['y_position'].values  # Values are already in mm
    else:
        final['-x (mm)'] = np.nan
        final['-y (mm)'] = np.nan

    force_cols_map = {'Fx': 'fx', 'Fy': 'fy', 'Fz': 'fz', 'Tx': 'tx', 'Ty': 'ty', 'Tz': 'tz'}
    for old_col, new_col in force_cols_map.items():
        if old_col in comb.columns:
            final[new_col] = comb[old_col].values
        elif new_col in comb.columns:
            final[new_col] = comb[new_col].values
        else:
            final[new_col] = np.nan

    # --- Keep ONLY rows where all 6 barometers are valid (strict overlap) ---
    baro_cols = [f"b{i}" for i in range(1, 7)]
    final = final[final[baro_cols].notna().all(axis=1)].copy()

    # Save to the OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path_main = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    try:
        # Keep index (time_ns) labeled for traceability
        final.to_csv(out_path_main, index_label='time_ns')
        print(f"Saved merged CSV: {out_path_main}")
    except Exception as e:
        print(f"Failed to save CSV to {out_path_main}: {e}")

    # --- Plots (same names + destination = OUTPUT_DIR) ---
    plot_barometers(final, OUTPUT_DIR, f"plot_barometers_{test_num}.png")
    plot_wrench(final,     OUTPUT_DIR, f"plot_ATI_{test_num}.png")
    # Only draw overlay if b1 and Fz/fz present
    if 'b1' in final.columns and (('Fz' in final.columns) or ('fz' in final.columns)):
        plot_b1_and_fz(final, OUTPUT_DIR, f"plot_b1_vs_fz_{test_num}.png")
    else:
        print("Skipping b1 vs Fz overlay (missing columns).")
    # Path plot (will fall back to x_position/y_position)
    plot_path(final, OUTPUT_DIR, f"plot_path_{test_num}.png")

    # Quick summary
    t_series = final['t'].dropna()
    span = (t_series.iloc[-1] - t_series.iloc[0]) if len(t_series) else np.nan
    
    # Calculate sampling rate
    if len(t_series) > 1:
        # Calculate time differences between consecutive samples
        dt = np.diff(t_series)
        mean_dt = np.mean(dt)
        std_dt = np.std(dt)
        sampling_rate = 1.0 / mean_dt
        print(f"\nSampling Rate Analysis:")
        print(f"Average sampling rate: {sampling_rate:.2f} Hz")
        print(f"Average time between samples: {mean_dt*1000:.2f} ms ± {std_dt*1000:.2f} ms")
    
    # Print force ranges
    print("\nForce Ranges:")
    for force in ['fx', 'fy', 'fz']:
        if force in final.columns:
            force_data = final[force].dropna()
            if len(force_data) > 0:
                print(f"{force.upper()}:")
                print(f"  Min: {force_data.min():.2f} N")
                print(f"  Max: {force_data.max():.2f} N")
                print(f"  Range: {force_data.max() - force_data.min():.2f} N")
    
    print(f"\nRows: {len(final)}, time span: {float(span):.3f} s")
    if final[[f"b{i}" for i in range(1,7)]].isna().all().all():
        print("Note: all barometer values are NaN (no overlap or wrong barometer file).")
        
    # Generate distribution plots
    plot_distributions(final, OUTPUT_DIR)

if __name__ == "__main__":
    pd.set_option('display.width', 140)
    main()
