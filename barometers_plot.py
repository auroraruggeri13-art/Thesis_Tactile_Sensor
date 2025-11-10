#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot six barometer pressures vs time in a single 3x2 subplot figure.
Run directly from VS Code (no command-line arguments required).
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# === USER SETTINGS ==========================================
# ============================================================
file_name = "datalog_2025-11-06_14-35-45.csv"
CSV_PATH = os.path.join(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\Sensor-Logs", file_name)

TITLE = "Barometer Pressures vs Time"
UNITS = "hPa"        # hectopascals = millibars
YLIM = None          # e.g. (980, 1020) for typical atmospheric pressure in hPa
SAVE_FIG = True
DPI = 200
# ============================================================

def guess_time_column(df):
    candidates = [c for c in df.columns if re.search(r"(time|timestamp|ts)", str(c), re.IGNORECASE)]
    if candidates:
        return candidates[0]
    df.insert(0, "time_index", np.arange(len(df), dtype=float))
    return "time_index"

def to_elapsed_seconds(series):
    # try datetime first
    try:
        dt = pd.to_datetime(series, errors="raise", utc=True)
        ns = dt.view("int64")
        return (ns - ns.iloc[0]) / 1e9
    except Exception:
        pass
    # numeric fallback
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.95:
        x = numeric.fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)
        if np.isfinite(x[0]): x -= x[0]
        if np.nanmax(x) - np.nanmin(x) > 1e4:  # probably ms
            x /= 1000.0
        return x
    return np.arange(len(series), dtype=float)

def find_six_barometers(df):
    # --- normalize column names for reliable matching ---
    original_cols = list(df.columns)
    norm = [re.sub(r"\s+", " ", str(c).strip()) for c in original_cols]  # trim & collapse spaces
    # Build a mapping normalized -> original
    colmap = {n.lower(): o for n, o in zip(norm, original_cols)}

    wanted = []
    # Prefer exact "barometer 1..6" (case-insensitive, normalized spaces)
    for i in range(1, 7):
        key = f"barometer {i}".lower()
        if key in colmap:
            wanted.append(colmap[key])

    # If any are missing, try a broader pattern (p/pressure/baro/dps/sensor + index)
    if len(wanted) < 6:
        pattern = re.compile(r"^(?:baro(?:meter)?|pressure|press|p|sensor|dps)\s*([0-9]+)$", re.IGNORECASE)
        candidates = []
        for n, o in zip(norm, original_cols):
            m = pattern.search(n)
            if m:
                idx = int(m.group(1))
                if 0 <= idx <= 99:
                    candidates.append((idx, o))
        # sort by numeric index and add any not already included
        for _, o in sorted(candidates, key=lambda x: x[0]):
            if o not in wanted:
                wanted.append(o)
        wanted = wanted[:6]

    if len(wanted) != 6:
        raise ValueError(
            f"Could not find six barometer columns. Found: {wanted}. "
            "Make sure your CSV has columns like 'barometer 1'...'barometer 6', "
            "or adjust 'find_six_barometers' to your naming."
        )
    return wanted

# ============================================================
# === MAIN PLOTTING SECTION =================================
# ============================================================

df = pd.read_csv(CSV_PATH, sep=None, engine="python")

time_col = guess_time_column(df)
t = to_elapsed_seconds(df[time_col])
pres_cols = find_six_barometers(df)

print(f"Detected time column: {time_col}")
print(f"Detected pressure columns: {pres_cols}")

fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
axes = axes.ravel()

for i, col in enumerate(pres_cols):
    # Convert raw values to actual pressure values
    y_raw = pd.to_numeric(df[col], errors="coerce").to_numpy()
    y = y_raw  # Use actual values without scaling
    
    ax = axes[i]
    ax.plot(t, y, linewidth=1.0)
    
    # Just show sensor name and units
    ax.set_ylabel(f"{col}\n[{UNITS}]")
    
    if YLIM:
        ax.set_ylim(*YLIM)
    
    # Show actual values on y-axis without any formatting
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_useOffset(False)  # Prevent offset notation
    ax.yaxis.get_major_formatter().set_scientific(False)  # Prevent scientific notation
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)

axes[-2].set_xlabel("Time [s]")
axes[-1].set_xlabel("Time [s]")
fig.suptitle(TITLE, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.97])

if SAVE_FIG:
    out_path = os.path.splitext(CSV_PATH)[0] + "_6subplots.png"
    fig.savefig(out_path, dpi=DPI)
    print(f"✅ Saved combined plot to {out_path}")

plt.show()
