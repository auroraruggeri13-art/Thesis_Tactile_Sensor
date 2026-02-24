#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot barometer pressures vs time in a 3x2 subplot figure.
Optional: if temperature columns exist (e.g., b1_T), plot them on a right axis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ============================================================
# === USER SETTINGS ==========================================
# ============================================================
test_num = 51000
version_num = 5
file_name = f"{test_num}barometers_trial.txt" # or f"barometers_trial{test_num}.txt"
directory_to_datasets = os.path.abspath(
    fr"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test {test_num} - sensor v{version_num}"
)
CSV_PATH = os.path.join(directory_to_datasets, file_name)

TITLE = "Barometer Pressures vs Time"
UNITS = "hPa"
TEMP_UNITS = "°C"
YLIM = None     # e.g. (980,1020)
SAVE_FIG = True
DPI = 200
# ============================================================

# Load file (auto-detect delimiter)
df = pd.read_csv(CSV_PATH, sep=None, engine="python")

# Time handling (use Epoch_s → convert to elapsed seconds)
if "Epoch_s" not in df.columns:
    raise ValueError("File has no 'Epoch_s' column — cannot extract time.")

t = df["Epoch_s"].astype(float).to_numpy()
t = t - t[0]     # convert to elapsed seconds

# Pressure columns (detect both old: b1..b6 and new: b1_P..b6_P)
pressure_cols = []
temp_cols = {}

for i in range(1, 7):
    # Pressure
    if f"b{i}" in df.columns:
        pressure_cols.append(f"b{i}")
    elif f"b{i}_P" in df.columns:
        pressure_cols.append(f"b{i}_P")
    else:
        raise ValueError(f"No pressure column found for sensor b{i}")

    # Temperature (optional)
    if f"b{i}_T" in df.columns:
        temp_cols[pressure_cols[-1]] = f"b{i}_T"

# Create plot
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
axes = axes.ravel()

for i, col in enumerate(pressure_cols):
    y = pd.to_numeric(df[col], errors="coerce").to_numpy()

    ax = axes[i]
    ax.plot(t, y, linewidth=1.0, label="Pressure")
    ax.set_ylabel(f"{col} [{UNITS}]")
    ax.grid(True, alpha=0.3)
    
    # Format y-axis to show plain numbers instead of scientific notation
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='y')

    # Apply y-limits if defined
    if YLIM:
        ax.set_ylim(*YLIM)

    # === Optional temperature plot on right axis ===
    if col in temp_cols:
        temp_col = temp_cols[col]
        temp_y = pd.to_numeric(df[temp_col], errors="coerce").to_numpy()
        ax2 = ax.twinx()
        ax2.plot(t, temp_y, color="tab:red", linewidth=0.5, alpha=0.8, label="Temperature")
        ax2.set_ylabel(f"{temp_col} [{TEMP_UNITS}]", color="tab:red")
        ax2.tick_params(axis='y', labelcolor="tab:red")
        ax2.set_ylim(24, 25)

axes[-2].set_xlabel("Time [s]")
axes[-1].set_xlabel("Time [s]")

fig.suptitle(TITLE)
fig.tight_layout()

if SAVE_FIG:
    out_path = os.path.splitext(CSV_PATH)[0] + "_6subplots.png"
    fig.savefig(out_path, dpi=DPI)
    print(f"✅ Saved combined plot to {out_path}")

plt.show()
