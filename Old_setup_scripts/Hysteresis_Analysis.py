#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hysteresis plots: Fz (x-axis) vs barometers B1..B6 (y-axis)

- Loads synchronized_events_{TEST_NUM}.csv from the standard test directory
- Creates 6 stacked subplots: B1..B6 vs Fz
- Saves the figure into the 'hysterisis' folder under the Thesis root
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# =========================== CONFIGURATION ===========================

TEST_NUM = 4350
VERSION_NUM = 4

# Smoothing parameters
SMOOTH_WINDOW = 51  # Window length (must be odd)
SMOOTH_POLY = 3     # Polynomial order

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data")
DATA_DIR = BASE_DIR / f"test {TEST_NUM} - sensor v{VERSION_NUM}"

# Input file (adjust name if needed)
CSV_FILE = DATA_DIR / f"synchronized_events_{TEST_NUM}.csv"

# Output folder for hysteresis plots
HYST_BASE = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab")
HYST_DIR = HYST_BASE / "hysterisis"
HYST_DIR.mkdir(parents=True, exist_ok=True)

# Output figure name
OUT_FIG = HYST_DIR / f"hysteresis_Fz_vs_Barometers_test{TEST_NUM}.png"

# ============================= LOAD DATA =============================

if not CSV_FILE.exists():
    raise FileNotFoundError(f"Could not find input file: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

# Expect columns named 'fz' and 'b1'..'b6'
required_cols = ["fz"] + [f"b{i}" for i in range(1, 7)]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# =============================== PLOTS ===============================

fz = df["fz"].values

# Apply smoothing to fz and barometer data
fz_smooth = savgol_filter(fz, SMOOTH_WINDOW, SMOOTH_POLY)

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)

for i, ax in enumerate(axes.flat, start=1):
    bi = df[f"b{i}"].values
    bi_smooth = savgol_filter(bi, SMOOTH_WINDOW, SMOOTH_POLY)
    
    # Calculate hysteresis using standard formula:
    # Hysteresis ratio (%) = (Δhyst / Full-scale output) × 100%
    # where:
    # - Δhyst = maximum difference between loading and unloading at same force
    # - Full-scale output = max output - min output
    
    # Split data into loading (ascending fz) and unloading (descending fz)
    midpoint = len(fz_smooth) // 2
    
    # Maximum hysteresis (Δhyst): max difference between curves
    delta_hyst = np.max(np.abs(bi_smooth[:midpoint] - bi_smooth[midpoint:][::-1][:midpoint]))
    
    # Full-scale output (FSO)
    full_scale_output = np.max(bi_smooth) - np.min(bi_smooth)
    
    # Hysteresis ratio (%)
    hysteresis_percent = (delta_hyst / full_scale_output * 100) if full_scale_output > 0 else 0
    
    ax.plot(-fz_smooth, bi_smooth, linewidth=2.5, color='red', alpha=0.7)
    ax.set_ylabel(f"B{i} Pressure", fontsize=10)
    ax.set_title(f"B{i} vs Fz", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.5, linestyle='--', linewidth=1)
    
    # Add hysteresis value in a text box
    textstr = f'Δhyst: {delta_hyst:.3f}\nHyst: {hysteresis_percent:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

# Add x-label to bottom row
for ax in axes[1, :]:
    ax.set_xlabel("Fz (N)", fontsize=10)

plt.suptitle(f"Hysteresis Analysis: Barometer Response vs Force (Test {TEST_NUM})", 
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
fig.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved hysteresis figure to: {OUT_FIG}")
