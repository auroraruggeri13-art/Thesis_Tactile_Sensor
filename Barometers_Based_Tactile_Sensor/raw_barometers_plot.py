#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot barometer pressures vs time in a 3x2 subplot figure.
Optional:
- Plot temperature on a right axis if temperature columns exist.
- Run linear regression to estimate pressure-vs-temperature sensitivity.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ============================================================
# === USER SETTINGS ==========================================
# ============================================================
test_num = 51701
version_num = 5
file_name = f"{test_num}barometers_trial.txt"  # or f"barometers_trial{test_num}.txt"
directory_to_datasets = os.path.abspath(
    fr"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test {test_num} - sensor v{version_num}"
)
CSV_PATH = os.path.join(directory_to_datasets, file_name)

TITLE = "Barometer Pressures vs Time"
UNITS = "hPa"
TEMP_UNITS = "degC"
YLIM = None  # e.g. (980, 1020)
SAVE_FIG = True
DPI = 200

# Optional: quantitative temperature-pressure sensitivity
RUN_TEMP_PRESSURE_REGRESSION = False
REGRESSION_WARMUP_SECONDS = 1.0
REGRESSION_USE_ROBUST_FILTER = False
REGRESSION_MAD_THRESHOLD = 5.0
SAVE_REGRESSION_CSV = False

# Color palette
COLORS = ["#292f56", "#005c7f", "#008780", "#44b155", "#d6c52e", "#3a7a9e"]
# ============================================================


def compute_temp_pressure_sensitivity(df_in, time_s, pressure_cols, temp_cols):
    """Compute per-channel pressure-vs-temperature sensitivity via linear regression."""
    dt_median = np.nanmedian(np.diff(time_s))
    warmup_samples = int(REGRESSION_WARMUP_SECONDS / max(dt_median, 1e-9))
    warmup_samples = min(max(warmup_samples, 0), len(df_in) - 1)

    results = []
    for p_col in pressure_cols:
        t_col = temp_cols.get(p_col)
        if t_col is None:
            continue

        p = pd.to_numeric(df_in[p_col], errors="coerce").to_numpy()[warmup_samples:]
        temp = pd.to_numeric(df_in[t_col], errors="coerce").to_numpy()[warmup_samples:]
        valid = np.isfinite(p) & np.isfinite(temp)

        if REGRESSION_USE_ROBUST_FILTER and valid.any():
            # Reject likely contact spikes by thresholding pressure derivative with MAD.
            dp = np.diff(p, prepend=p[0])
            med = np.nanmedian(dp[valid])
            mad = np.nanmedian(np.abs(dp[valid] - med)) + 1e-12
            robust_sigma = 1.4826 * mad
            keep = np.abs(dp - med) < (REGRESSION_MAD_THRESHOLD * robust_sigma)
            valid = valid & keep

        if valid.sum() < 10:
            continue

        slope_hpa_per_degC, intercept_hpa = np.polyfit(temp[valid], p[valid], 1)
        pred = slope_hpa_per_degC * temp[valid] + intercept_hpa
        sse = np.nansum((p[valid] - pred) ** 2)
        sst = np.nansum((p[valid] - np.nanmean(p[valid])) ** 2)
        r2 = 1.0 - (sse / (sst + 1e-12))

        results.append(
            {
                "channel": p_col,
                "temperature_col": t_col,
                "n_samples": int(valid.sum()),
                "slope_hPa_per_degC": float(slope_hpa_per_degC),
                "slope_Pa_per_degC": float(100.0 * slope_hpa_per_degC),
                "intercept_hPa": float(intercept_hpa),
                "R2": float(r2),
            }
        )

    return pd.DataFrame(results)


# Load file (auto-detect delimiter)
df = pd.read_csv(CSV_PATH, sep=None, engine="python")

# Time handling (use Epoch_s -> convert to elapsed seconds)
if "Epoch_s" not in df.columns:
    raise ValueError("File has no 'Epoch_s' column - cannot extract time.")

t = df["Epoch_s"].astype(float).to_numpy()
t = t - t[0]  # convert to elapsed seconds

# Pressure columns (detect both old: b1..b6 and new: b1_P..b6_P)
pressure_cols = []
temp_cols = {}

for i in range(1, 7):
    if f"b{i}" in df.columns:
        pressure_cols.append(f"b{i}")
    elif f"b{i}_P" in df.columns:
        pressure_cols.append(f"b{i}_P")
    else:
        raise ValueError(f"No pressure column found for sensor b{i}")

    # Temperature (optional): supports both raw format and renamed format
    if f"b{i}_T" in df.columns:
        temp_cols[pressure_cols[-1]] = f"b{i}_T"
    elif f"t{i}" in df.columns:
        temp_cols[pressure_cols[-1]] = f"t{i}"

# Plot style
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["grid.linewidth"] = 0.3
plt.rcParams["grid.alpha"] = 0.4
plt.rcParams["grid.color"] = "#999999"
plt.rcParams["font.size"] = 15

# Create plot
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, facecolor="white")
axes = axes.ravel()

for i, col in enumerate(pressure_cols):
    y = pd.to_numeric(df[col], errors="coerce").to_numpy()

    ax = axes[i]
    ax.set_facecolor("white")
    ax.plot(t, y, linewidth=1.0, color=COLORS[0], label="Pressure")
    ax.set_ylabel(f"{col} [{UNITS}]", color=COLORS[0])
    ax.tick_params(axis="y", labelcolor=COLORS[0])
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="y")

    if YLIM:
        ax.set_ylim(*YLIM)

    # Optional temperature plot on right axis
    if col in temp_cols:
        temp_col = temp_cols[col]
        temp_y = pd.to_numeric(df[temp_col], errors="coerce").to_numpy()
        ax2 = ax.twinx()
        ax2.plot(t, temp_y, color=COLORS[3], linewidth=2.0, alpha=0.9, label="Temperature")
        ax2.set_ylabel(f"{temp_col} [{TEMP_UNITS}]", color=COLORS[3])
        ax2.tick_params(axis="y", labelcolor=COLORS[3])

axes[-2].set_xlabel("Time [s]")
axes[-1].set_xlabel("Time [s]")

fig.suptitle(TITLE)
fig.tight_layout()

if SAVE_FIG:
    out_path = os.path.splitext(CSV_PATH)[0] + "_6subplots.png"
    fig.savefig(out_path, dpi=DPI)
    print(f"Saved combined plot to {out_path}")

plt.show()

if RUN_TEMP_PRESSURE_REGRESSION:
    if len(temp_cols) == 0:
        print("\n[Temp-Pressure Regression] No temperature columns found. Skipping.")
    else:
        reg_df = compute_temp_pressure_sensitivity(df, t, pressure_cols, temp_cols)
        if reg_df.empty:
            print("\n[Temp-Pressure Regression] Not enough valid samples per channel after filtering.")
        else:
            print("\n=== Temperature-Pressure Sensitivity (Linear Regression) ===")
            print(
                reg_df[
                    [
                        "channel",
                        "slope_hPa_per_degC",
                        "slope_Pa_per_degC",
                        "R2",
                        "n_samples",
                    ]
                ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
            )

            mean_sens = reg_df["slope_Pa_per_degC"].mean()
            std_sens = reg_df["slope_Pa_per_degC"].std(ddof=1) if len(reg_df) > 1 else 0.0
            print(f"\nMean sensitivity: {mean_sens:.6f} +/- {std_sens:.6f} Pa/degC")

            if SAVE_REGRESSION_CSV:
                out_csv = os.path.splitext(CSV_PATH)[0] + "_temp_pressure_sensitivity.csv"
                reg_df.to_csv(out_csv, index=False)
                print(f"Saved regression table to {out_csv}")
