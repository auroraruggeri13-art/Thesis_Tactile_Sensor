#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
TEST_NUM    = 4105
VERSION_NUM = 4

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data")
DATA_DIR = BASE_DIR / f"test {TEST_NUM} - sensor v{VERSION_NUM}"

PROCESSING_FILE = DATA_DIR / f"processing_test_{TEST_NUM}.csv"
BAROMETER_FILE  = DATA_DIR / "datalog_2025-11-21_13-28-01.csv"

ASOF_DIRECTION = "nearest"
ASOF_TOLERANCE_S = 0.05

# --- Manual time shift (seconds) ---
# Positive value shifts the barometer data FORWARD in time (delays it)
# Negative value shifts it backward (makes it earlier)
BARO_TIME_SHIFT_S = 0.00000234

SAVE_FIG = False
DPI = 200

# =============================================================================
# HELPERS
# =============================================================================
def ensure_seconds(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = s.dropna().abs().median()
    if med > 1e12: s *= 1e-9
    elif med > 1e9: s *= 1e-6
    elif med > 1e6: s *= 1e-3
    return s

def load_processing(path: Path):
    df = pd.read_csv(path)
    df.rename(columns={"time": "t"}, inplace=True)
    df["t"] = ensure_seconds(df["t"])
    df = df.sort_values("t", ignore_index=True)
    return df[["t","Fx","Fy","Fz"]]

def load_barometer(path: Path):
    df = pd.read_csv(path)
    df.rename(columns={"Epoch_s": "t"}, inplace=True)
    df["t"] = ensure_seconds(df["t"])
    # Apply manual time shift
    df["t"] = df["t"] + BARO_TIME_SHIFT_S
    df = df.sort_values("t", ignore_index=True)
    baro_cols = [c for c in df.columns if "barometer" in c.lower()]
    return df[["t"] + baro_cols]

# =============================================================================
# MAIN
# =============================================================================
def main():
    df_proc = load_processing(PROCESSING_FILE)
    df_baro = load_barometer(BAROMETER_FILE)

    merged = pd.merge_asof(
        df_proc.sort_values("t"),
        df_baro.sort_values("t"),
        on="t",
        direction=ASOF_DIRECTION,
        tolerance=ASOF_TOLERANCE_S
    )

    t_rel = merged["t"] - merged["t"].min()

    # --- Plot forces and barometers together ---
    fig, ax1 = plt.subplots(figsize=(12,6))

    # Left axis: barometers
    baro_cols = [c for c in merged.columns if "barometer" in c.lower()]
    for c in baro_cols:
        ax1.plot(t_rel, merged[c], label=c, alpha=0.7)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Pressure [hPa]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Right axis: forces (scaled to hPa range for visualization)
    ax2 = ax1.twinx()
    fz_scale = (merged[baro_cols].mean().mean() / merged[["Fx","Fy","Fz"]].abs().max().max())
    for c in ["Fx","Fy","Fz"]:
        ax2.plot(t_rel, merged[c]*fz_scale, label=c, linestyle="--")
    ax2.set_ylabel("Force [N] (scaled)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Unified legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, ncol=3, fontsize=9)

    plt.title(f"Test {TEST_NUM} - Forces and Barometer Pressures (Aligned, shift={BARO_TIME_SHIFT_S:+.3f}s)")
    plt.tight_layout()

    if SAVE_FIG:
        out = DATA_DIR / f"forces_barometers_sameplot_{TEST_NUM}.png"
        plt.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"[Saved] {out}")

    plt.show()

if __name__ == "__main__":
    main()
