#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot ATI Force/Torque logs (TXT/CSV)
------------------------------------
- Loads one or more ATI logs exported with columns like:
  %time, field.wrench.force.{x,y,z}, field.wrench.torque.{x,y,z}
- Converts time to seconds relative to the start
- Plots force magnitude |F| over time for each file
- Plots Fx, Fy, Fz over time for a selected file
- Writes a small summary CSV of means/stds and sampling rate

"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import apply_plot_style
apply_plot_style()

# ===========================
# ===== USER SETTINGS =======
# ===========================
# Set your data directory and filenames here:
DATA_DIR = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\ATI_calibration_files"  
FILES = [
    r"ati_middle_trial10.txt",   # unloaded
    r"ati_middle_trial11.txt",   # +100 g with rotations
    r"ati_middle_trial12.txt"   # +200 g with rotations
]
# Choose which file to use for the Fx/Fy/Fz component plot (index in FILES list)
COMPONENTS_FILE_INDEX = 1  # 0 for first file, 1 for second, etc.
# Whether to save outputs as PNG/CSV next to the data files
SAVE_OUTPUTS = True

# ===========================
# ======  HELPERS  ==========
# ===========================
def load_ati_txt(path: Path) -> pd.DataFrame:
    """
    Robust loader for ATI TXT/CSV logs exported by ROS/bridge-style tools.
    We expect columns like:
      %time, field.wrench.force.x, field.wrench.force.y, field.wrench.force.z,
             field.wrench.torque.x, field.wrench.torque.y, field.wrench.torque.z
    """
    # Let pandas detect delimiter (comma or tab)
    df = pd.read_csv(path)
    # Standardize column names we need
    rename = {
        "%time": "time_ns",
        "field.wrench.force.x": "Fx",
        "field.wrench.force.y": "Fy",
        "field.wrench.force.z": "Fz",
        "field.wrench.torque.x": "Tx",
        "field.wrench.torque.y": "Ty",
        "field.wrench.torque.z": "Tz",
    }
    have = {k: v for k, v in rename.items() if k in df.columns}
    if not have:
        raise ValueError(f"Could not find ATI columns in file: {path.name}")
    df = df.rename(columns=have)

    # Time to seconds relative
    if "time_ns" in df.columns:
        t0 = df["time_ns"].iloc[0]
        df["t"] = (df["time_ns"] - t0) * 1e-9
    else:
        # Fall back to index-based time (assume ~100 Hz)
        df["t"] = np.arange(len(df)) * 0.01

    # Force magnitude
    for c in ["Fx","Fy","Fz","Tx","Ty","Tz"]:
        if c not in df.columns:
            df[c] = 0.0
    df["Fmag"] = np.sqrt(df["Fx"]**2 + df["Fy"]**2 + df["Fz"]**2)
    return df

def approx_sample_rate(df: pd.DataFrame) -> float:
    dt = np.diff(df["t"].values)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if len(dt) == 0:
        return float("nan")
    return float(1.0 / np.median(dt))

def summarize_df(df: pd.DataFrame, label: str) -> dict:
    return {
        "file": label,
        "samples": int(len(df)),
        "duration_s": float(df["t"].iloc[-1] - df["t"].iloc[0]),
        "sample_rate_Hz (approx)": round(approx_sample_rate(df), 2),
        "Fx_mean_N": round(float(df["Fx"].mean()), 6),
        "Fy_mean_N": round(float(df["Fy"].mean()), 6),
        "Fz_mean_N": round(float(df["Fz"].mean()), 6),
        "Fmag_mean_N": round(float(df["Fmag"].mean()), 6),
        "Fmag_std_N": round(float(df["Fmag"].std()), 6),
    }

# ===========================
# ========  MAIN  ===========
# ===========================
def main():
    data_dir = Path(DATA_DIR)
    files = [data_dir / f for f in FILES]
    dfs = []
    labels = []

    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")
        df = load_ati_txt(f)
        dfs.append(df)
        labels.append(f.name)

    # ---- Summary table
    rows = [summarize_df(df, lbl) for df, lbl in zip(dfs, labels)]
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # Save summary CSV next to first file
    if SAVE_OUTPUTS:
        out_csv = files[0].parent / "ati_force_summary.csv"
        summary.to_csv(out_csv, index=False)
        print(f"Saved summary CSV -> {out_csv}")

    # ---- Plot |F| for each file on one figure
    plt.figure()
    for df, lbl in zip(dfs, labels):
        plt.plot(df["t"], df["Fmag"], label=lbl)
    plt.xlabel("time [s]")
    plt.ylabel("Force magnitude |F| [N]")
    plt.title("ATI Force Magnitude vs Time")
    plt.legend()
    if SAVE_OUTPUTS:
        out_png = files[0].parent / "ati_force_magnitude.png"
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"Saved plot -> {out_png}")
    plt.show()

    # ---- Plot components for each file (Fx,Fy,Fz) and overlay |F| as red dotted line
    for i, (dfc, lblc) in enumerate(zip(dfs, labels)):
        plt.figure()
        plt.plot(dfc["t"], dfc["Fx"], label="Fx")
        plt.plot(dfc["t"], dfc["Fy"], label="Fy")
        plt.plot(dfc["t"], dfc["Fz"], label="Fz")
        # Overlay vectorial magnitude if present
        if "Fmag" in dfc.columns:
            plt.plot(dfc["t"], dfc["Fmag"], label="|F|", color='red', linestyle=':')
        plt.xlabel("time [s]")
        plt.ylabel("Force component [N]")
        plt.title(f"ATI Force Components vs Time — {lblc}")
        plt.legend()
        if SAVE_OUTPUTS:
            out_png2 = files[i].parent / f"{Path(lblc).stem}_components.png"
            plt.savefig(out_png2, dpi=200, bbox_inches="tight")
            print(f"Saved plot -> {out_png2}")
        plt.show()

if __name__ == "__main__":
    main()
