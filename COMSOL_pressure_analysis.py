#!/usr/bin/env python3

import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab")
CSV_NAME = "COMSOL_simulation_data.csv"
PARENT = BASE_DIR / "COMSOL_plots"
PARENT.mkdir(parents=True, exist_ok=True)

MIN_POINTS = 3
DPI = 200

COLOR = {"d_mm": "g", "h_mm": "r", "z_mm": "b"}

def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-\.]+", "_", str(s)).strip("_")

def load_csv() -> pd.DataFrame:
    p = BASE_DIR / CSV_NAME
    if p.exists():
        return pd.read_csv(p)
    need = {"shape","d_mm","h_mm","z_mm","Fz_N","Fy_N","Fx_N","dP_hPa"}
    for c in BASE_DIR.glob("*.csv"):
        try:
            tmp = pd.read_csv(c, nrows=5)
            if need <= set(tmp.columns):
                print(f"[INFO] Using {c.name}")
                return pd.read_csv(c)
        except Exception:
            pass
    raise FileNotFoundError(f"No '{CSV_NAME}' (or compatible CSV) in {BASE_DIR}")

def plot_var(df: pd.DataFrame, var: str):
    other = sorted({"d_mm", "h_mm", "z_mm"} - {var})
    out = PARENT / f"plots_{var[0]}"
    out.mkdir(exist_ok=True)
    cols = ["shape", "Fz_N", "Fy_N", "Fx_N"] + other
    n = 0
    for keys, g in df.groupby(cols):
        if g[var].nunique() < MIN_POINTS:
            continue
        xy = (g[[var, "dP_hPa"]]
              .groupby(var, as_index=False)
              .mean()
              .sort_values(var))
        col = COLOR.get(var, "k")
        plt.figure(figsize=(5.2, 4.0))
        plt.plot(xy[var].values, xy["dP_hPa"].values, "-o", color=col)
        title = (f"{keys[0]} — ΔP vs {var}\n"
                 f"Fx={keys[3]} N, Fy={keys[2]} N, Fz={keys[1]} N, "
                 f"{other[0]}={keys[4]} mm, {other[1]}={keys[5]} mm")
        plt.title(title)
        plt.xlabel({"d_mm": "d [mm]", "h_mm": "h [mm]", "z_mm": "z [mm]"}[var])
        plt.ylabel("ΔP [hPa]")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = (f"{keys[0]}__Fx{keys[3]}_Fy{keys[2]}_Fz{keys[1]}__"
                 f"{other[0]}{keys[4]}__{other[1]}{keys[5]}__vs_{var}.png")
        plt.savefig(out / safe_name(fname), dpi=DPI)
        plt.close()
        n += 1
    print(f"[DONE] Saved {n} plots to: {out}")

def plot_equal_forces_by_d(df: pd.DataFrame):
    out = PARENT / "plots_equal_forces"
    out.mkdir(exist_ok=True)

    # keep only rows where both forces are nonzero and equal (Fy=Fz)
    m = (df["Fy_N"] != 0) & (df["Fz_N"] != 0) & (df["Fy_N"] == df["Fz_N"])
    df2 = df.loc[m].copy()

    for c in ["Fy_N","Fz_N","d_mm","dP_hPa"]:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    SHAPE_COLS = {"Circle": "b", "Square": "g", "Triangle": "r"}

    # possible force levels to show on the x-axis
    FORCE_LEVELS = [-11, -5, -1]

    for dval in [1.8, 4.25]:
        g_d = df2[df2["d_mm"] == dval]
        if g_d.empty:
            continue

        plt.figure(figsize=(6, 4.2))
        nlines = 0

        for shape, gs in g_d.groupby("shape"):
            xy = (gs[["Fy_N", "dP_hPa"]]
                  .groupby("Fy_N", as_index=False)
                  .mean())

            # ensure all FORCE_LEVELS exist (fill missing with NaN)
            full = pd.DataFrame({"Fy_N": FORCE_LEVELS})
            xy = full.merge(xy, on="Fy_N", how="left").sort_values("Fy_N")

            plt.plot(xy["Fy_N"], xy["dP_hPa"], "-o",
                     label=shape, color=SHAPE_COLS.get(shape, "k"))
            nlines += 1

        if nlines == 0:
            plt.close()
            continue

        plt.title(f"ΔP vs Fy (Fy=Fz, both ≠ 0) — d={dval} mm")
        plt.xlabel("Fy [N]")
        plt.ylabel("ΔP [hPa]")
        plt.xticks(FORCE_LEVELS)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Shape")
        plt.tight_layout()

        fname = f"dp_vs_fy_equal_forces_d{dval}.png"
        plt.savefig(out / safe_name(fname), dpi=DPI)
        plt.close()

    print(f"[DONE] Saved plots to: {out}")

def main():
    df = load_csv()
    for c in ["d_mm","h_mm","z_mm","Fz_N","Fy_N","Fx_N","dP_hPa"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    plot_var(df, "d_mm")
    plot_var(df, "h_mm")
    plot_var(df, "z_mm")
    plot_equal_forces_by_d(df)

if __name__ == "__main__":
    main()
