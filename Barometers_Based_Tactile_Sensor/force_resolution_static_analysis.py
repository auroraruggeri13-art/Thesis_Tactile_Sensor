#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force resolution analysis on quasi-static-load intervals.

Workflow:
1) Load raw synchronized experiment CSVs.
2) Build sliding-window features and run LightGBM model (v5.18) to generate predictions.
3) Identify quasi-static windows using a rolling-std threshold on ATI GT forces.
4) Compute prediction variability (sigma) within each quasi-static window.
5) Estimate force resolution as DeltaF_min = 3 * sigma, per axis.
6) Save per-segment and summary results.
"""

from __future__ import annotations

import sys
import pickle
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils.io_utils import load_tabular_csv
from utils.plot_utils import plot_pred_vs_actual

# Thesis palette used in the rest of the codebase
COLORS = ["#292f56", "#005c7f", "#008780", "#44b155", "#d6c52e", "#3a7a9e"]


# ============================================================
# ======================= USER CONFIG ========================
# ============================================================
SENSOR_VERSION = 5.18

# Quasi-static data for Method A (train_data_v5.1892 = 2 force ramps, very quasi-static)
RAW_DATA_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data")
RAW_CSV_FILES = [
    r"train_data_v5.1893.csv",
]

# LightGBM model v5.180
LGBM_MODEL_DIR  = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models")
LGBM_MODEL_FILE = "lightgbm_sliding_window_model_v5.180.pkl"
LGBM_SCALER_FILE = "scaler_sliding_window_v5.180.pkl"

# Model hyperparameters (must match training)
LGBM_WINDOW_SIZE       = 10      # past samples per window
LGBM_USE_2ND_DERIV     = True    # v5.180 trained with 2nd derivatives → 198 features
LGBM_APPLY_DENOISING   = True    # rolling-mean smoothing on barometers
LGBM_DENOISE_WINDOW    = 5       # denoising window (samples)
LGBM_MAX_TIME_GAP      = 0.05    # skip windows spanning gaps > this (seconds)

# Target column order (must match training)
LGBM_TARGETS = ["x", "y", "fx", "fy", "fz"]   # model[0]=x, [1]=y, [2]=fx, [3]=fy, [4]=fz

# Test split CSV for RMSE-based force resolution (Method B)
TEST_SPLIT_CSV = Path(
    r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data\test_data_v5.18.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\Sensor Characterization\force_resolution"
)

FALLBACK_SAMPLE_RATE_HZ = 59.3

# Limit data to first N seconds (relative to each file's t[0]).
# Set to None to use all data. Use 31.0 to keep only ramp 1 of v5.1892.
DATA_MAX_T_S = None

# Time gap (s) inserted between concatenated experiments in the output plot
EXPERIMENT_GAP_S = 5.0

# ---- Quasi-static window detection ----
ROLLING_WINDOW_S     = 1.0    # seconds
GT_ROLLING_STD_MAX_N = 0.25   # N — max allowed GT Fz std within the window
CONTACT_FZ_MIN_N     = 0.5    # N — ignore near-zero contact samples
MIN_SEGMENT_SAMPLES  = 190    # minimum samples per accepted window

# Aggregate segment-level sigma values
AGGREGATION = "median"        # "median" (robust) or "mean"

# Optional outputs
SAVE_SEGMENT_TABLE = True
SAVE_SUMMARY_TABLE = True
SAVE_PLOT          = True
PLOT_MAX_SECONDS   = None   # None = show full dataset


# ============================================================
# ============== SLIDING WINDOW FEATURE BUILDER ==============
# ============================================================
BARO_COLS    = ["b1", "b2", "b3", "b4", "b5", "b6"]
ALL_EXPECTED = ["t", "b1", "b2", "b3", "b4", "b5", "b6",
                "x", "y", "fx", "fy", "fz", "tx", "ty", "tz"]


def _denoise(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling-mean smoothing on barometer channels (matches training preprocessing)."""
    if not LGBM_APPLY_DENOISING:
        return df
    df = df.copy()
    for col in BARO_COLS:
        df[col] = df[col].rolling(LGBM_DENOISE_WINDOW, center=True).mean().bfill().ffill()
    return df


def _build_window_features(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build sliding-window features identical to those used during LightGBM training.
    Returns:
        X          : (M, n_features) feature matrix
        aligned_df : rows of df at the centre (current) index of each valid window
    """
    df = df.sort_values("t").reset_index(drop=True).copy()
    df = _denoise(df)

    for col in BARO_COLS:
        d1 = df[col].diff().fillna(0.0)
        df[f"{col}_d1"] = d1
        if LGBM_USE_2ND_DERIV:
            df[f"{col}_d2"] = d1.diff().fillna(0.0)

    W          = LGBM_WINDOW_SIZE
    time_vals  = df["t"].values
    baro_data  = df[BARO_COLS].values
    d1_data    = df[[f"{c}_d1" for c in BARO_COLS]].values
    d2_data    = df[[f"{c}_d2" for c in BARO_COLS]].values if LGBM_USE_2ND_DERIV else None

    X_list, valid_indices = [], []
    for i in range(W, len(df)):
        s, e = i - W, i + 1
        if np.max(np.diff(time_vals[s:e])) > LGBM_MAX_TIME_GAP:
            continue
        bw = baro_data[s:e].flatten()
        d1 = d1_data[s:e].flatten()
        if LGBM_USE_2ND_DERIV:
            d2 = d2_data[s:e].flatten()
            X_list.append(np.concatenate([bw, d1, d2]))
        else:
            X_list.append(np.concatenate([bw, d1]))
        valid_indices.append(i)

    X = np.array(X_list)
    aligned_df = df.iloc[valid_indices].reset_index(drop=True)
    return X, aligned_df


# ============================================================
# =================== MODEL LOADING ==========================
# ============================================================

def load_lgbm_models() -> Tuple:
    model_path  = LGBM_MODEL_DIR / LGBM_MODEL_FILE
    scaler_path = LGBM_MODEL_DIR / LGBM_SCALER_FILE

    if not model_path.exists():
        raise FileNotFoundError(f"LightGBM model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    with open(model_path,  "rb") as f: lgbm_list = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler    = pickle.load(f)

    # Map target names to models
    models = {name: lgbm_list[i] for i, name in enumerate(LGBM_TARGETS)}
    return scaler, models


def compute_rmse_resolution(scaler, models) -> Optional[Dict]:
    """
    Method B: Compute RMSE-based force resolution from the held-out test split CSV.
    This reproduces the Predicted vs Actual scatter plot used in model evaluation.
    Returns a dict with MAE, RMSE, bias, R², and 3*RMSE for each force axis, or None if unavailable.
    """
    if not TEST_SPLIT_CSV.exists():
        print(f"  [WARN] Test split CSV not found: {TEST_SPLIT_CSV} — skipping Method B.")
        return None

    print(f"Loading test split for RMSE evaluation: {TEST_SPLIT_CSV.name}")
    df = load_tabular_csv(str(TEST_SPLIT_CSV), ALL_EXPECTED)
    df = df.sort_values("t").reset_index(drop=True)

    X, aligned = _build_window_features(df)
    if len(X) == 0:
        print("  [WARN] No valid windows in test split — skipping Method B.")
        return None

    X_sc = scaler.transform(X)
    pred_fx = models["fx"].predict(X_sc)
    pred_fy = models["fy"].predict(X_sc)
    pred_fz = models["fz"].predict(X_sc)

    gt_fx = aligned["fx"].values
    gt_fy = aligned["fy"].values
    gt_fz = aligned["fz"].values

    result = {}
    for axis, g, p in [("fx", gt_fx, pred_fx), ("fy", gt_fy, pred_fy), ("fz", gt_fz, pred_fz)]:
        err  = p - g
        result[axis] = {
            "n":     len(err),
            "gt":    g,
            "pred":  p,
            "mae":   float(np.nanmean(np.abs(err))),
            "rmse":  float(np.sqrt(np.nanmean(err ** 2))),
            "bias":  float(np.nanmean(err)),
            "std_err": float(np.nanstd(err, ddof=1)),
            "r2":    float(1 - np.nanvar(err) / np.nanvar(g)),
        }

    print(f"  Test split: {len(X)} valid windows from {len(df)} samples.")
    return result


def generate_predictions(raw_data_dir: Path, raw_csv_files: List[str]) -> pd.DataFrame:
    """
    Load each raw experiment CSV, build sliding-window features, apply LightGBM,
    concatenate with time offset between experiments.
    """
    print(f"Loading LightGBM model (v{SENSOR_VERSION:.2f}) from: {LGBM_MODEL_DIR}")
    scaler, models = load_lgbm_models()
    print("  Model loaded.")

    frames: List[pd.DataFrame] = []
    t_offset = 0.0

    for rel_path in raw_csv_files:
        full_path = raw_data_dir / rel_path
        if not full_path.exists():
            print(f"  [WARN] Not found, skipping: {full_path}")
            continue

        print(f"Loading: {full_path.name}")
        df = load_tabular_csv(str(full_path), ALL_EXPECTED)
        df = df.sort_values("t").reset_index(drop=True)
        if DATA_MAX_T_S is not None:
            t0 = df["t"].iloc[0]
            df = df[df["t"] - t0 <= DATA_MAX_T_S].reset_index(drop=True)
            print(f"  Trimmed to first {DATA_MAX_T_S}s: {len(df)} rows")

        X, aligned = _build_window_features(df)
        if len(X) == 0:
            print("  [WARN] No valid windows, skipping.")
            continue

        X_sc = scaler.transform(X)

        pred_fx = models["fx"].predict(X_sc)
        pred_fy = models["fy"].predict(X_sc)
        pred_fz = models["fz"].predict(X_sc)

        t = aligned["t"].values
        t = t - t[0]  # re-zero within this experiment

        frames.append(pd.DataFrame({
            "t":         t + t_offset,
            "Actual_fx": aligned["fx"].values,
            "Actual_fy": aligned["fy"].values,
            "Actual_fz": aligned["fz"].values,
            "Pred_fx":   pred_fx,
            "Pred_fy":   pred_fy,
            "Pred_fz":   pred_fz,
        }))
        print(f"  {len(X)} valid windows  (t_offset = {t_offset:.1f} s)")
        t_offset += float(t[-1]) + EXPERIMENT_GAP_S

    if not frames:
        raise FileNotFoundError(
            f"No raw CSV files found under {raw_data_dir}.\n"
            "Check RAW_DATA_DIR and RAW_CSV_FILES in the config section."
        )

    out = pd.concat(frames, ignore_index=True)
    print(f"  {len(out)} samples total across {len(frames)} experiments.")
    return out


# ============================================================
# ====================== COL DETECTION =======================
# ============================================================
TIME_CANDIDATES = ["t", "time", "Time_s", "Epoch_s"]

GT_CANDIDATES: Dict[str, List[str]] = {
    "fx": ["Actual_fx", "fx", "gt_fx", "true_fx"],
    "fy": ["Actual_fy", "fy", "gt_fy", "true_fy"],
    "fz": ["Actual_fz", "fz", "gt_fz", "true_fz"],
}
PRED_CANDIDATES: Dict[str, List[str]] = {
    "fx": ["Pred_fx", "pred_fx", "fx_pred"],
    "fy": ["Pred_fy", "pred_fy", "fy_pred"],
    "fz": ["Pred_fz", "pred_fz", "fz_pred"],
}


@dataclass
class QuasiStaticSegment:
    start: int
    end: int

    @property
    def n(self) -> int:
        return self.end - self.start + 1


def pick_first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def contiguous_segments(mask: np.ndarray) -> List[QuasiStaticSegment]:
    segments: List[QuasiStaticSegment] = []
    start: Optional[int] = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append(QuasiStaticSegment(start=start, end=i - 1))
            start = None
    if start is not None:
        segments.append(QuasiStaticSegment(start=start, end=len(mask) - 1))
    return segments


def infer_columns(df: pd.DataFrame) -> Tuple[Optional[str], Dict[str, str], Dict[str, str]]:
    cols = list(df.columns)
    time_col = pick_first_existing(cols, TIME_CANDIDATES)
    gt_cols: Dict[str, str] = {}
    pred_cols: Dict[str, str] = {}
    for ax in ["fx", "fy", "fz"]:
        gt = pick_first_existing(cols, GT_CANDIDATES[ax])
        pr = pick_first_existing(cols, PRED_CANDIDATES[ax])
        if gt is None or pr is None:
            missing = "GT" if gt is None else "PRED"
            raise ValueError(f"Missing {missing} column for axis '{ax}'. Available: {cols}")
        gt_cols[ax] = gt
        pred_cols[ax] = pr
    return time_col, gt_cols, pred_cols


def build_time_array(df: pd.DataFrame, time_col: Optional[str]) -> np.ndarray:
    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(t)
        if not finite.any():
            raise ValueError(f"Time column '{time_col}' has no valid values.")
        if not finite.all():
            idx = np.arange(len(t), dtype=float)
            t = np.interp(idx, idx[finite], t[finite])
        return t - t[0]
    return np.arange(len(df), dtype=float) / FALLBACK_SAMPLE_RATE_HZ


def detect_quasistatic_windows(
    gt_forces: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, List[QuasiStaticSegment], np.ndarray]:
    dt_arr   = np.diff(t)
    valid_dt = dt_arr[(dt_arr > 0) & np.isfinite(dt_arr)]
    sr       = 1.0 / np.median(valid_dt) if len(valid_dt) > 0 else FALLBACK_SAMPLE_RATE_HZ
    win      = max(3, int(round(ROLLING_WINDOW_S * sr)))

    rolling_std = (
        pd.Series(gt_forces[:, 2])
        .rolling(win, center=True, min_periods=max(3, win // 4))
        .std()
        .to_numpy()
    )

    contact    = np.abs(gt_forces[:, 2]) >= CONTACT_FZ_MIN_N
    quasi_mask = contact & (rolling_std < GT_ROLLING_STD_MAX_N) & np.isfinite(rolling_std)

    kept: List[QuasiStaticSegment] = []
    for seg in contiguous_segments(quasi_mask):
        if seg.n >= MIN_SEGMENT_SAMPLES:
            kept.append(seg)

    return quasi_mask, kept, rolling_std


def _print_detection_diagnostics(gt: np.ndarray, t: np.ndarray,
                                  quasi_mask: np.ndarray,
                                  rolling_std: np.ndarray) -> None:
    contact = np.abs(gt[:, 2]) >= CONTACT_FZ_MIN_N
    print("\n--- Quasi-static detection diagnostics ---")
    print(f"  Total samples          : {len(t)}")
    print(f"  Contact (|Fz|>={CONTACT_FZ_MIN_N} N)    : {contact.sum()} ({100*contact.mean():.1f}%)")
    print(f"  Low rolling-std (<{GT_ROLLING_STD_MAX_N}N) : {(rolling_std<GT_ROLLING_STD_MAX_N).sum()}")
    print(f"  Rolling-std p10={np.nanpercentile(rolling_std,10):.3f}  p25={np.nanpercentile(rolling_std,25):.3f}  p50={np.nanpercentile(rolling_std,50):.3f}  N")
    raw = contiguous_segments(quasi_mask)
    print(f"  Raw blobs: {len(raw)}, sizes: max={max((s.n for s in raw), default=0)}, need>={MIN_SEGMENT_SAMPLES}")
    print(f"  Try raising GT_ROLLING_STD_MAX_N (currently {GT_ROLLING_STD_MAX_N}) "
          f"to ~{np.nanpercentile(rolling_std[contact],20):.3f}")
    print("-------------------------------------------")


def aggregate(arr: np.ndarray, mode: str) -> float:
    return float(np.nanmean(arr)) if mode == "mean" else float(np.nanmedian(arr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Force resolution from quasi-static windows.")
    parser.add_argument("--raw-data-dir", type=str, default=str(RAW_DATA_DIR))
    parser.add_argument("--output-dir",   type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    raw_data_dir = Path(args.raw_data_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Generate LightGBM predictions from raw continuous time series ---
    df = generate_predictions(raw_data_dir, RAW_CSV_FILES)

    # --- 2. Extract arrays ---
    time_col, gt_cols, pred_cols = infer_columns(df)
    t = build_time_array(df, time_col)

    gt = np.column_stack([
        pd.to_numeric(df[gt_cols["fx"]], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df[gt_cols["fy"]], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df[gt_cols["fz"]], errors="coerce").to_numpy(dtype=float),
    ])
    pred = np.column_stack([
        pd.to_numeric(df[pred_cols["fx"]], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df[pred_cols["fy"]], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df[pred_cols["fz"]], errors="coerce").to_numpy(dtype=float),
    ])

    valid_rows = np.isfinite(gt).all(axis=1) & np.isfinite(pred).all(axis=1) & np.isfinite(t)
    if valid_rows.sum() < 10:
        raise RuntimeError("Not enough valid rows after numeric filtering.")

    gt   = gt[valid_rows];  pred = pred[valid_rows]
    t    = t[valid_rows];   t    = t - t[0]

    # --- 3. Detect quasi-static windows ---
    quasi_mask, segments, rolling_std = detect_quasistatic_windows(gt, t)
    if len(segments) == 0:
        _print_detection_diagnostics(gt, t, quasi_mask, rolling_std)
        raise RuntimeError("No valid quasi-static windows found — see diagnostics above.")

    print(f"  Detected {len(segments)} quasi-static windows "
          f"(rolling-std < {GT_ROLLING_STD_MAX_N} N, window = {ROLLING_WINDOW_S} s).")

    # --- 4. Segment-wise stats ---
    rows = []
    for i, seg in enumerate(segments, start=1):
        s, e = seg.start, seg.end
        sig_pred = np.nanstd(pred[s:e+1], axis=0, ddof=1)
        sig_gt   = np.nanstd(gt[s:e+1],   axis=0, ddof=1)
        mean_fz  = float(np.nanmean(np.abs(gt[s:e+1, 2])))   # mean |Fz| in segment
        rows.append({
            "segment":                i,
            "start_idx": s, "end_idx": e,
            "t_start_s":              float(t[s]),
            "t_end_s":                float(t[e]),
            "duration_s":             float(t[e] - t[s]),
            "n_samples":              int(seg.n),
            "mean_abs_Fz_N":          mean_fz,
            "sigma_pred_fx_N":        float(sig_pred[0]),
            "sigma_pred_fy_N":        float(sig_pred[1]),
            "sigma_pred_fz_N":        float(sig_pred[2]),
            "dFmin_pred_fx_N_3sigma": float(3.0 * sig_pred[0]),
            "dFmin_pred_fy_N_3sigma": float(3.0 * sig_pred[1]),
            "dFmin_pred_fz_N_3sigma": float(3.0 * sig_pred[2]),
            "sigma_gt_fx_N":          float(sig_gt[0]),
            "sigma_gt_fy_N":          float(sig_gt[1]),
            "sigma_gt_fz_N":          float(sig_gt[2]),
        })

    seg_df = pd.DataFrame(rows)

    # --- 5. Aggregate quasi-static sigma ---
    sigma_fx = aggregate(seg_df["sigma_pred_fx_N"].to_numpy(), AGGREGATION)
    sigma_fy = aggregate(seg_df["sigma_pred_fy_N"].to_numpy(), AGGREGATION)
    sigma_fz = aggregate(seg_df["sigma_pred_fz_N"].to_numpy(), AGGREGATION)

    # --- 5c. Per-force-level breakdown ---
    # Sort segments by mean |Fz| and print resolution at each load level
    seg_by_fz = seg_df.sort_values("mean_abs_Fz_N").copy()

    # --- 5b. RMSE-based force resolution (Method B: test-split scatter plot) ---
    # Load the scaler+models again (already in memory via generate_predictions, so reload)
    scaler_b, models_b = load_lgbm_models()
    rmse_res = compute_rmse_resolution(scaler_b, models_b)

    # Build summary dict
    rmse_row = {}
    if rmse_res is not None:
        for ax in ["fx", "fy", "fz"]:
            r = rmse_res[ax]
            rmse_row[f"n_samples_rmse"]      = r["n"]
            rmse_row[f"MAE_{ax}_N"]          = r["mae"]
            rmse_row[f"RMSE_{ax}_N"]         = r["rmse"]
            rmse_row[f"bias_{ax}_N"]         = r["bias"]
            rmse_row[f"R2_{ax}"]             = r["r2"]
            rmse_row[f"DeltaFmin_{ax}_rmse_N"] = 3.0 * r["rmse"]

    summary = pd.DataFrame([{
        # quasi-static window method
        "aggregation":              AGGREGATION,
        "model":                    "LightGBM",
        "n_segments_qstatic":       len(seg_df),
        "rolling_window_s":         ROLLING_WINDOW_S,
        "gt_std_threshold_N":       GT_ROLLING_STD_MAX_N,
        "sigma_Fx_qstatic_N":       sigma_fx,
        "sigma_Fy_qstatic_N":       sigma_fy,
        "sigma_Fz_qstatic_N":       sigma_fz,
        "DeltaFmin_Fx_qstatic_N":   3.0 * sigma_fx,
        "DeltaFmin_Fy_qstatic_N":   3.0 * sigma_fy,
        "DeltaFmin_Fz_qstatic_N":   3.0 * sigma_fz,
        **rmse_row,
    }])

    # --- Console report ---
    print(f"\n=== Force Resolution for LightGBM ===")
    print()
    print("--- Method A: Quasi-static window repeatability ---")
    print(f"  Criterion : rolling-std(GT Fz) < {GT_ROLLING_STD_MAX_N} N  over {ROLLING_WINDOW_S} s")
    print(f"  Segments  : {len(seg_df)}    Samples: {seg_df['n_samples'].sum()}")
    print(f"  Aggregation: {AGGREGATION}")
    print(f"  sigma_Fx = {sigma_fx:.4f} N   -->  DeltaFx_min = 3*sigma = {3*sigma_fx:.4f} N")
    print(f"  sigma_Fy = {sigma_fy:.4f} N   -->  DeltaFy_min = 3*sigma = {3*sigma_fy:.4f} N")
    print(f"  sigma_Fz = {sigma_fz:.4f} N   -->  DeltaFz_min = 3*sigma = {3*sigma_fz:.4f} N")
    print()
    print("  Per-segment breakdown (sorted by load level):")
    print(f"  {'Seg':>4} {'|Fz| mean':>10} {'sigma_Fx':>10} {'DeltaFx':>9} {'sigma_Fy':>10} {'DeltaFy':>9} {'sigma_Fz':>10} {'DeltaFz':>9}")
    for _, row in seg_by_fz.iterrows():
        print(f"  {int(row['segment']):>4} {row['mean_abs_Fz_N']:>9.2f}N "
              f"{row['sigma_pred_fx_N']:>9.4f}N {row['dFmin_pred_fx_N_3sigma']:>8.4f}N "
              f"{row['sigma_pred_fy_N']:>9.4f}N {row['dFmin_pred_fy_N_3sigma']:>8.4f}N "
              f"{row['sigma_pred_fz_N']:>9.4f}N {row['dFmin_pred_fz_N_3sigma']:>8.4f}N")
    print()
    if rmse_res is not None:
        print("--- Method B: RMSE on test split (= scatter plot sigma) ---")
        print(f"  Samples : {rmse_res['fx']['n']}")
        print(f"  {'Axis':<6} {'MAE':>8} {'RMSE=sigma':>12} {'Bias':>8} {'R2':>8}  {'3*RMSE=DeltaF':>15}")
        for ax, label in [("fx","Fx"), ("fy","Fy"), ("fz","Fz")]:
            r = rmse_res[ax]
            print(f"  {label:<6} {r['mae']:>8.4f} {r['rmse']:>12.4f} {r['bias']:>8.4f} {r['r2']:>8.4f}  {3*r['rmse']:>13.4f} N")
        print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if SAVE_SEGMENT_TABLE:
        out_seg = output_dir / "force_resolution_segments.csv"
        seg_df.to_csv(out_seg, index=False)
        print(f"\nSaved segment table : {out_seg}")

    if SAVE_SUMMARY_TABLE:
        out_sum = output_dir / "force_resolution_summary.csv"
        summary.to_csv(out_sum, index=False)
        print(f"Saved summary table : {out_sum}")

    if SAVE_PLOT:
        # Plot 1 — QC time-series plot
        if PLOT_MAX_SECONDS is not None and PLOT_MAX_SECONDS > 0:
            m = t <= PLOT_MAX_SECONDS
            t_plot = t[m]; gt_plot = gt[m]; pred_plot = pred[m]
            qmask_plot = quasi_mask[m]; rs_plot = rolling_std[m]
        else:
            t_plot = t; gt_plot = gt; pred_plot = pred
            qmask_plot = quasi_mask; rs_plot = rolling_std

        fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
        ax.plot(t_plot, gt_plot[:, 2], label="GT Fz", color="#292f56", linewidth=1.5)
        ax.plot(t_plot, pred_plot[:, 2], label="Pred Fz", color="#44b155", linewidth=1.0, alpha=0.85)
        ax.set_ylabel("Force [N]")
        ax.set_xlabel("Time [s]")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Force Resolution for LightGBM"
        )
        fig.tight_layout()
        out_plot = output_dir / "force_resolution_qc_plot.png"
        fig.savefig(out_plot, dpi=250)
        plt.close(fig)
        print(f"Saved QC plot       : {out_plot}")

        # Plot 2 — Predicted vs Actual scatter from TEST SPLIT (matches thesis scatter plot)
        if rmse_res is not None:
            y_true_plot = np.column_stack([
                rmse_res["fx"]["gt"],
                rmse_res["fy"]["gt"],
                rmse_res["fz"]["gt"],
            ])
            y_pred_plot = np.column_stack([
                rmse_res["fx"]["pred"],
                rmse_res["fy"]["pred"],
                rmse_res["fz"]["pred"],
            ])

            # Same visual style used in LightGBM evaluation plots, with petrol-green dots.
            fig2 = plot_pred_vs_actual(
                y_true_plot,
                y_pred_plot,
                ["fx", "fy", "fz"],
                title_suffix=f"LightGBM",
                scatter_color=COLORS[3],      # same green used for predictions in plot 1
                ideal_line_color=COLORS[0],   # same blue used for GT in plot 1
                show_units=True,
                alpha=0.35,
                s=8,
                figsize_factor=(4.4, 4.2),
            )
            # Replace R² with force resolution metric in subplot titles.
            for ax_obj, key, axis_label in zip(fig2.axes, ["fx", "fy", "fz"], ["Fx", "Fy", "Fz"]):
                mae = rmse_res[key]["mae"]
                sigma = rmse_res[key]["rmse"]  # sigma estimate from test split
                delta_f = 3.0 * sigma
                ax_obj.set_title(
                    f"{axis_label}\nMAE: {mae:.2f} N | ΔF_min (3σ): {delta_f:.2f} N",
                    fontsize=10,
                )
            out_scatter = output_dir / "force_resolution_scatter.png"
            fig2.savefig(out_scatter, dpi=250)
            plt.close(fig2)
            print(f"Saved scatter plot  : {out_scatter}")


if __name__ == "__main__":
    main()
