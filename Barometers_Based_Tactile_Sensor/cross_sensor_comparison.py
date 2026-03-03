#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-sensor repeatability comparison.

For three sensors (train datasets v5.1795 / v5.1895 / v5.1995):
  1. Build sliding-window LightGBM features (same pipeline as training).
  2. Run all sensor-specific models (v5.170, v5.180, v5.190) and the
     general model (v5.010) on every dataset.
  3. Plot A — 3 subplots (one per dataset): GT + all 3 sensor-specific
              model predictions.
  4. Plot B — 3 subplots (one per dataset): GT + general model prediction.
  5. Plot C — Repeatability in force domain: sensor predictions aligned
              on a common GT-Fz grid → CV across sensors.

Repeatability metric
--------------------
Because the three sensors record similar-but-not-identical experiments
(different timing, magnitudes), we cannot compare predictions at the same
time step.  Instead we use **force-domain alignment**:

  For each sensor's own model prediction, we interpolate Pred_Fz as a
  function of GT_Fz onto a shared GT-Fz grid.  At each force level we
  then compute std / mean (coefficient of variation, CV).  A low mean CV
  across the force range indicates consistent, repeatable predictions.

  We also compute pairwise RMSE on the aligned force domain and a
  normalised-time view for visual inspection.
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils.io_utils import load_tabular_csv

# ============================================================
# ======================== CONFIG ============================
# ============================================================

BASE      = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab")
DATA_DIR  = BASE / "train_validation_test_data"
MODEL_DIR = BASE / "models parameters" / "averaged models"
OUT_DIR   = BASE / "Sensor Characterization" / "cross_sensor_comparison"

# ---- Three sensor datasets (one per physical sensor) ----
DATASETS = {
    "Sensor 1 (v5.17)": DATA_DIR / "train_data_v5.1795.csv",
    "Sensor 2 (v5.18)": DATA_DIR / "train_data_v5.1895.csv",
    "Sensor 3 (v5.19)": DATA_DIR / "train_data_v5.1995.csv",
}

# ---- Sensor-specific models ----
SENSOR_MODELS = {
    "v5.170": {
        "model":  MODEL_DIR / "lightgbm_sliding_window_model_v5.170.pkl",
        "scaler": MODEL_DIR / "scaler_sliding_window_v5.170.pkl",
    },
    "v5.180": {
        "model":  MODEL_DIR / "lightgbm_sliding_window_model_v5.180.pkl",
        "scaler": MODEL_DIR / "scaler_sliding_window_v5.180.pkl",
    },
    "v5.190": {
        "model":  MODEL_DIR / "lightgbm_sliding_window_model_v5.190.pkl",
        "scaler": MODEL_DIR / "scaler_sliding_window_v5.190.pkl",
    },
}

# ---- General (cross-sensor) model ----
GENERAL_MODEL = {
    "model":  MODEL_DIR / "lightgbm_sliding_window_model_v5.010.pkl",
    "scaler": MODEL_DIR / "scaler_sliding_window_v5.010.pkl",
}

# ---- Each sensor's "own" model (same ordering as DATASETS) ----
OWN_MODEL_MAP = {
    "Sensor 1 (v5.17)": "v5.170",
    "Sensor 2 (v5.18)": "v5.180",
    "Sensor 3 (v5.19)": "v5.190",
}

# ---- Feature-engineering settings (MUST match training) ----
BARO_COLS  = ["b1", "b2", "b3", "b4", "b5", "b6"]
ALL_COLS   = ["t", "b1", "b2", "b3", "b4", "b5", "b6",
              "x", "y", "fx", "fy", "fz", "tx", "ty", "tz"]

WINDOW_SIZE    = 10
DENOISE_WINDOW = 5
MAX_TIME_GAP   = 0.05
FZ_MODEL_IDX   = 4      # models[4] → Fz  (targets: x, y, fx, fy, fz)

# Feature counts — used to auto-detect which config each model was trained with
_FEATS_NO_D2   = 6 * (WINDOW_SIZE + 1) * 2   # 132  (baro + d1)
_FEATS_WITH_D2 = 6 * (WINDOW_SIZE + 1) * 3   # 198  (baro + d1 + d2)

# ---- Prediction smoothing ----
MA_WIN = 11             # moving-average samples for display

# ---- Force-domain repeatability grid ----
FZ_GRID_N     = 300     # number of equally-spaced GT-Fz levels
CONTACT_THR   = 0.3     # N  — minimum GT Fz to classify as contact

# ---- Normalised-time grid (for Plot C visual) ----
N_GRID = 300

# ---- Thesis colour palette ----
GT_COLOR      = "#292f56"   # dark navy
MODEL_COLORS  = {
    "v5.170": "#005c7f",
    "v5.180": "#008780",
    "v5.190": "#44b155",
}
GEN_COLOR     = "#d6c52e"   # yellow / amber
SENSOR_COLORS = ["#005c7f", "#008780", "#44b155"]   # for per-dataset lines

# ============================================================
# ============== FEATURE BUILDER (matches training) ===========
# ============================================================

def _denoise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BARO_COLS:
        df[col] = df[col].rolling(DENOISE_WINDOW, center=True).mean().bfill().ffill()
    return df


def build_features(df: pd.DataFrame, use_2nd_deriv: bool = True):
    """
    Build sliding-window features identical to those used at training time.

    Parameters
    ----------
    use_2nd_deriv : bool
        True  → 198 features (baro + d1 + d2)
        False → 132 features (baro + d1 only)

    Returns
    -------
    X        : (M, n_features) feature matrix
    aligned  : DataFrame of the centre (current) row for each valid window
    """
    df = df.sort_values("t").reset_index(drop=True).copy()
    df = _denoise(df)

    for col in BARO_COLS:
        d1 = df[col].diff().fillna(0.0)
        df[f"{col}_d1"] = d1
        if use_2nd_deriv:
            df[f"{col}_d2"] = d1.diff().fillna(0.0)

    time_v = df["t"].values
    bv     = df[BARO_COLS].values
    d1v    = df[[f"{c}_d1" for c in BARO_COLS]].values
    d2v    = df[[f"{c}_d2" for c in BARO_COLS]].values if use_2nd_deriv else None

    X_list, valid_idx = [], []
    for i in range(WINDOW_SIZE, len(df)):
        s, e = i - WINDOW_SIZE, i + 1
        if np.max(np.diff(time_v[s:e])) > MAX_TIME_GAP:
            continue
        row = np.concatenate([bv[s:e].flatten(), d1v[s:e].flatten()])
        if use_2nd_deriv:
            row = np.concatenate([row, d2v[s:e].flatten()])
        X_list.append(row)
        valid_idx.append(i)

    X       = np.array(X_list)
    aligned = df.iloc[valid_idx].reset_index(drop=True)
    return X, aligned


def scaler_use_d2(scaler) -> bool:
    """Detect from the fitted scaler whether the model was trained with 2nd derivatives."""
    n = scaler.n_features_in_
    if n == _FEATS_WITH_D2:
        return True
    if n == _FEATS_NO_D2:
        return False
    raise ValueError(
        f"Unexpected scaler n_features_in_={n}. "
        f"Expected {_FEATS_NO_D2} (no d2) or {_FEATS_WITH_D2} (with d2)."
    )


# ============================================================
# ============== HELPERS =====================================
# ============================================================

def load_model(cfg: dict):
    with open(cfg["model"],  "rb") as f:
        models = pickle.load(f)
    with open(cfg["scaler"], "rb") as f:
        scaler = pickle.load(f)
    return models, scaler


def smooth(arr: np.ndarray, win: int = MA_WIN) -> np.ndarray:
    return np.convolve(arr, np.ones(win) / win, mode="same")



# ============================================================
# ============== MAIN ========================================
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all models ────────────────────────────────────────────────────────
    print("Loading sensor-specific models ...")
    loaded_sensor_models = {}
    for mname, cfg in SENSOR_MODELS.items():
        print(f"  {mname}")
        loaded_sensor_models[mname] = load_model(cfg)

    print("Loading general model (v5.010) ...")
    gen_models, gen_scaler = load_model(GENERAL_MODEL)

    # ── Detect feature config (use_2nd_deriv) for every model ────────────────
    dataset_labels = list(DATASETS.keys())
    model_d2: dict[str, bool] = {}
    for mname, (mod, scl) in loaded_sensor_models.items():
        model_d2[mname] = scaler_use_d2(scl)
        print(f"  {mname}: use_2nd_deriv={model_d2[mname]}  "
              f"(n_features={scl.n_features_in_})")
    gen_d2 = scaler_use_d2(gen_scaler)
    print(f"  general:  use_2nd_deriv={gen_d2}  "
          f"(n_features={gen_scaler.n_features_in_})")

    # ── Build predictions for every (dataset × model) combination ─────────────
    # results[label] = {
    #   "t":    np.ndarray,   # time (zero-referenced from dataset start)
    #   "gt":   np.ndarray,   # |GT Fz|  (NaN for no-contact rows)
    #   "preds": {mname: np.ndarray},
    #   "gen_pred": np.ndarray,
    # }
    results = {}

    for label, csv_path in DATASETS.items():
        print(f"\nProcessing {label}: {csv_path.name}")
        df = load_tabular_csv(str(csv_path), ALL_COLS)
        df = df.sort_values("t").reset_index(drop=True)

        # Convert no-contact sentinel (-999) → NaN so they appear as gaps in
        # plots and are excluded from the contact-region metrics.
        for col in ["x", "y", "fx", "fy", "fz", "tx", "ty", "tz"]:
            if col in df.columns:
                df[col] = df[col].replace(-999.0, np.nan)
        n_contact = int(df["fz"].notna().sum())
        print(f"  Contact samples: {n_contact}/{len(df)} ({100*n_contact/len(df):.1f}%)")

        # Lazy feature cache: build features per d2 config at most once per dataset
        feat_cache: dict[bool, tuple] = {}

        def get_features(use_d2: bool):
            if use_d2 not in feat_cache:
                feat_cache[use_d2] = build_features(df, use_2nd_deriv=use_d2)
            return feat_cache[use_d2]

        # Reference t/gt come from the first sensor model's feature config
        ref_d2 = model_d2[list(loaded_sensor_models.keys())[0]]
        _, ref_aligned = get_features(ref_d2)
        t  = ref_aligned["t"].values - ref_aligned["t"].values[0]
        gt = np.abs(ref_aligned["fz"].values)   # NaN preserved for no-contact

        preds = {}
        for mname, (mod, scl) in loaded_sensor_models.items():
            X, aligned = get_features(model_d2[mname])
            raw = np.abs(mod[FZ_MODEL_IDX].predict(scl.transform(X)))
            # Re-align onto reference time grid when feature configs differ
            if model_d2[mname] != ref_d2:
                t_m = aligned["t"].values - aligned["t"].values[0]
                raw = np.interp(t, t_m, smooth(raw))
            else:
                raw = smooth(raw)
            preds[mname] = raw
            print(f"  [{mname}] max pred = {preds[mname].max():.2f} N")

        X_gen, aligned_gen = get_features(gen_d2)
        raw_gen = np.abs(gen_models[FZ_MODEL_IDX].predict(gen_scaler.transform(X_gen)))
        if gen_d2 != ref_d2:
            t_g = aligned_gen["t"].values - aligned_gen["t"].values[0]
            raw_gen = np.interp(t, t_g, smooth(raw_gen))
        else:
            raw_gen = smooth(raw_gen)
        print(f"  [general] max pred = {raw_gen.max():.2f} N")

        results[label] = {
            "t":        t,
            "gt":       gt,
            "preds":    preds,
            "gen_pred": raw_gen,
        }

    # ── PLOT A — sensor-specific models, one subplot per dataset ──────────────
    fig_a, axes_a = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig_a.suptitle(
        "Cross-sensor: GT vs sensor-specific model predictions", fontsize=13
    )

    for ax, label in zip(axes_a, dataset_labels):
        entry = results[label]
        t_full, gt_full = entry["t"], entry["gt"]
        # Show only contact rows so loading cycles are visible
        contact = ~np.isnan(gt_full)
        t  = t_full[contact]
        gt = gt_full[contact]

        ax.plot(t, gt, color=GT_COLOR, lw=2.0, label="GT $|F_z|$", zorder=3)
        for mname, col in MODEL_COLORS.items():
            ax.plot(t, entry["preds"][mname][contact], color=col, lw=1.4, alpha=0.85,
                    label=f"Model {mname}")

        ax.set_title(label, fontsize=11)
        ax.set_ylabel("$|F_z|$ [N]", fontsize=10)
        ax.legend(fontsize=9, ncol=4, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes_a[-1].set_xlabel("Time [s]", fontsize=11)
    fig_a.tight_layout()
    out_a = OUT_DIR / "cross_sensor_A_sensor_models.png"
    fig_a.savefig(out_a, dpi=250, bbox_inches="tight")
    plt.close(fig_a)
    print(f"\nSaved Plot A → {out_a}")

    # ── PLOT B — general model, one subplot per dataset ───────────────────────
    fig_b, axes_b = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig_b.suptitle(
        "Cross-sensor: GT vs general model (v5.010)", fontsize=13
    )

    for ax, label in zip(axes_b, dataset_labels):
        entry = results[label]
        t_full, gt_full = entry["t"], entry["gt"]
        contact = ~np.isnan(gt_full)
        t  = t_full[contact]
        gt = gt_full[contact]

        ax.plot(t, gt,                         color=GT_COLOR,  lw=2.0, label="GT $|F_z|$", zorder=3)
        ax.plot(t, entry["gen_pred"][contact],  color=GEN_COLOR, lw=1.5, alpha=0.88,
                label="General model v5.010")

        ax.set_title(label, fontsize=11)
        ax.set_ylabel("$|F_z|$ [N]", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes_b[-1].set_xlabel("Time [s]", fontsize=11)
    fig_b.tight_layout()
    out_b = OUT_DIR / "cross_sensor_B_general_model.png"
    fig_b.savefig(out_b, dpi=250, bbox_inches="tight")
    plt.close(fig_b)
    print(f"Saved Plot B → {out_b}")

    # ── REPEATABILITY METRIC ──────────────────────────────────────────────────
    # Force-domain alignment: interpolate each sensor's own-model prediction
    # as a function of GT Fz onto a shared GT-Fz grid.
    #
    # This is the right comparison because:
    #   - Physical experiments are similar but not time-synchronised.
    #   - GT Fz is the controlled stimulus; we want pred(GT=F) per sensor.
    #   - CV at each force level = direct repeatability measure.
    print("\n" + "=" * 58)
    print("REPEATABILITY METRIC  — force-domain alignment")
    print("=" * 58)

    # 1. Collect (gt, own_pred) pairs in contact region, per sensor
    force_data = {}   # label → (gt_contact, pred_contact)
    for label in dataset_labels:
        entry   = results[label]
        gt      = entry["gt"]
        own_key = OWN_MODEL_MAP[label]
        pred    = entry["preds"][own_key]

        mask    = gt > CONTACT_THR
        if mask.sum() < 20:
            print(f"  [WARN] {label}: only {mask.sum()} contact samples — skipping.")
            continue
        force_data[label] = (gt[mask], pred[mask])
        print(f"  {label}: {mask.sum()} contact samples  "
              f"(GT range {gt[mask].min():.2f}–{gt[mask].max():.2f} N, "
              f"model {own_key})")

    available_labels = list(force_data.keys())

    if len(available_labels) >= 2:
        # 2. Build a common force grid spanning the *intersection* of GT ranges
        max_common = min(fd[0].max() for fd in force_data.values())
        min_common = max(fd[0].min() for fd in force_data.values())
        if max_common <= min_common:
            print("  [WARN] No common force range — widening to union.")
            max_common = max(fd[0].max() for fd in force_data.values())
            min_common = min(fd[0].min() for fd in force_data.values())

        fz_grid = np.linspace(min_common, max_common, FZ_GRID_N)
        print(f"\n  Force grid: {min_common:.2f} → {max_common:.2f} N  "
              f"({FZ_GRID_N} points)")

        # 3. Interpolate each sensor's prediction onto the grid
        #    Sort by GT Fz first to make it a monotone mapping
        grid_interp = {}
        for label, (gt_c, pred_c) in force_data.items():
            idx     = np.argsort(gt_c)
            gt_s    = gt_c[idx]
            pred_s  = pred_c[idx]
            grid_interp[label] = np.interp(fz_grid, gt_s, pred_s)

        # 4. Statistics across sensors at each force level
        stacked  = np.vstack([grid_interp[l] for l in available_labels])  # (N_sensors, grid)
        mean_p   = stacked.mean(axis=0)
        std_p    = stacked.std(axis=0)
        cv_pct   = (std_p / (mean_p + 1e-9)) * 100.0
        mean_cv  = float(np.nanmean(cv_pct))
        max_cv   = float(np.nanmax(cv_pct))

        # 5. Pairwise RMSE in force domain
        pairs = []
        for i in range(len(available_labels)):
            for j in range(i + 1, len(available_labels)):
                li, lj = available_labels[i], available_labels[j]
                rmse_ij = float(np.sqrt(np.mean((grid_interp[li] - grid_interp[lj]) ** 2)))
                pairs.append((li, lj, rmse_ij))

        print(f"\n  Metric : Coefficient of Variation (CV) = std / mean × 100 %")
        print(f"  Mean CV across force range : {mean_cv:.2f} %  (lower = more repeatable)")
        print(f"  Max  CV across force range : {max_cv:.2f} %")
        print(f"\n  Pairwise RMSE on force-aligned predictions [N]:")
        for li, lj, r in pairs:
            print(f"    {li}  vs  {lj} :  RMSE = {r:.4f} N")

        # -- Normalised-time view for same analysis -------------------------
        # We also interpolate onto normalised time for the visual Panel C
        norm_grid = np.linspace(0, 1, N_GRID)
        norm_interp = {}
        norm_gt     = {}
        for label in available_labels:
            entry    = results[label]
            gt       = entry["gt"]
            own_key  = OWN_MODEL_MAP[label]
            pred     = entry["preds"][own_key]
            mask     = gt > CONTACT_THR
            if mask.sum() < 10:
                continue
            idx      = np.where(mask)[0]
            t_rel    = entry["t"][idx]
            t_norm   = (t_rel - t_rel[0]) / (t_rel[-1] - t_rel[0] + 1e-9)
            norm_interp[label] = np.interp(norm_grid, t_norm, pred[idx])
            norm_gt[label]     = np.interp(norm_grid, t_norm,  gt[idx])

        # Also for general model
        norm_gen = {}
        for label in available_labels:
            entry  = results[label]
            gt     = entry["gt"]
            gen_p  = entry["gen_pred"]
            mask   = gt > CONTACT_THR
            if mask.sum() < 10:
                continue
            idx    = np.where(mask)[0]
            t_rel  = entry["t"][idx]
            t_norm = (t_rel - t_rel[0]) / (t_rel[-1] - t_rel[0] + 1e-9)
            norm_gen[label] = np.interp(norm_grid, t_norm, gen_p[idx])

        # ── PLOT C — Repeatability figure (3 panels) ──────────────────────
        # Panel 1: GT on normalised time
        # Panel 2: Own-model predictions on normalised time
        # Panel 3: Force-domain CV ribbon (the key metric)
        fig_c, axes_c = plt.subplots(1, 3, figsize=(15, 5))
        fig_c.suptitle(
            f"Cross-sensor repeatability  |  "
            f"Mean CV (own models) = {mean_cv:.1f} %",
            fontsize=12,
        )

        # Panel 1 — GT on normalised time
        ax1 = axes_c[0]
        for label, col in zip(available_labels, SENSOR_COLORS):
            if label in norm_gt:
                ax1.plot(norm_grid, norm_gt[label], color=col, lw=1.8,
                         label=label, alpha=0.85)
        stacked_gt = np.vstack([norm_gt[l] for l in available_labels if l in norm_gt])
        if stacked_gt.shape[0] >= 2:
            m_gt  = stacked_gt.mean(axis=0)
            s_gt  = stacked_gt.std(axis=0)
            ax1.fill_between(norm_grid, m_gt - s_gt, m_gt + s_gt,
                             alpha=0.18, color="gray")
            ax1.plot(norm_grid, m_gt, color="black", lw=1.5, ls="--",
                     label="Mean GT")
        ax1.set_xlabel("Normalised time", fontsize=10)
        ax1.set_ylabel("GT $|F_z|$ [N]", fontsize=10)
        ax1.set_title("Ground truth\n(ATI wrench)", fontsize=11)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        # Panel 2 — Own-model predictions on normalised time
        ax2 = axes_c[1]
        stacked_norm = np.vstack([norm_interp[l] for l in available_labels if l in norm_interp])
        m_norm = stacked_norm.mean(axis=0) if stacked_norm.shape[0] >= 2 else None
        s_norm = stacked_norm.std(axis=0)  if stacked_norm.shape[0] >= 2 else None
        for label, col in zip(available_labels, SENSOR_COLORS):
            if label in norm_interp:
                ax2.plot(norm_grid, norm_interp[label], color=col, lw=1.8,
                         label=f"{label}\n→ {OWN_MODEL_MAP[label]}", alpha=0.85)
        if m_norm is not None:
            ax2.fill_between(norm_grid, m_norm - s_norm, m_norm + s_norm,
                             alpha=0.18, color="gray")
            ax2.plot(norm_grid, m_norm, color="black", lw=1.5, ls="--",
                     label="Mean pred")
        ax2.set_xlabel("Normalised time", fontsize=10)
        ax2.set_ylabel("Pred $|F_z|$ [N]", fontsize=10)
        ax2.set_title("Own-model predictions\n(normalised time)", fontsize=11)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)

        # Panel 3 — Force-domain CV with individual predictions
        ax3 = axes_c[2]
        for label, col in zip(available_labels, SENSOR_COLORS):
            ax3.plot(fz_grid, grid_interp[label], color=col, lw=1.6,
                     label=f"{label}\n→ {OWN_MODEL_MAP[label]}", alpha=0.85)
        ax3.fill_between(fz_grid, mean_p - std_p, mean_p + std_p,
                         alpha=0.22, color="gray", label=f"±1σ  (mean CV = {mean_cv:.1f} %)")
        ax3.plot(fz_grid, mean_p, color="black", lw=1.6, ls="--", label="Mean pred")
        ax3.plot(fz_grid, fz_grid, color=GT_COLOR, lw=1.2, ls=":", label="Ideal (y=x)")
        ax3.set_xlabel("GT $|F_z|$ [N]", fontsize=10)
        ax3.set_ylabel("Pred $|F_z|$ [N]", fontsize=10)
        ax3.set_title(
            f"Force-domain alignment\n(CV metric — lower = more repeatable)",
            fontsize=11,
        )
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(bottom=0)
        ax3.set_xlim(left=fz_grid[0])

        fig_c.tight_layout()
        out_c = OUT_DIR / "cross_sensor_C_repeatability.png"
        fig_c.savefig(out_c, dpi=250, bbox_inches="tight")
        plt.close(fig_c)
        print(f"\nSaved Plot C → {out_c}")

        # ── PLOT D — Same as C but with the GENERAL model ─────────────────
        # Force-domain for general model
        force_data_gen = {}
        for label in available_labels:
            entry  = results[label]
            gt     = entry["gt"]
            gen_p  = entry["gen_pred"]
            mask   = gt > CONTACT_THR
            if mask.sum() < 20:
                continue
            force_data_gen[label] = (gt[mask], gen_p[mask])

        if len(force_data_gen) >= 2:
            max_g = min(fd[0].max() for fd in force_data_gen.values())
            min_g = max(fd[0].min() for fd in force_data_gen.values())
            if max_g <= min_g:
                max_g = max(fd[0].max() for fd in force_data_gen.values())
                min_g = min(fd[0].min() for fd in force_data_gen.values())
            fz_grid_gen = np.linspace(min_g, max_g, FZ_GRID_N)

            grid_gen = {}
            for label, (gt_c, pred_c) in force_data_gen.items():
                idx   = np.argsort(gt_c)
                grid_gen[label] = np.interp(fz_grid_gen, gt_c[idx], pred_c[idx])

            stk_gen  = np.vstack([grid_gen[l] for l in available_labels if l in grid_gen])
            m_gen    = stk_gen.mean(axis=0)
            s_gen    = stk_gen.std(axis=0)
            cv_gen   = (s_gen / (m_gen + 1e-9)) * 100.0
            mean_cv_gen = float(np.nanmean(cv_gen))

            pairs_gen = []
            for i in range(len(available_labels)):
                for j in range(i + 1, len(available_labels)):
                    li, lj = available_labels[i], available_labels[j]
                    if li in grid_gen and lj in grid_gen:
                        r = float(np.sqrt(np.mean((grid_gen[li] - grid_gen[lj]) ** 2)))
                        pairs_gen.append((li, lj, r))

            print(f"\n  General model (v5.010):")
            print(f"  Mean CV : {mean_cv_gen:.2f} %")
            print(f"  Pairwise RMSE on force-aligned general predictions [N]:")
            for li, lj, r in pairs_gen:
                print(f"    {li}  vs  {lj} :  RMSE = {r:.4f} N")

            fig_d, axes_d = plt.subplots(1, 3, figsize=(15, 5))
            fig_d.suptitle(
                f"Cross-sensor repeatability — GENERAL model (v5.010)  |  "
                f"Mean CV = {mean_cv_gen:.1f} %",
                fontsize=12,
            )

            # Panel 1 — GT (same as Plot C)
            ax1d = axes_d[0]
            for label, col in zip(available_labels, SENSOR_COLORS):
                if label in norm_gt:
                    ax1d.plot(norm_grid, norm_gt[label], color=col, lw=1.8,
                              label=label, alpha=0.85)
            if stacked_gt.shape[0] >= 2:
                ax1d.fill_between(norm_grid, m_gt - s_gt, m_gt + s_gt,
                                  alpha=0.18, color="gray")
                ax1d.plot(norm_grid, m_gt, color="black", lw=1.5, ls="--",
                          label="Mean GT")
            ax1d.set_xlabel("Normalised time", fontsize=10)
            ax1d.set_ylabel("GT $|F_z|$ [N]", fontsize=10)
            ax1d.set_title("Ground truth\n(ATI wrench)", fontsize=11)
            ax1d.legend(fontsize=8)
            ax1d.grid(True, alpha=0.3)
            ax1d.set_ylim(bottom=0)

            # Panel 2 — General model predictions on normalised time
            ax2d = axes_d[1]
            stk_norm_gen = np.vstack([norm_gen[l] for l in available_labels if l in norm_gen])
            m_ng = stk_norm_gen.mean(axis=0) if stk_norm_gen.shape[0] >= 2 else None
            s_ng = stk_norm_gen.std(axis=0)  if stk_norm_gen.shape[0] >= 2 else None
            for label, col in zip(available_labels, SENSOR_COLORS):
                if label in norm_gen:
                    ax2d.plot(norm_grid, norm_gen[label], color=col, lw=1.8,
                              label=label, alpha=0.85)
            if m_ng is not None:
                ax2d.fill_between(norm_grid, m_ng - s_ng, m_ng + s_ng,
                                  alpha=0.18, color="gray")
                ax2d.plot(norm_grid, m_ng, color="black", lw=1.5, ls="--",
                          label="Mean pred")
            ax2d.set_xlabel("Normalised time", fontsize=10)
            ax2d.set_ylabel("Pred $|F_z|$ [N]", fontsize=10)
            ax2d.set_title("General model predictions\n(normalised time)", fontsize=11)
            ax2d.legend(fontsize=8)
            ax2d.grid(True, alpha=0.3)
            ax2d.set_ylim(bottom=0)

            # Panel 3 — Force-domain CV for general model
            ax3d = axes_d[2]
            for label, col in zip(available_labels, SENSOR_COLORS):
                if label in grid_gen:
                    ax3d.plot(fz_grid_gen, grid_gen[label], color=col, lw=1.6,
                              label=label, alpha=0.85)
            ax3d.fill_between(fz_grid_gen, m_gen - s_gen, m_gen + s_gen,
                              alpha=0.22, color="gray",
                              label=f"±1σ  (mean CV = {mean_cv_gen:.1f} %)")
            ax3d.plot(fz_grid_gen, m_gen, color="black", lw=1.6, ls="--",
                      label="Mean pred")
            ax3d.plot(fz_grid_gen, fz_grid_gen, color=GT_COLOR, lw=1.2, ls=":",
                      label="Ideal (y=x)")
            ax3d.set_xlabel("GT $|F_z|$ [N]", fontsize=10)
            ax3d.set_ylabel("Pred $|F_z|$ [N]", fontsize=10)
            ax3d.set_title(
                "Force-domain alignment\n(general model)", fontsize=11
            )
            ax3d.legend(fontsize=7)
            ax3d.grid(True, alpha=0.3)
            ax3d.set_ylim(bottom=0)
            ax3d.set_xlim(left=fz_grid_gen[0])

            fig_d.tight_layout()
            out_d = OUT_DIR / "cross_sensor_D_repeatability_general.png"
            fig_d.savefig(out_d, dpi=250, bbox_inches="tight")
            plt.close(fig_d)
            print(f"Saved Plot D → {out_d}")

            # ── Summary table ─────────────────────────────────────────────────
            print("\n" + "=" * 58)
            print("SUMMARY — Repeatability CV")
            print(f"  {'Model':<22}  {'Mean CV [%]':>12}  {'Max CV [%]':>10}")
            print(f"  {'Sensor-specific':<22}  {mean_cv:>12.2f}  {max_cv:>10.2f}")
            print(f"  {'General (v5.010)':<22}  {mean_cv_gen:>12.2f}")
            print("=" * 58)
            print("\n  Interpretation: CV = std / mean × 100 across the three sensors")
            print("  at each GT force level. Lower values indicate that all sensors")
            print("  produce consistent predictions for the same applied force.")

    print("\nDone. All outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
