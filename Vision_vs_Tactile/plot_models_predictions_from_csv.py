#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot predicted vs ground-truth force time series.
Two separate figures (Left sensor / Right sensor), each with 3x1 subplots (fx, fy, fz).

  - fff_clean_forces_for_free  -> teal   (Vision model)
  - lightxgb                  -> navy   (Tactile model)
  - fitf_tmm_fusion           -> green  (Fusion model)
  - Ground Truth              -> yellow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── paths ─────────────────────────────────────────────────────────────────────
DOWNLOADS = r"C:\Users\aurir\Downloads"

FFF_LEFT           = rf"{DOWNLOADS}\fff_clean_forces_for_free_left_test_predictions.csv"
FFF_RIGHT          = rf"{DOWNLOADS}\fff_clean_forces_for_free_right_test_predictions.csv"
XGB_LEFT           = rf"{DOWNLOADS}\lightxgb_test_predictions_L.csv"
XGB_RIGHT          = rf"{DOWNLOADS}\lightxgb_test_predictions_R.csv"
FITF_TMM_FUSION_LEFT  = rf"{DOWNLOADS}\fitf_fusion_test_predictions_L.csv"
FITF_TMM_FUSION_RIGHT = rf"{DOWNLOADS}\fitf_fusion_test_predictions_R.csv"

# ── config ────────────────────────────────────────────────────────────────────
START_OFFSET_S  = 0   # seconds after the data's own t0 to start plotting 287.4
PLOT_DURATION_S = 6.5     # how many seconds to plot after the offset 5.6
GT_COLOR   = "#d6c52e"
FFF_COLOR  = "#008780"
XGB_COLOR  = "#292f56"
FITF_COLOR = "#44b155"
GT_LW   = 2   # ground truth: thick solid
PRED_LW = 1.5   # predictions: thinner, with distinct linestyles
MS      = 4.0   # marker size for scatter dots on lines
MEVERY  = 1     # plot a marker at every data point


# ── helpers ───────────────────────────────────────────────────────────────────
def trim(df, start_s, duration_s):
    t = df["time"].values
    t_rel = t - start_s
    mask = (t_rel >= 0) & (t_rel <= duration_s)
    return t_rel[mask], mask


# ── load ──────────────────────────────────────────────────────────────────────
df_fff_L  = pd.read_csv(FFF_LEFT)
df_fff_R  = pd.read_csv(FFF_RIGHT)
df_xgb_L  = pd.read_csv(XGB_LEFT)
df_xgb_R  = pd.read_csv(XGB_RIGHT)
df_fitf_L = pd.read_csv(FITF_TMM_FUSION_LEFT)
df_fitf_R = pd.read_csv(FITF_TMM_FUSION_RIGHT)

T0 = min(df["time"].min() for df in [df_fff_L, df_fff_R, df_xgb_L, df_xgb_R, df_fitf_L, df_fitf_R])
t_max = df_fff_L["time"].max()
START_ABS = T0 + START_OFFSET_S
print(f"Data time range: {T0:.1f} → {t_max:.1f}  (span: {t_max - T0:.1f} s)")
print(f"Plotting window: {START_ABS:.1f} → {START_ABS + PLOT_DURATION_S:.1f}  (offset={START_OFFSET_S} s from data start)")

t_fff_L,  m_fff_L  = trim(df_fff_L,  START_ABS, PLOT_DURATION_S)
t_fff_R,  m_fff_R  = trim(df_fff_R,  START_ABS, PLOT_DURATION_S)
t_xgb_L,  m_xgb_L  = trim(df_xgb_L,  START_ABS, PLOT_DURATION_S)
t_xgb_R,  m_xgb_R  = trim(df_xgb_R,  START_ABS, PLOT_DURATION_S)
t_fitf_L, m_fitf_L = trim(df_fitf_L, START_ABS, PLOT_DURATION_S)
t_fitf_R, m_fitf_R = trim(df_fitf_R, START_ABS, PLOT_DURATION_S)

# ── column names ──────────────────────────────────────────────────────────────
fff_true_L  = ["fx_L_surf_true", "fy_L_surf_true", "fz_L_surf_true"]
fff_pred_L  = ["fx_L_surf_pred", "fy_L_surf_pred", "fz_L_surf_pred"]
fff_true_R  = ["fx_R_surf_true", "fy_R_surf_true", "fz_R_surf_true"]
fff_pred_R  = ["fx_R_surf_pred", "fy_R_surf_pred", "fz_R_surf_pred"]

xgb_true_L  = ["fx_L_true", "fy_L_true", "fz_L_true"]
xgb_pred_L  = ["fx_L_pred", "fy_L_pred", "fz_L_pred"]
xgb_true_R  = ["fx_R_true", "fy_R_true", "fz_R_true"]
xgb_pred_R  = ["fx_R_pred", "fy_R_pred", "fz_R_pred"]

fitf_true_L = ["fx_L_true", "fy_L_true", "fz_L_true"]
fitf_pred_L = ["fx_L_pred", "fy_L_pred", "fz_L_pred"]
fitf_true_R = ["fx_R_true", "fy_R_true", "fz_R_true"]
fitf_pred_R = ["fx_R_pred", "fy_R_pred", "fz_R_pred"]

force_labels = ["fx [N]", "fy [N]", "fz [N]"]

# columns: one per model
models = [
    ("Vision model",   FFF_COLOR),
    ("Tactile model",  XGB_COLOR),
    ("Fusion model",   FITF_COLOR),
]


legend_handles = [
    mlines.Line2D([], [], color=GT_COLOR,   lw=GT_LW,   ls='-', marker='o', ms=MS, label="Ground Truth"),
    mlines.Line2D([], [], color=FFF_COLOR,  lw=PRED_LW, ls='-', marker='o', ms=MS, label="Vision model"),
    mlines.Line2D([], [], color=XGB_COLOR,  lw=PRED_LW, ls='-', marker='o', ms=MS, label="Tactile model"),
    mlines.Line2D([], [], color=FITF_COLOR, lw=PRED_LW, ls='-', marker='o', ms=MS, label="Fusion model"),
]


# ── RMSE on full test set ─────────────────────────────────────────────────────
def component_rmse(true, pred):
    err = np.array(true) - np.array(pred)
    return float(np.sqrt(np.mean(err ** 2)))

def euclidean_rmse(true_arr, pred_arr):
    diff = np.array(true_arr) - np.array(pred_arr)          # (N, 3)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

def component_mae(true, pred):
    err = np.abs(np.array(true) - np.array(pred))
    return float(np.mean(err))

def euclidean_mae(true_arr, pred_arr):
    diff = np.array(true_arr) - np.array(pred_arr)          # (N, 3)
    return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=1))))

W = 65  # table width

for side, _df_fff, _df_xgb, _df_fitf in [
    ("Left",  df_fff_L, df_xgb_L, df_fitf_L),
    ("Right", df_fff_R, df_xgb_R, df_fitf_R),
]:
    is_left = (side == "Left")
    rows = []
    for model_name, df_m, t_cols, p_cols in [
        ("Vision",  _df_fff,  fff_true_L  if is_left else fff_true_R,  fff_pred_L  if is_left else fff_pred_R),
        ("Tactile", _df_xgb,  xgb_true_L  if is_left else xgb_true_R,  xgb_pred_L  if is_left else xgb_pred_R),
        ("Fusion",  _df_fitf, fitf_true_L if is_left else fitf_true_R, fitf_pred_L if is_left else fitf_pred_R),
    ]:
        df_clean = df_m[t_cols + p_cols].dropna()
        gt_v   = df_clean[t_cols].values.astype(float)
        pred_v = df_clean[p_cols].values.astype(float)
        rows.append((model_name, gt_v, pred_v, len(df_clean)))

    # ── RMSE table ──
    print(f"\n{'='*W}")
    print(f"  RMSE — {side} sensor   sqrt(1/N·Σ‖y-ŷ‖²)")
    print(f"{'='*W}")
    print(f"{'Model':<18} {'fx [N]':>10} {'fy [N]':>10} {'fz [N]':>10} {'Eucl. [N]':>12} {'N':>4}")
    print(f"{'-'*W}")
    for model_name, gt_v, pred_v, n in rows:
        c = [component_rmse(gt_v[:, i], pred_v[:, i]) for i in range(3)]
        e = euclidean_rmse(gt_v, pred_v)
        print(f"{model_name:<18} {c[0]:>10.4f} {c[1]:>10.4f} {c[2]:>10.4f} {e:>12.4f} {n:>4}")
    print(f"{'='*W}")

    # ── MAE table ──
    print(f"\n{'='*W}")
    print(f"  MAE  — {side} sensor   1/N·Σ‖y-ŷ‖")
    print(f"{'='*W}")
    print(f"{'Model':<18} {'fx [N]':>10} {'fy [N]':>10} {'fz [N]':>10} {'Eucl. [N]':>12} {'N':>4}")
    print(f"{'-'*W}")
    for model_name, gt_v, pred_v, n in rows:
        c = [component_mae(gt_v[:, i], pred_v[:, i]) for i in range(3)]
        e = euclidean_mae(gt_v, pred_v)
        print(f"{model_name:<18} {c[0]:>10.4f} {c[1]:>10.4f} {c[2]:>10.4f} {e:>12.4f} {n:>4}")
    print(f"{'='*W}")

# ── figure builder ─────────────────────────────────────────────────────────────
# Layout: 3×1 (fx, fy, fz), all models + GT per subplot, no markers
def make_figure(side):
    is_left = (side == "Left")
    t_fff,  m_fff  = (t_fff_L,  m_fff_L)  if is_left else (t_fff_R,  m_fff_R)
    t_xgb,  m_xgb  = (t_xgb_L,  m_xgb_L)  if is_left else (t_xgb_R,  m_xgb_R)
    t_fitf, m_fitf = (t_fitf_L, m_fitf_L) if is_left else (t_fitf_R, m_fitf_R)
    df_fff  = df_fff_L  if is_left else df_fff_R
    df_xgb  = df_xgb_L  if is_left else df_xgb_R
    df_fitf = df_fitf_L if is_left else df_fitf_R
    true_cols = fff_true_L if is_left else fff_true_R
    pred_fff  = fff_pred_L if is_left else fff_pred_R
    pred_xgb  = xgb_pred_L if is_left else xgb_pred_R
    pred_fitf = fitf_pred_L if is_left else fitf_pred_R

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(f"Time Series: Predicted vs Ground Truth — {side} sensor", fontsize=13)

    for fi in range(3):
        ax = axes[fi]
        # predictions first (underneath) — with scatter dots
        ax.plot(t_fff,  df_fff[pred_fff[fi]].values[m_fff],
                color=FFF_COLOR,  lw=3, alpha=0.85, label="Vision model",
                marker='o', ms=MS, markevery=MEVERY, zorder=3)
        ax.scatter(t_fff, df_fff[pred_fff[fi]].values[m_fff],
                   color=FFF_COLOR, s=MS**2, alpha=0.85, label="Vision model", zorder=3)
        ax.plot(t_xgb,  df_xgb[pred_xgb[fi]].values[m_xgb],
                color=XGB_COLOR,  lw=PRED_LW,  alpha=0.85, label="Tactile model",
                marker='o', ms=MS, markevery=MEVERY, zorder=3)
        ax.scatter(t_xgb, df_xgb[pred_xgb[fi]].values[m_xgb],
                   color=XGB_COLOR, s=MS**2, alpha=0.85, label="Tactile model", zorder=3)
        ax.plot(t_fitf, df_fitf[pred_fitf[fi]].values[m_fitf],
                color=FITF_COLOR, lw=PRED_LW, alpha=0.85, label="Fusion model",
                marker='o', ms=MS, markevery=MEVERY, zorder=3)
        ax.scatter(t_fitf, df_fitf[pred_fitf[fi]].values[m_fitf],
                   color=FITF_COLOR, s=MS**2, alpha=0.85, label="Fusion model", zorder=3)
        # GT on top — thick solid, fully opaque, with scatter dots
        ax.plot(t_fff,  df_fff[true_cols[fi]].values[m_fff],
                color=GT_COLOR,   lw=GT_LW,  alpha=1.0,  label="Ground Truth",
                marker='o', ms=MS, markevery=MEVERY, zorder=3.0)
        ax.scatter(t_fff, df_fff[true_cols[fi]].values[m_fff],
                   color=GT_COLOR, s=MS**2, alpha=1.0, label="Ground Truth", zorder=3.0)
        ax.set_ylabel(force_labels[fi])
        ax.set_xlim(0, PLOT_DURATION_S)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time [s]")
    fig.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── produce & save ────────────────────────────────────────────────────────────
fig_L = make_figure("Left")
fig_L.savefig(rf"{DOWNLOADS}\timeseries_left.png", dpi=200, bbox_inches="tight")
print(f"Saved: {DOWNLOADS}\\timeseries_left.png")

fig_R = make_figure("Right")
fig_R.savefig(rf"{DOWNLOADS}\timeseries_right.png", dpi=200, bbox_inches="tight")
print(f"Saved: {DOWNLOADS}\\timeseries_right.png")

plt.show()
