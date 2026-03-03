#!/usr/bin/env python3
import sys, pickle, matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils.io_utils import load_tabular_csv

BASE = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab")
DATA = BASE / "train_validation_test_data" / "train_data_v5.1894.csv"
MDIR = BASE / "models parameters" / "averaged models"
OUT  = BASE / "Sensor Characterization" / "repeatability_hysteresis"

W = 10; DN_WIN = 5; MAX_GAP = 0.05

BARO = ["b1","b2","b3","b4","b5","b6"]
COLS = ["t","b1","b2","b3","b4","b5","b6","x","y","fx","fy","fz","tx","ty","tz"]

# ── Raw GT (all rows) ─────────────────────────────────────────────────────────
raw = load_tabular_csv(str(DATA), COLS)
raw = raw.sort_values("t").reset_index(drop=True)
t0     = raw["t"].values[0]
raw_t  = raw["t"].values - t0
raw_gt = np.abs(raw["fz"].values)
print(f"File: {len(raw)} rows, t={raw_t[-1]:.1f}s, max|Fz|={raw_gt.max():.2f}N")

# Cut at 8 s
_cut = raw_t <= 8.0
raw_t  = raw_t[_cut]
raw_gt = raw_gt[_cut]

# ── Predictions ───────────────────────────────────────────────────────────────
df = raw.copy()
for c in BARO:
    df[c] = df[c].rolling(DN_WIN, center=True).mean().bfill().ffill()
for c in BARO:
    d1 = df[c].diff().fillna(0.0)
    df[f"{c}_d1"] = d1
    df[f"{c}_d2"] = d1.diff().fillna(0.0)

tv  = df["t"].values
bv  = df[BARO].values
d1v = df[[f"{c}_d1" for c in BARO]].values
d2v = df[[f"{c}_d2" for c in BARO]].values

X_lst, idx_lst = [], []
for i in range(W, len(df)):
    s, e = i - W, i + 1
    if np.max(np.diff(tv[s:e])) > MAX_GAP:
        continue
    X_lst.append(np.concatenate([bv[s:e].flatten(),
                                  d1v[s:e].flatten(),
                                  d2v[s:e].flatten()]))
    idx_lst.append(i)

X   = np.array(X_lst)
adf = df.iloc[idx_lst].reset_index(drop=True)

with open(MDIR / "lightgbm_sliding_window_model_v5.180.pkl", "rb") as f:
    models = pickle.load(f)
with open(MDIR / "scaler_sliding_window_v5.180.pkl", "rb") as f:
    scaler = pickle.load(f)

pred_t  = adf["t"].values - t0
pred_fz = np.abs(models[4].predict(scaler.transform(X)))
print(f"Predictions: {len(pred_fz)} windows, max pred={pred_fz.max():.2f}N")

MA_WIN       = 21
TIME_SHIFT_S = -0.00   # seconds — shift predictions forward (+) or backward (-)

pred_fz_sm = np.convolve(pred_fz, np.ones(MA_WIN) / MA_WIN, mode="same")

pred_t = pred_t + TIME_SHIFT_S   # manual time shift

_cut_p = pred_t <= 8.0
pred_t     = pred_t[_cut_p]
pred_fz_sm = pred_fz_sm[_cut_p]

# ── Plot 1: Time series ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(raw_t,  raw_gt,  color="#292f56", lw=2.0, label="GT $|F_z|$")
ax.plot(pred_t, pred_fz_sm, color="#008780", lw=1.2, alpha=0.85, label="Pred $|F_z|$")
ax.set_xlabel("Time [s]", fontsize=12)
ax.set_ylabel("$|F_z|$ [N]", fontsize=12)
ax.set_title("train_data_v5.1894 — GT vs Predicted $|F_z|$", fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "hysteresis_timeseries.png", dpi=250, bbox_inches="tight")
plt.close(fig)
print("Saved: hysteresis_timeseries.png")

# ── Plot 2: GT vs Pred — one color per cycle ─────────────────────────────────
from scipy.signal import find_peaks

pred_on_raw = np.interp(raw_t, pred_t, pred_fz_sm)
PALETTE = ["#292f56", "#008780", "#44b155"]   # navy, teal, green — codebase palette
THRESH  = 0.15
lim = max(raw_gt.max(), pred_on_raw.max()) * 1.05

gt_peaks, _ = find_peaks(raw_gt, height=0.5, distance=20)
gt_peaks = gt_peaks[:3]   # keep only first 3 cycles
print(f"GT peaks: {len(gt_peaks)}")

fig2, ax = plt.subplots(figsize=(7, 6))
for i, pk in enumerate(gt_peaks):
    pre  = np.where(raw_gt[:pk] < THRESH)[0]
    post = np.where(raw_gt[pk:] < THRESH)[0]
    s = pre[-1]  if len(pre)  else 0
    e = pk + post[0] if len(post) else len(raw_gt) - 1
    gt_seg   = np.concatenate([[0], raw_gt[s:e+1],    [0]])
    pred_seg = np.concatenate([[0], pred_on_raw[s:e+1], [0]])
    ax.plot(gt_seg, pred_seg,
            color=PALETTE[i % len(PALETTE)], lw=2.0, label=f"Cycle {i+1}")

ax.plot([0, lim], [0, lim], color="gray", lw=1.0, ls="--", label="Ideal (y = x)")
ax.set_xlabel("GT $|F_z|$ [N]", fontsize=12)
ax.set_ylabel("Pred $|F_z|$ [N]", fontsize=12)
ax.set_title("train_data_v5.1894 — GT vs Pred $|F_z|$", fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
fig2.tight_layout()
fig2.savefig(OUT / "hysteresis_loops.png", dpi=250, bbox_inches="tight")
plt.close(fig2)
print("Saved: hysteresis_loops.png")
