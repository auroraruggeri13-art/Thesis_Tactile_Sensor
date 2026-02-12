import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- CONFIG -----------------
BASE = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data")
version_num = 5

# put 2 or 3 tests here, code adapts automatically
tests = [5761, 5861, 5961]   # e.g. [5750, 5950]

FILES_BROMETERS = {f"sensor_{t}": str(BASE / f"test {t} - sensor v{version_num}" / f"{t}barometers_trial.txt") for t in tests}
FILES_ATI = {f"sensor_{t}": str(BASE / f"test {t} - sensor v{version_num}" / f"{t}ati_middle_trial.txt") for t in tests}

BASELINE_SECONDS = 5.0
SMOOTH_WINDOW_S = 0.1   # set 0 to disable smoothing (increased for more smoothing)
SKIP_INITIAL_SECONDS = 0.5  # drop first N seconds of unreliable data
SAVE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis - Tactile Sensor\sensor_compare_out")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

P_COLS = [f"b{i}_P" for i in range(1, 7)]

# ----------------- COLORS (consistent across all plots) -----------------
names_sorted = sorted(FILES_BROMETERS.keys())
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
COL = {name: cycle[i % len(cycle)] for i, name in enumerate(names_sorted)}

# ----------------- LOADERS -----------------
def load_barometers_txt(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
        
            s = line.strip()
            if not s:
                continue
            parts = s.split(",")
            try:
                if len(parts) == 15:
                    row = {"PcTime": parts[0], "Epoch_s": float(parts[1]), "Time_ms": int(float(parts[2]))}
                    vals = list(map(float, parts[3:]))
                elif len(parts) == 13:
                    row = {"PcTime": None, "Epoch_s": np.nan, "Time_ms": int(float(parts[0]))}
                    vals = list(map(float, parts[1:]))
                else:
                    continue
                for i in range(6):
                    row[f"b{i+1}_P"] = vals[2*i + 0]
                    row[f"b{i+1}_T"] = vals[2*i + 1]
                rows.append(row)
            except Exception:
                continue
    df = pd.DataFrame(rows).dropna(subset=["Epoch_s"]).copy()
    df = df.sort_values("Epoch_s").reset_index(drop=True)
    
    # Filter out unreliable initial data
    if len(df) > 0 and SKIP_INITIAL_SECONDS > 0:
        t0 = df["Epoch_s"].iloc[0]
        df = df[df["Epoch_s"] >= t0 + SKIP_INITIAL_SECONDS].copy()
        df = df.reset_index(drop=True)
    
    return df

def load_ati_middle_txt(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(",")
            if len(parts) != 10:
                continue
            try:
                stamp_ns = int(parts[0])
                rows.append({
                    "stamp_s": stamp_ns / 1e9,
                    "Fx": float(parts[4]), "Fy": float(parts[5]), "Fz": float(parts[6]),
                    "Tx": float(parts[7]), "Ty": float(parts[8]), "Tz": float(parts[9]),
                })
            except Exception:
                continue
    df = pd.DataFrame(rows).sort_values("stamp_s").reset_index(drop=True)
    return df

# ----------------- BASELINE + SYNC -----------------
def baseline_per_channel(df_baro: pd.DataFrame, baseline_seconds: float) -> pd.Series:
    """Baseline from first baseline_seconds only (pre-contact)."""
    t0 = df_baro["Epoch_s"].iloc[0]
    base = df_baro[df_baro["Epoch_s"] <= t0 + baseline_seconds]
    return base[P_COLS].mean()

def first_event_window_from_fz(ati_sync, t_col="t_sync", fz_col="Fz",
                               base_s=1.0, smooth_s=0.05,
                               contact_thr=0.8, release_thr=0.4,
                               hold_s=0.08):
    """Robust first event detection from ATI Fz: returns (t_start, t_end) or None."""
    t = ati_sync[t_col].to_numpy()
    fz = ati_sync[fz_col].to_numpy()

    # baseline on first base_s seconds
    m0 = t <= (t[0] + base_s)
    fz0 = fz[m0].mean() if np.any(m0) else fz[0]
    fz = fz - fz0

    # smooth
    dt = np.median(np.diff(t))
    n = max(3, int(round(smooth_s / dt)))
    if n % 2 == 0: n += 1
    fzs = pd.Series(fz).rolling(n, center=True, min_periods=max(2, n//3)).mean()
    fzs = fzs.interpolate(limit_direction="both").to_numpy()

    hold_n = max(1, int(round(hold_s / dt)))

    # detect first sustained fz < -contact_thr (negative load)
    contact = fzs < (-contact_thr)
    run = np.convolve(contact.astype(int), np.ones(hold_n, dtype=int), mode="same") >= hold_n
    idx = np.where(run)[0]
    if len(idx) == 0:
        return None

    i_start = int(idx[0])

    # end: first sustained |fz| < release_thr after start
    near0 = np.abs(fzs) < release_thr
    run2 = np.convolve(near0.astype(int), np.ones(hold_n, dtype=int), mode="same") >= hold_n
    idx2 = np.where(run2 & (np.arange(len(t)) > i_start))[0]
    if len(idx2) == 0:
        i_end = len(t) - 1
    else:
        i_end = int(idx2[0])

    return float(t[i_start]), float(t[i_end])

def sync_time_first_contact_peak(
    df_ati,
    skip_s=1.0,
    smooth_s=0.05,
    contact_thresh=0.4,   # N, after baseline removal (tune)
    hold_s=0.08,          # must stay beyond threshold for this long
    peak_window_s=0.8     # search first peak within this window after contact
):
    """Two-stage sync: detect first contact, then find first local extremum after contact."""
    t = df_ati["stamp_s"].to_numpy()
    fz = df_ati["Fz"].to_numpy()

    # skip beginning
    m = t >= (t[0] + skip_s)
    t, fz = t[m], fz[m]

    # baseline remove using first 1s
    m0 = t <= (t[0] + 1.0)
    fz0 = fz[m0].mean() if np.any(m0) else fz[0]
    fz = fz - fz0

    # smooth
    dt = np.median(np.diff(t))
    n = max(3, int(round(smooth_s / dt)))
    if n % 2 == 0:
        n += 1
    fzs = (
        pd.Series(fz)
        .rolling(n, center=True, min_periods=max(2, n // 3))
        .mean()
        .interpolate(limit_direction="both")
        .to_numpy()
    )

    # contact detection: exceed threshold and stay exceeded for hold_s
    hold_n = max(1, int(round(hold_s / dt)))
    above = np.abs(fzs) >= contact_thresh
    run = np.convolve(above.astype(int), np.ones(hold_n, dtype=int), mode="same") >= hold_n

    idx_contact = np.where(run)[0]
    if len(idx_contact) == 0:
        # fallback: largest event
        return float(t[int(np.argmax(np.abs(fzs)))])

    i0 = int(idx_contact[0])

    # decide direction at contact (negative or positive)
    sign = -1.0 if fzs[i0] < 0 else 1.0

    # search for first local extremum in a window after contact
    i1 = min(len(t) - 2, i0 + int(round(peak_window_s / dt)))
    y = sign * fzs  # convert to "positive peak" problem

    for i in range(max(1, i0 + 1), i1):
        if (y[i] > y[i - 1]) and (y[i] > y[i + 1]):
            return float(t[i])

    # fallback: max in the window
    j = i0 + int(np.argmax(y[i0:i1 + 1]))
    return float(t[j])

# ----------------- SMOOTHING -----------------
def smooth_df_time(df: pd.DataFrame, t_col: str, y_cols: list, window_s: float) -> pd.DataFrame:
    if window_s is None or window_s <= 0:
        return df
    dt = np.median(np.diff(df[t_col].values))
    if not np.isfinite(dt) or dt <= 0:
        return df
    n = int(round(window_s / dt))
    n = max(3, n)
    if n % 2 == 0:
        n += 1
    out = df.copy()
    for c in y_cols:
        out[c] = out[c].rolling(n, center=True, min_periods=max(2, n//3)).mean()
        out[c] = out[c].interpolate(limit_direction="both")
    return out

# ----------------- PIPELINE -----------------
data = {}

for name, baro_path in FILES_BROMETERS.items():
    baro_path = Path(baro_path)
    ati_path = Path(FILES_ATI[name])

    if not baro_path.exists():
        raise FileNotFoundError(str(baro_path))
    if not ati_path.exists():
        raise FileNotFoundError(f"Missing ATI file: {ati_path}")

    baro = load_barometers_txt(str(baro_path))
    ati = load_ati_middle_txt(str(ati_path))

    base = baseline_per_channel(baro, BASELINE_SECONDS)
    t_sync0 = sync_time_first_contact_peak(
        ati,
        skip_s=1.0,
        smooth_s=0.05,
        contact_thresh=0.4,
        hold_s=0.08,
        peak_window_s=0.8
    )

    baro_zero = baro.copy()
    for ch in P_COLS:
        baro_zero[ch] = baro_zero[ch] - base[ch]

    baro_zero["t_sync"] = baro_zero["Epoch_s"] - t_sync0
    ati_sync = ati.copy()
    ati_sync["t_sync"] = ati_sync["stamp_s"] - t_sync0

    # smooth the zeroed pressure (and optionally ATI Fz if you want)
    baro_zero = smooth_df_time(baro_zero, "t_sync", P_COLS, SMOOTH_WINDOW_S)

    data[name] = {"baro_zero": baro_zero, "ati": ati_sync, "baseline": base, "t_sync0": t_sync0}

print("\nFirst contact + peak times used for sync (epoch seconds):")
for name, d in data.items():
    print(name, d["t_sync0"])

# ----------------- PLOTS (SHOW BEFORE SAVE) -----------------

# 1) Baselines per channel: same colors as other plots
fig1, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.ravel()
x0, x1 = 0.0, 1.0

for i, ch in enumerate(P_COLS):
    ax = axes[i]
    for name in names_sorted:
        if name not in data:
            continue
        y = float(data[name]["baseline"][ch])
        ax.plot([x0, x1], [y, y], color=COL[name], linewidth=2, label=name)
    ax.set_title(f"Baseline {ch}")
    ax.set_xticks([])
    ax.set_ylabel("Pressure (kPa)")
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.show()
fig1.savefig(SAVE_DIR / "baseline_per_channel.png", dpi=200)

# 2) ATI Fz check (peaks aligned at t=0) same colors
figF, axF = plt.subplots(1, 1, figsize=(10, 3))
for name in names_sorted:
    if name not in data:
        continue
    axF.plot(data[name]["ati"]["t_sync"], data[name]["ati"]["Fz"], color=COL[name], label=name)
axF.axvline(0, linestyle="--")
axF.set_title("ATI Fz (synced to first contact peak at t=0)")
axF.set_xlabel("t_sync (s)")
axF.set_ylabel("Fz")
axF.legend()
plt.tight_layout()
plt.show()
figF.savefig(SAVE_DIR / "ati_fz_synced.png", dpi=200)

# 3) Zeroed channels overlay: one subplot per channel, same colors
fig2, axes2 = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
axes2 = axes2.ravel()

for i, ch in enumerate(P_COLS):
    ax = axes2[i]
    for name in names_sorted:
        if name not in data:
            continue
        dfz = data[name]["baro_zero"]
        ax.plot(dfz["t_sync"], dfz[ch], color=COL[name], label=name)
    ax.axvline(0, linestyle="--")
    ax.set_title(f"Zeroed {ch}")
    ax.set_xlabel("t_sync (s)")
    ax.set_ylabel("Delta P (kPa)")
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.show()
fig2.savefig(SAVE_DIR / "zeroed_channels_synced_overlay.png", dpi=200)

# =========================
# PLOT OPTION 2: normalized time (0..1) on first event
# =========================

def resample_norm(x, y, n=300):
    """Resample (x, y) to normalized time 0..1 with n points."""
    x = np.asarray(x); y = np.asarray(y)
    u = (x - x[0]) / (x[-1] - x[0] + 1e-12)
    ug = np.linspace(0, 1, n)
    yg = np.interp(ug, u, y)
    return ug, yg

# compute windows once
event_win = {}
for name in names_sorted:
    win = first_event_window_from_fz(data[name]["ati"], contact_thr=0.8, release_thr=0.4)
    event_win[name] = win

figN, axesN = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
axesN = axesN.ravel()

for i, ch in enumerate(P_COLS):
    ax = axesN[i]

    # collect for consistent ylim
    ys_all = []
    curves = []
    for name in names_sorted:
        win = event_win[name]
        if win is None:
            continue
        t0, t1 = win
        seg = data[name]["baro_zero"]
        seg = seg[(seg["t_sync"] >= t0) & (seg["t_sync"] <= t1)]
        if len(seg) < 10:
            continue
        u, y = resample_norm(seg["t_sync"].to_numpy(), seg[ch].to_numpy(), n=300)
        curves.append((name, u, y))
        ys_all.append(y)

    for name, u, y in curves:
        ax.plot(u, y, color=COL[name], label=name)

    if ys_all:
        ycat = np.concatenate(ys_all)
        lo, hi = np.nanpercentile(ycat, [2, 98])
        pad = 0.1 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_title(f"{ch} first event (normalized time)")
    ax.set_xlabel("0..1")
    ax.set_ylabel("deltaP (kPa)")
    ax.grid(True, alpha=0.25)

handles, labels = axesN[0].get_legend_handles_labels()
figN.legend(handles, labels, loc="upper right")
plt.tight_layout()
plt.show()
figN.savefig(SAVE_DIR / "option2_normalized_first_event.png", dpi=200)


# =========================
# PLOT OPTION 3: hysteresis deltaP vs Fz (first event)
# =========================

figH, axesH = plt.subplots(2, 3, figsize=(13, 6))
axesH = axesH.ravel()

for i, ch in enumerate(P_COLS):
    ax = axesH[i]
    xs_all, ys_all = [], []

    for name in names_sorted:
        win = event_win[name]
        if win is None:
            continue
        t0, t1 = win

        b = data[name]["baro_zero"]
        a = data[name]["ati"]

        segb = b[(b["t_sync"] >= t0) & (b["t_sync"] <= t1)]
        if len(segb) < 10:
            continue

        fz_interp = np.interp(segb["t_sync"].to_numpy(),
                              a["t_sync"].to_numpy(),
                              a["Fz"].to_numpy())

        x = fz_interp
        y = segb[ch].to_numpy()

        ax.plot(x, y, color=COL[name], label=name)

        xs_all.append(x); ys_all.append(y)

    if xs_all and ys_all:
        X = np.concatenate(xs_all); Y = np.concatenate(ys_all)
        xlo, xhi = np.nanpercentile(X, [2, 98])
        ylo, yhi = np.nanpercentile(Y, [2, 98])
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

    ax.set_title(f"{ch} hysteresis (first event)")
    ax.set_xlabel("Fz")
    ax.set_ylabel("deltaP (kPa)")
    ax.grid(True, alpha=0.25)

handles, labels = axesH[0].get_legend_handles_labels()
figH.legend(handles, labels, loc="upper right")
plt.tight_layout()
plt.show()
figH.savefig(SAVE_DIR / "option3_hysteresis_first_event.png", dpi=200)

