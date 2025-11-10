#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synchronize barometer, ATI F/T, and Atracsys fiducials; compute contact position from fiducials; export a CSV.

Author: Aurora Ruggeri
"""

import os
import re
import math
from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


# =========================
# ======  CONFIG  =========
# =========================

# --- Paths (edit these) ---
BARO_DIR = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\Sensor-Logs"
ATI_DIR  = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\ATI-FT-Data-Logs"
ATR_DIR  = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\atracsys data"

BARO_FILENAME = "datalog_2025-10-23_14-08-43.csv"   # e.g. 'datalog_YYYY-MM-DD_HH-MM-SS.csv'
ATI_FILENAME  = "ati_middle_trial6.txt"             # ROS-style CSV (with %time) or ATI RDT export
ATR_FILENAME  = "atracsys_trial6.txt"               # Atracsys CSV with marker poses

# --- Output ---
OUTPUT_DIR  = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\synchronized Data"
OUTPUT_NAME = "synchronized_events.csv"

# --- Which timebase to keep (recommended: 'ati') ---
TIMEBASE = "ati"   # 'ati' | 'atr' | 'baro'

# --- Time alignments ---
AUTO_ESTIMATE_BARO_SHIFT = True   # cross-correlate Σbaro vs Fz to estimate baro↔ATI offset
BARO_SHIFT_SECONDS = 0.0          # manual override (applied in addition to AUTO if desired)
ATI_SHIFT_SECONDS  = 0.0          # extra shift if you know ATI clock offset
ATR_SHIFT_SECONDS  = 0.0          # extra shift for Atracsys

# nearest-neighbor tolerance (pose→timebase). 50–100 ms is typical for motion that is not too fast.
MERGE_TOLERANCE_MS = 100

# --- Marker IDs ---
# You can set these explicitly or let the script guess (sensor = most stationary, indenter = most dynamic)
AUTO_GUESS_MARKER_IDS = True
SENSOR_MARKER_ID: Optional[int]   = None  # e.g. 1020
INDENTER_MARKER_ID: Optional[int] = None  # e.g. 991000000

# --- Constant marker→body transforms and tip offset (enter your calibration) ---

# Sensor marker frame (M): +x = down, +y = right, +z = out-of-box (RH)
# Sensor frame (S): x_S = -y_box, y_S = +z_box
SENSOR_MARKER_TO_SENSOR = dict(
    # 4.5 cm higher (−x_M), 4.0 cm "behind" (opposite of right → −y_M)
    translation_mm=[-45.0, -40.0, 0.0],
    # Rotation M → S (Euler xyz): Rx(-90) • Ry(-90) • Rz(0)
    euler_deg=[-90.0, -90.0, 0.0],
    order="xyz"
)

# Indenter marker: +x = up, (assume) +y = left, +z = out-of-page
INDENTER_MARKER_TO_INDENTER = dict(
    translation_mm=[0.0, 0.0, 0.0],   # marker == indenter frame
    euler_deg=[0.0, 0.0, 0.0],
    order="xyz"
)

# Tip offset in the indenter (I) frame:
# 11 cm below → −x_I (since +x_I is up), 3.5 cm behind → −y_I (opposite of left)
TIP_OFFSET_I_MM = [-110.0, -35.0, 0.0]


# --- Units / formatting ---
# If your ATI torques are in N·m and you want N·mm, set TORQUE_SCALE = 1000.0
TORQUE_SCALE = 1.0

# Export columns: set to True to keep your old names ('-x (mm)', '-y (mm)'); False -> 'cx_mm','cy_mm','cz_mm'
CSV_COMPAT_NAMES = True

# --- Path plot rectangle crop (mm) ---
# Width in x (mm) and height in y (mm)
PATH_RECT_WIDTH_MM = 45.0
PATH_RECT_HEIGHT_MM = 20.0
# Rectangle center (cx, cy) in mm (sensor frame). Use (0,0) for sensor origin.
PATH_RECT_CENTER_X_MM = 0.0
PATH_RECT_CENTER_Y_MM = 0.0

# =========================
# ======  HELPERS  ========
# =========================

def euler_deg_to_R(euler_deg: List[float], order: str = "xyz") -> np.ndarray:
    """Build rotation matrix from Euler angles in degrees; order is a sequence of 'x','y','z'."""
    ax = dict(x=0, y=1, z=2)
    angles = np.deg2rad(np.array(euler_deg, dtype=float))
    R = np.eye(3)
    for i, ch in enumerate(order.lower()):
        a = angles[i]
        c, s = math.cos(a), math.sin(a)
        if ch == 'x':
            Ri = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif ch == 'y':
            Ri = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif ch == 'z':
            Ri = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError("Euler order must use only x,y,z")
        R = R @ Ri
    return R

def make_T(R: np.ndarray, t_mm: List[float]) -> np.ndarray:
    """Homogeneous transform from R (3x3) and translation in mm (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(t_mm, dtype=float).reshape(3)
    return T

def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def apply_T(T: np.ndarray, p_mm: List[float]) -> np.ndarray:
    p = np.array(list(p_mm) + [1.0], dtype=float)
    out = T @ p
    return out[:3]

# =========================
# ======  READERS  ========
# =========================

def read_ati_auto(path: str) -> pd.DataFrame:
    """
    Auto-detect ATI format.
    Returns DataFrame indexed by epoch ns ('time_ns'), columns: fx,fy,fz,tx,ty,tz (float).
    Recognizes:
      - ROS wrench CSV with '%time' and 'field.wrench.force.x/y/z', 'field.wrench.torque.x/y/z'. :contentReference[oaicite:2]{index=2}
      - ATI RDT export (Status, Fx..Tz, Time) like your earlier script. :contentReference[oaicite:3]{index=3}
    """
    # Peek the file
    head = ""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        head = "".join([next(f) for _ in range(20)])

    if '%time' in head and 'wrench' in head:
        # ROS style
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        tcol = next(c for c in df.columns if c.startswith('%time'))
        ren = {}
        for k,v in {
            'field.wrench.force.x':'fx','field.wrench.force.y':'fy','field.wrench.force.z':'fz',
            'field.wrench.torque.x':'tx','field.wrench.torque.y':'ty','field.wrench.torque.z':'tz',
            'wrench.force.x':'fx','wrench.force.y':'fy','wrench.force.z':'fz',
            'wrench.torque.x':'tx','wrench.torque.y':'ty','wrench.torque.z':'tz',
            'Fx':'fx','Fy':'fy','Fz':'fz','Tx':'tx','Ty':'ty','Tz':'tz'
        }.items():
            if k in df.columns: ren[k]=v
        df = df.rename(columns=ren)
        keep = [tcol] + [c for c in ['fx','fy','fz','tx','ty','tz'] if c in df.columns]
        df = df[keep].copy()
        df[tcol] = pd.to_numeric(df[tcol], errors='coerce', downcast='integer')
        for c in keep[1:]: df[c]=pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=[tcol]).sort_values(tcol)
        df = df.set_index(tcol)
        df.index.name = 'time_ns'
        return df

    # ATI RDT export (as in your previous script)
    # Find header row containing 'Status' and 'Fx' and 'Time' etc. :contentReference[oaicite:4]{index=4}
    header_row = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if 'Status' in line and 'Fx' in line and 'Time' in line:
                header_row = i
                break
    if header_row is None:
        raise ValueError("Unrecognized ATI format (neither ROS wrench nor RDT found)")

    cols = ['Status (hex)','RDTSequence','F/T Sequence','Fx','Fy','Fz','Tx','Ty','Tz','Time']
    df = pd.read_csv(path, skiprows=header_row, names=cols, usecols=range(10), engine='python')
    df = df[df['Time'].astype(str).str.strip().ne('')].copy()
    df['Time'] = df['Time'].astype(str).str.strip()

    file_date = os.path.basename(path).split('_')[0]
    df['datetime'] = pd.to_datetime(file_date + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()

    for c in ['Fx','Fy','Fz','Tx','Ty','Tz']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Fx','Fy','Fz','Tx','Ty','Tz'])

    # Convert to epoch ns index
    df['time_ns'] = df.index.view('int64')
    out = df[['Fx','Fy','Fz','Tx','Ty','Tz','time_ns']].copy()
    out = out.set_index('time_ns')
    out = out.rename(columns={'Fx':'fx','Fy':'fy','Fz':'fz','Tx':'tx','Ty':'ty','Tz':'tz'})
    return out

def read_atracsys_csv(path: str) -> pd.DataFrame:
    """
    Read Atracsys CSV with: %time, field.marker_id, position0..2, rotation0..8, etc.
    Returns DataFrame indexed by 'time_ns', columns: marker_id, px,py,pz, R00..R22. :contentReference[oaicite:5]{index=5}
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    tcol = next((c for c in df.columns if c.startswith('%time')), None)
    mid  = next((c for c in df.columns if 'marker_id' in c), None)
    pos_cols = [c for c in df.columns if re.search(r'position[0-2]$', c)]
    rot_cols = [c for c in df.columns if re.search(r'rotation[0-8]$', c)]
    if tcol is None or mid is None or len(pos_cols)!=3 or len(rot_cols)!=9:
        raise ValueError("Atracsys file not recognized (need %time, marker_id, position0..2, rotation0..8).")

    keep = [tcol, mid] + pos_cols + rot_cols
    df = df[keep].copy()
    df[tcol] = pd.to_numeric(df[tcol], errors='coerce', downcast='integer')
    df[mid]  = pd.to_numeric(df[mid], errors='coerce', downcast='integer')
    for c in pos_cols+rot_cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=[tcol, mid]).sort_values(tcol)

    ren = {tcol:'time_ns', mid:'marker_id'}
    for i,c in enumerate(pos_cols): ren[c] = ['px','py','pz'][i]
    for i,c in enumerate(rot_cols):
        rname = f'R{int(i/3)}{i%3}'
        ren[c] = rname
    df = df.rename(columns=ren).set_index('time_ns')
    return df

def read_barometer_csv(path: str) -> pd.DataFrame:
    """
    Read Arduino/MCU barometer CSV: 'Timestamp' plus 6 channels (e.g. 'barometer 1'..'barometer 6').
    The date is taken from the filename 'datalog_YYYY-MM-DD_HH-MM-SS.csv'.
    Returns DF indexed by epoch ns ('time_ns', UTC), columns: b1..b6 (hPa).
    """
    import re
    import pandas as pd
    import numpy as np

    LOCAL_TZ = 'America/New_York'  # <- set your local timezone here

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Timestamp column (HH:MM:SS.sss[sss])
    ts_col = next((c for c in df.columns if c.lower().startswith('timestamp')), None)
    if ts_col is None:
        raise ValueError("Barometer CSV must have a 'Timestamp' column (HH:MM:SS.sss).")

    # Infer date from filename
    m = re.search(r'datalog_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}', os.path.basename(path))
    if not m:
        raise ValueError("Filename must contain date like 'datalog_YYYY-MM-DD_hh-mm-ss.csv'")
    date_str = m.group(1)

    # Build LOCAL time → then convert to UTC → then to epoch ns
    dt_local = pd.to_datetime(
        date_str + ' ' + df[ts_col].astype(str),
        format='%Y-%m-%d %H:%M:%S.%f',
        errors='coerce'
    ).dt.tz_localize(LOCAL_TZ, nonexistent='NaT', ambiguous='NaT')
    dt_utc = dt_local.dt.tz_convert('UTC')

    df = df.loc[~dt_utc.isna()].copy()
    df['time_ns'] = dt_utc.view('int64')  # epoch ns in UTC
    df = df.drop_duplicates(subset=['time_ns']).sort_values('time_ns')

    # Map pressure columns to b1..b6
    bcols = []
    for i in range(1,7):
        if f'barometer {i}' in df.columns: bcols.append(f'barometer {i}')
        elif f'b{i}' in df.columns:        bcols.append(f'b{i}')
    if len(bcols) != 6:
        cand = [c for c in df.columns if 'baro' in c.lower()]
        if len(cand) >= 6: bcols = cand[:6]
        else: raise ValueError(f"Could not find 6 barometer columns, found: {bcols or cand}")

    out = df.set_index('time_ns')[[*bcols]].copy()
    out.index.name = 'time_ns'
    out.columns = [f"b{i}" for i in range(1,7)]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors='coerce')

    return out


# ===============================
# ===  UTILS: IDs, OFFSETS   ===
# ===============================

def guess_marker_ids(atr_df: pd.DataFrame) -> Tuple[int,int]:
    """Sensor marker = most stationary (lowest 3D std); Indenter marker = most dynamic (highest)."""
    g = atr_df.groupby('marker_id')[['px','py','pz']].agg(['count','std'])
    ids = list(g.index)
    sds = {}
    for mid in ids:
        sub = atr_df[atr_df['marker_id']==mid]
        sds[mid] = float(sub[['px','py','pz']].std().mean())
    sensor_id = min(sds, key=sds.get)
    indenter_id = max(sds, key=sds.get)
    return int(sensor_id), int(indenter_id)

def estimate_baro_offset_ns(ati_df: pd.DataFrame, baro_df: pd.DataFrame,
                            fs_hz: float = 200.0, search_window_s: float = 5.0) -> int:
    """Cross-correlate Fz (ATI) with Σbaro to estimate offset (ns). Positive => shift baro forward."""
    if ati_df.empty or baro_df.empty: return 0
    t0 = max(int(ati_df.index.min()), int(baro_df.index.min()))
    t1 = min(int(ati_df.index.max()), int(baro_df.index.max()))
    if t1 <= t0: return 0
    dt_ns = int(1e9 / fs_hz)
    grid = np.arange(t0, t1, dt_ns, dtype=np.int64)
    # interpolate onto grid
    fz = np.interp(grid, ati_df.index.values.astype(np.int64), ati_df['fz'].values.astype(float))
    ps = np.interp(grid, baro_df.index.values.astype(np.int64), baro_df[[f'b{i}' for i in range(1,7)]].sum(axis=1).values.astype(float))
    # normalize
    fz = (fz - fz.mean()) / (fz.std() + 1e-9)
    ps = (ps - ps.mean()) / (ps.std() + 1e-9)
    max_lag = int(search_window_s * 1e9)
    steps = int(max_lag // dt_ns)
    best_corr, best_lag = -np.inf, 0
    for k in range(-steps, steps+1):
        if k < 0:
            corr = (ps[-k:] * fz[:len(fz)+k]).mean()
        elif k > 0:
            corr = (ps[:len(fz)-k] * fz[k:]).mean()
        else:
            corr = (ps * fz).mean()
        if corr > best_corr: best_corr, best_lag = corr, k
    return int(best_lag * dt_ns)

def interp_to_index(src_df: pd.DataFrame, target_index: pd.Index, columns: List[str]) -> pd.DataFrame:
    """Linear interpolate src_df[columns] to target_index (epoch ns)."""
    src = src_df.sort_index()
    si = src.index.values.astype(np.int64)
    ti = target_index.values.astype(np.int64)
    out = pd.DataFrame(index=target_index)
    for c in columns:
        out[c] = np.interp(ti, si, src[c].values.astype(float))
    return out

def plot_barometers(df: pd.DataFrame, out_dir: str, filename: str = "plot_barometers.png"):
    """
    6 subplots (one per barometer) vs time 't' (seconds).
    Saves out_dir/filename.
    """
    # arrange as 3 rows x 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for i in range(6):
        ax = axes[i]
        col = f"b{i+1}"
        if col in df.columns:
            mask = df['t'].notna() & df[col].notna()
            if mask.any():
                ax.plot(df.loc[mask, 't'].values, df.loc[mask, col].values, linewidth=1.0)
                ax.set_ylabel(col + " (hPa)")
        ax.grid(True, alpha=0.3)
    # set a shared title and xlabel
    fig.suptitle("Barometer pressures vs time")
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    p = Path(out_dir) / filename
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

def plot_wrench(df: pd.DataFrame, out_dir: str, filename: str = "plot_wrench.png"):
    """
    6 subplots (Fx,Fy,Fz,Tx,Ty,Tz) vs time 't' (seconds).
    Saves out_dir/filename.
    """
    names = ["fx","fy","fz","tx","ty","tz"]
    ylabels = ["Fx (N)","Fy (N)","Fz (N)","Tx","Ty","Tz"]
    # arrange as 3 rows x 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        if name in df.columns:
            mask = df['t'].notna() & df[name].notna()
            if mask.any():
                ax.plot(df.loc[mask, 't'].values, df.loc[mask, name].values, linewidth=1.0)
                ax.set_ylabel(ylabels[i])
        ax.grid(True, alpha=0.3)
    fig.suptitle("Force/Torque vs time")
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    p = Path(out_dir) / filename
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

def plot_b1_and_fz(df: pd.DataFrame, out_dir: str, filename: str = "plot_b1_vs_fz.png"):
    """
    One axes showing Barometer 1 and Fz over the same time with twin y-axes.
    Saves out_dir/filename.
    """
    fig, ax1 = plt.subplots(figsize=(12, 4))
    if 'b1' in df.columns:
        mask1 = df['t'].notna() & df['b1'].notna()
        if mask1.any():
            ax1.plot(df.loc[mask1, 't'].values, df.loc[mask1, 'b1'].values, linewidth=1.0, label='b1 (hPa)')
    else:
        ax1.plot([], [], linewidth=1.0, label='b1 (hPa)')
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("b1 (hPa)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    # Plot Fz as a continuous orange line
    if 'fz' in df.columns:
        mask2 = df['t'].notna() & df['fz'].notna()
        if mask2.any():
            ax2.plot(df.loc[mask2, 't'].values, df.loc[mask2, 'fz'].values, linewidth=2.0, color='orange', linestyle='-', label='Fz (N)')
    else:
        ax2.plot([], [], linewidth=2.0, color='orange', linestyle='-', label='Fz (N)')
    ax2.set_ylabel("Fz (N)")

    # Build a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Barometer 1 vs Fz")
    fig.tight_layout()
    p = Path(out_dir) / filename
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

def plot_path(df: pd.DataFrame, out_dir: str, filename: str = "plot_path.png"):
    """
    Plot the XY path of the contact point (x vs y in mm).
    Saves out_dir/filename.
    """
    # prefer internal names cx_mm/cy_mm, fallback to CSV compat names
    candidates = [('cx_mm', 'cy_mm'), ('-x (mm)', '-y (mm)')]
    xcol = ycol = None
    for xc, yc in candidates:
        if xc in df.columns and yc in df.columns and not (df[xc].isna().all() or df[yc].isna().all()):
            xcol, ycol = xc, yc
            break
    if xcol is None:
        print("Skipping path plot (x/y columns missing or all NaN).")
        return

    # restrict to overlapping (non-NaN) x,y
    mask = df[xcol].notna() & df[ycol].notna()
    if not mask.any():
        print("Skipping path plot (x/y columns are all NaN).")
        return

    # Crop to rectangle
    cx_center = PATH_RECT_CENTER_X_MM
    cy_center = PATH_RECT_CENTER_Y_MM
    half_w = PATH_RECT_WIDTH_MM / 2.0
    half_h = PATH_RECT_HEIGHT_MM / 2.0
    x = df.loc[mask, xcol].values
    y = df.loc[mask, ycol].values
    inside = (x >= (cx_center - half_w)) & (x <= (cx_center + half_w)) & (y >= (cy_center - half_h)) & (y <= (cy_center + half_h))
    if not inside.any():
        print("Skipping path plot (no points inside rectangle).")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x[inside], y[inside], linewidth=1.5, color='tab:blue')
    # draw rectangle border
    rect = Rectangle((cx_center-half_w, cy_center-half_h), PATH_RECT_WIDTH_MM, PATH_RECT_HEIGHT_MM,
                     edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    # mark start/end
    ax.plot(x[inside][0], y[inside][0], marker='o', color='green', label='start')
    ax.plot(x[inside][-1], y[inside][-1], marker='o', color='red', label='end')
    # lock axis limits to rectangle and keep equal scale
    ax.set_xlim(cx_center-half_w, cx_center+half_w)
    ax.set_ylim(cy_center-half_h, cy_center+half_h)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.set_xlabel(xcol + ' (mm)')
    ax.set_ylabel(ycol + ' (mm)')
    ax.set_title('Contact XY path')
    ax.grid(True, alpha=0.3)
    p = Path(out_dir) / filename
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved {p}")

# =========================
# ===  CORE PIPELINE   ====
# =========================

def main():
    # --- Build paths ---
    baro_path = os.path.join(BARO_DIR, BARO_FILENAME)
    ati_path  = os.path.join(ATI_DIR,  ATI_FILENAME)
    atr_path  = os.path.join(ATR_DIR,  ATR_FILENAME)

    print("Reading ATI ...")
    ati = read_ati_auto(ati_path)
    ati_local = pd.to_datetime(ati.index, unit='ns', utc=True).tz_convert('America/New_York')
    ati['t_local'] = ati_local
    print(f"  ATI: {len(ati)} samples, {ati_local.min()} .. {ati_local.max()} (local)")

    print("Reading Atracsys ...")
    atr = read_atracsys_csv(atr_path)
    atr_local = pd.to_datetime(atr.index, unit='ns', utc=True).tz_convert('America/New_York')
    atr['t_local'] = atr_local
    print(f"  Atracsys: {len(atr)} rows from {atr['marker_id'].nunique()} markers, " f"{atr_local.min()} .. {atr_local.max()} (local)")

    print("Reading barometers ...")
    baro = read_barometer_csv(baro_path)  # b1..b6; index=time_ns
    print(f"  Barometers: {len(baro)} samples, {pd.to_datetime(baro.index.min())} .. {pd.to_datetime(baro.index.max())}")

    # --- Optional shifts (coarse manual or auto xcorr) ---
    if ATI_SHIFT_SECONDS:
        ati.index = (ati.index.values.astype(np.int64) + int(ATI_SHIFT_SECONDS*1e9)).astype('int64')
    if ATR_SHIFT_SECONDS:
        atr.index = (atr.index.values.astype(np.int64) + int(ATR_SHIFT_SECONDS*1e9)).astype('int64')

    if AUTO_ESTIMATE_BARO_SHIFT:
        est = estimate_baro_offset_ns(ati, baro, fs_hz=200.0, search_window_s=5.0)
        print(f"  Auto-estimated barometer offset vs ATI: {est/1e6:.3f} ms")
        baro.index = (baro.index.values.astype(np.int64) + est).astype('int64')
    if BARO_SHIFT_SECONDS:
        baro.index = (baro.index.values.astype(np.int64) + int(BARO_SHIFT_SECONDS*1e9)).astype('int64')

    # --- Decide timebase ---
    base_df = {"ati": ati, "atr": atr, "baro": baro}[TIMEBASE]
    base_times = base_df.index.to_series().rename('time_ns')

    # --- Marker IDs ---
    global SENSOR_MARKER_ID, INDENTER_MARKER_ID
    if AUTO_GUESS_MARKER_IDS or SENSOR_MARKER_ID is None or INDENTER_MARKER_ID is None:
        SENSOR_MARKER_ID, INDENTER_MARKER_ID = guess_marker_ids(atr)
        print(f"  Marker IDs guessed: SENSOR={SENSOR_MARKER_ID}, INDENTER={INDENTER_MARKER_ID}")
    else:
        print(f"  Marker IDs (manual): SENSOR={SENSOR_MARKER_ID}, INDENTER={INDENTER_MARKER_ID}")

    atr_sensor = atr[atr['marker_id']==SENSOR_MARKER_ID].copy()
    atr_ind    = atr[atr['marker_id']==INDENTER_MARKER_ID].copy()

    # --- Precompute constant transforms ---
    T_Ms_to_S = make_T(euler_deg_to_R(SENSOR_MARKER_TO_SENSOR['euler_deg'], SENSOR_MARKER_TO_SENSOR.get('order','xyz')),  SENSOR_MARKER_TO_SENSOR['translation_mm'])
    T_Mi_to_I = make_T(euler_deg_to_R(INDENTER_MARKER_TO_INDENTER['euler_deg'], INDENTER_MARKER_TO_INDENTER.get('order','xyz')), INDENTER_MARKER_TO_INDENTER['translation_mm'])
    tip_I = np.array(TIP_OFFSET_I_MM, dtype=float)

    # --- Merge poses onto timebase (nearest) ---
    tol_ns = int(MERGE_TOLERANCE_MS * 1e6)
    sensor_pose = pd.merge_asof(base_times.to_frame(), atr_sensor.sort_index(), left_index=True, right_index=True, direction='nearest', tolerance=tol_ns)
    ind_pose    = pd.merge_asof(base_times.to_frame(), atr_ind.sort_index(), left_index=True, right_index=True, direction='nearest', tolerance=tol_ns)

    # Align ATI and BARO to the same timebase (interpolation for baro)
    ati_aligned  = pd.merge_asof(base_times.to_frame(), ati.sort_index(), left_index=True, right_index=True, direction='nearest', tolerance=tol_ns)
    baro_interp  = interp_to_index(baro, base_times.index, [f'b{i}' for i in range(1,7)])

    # Keep rows that have both sensor and indenter poses (baro can be NaN if no overlap)
    valid_idx = sensor_pose.dropna().index.intersection(ind_pose.dropna().index)
    if valid_idx.empty:
        raise RuntimeError("No timestamps where both sensor and indenter poses are available within tolerance. "
                           "Increase MERGE_TOLERANCE_MS or check clocks.")

    sensor_pose = sensor_pose.loc[valid_idx]
    ind_pose    = ind_pose.loc[valid_idx]
    ati_aligned = ati_aligned.loc[valid_idx]
    baro_interp = baro_interp.loc[valid_idx]

    # --- Compute contact (tip) position in the sensor frame S ---
    def row_to_RT(row):
        R = np.array([[row['R00'],row['R01'],row['R02']], [row['R10'],row['R11'],row['R12']], [row['R20'],row['R21'],row['R22']]], dtype=float)
        t = np.array([row['px'],row['py'],row['pz']], dtype=float)
        T = make_T(R, t)
        return T

    cx, cy, cz = [], [], []
    for ts in valid_idx:
        # ^A T_Ms (Atracsys camera A to sensor marker)
        T_A_Ms = row_to_RT(sensor_pose.loc[ts])
        # ^A T_Mi (Atracsys camera A to indenter marker)
        T_A_Mi = row_to_RT(ind_pose.loc[ts])
        # ^A T_S = ^A T_Ms * T_Ms_to_S  ->  ^S T_A = inverse
        T_A_S  = T_A_Ms @ T_Ms_to_S
        T_S_A  = invert_T(T_A_S)
        # Tip in Mi frame: p_tip^Mi = T_Mi_to_I * tip_I
        p_tip_Mi = apply_T(T_Mi_to_I, tip_I)
        # Tip in A: p_tip^A = ^A T_Mi * p_tip^Mi
        p_tip_A  = apply_T(T_A_Mi, p_tip_Mi)
        # Tip in S: p_tip^S = ^S T_A * p_tip^A
        p_tip_S  = apply_T(T_S_A, p_tip_A)
        cx.append(float(p_tip_S[0])); cy.append(float(p_tip_S[1])); cz.append(float(p_tip_S[2]))

    # --- Build final DataFrame on the valid timeline ---
    final = pd.DataFrame(index=valid_idx)
    t0 = int(final.index.min())
    final['t'] = (final.index.values.astype(np.int64) - t0) / 1e9

    # Barometers (interpolated onto the same times)
    for i in range(1,7):
        final[f'b{i}'] = baro_interp[f'b{i}'].values

    # Contact position (mm)
    final['cx_mm'] = cx
    final['cy_mm'] = cy
    final['cz_mm'] = cz

    # ATI wrench (nearest, scaled torque if needed)
    for c in ['fx','fy','fz','tx','ty','tz']:
        final[c] = ati_aligned[c].values
    if TORQUE_SCALE != 1.0:
        for c in ['tx','ty','tz']:
            final[c] = final[c] * TORQUE_SCALE

    # --- Column naming (compatibility with your earlier CSV) ---
    if CSV_COMPAT_NAMES:
        final['-x (mm)'] = final['cx_mm']
        final['-y (mm)'] = final['cy_mm']
        out_cols = ['t','b1','b2','b3','b4','b5','b6','-x (mm)','-y (mm)','fx','fy','fz','tx','ty','tz']
    else:
        out_cols = ['t','b1','b2','b3','b4','b5','b6','cx_mm','cy_mm','cz_mm','fx','fy','fz','tx','ty','tz']

    final = final[out_cols]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Extract timestamp from BARO_FILENAME to append to outputs (e.g. datalog_YYYY-MM-DD_HH-MM-SS.csv)
    m = re.search(r'datalog_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', BARO_FILENAME)
    if m:
        ts_suffix = f"_{m.group(1)}_{m.group(2)}"
    else:
        ts_suffix = ""
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME.replace('.csv', f"{ts_suffix}.csv"))
    # Save synchronized CSV to output path
    try:
        final.to_csv(out_path, index_label='time_ns')
        print(f"Saved synchronized CSV: {out_path}")
    except Exception as e:
        print(f"Failed to save CSV to {out_path}: {e}")
    # Make the same plots in the output folder
    plot_barometers(final, OUTPUT_DIR, f"plot_barometers{ts_suffix}.png")
    plot_wrench(final, OUTPUT_DIR, f"plot_ATI{ts_suffix}.png")

    # Only make the overlay if both columns exist and have data
    if 'b1' in final.columns and 'fz' in final.columns and final['b1'].notna().any() and final['fz'].notna().any():
        plot_b1_and_fz(final, OUTPUT_DIR, f"plot_b1_vs_fz{ts_suffix}.png")
    else:
        print("Skipping b1 vs Fz overlay (data missing or all NaN).")

    # Plot XY path of contact if available
    if ('cx_mm' in final.columns and 'cy_mm' in final.columns) or ('-x (mm)' in final.columns and '-y (mm)' in final.columns):
        plot_path(final, OUTPUT_DIR, f"plot_path{ts_suffix}.png")
    else:
        print("Skipping path plot (cx/cy columns missing).")

    print(f"Rows: {len(final)}, time span: {final['t'].iloc[0]:.3f} .. {final['t'].iloc[-1]:.3f} s")
    if final[['b1','b2','b3','b4','b5','b6']].isna().all().all():
        print("Note: barometer values are NaN (no time overlap with ATI/Atracsys or wrong barometer file).")


if __name__ == "__main__":
    pd.set_option('display.width', 140)
    main()
