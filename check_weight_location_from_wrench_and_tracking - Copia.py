#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tip path on the box top — *no flattening* — with force/torque transformed into the Top frame
using the same style of transformations from your external script (ATI→marker→camera→object/top),
while keeping your original I/O layout and plots.

What this does
--------------
1) Load Atracsys (probe + box) and auto-detect which is which via motion variance.
2) Pair by nearest timestamp.
3) Pivot-calibrate the probe tip (still available, but a fixed offset can be used).
4) Compute the tip position in CAMERA, then express it in the TOP frame that is
   rigidly tied to the box geometry (no plane flattening).
5) Transform ATI force/torque from ATI frame → probe marker frame → camera → top frame.
6) Save a CSV of tip positions in the top frame and a processing CSV that includes
   time, x/y positions (mm) and F/T in the top frame.
7) Reproduce the same plots you had: 2D tip path on top and a 3D path. The 2D plot
   shows a dashed rectangle giving ±1 mm margin around the path.

Notes
-----
- PLANE_FLATTEN is disabled by default (as requested).
- The ATI→marker rotation (R_ati_to_atimarker) is configurable. If you know the exact
  orientation from your rig, update the constant below. Identity is used by default.
"""

import os, sys, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Tuple
import argparse
from matplotlib.patches import Rectangle

# ==========================
# ======  PATHS  ===========
# ==========================
# Keep the same directory pattern you used

test_num = 109
version_num = 1

# Base folder where the raw files live for this test
# (same pattern as before, auto-created if missing)
directory_to_datasets = os.path.abspath(fr"C:\\Users\\aurir\\OneDrive - epfl.ch\\Thesis- Biorobotics Lab\\test data\\test {test_num} - sensor v{version_num}")
if not os.path.exists(directory_to_datasets):
    os.makedirs(directory_to_datasets)

# File names (as before)
ati_file = f"ati_middle_trial{test_num}.txt"
atracsys_file = f"atracsys_trial{test_num}.txt"
ATI_PATH = os.path.join(directory_to_datasets, ati_file)
ATR_PATH = os.path.join(directory_to_datasets, atracsys_file)

# Output folders (unchanged)
# 1) Out dir for plots + tip path CSV
out_dir = os.path.join(directory_to_datasets, f"lin{test_num}")
os.makedirs(out_dir, exist_ok=True)
# 2) Processing dir (x,y + F/T top-frame CSV)
processing_dir = os.path.abspath(fr"C:\\Users\\aurir\\OneDrive\\Desktop\\Thesis- Biorobotics Lab\\test data\\test {test_num} - sensor v{version_num}")
os.makedirs(processing_dir, exist_ok=True)

# ==========================
# ======  SETTINGS  ========
# ==========================
PAIR_TOL_S = 0.020
PIVOT_SECONDS = 0.02       # first seconds used for pivot LSQ (optional)
PLANE_FLATTEN = False      # <—— requested: do not flatten

# Tip offset: you can either fix it, or use pivot from the first seconds
USE_FIXED_TIP_OFFSET = True
FIXED_TIP_OFFSET_IN_PROBE_m = np.array([-0.09, 0.0, -0.025])  # meters

# Force convention: point "into" the object (matches your external code style)
FORCE_INTO_OBJECT = True

# ATI → probe-marker rotation.
# If ATI axes are already aligned to the probe marker axes, leave as identity.
# Otherwise, update with the measured calibration (e.g., Euler angles). 
R_ati_to_atimarker = np.eye(3, dtype=float)
# Example (commented):
# from scipy.spatial.transform import Rotation
# R_ati_to_atimarker = Rotation.from_euler('zx', [48, -90], degrees=True).as_matrix()

# ==========================
# ======  IO HELPERS  ======
# ==========================

def load_ati_data(path: str) -> pd.DataFrame:
    """Load ATI F/T sensor data and coerce numeric columns.
       Expected columns include one time column and 3 force + 3 torque columns.
    """
    try:
        df = pd.read_csv(path)
        # Guess columns by regex
        time_col = [c for c in df.columns if re.search('time', c, re.I)][0]
        force_cols = [c for c in df.columns if re.search(r'force', c, re.I)]
        torque_cols = [c for c in df.columns if re.search(r'torque', c, re.I)]
        df_processed = pd.DataFrame({
            'time': df[time_col].astype(float),
            'Fx': df[force_cols[0]].astype(float),
            'Fy': df[force_cols[1]].astype(float),
            'Fz': df[force_cols[2]].astype(float),
            'Tx': df[torque_cols[0]].astype(float),
            'Ty': df[torque_cols[1]].astype(float),
            'Tz': df[torque_cols[2]].astype(float),
        })
        # Normalize time units to seconds
        t = df_processed['time'].astype(float)
        if t.abs().median() > 1e12:
            df_processed['time'] = t * 1e-9
        elif t.abs().median() > 1e9:
            df_processed['time'] = t * 1e-6
        elif t.abs().median() > 1e6:
            df_processed['time'] = t * 1e-3
        return df_processed
    except Exception as e:
        print(f"Error loading ATI data: {e}")
        return None


def load_atracsys_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python")
    df.columns = [c.strip() for c in df.columns]

    def pick(regex: str):
        rx = re.compile(regex, re.I)
        cols = [c for c in df.columns if rx.fullmatch(c)]
        return cols if cols else [c for c in df.columns if re.search(regex, c, re.I)]

    tcol = (pick(r"(?:%?time|field\.timestamp)") or ["t"])[0]
    mid = [c for c in df.columns if re.search("marker_id", c, re.I)][0]
    pcols = [f"field.position{i}" for i in range(3)]
    if not all(c in df.columns for c in pcols):
        pcols = sorted([c for c in df.columns if re.search(r"position[0-2]$", c)],
                       key=lambda s: int(re.search(r"(\d)$", s).group(1)))
    rcols = [f"field.rotation{i}" for i in range(9)]
    if not all(c in df.columns for c in rcols):
        rcols = sorted([c for c in df.columns if re.search(r"rotation[0-8]$", c)],
                       key=lambda s: int(re.search(r"(\d)$", s).group(1)))

    df = df[[tcol, mid] + pcols + rcols].rename(columns={tcol: "t", mid: "marker_id"}).copy()

    # Normalize time to seconds
    t = df["t"].astype(float)
    if t.abs().median() > 1e12:
        df["t"] = t * 1e-9
    elif t.abs().median() > 1e9:
        df["t"] = t * 1e-6
    elif t.abs().median() > 1e6:
        df["t"] = t * 1e-3

    df["px"] = df[pcols[0]].astype(float) * 1e-3
    df["py"] = df[pcols[1]].astype(float) * 1e-3
    df["pz"] = df[pcols[2]].astype(float) * 1e-3
    R = df[rcols].to_numpy(float).reshape((-1, 3, 3))
    df["R"] = list(R)
    return df[["t", "marker_id", "px", "py", "pz", "R"]]


def split_probe_and_box(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ids = df["marker_id"].unique()
    assert len(ids) >= 2, "Need at least two marker_ids."
    stats = []
    for k in ids:
        p = df[df["marker_id"] == k][["px", "py", "pz"]].to_numpy()
        stats.append((k, float(np.var(p, axis=0).sum())))
    stats.sort(key=lambda x: x[1])
    box_id, probe_id = stats[0][0], stats[-1][0]
    box = df[df["marker_id"] == box_id].sort_values("t").reset_index(drop=True)
    probe = df[df["marker_id"] == probe_id].sort_values("t").reset_index(drop=True)
    return probe, box


def asof_join(probe: pd.DataFrame, box: pd.DataFrame, tol_s: float) -> pd.DataFrame:
    a = probe[["t", "px", "py", "pz", "R"]].copy(); a.columns = ["t", "ppx", "ppy", "ppz", "PR"]
    b = box[["t", "px", "py", "pz", "R"]].copy(); b.columns = ["t_box", "bpx", "bpy", "bpz", "BR"]
    out = pd.merge_asof(
        a.sort_values("t"), b.sort_values("t_box"),
        left_on="t", right_on="t_box", direction="nearest", tolerance=tol_s
    ).dropna(subset=["t_box"]).reset_index(drop=True)
    return out

# ==========================
# =======  MATH  ===========
# ==========================

def make_homogeneous_matrices(Rs: np.ndarray, ts: np.ndarray) -> np.ndarray:
    Rs = np.asarray(Rs); ts = np.asarray(ts)
    if Rs.ndim == 2: Rs = Rs[np.newaxis, ...]
    if ts.ndim == 1: ts = ts[np.newaxis, ...]
    if Rs.shape[0] != ts.shape[0]:
        raise ValueError("Rs and ts must have same leading length")
    N = Rs.shape[0]
    Ts = np.zeros((N, 4, 4), dtype=Rs.dtype)
    Ts[:, :3, :3] = Rs
    Ts[:, :3, 3] = ts
    Ts[:, 3, 3] = 1.0
    return Ts


def invert_homogeneous_matrices(Ts: np.ndarray) -> np.ndarray:
    Ts = np.asarray(Ts)
    if Ts.ndim == 2: Ts = Ts[np.newaxis, ...]
    R = Ts[:, :3, :3]
    t = Ts[:, :3, 3]
    Rinv = np.transpose(R, (0, 2, 1))
    tinv = -np.einsum("nij,nj->ni", Rinv, t)
    out = np.zeros_like(Ts)
    out[:, :3, :3] = Rinv
    out[:, :3, 3] = tinv
    out[:, 3, 3] = 1.0
    return out


def apply_homogeneous_matrices(Ts: np.ndarray, points: np.ndarray) -> np.ndarray:
    Ts = np.asarray(Ts); points = np.asarray(points)
    if Ts.ndim == 2: Ts = Ts[np.newaxis, ...]
    N = Ts.shape[0]
    if points.ndim == 1:
        p_h = np.concatenate([points, [1.0]])
        res_h = Ts @ p_h
        return res_h[:, :3]
    elif points.ndim == 2:
        if points.shape[0] != N:
            raise ValueError("When providing per-sample points, number of points must equal number of transforms")
        p_h = np.concatenate([points, np.ones((N, 1))], axis=1)
        res = np.einsum("nij,nj->ni", Ts, p_h)
        return res[:, :3]
    else:
        raise ValueError("points must be shape (3,) or (N,3)")


def set_axes_equal_3d(ax, xs, ys, zs, margin_mm=10.0):
    """Set equal scaling on a mplot3d axis using the provided data (in mm).
    Centers the axes on the data and adds an optional margin in mm.
    """
    # compute ranges
    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    z_min, z_max = float(np.nanmin(zs)), float(np.nanmax(zs))
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)
    rx = max(x_max - x_min, 1e-3) / 2.0
    ry = max(y_max - y_min, 1e-3) / 2.0
    rz = max(z_max - z_min, 1e-3) / 2.0
    R = max(rx, ry, rz) + margin_mm
    ax.set_xlim(x_mid - R, x_mid + R)
    ax.set_ylim(y_mid - R, y_mid + R)
    ax.set_zlim(z_mid - R, z_mid + R)
    # Prefer set_box_aspect for consistent rendering if available
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def pivot_calibrate(R_C_P: np.ndarray, pP_C: np.ndarray):
    """Least-squares pivot: solve R_i p + t_i = s for p (tip in probe) and s (tip in camera)."""
    N = R_C_P.shape[0]
    A = np.zeros((3*N, 6)); b = np.zeros(3*N)
    for i in range(N):
        A[3*i:3*i+3, 0:3] = R_C_P[i]
        A[3*i:3*i+3, 3:6] = -np.eye(3)
        b[3*i:3*i+3] = -pP_C[i]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    p_tip_probe = x[:3]; tip_cam = x[3:]
    return p_tip_probe, tip_cam

# ==========================
# =======  MAIN  ===========
# ==========================

def main(atracsys_path: str, test_num: int):
    # Load ATI (optional) and Atracsys
    ati_data = load_ati_data(ATI_PATH) if os.path.exists(ATI_PATH) else None
    df = load_atracsys_csv(atracsys_path)

    # Identify probe vs box; join by nearest time
    probe, box = split_probe_and_box(df)
    F = asof_join(probe, box, PAIR_TOL_S)

    # Arrays
    pP_C = F[["ppx", "ppy", "ppz"]].to_numpy()          # probe origin in camera
    R_C_P = np.stack(F["PR"].to_numpy())                   # probe rotation in camera
    pB_C = F[["bpx", "bpy", "bpz"]].to_numpy()          # box origin in camera
    R_C_B = np.stack(F["BR"].to_numpy())                   # box rotation in camera
    tt    = F["t"].to_numpy()

    # Tip offset
    M = min(len(R_C_P), max(1, int(PIVOT_SECONDS / max(1e-6, np.median(np.diff(tt))))) )
    p_tip_probe_pivot, _ = pivot_calibrate(R_C_P[:M], pP_C[:M])
    p_tip_probe = FIXED_TIP_OFFSET_IN_PROBE_m if USE_FIXED_TIP_OFFSET else p_tip_probe_pivot

    # Tip in camera
    T_C_P = make_homogeneous_matrices(R_C_P, pP_C)
    tip_C = apply_homogeneous_matrices(T_C_P, np.asarray(p_tip_probe, float))   # (N,3)

    # -------- Box→Top (no flattening) --------
    # Same rigid mapping as your previous script (columns = Top axes in Box frame)
    R_B_T = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0]], dtype=float)
    # Top origin offset in Box axes (meters). Adjust if needed.
    d_T_in_B = np.array([0.03, -0.041, 0.030], dtype=float)

    # Compose to Camera
    R_C_T = np.matmul(R_C_B, R_B_T)                               # (N,3,3)
    top_origin_C = pB_C + np.einsum('nij,j->ni', R_C_B, d_T_in_B) # (N,3)

    # Top transforms
    T_C_Top = make_homogeneous_matrices(R_C_T, top_origin_C)
    T_Top_C = invert_homogeneous_matrices(T_C_Top)

    # Tip in Top frame (unflattened)
    tip_in_top = apply_homogeneous_matrices(T_Top_C, tip_C)       # (N,3)
    X_top, Y_top, Z_top = tip_in_top[:, 0], tip_in_top[:, 1], tip_in_top[:, 2]

    # Center for prettier plots (doesn't change underlying data)
    cx = 0.5 * (np.nanmin(X_top) + np.nanmax(X_top))
    cy = 0.5 * (np.nanmin(Y_top) + np.nanmax(Y_top))
    cz = np.nanmedian(Z_top)
    Xc, Yc, Zc = X_top - cx, Y_top - cy, Z_top - cz

    # --------------- Save tip path CSV ---------------
    tip_csv = pd.DataFrame({
        't': tt,
        'tip_x_top_m': X_top, 'tip_y_top_m': Y_top, 'tip_z_top_m': Z_top,
        'tip_x_top_centered_m': Xc, 'tip_y_top_centered_m': Yc, 'tip_z_top_centered_m': Zc,
        'tip_x_cam_m': tip_C[:, 0], 'tip_y_cam_m': tip_C[:, 1], 'tip_z_cam_m': tip_C[:, 2],
        'p_tip_probe_x_m': [p_tip_probe[0]]*len(tt),
        'p_tip_probe_y_m': [p_tip_probe[1]]*len(tt),
        'p_tip_probe_z_m': [p_tip_probe[2]]*len(tt),
        'top_center_shift_m_x': [cx]*len(tt),
        'top_center_shift_m_y': [cy]*len(tt),
        'top_center_shift_m_z': [cz]*len(tt),
    })
    csv_path = os.path.join(out_dir, f"tip_path_top_frame_trial{test_num}.csv")
    tip_csv.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # --------------- Plots (same style, but no flatten) ---------------
    # 2D top plot with ±1 mm error rectangle around the path extents
    Xmm, Ymm = X_top*1e3, Y_top*1e3
    x_min, x_max = float(np.nanmin(Xmm)), float(np.nanmax(Xmm))
    y_min, y_max = float(np.nanmin(Ymm)), float(np.nanmax(Ymm))
    pad = 1.0  # mm

    plt.figure(figsize=(8, 6))
    plt.plot(Xmm, Ymm, linewidth=1.2)
    # Error rectangle (±1 mm around the path bounding box)
    rect = Rectangle((x_min - pad, y_min - pad),
                     (x_max - x_min) + 2*pad, (y_max - y_min) + 2*pad,
                     fill=False, linestyle='--', linewidth=1.0, alpha=0.7)
    ax = plt.gca(); ax.add_patch(rect)
    plt.grid(True, alpha=0.3)
    
    plt.xlim([-25, 25])
    plt.ylim([-10, 10])
    plt.xlabel("Top X [mm]"); plt.ylabel("Top Y [mm]")
    plt.title(f"Tip path on top surface (no flatten) — trial {test_num}")
    fig2d_path = os.path.join(out_dir, f"tip_path_top_xy_raw_trial{test_num}.png")
    plt.savefig(fig2d_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {fig2d_path}")

    # 3D path in Top frame
    fig3d = plt.figure()
    ax3 = fig3d.add_subplot(111, projection='3d')
    ax3.plot(Xmm, Ymm, Z_top*1e3, linewidth=1.0)
    ax3.set_xlabel("Top X [mm]"); ax3.set_ylabel("Top Y [mm]"); ax3.set_zlabel("Top Z [mm]")
    ax3.set_title(f"Tip path in Top frame (no flatten) — trial {test_num}")
    # Improve tick font sizes for readability
    try:
        ax3.xaxis.set_tick_params(labelsize=9)
        ax3.yaxis.set_tick_params(labelsize=9)
        ax3.zaxis.set_tick_params(labelsize=9)
    except Exception:
        pass
    # set equal aspect and reasonable limits around data (centered)
    set_axes_equal_3d(ax3, Xmm, Ymm, Z_top*1e3, margin_mm=12.0)
    fig3d_path = os.path.join(out_dir, f"tip_path_top_3d_raw_trial{test_num}.png")
    plt.savefig(fig3d_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {fig3d_path}")

    # 3D triads of Camera, Top, and Probe frames + path (like your OG plot)
    # Choose a representative time index (middle of the sequence)
    k = len(tt) // 2 if len(tt) > 0 else 0

    # Rotation camera->top (columns are camera axes expressed in Top)
    R_T_C = np.transpose(R_C_T, (0, 2, 1))

    # Probe rotation in Top: R_T_P = R_T_C @ R_C_P
    R_T_P = np.einsum('nij,njk->nik', R_T_C, R_C_P)

    # Origins of frames, expressed in Top coordinates (mm)
    origin_top_mm   = np.array([0.0, 0.0, 0.0])
    origin_cam_top  = apply_homogeneous_matrices(T_Top_C, np.array([0.0, 0.0, 0.0]))[k]
    origin_cam_mm   = origin_cam_top * 1e3
    origin_probe_top = apply_homogeneous_matrices(T_Top_C, pP_C)[k]
    origin_probe_mm  = origin_probe_top * 1e3
    # Box origin in Top (expressed in Top coords, mm)
    origin_box_top = apply_homogeneous_matrices(T_Top_C, pB_C)[k]
    origin_box_mm = origin_box_top * 1e3
    # Tip origin in Top (we already have tip_in_top in Top coords)
    origin_tip_top = tip_in_top[k]
    origin_tip_mm = origin_tip_top * 1e3
    # Rotation of Box in Top: R_T_B = R_T_C @ R_C_B
    R_T_B = np.einsum('nij,njk->nik', R_T_C, R_C_B)

    def draw_triad(ax, origin_mm, R_axes_in_top, scale_mm=15.0, label=''):
        o = origin_mm.astype(float)
        A = R_axes_in_top.astype(float) * scale_mm  # scale columns
        # draw thicker arrows using quiver (with small arrow heads)
        try:
            ax.quiver(o[0], o[1], o[2], A[0,0], A[1,0], A[2,0], color='r', linewidth=2.5, arrow_length_ratio=0.12)
            ax.quiver(o[0], o[1], o[2], A[0,1], A[1,1], A[2,1], color='g', linewidth=2.5, arrow_length_ratio=0.12)
            ax.quiver(o[0], o[1], o[2], A[0,2], A[1,2], A[2,2], color='b', linewidth=2.5, arrow_length_ratio=0.12)
        except Exception:
            # fallback to simple lines if quiver not available
            ax.plot([o[0], o[0]+A[0,0]],[o[1], o[1]+A[1,0]],[o[2], o[2]+A[2,0]], color='r', linewidth=2)
            ax.plot([o[0], o[0]+A[0,1]],[o[1], o[1]+A[1,1]],[o[2], o[2]+A[2,1]], color='g', linewidth=2)
            ax.plot([o[0], o[0]+A[0,2]],[o[1], o[1]+A[1,2]],[o[2], o[2]+A[2,2]], color='b', linewidth=2)

        # origin marker and label
        ax.scatter([o[0]], [o[1]], [o[2]], color='k', s=18)
        if label:
            ax.text(o[0], o[1], o[2], f" {label}", fontsize=10, fontweight='bold')
        # small axis labels near the arrow tips for clarity
        tip_offset = 0.06 * scale_mm
        ax.text(o[0]+A[0,0]+tip_offset, o[1]+A[1,0], o[2]+A[2,0], 'X', color='r', fontsize=9)
        ax.text(o[0]+A[0,1], o[1]+A[1,1]+tip_offset, o[2]+A[2,1], 'Y', color='g', fontsize=9)
        ax.text(o[0]+A[0,2], o[1]+A[1,2], o[2]+A[2,2]+tip_offset, 'Z', color='b', fontsize=9)

    fig_axes = plt.figure()
    axf = fig_axes.add_subplot(111, projection='3d')
    # Path in Top frame
    axf.plot(Xmm, Ymm, Z_top*1e3, linewidth=1.0, alpha=0.8)
    # Triads: Top (origin at top center), Camera, Probe (handle), Tip, and Box
    draw_triad(axf, origin_top_mm, np.eye(3), scale_mm=18.0, label='Top')
    draw_triad(axf, origin_cam_mm, R_T_C[k], scale_mm=18.0, label='Cam')
    # Handle / probe frame
    draw_triad(axf, origin_probe_mm, R_T_P[k], scale_mm=16.0, label='Handle')
    # Tip frame (orient using probe axes; origin at tip)
    draw_triad(axf, origin_tip_mm, R_T_P[k], scale_mm=12.0, label='Tip')
    # Box frame
    draw_triad(axf, origin_box_mm, R_T_B[k], scale_mm=20.0, label='Box')

    axf.set_xlabel("Top X [mm]"); axf.set_ylabel("Top Y [mm]"); axf.set_zlabel("Top Z [mm]")
    axf.set_title(f"Frames (Top/Cam/Probe) and tip path — trial {test_num}")
    # ensure equal scaling and center view
    set_axes_equal_3d(axf, Xmm, Ymm, Z_top*1e3, margin_mm=20.0)
    # Create a simple legend using proxy artists
    from matplotlib.lines import Line2D
    legend_items = [Line2D([0], [0], color='r', lw=3), Line2D([0], [0], color='g', lw=3), Line2D([0], [0], color='b', lw=3)]
    axf.legend(legend_items, ['X axis', 'Y axis', 'Z axis'], loc='upper right')
    # increase tick label sizes for readability
    try:
        axf.xaxis.set_tick_params(labelsize=9)
        axf.yaxis.set_tick_params(labelsize=9)
        axf.zaxis.set_tick_params(labelsize=9)
    except Exception:
        pass

    fig_axes_path = os.path.join(out_dir, f"frames_and_path_trial{test_num}.png")
    plt.savefig(fig_axes_path, dpi=220, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {fig_axes_path}")

    # --------------- Transform forces/torques into TOP frame ---------------
    if ati_data is not None and len(ati_data) > 0:
        # nearest-neighbor merge like before, but then rotate F/T into Top frame
        # Use the same mask windowing as your original (optional); here we keep all
        tip_data = pd.DataFrame({'time': tt, 'x_position': Xmm, 'y_position': Ymm})
        ati_clean = ati_data.sort_values('time')
        tip_data = tip_data.sort_values('time')
        merged = pd.merge_asof(tip_data, ati_clean, on='time', direction='nearest', tolerance=PAIR_TOL_S)

        # Precompute for speed
        R_T_C_all = np.transpose(R_C_T, (0, 2, 1))  # camera→top

        def nearest_index(t_query: float) -> int:
            idx = np.searchsorted(tt, t_query)
            if idx <= 0: return 0
            if idx >= len(tt): return len(tt) - 1
            # choose nearest between idx-1 and idx
            return idx if (tt[idx] - t_query) < (t_query - tt[idx-1]) else idx-1

        Fx_top = np.full(len(merged), np.nan)
        Fy_top = np.full(len(merged), np.nan)
        Fz_top = np.full(len(merged), np.nan)
        Tx_top = np.full(len(merged), np.nan)
        Ty_top = np.full(len(merged), np.nan)
        Tz_top = np.full(len(merged), np.nan)

        sign = -1.0 if FORCE_INTO_OBJECT else 1.0

        for i, row in merged.iterrows():
            if not np.isfinite(row.get('Fx', np.nan)):
                continue
            k = nearest_index(row['time'])
            # ATI→marker
            F_atim = R_ati_to_atimarker @ np.array([row['Fx'], row['Fy'], row['Fz']], float)
            T_atim = R_ati_to_atimarker @ np.array([row['Tx'], row['Ty'], row['Tz']], float)
            # marker (probe) → camera
            F_cam = R_C_P[k] @ F_atim
            T_cam = R_C_P[k] @ T_atim
            # camera → top
            F_top_vec = sign * (R_T_C_all[k] @ F_cam)
            T_top_vec = R_T_C_all[k] @ T_cam
            Fx_top[i], Fy_top[i], Fz_top[i] = F_top_vec
            Tx_top[i], Ty_top[i], Tz_top[i] = T_top_vec

        processing = merged.copy()
        processing['Fx_top'] = Fx_top
        processing['Fy_top'] = Fy_top
        processing['Fz_top'] = Fz_top
        processing['Tx_top'] = Tx_top
        processing['Ty_top'] = Ty_top
        processing['Tz_top'] = Tz_top

        # Keep primary columns first
        processing = processing[['time', 'x_position', 'y_position',
                                 'Fx_top', 'Fy_top', 'Fz_top', 'Tx_top', 'Ty_top', 'Tz_top',
                                 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']]

        processing_path = os.path.join(processing_dir, f"processing_test_{test_num}.csv")
        processing.to_csv(processing_path, index=False)
        print(f"Saved processing data: {processing_path}")
    else:
        print("No ATI data found — skipping F/T processing CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tip path on box top (no flatten) + F/T to Top frame")
    parser.add_argument("--atracsys", type=str, default=ATR_PATH, help="Path to atracsys_trialXX.txt/.csv")
    parser.add_argument("--test", type=int, default=test_num, help="Test number (for output names)")
    parser.add_argument("--pivot_seconds", type=float, default=PIVOT_SECONDS, help="Seconds used for pivot calibration window")
    parser.add_argument("--force_into_object", action="store_true", help="Flip force to point into object (default off unless FORCE_INTO_OBJECT=True)")
    args = parser.parse_args()

    if not os.path.exists(args.atracsys):
        sys.exit(f"Atracsys file not found: {args.atracsys}")

    # CLI toggles
    PIVOT_SECONDS = args.pivot_seconds
    if args.force_into_object:
        FORCE_INTO_OBJECT = True

    print("\nPaths in use:")
    print("  dataset dir:", directory_to_datasets)
    print("  ATI path   :", ATI_PATH)
    print("  Atracsys   :", args.atracsys)
    print("  out_dir    :", out_dir)
    print("  processing :", processing_dir)

    main(args.atracsys, args.test)
