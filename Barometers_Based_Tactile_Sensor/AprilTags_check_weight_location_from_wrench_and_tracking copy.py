#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indenter tip path on the box top — simplified.

Outputs (same as before):
- tip_path_top_frame_trial{test}.csv
- tip_path_top_xy_raw_trial{test}.png
- tip_path_top_3d_raw_trial{test}.png
- tip_path_forces_trial{test}.png
- coordinate_systems_trial{test}.png
- ati_wrenches_ati_frame_trial{test}.png
- ati_wrenches_top_frame_trial{test}.png
- processing_test_{test}.csv
"""

import os, re, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
from scipy import signal

# Make utils importable when running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from utils.sensor_io import load_ati_data, load_atracsys_data, asof_join

# =========================
# Inputs / settings (edit here)
# =========================
TEST_NUM = 51092
VERSION_NUM = 5
BOX_TAG_ID = 1

DATASET_DIR = os.path.abspath(
    fr"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test {TEST_NUM} - sensor v{VERSION_NUM}"
)

ATRACSYS_FILE = f"atracsys_trial{TEST_NUM}.txt"
ATI_FILE      = f"{TEST_NUM}ati_middle_trial.txt"  # assumes this pattern exists

PAIR_TOL_S = 0.020
Z_LIMIT_MM_2D = 10.0

# Top origin offset in Box axes (meters)
Z_SHIFT = 92 / 1000
X_SHIFT = -19 / 1000 
Y_SHIFT = 76 / 1000

# Box->Top rotation (matrix)
R_B_T = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=float).T

# Tip offset in PROBE frame (meters)
TIP_OFFSET_IN_PROBE_M = np.array([0.0, 0.01, -0.065], dtype=float) #-0.060

# ATI and TIP frame rotation: 180° around Z axis relative to PROBE frame
R_P_A = np.eye(3)
# R_180_Z = np.array([
#     [-1.0, 0.0, 0.0],
#     [0.0, -1.0, 0.0],
#     [0.0, 0.0, 1.0]
# ], dtype=float)
# R_P_A = R_180_Z
# R_P_TIP = R_180_Z  # Tip frame also rotated 180° around Z
AXIAL_DIR_IN_PROBE = np.array([0.0, 0.0, 1.0], dtype=float)
ATI_ABOVE_TIP_M = 0.025

# Smoothing (good options)
# - "butter": strong, smooth, no phase shift (filtfilt)
# - "savgol": preserves corners/shape better for trajectories
# - "none"
SMOOTH_METHOD = "savgol"     # "butter" | "savgol" | "none"
BUTTER_CUTOFF_HZ = 8.0
BUTTER_ORDER = 4
SAVGOL_WINDOW_S = 9 / 60       # ~0.1 to 0.25 s typical
SAVGOL_POLYORDER = 0

# In plane rotation correction
CORRECT_INPLANE_ROTATION = True
MANUAL_ROTATION_ANGLE_DEG = 0 # None = auto (PCA)

# =========================
# Loading
# =========================
def split_probe_and_box_fixed(df: pd.DataFrame, box_id: int = BOX_TAG_ID):
    box = df[df["marker_id"] == box_id].sort_values("t").reset_index(drop=True)
    if box.empty:
        raise ValueError(f"Box tag {box_id} not found")

    others = df[df["marker_id"] != box_id]
    if others.empty:
        raise ValueError("No probe tag found")

    # probe = tag with largest motion
    stats = []
    for k in others["marker_id"].unique():
        p = others[others["marker_id"] == k][["px","py","pz"]].to_numpy()
        stats.append((k, float(np.var(p, axis=0).sum())))
    probe_id = max(stats, key=lambda x: x[1])[0]

    probe = df[df["marker_id"] == probe_id].sort_values("t").reset_index(drop=True)
    return probe, box, probe_id

# =========================
# Math helpers
# =========================
def smooth_xyz(xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    if SMOOTH_METHOD == "none":
        return xyz

    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    if SMOOTH_METHOD == "butter":
        nyq = 0.5 * fs
        wn = BUTTER_CUTOFF_HZ / nyq
        b, a = signal.butter(BUTTER_ORDER, wn, btype="low")
        out = np.zeros_like(xyz)
        for i in range(3):
            out[:, i] = signal.filtfilt(b, a, xyz[:, i])
        return out

    if SMOOTH_METHOD == "savgol":
        win = int(round(SAVGOL_WINDOW_S * fs))
        win = max(win, SAVGOL_POLYORDER + 2)
        if win % 2 == 0:
            win += 1
        out = np.zeros_like(xyz)
        for i in range(3):
            out[:, i] = signal.savgol_filter(xyz[:, i], window_length=win, polyorder=SAVGOL_POLYORDER)
        return out

    raise ValueError(f"Unknown SMOOTH_METHOD: {SMOOTH_METHOD}")

def estimate_inplane_rotation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]; y = y[valid]
    x = x - np.mean(x); y = y - np.mean(y)
    cov = np.cov(x, y)
    w, v = np.linalg.eig(cov)
    axis = v[:, np.argmax(w)]
    return float(np.arctan2(axis[1], axis[0]))

def apply_inplane_rotation(x: np.ndarray, y: np.ndarray, angle_rad: float):
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    return xr, yr

def wrench_A_to_T(R_TA: np.ndarray, p_A_in_T: np.ndarray, F_A: np.ndarray, M_A: np.ndarray):
    F_T = np.einsum("nij,nj->ni", R_TA, F_A)
    M_rot = np.einsum("nij,nj->ni", R_TA, M_A)
    M_T = M_rot + np.cross(p_A_in_T, F_T)
    return F_T, M_T

def stabilize_box_orientation(R_C_B: np.ndarray) -> np.ndarray:
    """
    Stabilize box orientation by fixing orientation discontinuities.
    Uses the first orientation as reference and ensures all subsequent
    orientations are in the same hemisphere (dot product > 0).
    """
    N = len(R_C_B)
    R_stable = R_C_B.copy()
    
    # Use median orientation as reference (more robust than first frame)
    # Convert to quaternions for easier comparison
    def rotm_to_quat(R):
        """Convert rotation matrix to quaternion (w, x, y, z)"""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                w = (R[2,1] - R[1,2]) / s
                x = 0.25 * s
                y = (R[0,1] + R[1,0]) / s
                z = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                w = (R[0,2] - R[2,0]) / s
                x = (R[0,1] + R[1,0]) / s
                y = 0.25 * s
                z = (R[1,2] + R[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                w = (R[1,0] - R[0,1]) / s
                x = (R[0,2] + R[2,0]) / s
                y = (R[1,2] + R[2,1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])
    
    def quat_to_rotm(q):
        """Convert quaternion (w, x, y, z) to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
    
    # Convert all to quaternions
    quats = np.array([rotm_to_quat(R_C_B[i]) for i in range(N)])
    
    # Use first quaternion as reference
    q_ref = quats[0]
    
    # Fix sign flips: if dot product with reference is negative, flip the quaternion
    for i in range(N):
        if np.dot(quats[i], q_ref) < 0:
            quats[i] = -quats[i]
    
    # Average the quaternions (simple mean, since they're now all in same hemisphere)
    q_mean = np.mean(quats, axis=0)
    q_mean = q_mean / np.linalg.norm(q_mean)  # Renormalize
    
    # Use this fixed orientation for all frames
    R_fixed = quat_to_rotm(q_mean)
    
    for i in range(N):
        R_stable[i] = R_fixed
    
    print(f"Box orientation stabilized: using averaged orientation from all {N} frames")
    
    return R_stable

# =========================
# Plotting 
# =========================
def plot_top_xy(out_dir: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, test_num: int):
    plt.figure(figsize=(8, 6))
    z_mask = np.isfinite(Z) & (np.abs(Z * 1e3) <= Z_LIMIT_MM_2D)
    x_mm = X * 1e3
    y_mm = Y * 1e3
    x_plot = x_mm.copy(); y_plot = y_mm.copy()
    x_plot[~z_mask] = np.nan; y_plot[~z_mask] = np.nan
    plt.plot(x_plot, y_plot, linewidth=1.2)

    ax2d = plt.gca()
    rect_w, rect_h = 40.0, 16.0
    rect = Rectangle((-rect_w/2.0, -rect_h/2.0), rect_w, rect_h,
                     fill=False, edgecolor='k', linewidth=1.0, linestyle='--', alpha=0.9)
    ax2d.add_patch(rect)

    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim([-25, 25]); plt.ylim([-10, 10])
    plt.xlabel("Top X [mm]"); plt.ylabel("Top Y [mm]")
    plt.title(f"Tip path on top surface (unflattened) — trial {test_num}")
    fig_path = os.path.join(out_dir, f"tip_path_top_xy_raw_trial{test_num}.png")
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {fig_path}")

def plot_top_3d(out_dir: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, test_num: int, z_limit_mm: float):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x_mm, y_mm, z_mm = X * 1e3, Y * 1e3, Z * 1e3
    mask = (x_mm >= -20) & (x_mm <= 20) & (y_mm >= -8) & (y_mm <= 8) & (z_mm >= -z_limit_mm) & (z_mm <= z_limit_mm)

    if np.any(mask):
        mp = np.concatenate([[False], mask, [False]])
        d = np.diff(mp.astype(int))
        starts = np.where(d == 1)[0]
        ends   = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            ax.plot(x_mm[s:e], y_mm[s:e], z_mm[s:e], linewidth=1.0, color='C0')
        title_extra = " (filtered to ±20×±8×±2 mm)"
    else:
        ax.plot(x_mm, y_mm, z_mm, linewidth=1.0)
        title_extra = " (no points in filtered window — showing all)"

    ax.set_xlabel("Top X [mm]")
    ax.set_ylabel("Top Y [mm]")
    ax.set_zlabel("Top Z [mm]")
    ax.set_title(f"Tip path in Top frame (unflattened){title_extra} — trial {test_num}")
    ax.set_box_aspect([1, 1, 0.3])

    fig_path = os.path.join(out_dir, f"tip_path_top_3d_raw_trial{test_num}.png")
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {fig_path}")

def plot_top_path_with_forces(out_dir: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                             times: np.ndarray, ati_df: pd.DataFrame, test_num: int):
    tip_df = pd.DataFrame({
        'time': times,
        'x_mm': X * 1e3,
        'y_mm': Y * 1e3,
        'z_mm': Z * 1e3,
    }).sort_values('time').reset_index(drop=True)

    merged = pd.merge_asof(
        tip_df, ati_df.sort_values('time').reset_index(drop=True),
        on='time', direction='nearest', tolerance=PAIR_TOL_S
    ).dropna(subset=['Fx', 'Fy', 'Fz']).reset_index(drop=True)

    rng = np.random.default_rng()
    n_samples = min(3, len(merged))
    sel = merged.iloc[rng.choice(len(merged), size=n_samples, replace=False)]

    x_min, x_max = np.nanmin(tip_df['x_mm']), np.nanmax(tip_df['x_mm'])
    y_min, y_max = np.nanmin(tip_df['y_mm']), np.nanmax(tip_df['y_mm'])
    span = max(x_max - x_min, y_max - y_min, 1.0)
    forces = np.vstack([sel['Fx'].to_numpy(), sel['Fy'].to_numpy(), sel['Fz'].to_numpy()]).T
    maxF = np.linalg.norm(forces, axis=1).max()
    scale_mm_per_N = 0.2 * span / (maxF + 1e-9)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(tip_df['x_mm'], tip_df['y_mm'], tip_df['z_mm'], '-', color='tab:blue', linewidth=1.0, alpha=0.8, label='Tip path')

    added_labels = set()
    for _, row in sel.iterrows():
        ox, oy, oz = float(row['x_mm']), float(row['y_mm']), float(row['z_mm'])
        fx, fy, fz = float(row['Fx']), float(row['Fy']), float(row['Fz'])
        origin_offsets = [(-0.6, 0.0, 0.0), (0.0, 0.0, 0.0), (0.6, 0.0, 0.0)]
        comps = [('Fx', fx, (1.0, 0.0, 0.0)), ('Fy', fy, (0.0, 0.6, 0.0)), ('Fz', fz, (0.0, 0.0, 1.0))]

        for j, (name, val, col) in enumerate(comps):
            dx = (val * scale_mm_per_N) if name == 'Fx' else 0.0
            dy = (val * scale_mm_per_N) if name == 'Fy' else 0.0
            dz = (val * scale_mm_per_N) if name == 'Fz' else 0.0
            offx, offy, offz = origin_offsets[j]
            oxo, oyo, ozo = ox + offx, oy + offy, oz + offz
            label = name if name not in added_labels else None
            ax.quiver(oxo, oyo, ozo, dx, dy, dz, color=col, linewidth=1.5,
                      arrow_length_ratio=0.2, normalize=False, label=label)
            ax.text(oxo + dx, oyo + dy, ozo + dz,
                    f"{name}={val:.2f}N\nt={row['time']:.3f}s", color=col, fontsize=8)
            if label is not None:
                added_labels.add(name)

    ax.set_xlabel('Top X [mm]')
    ax.set_ylabel('Top Y [mm]')
    ax.set_zlabel('Top Z [mm]')
    ax.set_title(f'Tip path with force directions (3 random times) — trial {test_num}')
    ax.legend()
    outpath = os.path.join(out_dir, f"tip_path_forces_trial{test_num}.png")
    plt.savefig(outpath, dpi=220, bbox_inches='tight')
    plt.show()
    print(f"Saved force-arrow plot: {outpath}")

def plot_ati_wrenches(out_dir: str, ati_df: pd.DataFrame, test_num: int, frame_label: str = ""):
    ati_clean = ati_df.dropna(subset=['Fx','Fy','Fz','Tx','Ty','Tz'])
    t = ati_clean['time'].to_numpy()

    Fx, Fy, Fz = ati_clean['Fx'].to_numpy(), ati_clean['Fy'].to_numpy(), ati_clean['Fz'].to_numpy()
    Tx, Ty, Tz = ati_clean['Tx'].to_numpy(), ati_clean['Ty'].to_numpy(), ati_clean['Tz'].to_numpy()

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(t, Fx, label='Fx', color='r', linewidth=1.5)
    axs[0].plot(t, Fy, label='Fy', color='g', linewidth=1.5)
    axs[0].plot(t, Fz, label='Fz', color='b', linewidth=1.5)
    axs[0].set_ylabel('Force [N]', fontsize=12)
    title_force = f'ATI Force Time Series — trial {test_num}'
    if frame_label:
        title_force += f' ({frame_label})'
    axs[0].set_title(title_force, fontsize=13, fontweight='bold')
    axs[0].legend(loc='best', fontsize=10)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, Tx, label='Tx', color='r', linewidth=1.5)
    axs[1].plot(t, Ty, label='Ty', color='g', linewidth=1.5)
    axs[1].plot(t, Tz, label='Tz', color='b', linewidth=1.5)
    axs[1].set_xlabel('Time [s]', fontsize=12)
    axs[1].set_ylabel('Torque [Nm]', fontsize=12)
    title_torque = f'ATI Torque Time Series — trial {test_num}'
    if frame_label:
        title_torque += f' ({frame_label})'
    axs[1].set_title(title_torque, fontsize=13, fontweight='bold')
    axs[1].legend(loc='best', fontsize=10)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = f"_{frame_label.lower().replace(' ', '_')}" if frame_label else ""
    outpath = os.path.join(out_dir, f"ati_wrenches{suffix}_trial{test_num}.png")
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved ATI wrench plot ({frame_label}): {outpath}")

def plot_coordinate_systems(out_dir: str,
                            pP_C: np.ndarray, pB_C: np.ndarray, tip_C: np.ndarray,
                            R_C_P: np.ndarray, R_C_B: np.ndarray, R_C_T: np.ndarray,
                            top_origin_C: np.ndarray, ati_origin_C: np.ndarray, R_C_A: np.ndarray,
                            test_num: int):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(pP_C[:, 0]*1000, pP_C[:, 1]*1000, pP_C[:, 2]*1000, 'b-', label='Probe Path', alpha=0.5)
    ax.plot(pB_C[:, 0]*1000, pB_C[:, 1]*1000, pB_C[:, 2]*1000, 'r-', label='Box Path', alpha=0.5)
    ax.plot(tip_C[:, 0]*1000, tip_C[:, 1]*1000, tip_C[:, 2]*1000, 'g-', label='Tip Path', alpha=0.5)

    z_min_idx = np.argmin(tip_C[:, 2])
    axis_length = 0.05 * 1000

    # Tip frame (aligned with probe since rotation is commented out)
    origin_tip = tip_C[z_min_idx] * 1000
    for i, (color, label) in enumerate([('r','Tip X'),('g','Tip Y'),('b','Tip Z')]):
        d = R_C_P[z_min_idx, :, i] * axis_length
        ax.quiver(origin_tip[0], origin_tip[1], origin_tip[2], d[0], d[1], d[2],
                  color=color, alpha=1.0, label=label)

    origin_P = pP_C[z_min_idx] * 1000
    for i, (color, label) in enumerate([('r','Probe X'),('g','Probe Y'),('b','Probe Z')]):
        d = R_C_P[z_min_idx, :, i] * axis_length
        ax.quiver(origin_P[0], origin_P[1], origin_P[2], d[0], d[1], d[2],
                  color=color, alpha=0.6, label=label)

    origin_B = pB_C[z_min_idx] * 1000
    for i, (color, label) in enumerate([('r','Box X'),('g','Box Y'),('b','Box Z')]):
        d = R_C_B[z_min_idx, :, i] * axis_length
        ax.quiver(origin_B[0], origin_B[1], origin_B[2], d[0], d[1], d[2],
                  color=color, alpha=0.6, label=label)

    origin_T = top_origin_C[z_min_idx] * 1000
    for i, (color, label) in enumerate([('r','Top X'),('g','Top Y'),('b','Top Z')]):
        d = R_C_T[z_min_idx, :, i] * axis_length
        ax.quiver(origin_T[0], origin_T[1], origin_T[2], d[0], d[1], d[2],
                  color=color, alpha=0.6, label=label)

    origin_A = ati_origin_C[z_min_idx] * 1000
    for i, label in enumerate(['ATI X', 'ATI Y', 'ATI Z']):
        d = R_C_A[z_min_idx, :, i] * axis_length
        ax.quiver(origin_A[0], origin_A[1], origin_A[2], d[0], d[1], d[2],
                  color='orange', alpha=0.8, label=label, linewidth=2)

    ax.set_xlabel('Camera X [mm]')
    ax.set_ylabel('Camera Y [mm]')
    ax.set_zlabel('Camera Z [mm]')
    ax.legend()
    ax.set_title('Probe, Box, Top, and ATI Coordinate Systems in Camera Frame')

    all_points = np.vstack([pP_C, pB_C, tip_C, top_origin_C, ati_origin_C]) * 1000
    mins = all_points.min(axis=0); maxs = all_points.max(axis=0)
    max_range = np.max(maxs - mins)
    mid = 0.5 * (mins + maxs)
    ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
    ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
    ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)
    ax.set_box_aspect([1, 1, 1])

    outpath = os.path.join(out_dir, f"coordinate_systems_trial{test_num}.png")
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved coordinate systems plot: {outpath}")

# =========================
# Main
# =========================
def main(atracsys_path: str, test_num: int):
    base_dir = os.path.dirname(os.path.abspath(atracsys_path))
    out_dir = os.path.join(base_dir, f"lin{test_num}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    ati_path = os.path.join(base_dir, ATI_FILE)
    atr_path = atracsys_path

    print(f"Trial {test_num}")
    print(f"Atracsys: {atr_path}")
    print(f"ATI:      {ati_path}")
    print(f"Smoothing: {SMOOTH_METHOD}")

    ati_data = load_ati_data(ati_path)
    df = load_atracsys_data(atr_path)
    probe, box, probe_id = split_probe_and_box_fixed(df, box_id=BOX_TAG_ID)
    print(f"Box tag fixed to {BOX_TAG_ID}, probe tag = {probe_id}")
    F = asof_join(probe, box, PAIR_TOL_S)

    tt = F["t"].to_numpy()
    pP_C = F[["ppx","ppy","ppz"]].to_numpy()
    pB_C = F[["bpx","bpy","bpz"]].to_numpy()
    R_C_P = np.stack(F["PR"].to_numpy())
    R_C_B = np.stack(F["BR"].to_numpy())
    
    # Stabilize box orientation to fix tracking discontinuities
    R_C_B = stabilize_box_orientation(R_C_B)

    # Tip in camera: tip_C = pP_C + R_C_P @ tip_offset_in_probe
    tip_C = pP_C + np.einsum("nij,j->ni", R_C_P, TIP_OFFSET_IN_PROBE_M)

    d_T_in_B = np.array([X_SHIFT, Y_SHIFT, Z_SHIFT], dtype=float)
    R_C_T = np.matmul(R_C_B, R_B_T)
    top_origin_C = pB_C + np.einsum("nij,j->ni", R_C_B, d_T_in_B)

    # Tip in Top: tip_T = R_C_T^T (tip_C - top_origin_C)
    tip_T_raw = np.einsum("nij,nj->ni", np.transpose(R_C_T, (0,2,1)), (tip_C - top_origin_C))
    tip_T = smooth_xyz(tip_T_raw, tt)

    X_top, Y_top, Z_top = tip_T[:,0], tip_T[:,1], tip_T[:,2]

    rotation_angle_rad = 0.0
    if CORRECT_INPLANE_ROTATION:
        if MANUAL_ROTATION_ANGLE_DEG is None:
            rotation_angle_rad = estimate_inplane_rotation(X_top, Y_top)
            print(f"Detected in-plane rotation: {np.rad2deg(rotation_angle_rad):.2f}°")
        else:
            rotation_angle_rad = np.deg2rad(MANUAL_ROTATION_ANGLE_DEG)
            print(f"Applying manual in-plane rotation: {MANUAL_ROTATION_ANGLE_DEG:.2f}°")
        X_top, Y_top = apply_inplane_rotation(X_top, Y_top, rotation_angle_rad)

    # Centered
    cx = 0.5 * (np.nanmin(X_top) + np.nanmax(X_top))
    cy = 0.5 * (np.nanmin(Y_top) + np.nanmax(Y_top))
    cz = np.nanmedian(Z_top)
    Xc, Yc, Zc = X_top - cx, Y_top - cy, Z_top - cz

    # ATI origin in probe, then camera
    p_A_in_P = TIP_OFFSET_IN_PROBE_M + ATI_ABOVE_TIP_M * AXIAL_DIR_IN_PROBE
    pA_C = pP_C + np.einsum("nij,j->ni", R_C_P, p_A_in_P)
    R_C_A = R_C_P @ R_P_A

    # ATI in Top
    R_T_C = np.transpose(R_C_T, (0,2,1))
    p_A_in_T = np.einsum("nij,nj->ni", R_T_C, (pA_C - top_origin_C))
    R_T_A = np.einsum("nij,njk->nik", R_T_C, R_C_A)

    # Sync ATI to tt and transform wrenches
    times_df = pd.DataFrame({"t": tt})
    W = pd.merge_asof(
        times_df.sort_values("t"),
        ati_data.sort_values("time"),
        left_on="t", right_on="time",
        direction="nearest", tolerance=PAIR_TOL_S
    ).dropna()

    Fx_top = np.full(len(tt), np.nan); Fy_top = Fx_top.copy(); Fz_top = Fx_top.copy()
    Tx_top = Fx_top.copy(); Ty_top = Fx_top.copy(); Tz_top = Fx_top.copy()

    if len(W) > 0:
        idx = W.index.to_numpy()
        F_A = W[["Fx","Fy","Fz"]].to_numpy(float)
        M_A = W[["Tx","Ty","Tz"]].to_numpy(float)
        F_T, M_T = wrench_A_to_T(R_T_A[idx], p_A_in_T[idx], F_A, M_A)

        Fx_top[idx], Fy_top[idx], Fz_top[idx] = F_T[:,0], F_T[:,1], F_T[:,2]
        Tx_top[idx], Ty_top[idx], Tz_top[idx] = M_T[:,0], M_T[:,1], M_T[:,2]
    
    # Apply same in-plane rotation to forces/torques if position was rotated
    if CORRECT_INPLANE_ROTATION and rotation_angle_rad != 0.0:
        valid = np.isfinite(Fx_top) & np.isfinite(Fy_top)
        if np.any(valid):
            Fx_rot, Fy_rot = apply_inplane_rotation(Fx_top, Fy_top, rotation_angle_rad)
            Tx_rot, Ty_rot = apply_inplane_rotation(Tx_top, Ty_top, rotation_angle_rad)
            Fx_top, Fy_top = Fx_rot, Fy_rot
            Tx_top, Ty_top = Tx_rot, Ty_rot
            print("Applied in-plane rotation to force/torque components.")

    ati_top_df = pd.DataFrame({
        "time": tt,
        "Fx": Fx_top, "Fy": Fy_top, "Fz": Fz_top,
        "Tx": Tx_top, "Ty": Ty_top, "Tz": Tz_top
    }).dropna(subset=["Fx","Fy","Fz"])

    # Output CSV (same columns)
    out = pd.DataFrame({
        "t": tt,
        "tip_x_top_m": X_top, "tip_y_top_m": Y_top, "tip_z_top_m": Z_top,
        "tip_x_top_centered_m": Xc, "tip_y_top_centered_m": Yc, "tip_z_top_centered_m": Zc,
        "tip_x_top_raw_m": tip_T_raw[:,0], "tip_y_top_raw_m": tip_T_raw[:,1], "tip_z_top_raw_m": tip_T_raw[:,2],
        "tip_x_cam_m": tip_C[:,0], "tip_y_cam_m": tip_C[:,1], "tip_z_cam_m": tip_C[:,2],
        "p_tip_probe_x_m": [TIP_OFFSET_IN_PROBE_M[0]]*len(tt),
        "p_tip_probe_y_m": [TIP_OFFSET_IN_PROBE_M[1]]*len(tt),
        "p_tip_probe_z_m": [TIP_OFFSET_IN_PROBE_M[2]]*len(tt),
        "top_center_shift_m_x": [cx]*len(tt),
        "top_center_shift_m_y": [cy]*len(tt),
        "top_center_shift_m_z": [cz]*len(tt),
        "inplane_rotation_correction_rad": [rotation_angle_rad]*len(tt),
        "inplane_rotation_correction_deg": [np.rad2deg(rotation_angle_rad)]*len(tt),
        "Fx_top_N": Fx_top, "Fy_top_N": Fy_top, "Fz_top_N": Fz_top,
        "Tx_top_Nm": Tx_top, "Ty_top_Nm": Ty_top, "Tz_top_Nm": Tz_top,
    })

    # Torque about the tip point (Top axes)
    r_T_to_tip = tip_T_raw
    F_stack = np.vstack([Fx_top, Fy_top, Fz_top]).T
    r_cross_F = np.cross(r_T_to_tip, F_stack)
    out["Tx_tip_top_Nm"] = out["Tx_top_Nm"] - r_cross_F[:,0]
    out["Ty_tip_top_Nm"] = out["Ty_top_Nm"] - r_cross_F[:,1]
    out["Tz_tip_top_Nm"] = out["Tz_top_Nm"] - r_cross_F[:,2]

    csv_path = os.path.join(out_dir, f"tip_path_top_frame_trial{test_num}.csv")
    out.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    print("Tip position statistics (Top frame, unflattened):")
    print(f"X range: [{np.min(X_top):.3f}, {np.max(X_top):.3f}] m")
    print(f"Y range: [{np.min(Y_top):.3f}, {np.max(Y_top):.3f}] m")
    print(f"Z mean ± std: {np.mean(Z_top):.3f} ± {np.std(Z_top):.3f} m")

    # Plots
    plot_ati_wrenches(out_dir, ati_data, test_num, frame_label="ATI Frame")
    plot_ati_wrenches(out_dir, ati_top_df, test_num, frame_label="Top Frame")
    plot_top_xy(out_dir, X_top, Y_top, Z_top, test_num)
    plot_top_3d(out_dir, X_top, Y_top, Z_top, test_num, z_limit_mm=Z_LIMIT_MM_2D)

    if not ati_top_df.empty:
        plot_top_path_with_forces(out_dir, X_top, Y_top, Z_top, tt, ati_top_df, test_num)

    plot_coordinate_systems(
        out_dir,
        pP_C, pB_C, tip_C,
        R_C_P, R_C_B, R_C_T,
        top_origin_C, pA_C, R_C_A,
        test_num
    )

    # Processing file (same idea)
    tip_data = pd.DataFrame({
        "time": tt,
        "x_position_mm": X_top * 1000,
        "y_position_mm": Y_top * 1000,
        "z_position_mm": Z_top * 1000,
    }).sort_values("time")

    processing_data = pd.merge_asof(
        tip_data, ati_top_df.sort_values("time"),
        on="time", direction="nearest", tolerance=PAIR_TOL_S
    ).dropna(subset=["Fx","Fy","Fz"])

    processing_path = os.path.join(DATASET_DIR, f"processing_test_{test_num}.csv")
    processing_data.to_csv(processing_path, index=False)
    print(f"Saved processing data: {processing_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--atracsys", type=str, default=None)
    parser.add_argument("--test", type=int, default=TEST_NUM)
    args = parser.parse_args()

    atr_path = args.atracsys or os.path.join(DATASET_DIR, ATRACSYS_FILE)
    main(atr_path, args.test)
