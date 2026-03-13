#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indenter tip path on the box top surface.

Frame glossary
--------------
  C  : Camera frame      — Atracsys world frame (fixed)
  B  : Box frame         — tag1, rigidly fixed on the box
  T  : Top frame         — box surface; origin = B origin + d_T_in_B, axes parallel to B
  P  : Probe frame       — tag2, moving with the tool
  A  : ATI frame         — force/torque sensor, fixed offset+rotation relative to P

Pipeline
--------
  1. Load Atracsys (tag poses) and ATI (wrench) data.
  2. Extract R_C_P, R_C_B (orientations in C) and pP_C, pB_C (positions in C).
  3. Build Top frame T from Box frame B.
  4. Compute tip position in Top frame: tip_T = R_C_T^T (tip_C - top_origin_C).
  5. Build ATI frame A from Probe frame P using R_P_A.
  6. Transform ATI wrench → Top frame → transfer moment to tip contact point.
  7. Save CSV and plots.

Outputs
-------
  tip_path_top_frame_trial{N}.csv
  tip_path_top_xy_raw_trial{N}.png
  tip_path_top_3d_raw_trial{N}.png
  tip_path_forces_trial{N}.png
  coordinate_systems_trial{N}.png
  ati_wrenches_ati_frame_trial{N}.png
  ati_wrenches_top_frame_trial{N}.png
  processing_test_{N}.csv
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from utils.sensor_io import load_ati_data, asof_join


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — TRIAL / PATH SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

TEST_NUM    = 52092
VERSION_NUM = 5
BOX_TAG_ID  = 1          # marker_id of the fixed box AprilTag (tag1)

DATASET_DIR = os.path.abspath(
    fr"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data"
    fr"\test {TEST_NUM} - sensor v{VERSION_NUM}"
)
TAG1_FILE     = f"{TEST_NUM}tag1_pose_trial.txt"   # box tag (fixed)
TAG2_FILE     = f"{TEST_NUM}tag2_pose_trial.txt"   # probe tag (moving)
ATI_FILE      = f"{TEST_NUM}ati_middle_trial.txt"

PAIR_TOL_S    = 0.020    # max time gap [s] for sensor synchronisation
Z_LIMIT_MM_2D = 10.0     # Z-depth filter for 2-D surface plots [mm]


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — COORDINATE FRAME DEFINITIONS
#
#   All geometric relationships between frames are defined here.
#   Edit these when the physical setup changes.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Box (B) → Top (T) ─────────────────────────────────────────────────────────
# The Top frame origin is offset from the Box tag origin by d_T_in_B,
# expressed in Box frame axes.  Orientation is parallel (R_B_T = I).

X_SHIFT = -5 / 1000    # m — Top origin along Box X
Y_SHIFT = 70 / 1000    # m — Top origin along Box Y
Z_SHIFT = 120 / 1000    # m — Top origin along Box Z  (height of box surface)

CORRECT_INPLANE_ROTATION  = True
MANUAL_ROTATION_ANGLE_DEG = 3         # degrees; None → auto (PCA)

R_B_T = np.eye(3, dtype=float)   # Top axes aligned with Box axes

# ── Probe (P) → Tip ───────────────────────────────────────────────────────────
# Fixed vector from probe AprilTag origin to the physical indenter tip,
# expressed in Probe frame coordinates [m].

TIP_OFFSET_IN_PROBE_M = np.array([0.0, 0.01, -0.055], dtype=float)

# ── Probe (P) → ATI (A) ───────────────────────────────────────────────────────
# The ATI sensor is rigidly mounted on the probe shaft.
# Physical layout (probe frame: X=left, Y=down, Z=out-of-page):
#   • ATI Y axis points upper-left  (138° from drawing-right = 42° above probe X)
#   • ATI X axis points lower-left  (228° from drawing-right = 42° below probe X)
#   • ATI Z axis points INTO page   (inverted relative to probe Z)
#
# Decomposition: R_P_A = Rz(48°) @ Rx(180°)
#   Rz(48°)  : 48° CCW in-plane rotation
#   Rx(180°) : flips ATI Y and Z in the rotated frame (puts Z into page)
#
# R_P_A : maps a vector expressed in ATI frame into Probe frame coordinates.
#         Column i = i-th ATI axis expressed in Probe coordinates.

_angle_PA = np.deg2rad(48.0)
_Rz48 = np.array([
    [ np.cos(_angle_PA), -np.sin(_angle_PA), 0.0],
    [ np.sin(_angle_PA),  np.cos(_angle_PA), 0.0],
    [ 0.0,                0.0,               1.0],
], dtype=float)
_Rx180 = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=float)
R_P_A = _Rz48 @ _Rx180   # ATI Y→upper-left (138°), ATI X→lower-left (228°), ATI Z→into page

# ── ATI sensing point ─────────────────────────────────────────────────────────
# The ATI force-sensing origin is located above the tip along the probe shaft.
# Used in the moment-transfer step: M_tip = M_ATI + r_{tip→ATI} × F.

AXIAL_DIR_IN_PROBE = np.array([0.0, 0.0, 1.0], dtype=float)
ATI_ABOVE_TIP_M    = 0.015    # m — distance from tip to ATI sensing origin


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — SIGNAL PROCESSING SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

SMOOTH_METHOD             = "savgol"   # "butter" | "savgol" | "none"
BUTTER_CUTOFF_HZ          = 8.0
BUTTER_ORDER              = 4
SAVGOL_WINDOW_S           = 9 / 60    # window duration [s]
SAVGOL_POLYORDER          = 0

# ═══════════════════════════════════════════════════════════════════════════════
# 4 — DATA LOADING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def split_probe_and_box(df: pd.DataFrame, box_id: int = BOX_TAG_ID):
    """Return (probe_df, box_df, probe_marker_id).

    Box tag is identified by marker_id == box_id.
    Probe tag is whichever remaining tag has the highest positional variance
    (i.e. moves the most).
    """
    box = df[df["marker_id"] == box_id].sort_values("t").reset_index(drop=True)
    if box.empty:
        raise ValueError(f"Box tag {box_id} not found in data")

    others = df[df["marker_id"] != box_id]
    if others.empty:
        raise ValueError("No probe tag found in data")

    stats = [
        (k, float(np.var(others[others["marker_id"] == k][["px","py","pz"]].to_numpy(), axis=0).sum()))
        for k in others["marker_id"].unique()
    ]
    probe_id = max(stats, key=lambda x: x[1])[0]
    probe = df[df["marker_id"] == probe_id].sort_values("t").reset_index(drop=True)
    return probe, box, probe_id


def stabilize_box_orientation(R_C_B: np.ndarray) -> np.ndarray:
    """Replace per-frame box orientations with a single averaged orientation.

    Fixes AprilTag tracking discontinuities on a nominally static object by
    averaging all quaternions and broadcasting the result to every frame.
    """
    def rotm_to_quat(R):
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
        if R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])

    def quat_to_rotm(q):
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])

    N = len(R_C_B)
    quats = np.array([rotm_to_quat(R_C_B[i]) for i in range(N)])
    q_ref = quats[0]
    for i in range(N):                          # ensure same hemisphere
        if np.dot(quats[i], q_ref) < 0:
            quats[i] = -quats[i]
    q_mean = quats.mean(axis=0)
    q_mean /= np.linalg.norm(q_mean)
    R_fixed = quat_to_rotm(q_mean)
    print(f"Box orientation stabilised using mean of {N} frames.")
    return np.broadcast_to(R_fixed, R_C_B.shape).copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 4b — TAG FILE LOADER  (replaces Atracsys intermediate file)
# ═══════════════════════════════════════════════════════════════════════════════

def _time_to_seconds(t_raw: np.ndarray) -> np.ndarray:
    """Heuristic: ns/us/ms → seconds based on magnitude."""
    t = np.asarray(t_raw, dtype=np.float64)
    med = np.nanmedian(np.abs(t))
    if med > 1e12:   return t * 1e-9   # nanoseconds
    elif med > 1e9:  return t * 1e-6   # microseconds
    elif med > 1e6:  return t * 1e-3   # milliseconds
    return t


def _quat_xyzw_to_rotm(qx, qy, qz, qw) -> np.ndarray:
    """Vectorized quaternion (x,y,z,w) → rotation matrix (N,3,3).
    Convention: R_C_tag — maps tag-frame vectors into camera frame."""
    x = np.asarray(qx, float); y = np.asarray(qy, float)
    z = np.asarray(qz, float); w = np.asarray(qw, float)
    n = np.sqrt(x*x + y*y + z*z + w*w); n[n == 0] = 1.0
    x /= n; y /= n; z /= n; w /= n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    xw, yw, zw = x*w, y*w, z*w
    R = np.empty((len(x), 3, 3), dtype=np.float64)
    R[:,0,0] = 1-2*(yy+zz);  R[:,0,1] = 2*(xy-zw);   R[:,0,2] = 2*(xz+yw)
    R[:,1,0] = 2*(xy+zw);    R[:,1,1] = 1-2*(xx+zz);  R[:,1,2] = 2*(yz-xw)
    R[:,2,0] = 2*(xz-yw);    R[:,2,1] = 2*(yz+xw);    R[:,2,2] = 1-2*(xx+yy)
    return R


def load_tag_file(path: str, tag_id: int) -> pd.DataFrame:
    """Load a ROS AprilTag pose CSV directly.

    Expected columns: %time, field.pose.position.{x,y,z},
                      field.pose.orientation.{x,y,z,w}
    Positions are in metres (ROS convention).

    Returns DataFrame compatible with asof_join():
      t, marker_id, px, py, pz, R
    """
    df = pd.read_csv(path)
    t_s = _time_to_seconds(df["%time"].astype(np.int64).to_numpy())
    px  = df["field.pose.position.x"].astype(float).to_numpy()
    py  = df["field.pose.position.y"].astype(float).to_numpy()
    pz  = df["field.pose.position.z"].astype(float).to_numpy()
    qx  = df["field.pose.orientation.x"].astype(float).to_numpy()
    qy  = df["field.pose.orientation.y"].astype(float).to_numpy()
    qz  = df["field.pose.orientation.z"].astype(float).to_numpy()
    qw  = df["field.pose.orientation.w"].astype(float).to_numpy()
    R   = _quat_xyzw_to_rotm(qx, qy, qz, qw)
    out = pd.DataFrame({"t": t_s, "marker_id": int(tag_id),
                        "px": px, "py": py, "pz": pz})
    out["R"] = list(R)
    return out.sort_values("t").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — SIGNAL PROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def smooth_xyz(xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    if SMOOTH_METHOD == "none":
        return xyz
    fs = 1.0 / np.median(np.diff(t))
    if SMOOTH_METHOD == "butter":
        b, a = signal.butter(BUTTER_ORDER, BUTTER_CUTOFF_HZ / (0.5 * fs), btype="low")
        return np.column_stack([signal.filtfilt(b, a, xyz[:, i]) for i in range(3)])
    if SMOOTH_METHOD == "savgol":
        win = max(int(round(SAVGOL_WINDOW_S * fs)), SAVGOL_POLYORDER + 2)
        if win % 2 == 0:
            win += 1
        return np.column_stack([
            signal.savgol_filter(xyz[:, i], window_length=win, polyorder=SAVGOL_POLYORDER)
            for i in range(3)
        ])
    raise ValueError(f"Unknown SMOOTH_METHOD: {SMOOTH_METHOD!r}")


def estimate_inplane_rotation(x: np.ndarray, y: np.ndarray) -> float:
    """PCA-based in-plane rotation angle of a 2-D trajectory [radians]."""
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid] - x[valid].mean()
    y = y[valid] - y[valid].mean()
    w, v = np.linalg.eig(np.cov(x, y))
    axis = v[:, np.argmax(w)]
    return float(np.arctan2(axis[1], axis[0]))


def apply_inplane_rotation(x: np.ndarray, y: np.ndarray, angle_rad: float):
    ca, sa = np.cos(-angle_rad), np.sin(-angle_rad)
    return ca*x - sa*y, sa*x + ca*y


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — WRENCH TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

def transport_wrench(R_new_old: np.ndarray,
                     p_old_in_new: np.ndarray,
                     F_old: np.ndarray,
                     M_old: np.ndarray):
    """Rigid-body wrench transport.

    Rotates a wrench from frame OLD into frame NEW and transfers the moment
    reference point from OLD origin to NEW origin.

    Parameters
    ----------
    R_new_old   : (N,3,3)  rotation mapping OLD vectors → NEW frame
    p_old_in_new: (N,3)    position of OLD origin expressed in NEW frame
    F_old, M_old: (N,3)    force and moment in OLD frame

    Returns
    -------
    F_new : (N,3)  force in NEW frame          F_new = R @ F_old
    M_new : (N,3)  moment at NEW origin         M_new = R @ M_old + p_old_in_new × F_new
    """
    F_new = np.einsum("nij,nj->ni", R_new_old, F_old)
    M_new = np.einsum("nij,nj->ni", R_new_old, M_old) + np.cross(p_old_in_new, F_new)
    return F_new, M_new


# ═══════════════════════════════════════════════════════════════════════════════
# 7 — PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_top_xy(out_dir, X, Y, Z, test_num):
    plt.figure(figsize=(8, 6))
    z_mask = np.isfinite(Z) & (np.abs(Z * 1e3) <= Z_LIMIT_MM_2D)
    xp, yp = X * 1e3, Y * 1e3
    xp[~z_mask] = np.nan; yp[~z_mask] = np.nan
    plt.plot(xp, yp, linewidth=1.2)
    ax = plt.gca()
    ax.add_patch(Rectangle((-20, -8), 40, 16,
                            fill=False, edgecolor='k', linewidth=1.0, linestyle='--', alpha=0.9))
    plt.grid(True, alpha=0.3); plt.axis("equal")
    plt.xlim([-25, 25]); plt.ylim([-10, 10])
    plt.xlabel("Top X [mm]"); plt.ylabel("Top Y [mm]")
    plt.title(f"Tip path on top surface — trial {test_num}")
    path = os.path.join(out_dir, f"tip_path_top_xy_raw_trial{test_num}.png")
    plt.savefig(path, dpi=220, bbox_inches="tight"); plt.show()
    print(f"Saved: {path}")


def plot_top_3d(out_dir, X, Y, Z, test_num, z_limit_mm):
    fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
    x_mm, y_mm, z_mm = X*1e3, Y*1e3, Z*1e3
    mask = (x_mm >= -20) & (x_mm <= 20) & (y_mm >= -8) & (y_mm <= 8) & (np.abs(z_mm) <= z_limit_mm)
    if np.any(mask):
        mp = np.concatenate([[False], mask, [False]])
        d  = np.diff(mp.astype(int))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            ax.plot(x_mm[s:e], y_mm[s:e], z_mm[s:e], linewidth=1.0, color='C0')
        extra = " (filtered ±20×±8×±z_limit mm)"
    else:
        ax.plot(x_mm, y_mm, z_mm, linewidth=1.0)
        extra = " (no filtered points — showing all)"
    ax.set_xlabel("Top X [mm]"); ax.set_ylabel("Top Y [mm]"); ax.set_zlabel("Top Z [mm]")
    ax.set_title(f"Tip path in Top frame{extra} — trial {test_num}")
    ax.set_box_aspect([1, 1, 0.3])
    path = os.path.join(out_dir, f"tip_path_top_3d_raw_trial{test_num}.png")
    plt.savefig(path, dpi=220, bbox_inches="tight"); plt.show()
    print(f"Saved: {path}")


def plot_top_path_with_forces(out_dir, X, Y, Z, times, ati_df, test_num):
    tip_df = pd.DataFrame({"time": times, "x_mm": X*1e3, "y_mm": Y*1e3, "z_mm": Z*1e3}
                          ).sort_values("time").reset_index(drop=True)
    merged = pd.merge_asof(
        tip_df, ati_df.sort_values("time").reset_index(drop=True),
        on="time", direction="nearest", tolerance=PAIR_TOL_S,
    ).dropna(subset=["Fx","Fy","Fz"]).reset_index(drop=True)

    n_sel  = min(3, len(merged))
    sel    = merged.iloc[np.random.default_rng().choice(len(merged), size=n_sel, replace=False)]
    span   = max(tip_df["x_mm"].max()-tip_df["x_mm"].min(),
                 tip_df["y_mm"].max()-tip_df["y_mm"].min(), 1.0)
    maxF   = np.linalg.norm(sel[["Fx","Fy","Fz"]].to_numpy(), axis=1).max()
    scale  = 0.2 * span / (maxF + 1e-9)

    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(tip_df["x_mm"], tip_df["y_mm"], tip_df["z_mm"],
            "-", color="tab:blue", linewidth=1.0, alpha=0.8, label="Tip path")

    added = set()
    for _, row in sel.iterrows():
        ox, oy, oz = float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])
        offsets = [(-0.6,0,0), (0,0,0), (0.6,0,0)]
        comps   = [("Fx", float(row["Fx"]), (1,0,0)),
                   ("Fy", float(row["Fy"]), (0,.6,0)),
                   ("Fz", float(row["Fz"]), (0,0,1))]
        for j, (name, val, col) in enumerate(comps):
            dx = val*scale if name == "Fx" else 0.0
            dy = val*scale if name == "Fy" else 0.0
            dz = val*scale if name == "Fz" else 0.0
            offx, offy, offz = offsets[j]
            lbl = name if name not in added else None
            ax.quiver(ox+offx, oy+offy, oz+offz, dx, dy, dz,
                      color=col, linewidth=1.5, arrow_length_ratio=0.2, normalize=False, label=lbl)
            ax.text(ox+offx+dx, oy+offy+dy, oz+offz+dz,
                    f"{name}={val:.2f}N\nt={row['time']:.3f}s", color=col, fontsize=8)
            if lbl:
                added.add(name)

    ax.set_xlabel("Top X [mm]"); ax.set_ylabel("Top Y [mm]"); ax.set_zlabel("Top Z [mm]")
    ax.set_title(f"Tip path with ATI force directions — trial {test_num}"); ax.legend()
    path = os.path.join(out_dir, f"tip_path_forces_trial{test_num}.png")
    plt.savefig(path, dpi=220, bbox_inches="tight"); plt.show()
    print(f"Saved: {path}")


def plot_ati_wrenches(out_dir, ati_df, test_num, frame_label=""):
    df = ati_df.dropna(subset=["Fx","Fy","Fz","Tx","Ty","Tz"])
    t  = df["time"].to_numpy()
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    pairs = [
        (["Fx","Fy","Fz"], "Force [N]",  "ATI Force Time Series"),
        (["Tx","Ty","Tz"], "Torque [Nm]","ATI Torque Time Series"),
    ]
    for (comps, ylabel, base_title), ax in zip(pairs, axs):
        for c, col in zip(comps, ["r","g","b"]):
            ax.plot(t, df[c].to_numpy(), label=c, color=col, linewidth=1.5)
        title = f"{base_title} — trial {test_num}" + (f" ({frame_label})" if frame_label else "")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc="best", fontsize=10); ax.grid(True, alpha=0.3)
    axs[1].set_xlabel("Time [s]", fontsize=12)
    plt.tight_layout()
    suffix = f"_{frame_label.lower().replace(' ','_')}" if frame_label else ""
    path = os.path.join(out_dir, f"ati_wrenches{suffix}_trial{test_num}.png")
    plt.savefig(path, dpi=220, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_coordinate_systems(out_dir, pP_C, pB_C, tip_C,
                             R_C_P, R_C_B, R_C_T,
                             top_origin_C, ati_origin_C, R_C_A,
                             test_num):
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot(*[pP_C[:,i]*1000 for i in range(3)], "b-", label="Probe Path",  alpha=0.5)
    ax.plot(*[pB_C[:,i]*1000 for i in range(3)], "r-", label="Box Path",    alpha=0.5)
    ax.plot(*[tip_C[:,i]*1000 for i in range(3)],"g-", label="Tip Path",    alpha=0.5)

    k = np.argmin(tip_C[:, 2])   # frame closest to surface contact
    L = 0.05 * 1000               # arrow length [mm]

    def draw_frame(origin_mm, R_cam, colors, labels, alpha=0.8, lw=2):
        """Draw X/Y/Z axes of a frame given its rotation matrix in Camera frame."""
        for i, (c, lbl) in enumerate(zip(colors, labels)):
            ax.quiver(*origin_mm, *(R_cam[:, i] * L),
                      color=c, alpha=alpha, label=lbl, linewidth=lw)

    draw_frame(tip_C[k]*1000,        R_C_P[k] @ R_P_A,  ["r","g","b"],
               ["Tip X","Tip Y","Tip Z"],               alpha=1.0)
    draw_frame(pP_C[k]*1000,          R_C_P[k],           ["r","g","b"],
               ["Probe X","Probe Y","Probe Z"],          alpha=0.6)
    draw_frame(pB_C[k]*1000,          R_C_B[k],           ["r","g","b"],
               ["Box X","Box Y","Box Z"],                alpha=0.6)
    draw_frame(top_origin_C[k]*1000,  R_C_T[k],           ["r","g","b"],
               ["Top X","Top Y","Top Z"],                alpha=0.6)
    draw_frame(ati_origin_C[k]*1000,  R_C_A[k],           ["orange"]*3,
               ["ATI X","ATI Y","ATI Z"],                alpha=0.8)

    ax.set_xlabel("Camera X [mm]"); ax.set_ylabel("Camera Y [mm]"); ax.set_zlabel("Camera Z [mm]")
    ax.set_title("Coordinate systems in Camera frame"); ax.legend()
    all_pts = np.vstack([pP_C, pB_C, tip_C, top_origin_C, ati_origin_C]) * 1000
    mn, mx  = all_pts.min(axis=0), all_pts.max(axis=0)
    r = np.max(mx - mn) / 2; mid = (mn + mx) / 2
    ax.set_xlim(mid[0]-r, mid[0]+r); ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r); ax.set_box_aspect([1, 1, 1])
    path = os.path.join(out_dir, f"coordinate_systems_trial{test_num}.png")
    plt.savefig(path, dpi=220, bbox_inches="tight"); plt.show()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8 — MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main(test_num: int):
    base_dir = DATASET_DIR
    out_dir  = os.path.join(base_dir, f"lin{test_num}")
    os.makedirs(out_dir,    exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"Trial {test_num}  |  Smoothing: {SMOOTH_METHOD}")

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    ati_data  = load_ati_data(os.path.join(base_dir, ATI_FILE))
    box_df    = load_tag_file(os.path.join(base_dir, TAG1_FILE), tag_id=1)   # fixed box
    probe_df  = load_tag_file(os.path.join(base_dir, TAG2_FILE), tag_id=2)   # moving probe
    print(f"  Box tag = tag1 ({len(box_df)} frames)  |  Probe tag = tag2 ({len(probe_df)} frames)")
    F  = asof_join(probe_df, box_df, PAIR_TOL_S)
    tt = F["t"].to_numpy()

    # ── Step 2: Tracked poses in Camera frame (C) ─────────────────────────────
    pP_C  = F[["ppx","ppy","ppz"]].to_numpy()    # (N,3) probe origin in C
    pB_C  = F[["bpx","bpy","bpz"]].to_numpy()    # (N,3) box origin in C
    R_C_P = np.stack(F["PR"].to_numpy())           # (N,3,3) probe orientation in C
    R_C_B = np.stack(F["BR"].to_numpy())           # (N,3,3) box orientation in C
    R_C_B = stabilize_box_orientation(R_C_B)       # fix jitter on static box

    # ── Step 3: Build Top frame (T) from Box frame (B) ───────────────────────
    #   top_origin_C = pB_C + R_C_B @ d_T_in_B
    #   R_C_T        = R_C_B @ R_B_T
    d_T_in_B     = np.array([X_SHIFT, Y_SHIFT, Z_SHIFT])
    R_C_T        = np.matmul(R_C_B, R_B_T)                                    # (N,3,3)
    top_origin_C = pB_C + np.einsum("nij,j->ni", R_C_B, d_T_in_B)            # (N,3)
    R_T_C        = np.transpose(R_C_T, (0, 2, 1))                             # (N,3,3)

    # ── Step 4: Tip position in Top frame (T) ─────────────────────────────────
    #   tip_C   = pP_C + R_C_P @ tip_offset_in_P
    #   tip_T   = R_T_C @ (tip_C - top_origin_C)
    tip_C     = pP_C + np.einsum("nij,j->ni", R_C_P, TIP_OFFSET_IN_PROBE_M)  # (N,3)
    tip_T_raw = np.einsum("nij,nj->ni", R_T_C, tip_C - top_origin_C)         # (N,3) raw
    tip_T     = smooth_xyz(tip_T_raw, tt)                                      # (N,3) smooth
    X_top, Y_top, Z_top = tip_T[:, 0], tip_T[:, 1], tip_T[:, 2]

    # Optional in-plane rotation correction (aligns trajectory with box axes)
    rotation_angle_rad = 0.0
    if CORRECT_INPLANE_ROTATION:
        if MANUAL_ROTATION_ANGLE_DEG is None:
            rotation_angle_rad = estimate_inplane_rotation(X_top, Y_top)
            print(f"  Auto in-plane rotation: {np.rad2deg(rotation_angle_rad):.2f}°")
        else:
            rotation_angle_rad = np.deg2rad(MANUAL_ROTATION_ANGLE_DEG)
            print(f"  Manual in-plane rotation: {MANUAL_ROTATION_ANGLE_DEG:.2f}°")
        X_top, Y_top = apply_inplane_rotation(X_top, Y_top, rotation_angle_rad)

    cx = 0.5 * (np.nanmin(X_top) + np.nanmax(X_top))
    cy = 0.5 * (np.nanmin(Y_top) + np.nanmax(Y_top))
    cz = np.nanmedian(Z_top)
    Xc, Yc, Zc = X_top - cx, Y_top - cy, Z_top - cz

    # ── Step 5: Build ATI frame (A) from Probe frame (P) ─────────────────────
    #   ATI origin in Probe frame = tip_offset + ATI_ABOVE_TIP * axial_dir
    #   R_C_A = R_C_P @ R_P_A
    p_A_in_P = TIP_OFFSET_IN_PROBE_M + ATI_ABOVE_TIP_M * AXIAL_DIR_IN_PROBE
    pA_C     = pP_C + np.einsum("nij,j->ni", R_C_P, p_A_in_P)                # (N,3)
    R_C_A    = R_C_P @ R_P_A                                                   # (N,3,3)

    # ATI frame expressed in Top frame
    p_A_in_T = np.einsum("nij,nj->ni", R_T_C, pA_C - top_origin_C)           # (N,3)
    R_T_A    = np.einsum("nij,njk->nik", R_T_C, R_C_A)                        # (N,3,3)

    # ── Step 6: Transform ATI wrench → Top frame → transfer to tip ───────────
    #
    #   F_top        = R_T_A @ F_A                       (rotate forces into Top axes)
    #   M_top_origin = R_T_A @ M_A + p_A_in_T × F_top   (moment at Top origin)
    #   M_tip        = M_top_origin − tip_T × F_top      (transfer to tip/contact point)
    #
    #   Combined in one step:
    #   M_tip = R_T_A @ M_A + (p_A_in_T − tip_T_raw) × F_top
    #                          ↑_______________________↑
    #                          vector from tip to ATI origin, in Top frame
    #
    times_df = pd.DataFrame({"t": tt})
    W = pd.merge_asof(
        times_df.sort_values("t"),
        ati_data.sort_values("time"),
        left_on="t", right_on="time",
        direction="nearest", tolerance=PAIR_TOL_S,
    ).dropna()

    Fx_top = np.full(len(tt), np.nan)
    Fy_top, Fz_top = Fx_top.copy(), Fx_top.copy()
    Tx_top, Ty_top, Tz_top = Fx_top.copy(), Fx_top.copy(), Fx_top.copy()

    if len(W) > 0:
        idx = W.index.to_numpy()
        # Negate: ATI reports reaction force (environment→sensor); convention here
        # is applied force (sensor→environment), so flip sign on all components.
        F_A = -W[["Fx","Fy","Fz"]].to_numpy(float)
        M_A = -W[["Tx","Ty","Tz"]].to_numpy(float)

        # Wrench from ATI frame to Top frame, moment referenced to Top origin
        F_top_w, M_top_w = transport_wrench(R_T_A[idx], p_A_in_T[idx], F_A, M_A)
        Fx_top[idx], Fy_top[idx], Fz_top[idx] = F_top_w[:,0], F_top_w[:,1], F_top_w[:,2]
        Tx_top[idx], Ty_top[idx], Tz_top[idx] = M_top_w[:,0], M_top_w[:,1], M_top_w[:,2]

    # Apply same in-plane correction to force/torque XY components
    if CORRECT_INPLANE_ROTATION and rotation_angle_rad != 0.0:
        Fx_top, Fy_top = apply_inplane_rotation(Fx_top, Fy_top, rotation_angle_rad)
        Tx_top, Ty_top = apply_inplane_rotation(Tx_top, Ty_top, rotation_angle_rad)
        print("  Applied in-plane rotation to force/torque XY components.")

    # Transfer moment from Top origin to tip (contact point)
    F_stack      = np.vstack([Fx_top, Fy_top, Fz_top]).T
    r_top_to_tip = tip_T_raw                          # vector Top origin → tip, in Top frame
    r_cross_F    = np.cross(r_top_to_tip, F_stack)
    Mx_tip = Tx_top - r_cross_F[:, 0]
    My_tip = Ty_top - r_cross_F[:, 1]
    Mz_tip = Tz_top - r_cross_F[:, 2]

    ati_top_df = pd.DataFrame({
        "time": tt,
        "Fx": Fx_top, "Fy": Fy_top, "Fz": Fz_top,
        "Tx": Tx_top, "Ty": Ty_top, "Tz": Tz_top,
    }).dropna(subset=["Fx","Fy","Fz"])

    # ── Step 7: Save CSV ──────────────────────────────────────────────────────
    out = pd.DataFrame({
        # --- Tip position ---
        "t":                             tt,
        "tip_x_top_m":                   X_top,
        "tip_y_top_m":                   Y_top,
        "tip_z_top_m":                   Z_top,
        "tip_x_top_centered_m":          Xc,
        "tip_y_top_centered_m":          Yc,
        "tip_z_top_centered_m":          Zc,
        "tip_x_top_raw_m":               tip_T_raw[:, 0],
        "tip_y_top_raw_m":               tip_T_raw[:, 1],
        "tip_z_top_raw_m":               tip_T_raw[:, 2],
        "tip_x_cam_m":                   tip_C[:, 0],
        "tip_y_cam_m":                   tip_C[:, 1],
        "tip_z_cam_m":                   tip_C[:, 2],
        # --- Calibration metadata ---
        "p_tip_probe_x_m":               TIP_OFFSET_IN_PROBE_M[0],
        "p_tip_probe_y_m":               TIP_OFFSET_IN_PROBE_M[1],
        "p_tip_probe_z_m":               TIP_OFFSET_IN_PROBE_M[2],
        "top_center_shift_m_x":          cx,
        "top_center_shift_m_y":          cy,
        "top_center_shift_m_z":          cz,
        "inplane_rotation_correction_rad": rotation_angle_rad,
        "inplane_rotation_correction_deg": np.rad2deg(rotation_angle_rad),
        # --- Forces at tip, expressed in Top frame axes ---
        "Fx_top_N":                      Fx_top,
        "Fy_top_N":                      Fy_top,
        "Fz_top_N":                      Fz_top,
        # --- Moment at Top origin, Top frame axes (intermediate — rarely needed) ---
        "Tx_top_Nm":                     Tx_top,
        "Ty_top_Nm":                     Ty_top,
        "Tz_top_Nm":                     Tz_top,
        # --- Moment at tip (contact point), Top frame axes  ← USE THESE ---
        "Tx_tip_top_Nm":                 Mx_tip,
        "Ty_tip_top_Nm":                 My_tip,
        "Tz_tip_top_Nm":                 Mz_tip,
    })
    csv_path = os.path.join(out_dir, f"tip_path_top_frame_trial{test_num}.csv")
    out.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    print(f"  Tip X: [{np.min(X_top):.3f}, {np.max(X_top):.3f}] m")
    print(f"  Tip Y: [{np.min(Y_top):.3f}, {np.max(Y_top):.3f}] m")
    print(f"  Tip Z: {np.mean(Z_top):.3f} ± {np.std(Z_top):.3f} m")

    processing_data = pd.merge_asof(
        pd.DataFrame({"time": tt,
                      "x_position_mm": X_top*1000,
                      "y_position_mm": Y_top*1000,
                      "z_position_mm": Z_top*1000}).sort_values("time"),
        ati_top_df.sort_values("time"),
        on="time", direction="nearest", tolerance=PAIR_TOL_S,
    ).dropna(subset=["Fx","Fy","Fz"])
    proc_path = os.path.join(DATASET_DIR, f"processing_test_{test_num}.csv")
    processing_data.to_csv(proc_path, index=False)
    print(f"Saved processing data: {proc_path}")

    # ── Step 8: Plots ─────────────────────────────────────────────────────────
    plot_ati_wrenches(out_dir, ati_data,   test_num, frame_label="ATI Frame")
    plot_ati_wrenches(out_dir, ati_top_df, test_num, frame_label="Top Frame")
    plot_top_xy(out_dir, X_top, Y_top, Z_top, test_num)
    plot_top_3d(out_dir, X_top, Y_top, Z_top, test_num, z_limit_mm=Z_LIMIT_MM_2D)
    if not ati_top_df.empty:
        plot_top_path_with_forces(out_dir, X_top, Y_top, Z_top, tt, ati_top_df, test_num)
    plot_coordinate_systems(out_dir, pP_C, pB_C, tip_C, R_C_P, R_C_B, R_C_T,  top_origin_C, pA_C, R_C_A, test_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=TEST_NUM)
    args = parser.parse_args()
    main(args.test)
