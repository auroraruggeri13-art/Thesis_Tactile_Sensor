#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indenter tip path on the box top — with rotations + pivot calibration.

Steps
1) Load Atracsys (both rigid bodies).
2) Auto-detect PROBE vs BOX by motion variance.
3) Pair by nearest timestamp.
4) Pivot-calibrate tip offset in PROBE frame using an initial window.
5) Compute tip position in CAMERA, then express it in the TOP frame:
      Top origin = Box origin + R_box*[0,0,+0.045 m]
      Top axes   = R_box (optionally re-oriented to the best-fit plane)
6) Save CSV and 2D/3D plots into <dataset folder>/lin<test_num>/.

Only constant geometry assumption: BOX face balls are 45 mm below the top.
"""

import os, sys, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
from typing import Tuple
import argparse

# Make utils importable when running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from utils.sensor_io import load_ati_data, load_atracsys_data, asof_join

# If you run without CLI args, set your dataset directories here:
test_num = 1200
version_num = 1
directory_to_datasets = os.path.abspath(fr"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test {test_num} - sensor v{version_num}")
# Create directory if it doesn't exist
if not os.path.exists(directory_to_datasets):
    os.makedirs(directory_to_datasets)
    print(f"Created dataset directory: {directory_to_datasets}")

# Generate file paths with proper formatting
ati_file = "ati_middle_trial{}.txt".format(test_num)
atracsys_file = "atracsys_trial{}.txt".format(test_num)

DATASET_DIRS: list[str] = [
    os.path.join(directory_to_datasets, ati_file),
    os.path.join(directory_to_datasets, atracsys_file),
]

# ---------- Tunables ----------
PAIR_TOL_S = 0.020
z_limit = 3.0  # mm for 2D Top XY plot filtering
balls_to_top = -0.049  # meters
x_shift = 0.024  # meters, PROBE tip offset in Box X direction
y_shift = 0.0345    # meters, PROBE tip offset in Box Y direction

# --- Tip offset control ---
USE_FIXED_TIP_OFFSET = True
FIXED_TIP_OFFSET_IN_PROBE_m = np.array([-0.082, 0.0, -0.025])  # meters

# --- ATI mounting (A) relative to PROBE (P) ---
# If ATI axes are physically aligned with the PROBE axes, keep identity.
R_P_A = np.array([
    [0.0,  0.0,  1.0],
    [0.0,  1.0,  0.0],
    [1.0,  0.0,  0.0]
], dtype=float)

# Direction along the probe shaft in PROBE axes.
# Set this to the unit vector that points from the tip toward the sensor/handle.
AXIAL_DIR_IN_PROBE = np.array([1.0, 0.0, 0.0])  

ATI_ABOVE_TIP_m = 0.030  # 3 cm above the tip


# ---------- IO helpers ----------
def recognize_probe_and_box(df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    ids = df["marker_id"].unique()
    assert len(ids)>=2, "Need at least two marker_ids."
    stats=[]
    for k in ids:
        p=df[df["marker_id"]==k][["px","py","pz"]].to_numpy()
        stats.append((k, float(np.var(p,axis=0).sum())))
    stats.sort(key=lambda x: x[1])
    box_id, probe_id = stats[0][0], stats[-1][0]
    box   = df[df["marker_id"]==box_id  ].sort_values("t").reset_index(drop=True)
    probe = df[df["marker_id"]==probe_id].sort_values("t").reset_index(drop=True)
    return probe, box


# ---------- Homogeneous transform helpers ----------
def make_homogeneous_matrices(Rs: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Build Nx4x4 homogeneous transform matrices from Rs (N,3,3) and ts (N,3).
       If Rs or ts are single (3,3) or (3,), they are broadcast to length N=1.
    """
    Rs = np.asarray(Rs)
    ts = np.asarray(ts)
    if Rs.ndim == 2:
        Rs = Rs[np.newaxis, ...]
    if ts.ndim == 1:
        ts = ts[np.newaxis, ...]
    if Rs.shape[0] != ts.shape[0]:
        raise ValueError("Rs and ts must have same leading length")
    N = Rs.shape[0]
    Ts = np.zeros((N, 4, 4), dtype=Rs.dtype)
    Ts[:, :3, :3] = Rs
    Ts[:, :3, 3] = ts
    Ts[:, 3, 3] = 1.0
    return Ts

def invert_homogeneous_matrices(Ts: np.ndarray) -> np.ndarray:
    """Invert Nx4x4 homogeneous transforms efficiently using block inverse.
       inv([R, t; 0, 1]) = [R.T, -R.T @ t; 0, 1]
    """
    Ts = np.asarray(Ts)
    if Ts.ndim == 2:
        Ts = Ts[np.newaxis, ...]
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
    """Apply Nx4x4 transforms to either a single 3-vector (broadcast to all Ts)
       or to N 3-vectors (one-per-transform).

       - If points.shape == (3,), returns (N,3)
       - If points.shape == (N,3), returns (N,3)
    """
    Ts = np.asarray(Ts)
    points = np.asarray(points)
    if Ts.ndim == 2:
        Ts = Ts[np.newaxis, ...]
    N = Ts.shape[0]
    if points.ndim == 1:
        p_h = np.concatenate([points, [1.0]])
        # (N,4,4) @ (4,) -> (N,4)
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

def compose_homogeneous_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Return A ∘ B for 4x4 matrices.
             Supports broadcasting: (N,4,4)@(4,4) or (N,4,4)@(N,4,4).
        """
        A = np.asarray(A); B = np.asarray(B)
        if A.ndim == 2: A = A[np.newaxis, ...]
        if B.ndim == 2:
                return A @ B
        else:
                return np.einsum('nij,njk->nik', A, B)

def transform_wrenches_A_to_B(R_BA: np.ndarray, p_A_in_B: np.ndarray, F_A: np.ndarray, M_A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized wrench transform from frame A@A-origin to B@B-origin.

        Inputs (all length-N except constant rotations allowed):
            - R_BA: (N,3,3) rotation mapping A-vectors to B
            - p_A_in_B: (N,3) position of A-origin expressed in B
            - F_A: (N,3) force measured in A, at A-origin
            - M_A: (N,3) torque measured in A, about A-origin

        Returns:
            - F_B: (N,3)
            - M_B: (N,3)  where  M_B = R_BA M_A + (p_A_in_B × F_B)
        """
        # rotate force and torque
        F_B = np.einsum('nij,nj->ni', R_BA, F_A)
        M_rot = np.einsum('nij,nj->ni', R_BA, M_A)
        # shift torque to B-origin
        M_B = M_rot + np.cross(p_A_in_B, F_B)
        return F_B, M_B


# ---------- Plotting helpers ----------
def plot_top_path_with_forces(out_dir: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, times: np.ndarray, ati_df: pd.DataFrame, test_num: int):
    """Plot the tip path in the Top frame and draw force-direction arrows at 3 random times.

    - out_dir: folder to save the plot
    - X, Y, Z: tip positions in Top frame (meters)
    - times: timestamps (same length as X/Y/Z)
    - ati_df: ATI dataframe with columns ['time','Fx','Fy','Fz'] expressed in Top axes (or None)
    """
    # Require ATI data
    if ati_df is None:
        print("No ATI data available - skipping force-arrow plot")
        return

    # Build tip dataframe and merge with ATI by nearest timestamp
    tip_df = pd.DataFrame({
        'time': times,
        'x_mm': X * 1e3,
        'y_mm': Y * 1e3,
        'z_mm': Z * 1e3,
    }).sort_values('time').reset_index(drop=True)

    ati_sorted = ati_df.sort_values('time').reset_index(drop=True)
    merged = pd.merge_asof(tip_df, ati_sorted, on='time', direction='nearest', tolerance=PAIR_TOL_S)
    merged = merged.dropna(subset=['Fx','Fy','Fz']).reset_index(drop=True)
    if merged.empty:
        print("No synchronized ATI samples found within tolerance - skipping force-arrow plot")
        return

    # Choose up to 3 random samples
    rng = np.random.default_rng()
    n_samples = min(3, len(merged))
    idx = rng.choice(len(merged), size=n_samples, replace=False)
    sel = merged.iloc[idx]

    # Compute a scale so arrows are visible but not dominating the plot
    x_min, x_max = np.nanmin(tip_df['x_mm']), np.nanmax(tip_df['x_mm'])
    y_min, y_max = np.nanmin(tip_df['y_mm']), np.nanmax(tip_df['y_mm'])
    span = max(x_max - x_min, y_max - y_min, 1.0)
    forces = np.vstack([sel['Fx'].to_numpy(), sel['Fy'].to_numpy(), sel['Fz'].to_numpy()]).T
    maxF = np.linalg.norm(forces, axis=1).max()
    scale_mm_per_N = 0.2 * span / (maxF + 1e-9)

    # Create 3D figure in Top-frame coordinates
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(tip_df['x_mm'], tip_df['y_mm'], tip_df['z_mm'], '-', color='tab:blue', linewidth=1.0, alpha=0.8, label='Tip path')

    # For each selected time, draw three axis-aligned arrows representing Fx, Fy, Fz
    # Colors: Fx=red (X), Fy=green (Y), Fz=blue (Z)
    added_labels = set()
    for ii, row in sel.reset_index(drop=True).iterrows():
        ox, oy, oz = float(row['x_mm']), float(row['y_mm']), float(row['z_mm'])
        fx, fy, fz = float(row['Fx']), float(row['Fy']), float(row['Fz'])  # flip Fz so arrow points opposite in this plot
        # small offsets in X for arrow origins so the three arrows don't perfectly overlap
        origin_offsets = [(-0.6, 0.0, 0.0), (0.0, 0.0, 0.0), (0.6, 0.0, 0.0)]  # mm
        comps = [('Fx', fx, (1.0, 0.0, 0.0)), ('Fy', fy, (0.0, 0.6, 0.0)), ('Fz', fz, (0.0, 0.0, 1.0))]
        for j, (name, val, col) in enumerate(comps):
            dx = (val * scale_mm_per_N) if name == 'Fx' else 0.0
            dy = (val * scale_mm_per_N) if name == 'Fy' else 0.0
            dz = (val * scale_mm_per_N) if name == 'Fz' else 0.0
            offx, offy, offz = origin_offsets[j]
            oxo, oyo, ozo = ox + offx, oy + offy, oz + offz
            label = name if name not in added_labels else None
            ax.quiver(oxo, oyo, ozo, dx, dy, dz, color=col, linewidth=1.5, arrow_length_ratio=0.2, normalize=False, label=label)
            # annotate magnitude near arrow tip
            tipx, tipy, tipz = oxo + dx, oyo + dy, ozo + dz
            ax.text(tipx, tipy, tipz, f"{name}={val:.2f}N\nt={row['time']:.3f}s", color=col, fontsize=8)
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

def plot_top_xy(out_dir: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, test_num: int):
    """Plot 2D Top XY path and save PNG.

    Uses the same styling as the original inline code.
    """
    plt.figure(figsize=(8, 6))
    # select points within +-3 mm in Z (Top frame units -> convert to mm)
    z_mask = np.isfinite(Z) & (np.abs(Z * 1e3) <= z_limit)
    # Only plot the points that satisfy the masking condition. To avoid
    # joining separate segments (gaps where the mask is False), replace
    # masked-out coordinates with NaN — matplotlib will break the line at
    # NaNs, so distinct contiguous segments are not united.
    x_mm = X * 1e3
    y_mm = Y * 1e3
    x_plot = x_mm.copy()
    y_plot = y_mm.copy()
    x_plot[~z_mask] = np.nan
    y_plot[~z_mask] = np.nan
    plt.plot(x_plot, y_plot, linewidth=1.2)
    try:
        ax2d = plt.gca()
        rect_w, rect_h = 40.0, 16.0  # mm
        rect = Rectangle((-rect_w/2.0, -rect_h/2.0), rect_w, rect_h, fill=False, edgecolor='k', linewidth=1.0, linestyle='--', alpha=0.9)
        ax2d.add_patch(rect)
    except Exception:
        pass
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim([-25, 25]); plt.ylim([-10, 10])
    plt.xlabel("Top X [mm]"); plt.ylabel("Top Y [mm]")
    plt.title(f"Tip path on top surface (unflattened) — trial {test_num}")
    fig2d_nc_path = os.path.join(out_dir, f"tip_path_top_xy_raw_trial{test_num}.png")
    plt.savefig(fig2d_nc_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {fig2d_nc_path}")

def plot_top_3d(out_dir: str, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, test_num: int, z_limit):
    """Plot 3D tip path in Top frame and save PNG."""
    fig_nc = plt.figure()
    ax_nc = fig_nc.add_subplot(111, projection="3d")
    # Convert to mm for masking and plotting
    x_mm = X * 1e3
    y_mm = Y * 1e3
    z_mm = Z * 1e3
    
    # Keep only points within the requested Top-frame bounds (mm): X [-20,20], Y [-8,8], Z [-2,2]
    mask = (x_mm >= -20) & (x_mm <= 20) & (y_mm >= -8) & (y_mm <= 8) & (z_mm >= -z_limit) & (z_mm <= z_limit)
    
    if np.any(mask):
        # Find continuous segments within the mask
        # Identify transitions: where mask changes from False to True or True to False
        mask_padded = np.concatenate([[False], mask, [False]])
        diff = np.diff(mask_padded.astype(int))
        starts = np.where(diff == 1)[0]  # Start of valid segments
        ends = np.where(diff == -1)[0]    # End of valid segments
        
        # Plot each continuous segment separately with the same color
        for start, end in zip(starts, ends):
            ax_nc.plot(x_mm[start:end], y_mm[start:end], z_mm[start:end], 
                      linewidth=1.0, color='C0')  # C0 is matplotlib's default blue
        
        title_extra = " (filtered to ±20×±8×±2 mm)"
    else:
        # fallback: plot all points if none satisfy the mask
        ax_nc.plot(x_mm, y_mm, z_mm, linewidth=1.0)
        title_extra = " (no points in filtered window — showing all)"
    
    ax_nc.set_xlabel("Top X [mm]")
    ax_nc.set_ylabel("Top Y [mm]")
    ax_nc.set_zlabel("Top Z [mm]")
    ax_nc.set_title(f"Tip path in Top frame (unflattened){title_extra} — trial {test_num}")
    ax_nc.set_box_aspect([1, 1, 0.3])
    
    fig3d_nc_path = os.path.join(out_dir, f"tip_path_top_3d_raw_trial{test_num}.png")
    plt.savefig(fig3d_nc_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {fig3d_nc_path}")

def plot_coordinate_systems(out_dir: str,
                            pP_C: np.ndarray, pB_C: np.ndarray, tip_C: np.ndarray,
                            R_C_P: np.ndarray, R_C_B: np.ndarray, R_C_T: np.ndarray,
                            top_origin_C: np.ndarray, ati_origin_C: np.ndarray, R_C_A: np.ndarray,
                            test_num: int):
    """Plot probe/box/tip/ATI coordinate systems in CAMERA frame and save PNG."""
    fig_coords = plt.figure(figsize=(10, 8))
    ax_coords = fig_coords.add_subplot(111, projection="3d")
    ax_coords.plot(pP_C[:, 0]*1000, pP_C[:, 1]*1000, pP_C[:, 2]*1000, 'b-', label='Probe Path', alpha=0.5)
    ax_coords.plot(pB_C[:, 0]*1000, pB_C[:, 1]*1000, pB_C[:, 2]*1000, 'r-', label='Box Path', alpha=0.5)
    ax_coords.plot(tip_C[:, 0]*1000, tip_C[:, 1]*1000, tip_C[:, 2]*1000, 'g-', label='Tip Path', alpha=0.5)
    z_min_idx = np.argmin(tip_C[:, 2])
    axis_length = 0.05 * 1000
    origin_tip = tip_C[z_min_idx] * 1000
    for i, (color, label) in enumerate([('r','Tip X'),('g','Tip Y'),('b','Tip Z')]):
        direction = R_C_P[z_min_idx, :, i] * axis_length
        ax_coords.quiver(origin_tip[0], origin_tip[1], origin_tip[2], direction[0], direction[1], direction[2], color=color, alpha=1.0, label=label)
    origin_P = pP_C[z_min_idx] * 1000
    for i, (color, label) in enumerate([('r','Probe X'),('g','Probe Y'),('b','Probe Z')]):
        direction = R_C_P[z_min_idx, :, i] * axis_length
        ax_coords.quiver(origin_P[0], origin_P[1], origin_P[2], direction[0], direction[1], direction[2], color=color, alpha=0.6, label=label)
    origin_B = pB_C[z_min_idx] * 1000
    for i, (color, label) in enumerate([('r','Box X'),('g','Box Y'),('b','Box Z')]):
        direction = R_C_B[z_min_idx, :, i] * axis_length
        ax_coords.quiver(origin_B[0], origin_B[1], origin_B[2], direction[0], direction[1], direction[2], color=color, alpha=0.6, label=label)
    origin_T = top_origin_C[z_min_idx] * 1000
    for i, (color, label) in enumerate([('r','Top X'),('g','Top Y'),('b','Top Z')]):
        direction = R_C_T[z_min_idx, :, i] * axis_length
        ax_coords.quiver(origin_T[0], origin_T[1], origin_T[2], direction[0], direction[1], direction[2], color=color, alpha=0.6, label=label)
    origin_A = ati_origin_C[z_min_idx] * 1000
    for i, label in enumerate(['ATI X', 'ATI Y', 'ATI Z']):
        direction = R_C_A[z_min_idx, :, i] * axis_length
        ax_coords.quiver(origin_A[0], origin_A[1], origin_A[2], direction[0], direction[1], direction[2], color='orange', alpha=0.8, label=label, linewidth=2)
    ax_coords.set_xlabel('Camera X [mm]')
    ax_coords.set_ylabel('Camera Y [mm]')
    ax_coords.set_zlabel('Camera Z [mm]')
    ax_coords.legend()
    ax_coords.set_title('Probe, Box, Top, and ATI Coordinate Systems in Camera Frame')
    coords_path = os.path.join(out_dir, f"coordinate_systems_trial{test_num}.png")
    plt.savefig(coords_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved coordinate systems plot: {coords_path}")

def plot_ati_wrenches(out_dir: str, ati_df: pd.DataFrame, test_num: int, frame_label: str = ""):
    """Plot ATI force and torque time series and save PNG.
    
    Parameters:
    -----------
    out_dir : str
        Output directory for saving plots
    ati_df : pd.DataFrame
        DataFrame with columns ['time', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    test_num : int
        Test number for filename
    frame_label : str
        Label to add to title and filename (e.g., "ATI Frame" or "Top Frame")
    """
    if ati_df is None or ati_df.empty:
        print(f"No ATI data available for {frame_label} - skipping wrench plot")
        return

    # Remove NaN values
    ati_clean = ati_df.dropna(subset=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
    
    if ati_clean.empty:
        print(f"No valid ATI data after removing NaNs for {frame_label} - skipping wrench plot")
        return

    time = ati_clean['time'].to_numpy()
    Fx = ati_clean['Fx'].to_numpy()
    Fy = ati_clean['Fy'].to_numpy()
    Fz = ati_clean['Fz'].to_numpy()
    Tx = ati_clean['Tx'].to_numpy()
    Ty = ati_clean['Ty'].to_numpy()
    Tz = ati_clean['Tz'].to_numpy()

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Force plot
    axs[0].plot(time, Fx, label='Fx', color='r', linewidth=1.5)
    axs[0].plot(time, Fy, label='Fy', color='g', linewidth=1.5)
    axs[0].plot(time, Fz, label='Fz', color='b', linewidth=1.5)
    axs[0].set_ylabel('Force [N]', fontsize=12)
    title_force = f'ATI Force Time Series — trial {test_num}'
    if frame_label:
        title_force += f' ({frame_label})'
    axs[0].set_title(title_force, fontsize=13, fontweight='bold')
    axs[0].legend(loc='best', fontsize=10)
    axs[0].grid(True, alpha=0.3)

    # Torque plot
    axs[1].plot(time, Tx, label='Tx', color='r', linewidth=1.5)
    axs[1].plot(time, Ty, label='Ty', color='g', linewidth=1.5)
    axs[1].plot(time, Tz, label='Tz', color='b', linewidth=1.5)
    axs[1].set_xlabel('Time [s]', fontsize=12)
    axs[1].set_ylabel('Torque [Nm]', fontsize=12)
    title_torque = f'ATI Torque Time Series — trial {test_num}'
    if frame_label:
        title_torque += f' ({frame_label})'
    axs[1].set_title(title_torque, fontsize=13, fontweight='bold')
    axs[1].legend(loc='best', fontsize=10)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Generate filename with frame label
    filename_suffix = f"_{frame_label.lower().replace(' ', '_')}" if frame_label else ""
    wrench_path = os.path.join(out_dir, f"ati_wrenches{filename_suffix}_trial{test_num}.png")
    plt.savefig(wrench_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved ATI wrench plot ({frame_label}): {wrench_path}")
    
# ---------- Main ----------
def main(atracsys_path: str, test_num: int):
    # Original output directory for tip path data
    base_dir = os.path.dirname(os.path.abspath(atracsys_path))
    out_dir = os.path.join(base_dir, f"lin{test_num}")
    os.makedirs(out_dir, exist_ok=True)

    # New directory for processing files
    processing_dir = directory_to_datasets
    os.makedirs(processing_dir, exist_ok=True)

    # Load ATI data
    ati_path = os.path.join(base_dir, f"ati_middle_trial{test_num}.txt")
    ati_data = None
    if os.path.exists(ati_path):
        ati_data = load_ati_data(ati_path)
        if ati_data is not None:
            # Convert ATI timestamps to match Atracsys if needed
            if ati_data['time'].abs().median() > 1e12:
                ati_data['time'] = ati_data['time'] * 1e-9
            elif ati_data['time'].abs().median() > 1e9:
                ati_data['time'] = ati_data['time'] * 1e-6
            elif ati_data['time'].abs().median() > 1e6:
                ati_data['time'] = ati_data['time'] * 1e-3

    df = load_atracsys_data(atracsys_path)
    probe, box = recognize_probe_and_box(df)
    F = asof_join(probe, box, PAIR_TOL_S)

    # Data arrays
    pP_C = F[["ppx","ppy","ppz"]].to_numpy()
    R_C_P = np.stack(F["PR"].to_numpy())
    pB_C = F[["bpx","bpy","bpz"]].to_numpy()
    R_C_B = np.stack(F["BR"].to_numpy())
    tt    = F["t"].to_numpy()
    
    p_tip_probe = FIXED_TIP_OFFSET_IN_PROBE_m


    # 1) First apply tip offset in PROBE frame (before any frame transformations)    
    # Tip in CAMERA = T_C_P ∘ p_tip_probe
    T_C_P = make_homogeneous_matrices(R_C_P, pP_C)     # (N,4,4), Probe→Camera
    p_tip_probe = np.asarray(p_tip_probe, dtype=float) # ensure shape (3,)
    tip_C = apply_homogeneous_matrices(T_C_P, p_tip_probe)  # (N,3)


    # ---- Fixed Box→Top transform (LOCKED to Box geometry) ----
    # Top axes expressed in Box frame (columns = Top axes in Box):
    # Top X = Box X
    # Top Y = Box Z
    # Top Z = -Box Y      <-- sign flip to keep det = +1
    R_B_T = np.array([
        [ 1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  -1.0,  0.0],
    ], dtype=float).T
    # det(R_B_T) = +1

    # Top origin offset is given in **Box axes** (face balls are 41 mm below Top)
    d_T_in_B = np.array([x_shift, balls_to_top, y_shift], dtype=float)  # meters, in Box axes

    # Compose to Camera
    R_C_T = np.matmul(R_C_B, R_B_T)                                   # (N,3,3)
    top_origin_C = pB_C + np.einsum('nij,j->ni', R_C_B, d_T_in_B)     # (N,3)

    # Build Top transforms (frame locked to Box)
    T_C_Top = make_homogeneous_matrices(R_C_T, top_origin_C)
    T_Top_C = invert_homogeneous_matrices(T_C_Top)

    # ----- Compute ATI origin in PROBE and the chain to TOP -----
    # ATI origin in PROBE axes: tip position + 2 cm along the probe shaft
    p_A_in_P = p_tip_probe + ATI_ABOVE_TIP_m * AXIAL_DIR_IN_PROBE  # (3,)

    # Constant PROBE->ATI (A expressed in P)
    T_P_A = make_homogeneous_matrices(R_P_A, p_A_in_P)   # (1,4,4)

    # Per-sample CAMERA->ATI: T_C_A = T_C_P ∘ T_P_A
    T_C_A = compose_homogeneous_matrices(T_C_P, T_P_A)   # (N,4,4)

    # Per-sample TOP->ATI: T_T_A = T_Top_C ∘ T_C_A
    T_T_A = compose_homogeneous_matrices(T_Top_C, T_C_A) # (N,4,4)
    R_T_A = T_T_A[:, :3, :3]                # (N,3,3)
    p_A_in_T = T_T_A[:, :3, 3]              # (N,3)  (position of ATI-origin in Top axes)

    # ----- Synchronize ATI samples to F frames and transform wrenches -----
    ati_top_df = None
    if ati_data is not None and not ati_data.empty:
        # asof-merge ATI to the timestamps used in F (tt)
        times_df = pd.DataFrame({'t': tt})
        W = pd.merge_asof(times_df.sort_values('t'),
                          ati_data.sort_values('time'),
                          left_on='t', right_on='time',
                          direction='nearest', tolerance=PAIR_TOL_S).dropna()

        if not W.empty:
            # indices into per-sample transforms
            idx = W.index.to_numpy()

            F_A = W[['Fx','Fy','Fz']].to_numpy(float)
            M_A = W[['Tx','Ty','Tz']].to_numpy(float)

            # select transforms for those timestamps
            R_T_A_sel  = R_T_A[idx]
            p_A_in_T_sel = p_A_in_T[idx]

            # Transform the wrench (ATI->Top)
            F_T, M_T = transform_wrenches_A_to_B(R_T_A_sel, p_A_in_T_sel, F_A, M_A)

            # Allocate aligned arrays (NaN where no ATI sample)
            Fx_top = np.full(len(tt), np.nan); Fy_top = Fx_top.copy(); Fz_top = Fx_top.copy()
            Tx_top = Fx_top.copy(); Ty_top = Fx_top.copy(); Tz_top = Fx_top.copy()

            Fx_top[idx] = F_T[:,0]; Fy_top[idx] = F_T[:,1]; Fz_top[idx] = F_T[:,2]
            Tx_top[idx] = M_T[:,0]; Ty_top[idx] = M_T[:,1]; Tz_top[idx] = M_T[:,2]

            # Keep for plotting/processing merges
            ati_top_df = pd.DataFrame({
                'time': tt,
                'Fx': Fx_top, 'Fy': Fy_top, 'Fz': Fz_top,
                'Tx': Tx_top, 'Ty': Ty_top, 'Tz': Tz_top
            }).dropna(subset=['Fx','Fy','Fz'])  # keep only synchronized rows

            # Also attach to the main CSV for traceability
            # Will add columns to 'out' after out DataFrame exists (we'll add placeholders here)
            fx_col = Fx_top; fy_col = Fy_top; fz_col = Fz_top
            tx_col = Tx_top; ty_col = Ty_top; tz_col = Tz_top

    # Tip positions in Top frame
    tip_in_top = apply_homogeneous_matrices(T_Top_C, tip_C)  # (N,3)
    
    # --- Top-frame coordinates ---
    X_top = tip_in_top[:, 0]
    Y_top = tip_in_top[:, 1]
    Z_top = tip_in_top[:, 2]

    # Center 
    cx = 0.5 * (np.nanmin(X_top) + np.nanmax(X_top))
    cy = 0.5 * (np.nanmin(Y_top) + np.nanmax(Y_top))
    cz = np.nanmedian(Z_top)
    Xc, Yc, Zc = X_top - cx, Y_top - cy, Z_top - cz

    # ---- DataFrame ----
    out = pd.DataFrame({
        "t": tt,
        # Top-frame coordinates
        "tip_x_top_m": X_top, "tip_y_top_m": Y_top, "tip_z_top_m": Z_top,
        "tip_x_top_centered_m": Xc, "tip_y_top_centered_m": Yc, "tip_z_top_centered_m": Zc,
        # Camera-space tip (for traceability)
        "tip_x_cam_m": tip_C[:, 0], "tip_y_cam_m": tip_C[:, 1], "tip_z_cam_m": tip_C[:, 2],
        # Calibration/debug
        "p_tip_probe_x_m": [p_tip_probe[0]]*len(tt),
        "p_tip_probe_y_m": [p_tip_probe[1]]*len(tt),
        "p_tip_probe_z_m": [p_tip_probe[2]]*len(tt),
        "top_center_shift_m_x": [cx]*len(tt),
        "top_center_shift_m_y": [cy]*len(tt),
        "top_center_shift_m_z": [cz]*len(tt),
    })

    # Attach Top-frame wrench columns if they were computed above (aligned to tt)
    if 'fx_col' in locals():
        out['Fx_top_N']  = fx_col
        out['Fy_top_N']  = fy_col
        out['Fz_top_N']  = fz_col
        out['Tx_top_Nm'] = tx_col
        out['Ty_top_Nm'] = ty_col
        out['Tz_top_Nm'] = tz_col

        # Optional: torque about the tip point (still expressed in Top axes)
        # r: from Top origin to tip (tip_in_top is in Top axes already)
        r_T_to_tip = tip_in_top  # (N,3)
        F_stack = np.vstack([fx_col, fy_col, fz_col]).T
        # compute cross product r x F (per-sample)
        r_cross_F = np.cross(r_T_to_tip, F_stack)
        out['Tx_tip_top_Nm'] = out['Tx_top_Nm'] - r_cross_F[:, 0]
        out['Ty_tip_top_Nm'] = out['Ty_top_Nm'] - r_cross_F[:, 1]
        out['Tz_tip_top_Nm'] = out['Tz_top_Nm'] - r_cross_F[:, 2]
    else:
        out['Fx_top_N']  = [np.nan]*len(tt)
        out['Fy_top_N']  = [np.nan]*len(tt)
        out['Fz_top_N']  = [np.nan]*len(tt)
        out['Tx_top_Nm'] = [np.nan]*len(tt)
        out['Ty_top_Nm'] = [np.nan]*len(tt)
        out['Tz_top_Nm'] = [np.nan]*len(tt)

    # Plot ATI wrenches before and after transform 
    try:
        # Before transformation (ATI frame)
        if ati_data is not None and not ati_data.empty:
            plot_ati_wrenches(out_dir, ati_data, test_num, frame_label="ATI Frame")
        
        # After transformation (Top frame)
        if 'ati_top_df' in locals() and ati_top_df is not None and not ati_top_df.empty:
            plot_ati_wrenches(out_dir, ati_top_df, test_num, frame_label="Top Frame")
    except Exception as e:
        print(f"Skipping ATI wrench plots due to error: {e}")

    csv_path = os.path.join(out_dir, f"tip_path_top_frame_trial{test_num}.csv")
    out.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    print(f"\nTip position statistics (Top frame, unflattened):")
    print(f"X range: [{np.min(X_top):.3f}, {np.max(X_top):.3f}] m")
    print(f"Y range: [{np.min(Y_top):.3f}, {np.max(Y_top):.3f}] m")
    print(f"Z mean ± std: {np.mean(Z_top):.3f} ± {np.std(Z_top):.3f} m\n")

    # Use Top-frame coordinates for plotting (plotting on Top XY)
    Xp, Yp, Zp = X_top, Y_top, Z_top
    try:
        plot_top_xy(out_dir, Xp, Yp, Zp, test_num)
    except Exception as e:
        print(f"Skipping 2D Top XY plot due to error: {e}")

    # --- Extra plot: tip path with force direction arrows at 3 random times ---
    try:
        # Plot using ATI data transformed into Top axes (if available)
        plot_top_path_with_forces(out_dir, X_top, Y_top, Z_top, tt, 
                                 ati_top_df if 'ati_top_df' in locals() else None, test_num)
    except Exception as e:
        print(f"Skipping force-arrow plot due to error: {e}")

    try:
        plot_top_3d(out_dir, Xp, Yp, Zp, test_num, z_limit)
    except Exception as e:
        print(f"Skipping 3D Top plot due to error: {e}")

    try:
        # Extract ATI origin and rotation in Camera frame from transforms
        ati_origin_C = T_C_A[:, :3, 3]  # (N,3)
        R_C_A = T_C_A[:, :3, :3]  # (N,3,3)
        plot_coordinate_systems(out_dir, pP_C, pB_C, tip_C, R_C_P, R_C_B, R_C_T, top_origin_C, ati_origin_C, R_C_A, test_num)
    except Exception as e:
        print(f"Skipping coordinate systems plot due to error: {e}")

    # Create combined processing file with tip positions and Top-frame forces/torques
    if ati_data is not None and 'ati_top_df' in locals() and ati_top_df is not None and not ati_top_df.empty:
        # Use Top-frame coordinates for both position and wrench (unflattened)
        x_pos = X_top * 1000  # Convert to mm
        y_pos = Y_top * 1000
        z_pos = Z_top * 1000

        tip_data = pd.DataFrame({
            'time': tt,
            'x_position_mm': x_pos,
            'y_position_mm': y_pos,
            'z_position_mm': z_pos
        }).sort_values('time')

        # Already transformed to Top axes & Top origin
        ati_top_sorted = ati_top_df.sort_values('time')

        processing_data = pd.merge_asof(
            tip_data, ati_top_sorted, 
            left_on='time', right_on='time',
            direction='nearest', tolerance=PAIR_TOL_S
        ).dropna(subset=['Fx', 'Fy', 'Fz'])

        processing_path = os.path.join(processing_dir, f"processing_test_{test_num}.csv")
        processing_data.to_csv(processing_path, index=False)
        print(f"\nSaved processing data: {processing_path}")
        print(f"Processing file contains {len(processing_data)} rows (Top frame):")
        print("- time, x_position_mm, y_position_mm, z_position_mm, Fx, Fy, Fz, Tx, Ty, Tz")
        
        # Print statistics
        print(f"\nProcessing data statistics:")
        print(f"Time range: [{processing_data['time'].min():.3f}, {processing_data['time'].max():.3f}] s")
        print(f"X range: [{processing_data['x_position_mm'].min():.2f}, {processing_data['x_position_mm'].max():.2f}] mm")
        print(f"Y range: [{processing_data['y_position_mm'].min():.2f}, {processing_data['y_position_mm'].max():.2f}] mm")
        print(f"Z range: [{processing_data['z_position_mm'].min():.2f}, {processing_data['z_position_mm'].max():.2f}] mm")
        print(f"Fz range: [{processing_data['Fz'].min():.2f}, {processing_data['Fz'].max():.2f}] N")
    else:
        print("\nNo synchronized Top-frame ATI data found - skipping processing file creation")
        if ati_data is None:
            print("  Reason: No ATI data loaded")
        elif 'ati_top_df' not in locals() or ati_top_df is None:
            print("  Reason: ATI data not transformed to Top frame")
        elif ati_top_df.empty:
            print("  Reason: Transformed ATI data is empty")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tip path on box top with rotations + pivot calibration.")
    parser.add_argument("--atracsys", type=str, default=None, help="Path to atracsys_trialXX.txt/.csv")
    parser.add_argument("--test", type=int, default=test_num, help="Test number for output folder name")
    args = parser.parse_args()

    atr_path = args.atracsys or (DATASET_DIRS[1] if len(DATASET_DIRS)>1 else None)
    
    # Print debug information
    print(f"\nChecking paths:")
    print(f"Directory to datasets: {directory_to_datasets}")
    print(f"ATI data path: {DATASET_DIRS[0]}")
    print(f"Atracsys path: {atr_path}")
    
    if not os.path.exists(directory_to_datasets):
        sys.exit(f"Dataset directory not found: {directory_to_datasets}")
    
    if not atr_path or not os.path.exists(atr_path):
        sys.exit(f"Atracsys file not found: {atr_path}\nPlease check if the file exists or pass --atracsys with the correct path.")
    main(atr_path, args.test)
