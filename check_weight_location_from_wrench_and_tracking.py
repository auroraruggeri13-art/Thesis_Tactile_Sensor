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
from typing import Tuple
import argparse

# If you run without CLI args, set your dataset directories here:
test_num = 110
version_num = 4
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
PIVOT_SECONDS = 0.02       # first seconds used for pivot LSQ
PLANE_FLATTEN = True       # rotate Top so the traced surface is flat (z≈0)

# --- Tip offset control ---
USE_FIXED_TIP_OFFSET = True
FIXED_TIP_OFFSET_IN_PROBE_m = np.array([-0.09, 0.0, -0.025])  # meters


# ---------- IO helpers ----------
def load_ati_data(path: str) -> pd.DataFrame:
    """Load ATI F/T sensor data from text file."""
    try:
        # Read the CSV with its header
        df = pd.read_csv(path)
        
        # Extract relevant columns and rename them
        time_col = [col for col in df.columns if 'time' in col.lower()][0]
        force_cols = [col for col in df.columns if 'force' in col.lower()]
        torque_cols = [col for col in df.columns if 'torque' in col.lower()]
        
        df_processed = pd.DataFrame({
            'time': df[time_col],
            'Fx': df[force_cols[0]],
            'Fy': df[force_cols[1]],
            'Fz': df[force_cols[2]],
            'Tx': df[torque_cols[0]],
            'Ty': df[torque_cols[1]],
            'Tz': df[torque_cols[2]]
        })
        
        # Convert time and F/T columns to float
        df_processed = df_processed.astype(float)
        return df_processed
    except Exception as e:
        print(f"Error loading ATI data: {e}")
        return None

def load_atracsys_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python")
    df.columns = [c.strip() for c in df.columns]
    def pick(regex: str) -> list[str]:
        rx = re.compile(regex, re.IGNORECASE)
        cols = [c for c in df.columns if rx.fullmatch(c)]
        return cols if cols else [c for c in df.columns if re.search(regex, c, re.IGNORECASE)]

    tcol = (pick(r"(?:%?time|field\.timestamp)") or ["t"])[0]
    mid = ( [c for c in df.columns if re.search("marker_id", c, re.IGNORECASE)] )[0]
    pcols = [f"field.position{i}" for i in range(3)]
    if not all(c in df.columns for c in pcols):
        pcols = [c for c in df.columns if re.search(r"position[0-2]$", c)]
        assert len(pcols)==3, "position0..2 not found"
        pcols = sorted(pcols, key=lambda s:int(re.search(r"(\d)$",s).group(1)))
    rcols = [f"field.rotation{i}" for i in range(9)]
    if not all(c in df.columns for c in rcols):
        rcols = [c for c in df.columns if re.search(r"rotation[0-8]$", c)]
        assert len(rcols)==9, "rotation0..8 not found"
        rcols = sorted(rcols, key=lambda s:int(re.search(r"(\d)$",s).group(1)))

    df = df[[tcol, mid] + pcols + rcols].rename(columns={tcol:"t", mid:"marker_id"}).copy()

    t = df["t"].astype(float)
    if t.abs().median()>1e12: df["t"]=t*1e-9
    elif t.abs().median()>1e9: df["t"]=t*1e-6
    elif t.abs().median()>1e6: df["t"]=t*1e-3

    df["px"] = df[pcols[0]].astype(float)*1e-3
    df["py"] = df[pcols[1]].astype(float)*1e-3
    df["pz"] = df[pcols[2]].astype(float)*1e-3
    R = df[rcols].to_numpy(float).reshape((-1,3,3))
    df["R"] = list(R)
    return df[["t","marker_id","px","py","pz","R"]]

def split_probe_and_box(df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
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

def asof_join(probe: pd.DataFrame, box: pd.DataFrame, tol_s: float) -> pd.DataFrame:
    a = probe[["t","px","py","pz","R"]].copy(); a.columns=["t","ppx","ppy","ppz","PR"]
    b = box  [["t","px","py","pz","R"]].copy(); b.columns=["t_box","bpx","bpy","bpz","BR"]
    out = pd.merge_asof(
        a.sort_values("t"), b.sort_values("t_box"),
        left_on="t", right_on="t_box", direction="nearest", tolerance=tol_s
    ).dropna(subset=["t_box"])
    return out.reset_index(drop=True)

def pivot_calibrate(R_C_P, pP_C):
    # Solve R_i p + t_i = s (constant) for p (tip in probe) and s (tip in camera)
    N = R_C_P.shape[0]
    A = np.zeros((3*N, 6)); b = np.zeros(3*N)
    for i in range(N):
        A[3*i:3*i+3, 0:3] = R_C_P[i]
        A[3*i:3*i+3, 3:6] = -np.eye(3)
        b[3*i:3*i+3] = -pP_C[i]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    p_tip_probe = x[:3]; tip_cam = x[3:]
    return p_tip_probe, tip_cam

# ---------- Math ----------

def best_fit_plane(points: np.ndarray):
    """Return (origin, normal, R_top_align) that rotates Z to the plane normal."""
    ctr = points.mean(axis=0)
    P = points - ctr
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    n = vh[-1, :]  # smallest variance -> normal
    n = n / np.linalg.norm(n)
    # rotation that maps current z-axis [0,0,1] to n
    z = np.array([0.,0.,1.])
    v = np.cross(z, n); s = np.linalg.norm(v); c = float(np.dot(z, n))
    if s < 1e-8:
        R = np.eye(3) if c>0 else np.diag([1,1,-1])
    else:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + vx@vx * ((1-c)/(s**2))
    return ctr, n, R


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

# ---------- Main ----------
def main(atracsys_path: str, test_num: int):
    # Original output directory for tip path data
    base_dir = os.path.dirname(os.path.abspath(atracsys_path))
    out_dir = os.path.join(base_dir, f"lin{test_num}")
    os.makedirs(out_dir, exist_ok=True)

    # New directory for processing files
    processing_dir = os.path.abspath(fr"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\test data\test {test_num} - sensor v{version_num}")
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

    df = load_atracsys_csv(atracsys_path)
    probe, box = split_probe_and_box(df)
    F = asof_join(probe, box, PAIR_TOL_S)

    # Data arrays
    pP_C = F[["ppx","ppy","ppz"]].to_numpy()
    R_C_P = np.stack(F["PR"].to_numpy())
    pB_C = F[["bpx","bpy","bpz"]].to_numpy()
    R_C_B = np.stack(F["BR"].to_numpy())
    tt    = F["t"].to_numpy()

    # Pivot calibration (still compute for reference)
    M = min(len(R_C_P), int(PIVOT_SECONDS / np.median(np.diff(tt))))
    p_tip_probe_pivot, _ = pivot_calibrate(R_C_P[:M], pP_C[:M])

    # Choose tip offset source
    if USE_FIXED_TIP_OFFSET:
        p_tip_probe = FIXED_TIP_OFFSET_IN_PROBE_m
        print("Using FIXED tip offset in PROBE frame:", p_tip_probe)
        print("Pivot-estimated tip (ignored):", p_tip_probe_pivot)
    else:
        p_tip_probe = p_tip_probe_pivot
        print("Using PIVOT-calibrated tip offset from first", PIVOT_SECONDS, "seconds.")


    # 1) First apply tip offset in PROBE frame (before any frame transformations)    
    # Tip in CAMERA = T_C_P ∘ p_tip_probe
    T_C_P = make_homogeneous_matrices(R_C_P, pP_C)     # (N,4,4), Probe→Camera
    p_tip_probe = np.asarray(p_tip_probe, dtype=float) # ensure shape (3,)
    tip_C = apply_homogeneous_matrices(T_C_P, p_tip_probe)  # (N,3)


    # ---- Fixed Box→Top transform (LOCKED to Box geometry) ----
    # Top frame relative to Box frame:
    # Top X = Box X (horizontal)
    # Top Y = Box Z (depth)
    # Top Z = Box Y (vertical, up) - Note: Y axis is now reversed
    R_B_T = np.array([
        [ 1.0,  0.0,  0.0],  # Top X = Box X (horizontal)
        [ 0.0,  0.0,  1.0],  # Top Y = Box Z (depth)
        [ 0.0,  1.0,  0.0],  # Top Z = Box Y (vertical, up) - Y axis reversed
    ], dtype=float)
    # Sanity: det(R_B_T) == +1 and columns orthonormal.

    # Top origin offset is given in **Box axes** (face balls are 45 mm below Top)
    d_T_in_B = np.array([0.03, -0.045, 0.030], dtype=float)  # meters, in Box axes

    # Compose to Camera
    R_C_T = np.matmul(R_C_B, R_B_T)                                   # (N,3,3)
    top_origin_C = pB_C + np.einsum('nij,j->ni', R_C_B, d_T_in_B)     # (N,3)

    # Build Top transforms (frame locked to Box)
    T_C_Top = make_homogeneous_matrices(R_C_T, top_origin_C)
    T_Top_C = invert_homogeneous_matrices(T_C_Top)

    # Tip positions in the (unflattened) Top frame
    tip_in_top = apply_homogeneous_matrices(T_Top_C, tip_C)  # (N,3)

    # Optional: plane flatten **points only** (keep Box→Top mapping intact)
    tip_in_top_flat = None
    if PLANE_FLATTEN:
        ctr, n, R_flat = best_fit_plane(tip_in_top)  # R_flat rotates z->n
        # To map the plane normal back to +z (flatten), rotate points by R_n2z:
        # With row-vectors, multiply on the right by R_flat (see derivation).
        tip_in_top_flat = (tip_in_top - ctr) @ R_flat + ctr  # shape (N,3)


    
    # --- Unflattened Top-frame coordinates ---
    X_top = tip_in_top[:, 0]
    Y_top = tip_in_top[:, 1]
    Z_top = tip_in_top[:, 2]

    # --- Flattened Top-frame coordinates (if requested) ---
    if tip_in_top_flat is not None:
        X_top_flat = tip_in_top_flat[:, 0]
        Y_top_flat = tip_in_top_flat[:, 1]
        Z_top_flat = tip_in_top_flat[:, 2]
    else:
        X_top_flat = Y_top_flat = Z_top_flat = None

    # Center (unflattened)
    cx = 0.5 * (np.nanmin(X_top) + np.nanmax(X_top))
    cy = 0.5 * (np.nanmin(Y_top) + np.nanmax(Y_top))
    cz = np.nanmedian(Z_top)
    Xc, Yc, Zc = X_top - cx, Y_top - cy, Z_top - cz

    # Optionally center flattened, too
    if tip_in_top_flat is not None:
        cxf = 0.5 * (np.nanmin(X_top_flat) + np.nanmax(X_top_flat))
        cyf = 0.5 * (np.nanmin(Y_top_flat) + np.nanmax(Y_top_flat))
        czf = np.nanmedian(Z_top_flat)
        Xc_f, Yc_f, Zc_f = X_top_flat - cxf, Y_top_flat - cyf, Z_top_flat - czf

    # ---- DataFrame ----
    out = pd.DataFrame({
        "t": tt,
        # Unflattened (Box→Top mapping strictly holds here)
        "tip_x_top_m": X_top, "tip_y_top_m": Y_top, "tip_z_top_m": Z_top,
        "tip_x_top_centered_m": Xc, "tip_y_top_centered_m": Yc, "tip_z_top_centered_m": Zc,
        # Flattened (optional, for convenience/plots)
        **({} if tip_in_top_flat is None else {
            "tip_x_top_flat_m": X_top_flat, "tip_y_top_flat_m": Y_top_flat, "tip_z_top_flat_m": Z_top_flat,
            "tip_x_top_flat_centered_m": Xc_f, "tip_y_top_flat_centered_m": Yc_f, "tip_z_top_flat_centered_m": Zc_f,
        }),
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



    csv_path = os.path.join(out_dir, f"tip_path_top_frame_trial{test_num}.csv")
    out.to_csv(csv_path, index=False); print(f"Saved CSV: {csv_path}")

    '''# Sanity checks
    print("\nSanity check results:")
    det_median = np.median([np.linalg.det(R) for R in R_C_T])
    print(f"Median determinant of R_C_T: {det_median:.6f} (should be ~+1)")
    
    normal_dot_z = np.median(np.einsum('ni,i->n', R_C_T[:,:,2], np.array([0,0,1])))
    print(f"Median normal dot camera +Z: {normal_dot_z:.6f} (should be consistent sign)")'''
    
    print(f"\nTip position statistics (Top frame, {'flattened' if tip_in_top_flat is not None else 'unflattened'}):")
    if tip_in_top_flat is not None:
        print(f"X range: [{np.min(X_top_flat):.3f}, {np.max(X_top_flat):.3f}] m")
        print(f"Y range: [{np.min(Y_top_flat):.3f}, {np.max(Y_top_flat):.3f}] m")
        print(f"Z mean ± std: {np.mean(Z_top_flat):.3f} ± {np.std(Z_top_flat):.3f} m\n")
    else:
        print(f"X range: [{np.min(X_top):.3f}, {np.max(X_top):.3f}] m")
        print(f"Y range: [{np.min(Y_top):.3f}, {np.max(Y_top):.3f}] m")
        print(f"Z mean ± std: {np.mean(Z_top):.3f} ± {np.std(Z_top):.3f} m\n")

    # Choose which to plot: flattened if available, else unflattened
    Xp, Yp, Zp = (X_top_flat, Y_top_flat, Z_top_flat) if tip_in_top_flat is not None else (X_top, Y_top, Z_top)

    plt.figure(figsize=(8, 6))
    plt.plot(Xp*1e3, Yp*1e3, linewidth=1.2)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")  # First set equal aspect ratio
    plt.xlim([-25, 25])  # Then set X limits in mm
    plt.ylim([-10, 10])  # Then set Y limits in mm
    plt.xlabel("Top X [mm]"); plt.ylabel("Top Y [mm]")
    plt.title(f"Tip path on top surface ({'flattened' if tip_in_top_flat is not None else 'unflattened'}) — trial {test_num}")
    fig2d_nc_path = os.path.join(out_dir, f"tip_path_top_xy_{'flat' if tip_in_top_flat is not None else 'raw'}_trial{test_num}.png")
    plt.savefig(fig2d_nc_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {fig2d_nc_path}")

    fig_nc = plt.figure()
    ax_nc = fig_nc.add_subplot(111, projection="3d")
    ax_nc.plot(Xp*1e3, Yp*1e3, Zp*1e3, linewidth=1.0)
    ax_nc.set_xlabel("Top X [mm]"); ax_nc.set_ylabel("Top Y [mm]"); ax_nc.set_zlabel("Top Z [mm]")
    ax_nc.set_title(f"Tip path in Top frame ({'flattened' if tip_in_top_flat is not None else 'unflattened'}) — trial {test_num}")
    ax_nc.set_box_aspect([1, 1, 0.3])
    fig3d_nc_path = os.path.join(out_dir, f"tip_path_top_3d_{'flat' if tip_in_top_flat is not None else 'raw'}_trial{test_num}.png")
    plt.savefig(fig3d_nc_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved plot: {fig3d_nc_path}")

    # Plot coordinate systems in camera frame
    fig_coords = plt.figure(figsize=(10, 8))
    ax_coords = fig_coords.add_subplot(111, projection="3d")
    
    # Plot probe center trajectory
    ax_coords.plot(pP_C[:, 0]*1000, pP_C[:, 1]*1000, pP_C[:, 2]*1000, 'b-', label='Probe Path', alpha=0.5)
    
    # Plot box center trajectory
    ax_coords.plot(pB_C[:, 0]*1000, pB_C[:, 1]*1000, pB_C[:, 2]*1000, 'r-', label='Box Path', alpha=0.5)
    
    # Plot tip trajectory
    ax_coords.plot(tip_C[:, 0]*1000, tip_C[:, 1]*1000, tip_C[:, 2]*1000, 'g-', label='Tip Path', alpha=0.5)
    
    # Plot coordinate axes at a specific time point
    # Find a timestamp where all paths have minimum z-coordinate (typically contact point)
    z_min_idx = np.argmin(tip_C[:, 2])  # Use tip's lowest point as reference
    axis_length = 0.05 * 1000  # 5cm arrows in mm

    
    # Plot tip coordinate system (same orientation as probe, but at tip position)
    origin_tip = tip_C[z_min_idx] * 1000
    axes_info = [('r', 'Tip X'), ('g', 'Tip Y'), ('b', 'Tip Z')]
    for i, (color, label) in enumerate(axes_info):
        direction = R_C_P[z_min_idx, :, i] * axis_length
        ax_coords.quiver(origin_tip[0], origin_tip[1], origin_tip[2],
                        direction[0], direction[1], direction[2],
                        color=color, alpha=1.0, label=label)
    
    # Plot probe coordinate system
    origin_P = pP_C[z_min_idx] * 1000
    axes_info = [('r', 'Probe X'), ('g', 'Probe Y'), ('b', 'Probe Z')]
    for i, (color, label) in enumerate(axes_info):
        direction = R_C_P[z_min_idx, :, i] * axis_length
        ax_coords.quiver(origin_P[0], origin_P[1], origin_P[2],
                        direction[0], direction[1], direction[2],
                        color=color, alpha=0.6, label=label)
    
    # Plot box coordinate system
    origin_B = pB_C[z_min_idx] * 1000
    axes_info = [('r', 'Box X'), ('g', 'Box Y'), ('b', 'Box Z')]
    for i, (color, label) in enumerate(axes_info):
        direction = R_C_B[z_min_idx, :, i] * axis_length
        ax_coords.quiver(origin_B[0], origin_B[1], origin_B[2],
                        direction[0], direction[1], direction[2],
                        color=color, alpha=0.6, label=label)
    
    # Plot top coordinate system
    origin_T = top_origin_C[z_min_idx] * 1000
    axes_info = [('r', 'Top X'), ('g', 'Top Y'), ('b', 'Top Z')]
    for i, (color, label) in enumerate(axes_info):
        direction = R_C_T[z_min_idx, :, i] * axis_length
        ax_coords.quiver(origin_T[0], origin_T[1], origin_T[2],
                        direction[0], direction[1], direction[2],
                        color=color, alpha=0.6, label=label)
    
    ax_coords.set_xlabel('Camera X [mm]')
    ax_coords.set_ylabel('Camera Y [mm]')
    ax_coords.set_zlabel('Camera Z [mm]')
    ax_coords.legend()
    ax_coords.set_title('Probe and Box Coordinate Systems in Camera Frame')
    
    # Save the coordinate systems plot
    coords_path = os.path.join(out_dir, f"coordinate_systems_trial{test_num}.png")
    plt.savefig(coords_path, dpi=220, bbox_inches="tight")
    plt.show()
    print(f"Saved coordinate systems plot: {coords_path}")

    # Create combined processing file with tip positions and forces
    if ati_data is not None:
        # Merge ATI data with tip positions based on nearest timestamp
        # Use flattened coordinates if available, otherwise use unflattened
        if tip_in_top_flat is not None:
            x_pos = X_top_flat*1000  # Convert to mm
            y_pos = Y_top_flat*1000
        else:
            x_pos = X_top*1000  # Convert to mm
            y_pos = Y_top*1000
            
        # Create mask for points within the specified range
        mask = (x_pos >= -25) & (x_pos <= 25) & (y_pos >= -10) & (y_pos <= 10)
        tip_data = pd.DataFrame({
            'time': tt[mask],
            'x_position': x_pos[mask],
            'y_position': y_pos[mask]
        })
        
        # Ensure we only keep force and torque columns from ATI data
        force_torque_cols = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        ati_data_clean = ati_data[['time'] + force_torque_cols]
        
        # Sort both dataframes by time before merging
        tip_data = tip_data.sort_values('time')
        ati_data_clean = ati_data_clean.sort_values('time')
        
        processing_data = pd.merge_asof(
            tip_data,
            ati_data_clean,
            on='time',
            direction='nearest',
            tolerance=PAIR_TOL_S
        )
        
        # Double-check we don't have any duplicate columns
        processing_data = processing_data[['time', 'x_position', 'y_position'] + force_torque_cols]

        # Save to the new processing directory
        processing_path = os.path.join(processing_dir, f"processing_test_{test_num}.csv")
        processing_data.to_csv(processing_path, index=False)
        print(f"\nSaved processing data: {processing_path}")
        print("Processing file contains:")
        print(f"- {len(processing_data)} synchronized samples")
        print("- Columns: time, x_position, y_position, Fx, Fy, Fz, Tx, Ty, Tz")
    else:
        print("\nNo ATI data found - skipping processing file creation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tip path on box top with rotations + pivot calibration.")
    parser.add_argument("--atracsys", type=str, default=None, help="Path to atracsys_trialXX.txt/.csv")
    parser.add_argument("--test", type=int, default=test_num, help="Test number for output folder name")
    parser.add_argument("--no_plane_flatten", action="store_true", help="Disable plane flattening")
    parser.add_argument("--pivot_seconds", type=float, default=PIVOT_SECONDS, help="Seconds used for pivot")
    args = parser.parse_args()

    # allow quick toggles from CLI
    if args.no_plane_flatten: PLANE_FLATTEN = False
    PIVOT_SECONDS = args.pivot_seconds

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
