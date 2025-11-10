#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indenter tip path on the box top — rewritten to follow the wrench/contact
transformations used in FullForceTactile.py, while preserving:
- CLI args and tunables
- input/output paths and filenames
- CSV column names
- plots (2D/3D paths and coordinate systems)

Key differences vs the original:
- "Top" frame is bound to the Object frame (no 45 mm offset).
- "Tip" is the external contact point (marker-origin → contact offset),
  expressed in Object ("Top") and Camera ("world") frames.
- Forces/torques are transformed ATI → external marker → world (flip force)
  → object, like in FullForceTactile.py.

Refs: FullForceTactile.py (force_transformation & contact) — see notes.
"""

import os, sys, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Tuple
import argparse

# ------------------- User-tunable metadata (unchanged) -------------------
test_num = 101
version_num = 1
directory_to_datasets = os.path.abspath(
    fr"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\test data\test {test_num} - sensor v{version_num}"
)
if not os.path.exists(directory_to_datasets):
    os.makedirs(directory_to_datasets)
    print(f"Created dataset directory: {directory_to_datasets}")

ati_file = "ati_middle_trial{}.txt".format(test_num)
atracsys_file = "atracsys_trial{}.txt".format(test_num)

DATASET_DIRS: list[str] = [
    os.path.join(directory_to_datasets, ati_file),
    os.path.join(directory_to_datasets, atracsys_file),
]

print(f"\nLooking for files:")
print(f"ATI data file: {ati_file}")
print(f"Atracsys file: {atracsys_file}")
print(f"In directory: {directory_to_datasets}")

# ------------------- Tunables (kept) -------------------
PAIR_TOL_S = 0.020
PIVOT_SECONDS = 2.02           # unused now (tip comes from contact offset)
PLANE_FLATTEN = True

# ---- "Other-code" (FullForce) tunables ----
EXTERNAL_MOUNT = "bottom"      # 'bottom' | 'front' | 'side' (defaults from FullForce)
VERSION_TAG    = "2024-02"     # matches constants in FullForce
FLIP_FORCE_SIGN = True         # flip so force points into object
# Fixed rotation: ATI sensor frame -> External marker frame (per mount)
def Rz(deg): 
    th = np.deg2rad(deg); c,s = np.cos(th), np.sin(th)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
def Ry(deg):
    th = np.deg2rad(deg); c,s = np.cos(th), np.sin(th)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], float)
def Rx(deg):
    th = np.deg2rad(deg); c,s = np.cos(th), np.sin(th)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)

def get_R_marker_from_ATI(mount: str) -> np.ndarray:
    """
    Approximates FullForce's R_ati_to_atimarker_fingers for the external mounts.
    - bottom:  Rotation.from_euler('zy', [48, 90])  -> Rz(48) @ Ry(90)
    - front:   Rotation.from_euler('zy', [48, 180]) -> Rz(48) @ Ry(180)
    - side:    Rotation.from_euler('zyx',[48,180,-90]) -> Rz(48) @ Ry(180) @ Rx(-90)
    See: R_ati_to_atimarker_fingers[...] in FullForceTactile.py.
    """
    m = mount.lower()
    if m == "bottom":
        return Rz(48) @ Ry(90)
    elif m == "front":
        return Rz(48) @ Ry(180)
    elif m == "side":
        return Rz(48) @ Ry(180) @ Rx(-90)
    else:
        raise ValueError("EXTERNAL_MOUNT must be 'bottom'|'front'|'side'")

# Contact offset in the external marker frame (mm), VERSION_TAG='2024-02'
# marker_ori2pt = [86.48 + dx, 0.53 + dy, 28.57 + fiducial + dz]
FIDUCIAL_RADIUS_mm = 12.5/2
MOUNT_ERR = dict(bottom=(0.0, -1.0, 2.0), front=(-0.0, -1.0, 0.0), side=(0.0, 0.0, 0.0))
def get_marker_ori2pt_m(mount: str) -> np.ndarray:
    dx,dy,dz = MOUNT_ERR.get(mount.lower(), (0.0, -1.0, 2.0))
    if mount.lower()=="bottom":
        arr_mm = np.array([86.48 + dx, 0.53 + dy, 28.57 + FIDUCIAL_RADIUS_mm + dz], float)
    elif mount.lower()=="front":
        arr_mm = np.array([-(69-67), dy, -(54.95+20+FIDUCIAL_RADIUS_mm)], float)
    elif mount.lower()=="side":
        arr_mm = np.array([-(134.25-67), -41.59, 19+FIDUCIAL_RADIUS_mm], float)
    else:
        # default to bottom
        arr_mm = np.array([86.48 + dx, 0.53 + dy, 28.57 + FIDUCIAL_RADIUS_mm + dz], float)
    return arr_mm * 1e-3  # meters

# ------------------- IO helpers (kept; robust) -------------------
def load_ati_data(path: str) -> pd.DataFrame:
    """Load ATI F/T sensor data from text file."""
    try:
        df = pd.read_csv(path)
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
        }).astype(float)
        # Normalize time to seconds range
        t = df_processed['time'].astype(float)
        if t.abs().median()>1e12: df_processed['time']=t*1e-9
        elif t.abs().median()>1e9: df_processed['time']=t*1e-6
        elif t.abs().median()>1e6: df_processed['time']=t*1e-3
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
    mid = ([c for c in df.columns if re.search("marker_id", c, re.IGNORECASE)])[0]
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
    # time normalization
    t = df["t"].astype(float)
    if t.abs().median()>1e12: df["t"]=t*1e-9
    elif t.abs().median()>1e9: df["t"]=t*1e-6
    elif t.abs().median()>1e6: df["t"]=t*1e-3
    # mm → m; pack R
    df["px"] = df[pcols[0]].astype(float)*1e-3
    df["py"] = df[pcols[1]].astype(float)*1e-3
    df["pz"] = df[pcols[2]].astype(float)*1e-3
    R = df[rcols].to_numpy(float).reshape((-1,3,3))
    df["R"] = list(R)
    return df[["t","marker_id","px","py","pz","R"]]

def split_external_and_object(df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """Auto-detect moving 'external' vs steadier 'object' by motion variance (like probe/box)."""
    ids = df["marker_id"].unique()
    assert len(ids)>=2, "Need at least two marker_ids."
    stats=[]
    for k in ids:
        p=df[df["marker_id"]==k][["px","py","pz"]].to_numpy()
        stats.append((k, float(np.var(p,axis=0).sum())))
    stats.sort(key=lambda x: x[1])
    object_id, external_id = stats[0][0], stats[-1][0]
    ext   = df[df["marker_id"]==external_id].sort_values("t").reset_index(drop=True)
    obj   = df[df["marker_id"]==object_id ].sort_values("t").reset_index(drop=True)
    return ext, obj

def asof_join(a_df: pd.DataFrame, b_df: pd.DataFrame, tol_s: float, 
              a_cols=("t","px","py","pz","R"), b_cols=("t","px","py","pz","R")) -> pd.DataFrame:
    a = a_df[list(a_cols)].copy(); a.columns=["t","ax","ay","az","AR"]
    b = b_df[list(b_cols)].copy(); b.columns=["t_b","bx","by","bz","BR"]
    out = pd.merge_asof(a.sort_values("t"), b.sort_values("t_b"),
                        left_on="t", right_on="t_b",
                        direction="nearest", tolerance=tol_s).dropna(subset=["t_b"])
    return out.reset_index(drop=True)

# ------------------- Homogeneous transforms (kept) -------------------
def make_homogeneous_matrices(Rs: np.ndarray, ts: np.ndarray) -> np.ndarray:
    Rs = np.asarray(Rs); ts = np.asarray(ts)
    if Rs.ndim==2: Rs = Rs[np.newaxis,...]
    if ts.ndim==1: ts = ts[np.newaxis,...]
    if Rs.shape[0]!=ts.shape[0]: raise ValueError("Rs and ts length mismatch")
    N = Rs.shape[0]
    Ts = np.zeros((N,4,4), dtype=Rs.dtype)
    Ts[:, :3, :3] = Rs
    Ts[:, :3, 3]  = ts
    Ts[:, 3, 3]   = 1.0
    return Ts

def invert_homogeneous_matrices(Ts: np.ndarray) -> np.ndarray:
    Ts = np.asarray(Ts)
    if Ts.ndim==2: Ts = Ts[np.newaxis,...]
    R = Ts[:, :3, :3]; t = Ts[:, :3, 3]
    Rinv = np.transpose(R, (0,2,1))
    tinv = -np.einsum("nij,nj->ni", Rinv, t)
    out = np.zeros_like(Ts)
    out[:, :3, :3] = Rinv
    out[:, :3, 3]  = tinv
    out[:, 3, 3]   = 1.0
    return out

def apply_homogeneous_matrices(Ts: np.ndarray, points: np.ndarray) -> np.ndarray:
    Ts = np.asarray(Ts); points = np.asarray(points)
    if Ts.ndim==2: Ts = Ts[np.newaxis,...]
    N = Ts.shape[0]
    if points.ndim==1:
        p_h = np.concatenate([points,[1.0]])
        res_h = Ts @ p_h
        return res_h[:, :3]
    elif points.ndim==2:
        if points.shape[0]!=N: raise ValueError("points length != transforms")
        p_h = np.concatenate([points, np.ones((N,1))], axis=1)
        res = np.einsum("nij,nj->ni", Ts, p_h)
        return res[:, :3]
    else:
        raise ValueError("points must be (3,) or (N,3)")

# ------------------- Plane fit (kept) -------------------
def best_fit_plane(points: np.ndarray):
    ctr = points.mean(axis=0)
    P = points - ctr
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    n = vh[-1,:]; n /= np.linalg.norm(n)
    z = np.array([0.,0.,1.]); v = np.cross(z,n); s = np.linalg.norm(v); c = float(np.dot(z,n))
    if s<1e-8:
        R = np.eye(3) if c>0 else np.diag([1,1,-1])
    else:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + vx@vx * ((1-c)/(s**2))
    return ctr, n, R

# ------------------- Wrench/contact per FullForce -------------------
def transform_wrench_to_object(F_ati: np.ndarray, T_ati: np.ndarray,
                               R_C_ext: np.ndarray, R_C_obj: np.ndarray,
                               R_ext_ATI: np.ndarray, flip_force: bool=True) -> Tuple[np.ndarray,np.ndarray]:
    """
    F_ati, T_ati             : (N,3) in ATI frame
    R_C_ext, R_C_obj         : (N,3,3) Camera(world)←External/Obj rotation matrices
    R_ext_ATI                : (3,3) External marker ← ATI
    Returns F_obj, T_obj     : (N,3) in Object frame
    """
    # world ← ATI : R_C_ext @ R_ext_ATI
    R_world_ATI = np.einsum("nij,jk->nik", R_C_ext, R_ext_ATI)
    F_world = np.einsum("nij,nj->ni", R_world_ATI, F_ati)
    T_world = np.einsum("nij,nj->ni", R_world_ATI, T_ati)
    if flip_force: F_world = -F_world
    # object ← world : R_obj^T
    R_obj_T = np.transpose(R_C_obj, (0,2,1))
    F_obj = np.einsum("nij,nj->ni", R_obj_T, F_world)
    T_obj = np.einsum("nij,nj->ni", R_obj_T, T_world)
    return F_obj, T_obj

def compute_contact_positions(R_C_ext: np.ndarray, p_ext_C: np.ndarray,
                              R_C_obj: np.ndarray, p_obj_C: np.ndarray,
                              marker_ori2pt_m: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """
    Returns:
      tip_C   : (N,3) contact in Camera/world
      tip_obj : (N,3) contact in Object frame
    """
    # world contact
    offset_world = np.einsum("nij,j->ni", R_C_ext, marker_ori2pt_m)
    tip_C = p_ext_C + offset_world
    # object frame
    R_obj_T = np.transpose(R_C_obj, (0,2,1))
    tip_obj = np.einsum("nij,nj->ni", R_obj_T, (tip_C - p_obj_C))
    return tip_C, tip_obj

# ------------------- Main -------------------
def main(atracsys_path: str, test_num: int):
    base_dir = os.path.dirname(os.path.abspath(atracsys_path))
    out_dir = os.path.join(base_dir, f"lin{test_num}")
    os.makedirs(out_dir, exist_ok=True)

    processing_dir = os.path.abspath(
        fr"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\test data\test {test_num} - sensor v{version_num}"
    )
    os.makedirs(processing_dir, exist_ok=True)

    # Load ATI (raw, seconds)
    ati_path = os.path.join(base_dir, f"ati_middle_trial{test_num}.txt")
    ati_data = load_ati_data(ati_path) if os.path.exists(ati_path) else None

    # Load Atracsys
    df = load_atracsys_csv(atracsys_path)
    external, obj = split_external_and_object(df)

    # Pair external↔object
    F = asof_join(external, obj, PAIR_TOL_S)
    tt = F["t"].to_numpy()
    pExt_C = F[["ax","ay","az"]].to_numpy()
    pObj_C = F[["bx","by","bz"]].to_numpy()
    R_C_ext = np.stack(F["AR"].to_numpy())
    R_C_obj = np.stack(F["BR"].to_numpy())

    # --- Contact point (this is your "tip") ---
    R_ext_ATI = get_R_marker_from_ATI(EXTERNAL_MOUNT)
    marker_ori2pt = get_marker_ori2pt_m(EXTERNAL_MOUNT)
    tip_C, tip_in_obj = compute_contact_positions(R_C_ext, pExt_C, R_C_obj, pObj_C, marker_ori2pt)

    # Bind "Top" to Object to preserve your outputs/plots
    R_C_T = R_C_obj.copy()
    top_origin_C = pObj_C.copy()
    T_C_Top = make_homogeneous_matrices(R_C_T, top_origin_C)
    T_Top_C = invert_homogeneous_matrices(T_C_Top)
    tip_in_top = tip_in_obj  # since Top ≡ Object

    # Optional flatten (points only)
    tip_in_top_flat = None
    if PLANE_FLATTEN:
        ctr, n, R_flat = best_fit_plane(tip_in_top)
        tip_in_top_flat = (tip_in_top - ctr) @ R_flat + ctr

    # Un/flattened split
    X_top, Y_top, Z_top = tip_in_top[:,0], tip_in_top[:,1], tip_in_top[:,2]
    if tip_in_top_flat is not None:
        X_top_flat, Y_top_flat, Z_top_flat = tip_in_top_flat[:,0], tip_in_top_flat[:,1], tip_in_top_flat[:,2]
    else:
        X_top_flat = Y_top_flat = Z_top_flat = None

    # Center (unflattened)
    cx = 0.5*(np.nanmin(X_top)+np.nanmax(X_top))
    cy = 0.5*(np.nanmin(Y_top)+np.nanmax(Y_top))
    cz = np.nanmedian(Z_top)
    Xc, Yc, Zc = X_top-cx, Y_top-cy, Z_top-cz

    # Center (flattened)
    if tip_in_top_flat is not None:
        cxf = 0.5*(np.nanmin(X_top_flat)+np.nanmax(X_top_flat))
        cyf = 0.5*(np.nanmin(Y_top_flat)+np.nanmax(Y_top_flat))
        czf = np.nanmedian(Z_top_flat)
        Xc_f, Yc_f, Zc_f = X_top_flat-cxf, Y_top_flat-cyf, Z_top_flat-czf

    # ---- CSV (tip path) ----
    out_df = pd.DataFrame({
        "t": tt,
        # Object frame (named "Top" to keep your schema)
        "tip_x_top_m": X_top, "tip_y_top_m": Y_top, "tip_z_top_m": Z_top,
        "tip_x_top_centered_m": Xc, "tip_y_top_centered_m": Yc, "tip_z_top_centered_m": Zc,
        **({} if tip_in_top_flat is None else {
            "tip_x_top_flat_m": X_top_flat, "tip_y_top_flat_m": Y_top_flat, "tip_z_top_flat_m": Z_top_flat,
            "tip_x_top_flat_centered_m": Xc_f, "tip_y_top_flat_centered_m": Yc_f, "tip_z_top_flat_centered_m": Zc_f,
        }),
        # Camera/world contact for traceability
        "tip_x_cam_m": tip_C[:,0], "tip_y_cam_m": tip_C[:,1], "tip_z_cam_m": tip_C[:,2],
        # Legacy calibration/debug columns (kept, not used)
        "p_tip_probe_x_m": [0.0]*len(tt), "p_tip_probe_y_m": [0.0]*len(tt), "p_tip_probe_z_m": [0.0]*len(tt),
        "top_center_shift_m_x": [cx]*len(tt), "top_center_shift_m_y": [cy]*len(tt), "top_center_shift_m_z": [cz]*len(tt),
    })
    csv_path = os.path.join(out_dir, f"tip_path_top_frame_trial{test_num} - Zero version.csv")
    out_df.to_csv(csv_path, index=False); print(f"Saved CSV: {csv_path}")

    # ---- Sanity checks (determinant, etc.) ----
    det_median = np.median([np.linalg.det(R) for R in R_C_T])
    print(f"Median det(R_C_T): {det_median:.6f} (should be ~+1)")
    normal_dot_z = np.median(np.einsum('ni,i->n', R_C_T[:,:,2], np.array([0,0,1])))
    print(f"Median normal·camera+Z: {normal_dot_z:.6f}")
    print(f"\nContact (Top/Object) stats:")
    print(f"X range: [{np.min(X_top):.3f}, {np.max(X_top):.3f}] m")
    print(f"Y range: [{np.min(Y_top):.3f}, {np.max(Y_top):.3f}] m")
    print(f"Z mean ± std: {np.mean(Z_top):.3f} ± {np.std(Z_top):.3f} m\n")

    # ---- Plots (unchanged look/paths) ----
    Xp, Yp, Zp = (X_top_flat, Y_top_flat, Z_top_flat) if tip_in_top_flat is not None else (X_top, Y_top, Z_top)
    plt.figure()
    plt.plot(Xp*1e3, Yp*1e3, linewidth=1.2)
    plt.axis("equal"); plt.grid(True, alpha=0.3)
    plt.xlabel("Top X [mm]"); plt.ylabel("Top Y [mm]")
    plt.title(f"Tip path on top surface ({'flattened' if tip_in_top_flat is not None else 'unflattened'}) — trial {test_num} - Zero version")
    fig2d_nc_path = os.path.join(out_dir, f"tip_path_top_xy_{'flat' if tip_in_top_flat is not None else 'raw'}_trial{test_num} - Zero version.png")
    plt.savefig(fig2d_nc_path, dpi=220, bbox_inches="tight"); plt.show()
    print(f"Saved plot: {fig2d_nc_path}")

    fig_nc = plt.figure()
    ax_nc = fig_nc.add_subplot(111, projection="3d")
    ax_nc.plot(Xp*1e3, Yp*1e3, Zp*1e3, linewidth=1.0)
    ax_nc.set_xlabel("Top X [mm]"); ax_nc.set_ylabel("Top Y [mm]"); ax_nc.set_zlabel("Top Z [mm]")
    ax_nc.set_title(f"Tip path in Top frame ({'flattened' if tip_in_top_flat is not None else 'unflattened'}) — trial {test_num} - Zero version")
    ax_nc.set_box_aspect([1,1,0.3])
    fig3d_nc_path = os.path.join(out_dir, f"tip_path_top_3d_{'flat' if tip_in_top_flat is not None else 'raw'}_trial{test_num} - Zero version.png")
    plt.savefig(fig3d_nc_path, dpi=220, bbox_inches="tight"); plt.show()
    print(f"Saved plot: {fig3d_nc_path}")

    # Coordinate systems in camera
    fig_coords = plt.figure(figsize=(10, 8))
    ax_coords = fig_coords.add_subplot(111, projection="3d")
    ax_coords.plot(external["px"].to_numpy()*1e3, external["py"].to_numpy()*1e3, external["pz"].to_numpy()*1e3, 'b-', label='External Path', alpha=0.5)
    ax_coords.plot(obj["px"].to_numpy()*1e3, obj["py"].to_numpy()*1e3, obj["pz"].to_numpy()*1e3, 'r-', label='Object Path', alpha=0.5)
    ax_coords.plot(tip_C[:,0]*1e3, tip_C[:,1]*1e3, tip_C[:,2]*1e3, 'g-', label='Contact Path', alpha=0.5)
    z_min_idx = np.argmin(tip_C[:,2]); axis_length = 0.05*1e3
    print(f"\nAxes at index {z_min_idx} (t={tt[z_min_idx]:.3f}s)")
    origin_tip = tip_C[z_min_idx]*1e3
    for i,(c,lbl) in enumerate([('r','Tip X'),('g','Tip Y'),('b','Tip Z')]):
        direction = R_C_ext[z_min_idx,:,i]*axis_length
        ax_coords.quiver(origin_tip[0],origin_tip[1],origin_tip[2], direction[0],direction[1],direction[2], color=c, alpha=1.0, label=lbl)
    origin_E = pExt_C[z_min_idx]*1e3
    for i,(c,lbl) in enumerate([('r','Ext X'),('g','Ext Y'),('b','Ext Z')]):
        direction = R_C_ext[z_min_idx,:,i]*axis_length
        ax_coords.quiver(origin_E[0],origin_E[1],origin_E[2], direction[0],direction[1],direction[2], color=c, alpha=0.6, label=lbl)
    origin_O = pObj_C[z_min_idx]*1e3
    for i,(c,lbl) in enumerate([('r','Obj X'),('g','Obj Y'),('b','Obj Z')]):
        direction = R_C_obj[z_min_idx,:,i]*axis_length
        ax_coords.quiver(origin_O[0],origin_O[1],origin_O[2], direction[0],direction[1],direction[2], color=c, alpha=0.6, label=lbl)
    ax_coords.set_xlabel('Camera X [mm]'); ax_coords.set_ylabel('Camera Y [mm]'); ax_coords.set_zlabel('Camera Z [mm]')
    ax_coords.legend(); ax_coords.set_title('External/Object Coordinate Systems in Camera Frame')
    coords_path = os.path.join(out_dir, f"coordinate_systems_trial{test_num} - Zero version.png")
    plt.savefig(coords_path, dpi=220, bbox_inches="tight"); plt.show()
    print(f"Saved coordinate systems plot: {coords_path}")

    # ---- Processing CSV: centered XY + transformed wrenches (object frame) ----
    if ati_data is not None:
        # Pair ATI samples to nearest external/object poses to compute object-frame wrenches at ATI times
        ext_asof = pd.merge_asof(ati_data[["time"]].sort_values("time"),
                                 external[["t","px","py","pz","R"]].rename(columns={"t":"t_ext"}),
                                 left_on="time", right_on="t_ext", direction="nearest", tolerance=PAIR_TOL_S)
        obj_asof = pd.merge_asof(ati_data[["time"]].sort_values("time"),
                                 obj[["t","px","py","pz","R"]].rename(columns={"t":"t_obj"}),
                                 left_on="time", right_on="t_obj", direction="nearest", tolerance=PAIR_TOL_S)
        valid = ext_asof["t_ext"].notna() & obj_asof["t_obj"].notna()
        if valid.any():
            R_C_ext_ati = np.stack(ext_asof.loc[valid,"R"].to_numpy())
            R_C_obj_ati = np.stack(obj_asof.loc[valid,"R"].to_numpy())
            F_ati = ati_data.loc[valid,["Fx","Fy","Fz"]].to_numpy(float)
            T_ati = ati_data.loc[valid,["Tx","Ty","Tz"]].to_numpy(float)
            F_obj, T_obj = transform_wrench_to_object(F_ati, T_ati, R_C_ext_ati, R_C_obj_ati, R_ext_ATI, flip_force=FLIP_FORCE_SIGN)
            ati_obj = ati_data.loc[valid].copy()
            ati_obj[["Fx","Fy","Fz"]] = F_obj
            ati_obj[["Tx","Ty","Tz"]] = T_obj
        else:
            ati_obj = ati_data.copy()
            print("Warning: No valid pose pairing for ATI times; leaving forces in sensor frame.")

        # Keep your centered XY at the time base of tip points, then asof-merge onto ATI time
        tip_data = pd.DataFrame({'time': tt, 'x_position': Xc, 'y_position': Yc}).sort_values('time')
        processing_data = pd.merge_asof(
            ati_obj.sort_values('time'),
            tip_data,
            on='time',
            direction='nearest',
            tolerance=PAIR_TOL_S
        )[["time","x_position","y_position","Fx","Fy","Fz","Tx","Ty","Tz"]]

        processing_path = os.path.join(processing_dir, f"processing_test_{test_num} - Zero version.csv")
        processing_data.to_csv(processing_path, index=False)
        print(f"\nSaved processing data: {processing_path}")
        print("Processing file contains:")
        print(f"- {len(processing_data)} synchronized samples")
        print("- Columns: time, x_position, y_position, Fx, Fy, Fz, Tx, Ty, Tz")
    else:
        print("\nNo ATI data found - skipping processing file creation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tip/contact path with object-frame wrenches (FullForce-style).")
    parser.add_argument("--atracsys", type=str, default=None, help="Path to atracsys_trialXX.txt/.csv")
    parser.add_argument("--test", type=int, default=test_num, help="Test number for output folder name")
    parser.add_argument("--no_plane_flatten", action="store_true", help="Disable plane flattening")
    parser.add_argument("--pivot_seconds", type=float, default=PIVOT_SECONDS, help="(Unused) kept for API compat")
    args = parser.parse_args()

    if args.no_plane_flatten: PLANE_FLATTEN = False
    PIVOT_SECONDS = args.pivot_seconds  # kept for CLI compatibility

    atr_path = args.atracsys or (DATASET_DIRS[1] if len(DATASET_DIRS)>1 else None)
    print(f"\nChecking paths:")
    print(f"Directory to datasets: {directory_to_datasets}")
    print(f"ATI data path: {DATASET_DIRS[0]}")
    print(f"Atracsys path: {atr_path}")

    if not os.path.exists(directory_to_datasets):
        sys.exit(f"Dataset directory not found: {directory_to_datasets}")
    if not atr_path or not os.path.exists(atr_path):
        sys.exit(f"Atracsys file not found: {atr_path}\nPlease check if the file exists or pass --atracsys with the correct path.")
    main(atr_path, args.test)
