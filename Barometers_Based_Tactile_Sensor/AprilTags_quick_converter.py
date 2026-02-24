import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===================== EDIT THESE IN VS CODE =====================
TRIAL = 51092
DATA_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test {} - sensor v5".format(TRIAL))

TAG0_FILE = DATA_DIR / f"{TRIAL}tag0_pose_trial.txt"
TAG1_FILE = DATA_DIR / f"{TRIAL}tag1_pose_trial.txt"
TAG2_FILE = DATA_DIR / f"{TRIAL}tag2_pose_trial.txt"

# Export for your existing Atracsys-based pipeline (ONLY tag1+tag2)
OUT_ATRACSYS_LIKE = DATA_DIR / f"atracsys_trial{TRIAL}.txt"
# ================================================================

PAIR_TOL_S = 0.020


def time_to_seconds(t_raw: np.ndarray) -> np.ndarray:
    """Heuristic like your Atracsys loader: ns/us/ms -> seconds."""
    t = np.asarray(t_raw, dtype=np.float64)
    med = np.nanmedian(np.abs(t))
    if med > 1e12:   # nanoseconds
        return t * 1e-9
    elif med > 1e9:  # microseconds
        return t * 1e-6
    elif med > 1e6:  # milliseconds
        return t * 1e-3 
    else:
        return t


def sampling_rate_hz(t_s: np.ndarray) -> tuple[float, float]:
    """Return (Hz_median, dt_median_s)."""
    dt = np.diff(t_s)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if len(dt) == 0:
        return float("nan"), float("nan")
    dt_med = float(np.median(dt))
    return float(1.0 / dt_med), dt_med


def quat_xyzw_to_rotm(qx, qy, qz, qw) -> np.ndarray:
    """
    Vectorized quaternion (x,y,z,w) -> rotation matrix (N,3,3).
    Assumes quaternion expresses tag frame orientation in the camera frame
    (typical TF convention: R_camera_tag maps tag vectors into camera).
    """
    x = np.asarray(qx, dtype=np.float64)
    y = np.asarray(qy, dtype=np.float64)
    z = np.asarray(qz, dtype=np.float64)
    w = np.asarray(qw, dtype=np.float64)

    n = np.sqrt(x*x + y*y + z*z + w*w)
    n[n == 0] = 1.0
    x /= n; y /= n; z /= n; w /= n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    xw, yw, zw = x*w, y*w, z*w

    R = np.empty((len(x), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2*(yy + zz)
    R[:, 0, 1] = 2*(xy - zw)
    R[:, 0, 2] = 2*(xz + yw)

    R[:, 1, 0] = 2*(xy + zw)
    R[:, 1, 1] = 1 - 2*(xx + zz)
    R[:, 1, 2] = 2*(yz - xw)

    R[:, 2, 0] = 2*(xz - yw)
    R[:, 2, 1] = 2*(yz + xw)
    R[:, 2, 2] = 1 - 2*(xx + yy)
    return R


def load_tag_pose(path: Path, tag_id: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    t_ns = df["%time"].astype(np.int64).to_numpy()
    t_s = time_to_seconds(t_ns)

    px = df["field.pose.position.x"].astype(np.float64).to_numpy()
    py = df["field.pose.position.y"].astype(np.float64).to_numpy()
    pz = df["field.pose.position.z"].astype(np.float64).to_numpy()

    qx = df["field.pose.orientation.x"].astype(np.float64).to_numpy()
    qy = df["field.pose.orientation.y"].astype(np.float64).to_numpy()
    qz = df["field.pose.orientation.z"].astype(np.float64).to_numpy()
    qw = df["field.pose.orientation.w"].astype(np.float64).to_numpy()

    R = quat_xyzw_to_rotm(qx, qy, qz, qw)  # (N,3,3)

    out = pd.DataFrame({
        "t": t_s,
        "t_ns": t_ns,
        "marker_id": int(tag_id),
        "px_m": px,
        "py_m": py,
        "pz_m": pz,
    })
    out["R"] = list(R)
    return out


def export_atracsys_like(df_tags: pd.DataFrame, out_path: Path) -> None:
    """
    Make a CSV your existing load_atracsys_data() can read:
      %time, marker_id, field.position0..2 (mm), field.rotation0..8
    """
    df_tags = df_tags.sort_values("t_ns").reset_index(drop=True)
    Rs = np.stack(df_tags["R"].to_numpy())  # (N,3,3)
    R_flat = Rs.reshape((-1, 9))           # row-major

    export = pd.DataFrame({
        "%time": df_tags["t_ns"].to_numpy(),
        "marker_id": df_tags["marker_id"].to_numpy().astype(int),
        # Your loader multiplies by 1e-3, so we store mm here:
        "field.position0": df_tags["px_m"].to_numpy() * 1e3,
        "field.position1": df_tags["py_m"].to_numpy() * 1e3,
        "field.position2": df_tags["pz_m"].to_numpy() * 1e3,
    })
    for i in range(9):
        export[f"field.rotation{i}"] = R_flat[:, i]

    export.to_csv(out_path, index=False)
    print(f"[OK] Wrote Atracsys-like CSV: {out_path}")


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    r = 0.5 * max(x_range, y_range, z_range)
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - r, x_mid + r])
    ax.set_ylim3d([y_mid - r, y_mid + r])
    ax.set_zlim3d([z_mid - r, z_mid + r])


def plot_tags_3d(tag_dfs: dict[str, pd.DataFrame], units: str = "mm"):
    scale = 1000.0 if units == "mm" else 1.0

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for name, df in tag_dfs.items():
        x = df["px_m"].to_numpy() * scale
        y = df["py_m"].to_numpy() * scale
        z = df["pz_m"].to_numpy() * scale
        ax.plot(x, y, z, label=name)
        # mark start/end
        ax.scatter(x[0], y[0], z[0], marker="o")
        ax.scatter(x[-1], y[-1], z[-1], marker="x")
        
        # Plot coordinate axes at one sample point (middle of trajectory)
        mid_idx = len(df) // 2
        origin = np.array([x[mid_idx], y[mid_idx], z[mid_idx]])
        R = df["R"].iloc[mid_idx]  # (3,3) rotation matrix
        
        # Axis length in plot units
        axis_len = 20.0 if units == "mm" else 0.02
        
        # Plot X, Y, Z axes
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']
        for i, (color, label) in enumerate(zip(colors, labels)):
            axis_dir = R[:, i] * axis_len  # i-th column of R
            ax.quiver(origin[0], origin[1], origin[2],
                     axis_dir[0], axis_dir[1], axis_dir[2],
                     color=color, arrow_length_ratio=0.2, linewidth=2,
                     label=f"{name} {label}")

    ax.set_xlabel(f"Camera X [{units}]")
    ax.set_ylabel(f"Camera Y [{units}]")
    ax.set_zlabel(f"Camera Z [{units}]")
    ax.set_title("3D tag trajectories (camera frame)")
    ax.legend()
    set_axes_equal(ax)
    plt.tight_layout()
    plt.close()


def plot_probe_2d_projections(probe_df: pd.DataFrame, units: str = "mm"):
    """Plot 2D projections of probe path in camera frame (XY, XZ, YZ planes)."""
    scale = 1000.0 if units == "mm" else 1.0
    
    # Get probe positions in camera frame - these are already in a fixed frame!
    px_cam = probe_df["px_m"].to_numpy() * scale
    py_cam = probe_df["py_m"].to_numpy() * scale
    pz_cam = probe_df["pz_m"].to_numpy() * scale
    
    # Make relative to first position (origin at start)
    pos_in_cam = np.column_stack([
        px_cam - px_cam[0],
        py_cam - py_cam[0],
        pz_cam - pz_cam[0]
    ])
    
    # Create 2D projection plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # XY plane
    axes[0].plot(pos_in_cam[:, 0], pos_in_cam[:, 1], 'b-', linewidth=1.5)
    axes[0].scatter(pos_in_cam[0, 0], pos_in_cam[0, 1], c='g', s=50, marker='o', label='Start')
    axes[0].scatter(pos_in_cam[-1, 0], pos_in_cam[-1, 1], c='r', s=50, marker='x', label='End')
    axes[0].set_xlabel(f'Camera X [{units}]')
    axes[0].set_ylabel(f'Camera Y [{units}]')
    axes[0].set_title('XY Projection (Camera Frame)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    axes[0].legend()
    
    # XZ plane
    axes[1].plot(pos_in_cam[:, 0], pos_in_cam[:, 2], 'b-', linewidth=1.5)
    axes[1].scatter(pos_in_cam[0, 0], pos_in_cam[0, 2], c='g', s=50, marker='o', label='Start')
    axes[1].scatter(pos_in_cam[-1, 0], pos_in_cam[-1, 2], c='r', s=50, marker='x', label='End')
    axes[1].set_xlabel(f'Camera X [{units}]')
    axes[1].set_ylabel(f'Camera Z [{units}]')
    axes[1].set_title('XZ Projection (Camera Frame)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    axes[1].legend()
    
    # YZ plane
    axes[2].plot(pos_in_cam[:, 1], pos_in_cam[:, 2], 'b-', linewidth=1.5)
    axes[2].scatter(pos_in_cam[0, 1], pos_in_cam[0, 2], c='g', s=50, marker='o', label='Start')
    axes[2].scatter(pos_in_cam[-1, 1], pos_in_cam[-1, 2], c='r', s=50, marker='x', label='End')
    axes[2].set_xlabel(f'Camera Y [{units}]')
    axes[2].set_ylabel(f'Camera Z [{units}]')
    axes[2].set_title('YZ Projection (Camera Frame)')
    axes[2].grid(True, alpha=0.3)
    axes[2].axis('equal')
    axes[2].legend()
    
    plt.tight_layout()
    plt.close()



if __name__ == "__main__":
    tag0 = load_tag_pose(TAG0_FILE, 0)
    tag1 = load_tag_pose(TAG1_FILE, 1)
    tag2 = load_tag_pose(TAG2_FILE, 2)

    # ---- Sampling rates ----
    for name, df in [("tag0", tag0), ("tag1", tag1), ("tag2", tag2)]:
        hz, dt = sampling_rate_hz(df["t"].to_numpy())
        print(f"{name}: {len(df)} samples | median dt={dt:.4f}s | ~{hz:.2f} Hz")

    # ---- 3D plot of all 3 tags ----
    plot_tags_3d({"tag0": tag0, "tag1": tag1, "tag2": tag2}, units="mm")
    
    # ---- 2D projections of probe path in probe reference frame ----
    plot_probe_2d_projections(tag1, units="mm")  # tag1 is the probe

    # ---- Export ONLY tag1+tag2 for your existing pipeline ----
    export_atracsys_like(pd.concat([tag1, tag2], ignore_index=True), OUT_ATRACSYS_LIKE)