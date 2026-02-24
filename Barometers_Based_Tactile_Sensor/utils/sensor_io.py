#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor I/O utilities shared across tracking/wrench analysis scripts.

Functions:
- load_ati_data:       Load ATI F/T sensor data from CSV/TXT
- load_atracsys_data:  Load Atracsys rigid-body tracking data
- asof_join:           Nearest-timestamp join of probe and box dataframes
"""

import re
import numpy as np
import pandas as pd


def _to_seconds(series: pd.Series) -> np.ndarray:
    """Convert a time-like numeric Series to seconds (handles ns, µs, ms, s)."""
    t = series.astype(float).to_numpy()
    med = np.median(np.abs(t[np.isfinite(t)])) if np.any(np.isfinite(t)) else 0
    if med > 1e12:
        return t * 1e-9   # ns -> s
    if med > 1e9:
        return t * 1e-6   # µs -> s
    if med > 1e6:
        return t * 1e-3   # ms -> s
    return t              # already seconds (includes Unix epoch ~1.7e9)


def load_ati_data(path: str) -> pd.DataFrame:
    """
    Load ATI F/T sensor data from a CSV/TXT file.

    Detects time, force, and torque columns by name (case-insensitive).
    Converts the time column to seconds automatically.

    Args:
        path: Path to the ATI data file.

    Returns:
        DataFrame with columns: time, Fx, Fy, Fz, Tx, Ty, Tz (all float).
    """
    df = pd.read_csv(path)
    time_col   = [c for c in df.columns if 'time'   in c.lower()][0]
    force_cols  = [c for c in df.columns if 'force'  in c.lower()]
    torque_cols = [c for c in df.columns if 'torque' in c.lower()]

    out = pd.DataFrame({
        "time": df[time_col].astype(float),
        "Fx":   df[force_cols[0]].astype(float),
        "Fy":   df[force_cols[1]].astype(float),
        "Fz":   df[force_cols[2]].astype(float),
        "Tx":   df[torque_cols[0]].astype(float),
        "Ty":   df[torque_cols[1]].astype(float),
        "Tz":   df[torque_cols[2]].astype(float),
    })
    out["time"] = _to_seconds(out["time"])
    return out


def load_atracsys_data(path: str) -> pd.DataFrame:
    """
    Load Atracsys rigid-body tracking data from a CSV/TXT file.

    Handles multiple column-name conventions (ROS bag exports and direct exports)
    using regex-based detection. Converts positions from mm to metres and
    time to seconds.

    Args:
        path: Path to the Atracsys data file.

    Returns:
        DataFrame with columns: t, marker_id, px, py, pz (metres), R (list of 3×3 arrays).
    """
    df = pd.read_csv(path, engine="python")
    df.columns = [c.strip() for c in df.columns]

    def pick(regex: str) -> list:
        rx = re.compile(regex, re.IGNORECASE)
        cols = [c for c in df.columns if rx.fullmatch(c)]
        return cols if cols else [c for c in df.columns if re.search(regex, c, re.IGNORECASE)]

    tcol = (pick(r"(?:%?time|field\.timestamp)") or ["t"])[0]
    mid  = [c for c in df.columns if re.search("marker_id", c, re.IGNORECASE)][0]

    pcols = [f"field.position{i}" for i in range(3)]
    if not all(c in df.columns for c in pcols):
        pcols = sorted(
            [c for c in df.columns if re.search(r"position[0-2]$", c)],
            key=lambda s: int(re.search(r"(\d)$", s).group(1))
        )
        assert len(pcols) == 3, "position0..2 not found"

    rcols = [f"field.rotation{i}" for i in range(9)]
    if not all(c in df.columns for c in rcols):
        rcols = sorted(
            [c for c in df.columns if re.search(r"rotation[0-8]$", c)],
            key=lambda s: int(re.search(r"(\d)$", s).group(1))
        )
        assert len(rcols) == 9, "rotation0..8 not found"

    df = df[[tcol, mid] + pcols + rcols].rename(columns={tcol: "t", mid: "marker_id"}).copy()

    df["t"]  = _to_seconds(df["t"])
    df["px"] = df[pcols[0]].astype(float) * 1e-3
    df["py"] = df[pcols[1]].astype(float) * 1e-3
    df["pz"] = df[pcols[2]].astype(float) * 1e-3
    df["R"]  = list(df[rcols].to_numpy(float).reshape((-1, 3, 3)))

    return df[["t", "marker_id", "px", "py", "pz", "R"]]


def asof_join(probe: pd.DataFrame, box: pd.DataFrame, tol_s: float) -> pd.DataFrame:
    """
    Nearest-timestamp join of probe and box rigid-body dataframes.

    Args:
        probe: Probe rigid-body DataFrame (columns: t, px, py, pz, R).
        box:   Box rigid-body DataFrame   (columns: t, px, py, pz, R).
        tol_s: Maximum allowed time difference in seconds.

    Returns:
        Merged DataFrame with probe columns (t, ppx, ppy, ppz, PR) and
        box columns (t_box, bpx, bpy, bpz, BR).
    """
    a = probe[["t", "px", "py", "pz", "R"]].copy()
    a.columns = ["t", "ppx", "ppy", "ppz", "PR"]
    b = box[["t", "px", "py", "pz", "R"]].copy()
    b.columns = ["t_box", "bpx", "bpy", "bpz", "BR"]
    out = pd.merge_asof(
        a.sort_values("t"), b.sort_values("t_box"),
        left_on="t", right_on="t_box",
        direction="nearest", tolerance=tol_s
    ).dropna(subset=["t_box"])
    return out.reset_index(drop=True)
