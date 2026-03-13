#!/usr/bin/env python3
"""
Train LightGBM force models using the same dataset import style as forces_for_free.py.

Data source per test:
  code/test <TEST_NUM> - sensor v5/processed_data/
    - dataset.csv

Pipeline:
- Build sliding-window barometer features per side (R/L)
- Create non-leaking train/val/test splits per test (same split logic as FFF)
- Train one LightGBM regressor per target [fx, fy, fz] and side
- Use validation set for early stopping
- Evaluate on test set and save predictions/metrics/models/plots
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Suppress harmless LightGBM / sklearn interop warnings
warnings.filterwarnings("ignore", message=".*Usage of np.ndarray subset.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*", category=UserWarning)


# =========================== CONFIGURATION ===========================

DEFAULT_TEST_NUMS = [52021001, 52021002, 52021003, 52021004, 52021005, 52021006, 52021007, 52021008, 52021009, 52021010, 52021011]
DEFAULT_TIME_SERIES_TEST_NUM = DEFAULT_TEST_NUMS[0]
SENSOR_SIDES = ["R", "L"]
TIME_COL = "time"

WINDOW_SIZE = 10
USE_FIRST_DERIVATIVE = True
USE_SECOND_DERIVATIVE = True
MAX_TIME_GAP = 10.0

# Keep this equal to Forces For Free SEQ_LEN so anti-leak gap matches that style.
SPLIT_SEQ_LEN = 8
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

FORCE_RENAME_MAP = {
    "fx_R_surf": "fx_R",
    "fy_R_surf": "fy_R",
    "fz_R_surf": "fz_R",
    "fx_L_surf": "fx_L",
    "fy_L_surf": "fy_L",
    "fz_L_surf": "fz_L",
    "force_x_R": "fx_R",
    "force_y_R": "fy_R",
    "force_z_R": "fz_R",
    "force_x_L": "fx_L",
    "force_y_L": "fy_L",
    "force_z_L": "fz_L",
}


# =============================== HELPERS =============================

def load_test_dataframe(root_dir: Path, test_num: int) -> pd.DataFrame:
    code_dir = root_dir / "code"
    dataset_dir = code_dir / f"test {test_num} - sensor v5" / "processed_data"
    dataset_csv = dataset_dir / "dataset.csv"

    if not dataset_csv.exists():
        raise FileNotFoundError(f"[{test_num}] Missing file: {dataset_csv}")

    df = pd.read_csv(dataset_csv)
    df.columns = [c.strip() for c in df.columns]

    if TIME_COL not in df.columns:
        raise ValueError(f"[{test_num}] '{TIME_COL}' column not found in {dataset_csv}")

    df = df.rename(columns=FORCE_RENAME_MAP)

    required_force_cols = ["fx_R", "fy_R", "fz_R", "fx_L", "fy_L", "fz_L"]
    missing_forces = [c for c in required_force_cols if c not in df.columns]
    if missing_forces:
        raise ValueError(f"[{test_num}] Missing force columns: {missing_forces}")

    return df.sort_values(TIME_COL).reset_index(drop=True)


def time_based_split_masks(
    center_times: np.ndarray,
    raw_times: np.ndarray,
    train_split: float,
    val_split: float,
    seq_len: int,
):
    """
    Assign each window to train/val/test based on absolute time boundaries
    derived from the raw dataset time range.  This ensures that both LightGBM
    and FFF (which sort by time before splitting) use the exact same temporal
    window as their test set, regardless of how many windows each pipeline
    dropped for other reasons (missing images, large Δt, etc.).

    Boundaries:
        t_train_end  = t_min + train_split  * duration
        gap_secs     = (seq_len - 1) * median_dt   (anti-leak gap in seconds)
        t_val_start  = t_train_end  + gap_secs
        t_val_end    = t_min + (train_split + val_split) * duration
        t_test_start = t_val_end    + gap_secs
    """
    sorted_raw = np.sort(raw_times)
    if sorted_raw[0] == sorted_raw[-1]:
        raise ValueError("All timestamps are identical; cannot split.")

    dt_median  = float(np.median(np.diff(sorted_raw)))
    gap_secs   = max(dt_median * (seq_len - 1), 0.0)

    # Use quantile so that exactly train_split fraction of CSV *rows* (not
    # time range) defines each boundary.
    t_train_end  = float(np.quantile(sorted_raw, train_split))
    t_val_start  = t_train_end  + gap_secs
    t_val_end    = float(np.quantile(sorted_raw, train_split + val_split))
    t_test_start = t_val_end    + gap_secs

    train_mask = center_times <= t_train_end
    val_mask   = (center_times >= t_val_start) & (center_times <= t_val_end)
    test_mask  = center_times >= t_test_start

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        # Temporal gap in this dataset left one split empty — fall back to
        # index-based split (same logic as non_leaking_split_masks).
        print(
            f"    [time-split] WARNING: empty split "
            f"(train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}), "
            f"falling back to index-based split."
        )
        return non_leaking_split_masks(
            n_seq=len(center_times),
            seq_len=seq_len,
            train_split=train_split,
            val_split=val_split,
            test_split=1.0 - train_split - val_split,
        )

    print(
        f"    [time-split] duration={sorted_raw[-1] - sorted_raw[0]:.1f}s  "
        f"train≤{t_train_end:.2f}s  "
        f"val=[{t_val_start:.2f},{t_val_end:.2f}]s  "
        f"test≥{t_test_start:.2f}s  "
        f"gap={gap_secs:.3f}s"
    )
    return train_mask, val_mask, test_mask


def non_leaking_split_masks(
    n_seq: int,
    seq_len: int,
    train_split: float,
    val_split: float,
    test_split: float,
):
    """
    Build non-leaking train/val/test masks for sequence indices.
    """
    split_sum = train_split + val_split + test_split
    if not np.isclose(split_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0 (got {split_sum:.4f}).")

    if n_seq <= 0:
        raise ValueError(f"Not enough sequences for seq_len={seq_len}.")

    requested_gap = int(seq_len) - 1
    max_feasible_gap = max(0, (n_seq - 3) // 2)
    gap = min(requested_gap, max_feasible_gap)
    if gap < requested_gap:
        print(
            f"[non_leaking_split] n_seq={n_seq} is too small for requested "
            f"non-leak gap={requested_gap}; using reduced gap={gap}."
        )

    usable = n_seq - 2 * gap
    if usable < 3:
        raise ValueError(
            f"Not enough sequences ({n_seq}) to build train/val/test splits "
            f"with non-leak gap={gap}."
        )

    target = np.array([train_split, val_split, test_split], dtype=float) * usable
    counts = np.floor(target).astype(int)
    remainder = int(usable - counts.sum())
    if remainder > 0:
        frac = target - counts
        for idx in np.argsort(-frac)[:remainder]:
            counts[idx] += 1

    # Guarantee each split has at least one sample by borrowing from larger splits.
    for idx in np.where(counts == 0)[0]:
        donors = np.where(counts > 1)[0]
        if len(donors) == 0:
            raise ValueError(
                f"Unable to allocate non-empty train/val/test splits for n_seq={n_seq} "
                f"(usable={usable}, gap={gap})."
            )
        donor = donors[np.argmax(counts[donors])]
        counts[donor] -= 1
        counts[idx] += 1

    n_train, n_val, n_test = counts.tolist()

    train_start = 0
    train_end = n_train
    val_start = n_train + gap
    val_end = val_start + n_val
    test_start = val_end + gap
    test_end = test_start + n_test

    train_seq_starts = np.arange(train_start, train_end, dtype=int)
    val_seq_starts = np.arange(val_start, val_end, dtype=int)
    test_seq_starts = np.arange(test_start, test_end, dtype=int)

    train_mask = np.zeros(n_seq, dtype=bool)
    val_mask = np.zeros(n_seq, dtype=bool)
    test_mask = np.zeros(n_seq, dtype=bool)
    train_mask[train_seq_starts] = True
    val_mask[val_seq_starts] = True
    test_mask[test_seq_starts] = True
    return train_mask, val_mask, test_mask


def side_baro_cols(df: pd.DataFrame, side: str) -> list[str]:
    cols = [f"b{i}_{side}" for i in range(1, 7)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing barometer columns for side {side}: {missing}")
    return cols


def build_window_features(
    df: pd.DataFrame,
    baro_cols: list[str],
    side: str,
    window_size: int,
    max_time_gap: float,
    use_first_derivative: bool,
    use_second_derivative: bool,
):
    data = df.copy()

    for col in baro_cols:
        d1 = data[col].diff().fillna(0.0)
        data[f"{col}_d1"] = d1
        if use_second_derivative:
            data[f"{col}_d2"] = d1.diff().fillna(0.0)

    time_vals = data[TIME_COL].to_numpy()
    baro_data = data[baro_cols].to_numpy()
    d1_data = data[[f"{c}_d1" for c in baro_cols]].to_numpy() if use_first_derivative else None
    d2_data = data[[f"{c}_d2" for c in baro_cols]].to_numpy() if use_second_derivative else None

    x_list, y_list, keep_idx = [], [], []

    for cur in range(window_size, len(data)):
        start = cur - window_size
        end = cur + 1

        if np.max(np.diff(time_vals[start:end])) > max_time_gap:
            continue

        feat_blocks = [baro_data[start:end, :].reshape(-1)]
        if use_first_derivative:
            feat_blocks.append(d1_data[start:end, :].reshape(-1))
        if use_second_derivative:
            feat_blocks.append(d2_data[start:end, :].reshape(-1))

        x_list.append(np.concatenate(feat_blocks))
        y_list.append(
            [
                float(data.loc[cur, f"fx_{side}"]),
                float(data.loc[cur, f"fy_{side}"]),
                float(data.loc[cur, f"fz_{side}"]),
            ]
        )
        keep_idx.append(cur)

    if not x_list:
        raise ValueError("No valid windows found. Consider increasing MAX_TIME_GAP.")

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    center_df = data.iloc[keep_idx].reset_index(drop=True)
    return x, y, center_df


def collect_side_data(root_dir: Path, test_nums: list[int], side: str):
    x_train_all, y_train_all = [], []
    x_val_all, y_val_all = [], []
    x_test_all, y_test_all = [], []
    t_test_all, testnum_test_all = [], []

    for test_num in test_nums:
        df = load_test_dataframe(root_dir=root_dir, test_num=test_num)
        x, y, center_df = build_window_features(
            df=df,
            baro_cols=side_baro_cols(df, side),
            side=side,
            window_size=WINDOW_SIZE,
            max_time_gap=MAX_TIME_GAP,
            use_first_derivative=USE_FIRST_DERIVATIVE,
            use_second_derivative=USE_SECOND_DERIVATIVE,
        )
        # Use time-based split so both LightGBM and FFF share the same test
        # time window (boundaries derived from the raw dataset time range).
        train_mask, val_mask, test_mask = time_based_split_masks(
            center_times=center_df[TIME_COL].to_numpy(dtype=float),
            raw_times=df[TIME_COL].to_numpy(dtype=float),
            train_split=TRAIN_SPLIT,
            val_split=VAL_SPLIT,
            seq_len=SPLIT_SEQ_LEN,
        )

        x_train_all.append(x[train_mask])
        y_train_all.append(y[train_mask])
        x_val_all.append(x[val_mask])
        y_val_all.append(y[val_mask])
        x_test_all.append(x[test_mask])
        y_test_all.append(y[test_mask])

        t_center = center_df[TIME_COL].to_numpy(dtype=float)
        t_local = t_center[test_mask]
        t_test_all.append(t_local)
        testnum_test_all.append(np.full_like(t_local, fill_value=float(test_num), dtype=float))

        print(
            f"  [{test_num}][{side}] windows={len(x)} -> "
            f"train={int(np.sum(train_mask))} val={int(np.sum(val_mask))} "
            f"test={int(np.sum(test_mask))} dropped_unmatched=0"
        )

    return {
        "x_train": np.concatenate(x_train_all, axis=0),
        "y_train": np.concatenate(y_train_all, axis=0),
        "x_val": np.concatenate(x_val_all, axis=0),
        "y_val": np.concatenate(y_val_all, axis=0),
        "x_test": np.concatenate(x_test_all, axis=0),
        "y_test": np.concatenate(y_test_all, axis=0),
        "t_test": np.concatenate(t_test_all, axis=0),
        "test_num_test": np.concatenate(testnum_test_all, axis=0).astype(int),
        "dropped_for_gap": 0,
    }


def train_side_lightgbm(side: str, data: dict, args):
    scaler = StandardScaler()
    n_features = data["x_train"].shape[1]
    feat_cols = [f"f{i}" for i in range(n_features)]

    x_train = pd.DataFrame(
        np.ascontiguousarray(scaler.fit_transform(data["x_train"]), dtype=np.float32),
        columns=feat_cols,
    )
    x_val = pd.DataFrame(
        np.ascontiguousarray(scaler.transform(data["x_val"]), dtype=np.float32),
        columns=feat_cols,
    )
    x_test = pd.DataFrame(
        np.ascontiguousarray(scaler.transform(data["x_test"]), dtype=np.float32),
        columns=feat_cols,
    )

    target_short = ["fx", "fy", "fz"]
    models = {}
    preds = np.zeros_like(data["y_test"], dtype=np.float32)
    metrics = []

    for i, tname in enumerate(target_short):
        y_train = data["y_train"][:, i]
        y_val = data["y_val"][:, i]
        y_test = data["y_test"][:, i]

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            random_state=args.seed,
            n_jobs=-1,
        )

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
        )

        yp = model.predict(x_test)
        preds[:, i] = yp

        mae = float(mean_absolute_error(y_test, yp))
        rmse = float(np.sqrt(mean_squared_error(y_test, yp)))
        r2 = float(r2_score(y_test, yp))
        best_iter = int(getattr(model, "best_iteration_", -1) or -1)

        print(
            f"  [{side}] {tname:<2} | best_iter={best_iter:>4} "
            f"| MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}"
        )

        metrics.append(
            {
                "side": side,
                "target": tname,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "best_iteration": best_iter,
            }
        )
        models[tname] = model

    return scaler, models, preds, pd.DataFrame(metrics)


def plot_pred_vs_actual_by_side(pred_by_side: dict[str, np.ndarray], y_by_side: dict[str, np.ndarray], save_path: Path):
    ordered = [("L", 0, "fx_L"), ("L", 1, "fy_L"), ("L", 2, "fz_L"),
               ("R", 0, "fx_R"), ("R", 1, "fy_R"), ("R", 2, "fz_R")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, (side, idx, name) in zip(axes.flatten(), ordered):
        yt = y_by_side[side][:, idx]
        yp = pred_by_side[side][:, idx]

        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp))

        ax.scatter(yt, yp, s=12, alpha=0.35, color="#292f56")
        lo = float(min(np.min(yt), np.min(yp)))
        hi = float(max(np.max(yt), np.max(yp)))
        ax.plot([lo, hi], [lo, hi], "--", color="#d6c52e", linewidth=1.4)
        ax.set_title(f"{name}\nMAE: {mae:.2f} | R2: {r2:.3f}")
        ax.set_xlabel("Ground Truth [N]")
        ax.set_ylabel("Predicted [N]")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Tactile Sensor -  Predicted vs Actual", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries_by_side(
    pred_by_side: dict[str, np.ndarray],
    y_by_side: dict[str, np.ndarray],
    t_by_side: dict[str, np.ndarray],
    testnum_by_side: dict[str, np.ndarray],
    time_series_test_num: int,
    save_path: Path,
):
    """2×3 time-series panel (same style as FFF): top row = L, bottom row = R.
    Uses only the first test number, first 20 s, sorted by time."""
    CLR_GT   = "#d6c52e"   # same as CLR_TS_GT_COMMON in FFF
    CLR_PRED = "#292f56"   # same as scatter plot prediction color
    targets = ["fx", "fy", "fz"]
    side_rows = [("L", 0), ("R", 1)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=False)

    for side, row_idx in side_rows:
        preds   = pred_by_side[side]        # (N, 3)
        y_true  = y_by_side[side]           # (N, 3)
        t       = t_by_side[side]           # (N,)
        tnums   = testnum_by_side[side]     # (N,)

        # Keep only the selected test number.
        mask = tnums == time_series_test_num
        if not np.any(mask):
            available = np.unique(tnums).astype(int).tolist()
            print(
                f"  [time-series] WARNING: no samples for test {time_series_test_num} on side {side}. "
                f"Available tests: {available}"
            )
            continue
        t      = t[mask]
        preds  = preds[mask]
        y_true = y_true[mask]

        # Sort by time and crop to first 20 s
        order = np.argsort(t)
        t      = t[order]
        preds  = preds[order]
        y_true = y_true[order]
        if t.size > 0:
            t0   = float(t[0])
            keep = (t >= t0) & (t <= t0 + 20.0)
            t      = t[keep]
            preds  = preds[keep]
            y_true = y_true[keep]
            t      = t - t0  # relative time so x-axis starts at 0

        for col_idx, tname in enumerate(targets):
            ax = axes[row_idx, col_idx]
            label = f"{tname}_{side}"
            ax.plot(
                t, y_true[:, col_idx],
                color=CLR_GT, linewidth=2.4, alpha=0.95,
                label="Ground Truth",
            )
            ax.scatter(t, y_true[:, col_idx], color=CLR_GT, s=12, alpha=0.7, zorder=5)
            ax.plot(
                t, preds[:, col_idx],
                color=CLR_PRED, linewidth=2.4, alpha=0.95,
                label="Predicted",
            )
            ax.scatter(t, preds[:, col_idx], color=CLR_PRED, s=12, alpha=0.7, zorder=5)
            mae = float(mean_absolute_error(y_true[:, col_idx], preds[:, col_idx]))
            r2  = float(r2_score(y_true[:, col_idx], preds[:, col_idx]))
            ax.set_title(f"{label}", fontsize=11)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"{label} [N]")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right")

    fig.suptitle("Time Series: Predicted vs Ground Truth (Right + Left)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved time-series plot: {save_path}")


def print_split_target_summary(side: str, data: dict[str, np.ndarray]):
    print(f"  [{side}] target stats by split:")
    target_names = ["fx", "fy", "fz"]
    split_map = [
        ("train", data["y_train"]),
        ("val", data["y_val"]),
        ("test", data["y_test"]),
    ]
    for split_name, y_split in split_map:
        print(f"    {split_name}: n={len(y_split)}")
        for idx, tname in enumerate(target_names):
            vals = y_split[:, idx]
            print(
                f"      {tname}_{side}: "
                f"min={float(np.min(vals)):.3f} "
                f"max={float(np.max(vals)):.3f} "
                f"mean={float(np.mean(vals)):.3f}"
            )


def plot_force_histograms_combined_by_split(
    split_targets_by_side: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
):
    # Match existing plot colors used in this script.
    clr_pred_blue = "#292f56"
    clr_mean_yellow = "#d6c52e"
    clr_median = "#2c9a7a"
    target_names = ["fx", "fy", "fz"]

    for split_name in ["train", "val", "test"]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
        for row_idx, side in enumerate(["L", "R"]):
            y_split = split_targets_by_side[side][split_name]
            for col_idx, tname in enumerate(target_names):
                ax = axes[row_idx, col_idx]
                vals = y_split[:, col_idx]
                mean_v = float(np.mean(vals))
                median_v = float(np.median(vals))

                ax.hist(vals, bins=60, color=clr_pred_blue, alpha=0.9)
                ax.axvline(
                    mean_v,
                    color=clr_mean_yellow,
                    linewidth=2.2,
                    linestyle="--",
                    label=f"Mean: {mean_v:.3f}",
                )
                ax.axvline(
                    median_v,
                    color=clr_median,
                    linewidth=2.2,
                    linestyle="-.",
                    label=f"Median: {median_v:.3f}",
                )
                ax.set_title(f"{tname}_{side} ({split_name})")
                ax.set_xlabel("Force [N]")
                ax.set_ylabel("Count")
                ax.grid(alpha=0.25)
                ax.legend(loc="upper right", fontsize=9)

        fig.suptitle(f"Force Distribution (L+R) - {split_name}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = out_dir / f"lightgbm_force_distribution_left_right_{split_name}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved histogram plot (L+R, {split_name}): {save_path}")


# ============================== MAIN ==================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-nums",
        default=",".join(str(t) for t in DEFAULT_TEST_NUMS),
        help="Comma-separated test IDs, e.g. 52021001,52021002",
    )
    parser.add_argument(
        "--timeseries-test-num",
        type=int,
        default=DEFAULT_TIME_SERIES_TEST_NUM,
        help="Test ID used for the time-series plot.",
    )
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_nums = [int(x.strip()) for x in args.test_nums.split(",") if x.strip()]
    if not test_nums:
        raise ValueError("--test-nums resolved to an empty list.")
    time_series_test_num = int(args.timeseries_test_num)
    if time_series_test_num not in test_nums:
        print(
            f"[time-series] WARNING: selected test {time_series_test_num} is not in --test-nums. "
            f"Using first listed test {test_nums[0]} instead."
        )
        time_series_test_num = int(test_nums[0])

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    out_dir = root_dir / "outputs" / "lightgbm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Train LightGBM from raw test folders (FFF-style import + non-leaking split)")
    print(f"Tests: {test_nums}")
    print(f"Time-series plot test: {time_series_test_num}")
    print(f"Window: {WINDOW_SIZE} | Derivatives: d1={USE_FIRST_DERIVATIVE}, d2={USE_SECOND_DERIVATIVE}")
    print(f"Split: train/val/test = {TRAIN_SPLIT:.2f}/{VAL_SPLIT:.2f}/{TEST_SPLIT:.2f}")
    print(f"Anti-leak gap source seq_len: {SPLIT_SEQ_LEN}")
    print("=" * 72)

    side_scalers = {}
    side_models = {}
    side_preds = {}
    side_ytest = {}
    side_ttest = {}
    side_testnums = {}
    side_split_targets = {}
    metrics_frames = []

    for side in SENSOR_SIDES:
        print(f"\n--- Preparing side {side} ---")
        data = collect_side_data(root_dir=root_dir, test_nums=test_nums, side=side)
        print(
            f"  [{side}] total train={len(data['x_train'])} val={len(data['x_val'])} "
            f"test={len(data['x_test'])} dropped_gap_total={data['dropped_for_gap']}"
        )
        print_split_target_summary(side=side, data=data)
        side_split_targets[side] = {
            "train": data["y_train"],
            "val": data["y_val"],
            "test": data["y_test"],
        }

        print(f"--- Training side {side} models ---")
        scaler, models, preds, metrics_df = train_side_lightgbm(side=side, data=data, args=args)

        side_scalers[side] = scaler
        side_models[side] = models
        side_preds[side] = preds
        side_ytest[side] = data["y_test"]
        side_ttest[side] = data["t_test"]
        side_testnums[side] = data["test_num_test"]
        metrics_frames.append(metrics_df)

        pred_df = pd.DataFrame(
            {
                "test_num": data["test_num_test"],
                "time": data["t_test"],
                f"fx_{side}_true": data["y_test"][:, 0],
                f"fy_{side}_true": data["y_test"][:, 1],
                f"fz_{side}_true": data["y_test"][:, 2],
                f"fx_{side}_pred": preds[:, 0],
                f"fy_{side}_pred": preds[:, 1],
                f"fz_{side}_pred": preds[:, 2],
            }
        ).sort_values(["test_num", "time"])
        pred_path = out_dir / f"lightgbm_test_predictions_{side}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"  Saved predictions ({side}): {pred_path}")

    # Also save a single combined test-set CSV (L + R) keyed by test_num/time.
    combined_pred_df = (
        pd.merge(
            pd.DataFrame(
                {
                    "test_num": side_testnums["L"],
                    "time": side_ttest["L"],
                    "fx_L_true": side_ytest["L"][:, 0],
                    "fy_L_true": side_ytest["L"][:, 1],
                    "fz_L_true": side_ytest["L"][:, 2],
                    "fx_L_pred": side_preds["L"][:, 0],
                    "fy_L_pred": side_preds["L"][:, 1],
                    "fz_L_pred": side_preds["L"][:, 2],
                }
            ),
            pd.DataFrame(
                {
                    "test_num": side_testnums["R"],
                    "time": side_ttest["R"],
                    "fx_R_true": side_ytest["R"][:, 0],
                    "fy_R_true": side_ytest["R"][:, 1],
                    "fz_R_true": side_ytest["R"][:, 2],
                    "fx_R_pred": side_preds["R"][:, 0],
                    "fy_R_pred": side_preds["R"][:, 1],
                    "fz_R_pred": side_preds["R"][:, 2],
                }
            ),
            on=["test_num", "time"],
            how="outer",
            sort=True,
        )
        .sort_values(["test_num", "time"])
        .reset_index(drop=True)
    )
    combined_pred_path = out_dir / "lightgbm_test_predictions_LR.csv"
    combined_pred_df.to_csv(combined_pred_path, index=False)
    print(f"  Saved combined predictions (L+R): {combined_pred_path}")

    metrics_all = pd.concat(metrics_frames, axis=0, ignore_index=True)
    metrics_path = out_dir / "lightgbm_test_metrics.csv"
    metrics_all.to_csv(metrics_path, index=False)

    plot_path = out_dir / "lightgbm_pred_vs_actual_left_right.png"
    plot_pred_vs_actual_by_side(pred_by_side=side_preds, y_by_side=side_ytest, save_path=plot_path)
    plot_force_histograms_combined_by_split(split_targets_by_side=side_split_targets, out_dir=out_dir)

    ts_plot_path = out_dir / "lightgbm_timeseries_gt_vs_pred.png"
    plot_timeseries_by_side(
        pred_by_side=side_preds,
        y_by_side=side_ytest,
        t_by_side=side_ttest,
        testnum_by_side=side_testnums,
        time_series_test_num=time_series_test_num,
        save_path=ts_plot_path,
    )

    model_bundle = {
        "test_nums": test_nums,
        "window_size": WINDOW_SIZE,
        "use_first_derivative": USE_FIRST_DERIVATIVE,
        "use_second_derivative": USE_SECOND_DERIVATIVE,
        "max_time_gap": MAX_TIME_GAP,
        "split_seq_len": SPLIT_SEQ_LEN,
        "split": {"train": TRAIN_SPLIT, "val": VAL_SPLIT, "test": TEST_SPLIT},
        "side_scalers": side_scalers,
        "side_models": side_models,
        "model_params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_child_samples": args.min_child_samples,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "early_stopping_rounds": args.early_stopping_rounds,
            "seed": args.seed,
        },
    }
    model_path = script_dir / "lightgbm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    # Save explicit per-side artifacts in the same folder as this script.
    left_model_path = script_dir / "lightgbm_left_model.pkl"
    right_model_path = script_dir / "lightgbm_right_model.pkl"
    left_scaler_path = script_dir / "lightgbm_left_scaler.pkl"
    right_scaler_path = script_dir / "lightgbm_right_scaler.pkl"

    with open(left_model_path, "wb") as f:
        pickle.dump(side_models["L"], f)
    with open(right_model_path, "wb") as f:
        pickle.dump(side_models["R"], f)
    with open(left_scaler_path, "wb") as f:
        pickle.dump(side_scalers["L"], f)
    with open(right_scaler_path, "wb") as f:
        pickle.dump(side_scalers["R"], f)

    print("\nSaved artifacts:")
    print(f"  Model bundle: {model_path}")
    print(f"  Left model  : {left_model_path}")
    print(f"  Right model : {right_model_path}")
    print(f"  Left scaler : {left_scaler_path}")
    print(f"  Right scaler: {right_scaler_path}")
    print(f"  Metrics CSV : {metrics_path}")
    print(f"  Scatter plot: {plot_path}")


if __name__ == "__main__":
    main()
