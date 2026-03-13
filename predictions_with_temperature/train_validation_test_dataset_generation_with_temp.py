#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train / Validation / Test Dataset Generation — Temperature-Aware Pipeline (v5.20)

Reads synchronized_events CSVs produced by data_organization_with_temp.py
(which include t1..t6 temperature columns alongside b1..b6 barometer pressures)
and splits them into train / validation / test sets.

Temperature columns (t1..t6) are preserved as-is through the split: no extra
normalization is applied here — that is done inside the prediction script
(StandardScaler covers all feature columns uniformly).

Split strategy: chronological (no shuffle), per-file 70/15/15.

Key differences from the baseline script:
  - VERSION_NUM = 5.20  (distinct from v5.01 baseline)
  - Output filenames: train_data_v5.2.csv, validation_data_v5.2.csv, test_data_v5.2.csv
  - Prints temperature column availability report per file

Author: Aurora Ruggeri
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================== CONFIG ==================

DATA_DIRECTORY = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data")
VERSION_NUM = 5.20   # sensor version tag for v5.20 temperature-aware run

# Fractions must sum to 1.0
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# ── List of synchronized_events CSVs produced by data_organization_with_temp.py ──
# Paths are inside the with_temperature/ subfolder created by data_organization_with_temp.py.
# IMPORTANT: Run data_organization_with_temp.py first to populate these folders.
CSV_FILENAMES = [

    r"test 52000 - sensor v5\with_temperature\synchronized_events_52000.csv",
    r"test 52001 - sensor v5\with_temperature\synchronized_events_52001.csv",
    r"test 52002 - sensor v5\with_temperature\synchronized_events_52002.csv",
    r"test 52003 - sensor v5\with_temperature\synchronized_events_52003.csv",
    r"test 52004 - sensor v5\with_temperature\synchronized_events_52004.csv",
    r"test 52005 - sensor v5\with_temperature\synchronized_events_52005.csv",
    r"test 52006 - sensor v5\with_temperature\synchronized_events_52006.csv",

    # r"test 52100 - sensor v5\with_temperature\synchronized_events_52100.csv",

    # r"test 51700 - sensor v5\with_temperature\synchronized_events_51700.csv",
    # r"test 51701 - sensor v5\with_temperature\synchronized_events_51701.csv",
    # r"test 51702 - sensor v5\with_temperature\synchronized_events_51702.csv",
    # r"test 51704 - sensor v5\with_temperature\synchronized_events_51704.csv",
    # r"test 51705 - sensor v5\with_temperature\synchronized_events_51705.csv",

    # r"test 51800 - sensor v5\with_temperature\synchronized_events_51800.csv",
    # r"test 51801 - sensor v5\with_temperature\synchronized_events_51801.csv",
    # r"test 51802 - sensor v5\with_temperature\synchronized_events_51802.csv",
    # r"test 51803 - sensor v5\with_temperature\synchronized_events_51803.csv",

    # r"test 51900 - sensor v5\with_temperature\synchronized_events_51900.csv",
    # r"test 51901 - sensor v5\with_temperature\synchronized_events_51901.csv",
    # r"test 51902 - sensor v5\with_temperature\synchronized_events_51902.csv",
    # r"test 51903 - sensor v5\with_temperature\synchronized_events_51903.csv",
]

# Output directory for final combined datasets
OUTPUT_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data")

# Temperature columns to check (they may be NaN if sensor lacked thermistor readout)
TEMP_COLS = [f"t{i}" for i in range(1, 7)]

# ============================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_parts    = []
val_parts      = []
test_parts     = []
sampling_rates = []

files_with_temp    = 0
files_without_temp = 0

for rel_path in CSV_FILENAMES:
    input_path = DATA_DIRECTORY / rel_path

    if not input_path.is_file():
        print(f"[WARN] Could not find CSV at: {input_path}")
        continue

    df = pd.read_csv(input_path)
    n = len(df)

    if n == 0:
        print(f"[WARN] Empty CSV, skipping: {input_path}")
        continue

    # Report temperature availability
    temp_present = [c for c in TEMP_COLS if c in df.columns and df[c].notna().any()]
    if temp_present:
        files_with_temp += 1
        temp_status = f"temp OK ({len(temp_present)}/6 channels)"
    else:
        files_without_temp += 1
        temp_status = "NO temp data"
        # Insert NaN temperature columns so all files share the same schema
        for c in TEMP_COLS:
            if c not in df.columns:
                df[c] = np.nan

    # Sampling rate
    time_col = next((c for c in ['t', 'time', 'ros_time', 'Epoch_s'] if c in df.columns), None)
    if time_col and n > 1:
        diffs = np.diff(df[time_col].values.astype(float))
        fs = 1.0 / np.mean(diffs)
        print(f"  [{rel_path}]  {fs:.1f} Hz  n={n}  {temp_status}")
        sampling_rates.append(fs)
    else:
        print(f"  [{rel_path}]  n={n}  {temp_status}")
        sampling_rates.append(float('nan'))

    # Chronological split
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_parts.append(df.iloc[:n_train].copy())
    val_parts.append(df.iloc[n_train:n_train + n_val].copy())
    test_parts.append(df.iloc[n_train + n_val:].copy())

if not train_parts:
    raise RuntimeError("No data loaded from any CSV. Check paths / filenames.")

combined_train = pd.concat(train_parts, ignore_index=True)

# ── Pipeline guard: verify temperature columns are actually present ────────────
temp_cols_in_combined = [c for c in TEMP_COLS
                         if c in combined_train.columns and combined_train[c].notna().any()]
if not temp_cols_in_combined:
    print("\n" + "!" * 70)
    print("WARNING: No temperature data (t1..t6) found in any loaded CSV.")
    print("The prediction script will fall back to pressure-only mode.")
    print("Did you run data_organization_with_temp.py first?")
    print("Expected files inside:  <test_folder>/with_temperature/synchronized_events_XXXXX.csv")
    print("!" * 70 + "\n")
else:
    print(f"\n[OK] Temperature data confirmed in combined dataset: {temp_cols_in_combined}")
# ─────────────────────────────────────────────────────────────────────────────
combined_val   = pd.concat(val_parts,   ignore_index=True) if val_parts  else pd.DataFrame(columns=combined_train.columns)
combined_test  = pd.concat(test_parts,  ignore_index=True) if test_parts else pd.DataFrame(columns=combined_train.columns)

# Save
train_path = OUTPUT_DIR / f"train_data_v{VERSION_NUM}.csv"
val_path   = OUTPUT_DIR / f"validation_data_v{VERSION_NUM}.csv"
test_path  = OUTPUT_DIR / f"test_data_v{VERSION_NUM}.csv"

combined_train.to_csv(train_path, index=False)
combined_val.to_csv(val_path,     index=False)
combined_test.to_csv(test_path,   index=False)

print("\n========== SUMMARY ==========")
print(f"Sensor version      : {VERSION_NUM}")
print(f"Files with temp data: {files_with_temp}")
print(f"Files without temp  : {files_without_temp}")
print(f"Final train rows    : {len(combined_train)}  -> {train_path}")
print(f"Final val   rows    : {len(combined_val)}  -> {val_path}")
print(f"Final test  rows    : {len(combined_test)}  -> {test_path}")

# Columns present in combined data
baro_cols    = [c for c in combined_train.columns if c in [f"b{i}" for i in range(1, 7)]]
temp_cols_ok = [c for c in combined_train.columns if c in TEMP_COLS and combined_train[c].notna().any()]
print(f"\nBarometer cols      : {baro_cols}")
print(f"Temperature cols    : {temp_cols_ok if temp_cols_ok else '(none — NaN columns present)'}")

valid_rates = [r for r in sampling_rates if not np.isnan(r)]
if valid_rates:
    print(f"\nSampling rate avg   : {np.mean(valid_rates):.1f} Hz "
          f"(min {np.min(valid_rates):.1f}, max {np.max(valid_rates):.1f})")


# ========== DATA DISTRIBUTION PLOTS ==========

def plot_data_distribution(y_data, target_names, title_prefix=""):
    if isinstance(y_data, pd.DataFrame):
        y_data = y_data[target_names].values
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4))
    if n_targets == 1:
        axes = [axes]
    for i in range(n_targets):
        ax = axes[i]
        data = y_data[:, i]
        data_filtered = data[data != -999.0]
        ax.hist(data_filtered, bins=50, alpha=0.7, color='#005c7f', edgecolor='black')
        mean_val   = np.mean(data_filtered)
        std_val    = np.std(data_filtered)
        median_val = np.median(data_filtered)
        ax.axvline(mean_val,   color='#292f56', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='#d6c52e', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.2f}')
        unit = "mm" if target_names[i] in ['x', 'y'] else \
               "N·m" if target_names[i] in ['tx', 'ty', 'tz'] else "N"
        ax.set_title(f'{target_names[i]}\nStd: {std_val:.2f} {unit}', fontsize=10)
        ax.set_xlabel(f'{target_names[i]} ({unit})', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle(f'{title_prefix} Data Distribution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


plots_dir = OUTPUT_DIR / f"sensor_v{VERSION_NUM}"
plots_dir.mkdir(parents=True, exist_ok=True)
print(f"\nSaving distribution plots to: {plots_dir}")

position_cols = [c for c in ['x', 'y'] if c in combined_train.columns]
force_cols    = [c for c in ['fx', 'fy', 'fz'] if c in combined_train.columns]
all_targets   = position_cols + force_cols

for split_name, split_df in [("Training",   combined_train),
                              ("Validation", combined_val),
                              ("Test",       combined_test)]:
    if all_targets and len(split_df) > 0:
        fig = plot_data_distribution(split_df, all_targets, title_prefix=split_name)
        fig.savefig(plots_dir / f'{split_name.lower()}_distribution_v{VERSION_NUM}.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {split_name} distribution plot")

print("Done.")
