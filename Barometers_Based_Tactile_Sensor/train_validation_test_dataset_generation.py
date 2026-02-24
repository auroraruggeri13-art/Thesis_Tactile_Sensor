#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ================== CONFIG ==================
# Base directory that contains all test folders
DATA_DIRECTORY = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data")
VERSION_NUM = 5.1092

# Fractions must sum to 1.0
TRAIN_FRAC = 1
VAL_FRAC   = 0.0
TEST_FRAC  = 0.0

# Relative CSV paths inside DATA_DIRECTORY
CSV_FILENAMES = [
    
    r"test 51092 - sensor v5\synchronized_events_51092.csv",
    
    # r"test 2 - sensor v5\synchronized_events_2.csv",
    # r"test 3 - sensor v5\synchronized_events_3.csv",    
    # r"test 4 - sensor v5\synchronized_events_4.csv",
    # r"test 5 - sensor v5\synchronized_events_5.csv",    
    
    # r"test 4301 - sensor v4\synchronized_events_4301.csv",
    # r"test 4302 - sensor v4\synchronized_events_4302.csv",
    # r"test 4303 - sensor v4\synchronized_events_4303.csv",
    # r"test 4304 - sensor v4\synchronized_events_4304.csv",
    # r"test 4305 - sensor v4\synchronized_events_4305.csv",
    
    # r"test 4401 - sensor v4\synchronized_events_4401.csv",
    # r"test 4402 - sensor v4\synchronized_events_4402.csv",
    # r"test 4403 - sensor v4\synchronized_events_4403.csv",
    # r"test 4404 - sensor v4\synchronized_events_4404.csv",
    # r"test 4405 - sensor v4\synchronized_events_4405.csv",
    
    # r"test 4600 - sensor v4\synchronized_events_4600.csv",
    # r"test 4601 - sensor v4\synchronized_events_4601.csv",
    # r"test 4602 - sensor v4\synchronized_events_4602.csv",
    # r"test 4603 - sensor v4\synchronized_events_4603.csv",
    # r"test 4608 - sensor v4\synchronized_events_4608.csv",
    # r"test 4604 - sensor v4\synchronized_events_4604.csv",
    # r"test 4605 - sensor v4\synchronized_events_4605.csv",
    
    # r"test 4610 - sensor v4\synchronized_events_4610.csv",
    # r"test 4611 - sensor v4\synchronized_events_4611.csv",
    # r"test 4612 - sensor v4\synchronized_events_4612.csv",  
    # r"test 4613 - sensor v4\synchronized_events_4613.csv",
    # r"test 4614 - sensor v4\synchronized_events_4614.csv",
    # r"test 4615 - sensor v4\synchronized_events_4615.csv",
    # r"test 4616 - sensor v4\synchronized_events_4616.csv",
    
    # r"test 4620 - sensor v4\synchronized_events_4620.csv",
    # r"test 4622 - sensor v4\synchronized_events_4622.csv",
    
    # r"test 4621 - sensor v4\synchronized_events_4621.csv",
    # r"test 4623 - sensor v4\synchronized_events_4623.csv",
    
    # r"test 4700 - sensor v4\synchronized_events_4700.csv",
    # r"test 4701 - sensor v4\synchronized_events_4701.csv",
    # r"test 4702 - sensor v4\synchronized_events_4702.csv",
    
    # r"test 4800 - sensor v4\synchronized_events_4800.csv",
    # r"test 4801 - sensor v4\synchronized_events_4801.csv",
    
    # r"test 4900 - sensor v4\synchronized_events_4900.csv",
    # r"test 4901 - sensor v4\synchronized_events_4901.csv",
    
    # r"test 5200 - sensor v5\synchronized_events_5200.csv",
    # r"test 5201 - sensor v5\synchronized_events_5201.csv",
    # r"test 5202 - sensor v5\synchronized_events_5202.csv",
    # r"test 5203 - sensor v5\synchronized_events_5203.csv",
    
    # r"test 5210 - sensor v5\synchronized_events_5210.csv",
    # r"test 5211 - sensor v5\synchronized_events_5211.csv",
    # r"test 5212 - sensor v5\synchronized_events_5212.csv",
    # r"test 5213 - sensor v5\synchronized_events_5213.csv",
    
    # r"test 5700 - sensor v5\synchronized_events_5700.csv",
    # r"test 5701 - sensor v5\synchronized_events_5701.csv",
    # r"test 5702 - sensor v5\synchronized_events_5702.csv",
    # r"test 5703 - sensor v5\synchronized_events_5703.csv",
    # r"test 5704 - sensor v5\synchronized_events_5704.csv",
    
    # r"test 5800 - sensor v5\synchronized_events_5800.csv",
    # r"test 5801 - sensor v5\synchronized_events_5801.csv",
    # r"test 5802 - sensor v5\synchronized_events_5802.csv",
    # r"test 5803 - sensor v5\synchronized_events_5803.csv",
    # r"test 5804 - sensor v5\synchronized_events_5804.csv",
    # r"test 5805 - sensor v5\synchronized_events_5805.csv",
    # r"test 5806 - sensor v5\synchronized_events_5806.csv",
    
    # r"test 5900 - sensor v5\synchronized_events_5900.csv",
    # r"test 5901 - sensor v5\synchronized_events_5901.csv",
    # r"test 5902 - sensor v5\synchronized_events_5902.csv",
    # r"test 5903 - sensor v5\synchronized_events_5903.csv",
    # r"test 5904 - sensor v5\synchronized_events_5904.csv",
    # r"test 5905 - sensor v5\synchronized_events_5905.csv",
    # r"test 5906 - sensor v5\synchronized_events_5906.csv",
    
    # r"test 51000 - sensor v5\synchronized_events_51000.csv",
    # r"test 51001 - sensor v5\synchronized_events_51001.csv",
    # r"test 51002 - sensor v5\synchronized_events_51002.csv",
    # r"test 51003 - sensor v5\synchronized_events_51003.csv",
    # r"test 51004 - sensor v5\synchronized_events_51004.csv",
    # r"test 51005 - sensor v5\synchronized_events_51005.csv",
    # r"test 51006 - sensor v5\synchronized_events_51006.csv",
    # # r"test 51007 - sensor v5\synchronized_events_51007.csv",  
    # # r"test 51008 - sensor v5\synchronized_events_51008.csv", 
    # # r"test 51009 - sensor v5\synchronized_events_51009.csv",
    
    # # # # # r"test 51030 - sensor v5\synchronized_events_51030.csv",
    # # # # # r"test 51031 - sensor v5\synchronized_events_51031.csv",
    # # # # # r"test 51032 - sensor v5\synchronized_events_51032.csv",
    # # # # # r"test 51033 - sensor v5\synchronized_events_51033.csv",
    # # # # # r"test 51034 - sensor v5\synchronized_events_51034.csv",
    # # # # # r"test 51035 - sensor v5\synchronized_events_51035.csv",
    # # # # # r"test 51036 - sensor v5\synchronized_events_51036.csv",
    
    # r"test 51100 - sensor v5\synchronized_events_51100.csv", 
    # r"test 51101 - sensor v5\synchronized_events_51101.csv",
    # r"test 51102 - sensor v5\synchronized_events_51102.csv",
    # r"test 51103 - sensor v5\synchronized_events_51103.csv",
    # r"test 51104 - sensor v5\synchronized_events_51104.csv",
    # r"test 51105 - sensor v5\synchronized_events_51105.csv",
    # r"test 51106 - sensor v5\synchronized_events_51106.csv",
    # # r"test 51108 - sensor v5\synchronized_events_51108.csv",    
    # # r"test 51109 - sensor v5\synchronized_events_51109.csv",

    
    # # # # # r"test 51130 - sensor v5\synchronized_events_51130.csv",
    # # # # # r"test 51131 - sensor v5\synchronized_events_51131.csv",
    
    # # r"test 51200 - sensor v5\synchronized_events_51200.csv",

    # r"test 51201 - sensor v5\synchronized_events_51201.csv",
    # r"test 51202 - sensor v5\synchronized_events_51202.csv",
    # r"test 51203 - sensor v5\synchronized_events_51203.csv",
    # r"test 51204 - sensor v5\synchronized_events_51204.csv",
    # r"test 51205 - sensor v5\synchronized_events_51205.csv",
    # # r"test 51206 - sensor v5\synchronized_events_51206.csv",
    # # r"test 51207 - sensor v5\synchronized_events_51207.csv",
    
    # r"test 51300 - sensor v5\synchronized_events_51300.csv",
    # r"test 51301 - sensor v5\synchronized_events_51301.csv",
    # r"test 51302 - sensor v5\synchronized_events_51302.csv",
    # r"test 51303 - sensor v5\synchronized_events_51303.csv",
    # r"test 51304 - sensor v5\synchronized_events_51304.csv",
    
    # r"test 51400 - sensor v5\synchronized_events_51400.csv",
    # r"test 51401 - sensor v5\synchronized_events_51401.csv",
    # r"test 51402 - sensor v5\synchronized_events_51402.csv",
    # r"test 51403 - sensor v5\synchronized_events_51403.csv",
    # r"test 51404 - sensor v5\synchronized_events_51404.csv",
    # r"test 51405 - sensor v5\synchronized_events_51405.csv",
    
    # r"test 51500 - sensor v5\synchronized_events_51500.csv",
    # r"test 51501 - sensor v5\synchronized_events_51501.csv",
    # r"test 51502 - sensor v5\synchronized_events_51502.csv",
    # r"test 51503 - sensor v5\synchronized_events_51503.csv",
    
    # r"test 51600 - sensor v5\synchronized_events_51600.csv",
    # r"test 51601 - sensor v5\synchronized_events_51601.csv",
    # r"test 51602 - sensor v5\synchronized_events_51602.csv",
    # r"test 51603 - sensor v5\synchronized_events_51603.csv",
    # r"test 51604 - sensor v5\synchronized_events_51604.csv",
    
    # r"test 51700 - sensor v5\synchronized_events_51700.csv",
    
    # r"test 6000 - sensor v6\synchronized_events_6000.csv",
    # r"test 6001 - sensor v6\synchronized_events_6001.csv",
    
    # r"test 7000 - sensor v7\synchronized_events_7000.csv",
    # r"test 7001 - sensor v7\synchronized_events_7001.csv",
    
    # r"test 8000 - sensor v8\synchronized_events_8000.csv",
    # r"test 8001 - sensor v8\synchronized_events_8001.csv",
    
    
]

# Output directory for final combined datasets
OUTPUT_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data")
# ============================================

# Make sure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lists to accumulate per-split pieces from each file
train_parts = []
val_parts   = []
test_parts  = []

for rel_path in CSV_FILENAMES:
    input_path = DATA_DIRECTORY / rel_path

    if not input_path.is_file():
        print(f"[WARN] Could not find CSV at: {input_path}")
        continue

    df = pd.read_csv(input_path)

    n = len(df)
    if n == 0:
        print(f"[WARN] Input CSV is empty, nothing to split: {input_path}")
        continue

    # Compute split indices (no shuffle, keep order)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    # Whatever is left goes to test
    n_test  = n - n_train - n_val

    df_train = df.iloc[:n_train].copy()
    df_val   = df.iloc[n_train:n_train + n_val].copy()
    df_test  = df.iloc[n_train + n_val:].copy()

    # Accumulate
    train_parts.append(df_train)
    val_parts.append(df_val)
    test_parts.append(df_test)

# Combine across all files
if len(train_parts) == 0:
    raise RuntimeError("No data was loaded from any CSV. Check paths / filenames.")

combined_train = pd.concat(train_parts, ignore_index=True)
combined_val   = pd.concat(val_parts, ignore_index=True) if len(val_parts) > 0 else pd.DataFrame(columns=combined_train.columns)
combined_test  = pd.concat(test_parts, ignore_index=True) if len(test_parts) > 0 else pd.DataFrame(columns=combined_train.columns)

# Save final combined datasets
train_path = OUTPUT_DIR / f"train_data_v{VERSION_NUM}.csv"
val_path   = OUTPUT_DIR / f"validation_data_v{VERSION_NUM}.csv"
test_path  = OUTPUT_DIR / f"test_data_v{VERSION_NUM}.csv"

combined_train.to_csv(train_path, index=False)
combined_val.to_csv(val_path, index=False)
combined_test.to_csv(test_path, index=False)

print("\n========== SUMMARY ==========")
print(f"Final train rows: {len(combined_train)} -> {train_path}")
print(f"Final val   rows: {len(combined_val)} -> {val_path}")
print(f"Final test  rows: {len(combined_test)} -> {test_path}")

# ========== FUNCTION TO PLOT DATA DISTRIBUTION ==========
def plot_data_distribution(y_data, target_names, title_prefix=""):
    """
    Plot histograms showing the distribution of each target variable.
    
    Args:
        y_data: Data array (N, n_targets) or DataFrame
        target_names: List of target names
        title_prefix: Prefix for the plot title (e.g., "Training" or "Test")
    """
    # Convert to numpy array if DataFrame
    if isinstance(y_data, pd.DataFrame):
        y_data = y_data[target_names].values
    
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(4*n_targets, 4))
    if n_targets == 1:
        axes = [axes]
    
    for i in range(n_targets):
        ax = axes[i]
        data = y_data[:, i]
        
        # Filter out sentinel values (-999.0) for plotting
        data_filtered = data[data != -999.0]
        
        # Create histogram (only with real data, excluding sentinel values)
        ax.hist(data_filtered, bins=50, alpha=0.7, color='#005c7f', edgecolor='black')
        
        # Add statistics (computed on filtered data)
        mean_val = np.mean(data_filtered)
        std_val = np.std(data_filtered)
        median_val = np.median(data_filtered)
        
        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='#292f56', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='#d6c52e', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # Determine unit
        if target_names[i] in ['x', 'y']:
            unit = "mm"
        elif target_names[i] in ['tx', 'ty', 'tz']:
            unit = "N·m"
        else:
            unit = "N"
        
        ax.set_title(f'{target_names[i]}\nStd: {std_val:.2f} {unit}', fontsize=10)
        ax.set_xlabel(f'{target_names[i]} ({unit})', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'{title_prefix} Data Distribution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

# ========== CREATE DISTRIBUTION PLOTS ==========
# Create folder for plots named after sensor version
plots_dir = OUTPUT_DIR / f"sensor_v{VERSION_NUM}"
plots_dir.mkdir(parents=True, exist_ok=True)

print(f"\n========== CREATING DISTRIBUTION PLOTS ==========")
print(f"Plots will be saved in: {plots_dir}")

# Define target columns for position and force
position_cols = ['x', 'y']
force_cols = ['fx', 'fy', 'fz']

# Check which columns exist in the data
available_position_cols = [col for col in position_cols if col in combined_train.columns]
available_force_cols = [col for col in force_cols if col in combined_train.columns]

# Combine all targets
all_targets = available_position_cols + available_force_cols

# 1. Individual plots for each dataset (Training)
if all_targets:
    fig = plot_data_distribution(combined_train, all_targets, title_prefix="Training")
    fig.savefig(plots_dir / f'train_distribution_v{VERSION_NUM}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved training distribution plot")

# 4. Individual plots for each dataset (Validation)
if all_targets and len(combined_val) > 0:
    fig = plot_data_distribution(combined_val, all_targets, title_prefix="Validation")
    fig.savefig(plots_dir / f'validation_distribution_v{VERSION_NUM}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved validation distribution plot")

# 5. Individual plots for each dataset (Test)
if all_targets and len(combined_test) > 0:
    fig = plot_data_distribution(combined_test, all_targets, title_prefix="Test")
    fig.savefig(plots_dir / f'test_distribution_v{VERSION_NUM}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved test distribution plot")

print(f"\nAll distribution plots saved in: {plots_dir}")
print("Done.")
