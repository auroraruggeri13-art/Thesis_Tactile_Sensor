#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Cross-Sensor Reproducibility Analysis
Easy-to-use script for testing model generalization across sensor versions
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================
# Change these to match your sensors and setup
SENSOR_VERSIONS = [5.13, 5.14, 5.15]  # Which sensors to compare
DATA_SPLIT = 'test'  # 'test' or 'validation'
ACCEPTABLE_DEGRADATION = 10  # Maximum acceptable performance drop (%)

# Paths
BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab")
MODEL_DIR = BASE_DIR / "models parameters" / "averaged models"
DATA_DIR = BASE_DIR / "train_validation_test_data"
OUT_DIR = BASE_DIR / "sensors repeatability"

# Data columns
TIME_COL = "t"
BARO_COLS = ["b1", "b2", "b3", "b4", "b5", "b6"]
TARGET_COLS = ["x", "y", "fx", "fy", "fz"]

# Feature engineering (must match training!)
WINDOW_SIZE = 10
APPLY_DENOISING = True
DENOISE_WINDOW = 5
USE_SECOND_DERIVATIVE = False
MAX_TIME_GAP = 0.05

# ===================== HELPER FUNCTIONS =====================

def maybe_denoise(df, baro_cols):
    """Apply rolling mean denoising if enabled"""
    if not APPLY_DENOISING:
        return df
    df = df.copy()
    for col in baro_cols:
        df[col] = df[col].rolling(DENOISE_WINDOW, center=True).mean().bfill().ffill()
    return df


def build_window_features(df, baro_cols, time_col, target_cols, window_size,
                          max_time_gap=0.05, use_second_derivative=False):
    """Build sliding window features (identical to training code)"""
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df = maybe_denoise(df, baro_cols)

    # Compute derivatives
    for col in baro_cols:
        d1 = df[col].diff()
        df[f"{col}_d1"] = d1.fillna(0.0)
        if use_second_derivative:
            d2 = d1.diff()
            df[f"{col}_d2"] = d2.fillna(0.0)

    time_vals = df[time_col].values
    baro_data = df[baro_cols].values
    d1_data = df[[f"{c}_d1" for c in baro_cols]].values
    if use_second_derivative:
        d2_data = df[[f"{c}_d2" for c in baro_cols]].values

    W = window_size
    X_list, y_list, valid_idx = [], [], []

    for current_idx in range(W, len(df)):
        start = current_idx - W
        end = current_idx + 1

        if np.max(np.diff(time_vals[start:end])) > max_time_gap:
            continue

        baro_window = baro_data[start:end, :].flatten()
        d1_window = d1_data[start:end, :].flatten()

        if use_second_derivative:
            d2_window = d2_data[start:end, :].flatten()
            X_list.append(np.concatenate([baro_window, d1_window, d2_window]))
        else:
            X_list.append(np.concatenate([baro_window, d1_window]))

        y_list.append(df.loc[current_idx, target_cols].values)
        valid_idx.append(current_idx)

    X = np.asarray(X_list)
    y = np.asarray(y_list)
    center_df = df.iloc[valid_idx].reset_index(drop=True)
    return X, y, center_df


def predict_and_evaluate(models, scaler, X_test, y_test, target_cols):
    """Run predictions and compute metrics"""
    X_scaled = scaler.transform(X_test)
    
    # Predict each target
    preds = []
    for i, col in enumerate(target_cols):
        yp = np.asarray(models[i].predict(X_scaled)).reshape(-1)
        preds.append(yp)
    y_pred = np.column_stack(preds)
    
    # Compute metrics
    metrics = {}
    for i, col in enumerate(target_cols):
        yt = y_test[:, i]
        yp = y_pred[:, i]
        
        metrics[col] = {
            'mae': mean_absolute_error(yt, yp),
            'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
            'r2': r2_score(yt, yp)
        }
    
    return y_pred, metrics


# ===================== MAIN ANALYSIS =====================

def main():
    # Setup output directory
    sensor_names = "_".join([f"{s:.2f}" for s in SENSOR_VERSIONS])
    run_dir = OUT_DIR / f"sensors_{sensor_names}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CROSS-SENSOR REPRODUCIBILITY ANALYSIS")
    print("="*80)
    print(f"Sensors: {SENSOR_VERSIONS}")
    print(f"Data split: {DATA_SPLIT}")
    print(f"Acceptable degradation: {ACCEPTABLE_DEGRADATION}%")
    print(f"Output: {run_dir}")
    print()
    
    # ===== STEP 1: Test All Combinations =====
    print("Testing all sensor combinations...")
    print("-"*80)
    
    all_results = []
    
    for train_sensor in SENSOR_VERSIONS:
        # Load model
        model_path = MODEL_DIR / f"lightgbm_sliding_window_model_v{train_sensor:.2f}.pkl"
        scaler_path = MODEL_DIR / f"scaler_sliding_window_v{train_sensor:.2f}.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            print(f"⚠ Model not found for sensor {train_sensor}, skipping...")
            continue
        
        models = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        for test_sensor in SENSOR_VERSIONS:
            # Load test data
            test_path = DATA_DIR / f"{DATA_SPLIT}_data_v{test_sensor}.csv"
            
            if not test_path.exists():
                print(f"⚠ Data not found for sensor {test_sensor}, skipping...")
                continue
            
            df = pd.read_csv(test_path, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            
            # Build features
            X_test, y_test, center_df = build_window_features(
                df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
                max_time_gap=MAX_TIME_GAP,
                use_second_derivative=USE_SECOND_DERIVATIVE
            )
            
            # Predict
            y_pred, metrics = predict_and_evaluate(models, scaler, X_test, y_test, TARGET_COLS)
            
            # Print results
            same_sensor = train_sensor == test_sensor
            label = "BASELINE" if same_sensor else "cross-sensor"
            print(f"\nTrain {train_sensor:.2f} → Test {test_sensor:.2f} ({label})")
            
            for target in TARGET_COLS:
                m = metrics[target]
                all_results.append({
                    'train_sensor': train_sensor,
                    'test_sensor': test_sensor,
                    'target': target,
                    'same_sensor': same_sensor,
                    **m
                })
                print(f"  {target:>3}: R²={m['r2']:6.3f}  MAE={m['mae']:6.3f}")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(run_dir / "all_results.csv", index=False)
    print(f"\n✓ Results saved to: {run_dir / 'all_results.csv'}")
    
    # ===== STEP 2: Calculate Degradation =====
    print("\n" + "="*80)
    print("PERFORMANCE DEGRADATION ANALYSIS")
    print("="*80)
    
    degradation_results = []
    
    for train_sensor in SENSOR_VERSIONS:
        for target in TARGET_COLS:
            # Get baseline (same-sensor)
            baseline = results_df[
                (results_df['train_sensor'] == train_sensor) &
                (results_df['test_sensor'] == train_sensor) &
                (results_df['target'] == target)
            ]
            
            if baseline.empty:
                continue
            
            baseline_r2 = baseline['r2'].values[0]
            
            # Compare to cross-sensor
            for test_sensor in SENSOR_VERSIONS:
                if train_sensor == test_sensor:
                    continue
                
                cross = results_df[
                    (results_df['train_sensor'] == train_sensor) &
                    (results_df['test_sensor'] == test_sensor) &
                    (results_df['target'] == target)
                ]
                
                if cross.empty:
                    continue
                
                cross_r2 = cross['r2'].values[0]
                
                # Calculate degradation
                if baseline_r2 > 0:
                    degradation_pct = 100 * (1 - cross_r2 / baseline_r2)
                else:
                    degradation_pct = float('inf')
                
                passes = degradation_pct <= ACCEPTABLE_DEGRADATION
                status = "✓ PASS" if passes else "✗ FAIL"
                
                degradation_results.append({
                    'train_sensor': train_sensor,
                    'test_sensor': test_sensor,
                    'target': target,
                    'baseline_r2': baseline_r2,
                    'cross_r2': cross_r2,
                    'degradation_%': degradation_pct,
                    'passes': passes
                })
                
                print(f"{train_sensor:.2f}→{test_sensor:.2f} [{target}]: "
                      f"degradation={degradation_pct:6.2f}% {status}")
    
    degradation_df = pd.DataFrame(degradation_results)
    degradation_df.to_csv(run_dir / "degradation.csv", index=False)
    
    # ===== STEP 3: Summary =====
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_tests = len(degradation_df)
    passed = degradation_df['passes'].sum()
    pass_rate = 100 * passed / total_tests if total_tests > 0 else 0
    
    print(f"\nOverall: {passed}/{total_tests} tests passed ({pass_rate:.1f}%)")
    
    print("\nBy target:")
    for target in TARGET_COLS:
        target_df = degradation_df[degradation_df['target'] == target]
        target_passed = target_df['passes'].sum()
        target_total = len(target_df)
        avg_deg = target_df['degradation_%'].mean()
        
        print(f"  {target}: {target_passed}/{target_total} passed, "
              f"avg degradation = {avg_deg:.1f}%")
    
    # ===== STEP 4: Create Heatmap =====
    print("\n" + "="*80)
    print("GENERATING HEATMAP")
    print("="*80)
    
    # Average R² across all targets for each combination
    heatmap_data = results_df.groupby(['train_sensor', 'test_sensor'])['r2'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, center=0.85, cbar_kws={'label': 'R²'},
                linewidths=1, linecolor='black', ax=ax)
    
    ax.set_title('Cross-Sensor Performance (Average R² across all targets)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Test Sensor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Train Sensor', fontsize=12, fontweight='bold')
    
    # Highlight diagonal
    for i in range(len(heatmap_data)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                   edgecolor='blue', lw=3))
    
    plt.tight_layout()
    plt.savefig(run_dir / "heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved to: {run_dir / 'heatmap.png'}")
    
    # ===== STEP 5: Create Degradation Bar Chart =====
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot degradation for each target
    targets = degradation_df['target'].unique()
    x = np.arange(len(targets))
    width = 0.8 / len(degradation_df['train_sensor'].unique())
    
    for i, train_sensor in enumerate(SENSOR_VERSIONS):
        train_data = degradation_df[degradation_df['train_sensor'] == train_sensor]
        avg_by_target = train_data.groupby('target')['degradation_%'].mean()
        
        positions = x + (i - len(SENSOR_VERSIONS)/2 + 0.5) * width
        bars = ax.bar(positions, avg_by_target.values, width,
                     label=f'Train: {train_sensor:.2f}', alpha=0.8)
        
        # Color bars by pass/fail
        for j, (bar, deg) in enumerate(zip(bars, avg_by_target.values)):
            if deg <= ACCEPTABLE_DEGRADATION:
                bar.set_color('green')
            else:
                bar.set_color('red')
    
    ax.axhline(y=ACCEPTABLE_DEGRADATION, color='black', linestyle='--',
              linewidth=2, label=f'{ACCEPTABLE_DEGRADATION}% threshold')
    ax.set_xlabel('Target', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average R² Degradation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Degradation by Target', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(run_dir / "degradation_by_target.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Degradation chart saved to: {run_dir / 'degradation_by_target.png'}")
    
    # ===== Save Summary Report =====
    summary_path = run_dir / "summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CROSS-SENSOR REPRODUCIBILITY SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Sensors tested: {SENSOR_VERSIONS}\n")
        f.write(f"Data split: {DATA_SPLIT}\n")
        f.write(f"Threshold: {ACCEPTABLE_DEGRADATION}%\n\n")
        f.write(f"Overall pass rate: {pass_rate:.1f}% ({passed}/{total_tests})\n\n")
        f.write("By target:\n")
        for target in TARGET_COLS:
            target_df = degradation_df[degradation_df['target'] == target]
            target_passed = target_df['passes'].sum()
            target_total = len(target_df)
            avg_deg = target_df['degradation_%'].mean()
            f.write(f"  {target}: {target_passed}/{target_total} passed "
                   f"(avg degradation: {avg_deg:.1f}%)\n")
        
        if pass_rate >= 80:
            f.write("\n[PASS] CONCLUSION: Sensors show good reproducibility\n")
            f.write("  A single model can be used across sensors.\n")
        elif pass_rate >= 50:
            f.write("\n[WARNING] CONCLUSION: Moderate reproducibility\n")
            f.write("  Consider transfer learning or sensor-specific fine-tuning.\n")
        else:
            f.write("\n[FAIL] CONCLUSION: Poor reproducibility\n")
            f.write("  Sensor-specific models are required.\n")
    
    print(f"✓ Summary saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll files saved to: {run_dir}")
    print("\nGenerated files:")
    print("  - all_results.csv (all metrics)")
    print("  - degradation.csv (degradation analysis)")
    print("  - heatmap.png (visual overview)")
    print("  - degradation_by_target.png (bar chart)")
    print("  - summary.txt (text summary)")


if __name__ == "__main__":
    main()