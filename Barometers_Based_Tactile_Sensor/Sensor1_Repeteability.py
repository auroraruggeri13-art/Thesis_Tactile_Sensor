import sys
from pathlib import Path

# Make utils importable when running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.signal_utils import maybe_denoise, convert_sentinel_to_nan

# ===================== CONFIG =====================
TRAIN_SENSOR = 5.010
TEST_SENSOR  = 5.010

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab")
MODEL_PATH  = BASE_DIR / "models parameters" / "averaged models" / f"lightgbm_sliding_window_model_v{TRAIN_SENSOR:.3f}.pkl"
SCALER_PATH = BASE_DIR / "models parameters" / "averaged models" / f"scaler_sliding_window_v{TRAIN_SENSOR:.3f}.pkl"
TEST_PATH   = BASE_DIR / "train_validation_test_data" / f"test_data_v{TEST_SENSOR}.csv"

TIME_COL   = "t"
BARO_COLS  = ["b1","b2","b3","b4","b5","b6"]
TARGET_COLS = ["x","y", "fx", "fy", "fz"] # []

WINDOW_SIZE = 10          # MUST match training
APPLY_DENOISING = True    # MUST match training
DENOISE_WINDOW  = 5       # MUST match training
USE_SECOND_DERIVATIVE = True  # MUST match training
MAX_TIME_GAP = 0.05       # same default as your function

# Sentinel value conversion (MUST match data_organization and training)
CONVERT_SENTINEL_TO_NAN = True
NO_CONTACT_SENTINEL = -999.0

# Barometer Failure Analysis
RUN_BAROMETER_FAILURE_ANALYSIS = True  # Set to True to analyze impact of missing barometer inputs

OUT_DIR = BASE_DIR / "sensors repeatability"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / f"pred_train{TRAIN_SENSOR}_on_test{TEST_SENSOR}.csv"

# ===================== FEATURE BUILD (same as your training) =====================
def build_window_features(df, baro_cols, time_col, target_cols, window_size,
                          max_time_gap=0.05, use_second_derivative=False):
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df = maybe_denoise(df, baro_cols, apply_denoising=APPLY_DENOISING, denoise_window=DENOISE_WINDOW)

    for col in baro_cols:
        d1 = df[col].diff()
        df[f"{col}_d1"] = d1.fillna(0.0)
        if use_second_derivative:
            d2 = d1.diff()
            df[f"{col}_d2"] = d2.fillna(0.0)

    time_vals = df[time_col].values
    baro_data = df[baro_cols].values
    d1_data   = df[[f"{c}_d1" for c in baro_cols]].values
    if use_second_derivative:
        d2_data = df[[f"{c}_d2" for c in baro_cols]].values

    W = window_size
    X_list, y_list, valid_idx = [], [], []

    for current_idx in range(W, len(df)):
        start = current_idx - W
        end = current_idx + 1  # includes current

        if np.max(np.diff(time_vals[start:end])) > max_time_gap:
            continue

        baro_window = baro_data[start:end, :].flatten()
        d1_window   = d1_data[start:end, :].flatten()

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


def analyze_barometer_failure_impact(df_original, models, scaler, baro_cols, time_col, target_cols, 
                                      window_size, max_time_gap, use_second_derivative, out_dir):
    """
    Analyze the impact of missing barometer inputs on prediction accuracy.
    Tests various sensor failure scenarios by zeroing out barometer inputs.
    
    Args:
        df_original: Original test dataframe
        models: List of trained models (one per target)
        scaler: Fitted scaler
        baro_cols: List of barometer column names
        time_col: Name of time column
        target_cols: List of target column names
        window_size: Window size for features
        max_time_gap: Maximum time gap for valid windows
        use_second_derivative: Whether to use second derivatives
        out_dir: Output directory for results
    """
    print("\n" + "="*80)
    print("BAROMETER FAILURE ANALYSIS")
    print("="*80)
    
    # Convert sentinel values if needed
    if CONVERT_SENTINEL_TO_NAN:
        df_original = convert_sentinel_to_nan(df_original, target_cols, NO_CONTACT_SENTINEL)
    
    # Define failure scenarios
    scenarios = []
    
    # Baseline: all sensors working
    scenarios.append({
        'name': 'Baseline (All Sensors)',
        'zero_sensors': []
    })
    
    # Single sensor failures
    for sensor in baro_cols:
        scenarios.append({
            'name': f'Missing {sensor}',
            'zero_sensors': [sensor]
        })
    
    # Double sensor failures (adjacent pairs)
    for i in range(len(baro_cols) - 1):
        scenarios.append({
            'name': f'Missing {baro_cols[i]}+{baro_cols[i+1]}',
            'zero_sensors': [baro_cols[i], baro_cols[i+1]]
        })
    
    # Triple sensor failure (first three)
    scenarios.append({
        'name': f'Missing b1+b2+b3',
        'zero_sensors': ['b1', 'b2', 'b3']
    })
    
    # Half sensors (alternating)
    scenarios.append({
        'name': 'Missing b1+b3+b5',
        'zero_sensors': ['b1', 'b3', 'b5']
    })
    
    scenarios.append({
        'name': 'Missing b2+b4+b6',
        'zero_sensors': ['b2', 'b4', 'b6']
    })
    
    # Collect results
    results = []
    
    for scenario in scenarios:
        print(f"\nTesting: {scenario['name']}")
        
        # Create modified dataframe
        df_test = df_original.copy()
        
        # Zero out specified sensors
        for sensor in scenario['zero_sensors']:
            if sensor in df_test.columns:
                df_test[sensor] = 0.0
        
        # Build features and predict
        X_test, y_test, center_df_test = build_window_features(
            df_test, baro_cols, time_col, target_cols, window_size,
            max_time_gap=max_time_gap,
            use_second_derivative=use_second_derivative
        )
        
        X_test_scaled = scaler.transform(X_test)
        
        # Predict with each model
        preds = []
        for i, col in enumerate(target_cols):
            yp = np.asarray(models[i].predict(X_test_scaled)).reshape(-1)
            preds.append(yp)
        y_pred_test = np.column_stack(preds)
        
        # Calculate metrics for each target
        scenario_metrics = {'scenario': scenario['name'], 'n_zeros': len(scenario['zero_sensors'])}
        
        for i, col in enumerate(target_cols):
            yt = y_test[:, i]
            yp = y_pred_test[:, i]
            
            valid_mask = ~np.isnan(yt)
            if np.sum(valid_mask) > 0:
                mae = mean_absolute_error(yt[valid_mask], yp[valid_mask])
                rmse = float(np.sqrt(mean_squared_error(yt[valid_mask], yp[valid_mask])))
                r2 = r2_score(yt[valid_mask], yp[valid_mask])
                
                scenario_metrics[f'{col}_MAE'] = mae
                scenario_metrics[f'{col}_RMSE'] = rmse
                scenario_metrics[f'{col}_R2'] = r2
                
                print(f"  {col}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            else:
                scenario_metrics[f'{col}_MAE'] = np.nan
                scenario_metrics[f'{col}_RMSE'] = np.nan
                scenario_metrics[f'{col}_R2'] = np.nan
        
        results.append(scenario_metrics)
    
    # Create dedicated output directory
    analysis_dir = out_dir / "missing barometers analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = analysis_dir / f"barometer_failure_analysis_train{TRAIN_SENSOR}_test{TEST_SENSOR}.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved failure analysis results to: {results_csv}")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    
    # Plot 1: MAE comparison for each target
    fig, axes = plt.subplots(1, len(target_cols), figsize=(5*len(target_cols), 6))
    if len(target_cols) == 1:
        axes = [axes]
    
    fig.suptitle('Prediction Error vs Barometer Failures', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(target_cols):
        ax = axes[i]
        mae_values = results_df[f'{col}_MAE'].values
        scenario_names = results_df['scenario'].values
        
        # Color code: baseline green, single failures yellow, multiple failures red
        colors = []
        for n_zeros in results_df['n_zeros'].values:
            if n_zeros == 0:
                colors.append('green')
            elif n_zeros == 1:
                colors.append('orange')
            else:
                colors.append('red')
        
        bars = ax.barh(range(len(scenario_names)), mae_values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(scenario_names)))
        ax.set_yticklabels(scenario_names, fontsize=9)
        ax.set_xlabel('MAE', fontsize=11, fontweight='bold')
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add baseline reference line
        baseline_mae = mae_values[0]
        ax.axvline(baseline_mae, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plot1_path = analysis_dir / f"barometer_failure_mae_comparison_train{TRAIN_SENSOR}_test{TEST_SENSOR}.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"Saved MAE comparison plot to: {plot1_path}")
    plt.show()
    
    # Plot 2: Degradation percentage heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate percentage degradation relative to baseline
    degradation_matrix = []
    for i, row in results_df.iterrows():
        if i == 0:  # Skip baseline
            continue
        deg_row = []
        for col in target_cols:
            baseline_mae = results_df.loc[0, f'{col}_MAE']
            current_mae = row[f'{col}_MAE']
            if not np.isnan(baseline_mae) and baseline_mae > 0:
                degradation = ((current_mae - baseline_mae) / baseline_mae) * 100
                deg_row.append(degradation)
            else:
                deg_row.append(0)
        degradation_matrix.append(deg_row)
    
    degradation_matrix = np.array(degradation_matrix)
    scenario_labels = results_df['scenario'].values[1:]  # Exclude baseline
    
    im = ax.imshow(degradation_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(range(len(target_cols)))
    ax.set_xticklabels(target_cols, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(scenario_labels)))
    ax.set_yticklabels(scenario_labels, fontsize=9)
    
    # Add percentage values in cells
    for i in range(len(scenario_labels)):
        for j in range(len(target_cols)):
            text = ax.text(j, i, f'{degradation_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    ax.set_title('Prediction Error Degradation (% increase vs Baseline)', fontsize=14, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MAE Increase (%)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plot2_path = analysis_dir / f"barometer_failure_degradation_heatmap_train{TRAIN_SENSOR}_test{TEST_SENSOR}.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"Saved degradation heatmap to: {plot2_path}")
    plt.show()
    
    # ===== PLOT 3: TORNADO PLOT (Sensitivity Analysis) =====
    print("\nGenerating tornado plot...")
    
    # Extract single sensor failure results only
    single_failures = results_df[results_df['n_zeros'] == 1].copy()
    baseline_row = results_df[results_df['n_zeros'] == 0].iloc[0]
    
    # Create subplots for each target
    n_targets = len(target_cols)
    fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 8))
    if n_targets == 1:
        axes = [axes]
    
    fig.suptitle('Tornado Plot: Single Barometer Failure Impact', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(target_cols):
        ax = axes[i]
        
        # Calculate MAE increase for each sensor IN ORIGINAL ORDER (b1-b6)
        sensor_impacts = []
        sensor_names = []
        for sensor in baro_cols:  # Use original barometer order
            sensor_row = single_failures[single_failures['scenario'] == f'Missing {sensor}']
            if len(sensor_row) > 0:
                baseline_mae = baseline_row[f'{col}_MAE']
                failure_mae = sensor_row.iloc[0][f'{col}_MAE']
                mae_increase = failure_mae - baseline_mae
                
                sensor_impacts.append(mae_increase)
                sensor_names.append(sensor)
        
        # Keep original order (no sorting)
        sensor_impacts_ordered = sensor_impacts
        sensor_names_ordered = sensor_names
        
        # Color code by magnitude
        colors = ['#d62728' if x > np.mean(sensor_impacts) else '#ff7f0e' for x in sensor_impacts_ordered]
        
        # Horizontal bar chart
        y_pos = np.arange(len(sensor_names_ordered))
        bars = ax.barh(y_pos, sensor_impacts_ordered, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sensor_names_ordered, fontsize=11, fontweight='bold')
        ax.set_xlabel('MAE Increase', fontsize=12, fontweight='bold')
        ax.set_title(f'{col}', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axvline(0, color='black', linewidth=1)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, sensor_impacts_ordered)):
            ax.text(val, j, f' {val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    tornado_path = analysis_dir / f"tornado_plot_train{TRAIN_SENSOR}_test{TEST_SENSOR}.png"
    plt.savefig(tornado_path, dpi=300, bbox_inches='tight')
    print(f"Saved tornado plot to: {tornado_path}")
    plt.show()
    
    # ===== PLOT 4: SENSOR IMPORTANCE MATRIX =====
    print("\nGenerating sensor importance matrix...")
    
    # Build matrix: rows=sensors, columns=targets
    importance_matrix = np.zeros((len(baro_cols), len(target_cols)))
    
    for sensor_idx, sensor in enumerate(baro_cols):
        # Find the row for this single sensor failure
        sensor_row = single_failures[single_failures['scenario'] == f'Missing {sensor}']
        if len(sensor_row) > 0:
            sensor_row = sensor_row.iloc[0]
            for target_idx, col in enumerate(target_cols):
                baseline_mae = baseline_row[f'{col}_MAE']
                failure_mae = sensor_row[f'{col}_MAE']
                pct_increase = ((failure_mae - baseline_mae) / baseline_mae * 100) if baseline_mae > 0 else 0
                importance_matrix[sensor_idx, target_idx] = pct_increase
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
    
    ax.set_xticks(range(len(target_cols)))
    ax.set_xticklabels(target_cols, fontsize=13, fontweight='bold')
    ax.set_yticks(range(len(baro_cols)))
    ax.set_yticklabels(baro_cols, fontsize=13, fontweight='bold')
    
    # Add percentage values in cells
    for i in range(len(baro_cols)):
        for j in range(len(target_cols)):
            text = ax.text(j, i, f'{importance_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    ax.set_title('Sensor Importance Matrix\n(MAE % Increase When Sensor Missing)', 
                 fontsize=15, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MAE Increase (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    matrix_path = analysis_dir / f"sensor_importance_matrix_train{TRAIN_SENSOR}_test{TEST_SENSOR}.png"
    plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
    print(f"Saved sensor importance matrix to: {matrix_path}")
    plt.show()
    
    # ===== PLOT 5: SPIDER/RADAR PLOT =====
    print("\nGenerating spider/radar plot...")
    
    # Create radar chart
    num_sensors = len(baro_cols)
    # Start at π/2 (90 degrees, top of circle) and go clockwise
    angles = np.linspace(np.pi/2, np.pi/2 + 2 * np.pi, num_sensors, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    fig, axes = plt.subplots(1, len(target_cols), figsize=(6*len(target_cols), 6), 
                             subplot_kw=dict(projection='polar'))
    if len(target_cols) == 1:
        axes = [axes]
    
    fig.suptitle('Radar Plot: Sensor Failure Impact', fontsize=16, fontweight='bold')
    
    for target_idx, col in enumerate(target_cols):
        ax = axes[target_idx]
        
        # Set the direction to clockwise and start position at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_theta_offset(np.pi/2)  # Start at top
        
        # Get MAE values for each sensor failure
        mae_values = []
        for sensor in baro_cols:
            sensor_row = single_failures[single_failures['scenario'] == f'Missing {sensor}']
            if len(sensor_row) > 0:
                mae = sensor_row.iloc[0][f'{col}_MAE']
            else:
                mae = baseline_row[f'{col}_MAE']
            mae_values.append(mae)
        
        # Close the circle
        mae_values += mae_values[:1]
        
        # Recalculate angles for proper positioning
        sensor_angles = np.linspace(0, 2 * np.pi, num_sensors, endpoint=False).tolist()
        sensor_angles += sensor_angles[:1]
        
        # Plot
        ax.plot(sensor_angles, mae_values, 'o-', linewidth=2, label=col, color='#1f77b4', markersize=8)
        ax.fill(sensor_angles, mae_values, alpha=0.25, color='#1f77b4')
        
        # Add baseline reference circle
        baseline_mae = baseline_row[f'{col}_MAE']
        baseline_circle = [baseline_mae] * len(sensor_angles)
        ax.plot(sensor_angles, baseline_circle, '--', linewidth=2, color='green', alpha=0.7, label='Baseline')
        
        # Set labels
        ax.set_xticks(sensor_angles[:-1])
        ax.set_xticklabels(baro_cols, fontsize=11, fontweight='bold')
        ax.set_title(f'{col}', fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.15, 1.1))
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0
        ax.set_ylim(0, None)
    
    plt.tight_layout()
    radar_path = analysis_dir / f"radar_plot_train{TRAIN_SENSOR}_test{TEST_SENSOR}.png"
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    print(f"Saved radar plot to: {radar_path}")
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Most Critical Sensors")
    print("="*80)
    
    for col in target_cols:
        print(f"\n{col}:")
        # Sort scenarios by MAE for this target
        sorted_results = results_df.sort_values(f'{col}_MAE', ascending=False)
        print("  Worst scenarios:")
        for idx in range(min(3, len(sorted_results))):
            row = sorted_results.iloc[idx]
            mae = row[f'{col}_MAE']
            baseline_mae = results_df.loc[0, f'{col}_MAE']
            pct_increase = ((mae - baseline_mae) / baseline_mae * 100) if baseline_mae > 0 else 0
            print(f"    {row['scenario']}: MAE={mae:.4f} (+{pct_increase:.1f}%)")
    
    print("\n" + "="*80 + "\n")


# ===================== LOAD + PREDICT =====================
df = pd.read_csv(TEST_PATH, skipinitialspace=True)
df.columns = df.columns.str.strip()

# Convert sentinel values to NaN if enabled
if CONVERT_SENTINEL_TO_NAN:
    print("Converting sentinel values to NaN in target columns...")
    df = convert_sentinel_to_nan(df, TARGET_COLS, NO_CONTACT_SENTINEL)

X, y, center_df = build_window_features(
    df, BARO_COLS, TIME_COL, TARGET_COLS, WINDOW_SIZE,
    max_time_gap=MAX_TIME_GAP,
    use_second_derivative=USE_SECOND_DERIVATIVE
)

scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X)

models = joblib.load(MODEL_PATH)
if not isinstance(models, (list, tuple)):
    raise ValueError("Expected a list of 5 models (one per target), but got a single object.")

# predict each target with its corresponding model
pred_cols = []
preds = []
for i, col in enumerate(TARGET_COLS):
    yp = np.asarray(models[i].predict(X_scaled)).reshape(-1)
    preds.append(yp)
    pred_cols.append(f"{col}_pred")

y_pred = np.column_stack(preds)

# ===================== METRICS =====================
print("\nCross-sensor metrics (train -> test):")
for i, col in enumerate(TARGET_COLS):
    yt = y[:, i]
    yp = y_pred[:, i]
    
    # Handle NaN values from sentinel conversion
    valid_mask = ~np.isnan(yt)
    if np.sum(valid_mask) > 0:
        mae = mean_absolute_error(yt[valid_mask], yp[valid_mask])
        rmse = float(np.sqrt(mean_squared_error(yt[valid_mask], yp[valid_mask])))
        r2 = r2_score(yt[valid_mask], yp[valid_mask])
        print(f"{col}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f} | Valid: {np.sum(valid_mask)}/{len(valid_mask)}")
    else:
        print(f"{col}: No valid samples (all NaN)")

# ===================== SAVE =====================
out = center_df.copy()
for i, col in enumerate(pred_cols):
    out[col] = y_pred[:, i]

out.to_csv(OUT_CSV, index=False)
print(f"Saved predictions to: {OUT_CSV}")

# ===================== SENSITIVITY ESTIMATION =====================
print("\n========== Sensor Sensitivity Estimation ==========")

# Calculate sensitivity as the ratio of sensor signal change to physical quantity change
# Sensitivity = ΔSensor_Signal / ΔPhysical_Quantity

# Get barometer data for the samples
baro_data_samples = center_df[BARO_COLS].values

# For force sensitivity (focusing on fz)
if 'fz' in TARGET_COLS:
    fz_idx = TARGET_COLS.index('fz')
    fz_real = y[:, fz_idx]
    
    # Calculate range of force
    valid_fz_mask = ~np.isnan(fz_real)
    if np.sum(valid_fz_mask) > 0:
        fz_range = np.ptp(fz_real[valid_fz_mask])  # peak-to-peak (max - min)
        
        # Calculate range of barometer signals
        baro_ranges = []
        for i, col in enumerate(BARO_COLS):
            baro_range = np.ptp(baro_data_samples[valid_fz_mask, i])
            baro_ranges.append(baro_range)
            sensitivity = baro_range / fz_range if fz_range != 0 else 0
            print(f"  Force Sensitivity ({col}): {sensitivity:.4f} units/N (ΔSignal={baro_range:.2f}, ΔFz={fz_range:.2f} N)")
        
        avg_baro_range = np.mean(baro_ranges)
        avg_force_sensitivity = avg_baro_range / fz_range if fz_range != 0 else 0
        print(f"  Average Force Sensitivity: {avg_force_sensitivity:.4f} units/N")
        
        # Calculate relative sensitivity (normalized by signal magnitude)
        mean_baro_signal = np.mean([np.mean(np.abs(baro_data_samples[valid_fz_mask, i])) for i in range(len(BARO_COLS))])
        mean_fz = np.mean(np.abs(fz_real[valid_fz_mask]))
        relative_sensitivity = (avg_baro_range / mean_baro_signal) / (fz_range / mean_fz) if mean_baro_signal != 0 and mean_fz != 0 else 0
        print(f"  Relative Force Sensitivity: {relative_sensitivity:.4f}")

# For location sensitivity (x and y)
print("\n--- Location Sensitivity ---")
for loc_var in ['x', 'y']:
    if loc_var in TARGET_COLS:
        loc_idx = TARGET_COLS.index(loc_var)
        loc_real = y[:, loc_idx]
        
        # Calculate range of location
        valid_loc_mask = ~np.isnan(loc_real)
        if np.sum(valid_loc_mask) > 0:
            loc_range = np.ptp(loc_real[valid_loc_mask])  # peak-to-peak
            
            # Calculate range of barometer signals
            baro_ranges = []
            for i, col in enumerate(BARO_COLS):
                baro_range = np.ptp(baro_data_samples[valid_loc_mask, i])
                baro_ranges.append(baro_range)
                sensitivity = baro_range / loc_range if loc_range != 0 else 0
                print(f"  {loc_var.upper()} Location Sensitivity ({col}): {sensitivity:.4f} units/mm (ΔSignal={baro_range:.2f}, Δ{loc_var}={loc_range:.2f} mm)")
            
            avg_baro_range = np.mean(baro_ranges)
            avg_loc_sensitivity = avg_baro_range / loc_range if loc_range != 0 else 0
            print(f"  Average {loc_var.upper()} Location Sensitivity: {avg_loc_sensitivity:.4f} units/mm")
            
            # Calculate relative sensitivity (normalized by signal magnitude)
            mean_baro_signal = np.mean([np.mean(np.abs(baro_data_samples[valid_loc_mask, i])) for i in range(len(BARO_COLS))])
            mean_loc = np.mean(np.abs(loc_real[valid_loc_mask]))
            relative_sensitivity = (avg_baro_range / mean_baro_signal) / (loc_range / mean_loc) if mean_baro_signal != 0 and mean_loc != 0 else 0
            print(f"  Relative {loc_var.upper()} Location Sensitivity: {relative_sensitivity:.4f}\n")

# Calculate Signal-to-Noise Ratio (SNR) based on prediction errors
print("\n--- Signal-to-Noise Ratio (based on prediction accuracy) ---")
for i, col in enumerate(TARGET_COLS):
    yt = y[:, i]
    yp = y_pred[:, i]
    
    valid_mask = ~np.isnan(yt)
    if np.sum(valid_mask) > 0:
        yt_valid = yt[valid_mask]
        yp_valid = yp[valid_mask]
        
        signal_power = np.var(yt_valid)
        noise_power = np.var(yt_valid - yp_valid)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else np.inf
        
        print(f"  {col} SNR: {snr:.2f} dB (Signal Var={signal_power:.4f}, Noise Var={noise_power:.4f})")

print("\n" + "="*60 + "\n")

# ===================== PLOT REAL VS PREDICTED =====================
n_targets = len(TARGET_COLS)
n_cols = min(5, n_targets)  # Max 5 plots per row
n_rows = (n_targets + n_cols - 1) // n_cols  # Calculate needed rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
if n_targets == 1:
    axes = np.array([axes])
axes = axes.flatten() if n_targets > 1 else axes

fig.suptitle('Predicted vs Actual (LightGBM)', fontsize=16, fontweight='bold', y=1.02)

for i, col in enumerate(TARGET_COLS):
    ax = axes[i]
    yt = y[:, i]
    yp = y_pred[:, i]
    
    # Filter out NaN values from ground truth
    valid_mask = ~np.isnan(yt)
    yt_valid = yt[valid_mask]
    yp_valid = yp[valid_mask]
    
    # Scatter plot with smaller points and transparency
    ax.scatter(yt_valid, yp_valid, alpha=0.4, s=8, color='steelblue', edgecolors='none')
    
    # Perfect prediction line
    if len(yt_valid) > 0:
        min_val = min(yt_valid.min(), yp_valid.min())
        max_val = max(yt_valid.max(), yp_valid.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, alpha=0.8)
    
    # Metrics
    if len(yt_valid) > 0:
        mae = mean_absolute_error(yt_valid, yp_valid)
        r2 = r2_score(yt_valid, yp_valid)
    else:
        mae, r2 = np.nan, np.nan
    
    ax.set_xlabel('Actual', fontsize=10)
    ax.set_ylabel('Predicted', fontsize=10)
    ax.set_title(f'{col}\nMAE: {mae:.2f} | R²: {r2:.3f}', fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Add light background
    ax.set_facecolor('#f8f9fa')

# Hide unused subplots if any
for j in range(n_targets, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plot_path = OUT_DIR / f"real_vs_pred_train{TRAIN_SENSOR}_on_test{TEST_SENSOR}.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {plot_path}")
plt.show()

# ===================== PLOT TIME SERIES WITH ERROR =====================
n_targets = len(TARGET_COLS)
fig, axes = plt.subplots(n_targets, 1, figsize=(10, 2.8*n_targets))
if n_targets == 1:
    axes = [axes]

time_values = center_df[TIME_COL].values

for i, col in enumerate(TARGET_COLS):
    ax = axes[i]
    yt = y[:, i]
    yp = y_pred[:, i]
    error = yp - yt
    
    # Create twin axis for error
    ax2 = ax.twinx()
    
    # Plot real and predicted
    ax.plot(time_values, yt, 'b-', linewidth=3, label='Real', alpha=0.8)
    ax.plot(time_values, yp, 'r--', linewidth=3, label='Predicted', alpha=0.8)
    
    # Plot error on secondary axis
    ax2.plot(time_values, error, 'g-', linewidth=1, label='Error', alpha=0.6)
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel(f'{col}', fontsize=11, color='black')
    ax2.set_ylabel('Error (Predicted - Real)', fontsize=11, color='green')
    
    # Metrics - handle NaN values
    valid_mask = ~np.isnan(yt)
    if np.sum(valid_mask) > 0:
        mae = mean_absolute_error(yt[valid_mask], yp[valid_mask])
        rmse = float(np.sqrt(mean_squared_error(yt[valid_mask], yp[valid_mask])))
        r2 = r2_score(yt[valid_mask], yp[valid_mask])
    else:
        mae, rmse, r2 = np.nan, np.nan, np.nan
    
    ax.set_title(f'{col}: R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}', fontsize=12, fontweight='bold')
    
    # Legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='green')

plt.tight_layout()
timeseries_path = OUT_DIR / f"timeseries_train{TRAIN_SENSOR}_on_test{TEST_SENSOR}.png"
plt.savefig(timeseries_path, dpi=300, bbox_inches='tight')
print(f"Saved time series plot to: {timeseries_path}")
plt.show()

# ===================== PLOT XY PATH (REAL VS PREDICTED) =====================
# Only plot if both x and y are in targets
if 'x' in TARGET_COLS and 'y' in TARGET_COLS:
    fig, ax = plt.subplots(figsize=(10, 10))

    # Extract x and y positions (filtered data)
    x_idx = TARGET_COLS.index('x')
    y_idx = TARGET_COLS.index('y')

    x_real = y[:, x_idx]
    y_real = y[:, y_idx]
    x_pred = y_pred[:, x_idx]
    y_pred_vals = y_pred[:, y_idx]

    # Filter out NaN values from real path
    valid_mask_real = ~(np.isnan(x_real) | np.isnan(y_real))
    valid_mask_pred = ~(np.isnan(x_pred) | np.isnan(y_pred_vals))

    # Plot real path - break into segments where there are gaps
    if np.sum(valid_mask_real) > 0:
        # Find indices of valid points
        valid_indices_real = np.where(valid_mask_real)[0]
        # Find breaks in the sequence (gaps > 1 index)
        breaks_real = np.where(np.diff(valid_indices_real) > 1)[0] + 1
        # Split into continuous segments
        segments_real = np.split(valid_indices_real, breaks_real)
        
        for seg_idx, segment in enumerate(segments_real):
            if len(segment) > 0:
                label = 'Real Path' if seg_idx == 0 else None
                ax.plot(x_real[segment], y_real[segment], 'b-', linewidth=2, label=label, alpha=0.7)
        
        # Mark start point
        first_valid_idx = valid_indices_real[0]
        ax.scatter(x_real[first_valid_idx], y_real[first_valid_idx], c='blue', s=200, marker='o', 
                   edgecolors='black', linewidths=2, label='Start (Real)', zorder=5)

    # Plot predicted path - break into segments where there are gaps
    if np.sum(valid_mask_pred) > 0:
        # Find indices of valid points
        valid_indices_pred = np.where(valid_mask_pred)[0]
        # Find breaks in the sequence (gaps > 1 index)
        breaks_pred = np.where(np.diff(valid_indices_pred) > 1)[0] + 1
        # Split into continuous segments
        segments_pred = np.split(valid_indices_pred, breaks_pred)
        
        for seg_idx, segment in enumerate(segments_pred):
            if len(segment) > 0:
                label = 'Predicted Path' if seg_idx == 0 else None
                ax.plot(x_pred[segment], y_pred_vals[segment], 'r--', linewidth=2, label=label, alpha=0.7)
        
        # Mark start point
        first_valid_idx = valid_indices_pred[0]
        ax.scatter(x_pred[first_valid_idx], y_pred_vals[first_valid_idx], c='red', s=200, marker='s', 
                   edgecolors='black', linewidths=2, label='Start (Pred)', zorder=5)
    ax.scatter(x_pred[0], y_pred_vals[0], c='red', s=200, marker='s', edgecolors='black', linewidths=2, label='Start (Pred)', zorder=5)

    # Labels and formatting
    ax.set_xlabel('X Position (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (mm)', fontsize=14, fontweight='bold')
    ax.set_title(f'Contact Position Path: Real vs Predicted\nTrain={TRAIN_SENSOR}  Test={TEST_SENSOR}', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    path_plot = OUT_DIR / f"xy_path_train{TRAIN_SENSOR}_on_test{TEST_SENSOR}.png"
    plt.savefig(path_plot, dpi=300, bbox_inches='tight')
    print(f"Saved XY path plot to: {path_plot}")
    plt.show()
else:
    print("\nSkipping XY path plot (x or y not in targets)")

# ===================== BAROMETER FAILURE ANALYSIS (OPTIONAL) =====================
if RUN_BAROMETER_FAILURE_ANALYSIS:
    analyze_barometer_failure_impact(
        df_original=df,  # Original unmodified dataframe
        models=models,
        scaler=scaler,
        baro_cols=BARO_COLS,
        time_col=TIME_COL,
        target_cols=TARGET_COLS,
        window_size=WINDOW_SIZE,
        max_time_gap=MAX_TIME_GAP,
        use_second_derivative=USE_SECOND_DERIVATIVE,
        out_dir=OUT_DIR
    )
else:
    print("\nSkipping barometer failure analysis (RUN_BAROMETER_FAILURE_ANALYSIS=False)")
