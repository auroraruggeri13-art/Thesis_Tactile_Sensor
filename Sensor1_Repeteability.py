from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================== CONFIG =====================
TRAIN_SENSOR = 5.103 
TEST_SENSOR  = 5.103 

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab")
MODEL_PATH  = BASE_DIR / "models parameters" / "averaged models" / f"lightgbm_sliding_window_model_v{TRAIN_SENSOR:.2f}.pkl"
SCALER_PATH = BASE_DIR / "models parameters" / "averaged models" / f"scaler_sliding_window_v{TRAIN_SENSOR:.2f}.pkl"
TEST_PATH   = BASE_DIR / "train_validation_test_data" / f"test_data_v{TEST_SENSOR}.csv"

TIME_COL   = "t"
BARO_COLS  = ["b1","b2","b3","b4","b5","b6"]
TARGET_COLS = ["x","y","fz","tz"] # []

WINDOW_SIZE = 10          # MUST match training
APPLY_DENOISING = True    # MUST match training
DENOISE_WINDOW  = 5       # MUST match training
USE_SECOND_DERIVATIVE = False  # MUST match training
MAX_TIME_GAP = 0.05       # same default as your function

# Sentinel value conversion (MUST match data_organization and training)
CONVERT_SENTINEL_TO_NAN = True
NO_CONTACT_SENTINEL = -999.0

OUT_DIR = BASE_DIR / "sensors repeatability"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / f"pred_train{TRAIN_SENSOR}_on_test{TEST_SENSOR}.csv"

# ===================== FEATURE BUILD (same as your training) =====================
def convert_sentinel_to_nan(df, target_cols, sentinel=-999.0):
    """Convert sentinel values to NaN in target columns."""
    df = df.copy()
    n_converted = 0
    
    for col in target_cols:
        if col in df.columns:
            sentinel_mask = df[col] == sentinel
            n_col_converted = sentinel_mask.sum()
            if n_col_converted > 0:
                df.loc[sentinel_mask, col] = np.nan
                n_converted += n_col_converted
    
    if n_converted > 0:
        pct = 100 * n_converted / (len(df) * len(target_cols))
        print(f"Converted {n_converted} sentinel values ({sentinel}) to NaN ({pct:.2f}% of target values)")
    
    return df


def maybe_denoise(df, baro_cols):
    if not APPLY_DENOISING:
        return df
    df = df.copy()
    for col in baro_cols:
        df[col] = df[col].rolling(DENOISE_WINDOW, center=True).mean().bfill().ffill()
    return df

def build_window_features(df, baro_cols, time_col, target_cols, window_size,
                          max_time_gap=0.05, use_second_derivative=False):
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df = maybe_denoise(df, baro_cols)

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

    # Extract x and y positions
    x_idx = TARGET_COLS.index('x')
    y_idx = TARGET_COLS.index('y')

    x_real = y[:, x_idx]
    y_real = y[:, y_idx]
    x_pred = y_pred[:, x_idx]
    y_pred_vals = y_pred[:, y_idx]

    # Filter out NaN values from real path
    valid_mask_real = ~(np.isnan(x_real) | np.isnan(y_real))
    x_real_valid = x_real[valid_mask_real]
    y_real_valid = y_real[valid_mask_real]

    # Plot real path (only valid samples)
    if len(x_real_valid) > 0:
        ax.plot(x_real_valid, y_real_valid, 'b-', linewidth=2, label='Real Path', alpha=0.7)
        ax.scatter(x_real_valid[0], y_real_valid[0], c='blue', s=200, marker='o', edgecolors='black', linewidths=2, label='Start (Real)', zorder=5)

    # Plot predicted path
    ax.plot(x_pred, y_pred_vals, 'r--', linewidth=2, label='Predicted Path', alpha=0.7)
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
