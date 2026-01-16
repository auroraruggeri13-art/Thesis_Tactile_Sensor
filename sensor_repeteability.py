from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================== CONFIG =====================
TRAIN_SENSOR = 4.8
TEST_SENSOR  = 4.9

BASE_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab")
MODEL_PATH  = BASE_DIR / "models parameters" / "averaged models" / f"lightgbm_sliding_window_model_v{TRAIN_SENSOR:.2f}.pkl"
SCALER_PATH = BASE_DIR / "models parameters" / "averaged models" / f"scaler_sliding_window_v{TRAIN_SENSOR:.2f}.pkl"
TEST_PATH   = BASE_DIR / "train_validation_test_data" / f"test_data_v{TEST_SENSOR}.csv"

TIME_COL   = "t"
BARO_COLS  = ["b1","b2","b3","b4","b5","b6"]
TARGET_COLS = ["x","y","fx","fy","fz"]

WINDOW_SIZE = 10          # MUST match training
APPLY_DENOISING = True    # MUST match training
DENOISE_WINDOW  = 5       # MUST match training
USE_SECOND_DERIVATIVE = False  # MUST match training
MAX_TIME_GAP = 0.05       # same default as your function

OUT_DIR = BASE_DIR / "sensors repeatability"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / f"pred_train{TRAIN_SENSOR}_on_test{TEST_SENSOR}.csv"

# ===================== FEATURE BUILD (same as your training) =====================
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
print("Cross-sensor metrics (train -> test):")
for i, col in enumerate(TARGET_COLS):
    yt = y[:, i]
    yp = y_pred[:, i]
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = r2_score(yt, yp)
    print(f"{col}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

# ===================== SAVE =====================
out = center_df.copy()
for i, col in enumerate(pred_cols):
    out[col] = y_pred[:, i]

out.to_csv(OUT_CSV, index=False)
print(f"Saved predictions to: {OUT_CSV}")

# ===================== PLOT REAL VS PREDICTED =====================
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, col in enumerate(TARGET_COLS):
    ax = axes[i]
    yt = y[:, i]
    yp = y_pred[:, i]
    
    # Scatter plot
    ax.scatter(yt, yp, alpha=0.5, s=10, label='Predictions')
    
    # Perfect prediction line
    min_val = min(yt.min(), yp.min())
    max_val = max(yt.max(), yp.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Metrics
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = r2_score(yt, yp)
    
    ax.set_xlabel(f'Real {col}', fontsize=12)
    ax.set_ylabel(f'Predicted {col}', fontsize=12)
    ax.set_title(f'{col}: R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plot_path = OUT_DIR / f"real_vs_pred_train{TRAIN_SENSOR}_on_test{TEST_SENSOR}.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {plot_path}")
plt.show()

# ===================== PLOT TIME SERIES WITH ERROR =====================
fig, axes = plt.subplots(5, 1, figsize=(16, 14))

time_values = center_df[TIME_COL].values

for i, col in enumerate(TARGET_COLS):
    ax = axes[i]
    yt = y[:, i]
    yp = y_pred[:, i]
    error = yp - yt
    
    # Create twin axis for error
    ax2 = ax.twinx()
    
    # Plot real and predicted
    ax.plot(time_values, yt, 'b-', linewidth=1.5, label='Real', alpha=0.8)
    ax.plot(time_values, yp, 'r--', linewidth=1.5, label='Predicted', alpha=0.8)
    
    # Plot error on secondary axis
    ax2.plot(time_values, error, 'g-', linewidth=1, label='Error', alpha=0.6)
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel(f'{col}', fontsize=11, color='black')
    ax2.set_ylabel('Error (Predicted - Real)', fontsize=11, color='green')
    
    # Metrics
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = r2_score(yt, yp)
    
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
