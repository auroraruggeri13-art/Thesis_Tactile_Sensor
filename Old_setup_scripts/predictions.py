import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

# ========================= CONFIG ==========================
BASE = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab"

DATA_FILE  = r"c:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test 4605 - sensor v4\synchronized_events_4605.csv"
MODEL_FILE = os.path.join(BASE, r"models parameters\averaged models\lightgbm_sliding_window_model_v4.6.pkl")
SCALER_FILE = os.path.join(BASE, r"models parameters\averaged models\scaler_sliding_window_v4.6.pkl")

WINDOW_RADIUS = 10
BARO_COLS  = ["b1","b2","b3","b4","b5","b6"]
TARGET_COLS = ["x","y","fx","fy","fz"]
TIME_COL = "t"

# CRITICAL: Match training config
APPLY_DENOISING = True
DENOISE_WINDOW = 5

# ===================== LOAD MODEL & SCALER ==================
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(SCALER_FILE, "rb") as f:
    scaler = pickle.load(f)

# ========================= LOAD DATA =========================
df = pd.read_csv(DATA_FILE, skipinitialspace=True)
df.columns = df.columns.str.strip()
df = df.dropna(subset=[TIME_COL] + BARO_COLS + TARGET_COLS).reset_index(drop=True)


# ================== EXACT COPY FROM TRAINING ============
def maybe_denoise(df, baro_cols):
    """Apply denoising EXACTLY as in training."""
    if not APPLY_DENOISING:
        return df
    
    df = df.copy()
    win = DENOISE_WINDOW
    for col in baro_cols:
        df[col] = df[col].rolling(win, center=True).mean().bfill().ffill()
    return df


def build_features(df, window_radius):
    """Build features EXACTLY as in training - NO TRANSPOSE!"""
    df = df.copy().sort_values(TIME_COL).reset_index(drop=True)
    
    # Apply denoising FIRST (before derivatives)
    df = maybe_denoise(df, BARO_COLS)
    
    N = len(df)
    R = window_radius
    
    # Calculate derivatives EXACTLY as in training
    for col in BARO_COLS:
        d1 = df[col].diff()
        d2 = d1.diff()
        df[f"{col}_d1"] = d1.fillna(0.0)
        df[f"{col}_d2"] = d2.fillna(0.0)
    
    # Get numpy arrays
    baro_data = df[BARO_COLS].values  # (N, 6)
    d1_cols = [f"{col}_d1" for col in BARO_COLS]
    d2_cols = [f"{col}_d2" for col in BARO_COLS]
    d1_data = df[d1_cols].values      # (N, 6)
    d2_data = df[d2_cols].values      # (N, 6)
    
    M = N - 2 * R
    window_size = 2 * R + 1
    n_baro = len(BARO_COLS)
    
    X = np.zeros((M, n_baro * window_size * 3), dtype=float)
    y = df.loc[R:N-R-1, TARGET_COLS].values
    
    # Build features - NO .T TRANSPOSE! Use same order as training
    for i, center_idx in enumerate(range(R, N - R)):
        start = center_idx - R
        end = center_idx + R + 1
        
        # CRITICAL: Match training exactly (time-major, no transpose)
        baro_window = baro_data[start:end, :].flatten()  # NO .T!
        d1_window = d1_data[start:end, :].flatten()      # NO .T!
        d2_window = d2_data[start:end, :].flatten()      # NO .T!
        
        X[i, :] = np.concatenate([baro_window, d1_window, d2_window])
    
    return X, y, df.loc[R:N-R-1, TIME_COL].values


X_raw, y_true, t = build_features(df, WINDOW_RADIUS)

# ============== APPLY SCALER ==========
X = scaler.transform(X_raw)

# ======================== PREDICT ============================
if isinstance(model, list):
    preds = [m.predict(X) for m in model]
    y_pred = np.column_stack(preds)
else:
    y_pred = model.predict(X)

print(f"Predictions shape: {y_pred.shape}")
print(f"Ground truth shape: {y_true.shape}")

# Quick sanity check
for i, col in enumerate(TARGET_COLS):
    mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
    print(f"{col}: MAE = {mae:.3f}")

# ========================= PLOTTING ==========================

# ---- FIGURE 1: fx, fy, fz ----
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
force_cols = ["fx", "fy", "fz"]
force_idx  = [2, 3, 4]

for ax, col, idx in zip(axes, force_cols, force_idx):
    ax.plot(t, y_true[:, idx], label=f"{col} real", alpha=0.7)
    ax.plot(t, y_pred[:, idx], "--", label=f"{col} pred", alpha=0.7)
    ax.set_ylabel(col)
    ax.grid(alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Time [s]")
plt.tight_layout()
plt.show()


# ---- FIGURE 2: XY PATH inside 40×16 rectangle ----
fig2, ax2 = plt.subplots(figsize=(6, 5))

# Draw rectangle
rect_w, rect_h = 40, 16
ax2.add_patch(plt.Rectangle((-rect_w/2, -rect_h/2),
                            rect_w, rect_h,
                            fill=False, edgecolor="black", linewidth=1.5))

# Plot paths
ax2.plot(y_true[:,0], y_true[:,1], label="True", alpha=0.8, linewidth=2)
ax2.plot(y_pred[:,0], y_pred[:,1], "--", label="Pred", alpha=0.8, linewidth=2)

ax2.set_xlim(-rect_w/2, rect_w/2)
ax2.set_ylim(-rect_h/2, rect_h/2)
ax2.set_aspect("equal")
ax2.grid(alpha=0.3)
ax2.set_xlabel("x [mm]")
ax2.set_ylabel("y [mm]")
ax2.set_title("Predicted XY Path (40 × 16 mm)")
ax2.legend()

plt.tight_layout()
plt.show()


# ---- FIGURE 3: ERROR OVER TIME ----
errors = y_true - y_pred

fig3, axes3 = plt.subplots(len(TARGET_COLS), 1, figsize=(12, 10), sharex=True)
if len(TARGET_COLS) == 1:
    axes3 = [axes3]

for i, (ax, col) in enumerate(zip(axes3, TARGET_COLS)):
    error = errors[:, i]
    ax.plot(t, error, label=f"{col} error", alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Add mean and std lines
    mean_err = np.mean(error)
    std_err = np.std(error)
    ax.axhline(y=mean_err, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_err:.3f}')
    ax.axhline(y=mean_err + std_err, color='orange', linestyle=':', alpha=0.5)
    ax.axhline(y=mean_err - std_err, color='orange', linestyle=':', alpha=0.5, label=f'±1σ: {std_err:.3f}')
    
    unit = "mm" if col in ['x', 'y'] else "N"
    ax.set_ylabel(f"Error [{unit}]")
    ax.set_title(f"{col} Prediction Error")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

axes3[-1].set_xlabel("Time [s]")
plt.suptitle("Prediction Errors Over Time", fontsize=12, y=1.0)
plt.tight_layout()
plt.show()


# ---- FIGURE 4: ERROR DISTRIBUTION (HISTOGRAMS) ----
fig4, axes4 = plt.subplots(2, 3, figsize=(14, 8))
axes4 = axes4.flatten()

for i, col in enumerate(TARGET_COLS):
    ax = axes4[i]
    error = errors[:, i]
    
    # Histogram
    ax.hist(error, bins=50, alpha=0.7, edgecolor='black')
    
    # Statistics
    mean_err = np.mean(error)
    std_err = np.std(error)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    
    # Add vertical lines for mean and std
    ax.axvline(mean_err, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.3f}')
    ax.axvline(mean_err + std_err, color='orange', linestyle=':', linewidth=1.5)
    ax.axvline(mean_err - std_err, color='orange', linestyle=':', linewidth=1.5, label=f'Std: {std_err:.3f}')
    
    unit = "mm" if col in ['x', 'y'] else "N"
    ax.set_xlabel(f"Error [{unit}]")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{col}: MAE={mae:.3f}, RMSE={rmse:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Hide extra subplot if we have 5 targets
if len(TARGET_COLS) == 5:
    axes4[5].axis('off')

plt.suptitle("Error Distribution (Histograms)", fontsize=13)
plt.tight_layout()
plt.show()


# ---- FIGURE 5: 2D ERROR SCATTER (X-Y position errors) ----
if 'x' in TARGET_COLS and 'y' in TARGET_COLS:
    x_idx = TARGET_COLS.index('x')
    y_idx = TARGET_COLS.index('y')
    
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    
    # Calculate Euclidean error for each point
    euclidean_errors = np.sqrt(errors[:, x_idx]**2 + errors[:, y_idx]**2)
    
    # Scatter plot colored by Euclidean error
    scatter = ax5.scatter(y_true[:, x_idx], y_true[:, y_idx], 
                         c=euclidean_errors, cmap='hot', 
                         s=30, alpha=0.6, edgecolors='k', linewidths=0.5)
    
    # Draw rectangle
    rect_w, rect_h = 40, 16
    ax5.add_patch(plt.Rectangle((-rect_w/2, -rect_h/2),
                                rect_w, rect_h,
                                fill=False, edgecolor="blue", linewidth=2, linestyle='--'))
    
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Euclidean Position Error [mm]', rotation=270, labelpad=20)
    
    ax5.set_xlim(-rect_w/2, rect_w/2)
    ax5.set_ylim(-rect_h/2, rect_h/2)
    ax5.set_aspect("equal")
    ax5.set_xlabel("x [mm]")
    ax5.set_ylabel("y [mm]")
    ax5.set_title(f"Spatial Error Distribution\nMean Error: {np.mean(euclidean_errors):.3f} mm, Max Error: {np.max(euclidean_errors):.3f} mm")
    ax5.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()