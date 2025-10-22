import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def plot_all_targets_summary(y_true, y_pred, target_names):
    """Creates a grid of scatter plots comparing actual vs. predicted values for each target."""
    n_targets = y_true.shape[1]
    n_rows = (n_targets + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    for i in range(n_targets):
        ax = axes[i]
        true_vals, pred_vals = y_true[:, i], y_pred[:, i]
        ax.scatter(true_vals, pred_vals, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
        min_val, max_val = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
        lims = [min_val - (max_val - min_val) * 0.05, max_val + (max_val - min_val) * 0.05]
        ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=1.5)
        mae, r2 = mean_absolute_error(true_vals, pred_vals), r2_score(true_vals, pred_vals)
        ax.set_title(f'{target_names[i]}\nMAE: {mae:.2f} | R²: {r2:.3f}', fontsize=10)
        ax.set_xlabel('Actual', fontsize=8); ax.set_ylabel('Predicted', fontsize=8)
        ax.grid(True, alpha=0.3); ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
    for i in range(n_targets, len(axes)):
        axes[i].axis('off')
    plt.tight_layout(); plt.show()

def calculate_grouped_rmse(y_true, y_pred):
    """Calculates and prints specialized RMSE metrics for location and force."""
    contact_location_true, contact_location_pred = y_true[:, :2], y_pred[:, :2]
    force_vector_true, force_vector_pred = y_true[:, 2:5], y_pred[:, 2:5]
    
    contact_location_errors = contact_location_true - contact_location_pred
    contact_location_rmse = np.sqrt(np.mean(contact_location_errors ** 2))
    force_vector_errors = force_vector_true - force_vector_pred
    force_vector_rmse = np.sqrt(np.mean(force_vector_errors ** 2))
    
    contact_euclidean_errors = np.sqrt(np.sum(contact_location_errors ** 2, axis=1))
    contact_euclidean_rmse = np.sqrt(np.mean(contact_euclidean_errors ** 2))
    force_euclidean_errors = np.sqrt(np.sum(force_vector_errors ** 2, axis=1))
    force_euclidean_rmse = np.sqrt(np.mean(force_euclidean_errors ** 2))
    
    print("\n" + "="*70 + "\nGROUPED RMSE METRICS\n" + "="*70)
    print(f"\nContact Location (x, y):\n  - Component-wise RMSE: {contact_location_rmse:.4f} mm\n  - Euclidean RMSE:      {contact_euclidean_rmse:.4f} mm\n  - Mean error distance: {np.mean(contact_euclidean_errors):.4f} mm")
    print(f"\n3-DOF Force Vector (fx, fy, fz):\n  - Component-wise RMSE: {force_vector_rmse:.4f} N\n  - Euclidean RMSE:      {force_euclidean_rmse:.4f} N\n  - Mean error magnitude: {np.mean(force_euclidean_errors):.4f} N")


if __name__ == "__main__":
    # --- 1. Load, Clean, and Combine Data ---
    DATA_DIRECTORY = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\synchronized sensor and ATI"
    CSV_FILENAMES = [
        "synchronized_events_1.csv", "synchronized_events_2.csv", "synchronized_events_3.csv",
        "synchronized_events_4.csv", "synchronized_events_5.csv", "synchronized_events_6.csv",
        "synchronized_events_7.csv", "synchronized_events_8.csv", "synchronized_events_12.csv",
        "synchronized_events_21.csv", "synchronized_events_22.csv", "synchronized_events_23.csv",
        "synchronized_events_24.csv", "synchronized_events_25.csv", "synchronized_events_26.csv",
        "synchronized_events_27.csv", "synchronized_events_28.csv", "synchronized_events_29.csv",
        "synchronized_events_30.csv", "synchronized_events_31.csv", "synchronized_events_32.csv",        
    ]
    
    all_dfs = []
    expected_cols = ['t', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', '-x (mm)', '-y (mm)', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']

    for filename in CSV_FILENAMES:
        full_path = os.path.join(DATA_DIRECTORY, filename)
        if not os.path.exists(full_path):
            print(f"Warning: Data file '{filename}' not found. Skipping.")
            continue
        
        temp_df = pd.read_csv(full_path)

        # Fix for corrupted headers from spreadsheet software
        if '-x (mm)' not in temp_df.columns:
            print(f"Found corrupted headers in {filename}. Renaming columns by position...")
            current_cols = temp_df.columns.to_list()
            if len(current_cols) >= 9:
                rename_map = {current_cols[7]: '-x (mm)', current_cols[8]: '-y (mm)'}
                temp_df.rename(columns=rename_map, inplace=True)
        
        # Ensure the dataframe only contains the 15 expected columns
        temp_df = temp_df[expected_cols]
        all_dfs.append(temp_df)

    if not all_dfs:
        raise FileNotFoundError("No valid data files were found. Exiting.")

    df = pd.concat(all_dfs, ignore_index=True)
    df.dropna(inplace=True)
    print(f"Successfully prepared dataset with {len(df)} total data points.")

    INPUT_FEATURES = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    OUTPUT_TARGETS = ['-x (mm)', '-y (mm)', 'fx', 'fy', 'fz']
    X = df[INPUT_FEATURES].values
    Y = df[OUTPUT_TARGETS].values

    # --- 2. Split and Scale Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    # --- 3. Initialize and Train the Multi-Output SVR Model ---
    print("\n" + "="*70 + "\nTRAINING MULTI-OUTPUT SVR MODEL\n" + "="*70)
    
    svr_base = SVR(kernel='rbf', C=10.0, epsilon=0.1)
    multi_output_svr = MultiOutputRegressor(svr_base, n_jobs=-1)
    
    print("Training SVR models for all targets...")
    multi_output_svr.fit(X_train_scaled, y_train_scaled)
    print("Training complete.")

    # --- 4. Evaluate on the Test Set ---
    print("\n" + "="*70 + "\nFINAL MODEL PERFORMANCE ON TEST SET\n" + "="*70)
    
    predictions_scaled = multi_output_svr.predict(X_test_scaled)
    predictions = y_scaler.inverse_transform(predictions_scaled)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\nOverall Test Set Metrics:\n  - MSE:       {mse:.4f}\n  - R-squared: {r2:.4f}")
    
    calculate_grouped_rmse(y_test, predictions)
    
    print("\nGenerating prediction plots...")
    plot_all_targets_summary(y_test, predictions, OUTPUT_TARGETS)

    print("\n" + "="*70 + "\nPER-TARGET PERFORMANCE METRICS\n" + "="*70)
    for i, target in enumerate(OUTPUT_TARGETS):
        mae_target = mean_absolute_error(y_test[:, i], predictions[:, i])
        r2_target = r2_score(y_test[:, i], predictions[:, i])
        print(f"{target:12s} | MAE: {mae_target:8.4f} | R²: {r2_target:7.4f}")

    # --- 5. Save the Model and Scalers ---
    print("\n" + "="*70 + "\nSAVING MODEL AND SCALERS\n" + "="*70)

    with open('multi_output_svr.pkl', 'wb') as f:
        pickle.dump(multi_output_svr, f)
    with open('x_scaler_svr.pkl', 'wb') as f:
        pickle.dump(x_scaler, f)
    with open('y_scaler_svr.pkl', 'wb') as f:
        pickle.dump(y_scaler, f)

    print("Multi-output SVR model saved to: multi_output_svr.pkl")
    print("X scaler saved to: x_scaler_svr.pkl")
    print("Y scaler saved to: y_scaler_svr.pkl")
    print("\nProcess complete!")