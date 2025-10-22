import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

# Plotting and metric functions remain the same
def plot_all_targets_summary(y_true, y_pred, target_names):
    """Plot a summary of all target predictions"""
    n_targets = y_true.shape[1]
    n_rows = (n_targets + 3) // 4  # Calculate rows needed
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_targets):
        ax = axes[i]
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        ax.scatter(true_vals, pred_vals, alpha=0.5, s=20)
        
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        buffer = (max_val - min_val) * 0.05
        lims = [min_val - buffer, max_val + buffer]
        
        ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=1.5)
        
        mae = mean_absolute_error(true_vals, pred_vals)
        r2 = r2_score(true_vals, pred_vals)
        
        ax.set_title(f'{target_names[i]}\nMAE: {mae:.2f} | R²: {r2:.3f}', fontsize=10)
        ax.set_xlabel('Actual', fontsize=8)
        ax.set_ylabel('Predicted', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    # Hide unused subplots
    for i in range(n_targets, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_grouped_rmse(y_true, y_pred):
    """Calculate RMSE for contact location and force vector separately"""
    
    # Extract contact location (x, y) - indices 0, 1
    contact_location_true = y_true[:, :2]
    contact_location_pred = y_pred[:, :2]
    
    # Extract 3-DOF force vector (fx, fy, fz) - indices 2, 3, 4
    force_vector_true = y_true[:, 2:5]
    force_vector_pred = y_pred[:, 2:5]
    
    # Calculate RMSE for contact location
    contact_location_errors = contact_location_true - contact_location_pred
    contact_location_rmse = np.sqrt(np.mean(contact_location_errors ** 2))
    
    # Calculate RMSE for 3-DOF force vector
    force_vector_errors = force_vector_true - force_vector_pred
    force_vector_rmse = np.sqrt(np.mean(force_vector_errors ** 2))
    
    # Calculate Euclidean distance RMSE (error magnitude per sample)
    contact_euclidean_errors = np.sqrt(np.sum(contact_location_errors ** 2, axis=1))
    contact_euclidean_rmse = np.sqrt(np.mean(contact_euclidean_errors ** 2))
    
    force_euclidean_errors = np.sqrt(np.sum(force_vector_errors ** 2, axis=1))
    force_euclidean_rmse = np.sqrt(np.mean(force_euclidean_errors ** 2))
    
    print("\n" + "="*70)
    print("GROUPED RMSE METRICS")
    print("="*70)
    print(f"\nContact Location (x, y):")
    print(f"  - Component-wise RMSE: {contact_location_rmse:.4f} mm")
    print(f"  - Euclidean RMSE:      {contact_euclidean_rmse:.4f} mm")
    print(f"  - Mean error distance: {np.mean(contact_euclidean_errors):.4f} mm")
    
    print(f"\n3-DOF Force Vector (fx, fy, fz):")
    print(f"  - Component-wise RMSE: {force_vector_rmse:.4f} N")
    print(f"  - Euclidean RMSE:      {force_euclidean_rmse:.4f} N")
    print(f"  - Mean error magnitude: {np.mean(force_euclidean_errors):.4f} N")


if __name__ == "__main__":
    # --- 1. Load and Combine Data ---
    # This section remains unchanged
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
            
        try:
            temp_df = pd.read_csv(full_path, skipinitialspace=True)
            temp_df.columns = temp_df.columns.str.strip()
            
            if '#NOME?' in temp_df.columns and '#NOME?.1' in temp_df.columns:
                temp_df = temp_df.rename(columns={'#NOME?': '-x (mm)', '#NOME?.1': '-y (mm)'})
            
            missing_cols = [col for col in expected_cols if col not in temp_df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {filename}: {missing_cols}")
                continue
                
            all_dfs.append(temp_df)
                
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            continue
            
    if not all_dfs:
        raise FileNotFoundError("No valid data files found.")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Successfully combined {len(all_dfs)} files with {len(df)} total data points.")

    INPUT_FEATURES = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    OUTPUT_TARGETS = ['-x (mm)', '-y (mm)', 'fx', 'fy', 'fz']
    X = df[INPUT_FEATURES].values
    Y = df[OUTPUT_TARGETS].values

    # --- 2. Preprocess Data with a standard Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    
    #-- MODIFIED: Use a single scaler for all targets
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    # --- 3. Initialize and Train a SINGLE Multi-Output Model --- #-- MODIFIED SECTION
    print("\n" + "="*70)
    print("TRAINING A SINGLE MULTI-OUTPUT LINEAR REGRESSION MODEL")
    print("="*70)
    
    #-- MODIFIED: Initialize one model instead of a dictionary of models
    model = LinearRegression()
    
    print("Training the multi-output model...")
    #-- MODIFIED: Fit the model once on all targets simultaneously
    model.fit(X_train_scaled, y_train_scaled)

    print("Model trained successfully.")

    # --- 4. Evaluate on TEST SET using the single model --- #-- MODIFIED SECTION
    print("\n" + "="*70)
    print("FINAL MODEL PERFORMANCE ON TEST SET")
    print("="*70)
    
    #-- MODIFIED: Predict all targets in a single step
    predictions_scaled = model.predict(X_test_scaled)
    
    #-- MODIFIED: Inverse transform all predictions at once
    predictions = y_scaler.inverse_transform(predictions_scaled)

    # --- Evaluation Metrics --- (This section remains the same)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nFinal Test Set Metrics:")
    print(f"  - MSE:         {mse:.4f}")
    print(f"  - R-squared:   {r2:.4f}")

    print("\n" + "="*70)
    print("PER-TARGET PERFORMANCE METRICS")
    print("="*70)
    for i, target in enumerate(OUTPUT_TARGETS):
        mae_target = mean_absolute_error(y_test[:, i], predictions[:, i])
        r2_target = r2_score(y_test[:, i], predictions[:, i])
        print(f"{target:12s} | MAE: {mae_target:8.4f} | R²: {r2_target:7.4f}")

    # --- PLOTTING SECTION --- (This section remains the same)
    calculate_grouped_rmse(y_test, predictions)
    print("\nGenerating prediction plots...")
    plot_all_targets_summary(y_test, predictions, OUTPUT_TARGETS)

    # --- 5. Save the Model and Scalers --- #-- MODIFIED SECTION
    print("\n" + "="*70)
    print("SAVING MODEL AND SCALERS")
    print("="*70)

    #-- MODIFIED: Save the single model object
    with open('linear_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('x_scaler_lr.pkl', 'wb') as f:
        pickle.dump(x_scaler, f)
        
    #-- MODIFIED: Save the single y_scaler object
    with open('y_scaler_lr.pkl', 'wb') as f:
        pickle.dump(y_scaler, f)

    print("Single model saved to: linear_regression_model.pkl")
    print("X scaler saved to: x_scaler_lr.pkl")
    print("Y scaler saved to: y_scaler_lr.pkl")
    print("\nProcess complete!")