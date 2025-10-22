import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle


def plot_all_targets_summary(y_true, y_pred, target_names):
    n_targets = y_true.shape[1]
    n_rows = (n_targets + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    for i in range(n_targets):
        ax = axes[i]
        true_vals = y_true[:, i]; pred_vals = y_pred[:, i]
        ax.scatter(true_vals, pred_vals, alpha=0.5, s=20)
        min_val, max_val = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
        lims = [min_val - (max_val - min_val) * 0.05, max_val + (max_val - min_val) * 0.05]
        ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=1.5)
        mae, r2 = mean_absolute_error(true_vals, pred_vals), r2_score(true_vals, pred_vals)
        ax.set_title(f'{target_names[i]}\nMAE: {mae:.2f} | R²: {r2:.3f}', fontsize=10)
        ax.set_xlabel('Actual', fontsize=8); ax.set_ylabel('Predicted', fontsize=8)
        ax.grid(True, alpha=0.3); ax.set_xlim(lims); ax.set_ylim(lims)
    for i in range(n_targets, len(axes)):
        axes[i].axis('off')
    plt.tight_layout(); plt.show()

def calculate_grouped_rmse(y_true, y_pred):
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

def plot_predictions(y_true, y_pred, target_index=2, target_names=None):
    if target_names is None:
        target_names = ['Target']
        
    true_vals = y_true[:, target_index]
    pred_vals = y_pred[:, target_index]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(true_vals, pred_vals, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    buffer = (max_val - min_val) * 0.05 
    lims = [min_val - buffer, max_val + buffer]
    
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics for this target
    mae = mean_absolute_error(true_vals, pred_vals)
    r2 = r2_score(true_vals, pred_vals)
    
    plt.title(f'Actual vs. Predicted: "{target_names[target_index]}"\nMAE: {mae:.3f} | R²: {r2:.3f}')
    plt.xlabel(f'Actual {target_names[target_index]}')
    plt.ylabel(f'Predicted {target_names[target_index]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # --- 1. Load and Preprocess Data ---
    DATA_DIRECTORY = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\synchronized sensor and ATI" 
    CSV_FILENAMES = [
        "synchronized_events_1.csv",
        "synchronized_events_2.csv",
        "synchronized_events_3.csv",
        "synchronized_events_4.csv",
        "synchronized_events_5.csv",
        "synchronized_events_6.csv",
        "synchronized_events_7.csv",
        "synchronized_events_8.csv",
        "synchronized_events_12.csv",
        "synchronized_events_21.csv",
        "synchronized_events_22.csv",
        "synchronized_events_23.csv",
        "synchronized_events_24.csv",
        "synchronized_events_25.csv",
        "synchronized_events_26.csv",
        "synchronized_events_27.csv",
        "synchronized_events_28.csv",
        "synchronized_events_29.csv",
        "synchronized_events_30.csv",
        "synchronized_events_31.csv",
        "synchronized_events_32.csv",        
    ]
    
    all_dfs = []
    # Define the columns you expect to have in the final clean DataFrame
    expected_cols = ['t', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 
                     '-x (mm)', '-y (mm)', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']

    for filename in CSV_FILENAMES:
        full_path = os.path.join(DATA_DIRECTORY, filename)
        if not os.path.exists(full_path):
            print(f"Warning: Data file '{filename}' not found. Skipping.")
            continue
            
        try:
            # Read CSV with more robust settings
            temp_df = pd.read_csv(full_path, skipinitialspace=True)
            
            # Clean column names
            temp_df.columns = temp_df.columns.str.strip()
            
            # Print column names for debugging
            print(f"\nFile: {filename}")
            print("Columns found:", temp_df.columns.tolist())
            
            # Fix corrupted column names
            if '#NOME?' in temp_df.columns and '#NOME?.1' in temp_df.columns:
                temp_df = temp_df.rename(columns={
                    '#NOME?': '-x (mm)',
                    '#NOME?.1': '-y (mm)'
                })
            
            # Verify all expected columns are present
            missing_cols = [col for col in expected_cols if col not in temp_df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {filename}: {missing_cols}")
                continue
                
            all_dfs.append(temp_df)
                
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            continue
            
    if not all_dfs:
        raise FileNotFoundError("No valid data files found. Please check the data files and their column names.")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Successfully combined {len(all_dfs)} files with {len(df)} total data points.")

    INPUT_FEATURES = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    OUTPUT_TARGETS = ['-x (mm)', '-y (mm)', 'fx', 'fy', 'fz']
    X = df[INPUT_FEATURES].values
    Y = df[OUTPUT_TARGETS].values

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    
    # --- 3. Initialize and Train a SEPARATE Model for Each Target ---
    print("\n" + "="*70 + "\nTRAINING SEPARATE RANDOM FOREST MODELS PER TARGET\n" + "="*70)
    
    models = {}
    
    # Define the different hyperparameter sets
    # Params for the position model (more regularized)
    position_params = {'n_estimators': 300, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1}
    
    # Params for the force models (less regularized to capture complex signals)
    force_params = {'n_estimators': 300, 'min_samples_leaf': 1, 'max_features': 1.0, 'random_state': 42, 'n_jobs': -1}

    for i, target_name in enumerate(OUTPUT_TARGETS):
        print(f"Training model for target: {target_name}...")
        
        # Choose the correct set of parameters
        if target_name == '-x (mm)' or target_name == '-y (mm)':
            params = position_params
        else: # Use force_params for fx, fy, and fz
            params = force_params
            
        model = RandomForestRegressor(**params)
        
        # Fit the model on its specific target column (y_train[:, i])
        model.fit(X_train_scaled, y_train[:, i])
        models[target_name] = model

    print("All specialized models trained successfully.")

    # --- 4. Evaluate on TEST SET using each specialized model ---
    print("\n" + "="*70 + "\nFINAL MODEL PERFORMANCE ON TEST SET\n" + "="*70)
    
    all_predictions = []
    
    # IMPORTANT: Iterate in the same order as OUTPUT_TARGETS to keep columns aligned
    for target_name in OUTPUT_TARGETS:
        model = models[target_name]
        # Predict using the specialized model
        single_prediction = model.predict(X_test_scaled)
        # Reshape for horizontal stacking
        all_predictions.append(single_prediction.reshape(-1, 1))

    # Combine all prediction columns into a single (N_samples, 5) array
    predictions = np.hstack(all_predictions)

    # --- The rest of the evaluation is the same ---
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\nOverall Test Set Metrics:\n  - MSE:         {mse:.4f}\n  - R-squared:   {r2:.4f}")
    
    calculate_grouped_rmse(y_test, predictions)
    plot_all_targets_summary(y_test, predictions, OUTPUT_TARGETS)

    print("\n" + "="*70 + "\nPER-TARGET PERFORMANCE METRICS\n" + "="*70)
    for i, target in enumerate(OUTPUT_TARGETS):
        mae_target = mean_absolute_error(y_test[:, i], predictions[:, i])
        r2_target = r2_score(y_test[:, i], predictions[:, i])
        print(f"{target:12s} | MAE: {mae_target:8.4f} | R²: {r2_target:7.4f}")

    # --- 5. Save the Models and Scaler ---
    print("\n" + "="*70 + "\nSAVING MODELS AND SCALER\n" + "="*70)

    with open('specialized_rf_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    with open('x_scaler_rf.pkl', 'wb') as f:
        pickle.dump(x_scaler, f)

    print("Dictionary of specialized models saved to: specialized_rf_models.pkl")
    print("X scaler saved to: x_scaler_rf.pkl")
    print("\nProcess complete!")