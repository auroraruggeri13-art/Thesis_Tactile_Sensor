import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb  # Import XGBoost
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

def plot_all_targets_summary(y_true, y_pred, target_names, save_dir=None, version=None):
    n_targets = y_true.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(4*n_targets, 4))
    if n_targets == 1:
        axes = [axes]
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
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_name = f'xgboost_predictions_summary_{version}.png' if version else 'xgboost_predictions_summary.png'
        plot_path = os.path.join(save_dir, plot_name)
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Saved predictions plot to: {plot_path}")
    
    plt.show()
    return fig

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


if __name__ == "__main__":
    # --- 1. Load and Preprocess Data ---
    DATA_DIRECTORY = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\test data" 
    CSV_FILENAMES = [
        "test 101 - sensor v1\synchronized_events_101.csv" 
    ]
    
    all_dfs = []
    # Define the columns you expect to have in the final clean DataFrame
    expected_cols = ['t', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', '-x (mm)', '-y (mm)', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']

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
    
    # --- 3. Initialize and Train a SEPARATE XGBoost Model for Each Target ---
    print("\n" + "="*70 + "\nTRAINING SEPARATE XGBOOST MODELS PER TARGET\n" + "="*70)
    
    models = {}
    
    # Define a strong set of starting hyperparameters for XGBoost
    params = {
        'n_estimators': 500,          # Number of boosting rounds
        'learning_rate': 0.05,        # Step size shrinkage
        'max_depth': 4,               # Maximum depth of a tree
        'subsample': 0.8,             # Fraction of samples to be used for fitting the individual base learners
        'colsample_bytree': 0.8,      # Fraction of columns to be random samples for each tree
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }

    for i, target_name in enumerate(OUTPUT_TARGETS):
        print(f"Training model for target: {target_name}...")
        
        # Create a new XGBoost Regressor model for each target
        model = xgb.XGBRegressor(**params)
        
        # Fit the model on its specific target column (y_train[:, i])
        model.fit(X_train_scaled, y_train[:, i])
        models[target_name] = model

    print("All specialized XGBoost models trained successfully.")

    # --- 4. Evaluate on TEST SET ---
    print("\n" + "="*70 + "\nFINAL MODEL PERFORMANCE ON TEST SET\n" + "="*70)
    
    all_predictions = []
    for target_name in OUTPUT_TARGETS:
        model = models[target_name]
        single_prediction = model.predict(X_test_scaled)
        all_predictions.append(single_prediction.reshape(-1, 1))

    predictions = np.hstack(all_predictions)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\nOverall Test Set Metrics:\n  - MSE:         {mse:.4f}\n  - R-squared:   {r2:.4f}")
    
    calculate_grouped_rmse(y_test, predictions)
    
    # Extract version number for plots
    version = CSV_FILENAMES[0].split("sensor ")[1].split("\\")[0]
    
    # Plot all targets summary
    print("\nGenerating prediction plots...")
    save_dir = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\Thesis - Tactile Sensor\models paramters\XGBoost"
    plot_all_targets_summary(y_test, predictions, OUTPUT_TARGETS, save_dir=save_dir, version=version)

    print("\n" + "="*70 + "\nPER-TARGET PERFORMANCE METRICS\n" + "="*70)
    for i, target in enumerate(OUTPUT_TARGETS):
        mae_target = mean_absolute_error(y_test[:, i], predictions[:, i])
        r2_target = r2_score(y_test[:, i], predictions[:, i])
        print(f"{target:12s} | MAE: {mae_target:8.4f} | R²: {r2_target:7.4f}")

    # --- 5. Save the Models and Scaler ---
    print("\n" + "="*70 + "\nSAVING MODELS AND SCALER\n" + "="*70)

    # Extract version number from the CSV filename
    version = CSV_FILENAMES[0].split("sensor ")[1].split("\\")[0]  # This will extract "v2" from the filename
    
    save_dir = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\Thesis - Tactile Sensor\models paramters\XGBoost"
    models_path = os.path.join(save_dir, f'specialized_xgb_models_{version}.pkl')
    scaler_path = os.path.join(save_dir, f'x_scaler_xgb_{version}.pkl')
    
    with open(models_path, 'wb') as f:
        pickle.dump(models, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(x_scaler, f)

    print(f"Dictionary of specialized XGBoost models saved to: {models_path}")
    print(f"X scaler saved to: {scaler_path}")
    print("X scaler saved to: x_scaler_xgb.pkl")
    print("\nProcess complete!")