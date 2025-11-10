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
    fig, axes = plt.subplots(1, n_targets, figsize=(4*n_targets, 4))
    if n_targets == 1:
        axes = [axes]
    
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
    
    plt.tight_layout()
    return fig

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

    # --- 2. Preprocess Data with a standard Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    
    # Each target will have its own scaler 
    y_scalers = {}
    y_train_scaled = np.zeros_like(y_train)
    for i in range(Y.shape[1]):
        y_scalers[i] = StandardScaler()
        # Scale each column of y_train individually
        y_train_scaled[:, i] = y_scalers[i].fit_transform(y_train[:, i].reshape(-1, 1)).flatten()

    # --- 3. Initialize and Train Multiple Models ---
    print("\n" + "="*70)
    print("TRAINING A SEPARATE LINEAR REGRESSION MODEL PER TARGET")
    print("="*70)
    
    models = {}
    for i, target_name in enumerate(OUTPUT_TARGETS):
        print(f"Training model for target: {target_name}...")
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled[:, i])
        models[target_name] = model

    print("All models trained successfully.")

    # --- 4. Evaluate on TEST SET using each model ---
    print("\n" + "="*70)
    print("FINAL MODEL PERFORMANCE ON TEST SET")
    print("="*70)
    
    all_predictions = []
    for i, target_name in enumerate(OUTPUT_TARGETS):
        model = models[target_name]
        predictions_scaled_single = model.predict(X_test_scaled)
        predictions_single = y_scalers[i].inverse_transform(predictions_scaled_single.reshape(-1, 1))
        all_predictions.append(predictions_single)
        
    predictions = np.hstack(all_predictions)

    # --- Evaluation Metrics ---
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

    # --- PLOTTING SECTION ---
    # Create the models parameters directory if it doesn't exist
    MODELS_DIR = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\Thesis - Tactile Sensor\models paramters\regression"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Extract version number from the CSV filename
    version = CSV_FILENAMES[0].split("sensor ")[1].split("\\")[0]
    
    # Calculate grouped RMSE metrics
    calculate_grouped_rmse(y_test, predictions)

    # Plot all targets summary
    print("\nGenerating prediction plots...")
    summary_fig = plot_all_targets_summary(y_test, predictions, OUTPUT_TARGETS)
    
    # Save the summary plot with version number
    summary_plot_path = os.path.join(MODELS_DIR, f'linear_regression_summary_{version}.png')
    summary_fig.savefig(summary_plot_path, bbox_inches='tight', dpi=300)
    plt.close(summary_fig)

    # --- 5. Save the Model and Scalers ---
    print("\n" + "="*70)
    print("SAVING MODEL AND SCALERS")
    print("="*70)

    # Extract version number from the CSV filename
    version = CSV_FILENAMES[0].split("sensor ")[1].split("\\")[0]  # This will extract "v2" from the filename
    
    # Create the models parameters directory if it doesn't exist
    MODELS_DIR = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\Thesis - Tactile Sensor\models paramters\regression"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save models dictionary with version
    models_path = os.path.join(MODELS_DIR, f'linear_regression_models_{version}.pkl')
    with open(models_path, 'wb') as f:
        pickle.dump(models, f)

    # Save input features scaler with version
    x_scaler_path = os.path.join(MODELS_DIR, f'x_scaler_lr_{version}.pkl')
    with open(x_scaler_path, 'wb') as f:
        pickle.dump(x_scaler, f)

    # Save output scalers dictionary with version
    y_scalers_path = os.path.join(MODELS_DIR, f'y_scalers_dict_{version}.pkl')
    with open(y_scalers_path, 'wb') as f:
        pickle.dump(y_scalers, f)

    print(f"\nModels saved to: {models_path}")
    print(f"X scaler saved to: {x_scaler_path}")
    print(f"Y scalers saved to: {y_scalers_path}")
    print("\nProcess complete!")