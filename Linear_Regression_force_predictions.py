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
    DATA_DIRECTORY = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\synchronized Data"
    CSV_FILENAMES = [
        "synchronized_events_62.csv"    
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
    
    # Scale input features
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    
    # Create separate scalers and models for each target
    y_scalers = {}
    models = {}
    
    print("\n" + "="*70)
    print("TRAINING SEPARATE LINEAR REGRESSION MODELS FOR EACH TARGET")
    print("="*70)
    
    # Train a separate model for each target using all inputs
    for i, target in enumerate(OUTPUT_TARGETS):
        print(f"\nTraining model for {target}...")
        
        # Scale this target
        y_scalers[target] = StandardScaler()
        y_train_target = y_train[:, i].reshape(-1, 1)
        y_train_scaled = y_scalers[target].fit_transform(y_train_target)
        
        # Train model for this target
        models[target] = LinearRegression()
        models[target].fit(X_train_scaled, y_train_scaled.ravel())
        
        # Quick training set R² score
        train_r2 = models[target].score(X_train_scaled, y_train_scaled)
        print(f"  - Training R² score: {train_r2:.4f}")
        
        # Print feature coefficients
        coef = pd.DataFrame({
            'Feature': INPUT_FEATURES,
            'Coefficient': models[target].coef_
        })
        coef = coef.sort_values('Coefficient', key=abs, ascending=False)
        print("\nFeature importance:")
        print(coef.to_string(index=False))

    # --- 4. Evaluate on TEST SET using separate models ---
    print("\n" + "="*70)
    print("FINAL MODEL PERFORMANCE ON TEST SET")
    print("="*70)
    
    # Initialize array for all predictions
    predictions = np.zeros_like(y_test)
    
    # Evaluate each model separately
    overall_r2 = []
    print("\nPer-Target Performance Metrics:")
    print("-" * 50)
    
    for i, target in enumerate(OUTPUT_TARGETS):
        # Get predictions for this target
        y_test_target = y_test[:, i].reshape(-1, 1)
        pred_scaled = models[target].predict(X_test_scaled).reshape(-1, 1)
        pred_target = y_scalers[target].inverse_transform(pred_scaled)
        
        # Store predictions
        predictions[:, i] = pred_target.ravel()
        
        # Calculate metrics
        mae_target = mean_absolute_error(y_test_target, pred_target)
        r2_target = r2_score(y_test_target, pred_target)
        rmse_target = np.sqrt(mean_squared_error(y_test_target, pred_target))
        
        overall_r2.append(r2_target)
        
        print(f"\n{target}:")
        print(f"  - MAE:  {mae_target:8.4f}")
        print(f"  - RMSE: {rmse_target:8.4f}")
        print(f"  - R²:   {r2_target:8.4f}")
        print(f"  - Top coefficients:")
        coef = pd.DataFrame({
            'Feature': INPUT_FEATURES,
            'Coefficient': models[target].coef_
        })
        coef = coef.sort_values('Coefficient', key=abs, ascending=False)
        print(coef.head(3).to_string(index=False))

    print("\n" + "="*70)
    print("OVERALL MODEL PERFORMANCE")
    print("="*70)
    print(f"Average R² across all targets: {np.mean(overall_r2):.4f}")

    # --- PLOTTING SECTION ---
    calculate_grouped_rmse(y_test, predictions)
    print("\nGenerating prediction plots...")
    plot_all_targets_summary(y_test, predictions, OUTPUT_TARGETS)

    # --- 5. Save all Models and Scalers ---
    print("\n" + "="*70)
    print("SAVING MODELS AND SCALERS")
    print("="*70)

    # Set models directory
    MODELS_DIR = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\Thesis - Tactile Sensor\models paramters\regression"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save input scaler
    x_scaler_path = os.path.join(MODELS_DIR, 'x_scaler_lr.pkl')
    with open(x_scaler_path, 'wb') as f:
        pickle.dump(x_scaler, f)
    print(f"Input scaler saved to: {x_scaler_path}")
    
    # Save each model and its scaler
    for target in OUTPUT_TARGETS:
        # Clean filename by removing spaces and parentheses
        clean_target = target.replace(" ", "_").replace("(", "").replace(")", "")
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f'linear_regression_model_{clean_target}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(models[target], f)
            
        # Save scaler
        scaler_path = os.path.join(MODELS_DIR, f'y_scaler_lr_{clean_target}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(y_scalers[target], f)
            
        print(f"Model and scaler for {target} saved to:")
        print(f"  - Model:  {model_path}")
        print(f"  - Scaler: {scaler_path}")

    print("\nProcess complete!")