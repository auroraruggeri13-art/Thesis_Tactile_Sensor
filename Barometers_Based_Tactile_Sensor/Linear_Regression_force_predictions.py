import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

# Plotting and metric functions remain the same
def plot_all_targets_summary(y_true, y_pred, target_names, save_path=None):
    """Plot a summary of all target predictions"""
    n_targets = y_true.shape[1]
    # Arrange all plots in one row
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]  # Make it iterable if only one subplot
    
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
    
    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def calculate_grouped_rmse(y_true, y_pred, target_names):
    """Calculate RMSE for contact location and force vector separately"""
    
    print("\n" + "="*70)
    print("GROUPED RMSE METRICS")
    print("="*70)
    
    # Check for contact location targets
    contact_indices = [i for i, name in enumerate(target_names) if name in ['x', 'y']]
    if len(contact_indices) >= 2:
        contact_location_true = y_true[:, contact_indices]
        contact_location_pred = y_pred[:, contact_indices]
        
        contact_location_errors = contact_location_true - contact_location_pred
        contact_location_rmse = np.sqrt(np.mean(contact_location_errors ** 2))
        
        contact_euclidean_errors = np.sqrt(np.sum(contact_location_errors ** 2, axis=1))
        contact_euclidean_rmse = np.sqrt(np.mean(contact_euclidean_errors ** 2))
        
        print(f"\nContact Location (x, y):")
        print(f"  - Component-wise RMSE: {contact_location_rmse:.4f} mm")
        print(f"  - Euclidean RMSE:      {contact_euclidean_rmse:.4f} mm")
        print(f"  - Mean error distance: {np.mean(contact_euclidean_errors):.4f} mm")
    
    # Check for force targets
    force_indices = [i for i, name in enumerate(target_names) if name in ['fx', 'fy', 'fz']]
    if force_indices:
        force_vector_true = y_true[:, force_indices]
        force_vector_pred = y_pred[:, force_indices]
        
        force_vector_errors = force_vector_true - force_vector_pred
        force_vector_rmse = np.sqrt(np.mean(force_vector_errors ** 2))
        
        force_euclidean_errors = np.sqrt(np.sum(force_vector_errors ** 2, axis=1))
        force_euclidean_rmse = np.sqrt(np.mean(force_euclidean_errors ** 2))
        
        force_names = [target_names[i] for i in force_indices]
        print(f"\n{len(force_indices)}-DOF Force Vector ({', '.join(force_names)}):")
        print(f"  - Component-wise RMSE: {force_vector_rmse:.4f} N")
        print(f"  - Euclidean RMSE:      {force_euclidean_rmse:.4f} N")
        print(f"  - Mean error magnitude: {np.mean(force_euclidean_errors):.4f} N")


if __name__ == "__main__":
    # --- 1. Load Pre-split Train/Validation/Test Data ---
    DATA_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data"
    
    sensor_version = 4.6

    # Files already created by the train_validation_test_dataset_generation.py script
    TRAIN_FILENAME = f"train_data_v{sensor_version}.csv"
    VAL_FILENAME   = f"validation_data_v{sensor_version}.csv"
    TEST_FILENAME  = f"test_data_v{sensor_version}.csv"
    
    # Define expected columns
    expected_cols = ['t', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'x', 'y', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']

    # ---- Load TRAIN ----
    train_path = os.path.join(DATA_DIRECTORY, TRAIN_FILENAME)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data file not found: {train_path}")

    train_df = pd.read_csv(train_path, skipinitialspace=True)
    train_df.columns = train_df.columns.str.strip()

    missing_cols_train = [col for col in expected_cols if col not in train_df.columns]
    if missing_cols_train:
        raise ValueError(f"Train file is missing columns: {missing_cols_train}")

    # ---- Load VALIDATION ----
    val_path = os.path.join(DATA_DIRECTORY, VAL_FILENAME)
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data file not found: {val_path}")

    val_df = pd.read_csv(val_path, skipinitialspace=True)
    val_df.columns = val_df.columns.str.strip()

    missing_cols_val = [col for col in expected_cols if col not in val_df.columns]
    if missing_cols_val:
        raise ValueError(f"Validation file is missing columns: {missing_cols_val}")

    # ---- Load TEST ----
    test_path = os.path.join(DATA_DIRECTORY, TEST_FILENAME)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data file not found: {test_path}")

    test_df = pd.read_csv(test_path, skipinitialspace=True)
    test_df.columns = test_df.columns.str.strip()

    missing_cols_test = [col for col in expected_cols if col not in test_df.columns]
    if missing_cols_test:
        raise ValueError(f"Test file is missing columns: {missing_cols_test}")

    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

    # --- 2. Prepare Input/Output Arrays ---
    INPUT_FEATURES = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    OUTPUT_TARGETS = ['x', 'y', 'fx', 'fy', 'fz']
    
    X_train = train_df[INPUT_FEATURES].values
    y_train = train_df[OUTPUT_TARGETS].values

    X_val = val_df[INPUT_FEATURES].values
    y_val = val_df[OUTPUT_TARGETS].values

    X_test = test_df[INPUT_FEATURES].values
    y_test = test_df[OUTPUT_TARGETS].values
    
    # --- 3. Scale Input Features ---
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)
    
    # --- 4. Train Separate Linear Regression Models for Each Target ---
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
        
        # Validation set performance
        y_val_target = y_val[:, i].reshape(-1, 1)
        pred_val_scaled = models[target].predict(X_val_scaled).reshape(-1, 1)
        pred_val = y_scalers[target].inverse_transform(pred_val_scaled)
        
        val_mae = mean_absolute_error(y_val_target, pred_val)
        val_r2 = r2_score(y_val_target, pred_val)
        print(f"  - Validation MAE: {val_mae:.4f}")
        print(f"  - Validation R²:  {val_r2:.4f}")
        
        # Print feature coefficients
        coef = pd.DataFrame({
            'Feature': INPUT_FEATURES,
            'Coefficient': models[target].coef_
        })
        coef = coef.sort_values('Coefficient', key=abs, ascending=False)
        print("\nFeature importance:")
        print(coef.to_string(index=False))

    # --- 5. Evaluate on TEST SET using separate models ---
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
    calculate_grouped_rmse(y_test, predictions, OUTPUT_TARGETS)
    print("\nGenerating prediction plots...")
    
    # Define save path in the current directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    plot_save_path = os.path.join(SCRIPT_DIR, f'linear_regression_predictions_v{sensor_version}.png')
    
    plot_all_targets_summary(y_test, predictions, OUTPUT_TARGETS, save_path=plot_save_path)

    # --- 6. Save all Models and Scalers ---
    print("\n" + "="*70)
    print("SAVING MODELS AND SCALERS")
    print("="*70)

    # Set models directory with version
    version = f"v{sensor_version}"
    MODELS_DIR = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\regression"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save input scaler
    x_scaler_path = os.path.join(MODELS_DIR, f'x_scaler_lr_{version}.pkl')
    with open(x_scaler_path, 'wb') as f:
        pickle.dump(x_scaler, f)
    print(f"Input scaler saved to: {x_scaler_path}")
    
    # Save each model and its scaler
    for target in OUTPUT_TARGETS:
        # Clean filename by removing spaces and parentheses
        clean_target = target.replace(" ", "_").replace("(", "").replace(")", "")
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f'linear_regression_model_{clean_target}_{version}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(models[target], f)
            
        # Save scaler
        scaler_path = os.path.join(MODELS_DIR, f'y_scaler_lr_{clean_target}_{version}.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(y_scalers[target], f)
            
        print(f"Model and scaler for {target} saved to:")
        print(f"  - Model:  {model_path}")
        print(f"  - Scaler: {scaler_path}")

    print("\nProcess complete!")