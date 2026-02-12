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
    return fig

def plot_data_distribution(y_data, target_names, title_prefix=""):
    """
    Plot histograms showing the distribution of each target variable.
    
    Args:
        y_data: Data array (N, n_targets)
        target_names: List of target names
        title_prefix: Prefix for the plot title (e.g., "Training" or "Test")
    """
    n_targets = y_data.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(4*n_targets, 4))
    if n_targets == 1:
        axes = [axes]
    
    for i in range(n_targets):
        ax = axes[i]
        data = y_data[:, i]
        
        # Create histogram
        ax.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        median_val = np.median(data)
        
        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # Determine unit
        if target_names[i] in ['x', 'y']:
            unit = "mm"
        elif target_names[i] in ['tx', 'ty', 'tz']:
            unit = "N·m"
        else:
            unit = "N"
        
        ax.set_title(f'{target_names[i]}\nStd: {std_val:.2f} {unit}', fontsize=10)
        ax.set_xlabel(f'{target_names[i]} ({unit})', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'{title_prefix} Data Distribution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_error_distribution(y_true, y_pred, target_names):
    """
    Plot error distributions (residuals) for each target variable.
    
    Args:
        y_true: True values (N, n_targets)
        y_pred: Predicted values (N, n_targets)
        target_names: List of target names
    """
    n_targets = y_true.shape[1]
    fig, axes = plt.subplots(2, n_targets, figsize=(4*n_targets, 8))
    
    if n_targets == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_targets):
        # Calculate errors (residuals)
        errors = y_true[:, i] - y_pred[:, i]
        
        # Determine unit and scale
        if target_names[i] in ['x', 'y']:
            unit = "mm"
            scale = 1
        elif target_names[i] in ['tx', 'ty', 'tz']:
            unit = "N·mm"
            scale = 1000
        else:
            unit = "N"
            scale = 1
        
        errors_scaled = errors * scale
        
        # Top subplot: Histogram of errors
        ax1 = axes[0, i]
        ax1.hist(errors_scaled, bins=50, alpha=0.7, color='coral', edgecolor='black')
        
        # Add statistics
        mean_error = np.mean(errors_scaled)
        std_error = np.std(errors_scaled)
        
        ax1.axvline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
        ax1.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_error:.2f}')
        
        ax1.set_title(f'{target_names[i]}\nStd Error: {std_error:.2f} {unit}', fontsize=10)
        ax1.set_xlabel(f'Error ({unit})', fontsize=8)
        ax1.set_ylabel('Frequency', fontsize=8)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Bottom subplot: Error vs Predicted Value (to check for bias)
        ax2 = axes[1, i]
        ax2.scatter(y_pred[:, i] * scale, errors_scaled, alpha=0.5, s=10, color='purple')
        ax2.axhline(0, color='green', linestyle='--', linewidth=2, label='Zero Error')
        ax2.axhline(mean_error, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Mean Error')
        
        ax2.set_xlabel(f'Predicted {target_names[i]} ({unit})', fontsize=8)
        ax2.set_ylabel(f'Error ({unit})', fontsize=8)
        ax2.set_title(f'Residual Plot', fontsize=10)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Error Distribution Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def calculate_grouped_rmse(y_true, y_pred, target_names):
    
    print("\n" + "="*70 + "\nGROUPED RMSE METRICS\n" + "="*70)
    
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
    
    # Check for torque targets
    torque_indices = [i for i, name in enumerate(target_names) if name in ['tx', 'ty', 'tz']]
    if torque_indices:
        torque_vector_true = y_true[:, torque_indices]
        torque_vector_pred = y_pred[:, torque_indices]
        
        torque_vector_errors = torque_vector_true - torque_vector_pred
        torque_vector_rmse = np.sqrt(np.mean(torque_vector_errors ** 2))
        
        torque_euclidean_errors = np.sqrt(np.sum(torque_vector_errors ** 2, axis=1))
        torque_euclidean_rmse = np.sqrt(np.mean(torque_euclidean_errors ** 2))
        
        torque_names = [target_names[i] for i in torque_indices]
        print(f"\n{len(torque_indices)}-DOF Torque Vector ({', '.join(torque_names)}):")
        print(f"  - Component-wise RMSE: {torque_vector_rmse*1000:.4f} N·mm")
        print(f"  - Euclidean RMSE:      {torque_euclidean_rmse*1000:.4f} N·mm")
        print(f"  - Mean error magnitude: {np.mean(torque_euclidean_errors)*1000:.4f} N·mm")
        
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
    DATA_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data"
    
    sensor_version = 5.15

    # Files already created by the previous script
    TRAIN_FILENAME = f"train_data_v{sensor_version}.csv"
    TEST_FILENAME   = f"test_data_v{sensor_version}.csv"                       
    output_targets = ['x', 'y', 'fx', 'fy', 'fz']  # , 'tx', 'ty', 'tz'

    # Define the columns you expect to have in the final clean DataFrame
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

    # ---- Load TEST ----
    test_path = os.path.join(DATA_DIRECTORY, TEST_FILENAME)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data file not found: {test_path}")

    test_df = pd.read_csv(test_path, skipinitialspace=True)
    test_df.columns = test_df.columns.str.strip()

    missing_cols_test = [col for col in expected_cols if col not in test_df.columns]
    if missing_cols_test:
        raise ValueError(f"Test file is missing columns: {missing_cols_test}")

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    INPUT_FEATURES = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    OUTPUT_TARGETS = output_targets

    # Build numpy arrays for model training
    X_train = train_df[INPUT_FEATURES].values
    y_train = train_df[OUTPUT_TARGETS].values

    X_test   = test_df[INPUT_FEATURES].values
    y_test   = test_df[OUTPUT_TARGETS].values
    
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    
    # --- 3. Initialize and Train a SEPARATE Model for Each Target ---
    print("\n" + "="*70 + "\nTRAINING SEPARATE RANDOM FOREST MODELS PER TARGET\n" + "="*70)
    
    models = {}
    
    # Define the different hyperparameter sets
    # Params for the position model (more regularized)
    position_params = {'n_estimators': 200, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1}
    
    # Params for the force models (less regularized to capture complex signals)
    force_params = {'n_estimators': 200, 'min_samples_leaf': 1, 'max_features': 1.0, 'random_state': 42, 'n_jobs': -1}

    for i, target_name in enumerate(OUTPUT_TARGETS):
        print(f"Training model for target: {target_name}...")
        
        # Choose the correct set of parameters
        if target_name == 'x' or target_name == 'y':
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
    
    calculate_grouped_rmse(y_test, predictions, OUTPUT_TARGETS)
    fig = plot_all_targets_summary(y_test, predictions, OUTPUT_TARGETS)
    plt.show()  # Display the plot during evaluation
    plt.close(fig)

    print("\n" + "="*70 + "\nPER-TARGET PERFORMANCE METRICS\n" + "="*70)
    for i, target in enumerate(OUTPUT_TARGETS):
        mae_target = mean_absolute_error(y_test[:, i], predictions[:, i])
        rmse_target = np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i]))
        r2_target = r2_score(y_test[:, i], predictions[:, i])
        # Add appropriate unit based on the target
        if target in ['x', 'y']:
            unit = "mm"
            scale = 1
        elif target in ['tx', 'ty', 'tz']:
            unit = "N·mm"
            scale = 1000  # Convert from N·m to N·mm
        else:
            unit = "N"
            scale = 1
        
        print(f"{target:12s} | MAE: {mae_target*scale:8.4f} {unit} | RMSE: {rmse_target*scale:8.4f} {unit} | R²: {r2_target:7.4f}")

    # Clear any existing plots
    plt.close('all')
    
    # --- 5. Save the Models, Scaler, and Plots ---
    print("\n" + "="*70 + "\nSAVING MODELS, SCALER, AND PLOTS\n" + "="*70)
    
    # Create directory if it doesn't exist
    save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\random forest"
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract version number from the CSV filename
    version = f'v{sensor_version}'
    
    # Save models and scaler with version
    models_path = os.path.join(save_dir, f'specialized_rf_models_{version}.pkl')
    scaler_path = os.path.join(save_dir, f'x_scaler_rf_{version}.pkl')
    
    with open(models_path, 'wb') as f:
        pickle.dump(models, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(x_scaler, f)

    # Save summary plot with version
    fig = plot_all_targets_summary(y_test, predictions, OUTPUT_TARGETS)
    fig.savefig(os.path.join(save_dir, f'rf_all_targets_summary_{version}.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Save data distribution plots
    fig_train_dist = plot_data_distribution(y_train, OUTPUT_TARGETS, title_prefix="Training Set")
    fig_train_dist.savefig(os.path.join(save_dir, f'rf_training_data_distribution_{version}.png'), bbox_inches='tight', dpi=300)
    plt.close(fig_train_dist)
    
    fig_test_dist = plot_data_distribution(y_test, OUTPUT_TARGETS, title_prefix="Test Set")
    fig_test_dist.savefig(os.path.join(save_dir, f'rf_test_data_distribution_{version}.png'), bbox_inches='tight', dpi=300)
    plt.close(fig_test_dist)
    
    # Save error distribution plot
    fig_errors = plot_error_distribution(y_test, predictions, OUTPUT_TARGETS)
    fig_errors.savefig(os.path.join(save_dir, f'rf_error_distribution_{version}.png'), bbox_inches='tight', dpi=300)
    plt.close(fig_errors)

    print(f"Models and scaler saved to: {save_dir}")
    print(f"All plots saved to: {save_dir}")
    print("  - Summary plot")
    print("  - Training data distribution")
    print("  - Test data distribution")
    print("  - Error distribution analysis")
    print("\nProcess complete!")