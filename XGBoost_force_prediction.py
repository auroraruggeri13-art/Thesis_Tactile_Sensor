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
    DATA_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data"
    
    sensor_version = 4

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
    
    # --- 3. Initialize and Train a SEPARATE XGBoost Model for Each Target ---
    print("\n" + "="*70 + "\nTRAINING SEPARATE XGBOOST MODELS PER TARGET\n" + "="*70)
    
    models = {}
    
    # Define separate hyperparameter sets for position and force targets
    # Position model (x, y) - more regularized for stable predictions
    position_params = {
        'n_estimators': 1500,
        'learning_rate': 0.02,
        'max_depth': 6,
        'min_child_weight': 2,
        'gamma': 0.15,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'reg_alpha': 0.2,
        'reg_lambda': 3,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Force model (fx, fy, fz) - optimized to capture complex force signals
    force_params = {
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'max_depth': 8,
        'min_child_weight': 1,
        'gamma': 0.03,
        'subsample': 0.9,
        'colsample_bytree': 0.95,
        'colsample_bylevel': 0.9,
        'colsample_bynode': 0.9,
        'reg_alpha': 0.01,
        'reg_lambda': 0.5,
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    for i, target_name in enumerate(OUTPUT_TARGETS):
        print(f"Training model for target: {target_name}...")
        
        # Choose the correct set of parameters based on target type
        if target_name in ['x', 'y']:
            params = position_params
        else:  # Use force_params for fx, fy, fz
            params = force_params
        
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
    
    # Use the sensor_version variable
    version = f"v{sensor_version}"
    
    # Plot all targets summary
    print("\nGenerating prediction plots...")
    save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models paramters\XGBoost"
    plot_all_targets_summary(y_test, predictions, OUTPUT_TARGETS, save_dir=save_dir, version=version)

    print("\n" + "="*70 + "\nPER-TARGET PERFORMANCE METRICS\n" + "="*70)
    for i, target in enumerate(OUTPUT_TARGETS):
        mae_target = mean_absolute_error(y_test[:, i], predictions[:, i])
        r2_target = r2_score(y_test[:, i], predictions[:, i])
        print(f"{target:12s} | MAE: {mae_target:8.4f} | R²: {r2_target:7.4f}")

    # --- 5. Save the Models and Scaler ---
    print("\n" + "="*70 + "\nSAVING MODELS AND SCALER\n" + "="*70)

    # Use the sensor_version variable
    version = f"v{sensor_version}"
    
    save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models paramters\XGBoost"
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