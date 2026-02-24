import os
import sys
from pathlib import Path

# Make utils importable when running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

from utils.metrics_utils import calculate_grouped_rmse
from utils.plot_utils import plot_pred_vs_actual
from utils.io_utils import load_tabular_csv


if __name__ == "__main__":
    DATA_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data"

    sensor_version = 5.9

    # Files already created by the previous script
    TRAIN_FILENAME = f"train_data_v{sensor_version}.csv"
    TEST_FILENAME   = f"test_data_v{sensor_version}.csv"
    output_targets = ['x', 'y', 'fx', 'fy', 'fz']  # , 'tx', 'ty', 'tz'

    # Define the columns you expect to have in the final clean DataFrame
    expected_cols = ['t', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'x', 'y', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']

    # ---- Load TRAIN ----
    train_path = os.path.join(DATA_DIRECTORY, TRAIN_FILENAME)
    train_df = load_tabular_csv(train_path, expected_cols)

    # ---- Load TEST ----
    test_path = os.path.join(DATA_DIRECTORY, TEST_FILENAME)
    test_df = load_tabular_csv(test_path, expected_cols)

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

    calculate_grouped_rmse(y_test, predictions, OUTPUT_TARGETS)

    # Use the sensor_version variable
    version = f"v{sensor_version}"

    # Plot all targets summary
    print("\nGenerating prediction plots...")
    save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models paramters\XGBoost"
    plot_name = f'xgboost_predictions_summary_{version}.png'
    save_path = os.path.join(save_dir, plot_name)
    fig = plot_pred_vs_actual(y_test, predictions, OUTPUT_TARGETS, save_path=save_path)
    plt.show()
    plt.close(fig)

    print("\n" + "="*70 + "\nPER-TARGET PERFORMANCE METRICS\n" + "="*70)
    for i, target in enumerate(OUTPUT_TARGETS):
        mae_target = mean_absolute_error(y_test[:, i], predictions[:, i])
        r2_target = r2_score(y_test[:, i], predictions[:, i])
        print(f"{target:12s} | MAE: {mae_target:8.4f} | R\u00b2: {r2_target:7.4f}")

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
