import os
import sys
from pathlib import Path

# Make utils importable when running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

from utils.metrics_utils import calculate_grouped_rmse
from utils.plot_utils import plot_pred_vs_actual
from utils.io_utils import load_tabular_csv


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
    train_df = load_tabular_csv(train_path, expected_cols)

    # ---- Load VALIDATION ----
    val_path = os.path.join(DATA_DIRECTORY, VAL_FILENAME)
    val_df = load_tabular_csv(val_path, expected_cols)

    # ---- Load TEST ----
    test_path = os.path.join(DATA_DIRECTORY, TEST_FILENAME)
    test_df = load_tabular_csv(test_path, expected_cols)

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
        print(f"  - Validation R\u00b2:  {val_r2:.4f}")

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
        print(f"  - R\u00b2:   {r2_target:8.4f}")
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
    print(f"Average R\u00b2 across all targets: {np.mean(overall_r2):.4f}")

    # --- PLOTTING SECTION ---
    calculate_grouped_rmse(y_test, predictions, OUTPUT_TARGETS)
    print("\nGenerating prediction plots...")

    # Define save path in the current directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    plot_save_path = os.path.join(SCRIPT_DIR, f'linear_regression_predictions_v{sensor_version}.png')

    # Linear Regression historically used slightly larger subplots (5 wide x 5 tall each)
    fig = plot_pred_vs_actual(
        y_test, predictions, OUTPUT_TARGETS,
        save_path=plot_save_path,
        figsize_factor=(5, 5),
    )
    plt.show()
    plt.close(fig)

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
