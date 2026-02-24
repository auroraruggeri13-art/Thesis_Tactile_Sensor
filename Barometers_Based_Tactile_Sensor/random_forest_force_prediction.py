import os
import sys
from pathlib import Path

# Make utils importable when running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

from utils.metrics_utils import calculate_grouped_rmse
from utils.plot_utils import plot_pred_vs_actual, plot_data_distribution
from utils.io_utils import load_tabular_csv


def plot_error_distribution(y_true, y_pred, target_names):
    """
    Plot error distributions (residuals) for each target variable.
    Bottom row shows residuals vs. predicted value (scatter), not box plots.

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
            unit = "N\u00b7mm"
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

    plt.title(f'Actual vs. Predicted: "{target_names[target_index]}"\nMAE: {mae:.3f} | R\u00b2: {r2:.3f}')
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
    fig = plot_pred_vs_actual(y_test, predictions, OUTPUT_TARGETS)
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
            unit = "N\u00b7mm"
            scale = 1000  # Convert from N·m to N·mm
        else:
            unit = "N"
            scale = 1

        print(f"{target:12s} | MAE: {mae_target*scale:8.4f} {unit} | RMSE: {rmse_target*scale:8.4f} {unit} | R\u00b2: {r2_target:7.4f}")

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
    fig = plot_pred_vs_actual(
        y_test, predictions, OUTPUT_TARGETS,
        save_path=os.path.join(save_dir, f'rf_all_targets_summary_{version}.png'),
    )
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
