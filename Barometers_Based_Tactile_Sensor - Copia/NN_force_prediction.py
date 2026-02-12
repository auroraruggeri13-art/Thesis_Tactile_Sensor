import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle


class RegressionModelNN(nn.Module):
    def __init__(self, input_dims, output_dims, fc_dims=[512, 256, 128], 
                 dropout_prob=0.1, lr=0.001, weight_decay=1e-4):
        super().__init__()

        # Build network with BatchNorm
        all_fc_dims = [input_dims] + fc_dims
        layers = []
        for i in range(len(all_fc_dims) - 1):
            layers.append(nn.Linear(all_fc_dims[i], all_fc_dims[i + 1]))
            layers.append(nn.BatchNorm1d(all_fc_dims[i + 1]))  # Batch normalization
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(all_fc_dims[-1], output_dims))
        self.model = nn.Sequential(*layers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Use AdamW with weight decay
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', 
                                          factor=0.5, patience=15, 
                                          min_lr=1e-6)

        self.training_losses = []
        self.validation_metrics = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def forward(self, x):
        return self.model(x.to(self.device))

    def learn(self, train_loader, val_loader, num_epochs=200, y_scaler=None, 
              early_stopping_patience=30):
        print(f"Starting training on {self.device} for up to {num_epochs} epochs...")
        print(f"Early stopping patience: {early_stopping_patience}")
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            running_train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / len(train_loader)
            self.training_losses.append(avg_train_loss)

            # Validation phase
            val_metrics = self.validate(val_loader, y_scaler)
            self.validation_metrics.append(val_metrics)
            
            # Learning rate scheduler step
            self.scheduler.step(val_metrics['loss'])

            # Print progress
            if (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:03d}/{num_epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val MAE: {val_metrics['mae']:.4f} | "
                      f"Val R2: {val_metrics['r2']:.4f} | "
                      f"LR: {current_lr:.6f}")

            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                # Save best model
                self.best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                # Restore best model
                self.load_state_dict(self.best_model_state)
                break

        print("Training complete.")

    def validate(self, loader, y_scaler):
        self.eval()
        running_loss = 0.0
        all_y_true = []
        all_y_pred = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                running_loss += loss.item()

                all_y_pred.append(y_pred.cpu().numpy())
                all_y_true.append(y_batch.cpu().numpy())

        all_y_pred = np.concatenate(all_y_pred)
        all_y_true = np.concatenate(all_y_true)

        y_pred_unscaled = y_scaler.inverse_transform(all_y_pred)
        y_true_unscaled = y_scaler.inverse_transform(all_y_true)

        mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
        r2 = r2_score(y_true_unscaled, y_pred_unscaled)
        mse = running_loss / len(loader)
        rmse = np.sqrt(mse)

        return {'loss': mse, 'mae': mae, 'r2': r2, 'rmse': rmse}

    def predict(self, x_tensor):
        self.eval()
        with torch.no_grad():
            predictions = self(x_tensor.to(self.device))
        return predictions.cpu().numpy()

def plot_training_history(model):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(model.training_losses, label='Training Loss', alpha=0.8)
    val_losses = [m['loss'] for m in model.validation_metrics]
    ax1.plot(val_losses, label='Validation Loss', alpha=0.8)
    ax1.set_title('Model Training History - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # R2 plot
    val_r2 = [m['r2'] for m in model.validation_metrics]
    ax2.plot(val_r2, label='Validation R²', color='green', alpha=0.8)
    ax2.set_title('Model Training History - R² Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R² Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
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
    
    plt.title(f'Actual vs. Predicted: "{target_names[target_index]}"\nMAE: {mae:.3f} | R²: {r2:.3f}')
    plt.xlabel(f'Actual {target_names[target_index]}')
    plt.ylabel(f'Predicted {target_names[target_index]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.axis('equal')
    plt.show()

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
    plt.show()
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

# Main 

if __name__ == "__main__":
    DATA_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\train_validation_test_data"
    
    sensor_version = 5.15

    # Files already created by the previous script
    TRAIN_FILENAME = f"train_data_v{sensor_version}.csv"
    VAL_FILENAME   = f"validation_data_v{sensor_version}.csv"
    TEST_FILENAME  = f"test_data_v{sensor_version}.csv"                       
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

    INPUT_FEATURES = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    OUTPUT_TARGETS = output_targets

    # Build numpy arrays for model training
    X_train = train_df[INPUT_FEATURES].values
    y_train = train_df[OUTPUT_TARGETS].values

    X_val   = val_df[INPUT_FEATURES].values
    y_val   = val_df[OUTPUT_TARGETS].values

    X_test  = test_df[INPUT_FEATURES].values
    y_test  = test_df[OUTPUT_TARGETS].values

    
    # Scale the data using StandardScaler
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    # --- 3. Convert to PyTorch Tensors & Create DataLoaders ---
    X_train_tensor = torch.from_numpy(X_train_scaled).float()
    y_train_tensor = torch.from_numpy(y_train_scaled).float()
    X_val_tensor = torch.from_numpy(X_val_scaled).float()
    y_val_tensor = torch.from_numpy(y_val_scaled).float()
    X_test_tensor = torch.from_numpy(X_test_scaled).float()
    y_test_tensor = torch.from_numpy(y_test_scaled).float()

    batch_size = 2048*4
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # --- 4. Initialize and Train the Model ---
    print("\n" + "="*70)
    print("MODEL WITH BATCH NORM, LR SCHEDULER & EARLY STOPPING")
    print("="*70)
    
    model = RegressionModelNN(
        input_dims=len(INPUT_FEATURES),
        output_dims=len(OUTPUT_TARGETS),
        fc_dims=[512, 256, 128, 64],
        dropout_prob=0.3,
        lr=0.01,
        weight_decay=1e-4
    )
    #print(model)

    # Train using VALIDATION set for monitoring (not test set!)
    model.learn(train_loader, val_loader, num_epochs=500, y_scaler=y_scaler, early_stopping_patience=100)

    # --- 5. Evaluate on TEST SET (never seen during training) ---
    print("\n" + "="*70)
    print("FINAL MODEL PERFORMANCE ON TEST SET")
    print("="*70)
    
    final_metrics = model.validate(test_loader, y_scaler=y_scaler)
    print("\nFinal Test Set Metrics:")
    print(f"  - Loss (MSE):  {final_metrics['loss']:.4f}")
    print(f"  - RMSE:        {final_metrics['rmse']:.4f}")
    print(f"  - MAE:         {final_metrics['mae']:.4f}")
    print(f"  - R-squared:   {final_metrics['r2']:.4f}")

    # Make predictions on test set
    predictions_scaled = model.predict(X_test_tensor)
    predictions = y_scaler.inverse_transform(predictions_scaled)

    # Calculate grouped RMSE metrics
    calculate_grouped_rmse(y_test, predictions, OUTPUT_TARGETS)

    # Create plots
    print("\nGenerating prediction plots...")
    history_fig = plot_training_history(model)
    summary_fig = plot_all_targets_summary(y_test, predictions, OUTPUT_TARGETS)

    # Comparison table
    results_df = pd.DataFrame(y_test, columns=[f'Actual_{col}' for col in OUTPUT_TARGETS])
    predictions_df = pd.DataFrame(predictions, columns=[f'Pred_{col}' for col in OUTPUT_TARGETS])
    comparison_df = pd.concat([results_df, predictions_df], axis=1)
    
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (First 10 rows)")
    print("="*70)
    print(comparison_df.head(10).round(2))

    # Per-target metrics
    print("\n" + "="*70)
    print("PER-TARGET PERFORMANCE METRICS (TEST SET)")
    print("="*70)
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
        
    # --- 6. Save the Model and Scalers ---
    print("\n" + "="*70)
    print("SAVING MODEL AND SCALERS")
    print("="*70)

    # Extract version number from the filename
    version = f"v{sensor_version}"
    
    # Define save directory
    save_dir = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\NN"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model and scalers with version
    model_path = os.path.join(save_dir, f'improved_regression_model_{version}.pth')
    x_scaler_path = os.path.join(save_dir, f'x_scaler_{version}.pkl')
    y_scaler_path = os.path.join(save_dir, f'y_scaler_{version}.pkl')

    torch.save(model.state_dict(), model_path)
    with open(x_scaler_path, 'wb') as f:
        pickle.dump(x_scaler, f)
    with open(y_scaler_path, 'wb') as f:
        pickle.dump(y_scaler, f)

    print(f"Model weights saved to: {model_path}")
    print(f"X scaler saved to: {x_scaler_path}")
    print(f"Y scaler saved to: {y_scaler_path}")
    
    # Save plots to files with version
    history_fig.savefig(os.path.join(save_dir, f'training_history_{version}.png'), bbox_inches='tight', dpi=300)
    summary_fig.savefig(os.path.join(save_dir, f'predictions_summary_{version}.png'), bbox_inches='tight', dpi=300)
    plt.close('all')

    print(f"\nAll plots saved to: {save_dir}")
    print("Training complete!")