import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
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
                                          verbose=True, min_lr=1e-6)

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

# Main 

if __name__ == "__main__":
    # --- 1. Load and Combine Data ---
    DATA_DIRECTORY = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data" 
    CSV_FILENAMES = [
        "test 109 - sensor v1\synchronized_events_109.csv" 
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
                
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            continue

        # Ensure the dataframe contains only the expected columns in the correct order
        # Remove any extra columns (like time_ns) and keep only the ones we need
        temp_df = temp_df[expected_cols]
        
        all_dfs.append(temp_df)

    if not all_dfs:
        print("Error: No data files were found. Exiting.")
        exit()

    df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"Successfully combined {len(all_dfs)} files into a single dataset with {len(df)} total data points.")
    
    #I want to print the whole dataframe
    print(df)
 
    INPUT_FEATURES = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    OUTPUT_TARGETS = ['-x (mm)', '-y (mm)', 'fx', 'fy', 'fz']

    X = df[INPUT_FEATURES].values
    Y = df[OUTPUT_TARGETS].values

    # --- 2. Preprocess Data with PROPER 3-WAY SPLIT ---
    print("\n" + "="*70)
    print("SPLITTING DATA INTO TRAIN / VALIDATION / TEST")
    print("="*70)
    
    # First split: separate test set (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    
    # Second split: split remaining 90% into train (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)
    
    print(f"Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Scale the data using StandardScaler
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    # --- 3. Convert to PyTorch Tensors & Create DataLoaders ---
    X_train_tensor = torch.from_numpy(X_train_scaled).float()
    y_train_tensor = torch.from_numpy(y_train_scaled).float()
    X_val_tensor = torch.from_numpy(X_val_scaled).float()
    y_val_tensor = torch.from_numpy(y_val_scaled).float()
    X_test_tensor = torch.from_numpy(X_test_scaled).float()
    y_test_tensor = torch.from_numpy(y_scaler.transform(y_test)).float()

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

    # Train using VALIDATION set (not test set!)
    model.learn(train_loader, val_loader, num_epochs=500, y_scaler=y_scaler, early_stopping_patience=50)

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
    calculate_grouped_rmse(y_test, predictions)

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

    # Per-target metrics for both validation and test sets
    print("\n" + "="*70)
    print("VALIDATION VS TEST SET PERFORMANCE METRICS")
    print("="*70)
    
    # Get validation set predictions
    val_predictions_scaled = model.predict(X_val_tensor)
    val_predictions = y_scaler.inverse_transform(val_predictions_scaled)
    
    print("\nValidation Set Metrics:")
    print("-" * 50)
    for i, target in enumerate(OUTPUT_TARGETS):
        val_mae = mean_absolute_error(y_val[:, i], val_predictions[:, i])
        val_r2 = r2_score(y_val[:, i], val_predictions[:, i])
        val_rmse = np.sqrt(np.mean((y_val[:, i] - val_predictions[:, i])**2))
        print(f"{target:12s} | MAE: {val_mae:8.4f} | RMSE: {val_rmse:8.4f} | R²: {val_r2:7.4f}")
    
    print("\nTest Set Metrics:")
    print("-" * 50)
    for i, target in enumerate(OUTPUT_TARGETS):
        test_mae = mean_absolute_error(y_test[:, i], predictions[:, i])
        test_r2 = r2_score(y_test[:, i], predictions[:, i])
        test_rmse = np.sqrt(np.mean((y_test[:, i] - predictions[:, i])**2))
        print(f"{target:12s} | MAE: {test_mae:8.4f} | RMSE: {test_rmse:8.4f} | R²: {test_r2:7.4f}")
    
    # Define save directory for plots
    save_dir = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\Thesis - Tactile Sensor\models paramters\NN"
    
    # Extract version number from the CSV filename (for plots)
    version = CSV_FILENAMES[0].split("sensor ")[1].split("\\")[0]
    
    # Plot comparison of validation vs test performance
    plt.figure(figsize=(12, 6))
    x = np.arange(len(OUTPUT_TARGETS))
    width = 0.35
    
    # Calculate R² scores
    val_r2_scores = [r2_score(y_val[:, i], val_predictions[:, i]) for i in range(len(OUTPUT_TARGETS))]
    test_r2_scores = [r2_score(y_test[:, i], predictions[:, i]) for i in range(len(OUTPUT_TARGETS))]
    
    plt.bar(x - width/2, val_r2_scores, width, label='Validation R²', alpha=0.8)
    plt.bar(x + width/2, test_r2_scores, width, label='Test R²', alpha=0.8)
    
    plt.xlabel('Output Variables')
    plt.ylabel('R² Score')
    plt.title('Validation vs Test Set Performance')
    plt.xticks(x, OUTPUT_TARGETS, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save comparison plot with version
    comparison_fig = plt.gcf()
    comparison_fig.savefig(os.path.join(save_dir, f'validation_vs_test_comparison_{version}.png'))
    plt.close()

    # --- 6. Save the Model and Scalers ---
    print("\n" + "="*70)
    print("SAVING MODEL AND SCALERS")
    print("="*70)

    # Extract version number from the CSV filename
    version = CSV_FILENAMES[0].split("sensor ")[1].split("\\")[0]  # This will extract "v2" from the filename
    
    # Define save directory
    save_dir = r"C:\Users\aurir\OneDrive\Desktop\Thesis- Biorobotics Lab\Thesis - Tactile Sensor\models paramters\NN"
    
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
    history_fig.savefig(os.path.join(save_dir, f'training_history_{version}.png'))
    summary_fig.savefig(os.path.join(save_dir, f'predictions_summary_{version}.png'))

    print("\nTraining complete!")