"""
Forces for Free — Vision-Based Force Prediction (CNN + Transformer)

Pipeline:
1. Extract image sequence from ROS bag (with timestamps)
2. Segment green gripper (HSV masking, zero out background)
3. Shared ResNet18 CNN feature extractor (512-dim per frame)
4. Transformer encoder on the sequence of features
5. Last output token -> MLP head -> [fx, fy, fz] prediction

Author: Aurora Ruggeri
"""

import sys
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from rosbags.highlevel import AnyReader

# =========================== CONFIGURATION ===========================

TEST_NUM = 51011002
DATASET_DIR = Path(
    rf"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test {TEST_NUM} - sensor v5"
)

# --- Data ---
SEQ_LEN = 8            # Number of consecutive frames per sequence
IMAGE_SIZE = 224        # Resize for ResNet input
TARGETS = ["fx", "fy", "fz"]

# --- Model ---
D_MODEL = 512           # ResNet18 feature dim (fixed)
N_HEADS = 4
N_TRANSFORMER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# --- Training ---
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
FREEZE_CNN_EPOCHS = 5   # Freeze ResNet backbone for first N epochs
EARLY_STOP_PATIENCE = 15
VAL_SPLIT = 0.2

# --- Green segmentation HSV bounds (same as direct_jpeg_extractor.py) ---
GREEN_LOWER = np.array([35, 40, 40])
GREEN_UPPER = np.array([85, 255, 255])

# --- ImageNet normalization ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== PREPROCESSING ===============================

def segment_green(image_bgr: np.ndarray) -> np.ndarray:
    """
    Segment the green gripper and zero out background.

    Args:
        image_bgr: BGR image (H, W, 3) from OpenCV

    Returns:
        Masked BGR image with non-green pixels set to black
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find largest contour to clean up noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    # Apply mask: zero out background
    masked = image_bgr.copy()
    masked[mask == 0] = 0
    return masked


def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Full preprocessing: segment green -> resize -> normalize -> tensor.

    Args:
        image_bgr: BGR image (H, W, 3)

    Returns:
        Tensor (3, 224, 224) normalized for ImageNet
    """
    masked = segment_green(image_bgr)

    # BGR -> RGB
    rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    # Resize
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

    # To float tensor [0, 1]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    # ImageNet normalization
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = normalize(tensor)

    return tensor


# ========================== IMAGE LOADER ==============================

def extract_images_from_bag(bag_path: Path):
    """
    Extract all CompressedImage frames from a ROS bag with timestamps.

    Returns:
        List of dicts: {'index', 'image', 'timestamp_sec'}
    """
    print(f"Reading bag file: {bag_path}")
    images = []

    with AnyReader([bag_path]) as reader:
        # Auto-detect CompressedImage topic
        image_connections = [
            c for c in reader.connections
            if 'CompressedImage' in c.msgtype
        ]
        if not image_connections:
            raise ValueError("No CompressedImage topic found in bag file.")

        topic = image_connections[0].topic
        print(f"  Using topic: {topic}")

        msg_count = sum(
            c.msgcount for c in image_connections
            if hasattr(c, 'msgcount') and c.msgcount
        )

        idx = 0
        for connection, timestamp, rawdata in tqdm(
            reader.messages(connections=image_connections),
            total=msg_count if msg_count else None,
            desc="Extracting images"
        ):
            msg = reader.deserialize(rawdata, connection.msgtype)
            jpeg_bytes = bytes(msg.data)
            img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image is not None:
                images.append({
                    'index': idx,
                    'image': image,
                    'timestamp_sec': timestamp / 1e9,
                })
                idx += 1

    print(f"  Extracted {len(images)} images")
    return images


# ============================ DATASET =================================

class ForceImageSequenceDataset(Dataset):
    """
    Dataset of image sequences with force labels.

    Each sample is a sequence of `seq_len` consecutive frames.
    The target is [fx, fy, fz] of the LAST frame in the sequence.
    """

    def __init__(self, images, metadata_df, seq_len=SEQ_LEN, targets=TARGETS):
        """
        Args:
            images: List of dicts from extract_images_from_bag()
            metadata_df: DataFrame with columns [image_idx, time, fx, fy, fz, ...]
            seq_len: Number of consecutive frames per sequence
            targets: List of target column names
        """
        self.seq_len = seq_len
        self.targets = targets

        # Build image lookup
        self.image_lookup = {img['index']: img['image'] for img in images}

        # Only keep metadata rows that have images
        valid_indices = set(self.image_lookup.keys())
        self.metadata = metadata_df[
            metadata_df['image_idx'].isin(valid_indices)
        ].sort_values('time').reset_index(drop=True)

        # Valid starting indices for sequences (need seq_len consecutive frames)
        self.valid_starts = list(range(len(self.metadata) - seq_len + 1))

        print(f"  Dataset: {len(self.valid_starts)} sequences "
              f"({len(self.metadata)} frames, seq_len={seq_len})")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len

        # Build sequence of preprocessed frames
        frames = []
        for i in range(start, end):
            row = self.metadata.iloc[i]
            image_bgr = self.image_lookup[int(row['image_idx'])]
            tensor = preprocess_image(image_bgr)
            frames.append(tensor)

        # Stack to (seq_len, 3, H, W)
        sequence = torch.stack(frames, dim=0)

        # Target: force labels of the last frame
        last_row = self.metadata.iloc[end - 1]
        target = torch.tensor(
            [float(last_row[t]) for t in self.targets],
            dtype=torch.float32
        )

        return sequence, target


# ============================== MODEL =================================

class ForcesForFree(nn.Module):
    """
    CNN + Transformer model for force prediction from image sequences.

    Architecture:
        1. Shared ResNet18 backbone (per-frame) -> 512-dim features
        2. Learnable positional encoding
        3. Transformer encoder (4 layers, 8 heads)
        4. Take last output token -> MLP head -> [fx, fy, fz]
    """

    def __init__(self, num_targets=3, seq_len=SEQ_LEN,
                 n_layers=N_TRANSFORMER_LAYERS, n_heads=N_HEADS,
                 dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT):
        super().__init__()

        self.seq_len = seq_len

        # CNN backbone: ResNet18 without final FC
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, D_MODEL) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_targets),
        )

    def freeze_cnn(self):
        """Freeze CNN backbone weights."""
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        """Unfreeze CNN backbone weights for fine-tuning."""
        for param in self.cnn.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, 3, H, W) image sequence

        Returns:
            (B, num_targets) force predictions
        """
        B, S, C, H, W = x.shape

        # Reshape to process all frames through CNN at once
        x = x.view(B * S, C, H, W)             # (B*S, 3, H, W)
        features = self.cnn(x)                   # (B*S, 512, 1, 1)
        features = features.view(B, S, D_MODEL)  # (B, S, 512)

        # Add positional encoding
        features = features + self.pos_embedding[:, :S, :]

        # Transformer encoder
        features = self.transformer(features)     # (B, S, 512)

        # Take last token
        last_token = features[:, -1, :]           # (B, 512)

        # Predict forces
        out = self.head(last_token)               # (B, num_targets)
        return out


# =========================== TRAINING =================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set, return average loss."""
    model.eval()
    total_loss = 0.0

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        predictions = model(sequences)
        loss = criterion(predictions, targets)
        total_loss += loss.item() * sequences.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def compute_metrics(model, loader, device, targets_names):
    """Compute per-target MAE and RMSE on a dataset."""
    model.eval()
    all_preds = []
    all_targets = []

    for sequences, targets in loader:
        sequences = sequences.to(device)
        preds = model(sequences).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = {}
    for i, name in enumerate(targets_names):
        errors = all_preds[:, i] - all_targets[:, i]
        metrics[name] = {
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors ** 2)),
        }

    return metrics, all_preds, all_targets


# =========================== PLOTTING =================================

def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Train Loss', linewidth=1.5)
    ax.plot(val_losses, label='Val Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_predictions_vs_gt(predictions, ground_truth, target_names, save_path):
    """Scatter plots: predicted vs ground truth for each target."""
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, target_names)):
        pred = predictions[:, i]
        gt = ground_truth[:, i]

        ax.scatter(gt, pred, alpha=0.3, s=10, color='#005c7f')

        # Perfect prediction line
        lims = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.8)

        ax.set_xlabel(f'Ground Truth {name}')
        ax.set_ylabel(f'Predicted {name}')
        unit = 'N'
        ax.set_title(f'{name} [{unit}]')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    fig.suptitle('Predictions vs Ground Truth', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ============================== MAIN ==================================

def main():
    print("=" * 60)
    print("FORCES FOR FREE - Vision-Based Force Prediction")
    print(f"CNN (ResNet18) + Transformer -> [{', '.join(TARGETS)}]")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    bag_file = DATASET_DIR / f"{TEST_NUM}_rgb_compressed.bag"
    metadata_file = DATASET_DIR / "processed_data" / "metadata.csv"
    output_dir = DATASET_DIR / "processed_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract images from ROS bag
    print("\n--- Step 1: Extract images ---")
    images = extract_images_from_bag(bag_file)

    # 2. Load metadata (force labels)
    print("\n--- Step 2: Load metadata ---")
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            f"Run direct_jpeg_extractor.py first to generate it."
        )
    metadata = pd.read_csv(metadata_file)
    print(f"  Loaded {len(metadata)} metadata rows")
    print(f"  Columns: {list(metadata.columns)}")

    # Verify target columns exist
    missing = [t for t in TARGETS if t not in metadata.columns]
    if missing:
        raise ValueError(f"Missing target columns in metadata: {missing}")

    # 3. Create dataset
    print("\n--- Step 3: Create dataset ---")
    dataset = ForceImageSequenceDataset(
        images=images,
        metadata_df=metadata,
        seq_len=SEQ_LEN,
        targets=TARGETS,
    )

    # Train/val split
    n_val = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # 4. Create model
    print("\n--- Step 4: Create model ---")
    model = ForcesForFree(
        num_targets=len(TARGETS),
        seq_len=SEQ_LEN,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")

    # 5. Training
    print("\n--- Step 5: Training ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    # Freeze CNN for initial epochs
    model.freeze_cnn()
    print(f"  CNN frozen for first {FREEZE_CNN_EPOCHS} epochs")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_path = output_dir / "forces_for_free_best.pt"

    for epoch in range(1, EPOCHS + 1):
        # Unfreeze CNN after warmup
        if epoch == FREEZE_CNN_EPOCHS + 1:
            model.unfreeze_cnn()
            print(f"  CNN unfrozen at epoch {epoch}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {current_lr:.2e}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # 6. Evaluation
    print("\n--- Step 6: Evaluation ---")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    metrics, predictions, ground_truth = compute_metrics(
        model, val_loader, DEVICE, TARGETS
    )

    print("\n  Validation Metrics:")
    print(f"  {'Target':<8} {'MAE':>10} {'RMSE':>10}")
    print(f"  {'-'*30}")
    for name, m in metrics.items():
        print(f"  {name:<8} {m['mae']:>10.4f} {m['rmse']:>10.4f}")

    # 7. Save plots
    print("\n--- Step 7: Save plots ---")
    plot_training_curves(
        train_losses, val_losses,
        save_path=output_dir / "forces_for_free_training_curves.png"
    )
    plot_predictions_vs_gt(
        predictions, ground_truth, TARGETS,
        save_path=output_dir / "forces_for_free_pred_vs_gt.png"
    )
    print(f"  Saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Model saved:   {best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
