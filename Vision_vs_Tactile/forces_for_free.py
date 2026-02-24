"""
Forces for Free — Vision-Based Force Prediction (CNN + Transformer)

Pipeline:
1. Extract image sequence from ROS bag (with timestamps)
2. Segment purple gripper (HSV masking, zero out background)
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
SEQ_LEN = 4            # Number of consecutive frames per sequence
IMAGE_SIZE = 112        # Resize for ResNet input (use 224 for full training)
TARGETS = ["fx_R", "fy_R", "fz_R", "fx_L", "fy_L", "fz_L"]
# If True, load pre-saved segmented JPEGs from the seg_path column in
# metadata.csv (written by direct_jpeg_extractor.py). Much faster than
# re-extracting the bag and re-running HSV segmentation every run.
USE_PRECOMPUTED_SEGMENTATION = True

# --- Model ---
USE_LSTM = True        # True = LSTM, False = Transformer
D_MODEL = 512           # ResNet18 feature dim (fixed)
N_HEADS = 4
N_TRANSFORMER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# --- LSTM-specific ---
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2

# --- Training ---
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
FREEZE_CNN_EPOCHS = 2   # Freeze ResNet backbone for first N epochs
EARLY_STOP_PATIENCE = 15
VAL_SPLIT = 0.2

# --- Purple segmentation HSV bounds (same as direct_jpeg_extractor.py) ---
PURPLE_LOWER = np.array([125, 50, 50])
PURPLE_UPPER = np.array([160, 255, 255])

# --- ImageNet normalization ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== PREPROCESSING ===============================

def segment_purple(image_bgr: np.ndarray) -> np.ndarray:
    """
    Segment the purple gripper and zero out background.

    Args:
        image_bgr: BGR image (H, W, 3) from OpenCV

    Returns:
        Masked BGR image with non-purple pixels set to black
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PURPLE_LOWER, PURPLE_UPPER)

    # Close: fills small noise gaps within each rib (~10-20 px) without
    # merging separate arms or overriding larger structural rib gaps
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    # Open: removes isolated speckles smaller than the kernel
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Keep all arm-sized contours (both fin ray arms), not just the largest one
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        min_area = image_bgr.shape[0] * image_bgr.shape[1] * 0.003  # 0.3% of image
        mask = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(mask, [cnt], -1, 255, -1)

    # Apply mask: zero out background
    masked = image_bgr.copy()
    masked[mask == 0] = 0
    return masked


def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Full preprocessing: segment purple -> resize -> normalize -> tensor.

    Args:
        image_bgr: BGR image (H, W, 3)

    Returns:
        Tensor (3, 224, 224) normalized for ImageNet
    """
    masked = segment_purple(image_bgr)

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


def preprocess_segmented_image(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Preprocess an already-segmented (background-zeroed) BGR image.
    Skips HSV segmentation — just converts colour, resizes, and normalises.

    Args:
        image_bgr: BGR image (H, W, 3) already masked by direct_jpeg_extractor

    Returns:
        Tensor (3, IMAGE_SIZE, IMAGE_SIZE) normalised for ImageNet
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return normalize(tensor)


# ========================== IMAGE LOADER ==============================

_ENC_INFO = {
    'rgb8': (np.uint8, 3),
    'bgr8': (np.uint8, 3),
    'rgba8': (np.uint8, 4),
    'bgra8': (np.uint8, 4),
    'mono8': (np.uint8, 1),
    '8UC1': (np.uint8, 1),
    '8UC3': (np.uint8, 3),
    '16UC1': (np.uint16, 1),
}


def _raw_to_bgr(data, height, width, encoding):
    """Convert raw Image data to a BGR numpy array."""
    info = _ENC_INFO.get(encoding)
    if info is None:
        raise ValueError(f"Unsupported encoding: {encoding}")
    dtype, channels = info
    img = np.frombuffer(data, dtype=dtype).reshape(height, width, channels)
    if encoding in ('rgb8',):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding in ('rgba8',):
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif encoding in ('bgra8',):
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif encoding in ('mono8', '8UC1'):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def extract_images_from_bag(bag_path: Path):
    """
    Extract all raw Image frames from a ROS bag with timestamps.

    Returns:
        List of dicts: {'index', 'image', 'timestamp_sec'}
    """
    print(f"Reading bag file: {bag_path}")
    images = []

    with AnyReader([bag_path]) as reader:
        # Auto-detect Image topic (raw, not compressed)
        image_connections = [
            c for c in reader.connections
            if c.msgtype in ('sensor_msgs/msg/Image', 'sensor_msgs/Image')
        ]
        if not image_connections:
            raise ValueError("No sensor_msgs/Image topic found in bag file.")

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
            image = _raw_to_bgr(
                bytes(msg.data), msg.height, msg.width, msg.encoding
            )

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

    def __init__(self, images, metadata_df, seq_len=SEQ_LEN, targets=TARGETS,
                 use_precomputed=USE_PRECOMPUTED_SEGMENTATION):
        """
        Args:
            images: List of dicts from extract_images_from_bag(), or None when
                    use_precomputed=True (images loaded from seg_path column).
            metadata_df: DataFrame with columns [image_idx, time, fx, fy, fz, ...]
                         Must contain 'seg_path' when use_precomputed=True.
            seq_len: Number of consecutive frames per sequence
            targets: List of target column names
            use_precomputed: If True, load pre-saved segmented JPEGs from seg_path.
        """
        self.seq_len = seq_len
        self.targets = targets
        self.use_precomputed = use_precomputed

        if use_precomputed:
            if 'seg_path' not in metadata_df.columns:
                raise ValueError(
                    "'seg_path' column not found in metadata.csv. "
                    "Run direct_jpeg_extractor.py first or set "
                    "USE_PRECOMPUTED_SEGMENTATION = False."
                )
            self.image_lookup = None
            self.metadata = (
                metadata_df.dropna(subset=['seg_path'])
                .sort_values('time')
                .reset_index(drop=True)
            )
        else:
            # Build image lookup from bag-extracted frames
            self.image_lookup = {img['index']: img['image'] for img in images}
            valid_indices = set(self.image_lookup.keys())
            self.metadata = metadata_df[
                metadata_df['image_idx'].isin(valid_indices)
            ].sort_values('time').reset_index(drop=True)

        # Valid starting indices for sequences (need seq_len consecutive frames)
        self.valid_starts = list(range(len(self.metadata) - seq_len + 1))

        mode = "pre-segmented JPEGs" if use_precomputed else "on-the-fly from bag"
        print(f"  Dataset: {len(self.valid_starts)} sequences "
              f"({len(self.metadata)} frames, seq_len={seq_len}, mode={mode})")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len

        # Build sequence of preprocessed frames
        frames = []
        for i in range(start, end):
            row = self.metadata.iloc[i]
            if self.use_precomputed:
                image_bgr = cv2.imread(str(row['seg_path']))
                if image_bgr is None:
                    raise FileNotFoundError(
                        f"Could not read segmented image: {row['seg_path']}"
                    )
                tensor = preprocess_segmented_image(image_bgr)
            else:
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


class ForcesForFreeLSTM(nn.Module):
    """
    CNN + LSTM model for force prediction from image sequences.

    Architecture:
        1. Shared ResNet18 backbone (per-frame) -> 512-dim features
        2. LSTM on the sequence of features
        3. Last hidden state -> MLP head -> force predictions
    """

    def __init__(self, num_targets=3, hidden_size=LSTM_HIDDEN_SIZE,
                 num_layers=LSTM_NUM_LAYERS, dropout=DROPOUT):
        super().__init__()

        # CNN backbone: ResNet18 without final FC
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)

        self.lstm = nn.LSTM(
            input_size=D_MODEL,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_targets),
        )

    def freeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
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

        x = x.view(B * S, C, H, W)
        features = self.cnn(x)                   # (B*S, 512, 1, 1)
        features = features.view(B, S, D_MODEL)  # (B, S, 512)

        # LSTM
        lstm_out, _ = self.lstm(features)        # (B, S, hidden_size)

        # Take last time step
        last = lstm_out[:, -1, :]                # (B, hidden_size)

        out = self.head(last)                    # (B, num_targets)
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
    n_cols = min(n_targets, 3)
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, (ax, name) in enumerate(zip(axes, target_names)):
        pred = predictions[:, i]
        gt = ground_truth[:, i]

        ax.scatter(gt, pred, alpha=0.3, s=10, color='#005c7f')

        # Perfect prediction line
        lims = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.8)

        ax.set_xlabel(f'Ground Truth {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} [N]')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    # Hide unused subplots
    for j in range(n_targets, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Predictions vs Ground Truth', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ============================== MAIN ==================================

def main():
    arch_name = "LSTM" if USE_LSTM else "Transformer"
    print("=" * 60)
    print("FORCES FOR FREE - Vision-Based Force Prediction")
    print(f"CNN (ResNet18) + {arch_name} -> [{', '.join(TARGETS)}]")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    bag_file = DATASET_DIR / f"{TEST_NUM}_rgb_raw.bag"
    metadata_file = DATASET_DIR / "processed_data" / "metadata.csv"
    output_dir = DATASET_DIR / "processed_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract images from ROS bag (skipped when using pre-saved segmentation)
    if USE_PRECOMPUTED_SEGMENTATION:
        print("\n--- Step 1: Skipping bag extraction (USE_PRECOMPUTED_SEGMENTATION=True) ---")
        images = None
    else:
        print("\n--- Step 1: Extract images ---")
        images = extract_images_from_bag(bag_file)

    # 2. Load metadata (force labels + seg_path)
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

    # Chronological train/val split (no shuffling — time series)
    n_val = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_dataset = torch.utils.data.Subset(dataset, range(0, n_train))
    val_dataset = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
    print(f"  Train: {n_train}, Val: {n_val} (chronological split)")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # 4. Create model
    print(f"\n--- Step 4: Create model ({arch_name}) ---")
    if USE_LSTM:
        model = ForcesForFreeLSTM(
            num_targets=len(TARGETS),
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
        ).to(DEVICE)
    else:
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
    best_model_path = output_dir / f"forces_for_free_{arch_name.lower()}_best.pt"

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
