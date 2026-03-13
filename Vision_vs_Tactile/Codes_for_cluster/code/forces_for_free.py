#!/usr/bin/env python3
"""
Forces for Free — Vision-Based Force Prediction (CNN + Transformer)

Supports combining multiple test datasets for training.
Each dataset lives in:  <code_dir>/test <TEST_NUM> - sensor v5/processed_data/

Speed optimizations:
- Mixed precision training (torch.amp)
- Image cache: pre-loads all images into RAM
- torch.compile() for GPU kernel fusion (PyTorch 2.0+)
- non_blocking data transfers

Output files are tagged with the SLURM job ID (or 'local' if run outside SLURM).

Author: Aurora Ruggeri
"""

import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import models, transforms
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


# =========================== CONFIGURATION ===========================

# List all test numbers you want to combine for training.

#TEST_NUMS = [51011003, 51011006, 51011007, 51011008, 51011009] # flat
TEST_NUMS = [52021001, 52021002, 52021003, 52021004, 52021005, 52021006, 52021007, 52021008, 52021009, 52021010, 52021011] # round
TIME_SERIES_TEST_NUM = TEST_NUMS[0]

# SLURM job ID — tags all output files so runs don't overwrite each other
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

# --- Data ---
SEQ_LEN    = 8
IMAGE_SIZE = 224

SIDE_TARGETS = {
    "right": ["fx_R_surf", "fy_R_surf", "fz_R_surf"],
    "left":  ["fx_L_surf", "fy_L_surf", "fz_L_surf"],
}
ALL_TARGETS = SIDE_TARGETS["right"] + SIDE_TARGETS["left"]

# --- Model ---
D_MODEL         = 512
N_HEADS         = 8
N_TRANSFORMER_LAYERS = 2
DIM_FEEDFORWARD = 1024
DROPOUT         = 0.25

# --- Training ---
BATCH_SIZE         = 32
EPOCHS             = 100
LR                 = 2e-5
WEIGHT_DECAY       = 2e-3
FREEZE_CNN_EPOCHS  = 0  # Number of initial epochs to freeze the entire CNN backbone (ResNet18).
FREEZE_RESNET_LAYERS = 3   # 0 disables; 1..4 freezes layer1..layer4 (plus stem)
EARLY_STOP_PATIENCE = 3
TRAIN_SPLIT        = 0.70
VAL_SPLIT          = 0.15
TEST_SPLIT         = 0.15

# --- Speed settings ---
CACHE_IMAGES = True   # Pre-load all images into RAM (recommended)
USE_COMPILE  = False  # keep False for maximum run-to-run reproducibility
SEED = 777
DETERMINISTIC = True

# --- Stress-test augmentation (optional) ---
# Enable to test robustness to partial occlusions and lighting changes.
ENABLE_STRESS_TEST_AUG = False
STRESS_AUG_SEED = 42
STRESS_OCCLUSION_PROB = 0.65
STRESS_LIGHT_PROB = 0.75
STRESS_OCCLUSION_SEGMENT_RATIO_RANGE = (0.12, 0.70)  # fraction of segmented gripper area
STRESS_OCCLUSION_VERTICES_RANGE = (5, 9)
STRESS_CONTRAST_RANGE = (0.65, 1.35)
STRESS_BRIGHTNESS_SHIFT = (-35, 35)
STRESS_SAVE_PREVIEW_FRAMES = 2

# --- ImageNet normalization ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DEVICE_TYPE = DEVICE.type  # "cuda" or "cpu"

PALETTE = ["#292f56", "#3d7be0", "#008780", "#44b155", "#d6c52e"]
CLR_NAVY, CLR_BLUE, CLR_TEAL, CLR_GREEN, CLR_YELLOW = PALETTE
# Plot-style colors
CLR_FFF_DOTS = "#008780"         # petrol blue-green
CLR_TS_GT_COMMON = "#d6c52e"     # shared GT color across pipelines
CLR_TS_PRED_FFF = "#008780"      # FFF predictions
PLOT_TITLE_TAG = " [Stress]" if ENABLE_STRESS_TEST_AUG else ""
STRESS_PREVIEW_DIR = Path(__file__).parent.parent / "outputs" / "forces_for_free" / "stress_examples"
_STRESS_PREVIEW_COUNT = 0


# ======================== PREPROCESSING ===============================

_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def set_global_reproducibility(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            print(f"WARNING: could not enable full deterministic algorithms: {exc}")


def resolve_image_path(raw_path: str, dataset_root: Path) -> Path:
    raw = str(raw_path).strip()
    if not raw:
        return dataset_root / "__missing__"

    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p

    norm = raw.replace("\\", "/")
    name = Path(norm).name
    candidates = [
        (dataset_root / norm),
        (dataset_root / name),
        (dataset_root / "segmented_images" / name),
        (dataset_root.parent / "processed_data" / "segmented_images" / name),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (dataset_root / "segmented_images" / name).resolve()


def _make_stress_rng(test_num: int, frame_idx: int) -> np.random.Generator:
    # Deterministic per frame to keep cache/no-cache behavior consistent.
    seed = (
        int(STRESS_AUG_SEED)
        + 1_000_003 * int(test_num)
        + 9_176 * int(frame_idx)
    ) % (2**32)
    return np.random.default_rng(seed)


def _build_polygon_mask_over_segment(
    seg_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    seg_pixels = int(seg_mask.sum())
    if seg_pixels == 0:
        return np.zeros_like(seg_mask, dtype=bool), 0.0

    ys, xs = np.where(seg_mask)
    target_ratio = float(rng.uniform(*STRESS_OCCLUSION_SEGMENT_RATIO_RANGE))
    target_pixels = max(16, int(target_ratio * seg_pixels))

    h, w = seg_mask.shape
    best_mask = np.zeros_like(seg_mask, dtype=bool)
    best_diff = float("inf")

    for _ in range(12):
        center_i = int(rng.integers(0, len(xs)))
        cx = int(xs[center_i])
        cy = int(ys[center_i])

        n_vertices = int(rng.integers(
            STRESS_OCCLUSION_VERTICES_RANGE[0],
            STRESS_OCCLUSION_VERTICES_RANGE[1] + 1,
        ))
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n_vertices))

        base_radius = np.sqrt(target_pixels / np.pi)
        radii = base_radius * rng.uniform(0.65, 1.45, size=n_vertices)

        pts = np.stack(
            [
                cx + radii * np.cos(angles),
                cy + radii * np.sin(angles),
            ],
            axis=1,
        )
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

        poly_mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask_u8, [poly], 255)
        poly_mask = poly_mask_u8 > 0
        on_gripper = poly_mask & seg_mask
        occ_pixels = int(on_gripper.sum())

        diff = abs(occ_pixels - target_pixels)
        if occ_pixels > 0 and diff < best_diff:
            best_diff = diff
            best_mask = on_gripper

        if occ_pixels >= int(0.85 * target_pixels):
            break

    occ_ratio = float(best_mask.sum()) / float(seg_pixels)
    return best_mask, occ_ratio


def _save_stress_preview(
    original_bgr: np.ndarray,
    augmented_bgr: np.ndarray,
    test_num: int,
    frame_idx: int,
    occ_ratio: float,
):
    global _STRESS_PREVIEW_COUNT
    if _STRESS_PREVIEW_COUNT >= STRESS_SAVE_PREVIEW_FRAMES:
        return

    STRESS_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    panel = np.hstack([original_bgr, augmented_bgr])
    out_path = STRESS_PREVIEW_DIR / (
        f"stress_preview_test{test_num}_frame{frame_idx:06d}_occ{occ_ratio:.2f}.jpg"
    )
    cv2.imwrite(str(out_path), panel)
    _STRESS_PREVIEW_COUNT += 1


def _apply_stress_augmentation(
    image_bgr: np.ndarray,
    test_num: int,
    frame_idx: int,
) -> np.ndarray:
    rng = _make_stress_rng(test_num=test_num, frame_idx=frame_idx)
    original = image_bgr.copy()
    out = original.copy()
    seg_mask = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) > 5
    occ_ratio = 0.0

    if rng.random() < STRESS_OCCLUSION_PROB:
        occ_mask, occ_ratio = _build_polygon_mask_over_segment(seg_mask=seg_mask, rng=rng)
        if np.any(occ_mask):
            occlusion_color = int(rng.integers(0, 40))
            out[occ_mask] = occlusion_color

    if rng.random() < STRESS_LIGHT_PROB:
        alpha = float(rng.uniform(*STRESS_CONTRAST_RANGE))
        beta = int(rng.integers(STRESS_BRIGHTNESS_SHIFT[0], STRESS_BRIGHTNESS_SHIFT[1] + 1))
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if ENABLE_STRESS_TEST_AUG and STRESS_SAVE_PREVIEW_FRAMES > 0:
        _save_stress_preview(
            original_bgr=original,
            augmented_bgr=out,
            test_num=test_num,
            frame_idx=frame_idx,
            occ_ratio=occ_ratio,
        )

    return out


def preprocess_image(
    image_bgr: np.ndarray,
    test_num: int,
    frame_idx: int,
) -> torch.Tensor:
    """BGR -> RGB -> resize -> tensor -> ImageNet normalize."""
    if ENABLE_STRESS_TEST_AUG:
        image_bgr = _apply_stress_augmentation(
            image_bgr=image_bgr,
            test_num=test_num,
            frame_idx=frame_idx,
        )

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return _normalize(tensor)


# ============================ DATASET =================================

class ForceImageSequenceDataset(Dataset):
    """
    Dataset for a SINGLE test folder.

    Each sample : seq_len consecutive frames (from the same recording)
    Target      : force labels of the LAST frame in the sequence

    Sequences never cross file/recording boundaries because the metadata
    is sorted by time and each dataset object covers exactly one folder.

    With cache=True all images are preprocessed once at startup and kept
    in RAM, so __getitem__ never touches the disk during training.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        dataset_root: Path,
        image_col: str,
        seq_len: int,
        targets: list[str],
        test_num: int,
        cache: bool = True,
    ):
        self.seq_len  = seq_len
        self.targets  = targets
        self.test_num = test_num
        self.image_col = image_col

        required_cols = ["time", image_col, *targets]
        missing_cols = [c for c in required_cols if c not in metadata_df.columns]
        if missing_cols:
            raise ValueError(f"dataset.csv missing columns: {missing_cols}")

        metadata_df = metadata_df.copy()
        metadata_df[self.image_col] = metadata_df[self.image_col].apply(
            lambda p: str(resolve_image_path(p, dataset_root))
        )

        # Drop metadata rows pointing to missing segmented images.
        exists_mask = metadata_df[self.image_col].apply(lambda p: Path(p).exists())
        missing_count = int((~exists_mask).sum())
        if missing_count > 0:
            print(
                f"  [{test_num}] Warning: {missing_count} missing segmented image(s) "
                "listed in metadata; dropping those rows."
            )
            metadata_df = metadata_df.loc[exists_mask].copy()

        self.metadata = (
            metadata_df.dropna(subset=[self.image_col])
            .sort_values("time")
            .reset_index(drop=True)
        )
        if len(self.metadata) < seq_len:
            raise ValueError(
                f"[{test_num}] Not enough valid frames after filtering missing images: "
                f"{len(self.metadata)} < seq_len ({seq_len})."
            )
        self.valid_starts = list(range(len(self.metadata) - seq_len + 1))

        # ---- optional image cache ---------------------------------------
        self.cache = cache
        self.image_cache: list = [None] * len(self.metadata)

        if cache:
            print(f"  [{test_num}] Pre-loading {len(self.metadata)} images into RAM ...")
            for i, row in enumerate(
                tqdm(self.metadata.itertuples(), total=len(self.metadata),
                     desc=f"  [{test_num}] Caching")
            ):
                img = cv2.imread(str(getattr(row, self.image_col)))
                if img is None:
                    raise FileNotFoundError(
                        f"Could not read image: {getattr(row, self.image_col)}"
                    )
                self.image_cache[i] = preprocess_image(
                    img,
                    test_num=self.test_num,
                    frame_idx=i,
                )
            print(f"  [{test_num}] Cache ready.")
        # -----------------------------------------------------------------

        print(f"  [{test_num}] {len(self.valid_starts)} sequences "
              f"({len(self.metadata)} frames)")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end   = start + self.seq_len

        frames = []
        for i in range(start, end):
            if self.cache:
                frames.append(self.image_cache[i])
            else:
                row = self.metadata.iloc[i]
                img = cv2.imread(str(row[self.image_col]))
                if img is None:
                    raise FileNotFoundError(f"Could not read image: {row[self.image_col]}")
                frames.append(
                    preprocess_image(
                        img,
                        test_num=self.test_num,
                        frame_idx=i,
                    )
                )

        sequence = torch.stack(frames, dim=0)   # (seq_len, C, H, W)

        last_row = self.metadata.iloc[end - 1]
        target   = torch.tensor(
            [float(last_row[t]) for t in self.targets], dtype=torch.float32
        )
        last_time = float(last_row["time"])
        return sequence, target, last_time, self.test_num


def load_dataset_for_test(
    test_num: int,
    image_col: str,
    seq_len: int,
    targets: list[str],
    cache: bool,
) -> ForceImageSequenceDataset:
    """Load and return a ForceImageSequenceDataset for one test number."""
    code_dir      = Path(__file__).parent
    dataset_dir   = code_dir / f"test {test_num} - sensor v5"
    processed_dir = dataset_dir / "processed_data"
    dataset_csv   = processed_dir / "dataset.csv"

    if not dataset_csv.exists():
        raise FileNotFoundError(f"[{test_num}] Dataset not found: {dataset_csv}")

    metadata = pd.read_csv(dataset_csv)
    missing  = [t for t in targets if t not in metadata.columns]
    if missing:
        raise ValueError(f"[{test_num}] Missing target columns: {missing}")

    return ForceImageSequenceDataset(
        metadata_df=metadata,
        dataset_root=processed_dir,
        image_col=image_col,
        seq_len=seq_len,
        targets=targets,
        test_num=test_num,
        cache=cache,
    )


# ============================== MODEL =================================

class ForcesForFree(nn.Module):
    """CNN (ResNet18) + Transformer -> last token -> MLP."""

    def __init__(
        self,
        num_targets: int,
        seq_len: int,
        n_layers: int = N_TRANSFORMER_LAYERS,
        n_heads: int = N_HEADS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.seq_len = seq_len

        resnet   = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, D_MODEL) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_targets),
        )

    def freeze_cnn(self):
        for p in self.cnn.parameters():
            p.requires_grad = False

    def unfreeze_cnn(self):
        for p in self.cnn.parameters():
            p.requires_grad = True

    def freeze_resnet_layers(self, n_blocks: int):
        """
        Freeze first n ResNet blocks in cnn sequential:
        stem(conv1+bn1) + layer1..layer4. n_blocks in [1..4].
        """
        if n_blocks <= 0:
            return
        n_blocks = min(4, int(n_blocks))
        for stem_idx in (0, 1):  # conv1, bn1
            for p in self.cnn[stem_idx].parameters():
                p.requires_grad = False
        block_indices = [4, 5, 6, 7]  # layer1..layer4
        for idx in block_indices[:n_blocks]:
            for p in self.cnn[idx].parameters():
                p.requires_grad = False

    def forward(self, x):
        B, S, C, H, W = x.shape
        feats = self.cnn(x.reshape(B * S, C, H, W)).reshape(B, S, D_MODEL)
        feats = feats + self.pos_embedding[:, :S, :]
        feats = self.transformer(feats)
        return self.head(feats[:, -1, :])


# =========================== TRAINING =================================

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, target_indices):
    model.train()
    total = 0.0
    for sequences, targets, _times, _test_nums in loader:
        sequences = sequences.to(device, non_blocking=True)
        targets = targets[:, target_indices].to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE):
            loss = criterion(model(sequences), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * sequences.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, target_indices):
    model.eval()
    total = 0.0
    for sequences, targets, _times, _test_nums in loader:
        sequences = sequences.to(device, non_blocking=True)
        targets = targets[:, target_indices].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE):
            total += criterion(model(sequences), targets).item() * sequences.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def compute_metrics(model, loader, device, target_names, target_indices):
    model.eval()
    all_preds, all_targets, all_times, all_test_nums = [], [], [], []
    for sequences, targets, times, test_nums in loader:
        with torch.amp.autocast(device_type=AMP_DEVICE_TYPE):
            preds = model(sequences.to(device, non_blocking=True)).cpu().float().numpy()
        all_preds.append(preds)
        all_targets.append(targets[:, target_indices].numpy())
        all_times.append(np.asarray(times))
        all_test_nums.append(np.asarray(test_nums))

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_times   = np.concatenate(all_times,   axis=0).astype(float)
    all_test_nums = np.concatenate(all_test_nums, axis=0).astype(int)

    metrics = {}
    for i, name in enumerate(target_names):
        err = all_preds[:, i] - all_targets[:, i]
        metrics[name] = {
            "mae":  float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
        }
    return metrics, all_preds, all_targets, all_times, all_test_nums


def save_test_predictions_csv(
    output_path: Path,
    target_names: list[str],
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    times: np.ndarray,
    test_nums: np.ndarray,
):
    rows = {
        "test_num": np.asarray(test_nums, dtype=int),
        "time": np.asarray(times, dtype=float),
    }
    for idx, target_name in enumerate(target_names):
        rows[f"{target_name}_true"] = ground_truth[:, idx]
        rows[f"{target_name}_pred"] = predictions[:, idx]

    pred_df = pd.DataFrame(rows).sort_values(["test_num", "time"])
    pred_df.to_csv(output_path, index=False)
    print(f"  Saved test predictions CSV: {output_path}")


# =========================== PLOTTING =================================

def plot_training_curves(train_losses, val_losses, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train Loss", linewidth=1.8, color=CLR_BLUE)
    ax.plot(val_losses,   label="Val Loss",   linewidth=1.8, color=CLR_GREEN)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss [N²]")
    ax.set_title(f"Training Curves{PLOT_TITLE_TAG}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predictions_vs_gt(predictions, ground_truth, target_names, save_path: Path):
    n_targets = len(target_names)
    n_cols    = min(n_targets, 3)
    n_rows    = (n_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, (ax, name) in enumerate(zip(axes, target_names)):
        pred, gt = predictions[:, i], ground_truth[:, i]
        mae      = np.mean(np.abs(pred - gt))
        rmse     = np.sqrt(np.mean((pred - gt) ** 2))
        ss_res   = np.sum((gt - pred) ** 2)
        ss_tot   = np.sum((gt - np.mean(gt)) ** 2)
        r2       = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

        ax.scatter(gt, pred, alpha=0.55, s=22, color=CLR_BLUE)
        lims = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
        ax.plot(lims, lims, "--", linewidth=1.4, alpha=0.95, color=CLR_YELLOW)
        ax.set_xlabel("Ground Truth [N]")
        ax.set_ylabel("Predicted [N]")
        ax.set_title(f"{name}\nMAE: {mae:.2f} | R\u00b2: {r2:.3f}", fontsize=12)
        ax.grid(alpha=0.28, linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")

    for j in range(n_targets, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Predicted vs Actual{PLOT_TITLE_TAG}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predictions_vs_gt_both_sides(
    right_predictions,
    right_ground_truth,
    left_predictions,
    left_ground_truth,
    save_path: Path,
):
    """Single 2x3 panel in FFTS style: top row left forces, bottom row right forces."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ordered_targets = [
        ("L", "fx", left_predictions, left_ground_truth),
        ("L", "fy", left_predictions, left_ground_truth),
        ("L", "fz", left_predictions, left_ground_truth),
        ("R", "fx", right_predictions, right_ground_truth),
        ("R", "fy", right_predictions, right_ground_truth),
        ("R", "fz", right_predictions, right_ground_truth),
    ]

    for ax, (side, name, predictions, ground_truth) in zip(axes.flatten(), ordered_targets):
        col_idx = ["fx", "fy", "fz"].index(name)
        pred = predictions[:, col_idx]
        gt = ground_truth[:, col_idx]
        mae = np.mean(np.abs(pred - gt))
        ss_res = np.sum((gt - pred) ** 2)
        ss_tot = np.sum((gt - np.mean(gt)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

        ax.scatter(gt, pred, alpha=0.55, s=22, color=CLR_FFF_DOTS)
        lims = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
        ax.plot(lims, lims, "--", linewidth=1.4, alpha=0.95, color=CLR_YELLOW)
        ax.set_xlabel("Ground Truth [N]")
        ax.set_ylabel("Predicted [N]")
        ax.set_title(f"{name}_{side}\nMAE: {mae:.2f} | R\u00b2: {r2:.3f}", fontsize=12)
        ax.grid(alpha=0.28, linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(f"Predicted vs Actual{PLOT_TITLE_TAG}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_time_series_both_sides(
    right_predictions,
    right_ground_truth,
    right_times,
    left_predictions,
    left_ground_truth,
    left_times,
    save_path: Path,
):
    """Single 2x3 panel of time-series: top row left forces, bottom row right forces."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=False)
    side_rows = [
        ("left", SIDE_TARGETS["left"], left_predictions, left_ground_truth, left_times),
        ("right", SIDE_TARGETS["right"], right_predictions, right_ground_truth, right_times),
    ]

    for row_idx, (side_name, target_names, predictions, ground_truth, times) in enumerate(side_rows):
        order = np.argsort(times)
        t = times[order]
        pred = predictions[order]
        gt = ground_truth[order]
        if t.size > 0:
            t0 = float(t[0])
            keep = (t >= t0) & (t <= t0 + 20.0)
            t = t[keep]
            pred = pred[keep]
            gt = gt[keep]
            t = t - t0  # relative time so x-axis starts at 0
        for col_idx, name in enumerate(target_names):
            ax = axes[row_idx, col_idx]
            ax.plot(
                t,
                gt[:, col_idx],
                color=CLR_TS_GT_COMMON,
                linewidth=2.4,
                alpha=0.95,
                label="Ground Truth",
            )
            ax.scatter(t, gt[:, col_idx], color=CLR_TS_GT_COMMON, s=12, alpha=0.7, zorder=5)
            ax.plot(
                t,
                pred[:, col_idx],
                color=CLR_TS_PRED_FFF,
                linewidth=2.4,
                alpha=0.95,
                label="Predicted",
            )
            ax.scatter(t, pred[:, col_idx], color=CLR_TS_PRED_FFF, s=12, alpha=0.7, zorder=5)
            ax.set_title(f"{name}_{'L' if side_name == 'left' else 'R'}", fontsize=12)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"{name} [N]")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right")

    fig.suptitle(f"Time Series: Predicted vs Ground Truth (Right + Left){PLOT_TITLE_TAG}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves_both_sides(
    right_train_losses,
    right_val_losses,
    left_train_losses,
    left_val_losses,
    save_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    panels = [
        ("L", left_train_losses, left_val_losses),
        ("R", right_train_losses, right_val_losses),
    ]
    for ax, (side, train_losses, val_losses) in zip(axes, panels):
        ax.plot(train_losses, label="Train Loss", linewidth=1.8, color=CLR_BLUE)
        ax.plot(val_losses, label="Val Loss", linewidth=1.8, color=CLR_GREEN)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss [N²]")
        ax.set_title(f"Training Curves ({side})")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Training Curves (Right + Left){PLOT_TITLE_TAG}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residual_histograms_both_sides(
    right_predictions,
    right_ground_truth,
    left_predictions,
    left_ground_truth,
    save_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ordered_targets = [
        ("L", "fx", left_predictions, left_ground_truth),
        ("L", "fy", left_predictions, left_ground_truth),
        ("L", "fz", left_predictions, left_ground_truth),
        ("R", "fx", right_predictions, right_ground_truth),
        ("R", "fy", right_predictions, right_ground_truth),
        ("R", "fz", right_predictions, right_ground_truth),
    ]
    for ax, (side, name, predictions, ground_truth) in zip(axes.flatten(), ordered_targets):
        col_idx = ["fx", "fy", "fz"].index(name)
        residuals = predictions[:, col_idx] - ground_truth[:, col_idx]
        ax.hist(residuals, bins=40, color=CLR_TEAL, alpha=0.8)
        ax.axvline(0.0, color=CLR_YELLOW, linestyle="--", linewidth=1.5)
        ax.set_title(f"Residuals {name}_{side}", fontsize=12)
        ax.set_xlabel("Prediction - Ground Truth [N]")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Residual Distributions (Right + Left){PLOT_TITLE_TAG}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_vs_gt_both_sides(
    right_predictions,
    right_ground_truth,
    left_predictions,
    left_ground_truth,
    save_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ordered_targets = [
        ("L", "fx", left_predictions, left_ground_truth),
        ("L", "fy", left_predictions, left_ground_truth),
        ("L", "fz", left_predictions, left_ground_truth),
        ("R", "fx", right_predictions, right_ground_truth),
        ("R", "fy", right_predictions, right_ground_truth),
        ("R", "fz", right_predictions, right_ground_truth),
    ]
    for ax, (side, name, predictions, ground_truth) in zip(axes.flatten(), ordered_targets):
        col_idx = ["fx", "fy", "fz"].index(name)
        gt = ground_truth[:, col_idx]
        residuals = predictions[:, col_idx] - gt
        ax.scatter(gt, residuals, alpha=0.35, s=12, color=CLR_TEAL)
        ax.axhline(0.0, color=CLR_YELLOW, linestyle="--", linewidth=1.4)
        ax.set_title(f"Residuals vs Ground Truth ({name}_{side})", fontsize=12)
        ax.set_xlabel("Ground Truth [N]")
        ax.set_ylabel("Prediction - Ground Truth [N]")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_abs_error_cdf_both_sides(
    right_predictions,
    right_ground_truth,
    left_predictions,
    left_ground_truth,
    save_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ordered_targets = [
        ("L", "fx", left_predictions, left_ground_truth),
        ("L", "fy", left_predictions, left_ground_truth),
        ("L", "fz", left_predictions, left_ground_truth),
        ("R", "fx", right_predictions, right_ground_truth),
        ("R", "fy", right_predictions, right_ground_truth),
        ("R", "fz", right_predictions, right_ground_truth),
    ]
    for ax, (side, name, predictions, ground_truth) in zip(axes.flatten(), ordered_targets):
        col_idx = ["fx", "fy", "fz"].index(name)
        e = np.abs(predictions[:, col_idx] - ground_truth[:, col_idx])
        xs = np.sort(e)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, linewidth=1.8, color=CLR_BLUE)
        ax.set_title(f"|Error| CDF ({name}_{side})", fontsize=12)
        ax.set_xlabel("|Prediction - Ground Truth| [N]")
        ax.set_ylabel("Fraction of samples")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Cumulative Absolute Error (Right + Left){PLOT_TITLE_TAG}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residual_histograms(predictions, ground_truth, target_names, save_path: Path):
    n_targets = len(target_names)
    n_cols = min(n_targets, 3)
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, (ax, name) in enumerate(zip(axes, target_names)):
        residuals = predictions[:, i] - ground_truth[:, i]
        ax.hist(residuals, bins=40, color=CLR_TEAL, alpha=0.8)
        ax.axvline(0.0, color=CLR_YELLOW, linestyle="--", linewidth=1.5)
        ax.set_title(f"Residuals {name}")
        ax.set_xlabel("Prediction - Ground Truth [N]")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)

    for j in range(n_targets, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Residual Distributions{PLOT_TITLE_TAG}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_vs_gt(predictions, ground_truth, target_names, save_path: Path):
    n_targets = len(target_names)
    n_cols = min(n_targets, 3)
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, (ax, name) in enumerate(zip(axes, target_names)):
        gt = ground_truth[:, i]
        residuals = predictions[:, i] - gt
        ax.scatter(gt, residuals, alpha=0.35, s=12, color=CLR_TEAL)
        ax.axhline(0.0, color=CLR_YELLOW, linestyle="--", linewidth=1.4)
        ax.set_title(f"Residuals vs Ground Truth ({name})")
        ax.set_xlabel("Ground Truth [N]")
        ax.set_ylabel("Prediction - Ground Truth [N]")
        ax.grid(alpha=0.25)

    for j in range(n_targets, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_abs_error_cdf(predictions, ground_truth, target_names, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(target_names):
        e = np.abs(predictions[:, i] - ground_truth[:, i])
        xs = np.sort(e)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, linewidth=1.8, label=name, color=PALETTE[i % len(PALETTE)])

    ax.set_title(f"Cumulative Absolute Error{PLOT_TITLE_TAG}")
    ax.set_xlabel("|Prediction - Ground Truth| [N]")
    ax.set_ylabel("Fraction of samples")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_time_series(predictions, ground_truth, times, target_names, save_path: Path):
    order = np.argsort(times)
    t = times[order]
    pred = predictions[order]
    gt = ground_truth[order]

    n_targets = len(target_names)
    fig, axes = plt.subplots(n_targets, 1, figsize=(12, 2.8 * n_targets), sharex=True)
    if n_targets == 1:
        axes = [axes]

    for i, name in enumerate(target_names):
        ax = axes[i]
        ax.plot(
            t,
            gt[:, i],
            color=CLR_NAVY,
            linestyle="-",
            linewidth=1.8,
            alpha=0.95,
            label="Ground Truth",
        )
        ax.plot(
            t,
            pred[:, i],
            color=CLR_GREEN,
            linestyle="-",
            linewidth=1.8,
            alpha=0.95,
            label="Predicted",
        )
        ax.set_ylabel(f"{name} [N]")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Time Series: Predicted vs Ground Truth{PLOT_TITLE_TAG}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================== MAIN ==================================

def build_non_leaking_splits(
    datasets: list[ForceImageSequenceDataset],
    train_split: float,
    val_split: float,
    test_split: float,
    seq_len: int,
):
    """
    Build train/val/test indices using absolute time boundaries so that both
    this pipeline and LightGBM share the same test-set time window.

    For each dataset the split boundaries are:
        t_train_end  = t_min + train_split  * duration
        gap_secs     = (seq_len - 1) * median_dt
        t_val_start  = t_train_end  + gap_secs
        t_val_end    = t_min + (train_split + val_split) * duration
        t_test_start = t_val_end    + gap_secs

    Each sequence is assigned by the timestamp of its LAST frame.
    """
    split_sum = train_split + val_split + test_split
    if not np.isclose(split_sum, 1.0):
        raise ValueError(
            f"TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must be 1.0 (got {split_sum:.4f})."
        )

    train_idx, val_idx, test_idx = [], [], []
    dropped_for_gap = 0
    offset = 0

    for ds in datasets:
        n_seq = len(ds)
        # last_time[i] = time of the last frame in sequence i
        last_times = ds.metadata["time"].iloc[seq_len - 1:].to_numpy(dtype=float)

        raw_times = ds.metadata["time"].to_numpy(dtype=float)
        sorted_raw = np.sort(raw_times)
        if sorted_raw[0] == sorted_raw[-1]:
            raise ValueError(f"[{ds.test_num}] All timestamps identical.")

        dt_median  = float(np.median(np.diff(sorted_raw)))
        gap_secs   = max(dt_median * (seq_len - 1), 0.0)

        # Use quantile so that exactly train_split fraction of CSV *rows* (not
        # time range) defines each boundary.
        t_train_end  = float(np.quantile(sorted_raw, train_split))
        t_val_start  = t_train_end  + gap_secs
        t_val_end    = float(np.quantile(sorted_raw, train_split + val_split))
        t_test_start = t_val_end    + gap_secs

        train_local = np.where(last_times <= t_train_end)[0].tolist()
        val_local   = np.where((last_times >= t_val_start) & (last_times <= t_val_end))[0].tolist()
        test_local  = np.where(last_times >= t_test_start)[0].tolist()

        if not train_local or not val_local or not test_local:
            # Temporal gaps in this dataset cause one split to be empty
            # (e.g. missing images create a hole in the val window).
            # Fall back to index-based split for this dataset.
            print(
                f"  [{ds.test_num}] WARNING: time-based split yielded empty split "
                f"(train={len(train_local)}, val={len(val_local)}, test={len(test_local)}), "
                f"falling back to index-based split."
            )
            gap = seq_len - 1
            usable = n_seq - 2 * gap
            if usable < 3:
                raise ValueError(
                    f"[{ds.test_num}] Not enough sequences ({n_seq}) even for index-based split."
                )
            n_train = int(usable * train_split)
            n_val   = int(usable * val_split)
            n_test  = usable - n_train - n_val
            train_local = list(range(0, n_train))
            val_start   = n_train + gap
            val_local   = list(range(val_start, val_start + n_val))
            test_start  = val_start + n_val + gap
            test_local  = list(range(test_start, test_start + n_test))
            print(
                f"  [{ds.test_num}] index-split fallback: "
                f"train={len(train_local)}  val={len(val_local)}  test={len(test_local)}"
            )
        else:
            print(
                f"  [{ds.test_num}] time-split: "
                f"train\u2264{t_train_end:.2f}s ({len(train_local)})  "
                f"val=[{t_val_start:.2f},{t_val_end:.2f}]s ({len(val_local)})  "
                f"test\u2265{t_test_start:.2f}s ({len(test_local)})"
            )

        train_idx.extend([offset + i for i in train_local])
        val_idx.extend([offset + i for i in val_local])
        test_idx.extend([offset + i for i in test_local])

        used_here = len(train_local) + len(val_local) + len(test_local)
        dropped_for_gap += n_seq - used_here
        offset += n_seq

    return train_idx, val_idx, test_idx, dropped_for_gap


def train_and_evaluate_side(
    side_name: str,
    targets: list[str],
    all_datasets: list[ForceImageSequenceDataset],
    out,
):
    print(f"\n--- Training side model: {side_name.upper()} -> {targets} ---")
    target_indices = [ALL_TARGETS.index(t) for t in targets]
    train_idx, val_idx, test_idx, dropped_for_gap = build_non_leaking_splits(
        datasets=all_datasets,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        seq_len=SEQ_LEN,
    )
    combined = ConcatDataset(all_datasets)
    train_dataset = Subset(combined, train_idx)
    val_dataset = Subset(combined, val_idx)
    test_dataset = Subset(combined, test_idx)
    print(
        f"  Split sizes -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)} "
        f"| Dropped(for anti-leakage gaps): {dropped_for_gap}"
    )

    loader_kwargs = dict(num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    # --- Create model ---
    model = ForcesForFree(num_targets=len(targets), seq_len=SEQ_LEN).to(DEVICE)

    if USE_COMPILE and hasattr(torch, "compile"):
        print("  Applying torch.compile() ...")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    def get_base_model():
        return model._orig_mod if hasattr(model, "_orig_mod") else model

    # --- Training ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler()

    get_base_model().freeze_cnn()
    print(f"  CNN frozen for first {FREEZE_CNN_EPOCHS} epochs")
    if FREEZE_RESNET_LAYERS > 0:
        get_base_model().freeze_resnet_layers(FREEZE_RESNET_LAYERS)
        print(f"  ResNet persistent freeze enabled: first {FREEZE_RESNET_LAYERS} blocks frozen")

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    best_model_path = out(f"forces_for_free_{side_name}_best.pt")
    best_state_dict = None

    try:
        best_model_path.unlink(missing_ok=True)
    except TypeError:
        if best_model_path.exists():
            best_model_path.unlink()

    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_CNN_EPOCHS + 1:
            get_base_model().unfreeze_cnn()
            if FREEZE_RESNET_LAYERS > 0:
                get_base_model().freeze_resnet_layers(FREEZE_RESNET_LAYERS)
            print(f"  CNN unfrozen at epoch {epoch}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, scaler, target_indices
        )
        val_loss = evaluate(model, val_loader, criterion, DEVICE, target_indices)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(
            f"  [{side_name}] Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in get_base_model().state_dict().items()
            }
            torch.save(best_state_dict, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # --- Evaluation ---
    eval_model = ForcesForFree(num_targets=len(targets), seq_len=SEQ_LEN).to(DEVICE)
    if best_state_dict is not None:
        state = best_state_dict
    else:
        if not best_model_path.exists():
            raise RuntimeError(
                f"[{side_name}] No best checkpoint available for evaluation. "
                "Training likely never produced a valid validation loss improvement."
            )
        try:
            state = torch.load(best_model_path, weights_only=True, map_location="cpu")
        except TypeError:
            state = torch.load(best_model_path, map_location="cpu")
    eval_model.load_state_dict(state)

    metrics, predictions_all, ground_truth_all, times, test_nums = compute_metrics(
        eval_model, test_loader, DEVICE, targets, target_indices
    )
    predictions = predictions_all
    ground_truth = ground_truth_all

    print(f"\n  [{side_name}] {'Target':<8} {'MAE':>10} {'RMSE':>10}")
    print(f"  {'-' * 36}")
    for name in targets:
        m = metrics[name]
        print(f"  {name:<8} {m['mae']:>10.4f} {m['rmse']:>10.4f}")

    return {
        "best_val_loss": best_val_loss,
        "best_model_path": best_model_path,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "times": times,
        "test_nums": test_nums,
    }


def main():
    set_global_reproducibility(SEED, deterministic=DETERMINISTIC)
    print("=" * 60)
    print("FORCES FOR FREE - Vision-Based Force Prediction")
    print(f"CNN (ResNet18) + Transformer -> [{', '.join(ALL_TARGETS)}]")
    print(f"Device: {DEVICE}  |  Job ID: {JOB_ID}")
    print(f"Seed: {SEED} | Deterministic: {DETERMINISTIC}")
    if ENABLE_STRESS_TEST_AUG:
        print("Data stress-test augmentation: ON (occlusion + lighting)")
        print(
            "Occlusion mode: random polygons over segmented gripper "
            f"(up to {int(100 * STRESS_OCCLUSION_SEGMENT_RATIO_RANGE[1])}% area)"
        )
        print(
            f"Preview frames: saving up to {STRESS_SAVE_PREVIEW_FRAMES} in {STRESS_PREVIEW_DIR}"
        )
    else:
        print("Data stress-test augmentation: OFF")
    print(f"Datasets: {TEST_NUMS}")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "outputs" / "forces_for_free"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tag every output file with both the tests and the job ID
    def out(filename: str) -> Path:
        stem, *ext = filename.rsplit(".", 1)
        suffix = f".{ext[0]}" if ext else ""
        run_tag = "stress" if ENABLE_STRESS_TEST_AUG else "clean"
        return output_dir / f"fff_{run_tag}_{stem}{suffix}"

    # --- Step 1/2/3/...: Train & evaluate two independent, side-specific models ---
    results = {}
    image_col_by_side = {"left": "seg_path_L", "right": "seg_path_R"}
    for side_name, targets in SIDE_TARGETS.items():
        print(f"\n--- Loading datasets for side: {side_name.upper()} ({image_col_by_side[side_name]}) ---")
        side_datasets = []
        for test_num in TEST_NUMS:
            print(f"\n  Loading test {test_num} ...")
            ds = load_dataset_for_test(
                test_num=test_num,
                image_col=image_col_by_side[side_name],
                seq_len=SEQ_LEN,
                targets=ALL_TARGETS,
                cache=CACHE_IMAGES,
            )
            side_datasets.append(ds)

        side_result = train_and_evaluate_side(
            side_name=side_name,
            targets=targets,
            all_datasets=side_datasets,
            out=out,
        )
        results[side_name] = side_result
        save_test_predictions_csv(
            output_path=out(f"forces_for_free_{side_name}_test_predictions.csv"),
            target_names=targets,
            predictions=side_result["predictions"],
            ground_truth=side_result["ground_truth"],
            times=side_result["times"],
            test_nums=side_result["test_nums"],
        )

    # Combined right+left Pred vs GT panel (2 rows x 3 cols)
    if "right" in results and "left" in results:
        right_times = results["right"]["times"]
        left_times = results["left"]["times"]
        right_test_nums = results["right"]["test_nums"]
        left_test_nums = results["left"]["test_nums"]

        if np.array_equal(right_times, left_times) and np.array_equal(right_test_nums, left_test_nums):
            plot_predictions_vs_gt_both_sides(
                right_predictions=results["right"]["predictions"],
                right_ground_truth=results["right"]["ground_truth"],
                left_predictions=results["left"]["predictions"],
                left_ground_truth=results["left"]["ground_truth"],
                save_path=out("forces_for_free_right_left_pred_vs_gt.png"),
            )
        else:
            print(
                "  Skipping combined right+left pred-vs-gt plot: test sample order mismatch "
                "between side loaders."
            )

        # Combined right+left training curves (1x2)
        plot_training_curves_both_sides(
            right_train_losses=results["right"]["train_losses"],
            right_val_losses=results["right"]["val_losses"],
            left_train_losses=results["left"]["train_losses"],
            left_val_losses=results["left"]["val_losses"],
            save_path=out("forces_for_free_right_left_training_curves.png"),
        )

        # Combined right+left residual plots (2x3)
        plot_residual_histograms_both_sides(
            right_predictions=results["right"]["predictions"],
            right_ground_truth=results["right"]["ground_truth"],
            left_predictions=results["left"]["predictions"],
            left_ground_truth=results["left"]["ground_truth"],
            save_path=out("forces_for_free_right_left_residual_hist.png"),
        )
        plot_residuals_vs_gt_both_sides(
            right_predictions=results["right"]["predictions"],
            right_ground_truth=results["right"]["ground_truth"],
            left_predictions=results["left"]["predictions"],
            left_ground_truth=results["left"]["ground_truth"],
            save_path=out("forces_for_free_right_left_residual_vs_gt.png"),
        )
        plot_abs_error_cdf_both_sides(
            right_predictions=results["right"]["predictions"],
            right_ground_truth=results["right"]["ground_truth"],
            left_predictions=results["left"]["predictions"],
            left_ground_truth=results["left"]["ground_truth"],
            save_path=out("forces_for_free_right_left_abs_error_cdf.png"),
        )

        # Combined right+left time-series panel (2 rows x 3 cols) for TIME_SERIES_TEST_NUM
        right_ts_mask = (right_test_nums == TIME_SERIES_TEST_NUM)
        left_ts_mask = (left_test_nums == TIME_SERIES_TEST_NUM)
        if np.sum(right_ts_mask) >= 2 and np.sum(left_ts_mask) >= 2:
            plot_time_series_both_sides(
                right_predictions=results["right"]["predictions"][right_ts_mask],
                right_ground_truth=results["right"]["ground_truth"][right_ts_mask],
                right_times=right_times[right_ts_mask],
                left_predictions=results["left"]["predictions"][left_ts_mask],
                left_ground_truth=results["left"]["ground_truth"][left_ts_mask],
                left_times=left_times[left_ts_mask],
                save_path=out("forces_for_free_right_left_timeseries_pred_vs_gt.png"),
            )
        else:
            print(
                "  Skipping combined right+left time-series plot: not enough samples for "
                f"test {TIME_SERIES_TEST_NUM} "
                f"(right={int(np.sum(right_ts_mask))}, left={int(np.sum(left_ts_mask))})."
            )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    for side_name, side_result in results.items():
        best_val_loss = side_result["best_val_loss"]
        best_model_path = side_result["best_model_path"]
        print(f"  [{side_name}] Best val loss : {best_val_loss:.6f}")
        print(f"  [{side_name}] Model saved   : {best_model_path}")
    print(f"  Datasets      : {TEST_NUMS}")
    print(f"  Job ID        : {JOB_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
