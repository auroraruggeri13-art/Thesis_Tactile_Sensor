#!/usr/bin/env python3
"""
Image + barometer fusion for 3-DoF force prediction.

Trains two independent models:
- Right: [fx_R, fy_R, fz_R]
- Left : [fx_L, fy_L, fz_L]
"""

import os
import pickle
import argparse
import random
import warnings
from bisect import bisect_right
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import models, transforms
import matplotlib.pyplot as plt

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:
    InconsistentVersionWarning = None


# =========================== CONFIGURATION ===========================

# TEST_NUMS = [51011003, 51011006, 51011007, 51011008, 51011009]  # flat
TEST_NUMS = [52021001, 52021002, 52021003, 52021004, 52021005, 52021006, 52021007, 52021008, 52021009, 52021010, 52021011]  # round
TIME_SERIES_TEST_NUM = TEST_NUMS[0]

SEQ_LEN = 8
IMAGE_SIZE = 224

BATCH_SIZE = 32
EPOCHS = 80
LR = 2e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 12
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
MODEL_VARIANT = "temporal_multimodal_tokens"

# Temporal multimodal token fusion (used when MODEL_VARIANT="temporal_multimodal_tokens")
TEMPORAL_D_MODEL = 512
TEMPORAL_HEADS = 8
TEMPORAL_LAYERS = 3
TEMPORAL_FF_DIM = 1024
TEMPORAL_DROPOUT = 0.25
TEMPORAL_HEAD_MODE = "concat_pooled_global"  # "global_only" or "concat_pooled_global"

CACHE_IMAGES = True

# Run fusion only (disable image-only/baro-only ablation runs).
MODALITY_MODES = ("fusion",) # "image_only", "baro_only", "fusion"


added_name = "no_baro_pretrain_no_baro_freeze_no_image_freeze_global_token"
BARO_PRETRAIN_ENABLED = True  # whether to do optional baro_net pretraining before main training
BARO_PRETRAIN_EPOCHS = 30
BARO_PRETRAIN_PATIENCE = 6
BARO_PRETRAIN_LR = 1e-3
BARO_PRETRAIN_MAX_TIME_GAP = 0.05
FREEZE_BARO_MLP = True  # False=no baro_net freezing during main training
FREEZE_RESNET_LAYERS = 3 # 0=no freeze, 3=freeze stem+layer1-3
MAX_GRAD_NORM = 1.0
USE_COMPILE = False  # keep False for maximum run-to-run reproducibility
SEED = 123
DETERMINISTIC = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DEVICE_TYPE = "cuda" if DEVICE.type == "cuda" else "cpu"
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

PALETTE = ["#292f56", "#44b155", "#008780", "#44b155", "#d6c52e"]
CLR_NAVY, CLR_BLUE, CLR_TEAL, CLR_GREEN, CLR_YELLOW = PALETTE
CLR_FUSION_DOTS = "#44b155"
CLR_TS_GT = "#d6c52e"
CLR_TS_PRED = "#44b155"


# =============================== HELPERS =============================

def configure_warning_filters():
    if InconsistentVersionWarning is not None:
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


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


class ChannelStandardScaler:
    """Simple 6-channel standard scaler fitted only on train rows."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_in_ = 6

    def fit(self, x: np.ndarray) -> "ChannelStandardScaler":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError(f"Expected train barometer rows with shape (N,6), got {x.shape}")
        self.mean_ = x.mean(axis=0).astype(np.float32)
        std = x.std(axis=0).astype(np.float32)
        std[std == 0.0] = 1.0
        self.scale_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("ChannelStandardScaler must be fitted before transform().")
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean_) / self.scale_).astype(np.float32, copy=False)


def fit_baro_scaler_from_train_rows(datasets: List["FusionSequenceDataset"], side_key: str) -> ChannelStandardScaler:
    train_rows = []
    for ds in datasets:
        sorted_raw = np.sort(ds.meta["time"].to_numpy(dtype=float))
        if sorted_raw.size < 2 or sorted_raw[0] == sorted_raw[-1]:
            continue
        t_train_end = float(np.quantile(sorted_raw, TRAIN_SPLIT))
        rows = ds.meta.loc[ds.meta["time"] <= t_train_end, ds.baro_cols].to_numpy(dtype=np.float32)
        if rows.size > 0:
            train_rows.append(rows)
    if not train_rows:
        raise RuntimeError(f"[{side_key}] Could not collect train rows to fit barometer scaler.")
    all_rows = np.concatenate(train_rows, axis=0)
    scaler = ChannelStandardScaler().fit(all_rows)
    print(f"  [{side_key}] fitted train-only barometer scaler on {all_rows.shape[0]} rows.")
    return scaler

def resolve_dataset_csv(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "dataset.csv",
        dataset_dir.parent / "dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No dataset.csv found in: {candidates}")


def resolve_image_path(raw_path: str, dataset_dir: Path) -> Path:
    raw = str(raw_path).strip()
    if not raw:
        return dataset_dir / "__missing__"

    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p

    norm = raw.replace("\\", "/")
    name = Path(norm).name
    candidates = [
        (dataset_dir / norm),
        (dataset_dir / name),
        (dataset_dir / "segmented_images" / name),
        (dataset_dir.parent / "processed_data" / "segmented_images" / name),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (dataset_dir / "segmented_images" / name).resolve()


def preprocess_image(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return _normalize(tensor)


def load_serialized_object(path: Path, description: str, required: bool = True) -> Optional[Any]:
    if not path.exists():
        msg = f"{description} not found: {path}"
        if required:
            raise FileNotFoundError(msg)
        print(f"WARNING: {msg}")
        return None
    try:
        if joblib is not None:
            return joblib.load(path)
        raise ImportError("joblib not available")
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


class RawBaroScalerAdapter:
    """
    Adapts a scaler fitted on flattened temporal features (e.g. 198 dims) to per-step raw barometers (6 dims).
    Assumes each temporal block is laid out as 18 features = [raw6, d1_6, d2_6].
    """

    def __init__(self, base_scaler: Any):
        self.base_scaler = base_scaler
        n_feat = int(getattr(base_scaler, "n_features_in_"))
        if n_feat < 18 or (n_feat % 18) != 0:
            raise ValueError(
                f"Cannot adapt scaler with n_features_in_={n_feat}; expected a multiple of 18."
            )
        if not hasattr(base_scaler, "mean_") or not hasattr(base_scaler, "scale_"):
            raise ValueError("Window scaler adaptation requires mean_ and scale_.")

        mean = np.asarray(base_scaler.mean_, dtype=np.float64)
        scale = np.asarray(base_scaler.scale_, dtype=np.float64)
        raw_idx = []
        for start in range(0, n_feat, 18):
            raw_idx.extend(range(start, start + 6))
        raw_mean = mean[raw_idx].reshape(-1, 6)
        raw_scale = scale[raw_idx].reshape(-1, 6)

        # Aggregate per-channel stats across temporal positions.
        self.mean_ = raw_mean.mean(axis=0)
        self.scale_ = raw_scale.mean(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        self.n_features_in_ = 6

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean_) / self.scale_).astype(np.float32, copy=False)


def adapt_scaler_for_raw_baro(scaler: Any, side_key: str) -> Any:
    if scaler is None:
        return None
    n_feat = getattr(scaler, "n_features_in_", None)
    if n_feat is None:
        return scaler
    n_feat = int(n_feat)
    if n_feat == 6:
        return scaler
    if n_feat % 18 == 0 and n_feat >= 18:
        print(
            f"  [{side_key}] adapting scaler with {n_feat} features to 6 raw-barometer features "
            "(using aggregated raw-channel stats across temporal blocks)."
        )
        return RawBaroScalerAdapter(scaler)
    return scaler


def validate_scaler(scaler: Any, side_key: str):
    if scaler is None:
        return
    if not hasattr(scaler, "transform"):
        raise TypeError(f"[{side_key}] scaler must expose transform(), got {type(scaler)}")
    n_feat = getattr(scaler, "n_features_in_", None)
    if n_feat is not None and int(n_feat) != 6:
        raise ValueError(
            f"[{side_key}] scaler expects {n_feat} features after adaptation, expected 6 for raw barometers."
        )


# ================================ DATA ===============================

class FusionSequenceDataset(Dataset):
    def __init__(
        self,
        dataset_df: pd.DataFrame,
        dataset_dir: Path,
        seq_len: int,
        side: str,
        test_num: int,
        cache_images: bool = True,
        baro_scaler: Any = None,
        use_baro_features: bool = True,
    ):
        self.seq_len = seq_len
        self.side = side
        self.test_num = test_num
        self.cache_images = cache_images
        self.use_baro_features = use_baro_features

        image_col = "seg_path_R" if side == "R" else "seg_path_L"
        force_cols = [f"fx_{side}_surf", f"fy_{side}_surf", f"fz_{side}_surf"]
        baro_cols = [f"b{i}_{side}" for i in range(1, 7)]
        required_cols = ["time", image_col, *force_cols]
        if use_baro_features:
            required_cols += baro_cols
        missing_cols = [c for c in required_cols if c not in dataset_df.columns]
        if missing_cols:
            raise ValueError(f"[{test_num}] dataset.csv missing columns: {missing_cols}")

        meta = dataset_df[required_cols].copy()
        meta[image_col] = meta[image_col].apply(lambda p: str(resolve_image_path(p, dataset_dir)))
        numeric_cols = ["time", *force_cols]
        if use_baro_features:
            numeric_cols += baro_cols
        for c in numeric_cols:
            meta[c] = pd.to_numeric(meta[c], errors="coerce")
        before = len(meta)
        meta = meta.dropna(subset=numeric_cols)
        dropped_nan = before - len(meta)
        if dropped_nan > 0:
            print(f"  [{test_num}][{side}] dropped {dropped_nan} rows with missing numeric values")
        exists = meta[image_col].apply(lambda p: Path(p).exists())
        if int((~exists).sum()) > 0:
            print(f"  [{test_num}][{side}] dropping {int((~exists).sum())} rows with missing images")
        meta = meta.loc[exists].sort_values("time").reset_index(drop=True)
        if len(meta) < seq_len:
            raise ValueError(
                f"[{test_num}][{side}] Not enough rows ({len(meta)}) for seq_len={seq_len}."
            )

        self.image_col = image_col
        self.force_cols = force_cols
        self.baro_cols = baro_cols
        self.meta = meta
        self.baro_scaler = baro_scaler
        self.valid_starts = list(range(0, len(self.meta) - seq_len + 1))

        self.image_cache = [None] * len(self.meta)
        if self.cache_images:
            print(f"  [{test_num}][{side}] caching {len(self.meta)} images ...")
            for i, p in enumerate(self.meta[self.image_col].tolist()):
                img = cv2.imread(str(p))
                if img is None:
                    raise FileNotFoundError(f"Could not read image: {p}")
                self.image_cache[i] = preprocess_image(img)

        print(
            f"  [{test_num}][{side}] aligned rows: {len(self.meta)} | sequences: {len(self.valid_starts)}"
        )

    def __len__(self):
        return len(self.valid_starts)

    def set_baro_scaler(self, baro_scaler: Any):
        self.baro_scaler = baro_scaler

    def __getitem__(self, idx: int):
        s = self.valid_starts[idx]
        e = s + self.seq_len

        frames = []
        for i in range(s, e):
            if self.cache_images:
                frame_t = self.image_cache[i]
                assert frame_t is not None
            else:
                p = self.meta.iloc[i][self.image_col]
                img = cv2.imread(str(p))
                if img is None:
                    raise FileNotFoundError(f"Could not read image: {p}")
                frame_t = preprocess_image(img)
            frames.append(frame_t)
        image_input = torch.stack(frames, dim=0)  # (S, C, H, W)

        if self.use_baro_features:
            baro_seq = self.meta.iloc[s:e][self.baro_cols].to_numpy(dtype=np.float32)  # (seq_len, 6)
            if self.baro_scaler is not None:
                baro_seq = self.baro_scaler.transform(baro_seq).astype(np.float32, copy=False)
            d1 = np.vstack([np.zeros((1, baro_seq.shape[1]), dtype=np.float32), np.diff(baro_seq, axis=0)])
            d2 = np.vstack([np.zeros((1, baro_seq.shape[1]), dtype=np.float32), np.diff(d1, axis=0)])
            baro_steps = np.concatenate([baro_seq, d1, d2], axis=1)  # (seq_len, 18)
            baro_input = torch.from_numpy(baro_steps)
        else:
            baro_input = torch.zeros((self.seq_len, 18), dtype=torch.float32)

        last_i = e - 1
        row = self.meta.iloc[last_i]
        target = np.array([row[self.force_cols[0]], row[self.force_cols[1]], row[self.force_cols[2]]], dtype=np.float32)
        time = float(row["time"])

        return (
            image_input,
            baro_input,
            torch.from_numpy(target),
            time,
            int(self.test_num),
        )


def load_side_dataset(
    test_num,
    side,
    seq_len,
    cache_images,
    baro_scaler: Any = None,
    use_baro_features: bool = True,
):
    base = Path(__file__).resolve().parent / f"test {test_num} - sensor v5" / "processed_data"
    dataset_csv = resolve_dataset_csv(base)
    dataset_df = pd.read_csv(dataset_csv)
    dataset_df.columns = [c.strip() for c in dataset_df.columns]

    return FusionSequenceDataset(
        dataset_df=dataset_df,
        dataset_dir=base,
        seq_len=seq_len,
        side=side,
        test_num=test_num,
        cache_images=cache_images,
        baro_scaler=baro_scaler,
        use_baro_features=use_baro_features,
    )


def build_non_leaking_splits(datasets, seq_len):
    split_sum = TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT
    if not np.isclose(split_sum, 1.0):
        raise ValueError(f"TRAIN/VAL/TEST split must sum to 1.0, got {split_sum:.4f}")

    train_idx, val_idx, test_idx = [], [], []
    dropped = 0
    offset = 0

    for ds in datasets:
        n_seq = len(ds)
        # last_time[i] = timestamp of the last frame of sequence i
        last_times = ds.meta["time"].iloc[seq_len - 1:].to_numpy(dtype=float)

        sorted_raw = np.sort(ds.meta["time"].to_numpy(dtype=float))
        if sorted_raw[0] == sorted_raw[-1]:
            raise ValueError(f"[{ds.test_num}] All timestamps identical.")

        dt_median = float(np.median(np.diff(sorted_raw)))
        gap_secs  = max(dt_median * (seq_len - 1), 0.0)

        # Use quantile so that exactly TRAIN_SPLIT fraction of CSV *rows*
        # (not time range) defines each boundary — same logic as FFF and LightGBM.
        t_train_end  = float(np.quantile(sorted_raw, TRAIN_SPLIT))
        t_val_start  = t_train_end + gap_secs
        t_val_end    = float(np.quantile(sorted_raw, TRAIN_SPLIT + VAL_SPLIT))
        t_test_start = t_val_end + gap_secs

        train_local = np.where(last_times <= t_train_end)[0].tolist()
        val_local   = np.where((last_times >= t_val_start) & (last_times <= t_val_end))[0].tolist()
        test_local  = np.where(last_times >= t_test_start)[0].tolist()

        if not train_local or not val_local or not test_local:
            # Temporal gap (e.g. missing images) left one split empty — fall
            # back to index-based split for this dataset.
            print(
                f"  [{ds.test_num}] WARNING: time-based split yielded empty split "
                f"(train={len(train_local)}, val={len(val_local)}, test={len(test_local)}), "
                f"falling back to index-based split."
            )
            gap    = seq_len - 1
            usable = n_seq - 2 * gap
            if usable < 3:
                raise ValueError(f"[{ds.test_num}] not enough sequences ({n_seq}) for seq_len={seq_len}.")
            n_train = int(usable * TRAIN_SPLIT)
            n_val   = int(usable * VAL_SPLIT)
            n_test  = usable - n_train - n_val
            train_local = list(range(0, n_train))
            val_start   = n_train + gap
            val_local   = list(range(val_start, val_start + n_val))
            test_start  = val_start + n_val + gap
            test_local  = list(range(test_start, test_start + n_test))
        else:
            print(
                f"  [{ds.test_num}] time-split: "
                f"train≤{t_train_end:.2f}s ({len(train_local)})  "
                f"val=[{t_val_start:.2f},{t_val_end:.2f}]s ({len(val_local)})  "
                f"test≥{t_test_start:.2f}s ({len(test_local)})"
            )

        train_idx.extend([offset + i for i in train_local])
        val_idx.extend([offset + i for i in val_local])
        test_idx.extend([offset + i for i in test_local])

        dropped += n_seq - (len(train_local) + len(val_local) + len(test_local))
        offset += n_seq

    return train_idx, val_idx, test_idx, dropped


# ================================ MODEL ==============================

class TemporalMultimodalTokenFusionModel(nn.Module):
    """
    Temporal multimodal fusion with joint self-attention over:
    [img_1, baro_1, img_2, baro_2, ..., img_S, baro_S, global]
    """

    def __init__(self, baro_step_dim: int, seq_len: int, modality_mode: str = "fusion"):
        super().__init__()
        if modality_mode not in MODALITY_MODES:
            raise ValueError(f"Unknown modality_mode='{modality_mode}', expected one of {MODALITY_MODES}")
        if TEMPORAL_HEAD_MODE not in {"global_only", "concat_pooled_global"}:
            raise ValueError(
                f"Unsupported TEMPORAL_HEAD_MODE='{TEMPORAL_HEAD_MODE}'. "
                "Use 'global_only' or 'concat_pooled_global'."
            )
        self.modality_mode = modality_mode
        self.head_mode = TEMPORAL_HEAD_MODE
        self.seq_len = seq_len
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # -> (B*S, 512, 1, 1)

        self.baro_net = nn.Sequential(
            nn.Linear(baro_step_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, TEMPORAL_D_MODEL),
        )
        self.img_proj = nn.Linear(512, TEMPORAL_D_MODEL)
        self.global_token = nn.Parameter(torch.zeros(1, 1, TEMPORAL_D_MODEL))
        self.modality_embed = nn.Parameter(torch.randn(1, 3, TEMPORAL_D_MODEL) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, TEMPORAL_D_MODEL) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TEMPORAL_D_MODEL,
            nhead=TEMPORAL_HEADS,
            dim_feedforward=TEMPORAL_FF_DIM,
            dropout=TEMPORAL_DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TEMPORAL_LAYERS)
        self.token_norm = nn.LayerNorm(TEMPORAL_D_MODEL)

        if self.head_mode == "global_only":
            head_in_dim = TEMPORAL_D_MODEL
        elif self.modality_mode == "fusion":
            head_in_dim = TEMPORAL_D_MODEL * 3
        else:
            head_in_dim = TEMPORAL_D_MODEL * 2

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(TEMPORAL_DROPOUT),
            nn.Linear(256, 3),
        )

    def forward(self, image_seq, baro_steps):
        bsz, seq_len, c, h, w = image_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        if baro_steps.ndim != 3 or baro_steps.size(1) != seq_len:
            raise ValueError(
                f"Expected baro_steps shape (B, {seq_len}, D), got {tuple(baro_steps.shape)}"
            )

        if self.modality_mode != "baro_only":
            img_tokens = self.img_proj(
                self.cnn(image_seq.reshape(bsz * seq_len, c, h, w)).reshape(bsz, seq_len, 512)
            )
        else:
            img_tokens = None

        if self.modality_mode != "image_only":
            baro_tokens = self.baro_net(baro_steps)
        else:
            baro_tokens = None
        pos = self.pos_embedding[:, :seq_len, :]

        if self.modality_mode == "fusion":
            img_tokens = img_tokens + pos
            baro_tokens = baro_tokens + pos
            pair = torch.stack([img_tokens, baro_tokens], dim=2).reshape(bsz, 2 * seq_len, TEMPORAL_D_MODEL)
            modality_ids = torch.empty((2 * seq_len,), dtype=torch.long, device=pair.device)
            modality_ids[0::2] = 0
            modality_ids[1::2] = 1
            main_tokens = pair + self.modality_embed[:, modality_ids, :]
        elif self.modality_mode == "image_only":
            main_tokens = img_tokens + pos + self.modality_embed[:, 0:1, :]
        else:  # baro_only
            main_tokens = baro_tokens + pos + self.modality_embed[:, 1:2, :]

        global_tok = self.global_token.expand(bsz, -1, -1) + self.modality_embed[:, 2:3, :]
        tokens = torch.cat([main_tokens, global_tok], dim=1)
        out = self.token_norm(self.transformer(tokens))

        if self.head_mode == "global_only":
            return self.head(out[:, -1, :])

        global_out = out[:, -1, :]
        if self.modality_mode == "fusion":
            img_pool = out[:, 0 : 2 * seq_len : 2, :].mean(dim=1)
            baro_pool = out[:, 1 : 2 * seq_len : 2, :].mean(dim=1)
            fused = torch.cat([img_pool, baro_pool, global_out], dim=1)
        else:
            main_pool = out[:, :-1, :].mean(dim=1)
            fused = torch.cat([main_pool, global_out], dim=1)
        return self.head(fused)


class ForcesForFreeImageModel(nn.Module):
    """FFF-style image-only architecture: CNN + temporal transformer + MLP head."""

    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # (B*S, 512, 1, 1)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, TEMPORAL_D_MODEL) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TEMPORAL_D_MODEL,
            nhead=TEMPORAL_HEADS,
            dim_feedforward=TEMPORAL_FF_DIM,
            dropout=TEMPORAL_DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TEMPORAL_LAYERS)
        self.head = nn.Sequential(
            nn.Linear(TEMPORAL_D_MODEL, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(TEMPORAL_DROPOUT),
            nn.Linear(256, 3),
        )

    def forward(self, image_seq, _baro_steps=None):
        bsz, seq_len, c, h, w = image_seq.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        feats = self.cnn(image_seq.reshape(bsz * seq_len, c, h, w)).reshape(bsz, seq_len, TEMPORAL_D_MODEL)
        feats = feats + self.pos_embedding[:, :seq_len, :]
        feats = self.transformer(feats)
        return self.head(feats[:, -1, :])


# =============================== TRAIN ==============================

def evaluate(model, loader, criterion, use_amp: bool = False):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for img, baro, y, _t, _n in loader:
            img = img.to(DEVICE, non_blocking=True)
            baro = baro.to(DEVICE, non_blocking=True).float()
            y = y.to(DEVICE, non_blocking=True).float()
            amp_ctx = torch.amp.autocast(device_type=AMP_DEVICE_TYPE) if use_amp else nullcontext()
            with amp_ctx:
                pred = model(img, baro)
            total += criterion(pred, y).item() * img.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def collect_predictions(model, loader, use_amp: bool = False):
    model.eval()
    all_pred, all_gt, all_t, all_n = [], [], [], []
    for img, baro, y, t, n in loader:
        img = img.to(DEVICE, non_blocking=True)
        baro = baro.to(DEVICE, non_blocking=True).float()
        amp_ctx = torch.amp.autocast(device_type=AMP_DEVICE_TYPE) if use_amp else nullcontext()
        with amp_ctx:
            pred = model(img, baro).cpu().numpy()
        all_pred.append(pred)
        all_gt.append(y.numpy())
        all_t.append(np.asarray(t, dtype=float))
        all_n.append(np.asarray(n, dtype=int))
    return (
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_gt, axis=0),
        np.concatenate(all_t, axis=0),
        np.concatenate(all_n, axis=0),
    )


def _concat_global_to_local(concat_ds: ConcatDataset, global_idx: int) -> Tuple[int, int]:
    ds_idx = bisect_right(concat_ds.cumulative_sizes, global_idx)
    prev_cum = 0 if ds_idx == 0 else concat_ds.cumulative_sizes[ds_idx - 1]
    local_idx = global_idx - prev_cum
    return ds_idx, local_idx


def build_baro_pretrain_tensors(
    combined: ConcatDataset,
    train_indices: List[int],
    max_time_gap: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    baro_steps_all = []
    targets_all = []
    dropped_gap = 0

    for gidx in train_indices:
        ds_idx, local_idx = _concat_global_to_local(combined, int(gidx))
        ds = combined.datasets[ds_idx]
        if not isinstance(ds, FusionSequenceDataset) or not ds.use_baro_features:
            continue

        s = ds.valid_starts[local_idx]
        e = s + ds.seq_len
        times = ds.meta.iloc[s:e]["time"].to_numpy(dtype=float)
        if times.size >= 2 and float(np.max(np.diff(times))) > max_time_gap:
            dropped_gap += 1
            continue

        baro_seq = ds.meta.iloc[s:e][ds.baro_cols].to_numpy(dtype=np.float32)
        if ds.baro_scaler is not None:
            baro_seq = ds.baro_scaler.transform(baro_seq).astype(np.float32, copy=False)
        d1 = np.vstack([np.zeros((1, baro_seq.shape[1]), dtype=np.float32), np.diff(baro_seq, axis=0)])
        d2 = np.vstack([np.zeros((1, baro_seq.shape[1]), dtype=np.float32), np.diff(d1, axis=0)])
        baro_steps = np.concatenate([baro_seq, d1, d2], axis=1)

        row = ds.meta.iloc[e - 1]
        target = np.array([row[ds.force_cols[0]], row[ds.force_cols[1]], row[ds.force_cols[2]]], dtype=np.float32)
        baro_steps_all.append(baro_steps)
        targets_all.append(target)

    if not baro_steps_all:
        return None, None

    x = torch.from_numpy(np.stack(baro_steps_all, axis=0))
    y = torch.from_numpy(np.stack(targets_all, axis=0))
    if dropped_gap > 0:
        print(f"  [baro-pretrain] dropped {dropped_gap} train sequences due to max gap>{max_time_gap:.3f}s")
    return x, y


def pretrain_baro_net(
    model: TemporalMultimodalTokenFusionModel,
    pretrain_x: torch.Tensor,
    pretrain_y: torch.Tensor,
    side: str,
    mode: str,
):
    if pretrain_x is None or pretrain_y is None or int(pretrain_x.size(0)) < 8:
        print(f"  [{side}][{mode}] skipping baro pretraining (not enough samples).")
        return

    n = int(pretrain_x.size(0))
    perm = torch.randperm(n)
    pretrain_x = pretrain_x[perm]
    pretrain_y = pretrain_y[perm]
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    if n_train < 1:
        print(f"  [{side}][{mode}] skipping baro pretraining (train split empty).")
        return

    x_train, y_train = pretrain_x[:n_train], pretrain_y[:n_train]
    x_val, y_val = pretrain_x[n_train:], pretrain_y[n_train:]

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    temp_head = nn.Linear(TEMPORAL_D_MODEL, 3).to(DEVICE)
    pretrain_params = list(model.baro_net.parameters()) + list(temp_head.parameters())
    optimizer = torch.optim.AdamW(pretrain_params, lr=BARO_PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.baro_net.state_dict().items()}
    patience = 0
    print(f"  [{side}][{mode}] baro pretrain: samples train={n_train} val={n_val}")
    for ep in range(1, BARO_PRETRAIN_EPOCHS + 1):
        model.baro_net.train()
        temp_head.train()
        running = 0.0
        for x_baro, y_force in train_loader:
            x_baro = x_baro.to(DEVICE, non_blocking=True).float()
            y_force = y_force.to(DEVICE, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast(device_type=AMP_DEVICE_TYPE) if use_amp else nullcontext()
            with amp_ctx:
                features = model.baro_net(x_baro).mean(dim=1)
                pred = temp_head(features)
                loss = criterion(pred, y_force)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running += loss.item() * x_baro.size(0)

        model.baro_net.eval()
        temp_head.eval()
        val_total = 0.0
        with torch.no_grad():
            for x_baro, y_force in val_loader:
                x_baro = x_baro.to(DEVICE, non_blocking=True).float()
                y_force = y_force.to(DEVICE, non_blocking=True).float()
                amp_ctx = torch.amp.autocast(device_type=AMP_DEVICE_TYPE) if use_amp else nullcontext()
                with amp_ctx:
                    features = model.baro_net(x_baro).mean(dim=1)
                    pred = temp_head(features)
                    vloss = criterion(pred, y_force)
                val_total += vloss.item() * x_baro.size(0)

        train_loss = running / max(1, n_train)
        val_loss = val_total / max(1, n_val)
        print(
            f"  [{side}][{mode}] baro-pretrain epoch {ep:03d}/{BARO_PRETRAIN_EPOCHS} "
            f"train={train_loss:.6f} val={val_loss:.6f}"
        )

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.baro_net.state_dict().items()}
        else:
            patience += 1
            if patience >= BARO_PRETRAIN_PATIENCE:
                print(f"  [{side}][{mode}] baro-pretrain early stop at epoch {ep}")
                break

    model.baro_net.load_state_dict(best_state)


def freeze_resnet_blocks(model: nn.Module, freeze_layers: int, side: str, mode: str):
    if freeze_layers <= 0 or not hasattr(model, "cnn"):
        return
    blocks = max(0, min(3, int(freeze_layers)))
    cnn_modules = list(model.cnn.children())
    if len(cnn_modules) < 8:
        print(f"  [{side}][{mode}] WARNING: unexpected CNN layout; skipping ResNet freezing")
        return

    modules_to_freeze = [cnn_modules[0], cnn_modules[1]]  # conv1, bn1
    for i in range(blocks):
        modules_to_freeze.append(cnn_modules[4 + i])  # layer1..layer3

    frozen_params = 0
    for module in modules_to_freeze:
        for p in module.parameters():
            if p.requires_grad:
                p.requires_grad = False
                frozen_params += p.numel()
    print(f"  [{side}][{mode}] froze ResNet stem + first {blocks} layer blocks ({frozen_params} params)")


def train_side(
    side,
    datasets,
    out_path_fn,
    baro_scaler: Any = None,
    modes: Tuple[str, ...] = MODALITY_MODES,
    freeze_resnet_layers: int = FREEZE_RESNET_LAYERS,
):
    print(f"\n--- Training multimodal model(s) for side {side.upper()} ---")
    combined = ConcatDataset(datasets)
    train_idx, val_idx, test_idx, dropped = build_non_leaking_splits(datasets, seq_len=SEQ_LEN)
    print(
        f"  sequences total={len(combined)} | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"| dropped_for_gap={dropped}"
    )
    train_ds = Subset(combined, train_idx)
    val_ds = Subset(combined, val_idx)
    test_ds = Subset(combined, test_idx)

    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    sample_item = combined[0]
    if sample_item[1].ndim != 2:
        raise ValueError(
            "Expected baro sequence features with shape (seq_len, feat_dim). "
            "Check dataset return_temporal_inputs flag."
        )
    baro_step_dim = int(sample_item[1].size(1))
    side_results = {}

    for mode in modes:
        print(f"\n  [{side}] mode={mode}")
        if mode == "image_only":
            model = ForcesForFreeImageModel(seq_len=SEQ_LEN).to(DEVICE)
        else:
            model = TemporalMultimodalTokenFusionModel(
                baro_step_dim=baro_step_dim,
                seq_len=SEQ_LEN,
                modality_mode=mode,
            ).to(DEVICE)

        def get_base_model():
            return model._orig_mod if hasattr(model, "_orig_mod") else model

        freeze_resnet_blocks(get_base_model(), freeze_resnet_layers, side, mode)

        if BARO_PRETRAIN_ENABLED and hasattr(get_base_model(), "baro_net") and mode != "image_only":
            pretrain_x, pretrain_y = build_baro_pretrain_tensors(
                combined,
                train_idx,
                max_time_gap=BARO_PRETRAIN_MAX_TIME_GAP,
            )
            pretrain_baro_net(get_base_model(), pretrain_x, pretrain_y, side, mode)

        if FREEZE_BARO_MLP and hasattr(get_base_model(), "baro_net"):
            frozen_params = 0
            for p in get_base_model().baro_net.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen_params += p.numel()
            print(f"  [{side}][{mode}] froze baro MLP (baro_net) ({frozen_params} params)")

        if mode == "image_only" and USE_COMPILE and hasattr(torch, "compile"):
            print(f"  [{side}][{mode}] applying torch.compile() ...")
            model = torch.compile(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        all_targets = []
        for _, _, y, _, _ in train_loader:
            all_targets.append(y)
        all_targets = torch.cat(all_targets, dim=0)
        target_var = all_targets.var(dim=0).clamp(min=1e-6).to(DEVICE)
        # inv_var_weights = (1.0 / target_var)
        # inv_var_weights = inv_var_weights / inv_var_weights.sum() * 3.0
        # print(f"  [{side}][{mode}] loss weights: {inv_var_weights.cpu().tolist()}")

        # def weighted_mse(pred, gt):
        #     return (inv_var_weights * (pred - gt) ** 2).mean()

        # criterion = weighted_mse
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
        use_amp = DEVICE.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)

        best_val = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in get_base_model().state_dict().items()}
        patience = 0
        train_losses = []
        val_losses = []
        for ep in range(1, EPOCHS + 1):
            model.train()
            running = 0.0
            for img, baro, y, _t, _n in train_loader:
                img = img.to(DEVICE, non_blocking=True)
                baro = baro.to(DEVICE, non_blocking=True).float()
                y = y.to(DEVICE, non_blocking=True).float()
                optimizer.zero_grad(set_to_none=True)
                amp_ctx = torch.amp.autocast(device_type=AMP_DEVICE_TYPE) if use_amp else nullcontext()
                with amp_ctx:
                    pred = model(img, baro)
                    loss = criterion(pred, y)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                running += loss.item() * img.size(0)
            train_loss = running / len(train_loader.dataset)
            val_loss = evaluate(model, val_loader, criterion, use_amp=use_amp)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if np.isfinite(val_loss):
                scheduler.step(val_loss)
            else:
                print(f"  [{side}][{mode}] WARNING: non-finite val loss ({val_loss}); skipping scheduler step")
            print(
                f"  [{side}][{mode}] epoch {ep:03d}/{EPOCHS} train={train_loss:.6f} val={val_loss:.6f} "
                f"lr(main)={optimizer.param_groups[-1]['lr']:.2e}"
            )

            if np.isfinite(val_loss) and (val_loss < best_val):
                best_val = val_loss
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in get_base_model().state_dict().items()}
            else:
                patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    print(f"  [{side}][{mode}] early stop at epoch {ep}")
                    break

        get_base_model().load_state_dict(best_state)
        ckpt = out_path_fn(f"fitf_{side}_{mode}_best.pt")
        torch.save(best_state, ckpt)

        pred, gt, times, test_nums = collect_predictions(model, test_loader, use_amp=use_amp)
        mae = np.mean(np.abs(pred - gt), axis=0)
        rmse = np.sqrt(np.mean((pred - gt) ** 2, axis=0))
        names = ["fx", "fy", "fz"]
        print(f"  [{side}][{mode}] {'target':<6} {'MAE':>10} {'RMSE':>10}")
        for i, n in enumerate(names):
            print(f"  [{side}][{mode}] {n:<6} {mae[i]:>10.4f} {rmse[i]:>10.4f}")

        side_results[mode] = {
            "model_path": ckpt,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "pred": pred,
            "gt": gt,
            "times": times,
            "test_nums": test_nums,
            "mae": mae,
            "rmse": rmse,
        }

    return side_results


# =============================== PLOTS ==============================

def plot_pred_vs_gt_both(left_pred, left_gt, right_pred, right_gt, save_path, mode_label: str = "fusion"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ordered = [
        ("L", "fx", left_pred, left_gt),
        ("L", "fy", left_pred, left_gt),
        ("L", "fz", left_pred, left_gt),
        ("R", "fx", right_pred, right_gt),
        ("R", "fy", right_pred, right_gt),
        ("R", "fz", right_pred, right_gt),
    ]
    for ax, (side, name, pred, gt) in zip(axes.flatten(), ordered):
        i = ["fx", "fy", "fz"].index(name)
        pred_i = pred[:, i]
        gt_i = gt[:, i]
        mae = float(np.mean(np.abs(pred_i - gt_i)))
        ss_res = float(np.sum((gt_i - pred_i) ** 2))
        ss_tot = float(np.sum((gt_i - np.mean(gt_i)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.scatter(gt_i, pred_i, alpha=0.55, s=22, color=CLR_FUSION_DOTS)
        lims = [min(gt_i.min(), pred_i.min()), max(gt_i.max(), pred_i.max())]
        ax.plot(lims, lims, "--", color=CLR_YELLOW, linewidth=1.4)
        ax.set_title(f"{name}_{side}\nMAE: {mae:.2f} | R²: {r2:.3f}", fontsize=12)
        ax.set_xlabel("Ground Truth [N]")
        ax.set_ylabel("Predicted [N]")
        ax.grid(alpha=0.28)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(f"Predicted vs Actual ({mode_label})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_time_series_both(left_pred, left_gt, left_t, right_pred, right_gt, right_t, save_path, mode_label: str = "fusion"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=False)
    ordered = [
        ("L", left_pred, left_gt, left_t),
        ("R", right_pred, right_gt, right_t),
    ]
    for row, (side, pred, gt, t) in enumerate(ordered):
        order = np.argsort(t)
        t = t[order]
        pred = pred[order]
        gt = gt[order]
        if t.size > 0:
            t0 = float(t[0])
            keep = (t >= t0) & (t <= t0 + 20.0)
            t = t[keep]
            pred = pred[keep]
            gt = gt[keep]
            t = t - t0  # relative time so x-axis starts at 0
        for col, name in enumerate(["fx", "fy", "fz"]):
            ax = axes[row, col]
            ax.plot(
                t,
                gt[:, col],
                color=CLR_TS_GT,
                linewidth=2.4,
                label="Ground Truth",
            )
            ax.scatter(t, gt[:, col], color=CLR_TS_GT, s=12, alpha=0.7, zorder=5)
            ax.plot(
                t,
                pred[:, col],
                color=CLR_TS_PRED,
                linewidth=2.4,
                alpha=0.95,
                label="Predicted",
            )
            ax.scatter(t, pred[:, col], color=CLR_TS_PRED, s=12, alpha=0.7, zorder=5)
            ax.set_title(f"{name}_{side}", fontsize=12)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"{name} [N]")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Time Series: Predicted vs Ground Truth ({mode_label})", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves_both(left_train, left_val, right_train, right_val, save_path, mode_label: str = "fusion"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    panels = [
        ("L", left_train, left_val),
        ("R", right_train, right_val),
    ]
    for ax, (side, train_losses, val_losses) in zip(axes, panels):
        ax.plot(train_losses, color=CLR_BLUE, linewidth=1.8, label="Train Loss")
        ax.plot(val_losses, color=CLR_TEAL, linewidth=1.8, label="Val Loss")
        ax.set_title(f"Loss Curves ({side})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss [N²]")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Train/Val Loss Curves ({mode_label})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_test_predictions_csv(
    output_path: Path,
    side_letter: str,
    pred: np.ndarray,
    gt: np.ndarray,
    times: np.ndarray,
    test_nums: np.ndarray,
):
    pred_df = pd.DataFrame(
        {
            "test_num": np.asarray(test_nums, dtype=int),
            "time": np.asarray(times, dtype=float),
            f"fx_{side_letter}_true": gt[:, 0],
            f"fy_{side_letter}_true": gt[:, 1],
            f"fz_{side_letter}_true": gt[:, 2],
            f"fx_{side_letter}_pred": pred[:, 0],
            f"fy_{side_letter}_pred": pred[:, 1],
            f"fz_{side_letter}_pred": pred[:, 2],
        }
    ).sort_values(["test_num", "time"])
    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions CSV: {output_path}")


# ================================ MAIN ===============================

def main():
    configure_warning_filters()
    set_global_reproducibility(SEED, deterministic=DETERMINISTIC)
    parser = argparse.ArgumentParser(description="Train image + barometer fusion model for force prediction.")
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(MODALITY_MODES),
        help=f"Comma-separated ablation modes from {MODALITY_MODES}.",
    )
    parser.add_argument(
        "--freeze-resnet",
        type=int,
        default=FREEZE_RESNET_LAYERS,
        help="Freeze ResNet stem and first N layer blocks (0-3).",
    )
    args = parser.parse_args()
    requested_modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    invalid = [m for m in requested_modes if m not in MODALITY_MODES]
    if invalid:
        raise ValueError(f"Invalid mode(s): {invalid}. Allowed modes: {MODALITY_MODES}")
    if not requested_modes:
        raise ValueError("At least one mode must be selected.")

    print("=" * 70)
    print("Image + Barometer Multimodal for Force Prediction")
    print(f"Tests: {TEST_NUMS} | Device: {DEVICE} | Job ID: {JOB_ID}")
    print(f"Seed: {SEED} | Deterministic: {DETERMINISTIC}")
    print(f"Model variant: {MODEL_VARIANT} | Temporal head mode: {TEMPORAL_HEAD_MODE}")
    print(
        f"LR/WD: {LR:.1e}/{WEIGHT_DECAY:.1e} | "
        f"Grad clip: {MAX_GRAD_NORM:.2f}"
    )
    print(
        f"Baro pretrain: {BARO_PRETRAIN_ENABLED} | "
        f"Freeze baro MLP: {FREEZE_BARO_MLP} | "
        f"Freeze ResNet layers: {args.freeze_resnet}"
    )
    print(f"Ablation modes: {requested_modes}")
    print("=" * 70)

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "forces_image_baro_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    def out(name):
        stem, *ext = name.rsplit(".", 1)
        suffix = f".{ext[0]}" if ext else ""
        return out_dir / f"fitf_{stem}{suffix}"

    results = {"left": {}, "right": {}}
    for side_key, side_letter in (("left", "L"), ("right", "R")):
        for mode in requested_modes:
            print(f"\nPreparing datasets for side {side_key.upper()} mode={mode} ...")
            use_baro_features = mode != "image_only"
            datasets = []
            for test_num in TEST_NUMS:
                ds = load_side_dataset(
                    test_num=test_num,
                    side=side_letter,
                    seq_len=SEQ_LEN,
                    cache_images=CACHE_IMAGES,
                    baro_scaler=None,
                    use_baro_features=use_baro_features,
                )
                datasets.append(ds)

            side_scaler = None
            if use_baro_features:
                side_scaler = fit_baro_scaler_from_train_rows(datasets, side_key=f"{side_key}_{mode}")
                validate_scaler(side_scaler, f"{side_key}_{mode}")
                for ds in datasets:
                    ds.set_baro_scaler(side_scaler)

            side_mode_result = train_side(
                side_key,
                datasets,
                out,
                baro_scaler=side_scaler,
                modes=(mode,),
                freeze_resnet_layers=args.freeze_resnet,
            )
            results[side_key][mode] = side_mode_result[mode]

    for mode in requested_modes:
        left = results["left"][mode]
        right = results["right"][mode]
        save_test_predictions_csv(
            output_path=out(f"{mode}_test_predictions_L.csv"),
            side_letter="L",
            pred=left["pred"],
            gt=left["gt"],
            times=left["times"],
            test_nums=left["test_nums"],
        )
        save_test_predictions_csv(
            output_path=out(f"{mode}_test_predictions_R.csv"),
            side_letter="R",
            pred=right["pred"],
            gt=right["gt"],
            times=right["times"],
            test_nums=right["test_nums"],
        )
        plot_loss_curves_both(
            left_train=left["train_losses"],
            left_val=left["val_losses"],
            right_train=right["train_losses"],
            right_val=right["val_losses"],
            save_path=out(f"{mode}_loss_curves_left_right______{added_name}.png"),
            mode_label=mode,
        )
        plot_pred_vs_gt_both(
            left_pred=left["pred"],
            left_gt=left["gt"],
            right_pred=right["pred"],
            right_gt=right["gt"],
            save_path=out(f"{mode}_pred_vs_gt_left_right______{added_name}.png"),
            mode_label=mode,
        )

        left_mask = left["test_nums"] == TIME_SERIES_TEST_NUM
        right_mask = right["test_nums"] == TIME_SERIES_TEST_NUM
        if int(np.sum(left_mask)) >= 2 and int(np.sum(right_mask)) >= 2:
            plot_time_series_both(
                left_pred=left["pred"][left_mask],
                left_gt=left["gt"][left_mask],
                left_t=left["times"][left_mask],
                right_pred=right["pred"][right_mask],
                right_gt=right["gt"][right_mask],
                right_t=right["times"][right_mask],
                save_path=out(f"{mode}_timeseries_pred_vs_gt_left_right______{added_name}.png"),
                mode_label=mode,
            )
        else:
            print(
                f"Skipping {mode} combined time-series plot due to low samples in TIME_SERIES_TEST_NUM "
                f"{TIME_SERIES_TEST_NUM}: left={int(np.sum(left_mask))} right={int(np.sum(right_mask))}"
            )

    rows = []
    target_names = ["fx", "fy", "fz"]
    for mode in requested_modes:
        for side_key, side_letter in (("left", "L"), ("right", "R")):
            side_res = results[side_key][mode]
            for i, name in enumerate(target_names):
                rows.append(
                    {
                        "mode": mode,
                        "side": side_letter,
                        "target": name,
                        "mae": float(side_res["mae"][i]),
                        "rmse": float(side_res["rmse"][i]),
                    }
                )
    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "fitf_ablation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("\nAblation metrics:")
    print(metrics_df.to_string(index=False))
    print(f"Saved metrics CSV: {metrics_path}")

    print("\nDone.")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
