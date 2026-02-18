"""
Direct JPEG Extractor - extracts JPEGs from ROS bag with timestamps,
processes left/right barometers, and synchronizes everything via merge_asof.

NOW INCLUDES interactive HSV colour picker:
  • Tab 1: click up to 4 pixels to INCLUDE (define target colour range)
  • Tab 2: click up to 4 pixels to EXCLUDE (remove false positives)
  After both windows are closed the picked bounds are patched into the
  processor and process_all() uses them automatically.

Requirements:
    pip install rosbags opencv-python tqdm matplotlib

Optional (SAM segmentation):
    pip install segment-anything torch torchvision
    Download a SAM checkpoint, e.g. vit_b (~375 MB, fastest):
      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    or vit_l (~1.2 GB) / vit_h (~2.5 GB, most accurate).
"""

import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from rosbags.highlevel import AnyReader

# ---- optional SAM import (guarded so the script works without it) ----
try:
    from segment_anything import sam_model_registry, SamPredictor
    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False

# barometers_processing is in a sibling folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Barometers_Based_Tactile_Sensor"))
from barometers_processing import (
    load_barometer_data,
    process_barometers,
    rezero_barometers_when_fz_zero,
)

# interactive HSV colour picker (same directory as this script)
from hsv_color_picker import pick_hsv_bounds, patch_processor


# =========================== CONFIGURATION ===========================

test_num = 51011004

# --- Barometer processing params ---
WARMUP_DURATION            = 1.0
ENABLE_STEP_LEVELING       = True
STEP_THRESHOLD_HPA         = 5.0
STEP_WINDOW_SIZE           = 200

ENABLE_OUTLIER_REMOVAL     = True
OUTLIER_THRESHOLD_MULTIPLIER = 30.0

DRIFT_REMOVAL_METHOD       = "ema"
EMA_ALPHA                  = 0.0001
EMA_ALPHA_OVERRIDE         = {}
ZERO_AT_START              = True

# --- Synchronization params ---
ASOF_DIRECTION             = "nearest"
ASOF_TOLERANCE_S           = 0.05

# --- Dynamic re-zero based on Fz ~ 0 ---
ENABLE_DYNAMIC_REZERO      = True
FZ_ZERO_THRESHOLD          = 0.2
MIN_ZERO_DURATION_S        = 0.01
MIN_ZERO_SAMPLES           = 5

# --- Segmentation ---
USE_SAM         = False
SAM_CHECKPOINT  = r"C:\Users\aurir\sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE  = "vit_h"

# --- HSV picker ---
# How many pixels to show for the picker preview (resize if larger)
PICKER_PREVIEW_MAX_DIM     = 900
# Frame index to use as picker sample (set None → middle frame)
PICKER_FRAME_IDX: int | None = None


# ======================== IMAGE EXTRACTION ============================

class RosBagImageExtractor:
    """Extract raw images from a ROS bag with timestamps using rosbags."""

    _ENC_INFO = {
        'rgb8':  (np.uint8,  3),
        'bgr8':  (np.uint8,  3),
        'rgba8': (np.uint8,  4),
        'bgra8': (np.uint8,  4),
        'mono8': (np.uint8,  1),
        '8UC1':  (np.uint8,  1),
        '8UC3':  (np.uint8,  3),
        '16UC1': (np.uint16, 1),
    }

    def __init__(self, bag_path):
        self.bag_path = Path(bag_path)

    @staticmethod
    def _raw_to_bgr(data, height, width, encoding):
        info = RosBagImageExtractor._ENC_INFO.get(encoding)
        if info is None:
            raise ValueError(f"Unsupported encoding: {encoding}")
        dtype, channels = info
        img = np.frombuffer(data, dtype=dtype).reshape(height, width, channels)
        if encoding   == 'rgb8':  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif encoding == 'rgba8': img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif encoding == 'bgra8': img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif encoding in ('mono8', '8UC1'):
                                   img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def extract_all_images(self):
        print(f"Reading bag file: {self.bag_path}")
        images = []
        with AnyReader([self.bag_path]) as reader:
            print("  Available topics:")
            for conn in reader.connections:
                print(f"    {conn.topic}  [{conn.msgtype}]")

            image_connections = [
                c for c in reader.connections
                if c.msgtype in ('sensor_msgs/msg/Image', 'sensor_msgs/Image')
            ]
            if not image_connections:
                raise ValueError("No sensor_msgs/Image topic found in bag file.")

            topic_name = image_connections[0].topic
            print(f"  Using topic: {topic_name}")

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
                image = self._raw_to_bgr(
                    bytes(msg.data), msg.height, msg.width, msg.encoding
                )
                images.append({
                    'index':          idx,
                    'image':          image,
                    'timestamp_sec':  timestamp / 1e9,
                    'width':          image.shape[1],
                    'height':         image.shape[0],
                })
                idx += 1

        print(f"Successfully extracted {len(images)} images with timestamps")
        return images


# ======================== FIN RAY PROCESSOR ===========================

class FinRayProcessor:
    """Process extracted images with force and barometer data."""

    def __init__(self, dataset_dir, test_num,
                 barometer_file_left=None, barometer_file_right=None):
        self.dataset_dir = Path(dataset_dir)
        self.test_num    = test_num

        self.ati_file_right        = self.dataset_dir / f"{test_num}_ati_right.txt"
        self.ati_file_left         = self.dataset_dir / f"{test_num}_ati_left.txt"
        self.barometer_file_left   = barometer_file_left
        self.barometer_file_right  = barometer_file_right

        self.output_dir = self.dataset_dir / "processed_data"
        self.viz_dir    = self.output_dir  / "visualizations"
        self.plots_dir  = self.output_dir  / "barometers_plots"

        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Default HSV thresholds — overridden by pick_hsv_bounds()
        self.green_lower   = np.array([35,  60,  60])
        self.green_upper   = np.array([85, 255, 255])
        self.exclude_lower = None
        self.exclude_upper = None

        # SAM predictor
        self.sam_predictor = None
        if USE_SAM:
            if not _SAM_AVAILABLE:
                print("  [SAM] segment_anything not installed — falling back to HSV.")
            elif not Path(SAM_CHECKPOINT).exists():
                print(f"  [SAM] Checkpoint not found: {SAM_CHECKPOINT}")
            else:
                import torch
                print(f"  [SAM] Loading {SAM_MODEL_TYPE} …")
                _sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
                _sam.to('cuda' if torch.cuda.is_available() else 'cpu')
                self.sam_predictor = SamPredictor(_sam)
                print("  [SAM] Ready.")

        self.images          = []
        self.ati_data_right  = None
        self.ati_data_left   = None
        self.baro_left       = None
        self.baro_right      = None
        self.synchronized_df = None

    # ------------------------------------------------------------------ ATI

    @staticmethod
    def _load_one_ati(filepath, suffix):
        df = pd.read_csv(filepath)
        df.columns = ['time','seq','stamp','frame_id',
                      'force_x','force_y','force_z',
                      'torque_x','torque_y','torque_z']
        df['time'] = df['stamp'] / 1e9
        rename = {c: f'{c}_{suffix}' for c in [
            'force_x','force_y','force_z','torque_x','torque_y','torque_z']}
        return df.rename(columns=rename)

    def load_ati_data(self):
        print("\nLoading ATI force data...")
        for side, fp in [('R', self.ati_file_right), ('L', self.ati_file_left)]:
            ati = self._load_one_ati(fp, side)
            print(f"  [{side}] {len(ati)} measurements  "
                  f"({ati['time'].min():.2f} – {ati['time'].max():.2f} s)")
            if side == 'R': self.ati_data_right = ati
            else:           self.ati_data_left  = ati

    # ------------------------------------------------------------------ images

    def extract_images(self, bag_file):
        print("\nExtracting images from bag file...")
        self.images = RosBagImageExtractor(bag_file).extract_all_images()
        return self.images

    # ------------------------------------------------------------------ interactive HSV picker

    def run_hsv_picker(self):
        """
        Show the interactive HSV picker on a representative frame, then
        patch this processor to use the resulting bounds.
        """
        if not self.images:
            raise RuntimeError("Call extract_images() before run_hsv_picker().")

        # Pick the sample frame
        if PICKER_FRAME_IDX is not None:
            frame_idx = min(PICKER_FRAME_IDX, len(self.images) - 1)
        else:
            frame_idx = len(self.images) // 2
        sample = self.images[frame_idx]['image'].copy()

        # Resize for display if very large
        h, w = sample.shape[:2]
        max_dim = max(h, w)
        if max_dim > PICKER_PREVIEW_MAX_DIM:
            scale  = PICKER_PREVIEW_MAX_DIM / max_dim
            sample = cv2.resize(sample, (int(w * scale), int(h * scale)))
            print(f"  [picker] Resized preview to {sample.shape[1]}×{sample.shape[0]} "
                  f"(original {w}×{h})")

        print(f"\n  [picker] Using frame {frame_idx} / {len(self.images) - 1} as sample.")

        green_lower, green_upper, exclude_lower, exclude_upper = pick_hsv_bounds(sample)
        patch_processor(self, green_lower, green_upper, exclude_lower, exclude_upper)

    # ------------------------------------------------------------------ barometers

    def _process_one_barometer(self, baro_path, side_label, plots_subdir):
        print(f"\n  Loading {side_label} barometers: {baro_path.name}")
        baro_df = load_barometer_data(baro_path)
        save_dir = self.plots_dir / plots_subdir
        save_dir.mkdir(parents=True, exist_ok=True)

        baro_df = process_barometers(
            baro_df,
            save_plots_dir=save_dir,
            warmup_duration=WARMUP_DURATION,
            enable_step_leveling=ENABLE_STEP_LEVELING,
            step_threshold_hpa=STEP_THRESHOLD_HPA,
            step_window_size=STEP_WINDOW_SIZE,
            enable_outlier_removal=ENABLE_OUTLIER_REMOVAL,
            outlier_threshold_multiplier=OUTLIER_THRESHOLD_MULTIPLIER,
            drift_removal_method=DRIFT_REMOVAL_METHOD,
            ema_alpha=EMA_ALPHA,
            ema_alpha_override=EMA_ALPHA_OVERRIDE,
            zero_at_start=ZERO_AT_START,
        )
        rename_map = {}
        for i in range(1, 7):
            if f'b{i}' in baro_df.columns: rename_map[f'b{i}'] = f'b{i}_{side_label}'
            if f't{i}' in baro_df.columns: rename_map[f't{i}'] = f't{i}_{side_label}'
        baro_df = baro_df.rename(columns=rename_map)
        print(f"  Processed {len(baro_df)} {side_label} barometer samples")
        return baro_df

    def load_and_process_barometers(self):
        print("\nProcessing barometers...")
        if self.barometer_file_left and self.barometer_file_left.exists():
            self.baro_left = self._process_one_barometer(
                self.barometer_file_left, 'L', 'barometers_left')
        else:
            print("  Warning: Left barometer file not found, skipping.")
        if self.barometer_file_right and self.barometer_file_right.exists():
            self.baro_right = self._process_one_barometer(
                self.barometer_file_right, 'R', 'barometers_right')
        else:
            print("  Warning: Right barometer file not found, skipping.")

    # ------------------------------------------------------------------ sync

    def synchronize_all(self):
        print("\nSynchronizing all data streams...")

        img_df = pd.DataFrame([
            {'image_idx': img['index'], 'time': img['timestamp_sec']}
            for img in self.images
        ]).sort_values('time').reset_index(drop=True)
        print(f"  Images: {len(img_df)} frames  "
              f"({img_df['time'].min():.2f} – {img_df['time'].max():.2f} s)")

        merged = img_df

        if self.ati_data_right is not None:
            cols = ['time'] + [c for c in self.ati_data_right.columns
                               if c.startswith(('force_', 'torque_'))]
            merged = pd.merge_asof(
                merged.sort_values('time'),
                self.ati_data_right[cols].sort_values('time'),
                on='time', direction=ASOF_DIRECTION, tolerance=ASOF_TOLERANCE_S)
            print(f"  After right ATI merge: {merged['force_z_R'].notna().sum()}/{len(merged)} matched")

        if self.ati_data_left is not None:
            cols = ['time'] + [c for c in self.ati_data_left.columns
                               if c.startswith(('force_', 'torque_'))]
            merged = pd.merge_asof(
                merged.sort_values('time'),
                self.ati_data_left[cols].sort_values('time'),
                on='time', direction=ASOF_DIRECTION, tolerance=ASOF_TOLERANCE_S)
            print(f"  After left ATI merge:  {merged['force_z_L'].notna().sum()}/{len(merged)} matched")

        if self.baro_left is not None:
            merged = pd.merge_asof(
                merged.sort_values('time'),
                self.baro_left.sort_values('time'),
                on='time', direction=ASOF_DIRECTION, tolerance=ASOF_TOLERANCE_S)
            n = merged['b1_L'].notna().sum() if 'b1_L' in merged.columns else 0
            print(f"  After left baro merge: {n}/{len(merged)} matched")

        if self.baro_right is not None:
            merged = pd.merge_asof(
                merged.sort_values('time'),
                self.baro_right.sort_values('time'),
                on='time', direction=ASOF_DIRECTION, tolerance=ASOF_TOLERANCE_S)
            n = merged['b1_R'].notna().sum() if 'b1_R' in merged.columns else 0
            print(f"  After right baro merge: {n}/{len(merged)} matched")

        fz_cols = [c for c in ['force_z_R', 'force_z_L'] if c in merged.columns]
        merged  = merged.dropna(subset=fz_cols).reset_index(drop=True)

        if ENABLE_DYNAMIC_REZERO:
            for side in ['L', 'R']:
                fz_col    = f'force_z_{side}'
                side_cols = [f'b{i}_{side}' for i in range(1, 7)
                             if f'b{i}_{side}' in merged.columns]
                if not side_cols or fz_col not in merged.columns:
                    continue
                to_generic   = {fz_col: 'Fz'}
                to_generic.update({f'b{i}_{side}': f'b{i}' for i in range(1, 7)
                                   if f'b{i}_{side}' in merged.columns})
                from_generic = {v: k for k, v in to_generic.items()}
                merged = merged.rename(columns=to_generic)
                merged = rezero_barometers_when_fz_zero(
                    merged,
                    fz_threshold=FZ_ZERO_THRESHOLD,
                    min_zero_duration=MIN_ZERO_DURATION_S,
                    min_samples=MIN_ZERO_SAMPLES,
                )
                merged = merged.rename(columns=from_generic)

        self.synchronized_df = merged
        print(f"  Final synchronized dataset: {len(merged)} rows")
        self._save_barometer_csv()
        return merged

    def _save_barometer_csv(self):
        if self.baro_left is None and self.baro_right is None:
            return
        if self.baro_left is not None and self.baro_right is not None:
            baro_all = pd.merge_asof(
                self.baro_left.sort_values('time'),
                self.baro_right.sort_values('time'),
                on='time', direction=ASOF_DIRECTION, tolerance=ASOF_TOLERANCE_S)
        elif self.baro_left is not None:
            baro_all = self.baro_left.copy()
        else:
            baro_all = self.baro_right.copy()

        for ati_data in [self.ati_data_right, self.ati_data_left]:
            if ati_data is not None:
                cols = ['time'] + [c for c in ati_data.columns
                                   if c.startswith(('force_', 'torque_'))]
                baro_all = pd.merge_asof(
                    baro_all.sort_values('time'),
                    ati_data[cols].sort_values('time'),
                    on='time', direction=ASOF_DIRECTION, tolerance=ASOF_TOLERANCE_S)

        out = self.output_dir / "barometers_synchronized.csv"
        baro_all.to_csv(out, index=False)
        print(f"  Saved barometer data ({len(baro_all)} rows) → {out}")

    # ------------------------------------------------------------------ segmentation

    def segment_green_gripper(self, image):
        """Dispatcher: SAM if loaded, else HSV."""
        if self.sam_predictor is not None:
            return self._segment_with_sam(image)
        return self._segment_with_hsv(image)

    def _segment_with_hsv(self, image):
        """
        HSV segmentation using self.green_lower / green_upper / exclude_lower / exclude_upper.
        Bounds are set by the default values above OR overridden by run_hsv_picker().

        Pipeline
        --------
        1. Bilateral filter + HSV threshold → raw include mask.
        2. Morphology + convex hull → arm ROI.
        3. Fine HSV clipped to arm ROI.
        4. Subtract exclude-colour mask (if defined).
        5. Fill tiny PLA-reflection holes; keep real rib gaps.
        """
        image_area = image.shape[0] * image.shape[1]

        filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

        # ── 1. raw include mask ───────────────────────────────────────────────
        raw = cv2.inRange(hsv, self.green_lower, self.green_upper)

        # ── 2. coarse arm ROI ─────────────────────────────────────────────────
        coarse = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                                  np.ones((15, 15), np.uint8), iterations=2)
        coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN,
                                  np.ones((5, 5), np.uint8),  iterations=1)

        num_labels, lbl_map, stats, _ = cv2.connectedComponentsWithStats(
            coarse, connectivity=8)
        arm_mask = np.zeros_like(raw)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] > image_area * 0.01:
                pts  = np.argwhere(lbl_map == lbl)[:, ::-1].astype(np.int32)
                hull = cv2.convexHull(pts)
                cv2.drawContours(arm_mask, [hull], -1, 255, -1)

        # ── 3. fine mask clipped to ROI ───────────────────────────────────────
        fine = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8), iterations=1)
        fine = cv2.bitwise_and(fine, arm_mask)

        # ── 4. subtract exclude colours ───────────────────────────────────────
        if self.exclude_lower is not None and self.exclude_upper is not None:
            exc = cv2.inRange(hsv, self.exclude_lower, self.exclude_upper)
            exc = cv2.dilate(exc, np.ones((5, 5), np.uint8), iterations=1)
            fine = cv2.bitwise_and(fine, cv2.bitwise_not(exc))

        # ── 5. fill tiny holes ────────────────────────────────────────────────
        holes = cv2.bitwise_and(cv2.bitwise_not(fine), arm_mask)
        num_h, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(
            holes, connectivity=8)
        max_fill = image_area * 0.0015
        result   = fine.copy()
        for lbl in range(1, num_h):
            if h_stats[lbl, cv2.CC_STAT_AREA] < max_fill:
                result[h_labels == lbl] = 255
        return result

    def _segment_with_sam(self, image):
        """SAM segmentation with HSV mask_input prior + two-pass refinement."""
        h, w = image.shape[:2]
        image_area = h * w

        filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        hsv_img  = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        raw      = cv2.inRange(hsv_img, self.green_lower, self.green_upper)

        coarse = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                                  np.ones((15, 15), np.uint8), iterations=2)
        coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN,
                                  np.ones((5, 5), np.uint8),  iterations=1)

        num_labels, lbl_map, stats, centroids = cv2.connectedComponentsWithStats(
            coarse, connectivity=8)

        BOX_MARGIN = 15
        arm_info   = []
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] <= image_area * 0.01:
                continue
            cx, cy = centroids[lbl]
            x0 = stats[lbl, cv2.CC_STAT_LEFT];  y0 = stats[lbl, cv2.CC_STAT_TOP]
            bw = stats[lbl, cv2.CC_STAT_WIDTH];  bh = stats[lbl, cv2.CC_STAT_HEIGHT]

            prompt_pts = np.array([
                [cx, cy], [cx, y0 + bh * 0.20], [cx, y0 + bh * 0.80]
            ], dtype=np.float32)
            box = np.array([
                max(0,   x0      - BOX_MARGIN), max(0,   y0      - BOX_MARGIN),
                min(w-1, x0 + bw + BOX_MARGIN), min(h-1, y0 + bh + BOX_MARGIN),
            ], dtype=np.float32)

            arm_blob    = np.uint8(lbl_map == lbl) * 255
            logit_prior = cv2.resize(arm_blob, (256, 256)).astype(np.float32)
            logit_prior = np.where(logit_prior > 0, 10.0, -10.0)[np.newaxis]

            arm_info.append({'centroid': np.array([cx, cy], dtype=np.float32),
                             'prompt_pts': prompt_pts, 'box': box,
                             'logit_prior': logit_prior})

        if not arm_info:
            return self._segment_with_hsv(image)

        self.sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        combined = np.zeros((h, w), dtype=np.uint8)

        for i, arm in enumerate(arm_info):
            neg_pts = np.array(
                [a['centroid'] for j, a in enumerate(arm_info) if j != i],
                dtype=np.float32).reshape(-1, 2)
            points  = np.vstack([arm['prompt_pts'], neg_pts]) if len(neg_pts) else arm['prompt_pts']
            plabels = np.array([1]*len(arm['prompt_pts']) + [0]*len(neg_pts), dtype=np.int32)

            _, _, logits_out = self.sam_predictor.predict(
                point_coords=points, point_labels=plabels,
                box=arm['box'], mask_input=arm['logit_prior'],
                multimask_output=False)
            masks, _, _ = self.sam_predictor.predict(
                point_coords=points, point_labels=plabels,
                box=arm['box'], mask_input=logits_out,
                multimask_output=False)
            combined = cv2.bitwise_or(combined, (masks[0] * 255).astype(np.uint8))

        min_blob = image_area * 0.005
        num_c, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(
            combined, connectivity=8)
        clean = np.zeros_like(combined)
        for lbl in range(1, num_c):
            if c_stats[lbl, cv2.CC_STAT_AREA] >= min_blob:
                clean[c_labels == lbl] = 255
        return clean

    # ------------------------------------------------------------------ process_all

    def process_all(self, show_preview=False):
        if self.synchronized_df is None:
            raise RuntimeError("Call synchronize_all() before process_all().")

        sync_df      = self.synchronized_df
        image_lookup = {img['index']: img['image'] for img in self.images}
        metadata_rows = []

        print(f"\nProcessing {len(sync_df)} synchronized images...")
        for i, row in enumerate(tqdm(sync_df.itertuples(), total=len(sync_df),
                                     desc="Processing")):
            image = image_lookup.get(row.image_idx)
            if image is None:
                continue

            mask = self.segment_green_gripper(image)

            if i % 10 == 0:
                force_info = {}
                for side in ['R', 'L']:
                    force_info[f'force_x_{side}'] = getattr(row, f'force_x_{side}', 0.0)
                    force_info[f'force_y_{side}'] = getattr(row, f'force_y_{side}', 0.0)
                    force_info[f'force_z_{side}'] = getattr(row, f'force_z_{side}', 0.0)
                    force_info[f'force_magnitude_{side}'] = np.sqrt(
                        force_info[f'force_x_{side}']**2 +
                        force_info[f'force_y_{side}']**2 +
                        force_info[f'force_z_{side}']**2)
                self.save_visualization(image, mask, force_info, i)

            meta = {'image_idx': int(row.image_idx), 'time': float(row.time)}
            for side in ['R', 'L']:
                fx = getattr(row, f'force_x_{side}',  0.0)
                fy = getattr(row, f'force_y_{side}',  0.0)
                fz = getattr(row, f'force_z_{side}',  0.0)
                meta[f'fx_{side}']   = float(fx)
                meta[f'fy_{side}']   = float(fy)
                meta[f'fz_{side}']   = float(fz)
                meta[f'tx_{side}']   = float(getattr(row, f'torque_x_{side}', 0.0))
                meta[f'ty_{side}']   = float(getattr(row, f'torque_y_{side}', 0.0))
                meta[f'tz_{side}']   = float(getattr(row, f'torque_z_{side}', 0.0))
                meta[f'fmag_{side}'] = float(np.sqrt(fx**2 + fy**2 + fz**2))
            metadata_rows.append(meta)

        df = pd.DataFrame(metadata_rows)
        df.to_csv(self.output_dir / "metadata.csv", index=False)
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata_rows, f, indent=2)

        print(f"\n  Processed {len(metadata_rows)} images")
        print(f"  Created ~{len(metadata_rows)//10} visualizations")
        print(f"  Output directory: {self.output_dir}")
        return df

    # ------------------------------------------------------------------ viz

    def save_visualization(self, image, mask, force_info, idx):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image'); axes[0].axis('off')
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Gripper Mask');   axes[1].axis('off')
        overlay = image.copy(); overlay[mask > 0] = [0, 255, 0]
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Segmentation Overlay'); axes[2].axis('off')
        force_text = (
            f"R: ({force_info['force_x_R']:.3f}, {force_info['force_y_R']:.3f}, "
            f"{force_info['force_z_R']:.3f}) N  |  "
            f"L: ({force_info['force_x_L']:.3f}, {force_info['force_y_L']:.3f}, "
            f"{force_info['force_z_L']:.3f}) N"
        )
        fig.suptitle(force_text, fontsize=10)
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"viz_{idx:06d}.png", dpi=100)
        plt.close()


# ================================ MAIN =================================

def main():
    TEST_NUM    = test_num
    DATASET_DIR = (
        rf"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab"
        rf"\test data\test {TEST_NUM} - sensor v5"
    )

    bag_file             = Path(DATASET_DIR) / f"{TEST_NUM}_rgb_raw.bag"
    barometer_file_left  = Path(DATASET_DIR) / f"{TEST_NUM}_barometers_left.txt"
    barometer_file_right = Path(DATASET_DIR) / f"{TEST_NUM}_barometers_right.txt"

    print("=" * 60)
    print("FIN RAY GRIPPER PROCESSING")
    print("ROS Bag + Barometers + ATI Synchronization")
    print("=" * 60)

    processor = FinRayProcessor(
        DATASET_DIR, TEST_NUM,
        barometer_file_left=barometer_file_left,
        barometer_file_right=barometer_file_right,
    )

    # 1. Load ATI force data
    processor.load_ati_data()

    # 2. Extract images
    processor.extract_images(bag_file)

    # 3. ── Interactive HSV colour picker ──────────────────────────────────
    #    Opens two windows:
    #      Tab 1 → click pixels to INCLUDE  (green gripper)
    #      Tab 2 → click pixels to EXCLUDE  (background / operator hand / …)
    #    Computed HSV bounds are patched into the processor automatically.
    processor.run_hsv_picker()

    # 4. Load and process barometers
    processor.load_and_process_barometers()

    # 5. Synchronize all data streams
    processor.synchronize_all()

    # 6. Process images with the picked HSV bounds
    metadata = processor.process_all(show_preview=False)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Images extracted:   {len(processor.images)}")
    print(f"Synchronized rows:  {len(processor.synchronized_df)}")
    if processor.ati_data_right is not None:
        print(f"Right ATI samples:  {len(processor.ati_data_right)}")
    if processor.ati_data_left is not None:
        print(f"Left ATI samples:   {len(processor.ati_data_left)}")
    if processor.baro_left is not None:
        print(f"Left baro samples:  {len(processor.baro_left)}")
    if processor.baro_right is not None:
        print(f"Right baro samples: {len(processor.baro_right)}")
    print(f"Output directory:   {processor.output_dir}")
    print("\nGenerated files:")
    print(f"  ~{len(processor.synchronized_df)//10} visualizations  (visualizations/)")
    print(f"  metadata.csv / metadata.json")
    print(f"  barometer plots  (barometers_plots/)")
    print("=" * 60)

    return processor, metadata


if __name__ == "__main__":
    processor, metadata = main()