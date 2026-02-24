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

# barometers_processing is in a sibling folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Barometers_Based_Tactile_Sensor"))

from utils.barometer_processing import (  # type: ignore[import]
    load_barometer_data,
    process_barometers,
    rezero_barometers_when_fz_zero,
)

# interactive HSV colour picker (same directory as this script)
from hsv_color_picker import pick_hsv_bounds, patch_processor


# =========================== CONFIGURATION ===========================

test_num = 51011008

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

# --- Colour segmentation toggles ---
DETECT_PURPLE   = False    # set False to skip purple segmentation entirely
DETECT_GREEN    = True    # set False to skip green segmentation entirely

# --- HSV colour bounds for the purple gripper ---
# Set USE_HSV_PICKER = False to skip the interactive picker and use the
# hardcoded bounds below directly (faster, no clicking needed).
# Set USE_HSV_PICKER = True only when lighting conditions change significantly.
USE_HSV_PICKER  = False

HSV_PURPLE_LOWER = np.array([106,  16,  54])   # [H_min, S_min, V_min]
HSV_PURPLE_UPPER = np.array([154, 122, 175])   # [H_max, S_max, V_max]

# --- HSV colour bounds for the green object ---
# Teal-green fin ray gripper: H 48-92, V≥38 keeps shadowed gripper areas
# while rejecting truly dark cables (V<30). S≥60 keeps real saturation.
HSV_GREEN_LOWER  = np.array([ 48,  60,  38])   # [H_min, S_min, V_min]
HSV_GREEN_UPPER  = np.array([ 92, 255, 215])   # [H_max, S_max, V_max]

# Optional: exclude a colour range (e.g. yellow background, skin tones).
# Set to None to disable.
HSV_EXCLUDE_LOWER = np.array([ 52,   0,   2])
HSV_EXCLUDE_UPPER = np.array([ 89, 154, 135])

# --- ROI (Region of Interest) — fractions of image size (0.0 – 1.0) ---
# Anything outside this rectangle is zeroed before segmentation to suppress
# noise sources (e.g. the purple LED glow at the bottom, edge reflections).
# Format: (x_left, y_top, x_right, y_bottom)  — all as fractions.
# Set to None to disable (use full frame).
ROI_FRAC: tuple | None = (0.10, 0.02, 0.87, 0.85)

# --- HSV picker display settings (only used when USE_HSV_PICKER = True) ---
PICKER_PREVIEW_MAX_DIM     = 900
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
        self.seg_dir    = self.output_dir  / "segmented_images"

        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.seg_dir.mkdir(parents=True, exist_ok=True)

        # HSV thresholds — set from config, overridden by run_hsv_picker() if used
        self.purple_lower  = HSV_PURPLE_LOWER.copy()
        self.purple_upper  = HSV_PURPLE_UPPER.copy()
        self.green_lower   = HSV_GREEN_LOWER.copy()
        self.green_upper   = HSV_GREEN_UPPER.copy()
        self.exclude_lower = HSV_EXCLUDE_LOWER.copy() if HSV_EXCLUDE_LOWER is not None else None
        self.exclude_upper = HSV_EXCLUDE_UPPER.copy() if HSV_EXCLUDE_UPPER is not None else None

        # ROI: (x1_frac, y1_frac, x2_frac, y2_frac) or None for full frame
        self.roi_frac = ROI_FRAC

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

        purple_lower, purple_upper, exclude_lower, exclude_upper = pick_hsv_bounds(sample)
        patch_processor(self, purple_lower, purple_upper, exclude_lower, exclude_upper)

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
        self._plot_barometers_with_forces(baro_all)

    def _plot_barometers_with_forces(self, baro_all):
        """
        For each side (L, R) that has barometer data, save a figure with:
          Row 1 – all available barometer channels (b1…b6)
          Row 2 – fx, fy, fz from the corresponding ATI sensor
        """
        t0 = baro_all['time'].min()
        t_rel = baro_all['time'] - t0   # relative time in seconds

        for side in ['L', 'R']:
            baro_cols  = [f'b{i}_{side}' for i in range(1, 7)
                          if f'b{i}_{side}' in baro_all.columns]
            force_cols = [f'force_{ax}_{side}' for ax in ['x', 'y', 'z']
                          if f'force_{ax}_{side}' in baro_all.columns]

            if not baro_cols:
                continue

            has_forces = len(force_cols) == 3
            n_rows = 2 if has_forces else 1
            fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows),
                                     sharex=True)
            if n_rows == 1:
                axes = [axes]

            ax_baro = axes[0]
            for col in baro_cols:
                ax_baro.plot(t_rel, baro_all[col], linewidth=0.8, label=col)
            ax_baro.set_ylabel('Pressure (hPa)')
            ax_baro.set_title(f'Barometers – side {side}')
            ax_baro.legend(fontsize=7, ncol=3, loc='upper right')
            ax_baro.grid(alpha=0.3)

            if has_forces:
                ax_f = axes[1]
                colors = {'x': '#e63946', 'y': '#2a9d8f', 'z': '#457b9d'}
                labels = {'x': f'Fx_{side}', 'y': f'Fy_{side}', 'z': f'Fz_{side}'}
                for ax_name in ['x', 'y', 'z']:
                    col = f'force_{ax_name}_{side}'
                    ax_f.plot(t_rel, baro_all[col], linewidth=0.9,
                              color=colors[ax_name], label=labels[ax_name])
                ax_f.axhline(0, color='k', linewidth=0.5, linestyle='--')
                ax_f.set_ylabel('Force (N)')
                ax_f.set_title(f'ATI Forces – side {side}')
                ax_f.legend(fontsize=8, loc='upper right')
                ax_f.grid(alpha=0.3)

            axes[-1].set_xlabel('Time (s)')
            fig.tight_layout()
            save_path = self.plots_dir / f'barometers_forces_{side}.png'
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"  Saved barometer+force plot → {save_path.name}")

    # ------------------------------------------------------------------ segmentation

    def _roi_mask(self, h, w):
        """Build a binary ROI mask (uint8, 255 inside, 0 outside)."""
        roi = np.zeros((h, w), dtype=np.uint8)
        if self.roi_frac is None:
            roi[:] = 255
        else:
            x1f, y1f, x2f, y2f = self.roi_frac
            x1 = int(x1f * w); y1 = int(y1f * h)
            x2 = int(x2f * w); y2 = int(y2f * h)
            roi[y1:y2, x1:x2] = 255
        return roi

    def _hsv_mask(self, hsv, lower, upper, h, w,
                  close_ksize=11, close_iters=2,
                  open_ksize=5,  open_iters=1,
                  min_blob_frac=0.003):
        """
        HSV colour segmentation.

        1. Threshold within [lower, upper].
        2. Apply ROI.
        3. Morph close (fill holes) then open (remove speckles).
        4. Drop blobs smaller than min_blob_frac of the image area.
        """
        image_area = h * w
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_and(mask, self._roi_mask(h, w))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel,
                                iterations=close_iters)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel,
                                iterations=open_iters)
        num_labels, lbl_map, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        result = np.zeros_like(mask)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= image_area * min_blob_frac:
                result[lbl_map == lbl] = 255
        return result

    @staticmethod
    def _fill_outer_contours(mask, min_area=0):
        """
        For each external contour in mask whose area >= min_area,
        draw it fully filled on a blank canvas.
        Everything inside the outermost boundary becomes white,
        regardless of internal holes or gaps.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(filled, [cnt], -1, 255, cv2.FILLED)
        return filled

    def segment_gripper(self, image):
        """
        HSV colour segmentation returning (purple_mask, green_mask).

        Purple: tight closing (11 px) to preserve fine detail.
        Green:  minimal closing (3 px) for pixel-precise HSV detection,
                then outer-contour fill so everything inside the gripper
                arm boundary is solid (no internal holes from rib gaps).
        The exclude range is applied only to the purple mask.

        Returns
        -------
        purple_mask : np.uint8 binary image (0 / 255)
        green_mask  : np.uint8 binary image (0 / 255)
        """
        h, w = image.shape[:2]

        filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

        if DETECT_PURPLE:
            purple_mask = self._hsv_mask(hsv, self.purple_lower, self.purple_upper, h, w)
            if self.exclude_lower is not None and self.exclude_upper is not None:
                exc = cv2.inRange(hsv, self.exclude_lower, self.exclude_upper)
                exc = cv2.dilate(exc, np.ones((5, 5), np.uint8), iterations=1)
                purple_mask = cv2.bitwise_and(purple_mask, cv2.bitwise_not(exc))
        else:
            purple_mask = np.zeros((h, w), dtype=np.uint8)

        if DETECT_GREEN:
            # Step 1: pixel-precise HSV detection
            raw_green = self._hsv_mask(
                hsv, self.green_lower, self.green_upper, h, w,
                close_ksize=3, close_iters=1,
                open_ksize=3,  open_iters=1,
                min_blob_frac=0.015,   # raise to kill small false-positive blobs
            )
            # Step 2: boundary smoothing — rounds off jagged pixel-level bumps,
            # fills small notches, and breaks thin cable connections before contour
            # tracing.  The open kernel must be larger than the cable neck width.
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
            raw_green = cv2.morphologyEx(raw_green, cv2.MORPH_CLOSE, close_k, iterations=3)
            raw_green = cv2.morphologyEx(raw_green, cv2.MORPH_OPEN,  open_k,  iterations=1)
            # Step 3: fill each outer contour solid — removes all internal holes
            green_mask = self._fill_outer_contours(
                raw_green, min_area=h * w * 0.015)   # consistent threshold with step 1
        else:
            green_mask = np.zeros((h, w), dtype=np.uint8)

        return purple_mask, green_mask

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

            purple_mask, green_mask = self.segment_gripper(image)
            combined_mask = cv2.bitwise_or(purple_mask, green_mask)

            # Save background-zeroed (segmented) image for forces_for_free.py
            seg_img = image.copy()
            seg_img[combined_mask == 0] = 0
            seg_filename = f"seg_{int(row.image_idx):06d}.jpg"
            seg_path = self.seg_dir / seg_filename
            cv2.imwrite(str(seg_path), seg_img)

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
                self.save_visualization(image, purple_mask, green_mask, force_info, i)

            meta = {'image_idx': int(row.image_idx), 'time': float(row.time),
                    'seg_path': str(seg_path)}
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

    def save_visualization(self, image, purple_mask, green_mask, force_info, idx):
        # Combined colour mask on black background
        # purple pixels → purple (BGR: 180, 0, 180), green pixels → green (BGR: 0, 200, 0)
        mask_bgr = np.zeros((*purple_mask.shape, 3), dtype=np.uint8)
        mask_bgr[purple_mask > 0] = [180,   0, 180]
        mask_bgr[green_mask  > 0] = [  0, 200,   0]

        # Overlay: blend both coloured regions onto the original
        overlay = image.copy()
        overlay[purple_mask > 0] = [180,   0, 180]
        overlay[green_mask  > 0] = [  0, 200,   0]
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image'); axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Colour Masks (purple + green)'); axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Overlay'); axes[2].axis('off')

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

    # 3. ── HSV colour bounds ───────────────────────────────────────────────
    if USE_HSV_PICKER:
        # Opens two interactive windows to click-pick include / exclude colours.
        # Use this only when lighting conditions have changed.
        processor.run_hsv_picker()
    else:
        print(f"\n  [HSV] Using hardcoded bounds: "
              f"lower={HSV_PURPLE_LOWER}  upper={HSV_PURPLE_UPPER}")

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
