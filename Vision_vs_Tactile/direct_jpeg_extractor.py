"""
Direct JPEG Extractor - extracts JPEGs from ROS bag with timestamps,
processes left/right barometers, and synchronizes everything via merge_asof.

Requirements:
    pip install rosbags opencv-python tqdm
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

# barometers_processing is located in a folder called Barometers_Based_Tactile_Sensor in the parent directory, so we add that to the path to import it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Barometers_Based_Tactile_Sensor"))
from barometers_processing import (
    load_barometer_data,
    process_barometers,
    rezero_barometers_when_fz_zero,
)


# =========================== CONFIGURATION ===========================

test_num = 51011002

# --- Barometer processing params (same defaults as data_organization) ---
WARMUP_DURATION = 1.0           # seconds to remove from beginning (0 = disabled)
ENABLE_STEP_LEVELING = True
STEP_THRESHOLD_HPA = 5.0
STEP_WINDOW_SIZE = 200          # ~2 seconds at 100Hz

ENABLE_OUTLIER_REMOVAL = True
OUTLIER_THRESHOLD_MULTIPLIER = 30.0

DRIFT_REMOVAL_METHOD = "ema"    # "ema", "temperature", "both", "none"
EMA_ALPHA = 0.0001
EMA_ALPHA_OVERRIDE = {}
ZERO_AT_START = True

# --- Synchronization params ---
ASOF_DIRECTION = "nearest"
ASOF_TOLERANCE_S = 0.05         # seconds

# --- Dynamic re-zero based on Fz ~ 0 ---
ENABLE_DYNAMIC_REZERO = True
FZ_ZERO_THRESHOLD = 0.2        # |Fz| [N] considered "zero"
MIN_ZERO_DURATION_S = 0.01
MIN_ZERO_SAMPLES = 5


# ======================== IMAGE EXTRACTION ============================

class RosBagImageExtractor:
    """Extract JPEG images from a ROS bag with timestamps using rosbags."""

    def __init__(self, bag_path):
        self.bag_path = Path(bag_path)

    def extract_all_images(self):
        """
        Extract all CompressedImage messages from the bag.

        Returns list of dicts:
            {'index', 'image', 'timestamp_sec', 'size_bytes', 'width', 'height'}
        """
        print(f"Reading bag file: {self.bag_path}")

        images = []
        with AnyReader([self.bag_path]) as reader:
            # Print all topics for reference
            print("  Available topics:")
            for conn in reader.connections:
                print(f"    {conn.topic}  [{conn.msgtype}]")

            # Auto-detect CompressedImage topic
            image_connections = [
                c for c in reader.connections
                if 'CompressedImage' in c.msgtype
            ]
            if not image_connections:
                raise ValueError("No CompressedImage topic found in bag file.")

            topic_name = image_connections[0].topic
            print(f"  Using topic: {topic_name}")

            # Count messages for progress bar
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
                timestamp_sec = timestamp / 1e9

                # Decode JPEG
                jpeg_bytes = bytes(msg.data)
                img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if image is not None:
                    images.append({
                        'index': idx,
                        'image': image,
                        'timestamp_sec': timestamp_sec,
                        'size_bytes': len(jpeg_bytes),
                        'width': image.shape[1],
                        'height': image.shape[0],
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
        self.test_num = test_num

        self.ati_file = self.dataset_dir / f"{test_num}_ati_middle_trial.txt"
        self.barometer_file_left = barometer_file_left
        self.barometer_file_right = barometer_file_right

        self.output_dir = self.dataset_dir / "processed_data"
        self.viz_dir = self.output_dir / "visualizations"
        self.plots_dir = self.output_dir / "barometers_plots"

        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # HSV thresholds for green gripper
        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([85, 255, 255])

        # Data holders
        self.images = []
        self.ati_data = None
        self.baro_left = None
        self.baro_right = None
        self.synchronized_df = None

    def load_ati_data(self):
        """Load ATI force sensor data with ROS timestamps."""
        print(f"\nLoading ATI force data...")
        self.ati_data = pd.read_csv(self.ati_file)
        self.ati_data.columns = [
            'time', 'seq', 'stamp', 'frame_id',
            'force_x', 'force_y', 'force_z',
            'torque_x', 'torque_y', 'torque_z'
        ]
        # ROS timestamp in seconds (for merge_asof)
        self.ati_data['time'] = self.ati_data['stamp'] / 1e9

        print(f"  Loaded {len(self.ati_data)} force measurements")
        print(f"  Time range: {self.ati_data['time'].min():.2f} - {self.ati_data['time'].max():.2f} sec")
        print(f"  Duration: {self.ati_data['time'].max() - self.ati_data['time'].min():.2f} sec")

        return self.ati_data

    def extract_images(self, bag_file):
        """Extract images from bag file with ROS timestamps."""
        print(f"\nExtracting images from bag file...")
        extractor = RosBagImageExtractor(bag_file)
        self.images = extractor.extract_all_images()
        return self.images

    def _process_one_barometer(self, baro_path, side_label, plots_subdir):
        """Load and process a single barometer file, renaming columns with side suffix."""
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

        # Rename b1..b6 -> b1_L..b6_L (or _R) and t1..t6 -> t1_L..t6_L
        rename_map = {}
        for i in range(1, 7):
            if f'b{i}' in baro_df.columns:
                rename_map[f'b{i}'] = f'b{i}_{side_label}'
            if f't{i}' in baro_df.columns:
                rename_map[f't{i}'] = f't{i}_{side_label}'
        baro_df = baro_df.rename(columns=rename_map)

        print(f"  Processed {len(baro_df)} {side_label} barometer samples")
        return baro_df

    def load_and_process_barometers(self):
        """Load and process both left and right barometer files."""
        print("\nProcessing barometers...")

        if self.barometer_file_left and self.barometer_file_left.exists():
            self.baro_left = self._process_one_barometer(
                self.barometer_file_left, 'L', 'barometers_left'
            )
        else:
            print(f"  Warning: Left barometer file not found, skipping.")

        if self.barometer_file_right and self.barometer_file_right.exists():
            self.baro_right = self._process_one_barometer(
                self.barometer_file_right, 'R', 'barometers_right'
            )
        else:
            print(f"  Warning: Right barometer file not found, skipping.")

    def synchronize_all(self):
        """
        Synchronize images, ATI, and barometers using merge_asof on ROS timestamps.

        Creates self.synchronized_df with one row per image, containing:
        - image_idx, time
        - force_x..torque_z (from ATI)
        - b1_L..b6_L (from left barometers)
        - b1_R..b6_R (from right barometers)
        """
        print("\nSynchronizing all data streams...")

        # Build image DataFrame
        img_df = pd.DataFrame([
            {'image_idx': img['index'], 'time': img['timestamp_sec']}
            for img in self.images
        ]).sort_values('time').reset_index(drop=True)

        print(f"  Images:     {len(img_df)} frames")
        print(f"    Time range: {img_df['time'].min():.2f} - {img_df['time'].max():.2f}")

        # Merge ATI force data onto image timestamps
        ati_cols = ['time', 'force_x', 'force_y', 'force_z',
                    'torque_x', 'torque_y', 'torque_z']
        ati_for_merge = self.ati_data[ati_cols].sort_values('time')

        merged = pd.merge_asof(
            img_df, ati_for_merge,
            on='time', direction=ASOF_DIRECTION, tolerance=ASOF_TOLERANCE_S
        )
        print(f"  After ATI merge: {merged['force_z'].notna().sum()}/{len(merged)} matched")

        # Merge left barometers
        if self.baro_left is not None:
            merged = pd.merge_asof(
                merged.sort_values('time'),
                self.baro_left.sort_values('time'),
                on='time', direction=ASOF_DIRECTION, tolerance=ASOF_TOLERANCE_S
            )
            n_matched = merged['b1_L'].notna().sum() if 'b1_L' in merged.columns else 0
            print(f"  After left baro merge: {n_matched}/{len(merged)} matched")

        # Merge right barometers
        if self.baro_right is not None:
            merged = pd.merge_asof(
                merged.sort_values('time'),
                self.baro_right.sort_values('time'),
                on='time', direction=ASOF_DIRECTION, tolerance=ASOF_TOLERANCE_S
            )
            n_matched = merged['b1_R'].notna().sum() if 'b1_R' in merged.columns else 0
            print(f"  After right baro merge: {n_matched}/{len(merged)} matched")

        # Drop rows where ATI didn't match (images outside ATI time range)
        merged = merged.dropna(subset=['force_z']).reset_index(drop=True)

        # Dynamic re-zeroing when Fz ~ 0
        # rezero_barometers_when_fz_zero expects columns named 'Fz' and 'b1..b6',
        # so temporarily rename to match, run rezero for each side, then rename back.
        if ENABLE_DYNAMIC_REZERO:
            merged = merged.rename(columns={'force_z': 'Fz'})

            for side in ['L', 'R']:
                side_cols = [f'b{i}_{side}' for i in range(1, 7)
                             if f'b{i}_{side}' in merged.columns]
                if not side_cols:
                    continue
                # Rename b1_L -> b1, etc. so rezero can find them
                to_generic = {f'b{i}_{side}': f'b{i}' for i in range(1, 7)
                              if f'b{i}_{side}' in merged.columns}
                from_generic = {v: k for k, v in to_generic.items()}

                merged = merged.rename(columns=to_generic)
                merged = rezero_barometers_when_fz_zero(
                    merged,
                    fz_threshold=FZ_ZERO_THRESHOLD,
                    min_zero_duration=MIN_ZERO_DURATION_S,
                    min_samples=MIN_ZERO_SAMPLES,
                )
                merged = merged.rename(columns=from_generic)

            merged = merged.rename(columns={'Fz': 'force_z'})

        self.synchronized_df = merged
        print(f"  Final synchronized dataset: {len(merged)} rows")

        # Save full-resolution barometer data (not downsampled to image rate)
        self._save_barometer_csv()

        return merged

    def _save_barometer_csv(self):
        """Save processed barometer data at native rate, merging left+right by time."""
        if self.baro_left is None and self.baro_right is None:
            return

        if self.baro_left is not None and self.baro_right is not None:
            # Merge left and right on time
            baro_all = pd.merge_asof(
                self.baro_left.sort_values('time'),
                self.baro_right.sort_values('time'),
                on='time', direction=ASOF_DIRECTION, tolerance=ASOF_TOLERANCE_S
            )
        elif self.baro_left is not None:
            baro_all = self.baro_left.copy()
        else:
            baro_all = self.baro_right.copy()

        # Also merge ATI forces so the barometer CSV has force context
        if self.ati_data is not None:
            ati_cols = ['time', 'force_x', 'force_y', 'force_z',
                        'torque_x', 'torque_y', 'torque_z']
            baro_all = pd.merge_asof(
                baro_all.sort_values('time'),
                self.ati_data[ati_cols].sort_values('time'),
                on='time', direction=ASOF_DIRECTION, tolerance=ASOF_TOLERANCE_S
            )

        baro_csv_path = self.output_dir / "barometers_synchronized.csv"
        baro_all.to_csv(baro_csv_path, index=False)
        print(f"  Saved barometer data ({len(baro_all)} rows) to: {baro_csv_path}")

    def segment_green_gripper(self, image):
        """Segment the green Fin Ray gripper."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            refined = np.zeros_like(mask)
            cv2.drawContours(refined, [largest], -1, 255, -1)
            return refined

        return mask

    def process_all(self, show_preview=False):
        """Process all synchronized images with segmentation."""
        if self.synchronized_df is None:
            raise RuntimeError("Call synchronize_all() before process_all().")

        sync_df = self.synchronized_df
        print(f"\nProcessing {len(sync_df)} synchronized images...")

        # Build image lookup by index
        image_lookup = {img['index']: img['image'] for img in self.images}

        metadata_rows = []

        for i, row in enumerate(tqdm(
            sync_df.itertuples(), total=len(sync_df), desc="Processing"
        )):
            image = image_lookup.get(row.image_idx)
            if image is None:
                continue

            # Segment gripper
            mask = self.segment_green_gripper(image)

            # Visualization every 10 images
            if i % 10 == 0:
                force_info = {
                    'force_x': row.force_x,
                    'force_y': row.force_y,
                    'force_z': row.force_z,
                    'torque_x': row.torque_x,
                    'torque_y': row.torque_y,
                    'torque_z': row.torque_z,
                    'force_magnitude': np.sqrt(
                        row.force_x**2 + row.force_y**2 + row.force_z**2
                    ),
                }
                self.save_visualization(image, mask, force_info, i)

            # Build metadata row: timestamp, image index, ATI forces only
            meta = {
                'image_idx': int(row.image_idx),
                'time': float(row.time),
                'fx': float(row.force_x),
                'fy': float(row.force_y),
                'fz': float(row.force_z),
                'tx': float(row.torque_x),
                'ty': float(row.torque_y),
                'tz': float(row.torque_z),
                'fmag': float(np.sqrt(
                    row.force_x**2 + row.force_y**2 + row.force_z**2
                )),
            }

            metadata_rows.append(meta)

        # Save metadata
        df = pd.DataFrame(metadata_rows)
        df.to_csv(self.output_dir / "metadata.csv", index=False)

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata_rows, f, indent=2)

        print(f"\n  Processed {len(metadata_rows)} images")
        print(f"  Created ~{len(metadata_rows)//10} visualizations")
        print(f"  Output directory: {self.output_dir}")

        return df

    def show_preview(self, image, mask, force_info, idx):
        """Show preview window."""
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 0]
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        top = np.hstack([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        bottom = np.hstack([blended, blended])
        display = np.vstack([top, bottom])

        h, w = display.shape[:2]
        if w > 1400:
            scale = 1400 / w
            display = cv2.resize(display, (int(w*scale), int(h*scale)))

        force_text = f"F: ({force_info['force_x']:.2f}, {force_info['force_y']:.2f}, {force_info['force_z']:.2f}) N"
        mag_text = f"Magnitude: {force_info['force_magnitude']:.2f} N"

        cv2.putText(display, f"Frame {idx}/{len(self.images)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, force_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, mag_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "Press any key to continue...", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Preview (Top: Original & Mask, Bottom: Overlay)', display)
        cv2.waitKey(1000)

    def save_visualization(self, image, mask, force_info, idx):
        """Save visualization to file."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Gripper Mask')
        axes[1].axis('off')

        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 0]
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Segmentation Overlay')
        axes[2].axis('off')

        force_text = (
            f"F: ({force_info['force_x']:.3f}, {force_info['force_y']:.3f}, "
            f"{force_info['force_z']:.3f}) N | Mag: {force_info['force_magnitude']:.3f} N"
        )
        fig.suptitle(force_text, fontsize=10)

        plt.tight_layout()
        plt.savefig(self.viz_dir / f"viz_{idx:06d}.png", dpi=100)
        plt.close()


# ================================ MAIN =================================

def main():
    """Main execution."""

    TEST_NUM = test_num
    DATASET_DIR = rf"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data\test {TEST_NUM} - sensor v5"

    bag_file = Path(DATASET_DIR) / f"{TEST_NUM}_rgb_compressed.bag"
    barometer_file_left = Path(DATASET_DIR) / f"{TEST_NUM}_barometers_left.txt"
    barometer_file_right = Path(DATASET_DIR) / f"{TEST_NUM}_barometers_right.txt"

    print("="*60)
    print("FIN RAY GRIPPER PROCESSING")
    print("ROS Bag + Barometers + ATI Synchronization")
    print("="*60)

    # Create processor with barometer file paths
    processor = FinRayProcessor(
        DATASET_DIR, TEST_NUM,
        barometer_file_left=barometer_file_left,
        barometer_file_right=barometer_file_right,
    )

    # 1. Load ATI force data
    processor.load_ati_data()

    # 2. Extract images with ROS timestamps
    processor.extract_images(bag_file)

    # 3. Load and process barometers (left + right)
    processor.load_and_process_barometers()

    # 4. Synchronize all data streams via merge_asof
    processor.synchronize_all()

    # 5. Process images (segmentation, visualization, metadata)
    metadata = processor.process_all(show_preview=False)

    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Images extracted:     {len(processor.images)}")
    print(f"Synchronized rows:    {len(processor.synchronized_df)}")
    print(f"Force measurements:   {len(processor.ati_data)}")
    if processor.baro_left is not None:
        print(f"Left baro samples:    {len(processor.baro_left)}")
    if processor.baro_right is not None:
        print(f"Right baro samples:   {len(processor.baro_right)}")
    print(f"Output directory:     {processor.output_dir}")
    print("\nGenerated files:")
    print(f"  - ~{len(processor.synchronized_df)//10} visualizations (in visualizations/)")
    print(f"  - metadata.csv (images + forces + barometers, synchronized)")
    print(f"  - metadata.json")
    print(f"  - barometer processing plots (in plots/)")
    print("="*60)

    return processor, metadata


if __name__ == "__main__":
    processor, metadata = main()
