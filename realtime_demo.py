#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Real-Time Tactile Sensor Demo

A minimal, standalone script for testing real-time predictions.
No ROS required - just connects to Arduino and prints predictions.

This is the simplest way to test your trained model in real-time.

Usage:
    python realtime_demo.py
    python realtime_demo.py --port COM3  # Windows
    python realtime_demo.py --port /dev/ttyACM0  # Linux
    
Requirements:
    pip install numpy pyserial lightgbm scikit-learn

Author: Based on Aurora Ruggeri's offline pipeline
"""

import argparse
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("Error: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)


# ============================================================
# ===================== CONFIGURATION ========================
# ============================================================

# UPDATE THESE PATHS TO YOUR ACTUAL MODEL LOCATION
MODEL_DIR = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models")
SENSOR_VERSION = "5.160"  # Match your trained model

# Feature engineering settings (MUST match training)
WINDOW_SIZE = 10
USE_SECOND_DERIVATIVE = True
APPLY_DENOISING = True
DENOISE_WINDOW = 5

# Drift removal settings
ENABLE_DRIFT_REMOVAL = True
EMA_ALPHA = 0.0001

# Other settings
WARMUP_SAMPLES = 100
BAUD_RATE = 115200


# ============================================================
# =================== PREPROCESSOR ===========================
# ============================================================

class SimplePreprocessor:
    """Minimal online preprocessor matching offline pipeline."""
    
    def __init__(self):
        self.n_sensors = 6
        self.ema = np.zeros(6)
        self.ema_init = False
        self.zero_offset = np.zeros(6)
        self.zero_init = False
        self.denoise_buf = deque(maxlen=DENOISE_WINDOW)
        self.prev = None
        self.prev_d1 = None
        
    def process(self, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process one sample, return (pressure, d1, d2)."""
        p = raw.astype(np.float64)
        
        # Zero at start
        if not self.zero_init:
            self.zero_offset = p.copy()
            self.zero_init = True
        p = p - self.zero_offset
        
        # EMA drift removal
        if ENABLE_DRIFT_REMOVAL:
            if not self.ema_init:
                self.ema = p.copy()
                self.ema_init = True
                p = np.zeros(6)
            else:
                self.ema = EMA_ALPHA * p + (1 - EMA_ALPHA) * self.ema
                p = p - self.ema
        
        # Denoise
        if APPLY_DENOISING:
            self.denoise_buf.append(p.copy())
            if len(self.denoise_buf) >= DENOISE_WINDOW:
                p = np.mean(np.array(list(self.denoise_buf)), axis=0)
        
        # Derivatives
        d1 = np.zeros(6) if self.prev is None else p - self.prev
        d2 = np.zeros(6) if self.prev_d1 is None else d1 - self.prev_d1
        
        self.prev = p.copy()
        self.prev_d1 = d1.copy()
        
        return p, d1, d2


class SimpleWindowBuilder:
    """Minimal sliding window feature builder."""
    
    def __init__(self):
        self.p_buf = deque(maxlen=WINDOW_SIZE + 1)
        self.d1_buf = deque(maxlen=WINDOW_SIZE + 1)
        self.d2_buf = deque(maxlen=WINDOW_SIZE + 1)
        
    def add(self, p: np.ndarray, d1: np.ndarray, d2: np.ndarray):
        self.p_buf.append(p)
        self.d1_buf.append(d1)
        self.d2_buf.append(d2)
        
    def ready(self) -> bool:
        return len(self.p_buf) >= WINDOW_SIZE + 1
    
    def features(self) -> np.ndarray:
        p_flat = np.array(list(self.p_buf)).flatten()
        d1_flat = np.array(list(self.d1_buf)).flatten()
        if USE_SECOND_DERIVATIVE:
            d2_flat = np.array(list(self.d2_buf)).flatten()
            return np.concatenate([p_flat, d1_flat, d2_flat])
        return np.concatenate([p_flat, d1_flat])


# ============================================================
# ======================= MAIN ===============================
# ============================================================

def find_arduino_port() -> str:
    """Auto-detect Arduino port."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description + p.device).lower()
        if any(x in desc for x in ['arduino', 'mkr', 'acm', 'usb serial']):
            return p.device
    if ports:
        return ports[0].device
    raise RuntimeError("No serial ports found!")


def parse_line(line: str) -> Optional[np.ndarray]:
    """Parse Arduino line, return 6 pressures or None."""
    try:
        parts = line.strip().split(',')
        if len(parts) != 13 or parts[0].lower().startswith('time'):
            return None
        # Pressures at indices 1,3,5,7,9,11
        return np.array([float(parts[i]) for i in [1, 3, 5, 7, 9, 11]])
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description='Real-time tactile demo')
    parser.add_argument('--port', type=str, help='Serial port')
    parser.add_argument('--model-dir', type=str, help='Model directory')
    parser.add_argument('--version', type=str, default=SENSOR_VERSION, help='Model version')
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    version = args.version
    
    # Load models
    model_path = model_dir / f"lightgbm_sliding_window_model_v{version}.pkl"
    scaler_path = model_dir / f"scaler_sliding_window_v{version}.pkl"
    
    print("=" * 60)
    print("Real-Time Tactile Sensor Demo")
    print("=" * 60)
    
    if not model_path.exists():
        print(f"\nERROR: Model not found at: {model_path}")
        print("\nPlease update MODEL_DIR and SENSOR_VERSION in this script,")
        print("or use --model-dir and --version arguments.")
        sys.exit(1)
        
    if not scaler_path.exists():
        print(f"\nERROR: Scaler not found at: {scaler_path}")
        sys.exit(1)
    
    print(f"\nLoading model: {model_path}")
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Loaded {len(models)} models for: x, y, fx, fy, fz")
    
    # Connect to Arduino
    port = args.port
    if not port:
        print("\nAuto-detecting Arduino port...")
        port = find_arduino_port()
    
    print(f"Connecting to {port}...")
    ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
    time.sleep(0.5)
    ser.write(b'S')  # Handshake
    ser.flush()
    print("Connected!")
    
    # Skip boot messages
    time.sleep(1.0)
    while ser.in_waiting:
        ser.readline()
    
    # Initialize
    prep = SimplePreprocessor()
    win = SimpleWindowBuilder()
    targets = ['x', 'y', 'fx', 'fy', 'fz']
    
    sample_count = 0
    pred_count = 0
    
    print(f"\nWarmup: {WARMUP_SAMPLES} samples...")
    print("\n" + "=" * 70)
    print(f"{'Sample':>8} {'X(mm)':>8} {'Y(mm)':>8} {'Fx(N)':>8} {'Fy(N)':>8} {'Fz(N)':>8} {'Contact':>8}")
    print("=" * 70)
    
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
                
            pressures = parse_line(line)
            if pressures is None:
                continue
                
            sample_count += 1
            
            # Warmup
            if sample_count < WARMUP_SAMPLES:
                if sample_count % 20 == 0:
                    print(f"\rWarmup: {sample_count}/{WARMUP_SAMPLES}", end='', flush=True)
                continue
            elif sample_count == WARMUP_SAMPLES:
                print(f"\rWarmup complete!                    ")
            
            # Process
            p, d1, d2 = prep.process(pressures)
            win.add(p, d1, d2)
            
            if not win.ready():
                continue
                
            # Predict
            features = win.features()
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            preds = {}
            for model, target in zip(models, targets):
                preds[target] = float(model.predict(features_scaled)[0])
            
            contact = "YES" if abs(preds['fz']) >= 0.1 else "NO"
            pred_count += 1
            
            print(f"\r{sample_count:>8} "
                  f"{preds['x']:>8.2f} "
                  f"{preds['y']:>8.2f} "
                  f"{preds['fx']:>8.3f} "
                  f"{preds['fy']:>8.3f} "
                  f"{preds['fz']:>8.3f} "
                  f"{contact:>8}", end='', flush=True)
                  
    except KeyboardInterrupt:
        print(f"\n\nStopped. Processed {sample_count} samples, made {pred_count} predictions.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
