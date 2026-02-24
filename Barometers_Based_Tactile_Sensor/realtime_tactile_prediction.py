#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Tactile Sensor Prediction System - 

PERFORMANCE TARGETS:
- Original: ~80-120 Hz
- This version: 200-500+ Hz (with Numba), 150-250 Hz (without)

Author: Optimized from Aurora Ruggeri's code
"""

import argparse
import pickle
import time
import threading
import warnings
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import serial
import serial.tools.list_ports

# Try to import Numba for JIT compilation
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

warnings.filterwarnings('ignore')


# ============================================================
# =================== CONFIGURATION ==========================
# ============================================================

class Config:
    """Ultra-optimized configuration."""
    
    def __init__(self):
        # Serial
        self.port = "/dev/ttyACM0"
        self.baud_rate = 115200
        self.serial_timeout = 0.0  # Non-blocking
        
        # Paths
        self.model_dir = Path(r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\averaged models")
        self.sensor_version = "5.020"
        
        # Sensor
        self.n_sensors = 6
        self.target_cols = ["x", "y", "fx", "fy", "fz"]
        
        # Features
        self.window_size = 10
        self.use_second_derivative = True
        self.warmup_samples = 200
        
        # EMA drift removal
        self.ema_alpha = 0.005
        
        # Contact
        self.no_contact_threshold = 0.5
        
        # Output
        self.enable_plot = False
        self.enable_csv_logging = False
        self.csv_log_path = None
        self.debug_log_interval = 100
        
        # Viz
        self.viz_update_interval_ms = 33
        self.viz_buffer_seconds = 30.0
        
    def get_model_path(self) -> Path:
        return self.model_dir / f"lightgbm_sliding_window_model_v{self.sensor_version}.pkl"
    
    def get_scaler_path(self) -> Path:
        return self.model_dir / f"scaler_sliding_window_v{self.sensor_version}.pkl"


# ============================================================
# ================ NUMBA-OPTIMIZED FUNCTIONS =================
# ============================================================

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def ema_update(ema, raw, alpha, in_contact):
        """Vectorized EMA update with contact gating."""
        if not in_contact:
            beta = 1.0 - alpha
            for i in range(len(ema)):
                ema[i] = beta * ema[i] + alpha * raw[i]
    
    @njit(cache=True, fastmath=True)
    def center_signal(raw, ema, out):
        """Vectorized signal centering."""
        for i in range(len(raw)):
            out[i] = raw[i] - ema[i]
    
    @njit(cache=True, fastmath=True)
    def compute_derivatives(current, prev, d1_out, prev_d1, d2_out):
        """Compute derivatives in-place."""
        for i in range(len(current)):
            d1_out[i] = current[i] - prev[i]
            d2_out[i] = d1_out[i] - prev_d1[i]
    
    @njit(cache=True, fastmath=True)
    def max_abs(arr):
        """Fast max absolute value."""
        m = 0.0
        for x in arr:
            v = abs(x)
            if v > m:
                m = v
        return m
    
    @njit(cache=True, fastmath=True)
    def roll_and_flatten(buffer, shift, n_rows, n_cols, output):
        """Fast roll and flatten."""
        idx = 0
        for row in range(n_rows):
            src_row = (row + shift) % n_rows
            for col in range(n_cols):
                output[idx] = buffer[src_row, col]
                idx += 1

else:
    # NumPy fallbacks (slower but functional)
    def ema_update(ema, raw, alpha, in_contact):
        if not in_contact:
            ema[:] = (1 - alpha) * ema + alpha * raw
    
    def center_signal(raw, ema, out):
        np.subtract(raw, ema, out=out)
    
    def compute_derivatives(current, prev, d1_out, prev_d1, d2_out):
        np.subtract(current, prev, out=d1_out)
        np.subtract(d1_out, prev_d1, out=d2_out)
    
    def max_abs(arr):
        return np.max(np.abs(arr))
    
    def roll_and_flatten(buffer, shift, n_rows, n_cols, output):
        rolled = np.roll(buffer, shift, axis=0)
        output[:] = rolled.ravel()


# ============================================================
# ============== ULTRA-FAST PREPROCESSOR =====================
# ============================================================

class UltraPreprocessor:
    """Zero-allocation preprocessor with JIT-optimized hot path."""
    
    def __init__(self, config: Config):
        self.config = config
        self.n = config.n_sensors
        
        # State
        self.ema = np.zeros(self.n, dtype=np.float64)
        self.ema_init = False
        self.alpha = config.ema_alpha
        
        self.prev_p = np.zeros(self.n, dtype=np.float64)
        self.prev_d1 = np.zeros(self.n, dtype=np.float64)
        self.deriv_init = False
        
        self.in_contact = False
        
        # Work arrays (pre-allocated, never reallocated)
        self._centered = np.zeros(self.n, dtype=np.float64)
        self._d1 = np.zeros(self.n, dtype=np.float64)
        self._d2 = np.zeros(self.n, dtype=np.float64)
    
    def process(self, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process sample - fully optimized hot path."""
        
        # Init EMA
        if not self.ema_init:
            self.ema[:] = raw
            self.ema_init = True
            return self._centered, self._d1, self._d2
        
        # Center
        center_signal(raw, self.ema, self._centered)
        
        # Contact detection with hysteresis
        max_dev = max_abs(self._centered)
        if self.in_contact:
            if max_dev < 0.2:
                self.in_contact = False
        else:
            if max_dev > 0.5:
                self.in_contact = True
        
        # Gated EMA
        ema_update(self.ema, raw, self.alpha, self.in_contact)
        if not self.in_contact:
            center_signal(raw, self.ema, self._centered)
        
        # Derivatives
        if not self.deriv_init:
            self.prev_p[:] = self._centered
            self.deriv_init = True
            return self._centered, self._d1, self._d2
        
        compute_derivatives(self._centered, self.prev_p, self._d1,
                          self.prev_d1, self._d2)
        self.prev_p[:] = self._centered
        self.prev_d1[:] = self._d1
        
        return self._centered, self._d1, self._d2
    
    def reset(self):
        """Reset all state."""
        self.ema.fill(0)
        self.ema_init = False
        self.prev_p.fill(0)
        self.prev_d1.fill(0)
        self.deriv_init = False
        self.in_contact = False


# ============================================================
# ============== ULTRA-FAST WINDOW BUILDER ===================
# ============================================================

class UltraWindowBuilder:
    """Zero-allocation window builder with JIT-optimized feature extraction."""
    
    def __init__(self, config: Config):
        self.window_size = config.window_size
        self.n = config.n_sensors
        self.use_d2 = config.use_second_derivative
        
        self.window_len = self.window_size + 1
        
        # Circular buffers (C-contiguous for cache efficiency)
        self.p_buf = np.zeros((self.window_len, self.n), dtype=np.float64, order='C')
        self.d1_buf = np.zeros((self.window_len, self.n), dtype=np.float64, order='C')
        self.d2_buf = np.zeros((self.window_len, self.n), dtype=np.float64, order='C')
        
        self.idx = 0
        self.count = 0
        
        # Feature array
        n_elem = self.window_len * self.n
        self.n_features = 3 * n_elem if self.use_d2 else 2 * n_elem
        self.features = np.zeros(self.n_features, dtype=np.float64)
    
    def add(self, p: np.ndarray, d1: np.ndarray, d2: np.ndarray):
        """Add sample to circular buffer."""
        i = self.idx % self.window_len
        self.p_buf[i] = p
        self.d1_buf[i] = d1
        self.d2_buf[i] = d2
        self.idx += 1
        self.count = min(self.count + 1, self.window_len)
    
    def ready(self) -> bool:
        """Check if window full."""
        return self.count >= self.window_len
    
    def get_features(self) -> np.ndarray:
        """Extract features - fully optimized."""
        if not self.ready():
            return None
        
        shift = -(self.idx % self.window_len)
        n_elem = self.window_len * self.n
        
        # Use JIT-optimized roll and flatten
        roll_and_flatten(self.p_buf, shift, self.window_len, self.n,
                        self.features[:n_elem])
        roll_and_flatten(self.d1_buf, shift, self.window_len, self.n,
                        self.features[n_elem:2*n_elem])
        if self.use_d2:
            roll_and_flatten(self.d2_buf, shift, self.window_len, self.n,
                           self.features[2*n_elem:])
        
        return self.features
    
    def reset(self):
        """Reset buffers."""
        self.p_buf.fill(0)
        self.d1_buf.fill(0)
        self.d2_buf.fill(0)
        self.idx = 0
        self.count = 0


# ============================================================
# =============== ULTRA-FAST SERIAL PARSER ===================
# ============================================================

class UltraSerialParser:
    """Optimized serial parser with compiled regex."""
    
    def __init__(self, n_sensors: int):
        self.n = n_sensors
        # Compile regex once
        self.pattern = re.compile(
            rb'(\d+),\s*([0-9.]+),\s*[0-9.]+,\s*([0-9.]+),\s*[0-9.]+,\s*([0-9.]+),\s*[0-9.]+,\s*([0-9.]+),\s*[0-9.]+,\s*([0-9.]+),\s*[0-9.]+,\s*([0-9.]+)'
        )
        self.pressures = np.zeros(n_sensors, dtype=np.float64)
    
    def parse(self, line: bytes) -> Optional[Tuple[int, np.ndarray]]:
        """Parse line to timestamp and pressures."""
        try:
            match = self.pattern.match(line)
            if not match:
                return None
            g = match.groups()
            ts = int(g[0])
            # Direct assignment
            self.pressures[0] = float(g[1])
            self.pressures[1] = float(g[2])
            self.pressures[2] = float(g[3])
            self.pressures[3] = float(g[4])
            self.pressures[4] = float(g[5])
            self.pressures[5] = float(g[6])
            return ts, self.pressures
        except:
            return None


# ============================================================
# ============== ULTRA-FAST PREDICTOR ========================
# ============================================================

class UltraPredictor:
    """Maximum performance predictor."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Components
        self.preprocessor = UltraPreprocessor(config)
        self.window = UltraWindowBuilder(config)
        self.parser = UltraSerialParser(config.n_sensors)
        
        # Load model
        self.models = None
        self.scaler = None
        self._load_model()
        
        # Serial
        self.serial_port: Optional[serial.Serial] = None
        
        # Threading
        self.running = False
        self.lock = threading.Lock()
        
        # Stats
        self.samples_received = 0
        self.samples_processed = 0
        self.predictions_made = 0
        self.warmup_complete = False
        self.start_time = None
        
        # Latest data (lock-protected)
        self.latest_prediction = None
        self.latest_processed = None
        self.latest_raw = None
        
        # Offsets
        self.pred_offsets = {t: 0.0 for t in config.target_cols}
        self.baro_offsets = np.zeros(config.n_sensors, dtype=np.float64)
        
        # Logging
        self.csv_file = None
        if config.enable_csv_logging and config.csv_log_path:
            self._setup_csv()
        
        self.debug_file = None
        self._setup_debug()
        
        # Performance
        self.last_stats_time = 0
        self.stats_interval = 5.0
        
        # Pre-allocated prediction arrays
        self.features_2d = np.zeros((1, self.window.n_features), dtype=np.float64)
    
    def _load_model(self):
        """Load model and scaler."""
        mp = self.config.get_model_path()
        sp = self.config.get_scaler_path()
        
        if not mp.exists():
            raise FileNotFoundError(f"Model not found: {mp}")
        if not sp.exists():
            raise FileNotFoundError(f"Scaler not found: {sp}")
        
        print(f"Loading model: {mp}")
        with open(mp, 'rb') as f:
            self.models = pickle.load(f)
        
        print(f"Loading scaler: {sp}")
        with open(sp, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Loaded {len(self.models)} models")
    
    def _setup_csv(self):
        """Setup CSV logging."""
        self.csv_file = open(self.config.csv_log_path, 'w')
        h = ['timestamp', 'sample_idx']
        h.extend([f'b{i+1}' for i in range(6)])
        h.extend([f"{t}_pred" for t in self.config.target_cols])
        self.csv_file.write(','.join(h) + '\n')
    
    def _setup_debug(self):
        """Setup debug logging."""
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = Path(f"debug_barometers_{ts}.csv")
        
        self.debug_file = open(p, 'w')
        h = ['pc_time_ms', 'arduino_time_ms', 'sample_idx']
        h.extend([f'raw_b{i}' for i in range(1, 7)])
        h.extend([f'proc_b{i}' for i in range(1, 7)])
        h.extend(['x', 'y', 'fx', 'fy', 'fz', 'contact'])
        self.debug_file.write(','.join(h) + '\n')
        print(f"Debug logging: {p.absolute()}")
    
    def connect(self):
        """Connect to serial port."""
        port = self.config.port
        
        try:
            self.serial_port = serial.Serial(port, self.config.baud_rate,
                                            timeout=self.config.serial_timeout)
            print(f"Connected: {port} @ {self.config.baud_rate} baud")
        except:
            print(f"Failed to open {port}, auto-detecting...")
            port = self._find_port()
            self.serial_port = serial.Serial(port, self.config.baud_rate,
                                            timeout=self.config.serial_timeout)
            print(f"Connected: {port}")
        
        # Handshake
        time.sleep(0.5)
        self.serial_port.write(b'S')
        self.serial_port.flush()
        print("Handshake sent")
        
        # Clear
        time.sleep(1.0)
        self.serial_port.reset_input_buffer()
    
    def _find_port(self) -> str:
        """Auto-detect port."""
        ports = serial.tools.list_ports.comports()
        keywords = ['arduino', 'mkr', 'acm', 'usb']
        
        for p in ports:
            desc = (p.description + p.device).lower()
            if any(k in desc for k in keywords):
                return p.device
        
        if ports:
            return ports[0].device
        
        raise RuntimeError("No serial ports found")
    
    def _serial_thread(self):
        """Serial reading and processing thread."""
        line_buffer = bytearray()
        
        while self.running:
            try:
                if self.serial_port.in_waiting > 0:
                    chunk = self.serial_port.read(self.serial_port.in_waiting)
                    line_buffer.extend(chunk)
                    
                    while b'\n' in line_buffer:
                        idx = line_buffer.find(b'\n')
                        line = bytes(line_buffer[:idx].strip())
                        line_buffer = line_buffer[idx+1:]
                        
                        if line and not line.startswith(b'Time'):
                            parsed = self.parser.parse(line)
                            if parsed:
                                self._process(*parsed)
                else:
                    time.sleep(0.00001)  # 10 microseconds
            except Exception as e:
                if self.running:
                    print(f"Serial error: {e}")
                time.sleep(0.001)
    
    def _process(self, timestamp: int, pressures: np.ndarray):
        """Process sample - fully optimized."""
        self.samples_processed += 1
        
        # Warmup
        if self.samples_processed <= self.config.warmup_samples:
            self.preprocessor.process(pressures)
            if self.samples_processed == self.config.warmup_samples:
                print(f"Warmup complete ({self.config.warmup_samples} samples)")
                self.warmup_complete = True
            return
        
        # Preprocess
        processed, d1, d2 = self.preprocessor.process(pressures)
        
        # Window
        self.window.add(processed, d1, d2)
        
        # Predict
        if self.window.ready():
            features = self.window.get_features()
            prediction = self._predict(features)
            
            # Update (minimal lock)
            with self.lock:
                self.latest_prediction = prediction
                self.latest_processed = processed.copy()
                self.latest_raw = pressures.copy()
            
            self.predictions_made += 1
            
            # Log
            if self.samples_processed % self.config.debug_log_interval == 0:
                self._log_debug(timestamp, pressures, processed, prediction)
            
            if self.csv_file:
                self._log_csv(timestamp, processed, prediction)
        
        # Stats
        t = time.time()
        if t - self.last_stats_time >= self.stats_interval:
            self._print_stats()
            self.last_stats_time = t
    
    def _predict(self, features: np.ndarray) -> Dict[str, float]:
        """Make prediction."""
        self.features_2d[0] = features
        features_scaled = self.scaler.transform(self.features_2d)
        
        predictions = {}
        for model, target in zip(self.models, self.config.target_cols):
            pred = model.predict(features_scaled)[0]
            predictions[target] = float(pred) - self.pred_offsets[target]
        
        fz = predictions.get('fz', 0)
        predictions['contact'] = abs(fz) >= self.config.no_contact_threshold
        
        return predictions
    
    def _log_debug(self, ts: int, raw: np.ndarray, proc: np.ndarray,
                   pred: Dict):
        """Debug logging."""
        if not self.debug_file:
            return
        
        pc = int(time.time() * 1000)
        row = [str(pc), str(ts), str(self.samples_processed)]
        row.extend([f'{v:.4f}' for v in raw])
        row.extend([f'{v:.4f}' for v in proc])
        row.extend([
            f'{pred.get("x", 0):.4f}', f'{pred.get("y", 0):.4f}',
            f'{pred.get("fx", 0):.4f}', f'{pred.get("fy", 0):.4f}',
            f'{pred.get("fz", 0):.4f}',
            '1' if pred.get('contact', False) else '0'
        ])
        self.debug_file.write(','.join(row) + '\n')
    
    def _log_csv(self, ts: int, proc: np.ndarray, pred: Dict):
        """CSV logging."""
        row = [str(ts), str(self.samples_processed)]
        row.extend([f'{p:.4f}' for p in proc])
        row.extend([f'{pred[t]:.4f}' for t in self.config.target_cols])
        self.csv_file.write(','.join(row) + '\n')
    
    def _print_stats(self):
        """Print statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 1.0
        rate = self.predictions_made / max(0.1, elapsed)
        
        print(f"[Stats] Processed: {self.samples_processed}, "
              f"Predictions: {self.predictions_made}, "
              f"Rate: {rate:.1f} Hz")
    
    def start(self):
        """Start system."""
        self.running = True
        self.start_time = time.time()
        
        self.preprocessor.reset()
        self.window.reset()
        self.samples_processed = 0
        self.predictions_made = 0
        
        self.thread = threading.Thread(target=self._serial_thread, daemon=True)
        self.thread.start()
        
        mode = "Numba JIT" if NUMBA_AVAILABLE else "NumPy"
        print(f"Started ULTRA-OPTIMIZED mode ({mode})")
        if NUMBA_AVAILABLE:
            print("First predictions will be slow (JIT compilation)")
    
    def stop(self):
        """Stop system."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print(f"Stopped. Processed: {self.samples_processed}, Predictions: {self.predictions_made}")
    
    def disconnect(self):
        """Disconnect."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        if self.csv_file:
            self.csv_file.close()
        if self.debug_file:
            self.debug_file.close()
    
    def zero_all(self):
        """Zero everything."""
        with self.lock:
            if self.latest_processed is not None:
                self.baro_offsets = self.latest_processed.copy()
                print(">>> BAROMETERS ZEROED <<<")
            
            if self.latest_prediction is not None:
                for t in self.config.target_cols:
                    self.pred_offsets[t] += self.latest_prediction.get(t, 0)
                print(">>> FORCES ZEROED <<<")
    
    def calculate_baseline(self, duration_s: float = 5.0):
        """Calculate baseline."""
        print(f"\nBaseline calculation ({duration_s}s)...")
        print("Keep sensor unloaded!")
        
        pred_lists = {t: [] for t in self.config.target_cols}
        baro_list = []
        
        start = time.time()
        while time.time() - start < duration_s:
            with self.lock:
                if self.latest_prediction and self.latest_processed is not None:
                    for t in self.config.target_cols:
                        pred_lists[t].append(self.latest_prediction.get(t, 0) + 
                                           self.pred_offsets[t])
                    baro_list.append(self.latest_processed.copy())
            time.sleep(0.01)
        
        if baro_list:
            for t in self.config.target_cols:
                if pred_lists[t]:
                    self.pred_offsets[t] = np.mean(pred_lists[t])
            
            self.baro_offsets = np.mean(baro_list, axis=0)
            
            print(f"Baseline from {len(baro_list)} samples")
            print(f"Offsets: fx={self.pred_offsets['fx']:.3f}, "
                  f"fy={self.pred_offsets['fy']:.3f}, "
                  f"fz={self.pred_offsets['fz']:.3f}")
    
    def get_latest_prediction(self) -> Optional[Dict]:
        """Thread-safe get latest."""
        with self.lock:
            return self.latest_prediction.copy() if self.latest_prediction else None
    
    def get_zeroed_barometers(self) -> Optional[np.ndarray]:
        """Get zeroed barometers."""
        with self.lock:
            if self.latest_processed is None:
                return None
            return self.latest_processed - self.baro_offsets


# ============================================================
# ================= VISUALIZATION ============================
# ============================================================

class Visualizer:
    """Real-time visualization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.enabled = False
        
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            from collections import deque
            
            self.plt = plt
            self.FuncAnimation = FuncAnimation
            self.deque = deque
            self.enabled = True
        except ImportError:
            print("Matplotlib not available")
    
    def setup(self, predictor: UltraPredictor):
        """Setup visualization."""
        if not self.enabled:
            return
        
        self.predictor = predictor
        
        # Buffers
        max_pts = int(self.config.viz_buffer_seconds * 100)
        self.time_buf = self.deque(maxlen=max_pts)
        self.force_buf = {
            'fx': self.deque(maxlen=max_pts),
            'fy': self.deque(maxlen=max_pts),
            'fz': self.deque(maxlen=max_pts)
        }
        self.baro_buf = {f'b{i}': self.deque(maxlen=max_pts) for i in range(1, 7)}
        
        self.pos_x = []
        self.pos_y = []
        self.last_contact_time = None
        self.was_in_contact = False
        
        self.start_time = None
        self.frame_count = 0
        
        # Create figure
        self.fig, self.axes = self.plt.subplots(2, 3, figsize=(16, 9))
        self.fig.suptitle('ULTRA-OPTIMIZED Tactile Sensor (Press Z to Zero)', fontsize=14)
        
        self._setup_plots()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.plt.tight_layout()
    
    def _on_key(self, event):
        """Key handler."""
        if event.key.lower() == 'z':
            self.predictor.zero_all()
    
    def _setup_plots(self):
        """Setup all plots."""
        # FZ
        ax = self.axes[0, 0]
        self.fz_line, = ax.plot([], [], 'b-', lw=0.8)
        ax.set_xlim(0, self.config.viz_buffer_seconds)
        ax.set_ylim(-15, 0)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('FZ (N)')
        ax.set_title('FZ Force')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='r', ls='--', alpha=0.5)
        
        # FX
        ax = self.axes[0, 1]
        self.fx_line, = ax.plot([], [], 'b-', lw=0.8)
        ax.set_xlim(0, self.config.viz_buffer_seconds)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('FX (N)')
        ax.set_title('FX Force')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='r', ls='--', alpha=0.5)
        
        # FY
        ax = self.axes[0, 2]
        self.fy_line, = ax.plot([], [], 'b-', lw=0.8)
        ax.set_xlim(0, self.config.viz_buffer_seconds)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('FY (N)')
        ax.set_title('FY Force')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='r', ls='--', alpha=0.5)
        
        # Position
        ax = self.axes[1, 0]
        self.pos_scatter, = ax.plot([], [], 'b-', lw=0.8, alpha=0.7)
        self.pos_point, = ax.plot([], [], 'ro', ms=8)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Position')
        ax.grid(True, alpha=0.3)
        ax.axhline(8, color='gray', ls='--', alpha=0.4)
        ax.axhline(-8, color='gray', ls='--', alpha=0.4)
        ax.set_aspect('equal')
        
        # Barometers
        ax = self.axes[1, 1]
        self.baro_lines = []
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, c in enumerate(colors):
            line, = ax.plot([], [], color=c, lw=0.8, label=f'B{i+1}')
            self.baro_lines.append(line)
        ax.set_xlim(0, self.config.viz_buffer_seconds)
        ax.set_ylim(-1, 6)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pressure (hPa)')
        ax.set_title('Barometers')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Stats
        ax = self.axes[1, 2]
        ax.axis('off')
        self.stats_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                                  fontsize=11, va='top', family='monospace')
        
        # Add ZERO button
        from matplotlib.widgets import Button
        button_ax = self.fig.add_axes([0.85, 0.02, 0.12, 0.04])
        self.zero_button = Button(button_ax, 'ZERO (Z)', color='lightcoral', hovercolor='red')
        self.zero_button.on_clicked(lambda event: self.predictor.zero_all())
    
    def update(self, frame):
        """Update visualization."""
        self.frame_count += 1
        
        pred = self.predictor.get_latest_prediction()
        baro = self.predictor.get_zeroed_barometers()
        
        if pred is None:
            return (self.fz_line, self.fx_line, self.fy_line,
                   self.pos_scatter, self.pos_point, *self.baro_lines)
        
        if self.start_time is None:
            self.start_time = time.time()
        
        t = time.time() - self.start_time
        is_contact = pred.get('contact', False)
        
        # Position
        if is_contact:
            if not self.was_in_contact:
                self.pos_x = []
                self.pos_y = []
            self.pos_x.append(pred.get('x', 0))
            self.pos_y.append(pred.get('y', 0))
            self.last_contact_time = t
        else:
            if self.last_contact_time and (t - self.last_contact_time) > 10:
                self.pos_x = []
                self.pos_y = []
        
        self.was_in_contact = is_contact
        
        # Buffers
        self.time_buf.append(t)
        self.force_buf['fx'].append(pred.get('fx', 0))
        self.force_buf['fy'].append(pred.get('fy', 0))
        self.force_buf['fz'].append(pred.get('fz', 0))
        
        if baro is not None:
            for i in range(6):
                self.baro_buf[f'b{i+1}'].append(baro[i])
        
        # Update plots
        time_list = list(self.time_buf)
        if time_list:
            t_max = time_list[-1]
            t_min = max(0, t_max - self.config.viz_buffer_seconds)
            time_shifted = [t - t_min for t in time_list]
            
            # Forces
            self.fz_line.set_data(time_shifted, list(self.force_buf['fz']))
            self.fx_line.set_data(time_shifted, list(self.force_buf['fx']))
            self.fy_line.set_data(time_shifted, list(self.force_buf['fy']))
            
            # Position
            if self.pos_x:
                self.pos_scatter.set_data(self.pos_x, self.pos_y)
                if is_contact:
                    self.pos_point.set_data([self.pos_x[-1]], [self.pos_y[-1]])
                else:
                    self.pos_point.set_data([], [])
            
            # Barometers
            for i, line in enumerate(self.baro_lines):
                data = list(self.baro_buf[f'b{i+1}'])
                if data:
                    line.set_data(time_shifted[-len(data):], data)
        
        # Stats (every 10 frames)
        if self.frame_count % 10 == 0:
            contact_str = "IN CONTACT" if is_contact else "NO CONTACT"
            rate = self.predictor.predictions_made / max(1, t)
            
            mode = "Numba" if NUMBA_AVAILABLE else "NumPy"
            
            stats = f"""Processed: {self.predictor.samples_processed}
Predictions: {self.predictor.predictions_made}
Rate: ~{rate:.0f} Hz ({mode})

Position:
  X: {pred.get('x', 0):>7.2f} mm
  Y: {pred.get('y', 0):>7.2f} mm

Forces:
  Fx: {pred.get('fx', 0):>7.3f} N
  Fy: {pred.get('fy', 0):>7.3f} N
  Fz: {pred.get('fz', 0):>7.3f} N

Status: {contact_str}

Press 'Z' to zero"""
            self.stats_text.set_text(stats)
        
        return (self.fz_line, self.fx_line, self.fy_line,
               self.pos_scatter, self.pos_point, *self.baro_lines)
    
    def run(self):
        """Run visualization."""
        if not self.enabled:
            return
        
        self.anim = self.FuncAnimation(self.fig, self.update,
                                      interval=self.config.viz_update_interval_ms,
                                      blit=True, cache_frame_data=False)
        self.plt.show()


# ============================================================
# ========================= MAIN =============================
# ============================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='ULTRA-OPTIMIZED real-time tactile sensor')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--model-dir', type=str, default=None)
    parser.add_argument('--version', type=str, default='5.020')
    
    args = parser.parse_args()
    
    config = Config()
    config.port = args.port
    config.enable_plot = args.plot
    config.sensor_version = args.version
    
    if args.log:
        config.enable_csv_logging = True
        config.csv_log_path = Path(args.log)
    
    if args.model_dir:
        config.model_dir = Path(args.model_dir)
    
    print("=" * 60)
    print("ULTRA-OPTIMIZED Real-Time Tactile Sensor")
    print("=" * 60)
    print(f"Numba JIT: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")
    if not NUMBA_AVAILABLE:
        print("  Install numba for 5-10x speedup: pip install numba")
    print(f"EMA Alpha: {config.ema_alpha}")
    print("=" * 60)
    
    try:
        predictor = UltraPredictor(config)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    
    try:
        predictor.connect()
    except Exception as e:
        print(f"\nError: {e}")
        return
    
    predictor.start()
    
    try:
        if config.enable_plot:
            print("\n" + "=" * 60)
            print("BASELINE CALIBRATION")
            print("=" * 60)
            print("Keep sensor UNLOADED for 5 seconds...\n")
            
            while not predictor.warmup_complete:
                time.sleep(0.1)
            
            predictor.calculate_baseline(duration_s=5.0)
            
            print("\n" + "=" * 60)
            print("CALIBRATION COMPLETE")
            print("=" * 60 + "\n")
            
            viz = Visualizer(config)
            viz.setup(predictor)
            viz.run()
        else:
            print("\nStreaming (Ctrl+C to stop)...\n")
            while predictor.running:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        predictor.stop()
        predictor.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()