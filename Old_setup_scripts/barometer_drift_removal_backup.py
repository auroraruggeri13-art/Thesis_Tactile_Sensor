#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Barometer Drift Removal - Real-time Compatible

Two methods implemented:
1. Simple Plateau Step Removal: Tracks sliding window median
2. EMA High-Pass Filter: Removes slow drift regardless of cause

All methods are real-time compatible (causal, sample-by-sample processing).

Author: Enhanced by Claude
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================== SIMPLE PLATEAU STEP REMOVAL ====================

# ==================== SIMPLE PLATEAU STEP REMOVAL ====================

def remove_plateau_steps(df, baro_cols=None, 
                        window=200, 
                        step_threshold=10.0):
    """
    Ultra-simple plateau step removal - just compare sliding window medians.
    
    Algorithm:
    1. Calculate median in sliding windows
    2. When median changes by more than threshold, it's a step
    3. Remove the step from that point forward
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with barometer columns
    baro_cols : list, optional
        Barometer columns (default: b1..b6)
    window : int
        Window size for median calculation (default: 200 samples)
        Larger = more robust to noise
    step_threshold : float
        Minimum change in median to detect step [hPa]
        For your data: use 5-15 hPa
    
    Returns:
    --------
    df : pd.DataFrame
        Corrected data
    step_info : dict
        Info about removed steps
    """
    df = df.copy()
    
    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    
    print(f"\n=== Simple Plateau Step Removal ===")
    print(f"Window size: {window} samples")
    print(f"Step threshold: {step_threshold} hPa")
    
    step_info = {}
    
    for col in baro_cols:
        if col not in df.columns:
            continue
        
        P = df[col].to_numpy().copy()
        n = len(P)
        
        if not np.isfinite(P).any():
            print(f"  {col}: no valid data")
            continue
        
        # Calculate rolling median
        P_series = pd.Series(P)
        median = P_series.rolling(window=window, center=False, min_periods=window//2).median()
        
        # Find where median changes significantly
        median_diff = median.diff().abs()
        step_locations = np.where(median_diff > step_threshold)[0]
        
        # Remove steps
        cumulative_offset = 0.0
        steps_removed = []
        
        for idx in step_locations:
            if idx < window:
                continue  # Skip early points
            
            # Calculate step size as difference in medians
            step_size = median.iloc[idx] - median.iloc[idx-1]
            
            # Apply correction from this point forward
            P[idx:] -= step_size
            cumulative_offset += step_size
            
            steps_removed.append({
                'index': idx,
                'time': df['time'].iloc[idx] if 'time' in df.columns else idx,
                'size': step_size
            })
            
            print(f"    Step at sample {idx}: {step_size:+.2f} hPa")
        
        df[col] = P
        
        step_info[col] = {
            'n_steps': len(steps_removed),
            'total_offset': cumulative_offset,
            'steps': steps_removed
        }
        
        if steps_removed:
            print(f"  {col}: removed {len(steps_removed)} steps, total offset = {cumulative_offset:+.2f} hPa")
        else:
            print(f"  {col}: no steps detected")
    
    return df, step_info


def remove_plateau_steps_robust(df, baro_cols=None,
                                window=300,
                                percentile=50,
                                step_threshold=8.0):
    """
    Simplest possible method: track percentile value in sliding window.
    
    When the percentile (median) jumps, that's a plateau step.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with barometer columns
    baro_cols : list, optional
        Barometer columns (default: b1..b6)
    window : int
        Samples to look at (300 = 3 seconds at 100Hz)
    percentile : float
        Which percentile to track (50 = median, very robust)
    step_threshold : float
        Minimum jump size to correct [hPa]
    
    Returns:
    --------
    df : pd.DataFrame
        Corrected data
    step_info : dict
        Info about removed steps
    """
    df = df.copy()
    
    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    
    print(f"\n=== Percentile-Based Step Removal ===")
    print(f"Window: {window}, Percentile: {percentile}, Threshold: {step_threshold} hPa")
    
    step_info = {}
    
    for col in baro_cols:
        if col not in df.columns:
            continue
        
        P = df[col].to_numpy().copy()
        
        if not np.isfinite(P).any():
            continue
        
        # Track percentile in sliding window
        baseline = pd.Series(P).rolling(
            window=window, 
            center=False, 
            min_periods=window//3
        ).quantile(percentile/100.0).values
        
        # Find jumps in baseline
        baseline_change = np.diff(baseline)
        baseline_change = np.insert(baseline_change, 0, 0)
        
        # Accumulate offsets
        cumulative_offset = 0.0
        steps = []
        
        for i in range(window, len(P)):
            if abs(baseline_change[i]) > step_threshold:
                step_size = baseline_change[i]
                cumulative_offset += step_size
                P[i:] -= step_size
                
                steps.append({'index': i, 'size': step_size})
                print(f"    {col}: step at {i}, size {step_size:+.2f} hPa")
        
        df[col] = P
        step_info[col] = {'n_steps': len(steps), 'total_offset': cumulative_offset, 'steps': steps}
        
        if steps:
            print(f"  {col}: {len(steps)} steps, total {cumulative_offset:+.2f} hPa")
    
    return df, step_info
    """
    Detect and remove plateau-based step changes in barometer data.
    
    This method detects when the sensor reading stabilizes at different pressure
    levels (plateaus) and removes the offset between them. Works for gradual
    transitions between plateaus, not just sudden jumps.
    
    Algorithm:
    1. Detect stable plateaus (low variance regions)
    2. Compare consecutive plateau levels
    3. Remove cumulative offset from subsequent data
    4. Handle gradual transitions between plateaus
    
    Real-time compatible: uses only past data (causal).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with barometer columns and 'time' column
    baro_cols : list of str, optional
        List of barometer columns. If None, auto-detect b1..b6
    stability_window : int
        Number of samples to check for stability (typical: 50-200)
        Larger = more robust to noise, slower to detect
    stability_threshold : float
        Maximum std dev [hPa] for data to be considered "stable"
        Typical: 0.3-1.0 hPa depending on sensor noise
    min_plateau_samples : int
        Minimum samples required to confirm a plateau
        Prevents brief stable moments from being treated as plateaus
    min_step_size : float
        Minimum step size [hPa] to correct
        Steps smaller than this are ignored as noise
    transition_buffer : int
        Samples to skip after detecting a step (transition period)
        Prevents detection during gradual transitions
    
    Returns:
    --------
    df : pd.DataFrame
        Modified dataframe with steps removed
    step_info : dict
        Dictionary with step detection info for each barometer
    """
    df = df.copy()
    
    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    
    if not baro_cols:
        print("remove_plateau_steps: No barometer columns found.")
        return df, {}
    
    print(f"\n=== Plateau-Based Step Removal ===")
    print(f"Stability window: {stability_window} samples")
    print(f"Stability threshold: {stability_threshold} hPa (std dev)")
    print(f"Min plateau samples: {min_plateau_samples}")
    print(f"Min step size: {min_step_size} hPa")
    print(f"Transition buffer: {transition_buffer} samples")
    
    step_info = {}
    
    for col in baro_cols:
        if col not in df.columns:
            continue
        
        P = df[col].to_numpy()
        n = len(P)
        
        valid_mask = np.isfinite(P)
        if not valid_mask.any():
            print(f"  {col}: no valid data, skipping")
            continue
        
        # Initialize tracking variables
        P_corrected = P.copy()
        cumulative_offset = 0.0
        plateaus_detected = []
        
        current_plateau = None  # {'start_idx', 'mean', 'samples'}
        in_transition = 0  # Countdown for transition buffer
        
        # Rolling window for stability check
        window = deque(maxlen=stability_window)
        
        for i in range(n):
            if not valid_mask[i]:
                continue
            
            # Add to rolling window
            window.append(P[i])
            
            # Skip if in transition period
            if in_transition > 0:
                in_transition -= 1
                continue
            
            # Need full window to assess stability
            if len(window) < stability_window:
                continue
            
            # Check if current window is stable
            window_array = np.array(window)
            window_std = np.std(window_array)
            window_mean = np.mean(window_array)
            
            is_stable = window_std < stability_threshold
            
            if is_stable:
                # We're in a stable region
                if current_plateau is None:
                    # Start of new plateau
                    current_plateau = {
                        'start_idx': i - stability_window + 1,
                        'mean': window_mean,
                        'samples': stability_window,
                        'std': window_std
                    }
                else:
                    # Continue current plateau
                    # Update plateau mean incrementally
                    current_plateau['samples'] += 1
                    current_plateau['mean'] = (
                        (current_plateau['mean'] * (current_plateau['samples'] - 1) + P[i]) 
                        / current_plateau['samples']
                    )
                    current_plateau['std'] = window_std
            
            else:
                # Not stable - check if we were on a plateau
                if current_plateau is not None:
                    # End of plateau
                    if current_plateau['samples'] >= min_plateau_samples:
                        # This was a valid plateau
                        
                        # Check if there's a step from previous plateau
                        if len(plateaus_detected) > 0:
                            prev_plateau = plateaus_detected[-1]
                            step_size = current_plateau['mean'] - prev_plateau['mean']
                            
                            if abs(step_size) >= min_step_size:
                                # Significant step detected!
                                cumulative_offset += step_size
                                
                                # Apply correction from this point forward
                                P_corrected[i:] -= step_size
                                
                                print(f"    Step detected at sample {i}:")
                                print(f"      From plateau: {prev_plateau['mean']:.2f} hPa "
                                      f"(samples: {prev_plateau['samples']})")
                                print(f"      To plateau: {current_plateau['mean']:.2f} hPa "
                                      f"(samples: {current_plateau['samples']})")
                                print(f"      Step size: {step_size:+.2f} hPa")
                                print(f"      Cumulative offset: {cumulative_offset:+.2f} hPa")
                                
                                # Enter transition buffer period
                                in_transition = transition_buffer
                        
                        # Store this plateau
                        plateaus_detected.append(current_plateau)
                    
                    # Reset plateau tracking
                    current_plateau = None
        
        # Handle plateau at end of data
        if current_plateau is not None and current_plateau['samples'] >= min_plateau_samples:
            if len(plateaus_detected) > 0:
                prev_plateau = plateaus_detected[-1]
                step_size = current_plateau['mean'] - prev_plateau['mean']
                if abs(step_size) >= min_step_size:
                    cumulative_offset += step_size
                    print(f"    Final step detected:")
                    print(f"      Step size: {step_size:+.2f} hPa")
            plateaus_detected.append(current_plateau)
        
        df[col] = P_corrected
        
        step_info[col] = {
            'n_plateaus': len(plateaus_detected),
            'n_steps': len(plateaus_detected) - 1 if len(plateaus_detected) > 1 else 0,
            'total_offset': cumulative_offset,
            'plateaus': plateaus_detected
        }
        
        if len(plateaus_detected) > 1:
            print(f"  {col}: detected {len(plateaus_detected)} plateaus, "
                  f"removed {len(plateaus_detected)-1} steps, "
                  f"total offset = {cumulative_offset:+.2f} hPa")
        else:
            print(f"  {col}: detected {len(plateaus_detected)} plateau(s), no steps removed")
    
    return df, step_info


def remove_plateau_steps_robust(df, baro_cols=None,
                                stability_window=150,
                                stability_threshold=0.8,
                                min_plateau_samples=100,
                                min_step_size=3.0,
                                use_median=True):
    """
    Robust plateau detection with stricter criteria.
    
    Differences from standard version:
    - Uses median instead of mean for robustness
    - Requires longer stable periods
    - More conservative step detection
    - Better handling of noisy data
    
    Parameters are similar to remove_plateau_steps() but with different defaults
    tuned for noisy data.
    """
    df = df.copy()
    
    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    
    if not baro_cols:
        print("remove_plateau_steps_robust: No barometer columns found.")
        return df, {}
    
    print(f"\n=== Robust Plateau-Based Step Removal ===")
    print(f"Stability window: {stability_window} samples")
    print(f"Stability threshold: {stability_threshold} hPa (std dev)")
    print(f"Min plateau samples: {min_plateau_samples}")
    print(f"Min step size: {min_step_size} hPa")
    print(f"Using median: {use_median}")
    
    step_info = {}
    
    for col in baro_cols:
        if col not in df.columns:
            continue
        
        P = df[col].to_numpy()
        n = len(P)
        
        valid_mask = np.isfinite(P)
        if not valid_mask.any():
            print(f"  {col}: no valid data, skipping")
            continue
        
        # Calculate rolling statistics
        P_series = pd.Series(P)
        rolling_std = P_series.rolling(window=stability_window, center=False).std()
        if use_median:
            rolling_mean = P_series.rolling(window=stability_window, center=False).median()
        else:
            rolling_mean = P_series.rolling(window=stability_window, center=False).mean()
        
        # Identify stable regions
        stable_mask = (rolling_std < stability_threshold).fillna(False).to_numpy()
        
        # Find contiguous stable regions (plateaus)
        plateaus = []
        in_plateau = False
        plateau_start = None
        
        for i in range(n):
            if stable_mask[i] and not in_plateau:
                # Start of plateau
                in_plateau = True
                plateau_start = i
            elif not stable_mask[i] and in_plateau:
                # End of plateau
                plateau_length = i - plateau_start
                if plateau_length >= min_plateau_samples:
                    plateau_mean = np.median(P[plateau_start:i]) if use_median else np.mean(P[plateau_start:i])
                    plateaus.append({
                        'start': plateau_start,
                        'end': i,
                        'mean': plateau_mean,
                        'length': plateau_length
                    })
                in_plateau = False
        
        # Handle plateau at end
        if in_plateau:
            plateau_length = n - plateau_start
            if plateau_length >= min_plateau_samples:
                plateau_mean = np.median(P[plateau_start:]) if use_median else np.mean(P[plateau_start:])
                plateaus.append({
                    'start': plateau_start,
                    'end': n,
                    'mean': plateau_mean,
                    'length': plateau_length
                })
        
        # Remove steps between plateaus
        P_corrected = P.copy()
        cumulative_offset = 0.0
        steps_removed = []
        
        for i in range(1, len(plateaus)):
            prev_plateau = plateaus[i-1]
            curr_plateau = plateaus[i]
            
            step_size = curr_plateau['mean'] - prev_plateau['mean']
            
            if abs(step_size) >= min_step_size:
                # Remove this step from current plateau onward
                cumulative_offset += step_size
                P_corrected[curr_plateau['start']:] -= step_size
                
                steps_removed.append({
                    'position': curr_plateau['start'],
                    'size': step_size,
                    'from_mean': prev_plateau['mean'],
                    'to_mean': curr_plateau['mean']
                })
                
                print(f"    Step at sample {curr_plateau['start']}: {step_size:+.2f} hPa "
                      f"({prev_plateau['mean']:.2f} → {curr_plateau['mean']:.2f})")
        
        df[col] = P_corrected
        
        step_info[col] = {
            'n_plateaus': len(plateaus),
            'n_steps': len(steps_removed),
            'total_offset': cumulative_offset,
            'plateaus': plateaus,
            'steps': steps_removed
        }
        
        if steps_removed:
            print(f"  {col}: detected {len(plateaus)} plateaus, "
                  f"removed {len(steps_removed)} steps, "
                  f"total offset = {cumulative_offset:+.2f} hPa")
        else:
            print(f"  {col}: detected {len(plateaus)} plateau(s), no significant steps")
    
    return df, step_info


# ==================== REAL-TIME PLATEAU DETECTOR CLASS ====================

class RealtimePlateauDetector:
    """
    Real-time plateau-based step removal.
    
    Processes samples one at a time and corrects for plateau steps on-the-fly.
    Maintains internal state to track current plateau and cumulative offset.
    
    Example usage:
    -------------
    detector = RealtimePlateauDetector(
        stability_window=100,
        stability_threshold=0.5,
        min_step_size=5.0
    )
    
    for pressure in data_stream:
        corrected = detector.process_sample(pressure)
        # Use corrected value...
    """
    
    def __init__(self, stability_window=100, stability_threshold=0.5,
                 min_plateau_samples=50, min_step_size=5.0,
                 transition_buffer=20):
        """
        Initialize real-time plateau detector.
        
        Parameters match remove_plateau_steps() function.
        """
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold
        self.min_plateau_samples = min_plateau_samples
        self.min_step_size = min_step_size
        self.transition_buffer = transition_buffer
        
        # State variables
        self.window = deque(maxlen=stability_window)
        self.cumulative_offset = 0.0
        self.current_plateau = None
        self.last_plateau = None
        self.in_transition = 0
        self.sample_count = 0
    
    def process_sample(self, pressure):
        """
        Process a single pressure sample.
        
        Parameters:
        -----------
        pressure : float
            Current pressure reading [hPa]
        
        Returns:
        --------
        corrected_pressure : float
            Step-corrected pressure
        """
        if not np.isfinite(pressure):
            return np.nan
        
        self.sample_count += 1
        
        # Apply current cumulative offset
        corrected = pressure - self.cumulative_offset
        
        # Add to rolling window
        self.window.append(pressure)
        
        # Skip if in transition period
        if self.in_transition > 0:
            self.in_transition -= 1
            return corrected
        
        # Need full window to assess stability
        if len(self.window) < self.stability_window:
            return corrected
        
        # Check stability
        window_array = np.array(self.window)
        window_std = np.std(window_array)
        window_mean = np.mean(window_array)
        
        is_stable = window_std < self.stability_threshold
        
        if is_stable:
            # We're in a stable region
            if self.current_plateau is None:
                # Start of new plateau
                self.current_plateau = {
                    'start_sample': self.sample_count - self.stability_window + 1,
                    'mean': window_mean,
                    'samples': self.stability_window,
                    'std': window_std
                }
            else:
                # Continue current plateau - update mean
                self.current_plateau['samples'] += 1
                n = self.current_plateau['samples']
                self.current_plateau['mean'] = (
                    (self.current_plateau['mean'] * (n - 1) + pressure) / n
                )
                self.current_plateau['std'] = window_std
        
        else:
            # Not stable - check if we were on a plateau
            if self.current_plateau is not None:
                # End of plateau
                if self.current_plateau['samples'] >= self.min_plateau_samples:
                    # Valid plateau
                    
                    # Check for step from previous plateau
                    if self.last_plateau is not None:
                        step_size = self.current_plateau['mean'] - self.last_plateau['mean']
                        
                        if abs(step_size) >= self.min_step_size:
                            # Significant step detected!
                            self.cumulative_offset += step_size
                            corrected = pressure - self.cumulative_offset
                            
                            # Enter transition buffer
                            self.in_transition = self.transition_buffer
                    
                    # Update last plateau
                    self.last_plateau = self.current_plateau
                
                # Reset current plateau
                self.current_plateau = None
        
        return corrected
    
    def reset(self):
        """Reset detector state."""
        self.window.clear()
        self.cumulative_offset = 0.0
        self.current_plateau = None
        self.last_plateau = None
        self.in_transition = 0
        self.sample_count = 0
    
    def get_state(self):
        """Get current detector state (for diagnostics)."""
        return {
            'cumulative_offset': self.cumulative_offset,
            'current_plateau': self.current_plateau,
            'last_plateau': self.last_plateau,
            'in_transition': self.in_transition,
            'sample_count': self.sample_count
        }


# ==================== METHOD A: EMA HIGH-PASS FILTER ====================

def remove_drift_ema(df, baro_cols=None, alpha=0.001, alpha_override=None, zero_at_start=True):
    """
    Remove drift using Exponential Moving Average high-pass filter.
    
    This removes slow drift (thermal, electronic, etc.) while preserving
    fast pressure changes. Works identically in real-time and post-processing.
    
    Algorithm (real-time compatible):
    - EMA tracks slow drift: EMA[n] = α × P[n] + (1-α) × EMA[n-1]
    - High-pass output: P_corrected[n] = P[n] - EMA[n]
    - Optionally zero at start by subtracting initial value
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with barometer columns
    baro_cols : list of str, optional
        List of barometer column names. If None, auto-detect b1..b6
    alpha : float
        EMA smoothing factor (0 < alpha < 1)
        - Smaller alpha = slower tracking, removes more drift
        - Time constant τ ≈ 1/alpha samples
        - For 100Hz: alpha=0.001 gives τ≈10s, alpha=0.0001 gives τ≈100s
    alpha_override : dict, optional
        Dictionary to override alpha for specific sensors
        Example: {'b6': 0.005, 'b3': 0.002}
    zero_at_start : bool
        If True, subtract the initial value to start at zero
    
    Returns:
    --------
    df : pd.DataFrame
        Modified dataframe with drift-removed barometers
    ema_trends : dict
        Dictionary of EMA trends for each barometer (for diagnostics)
    """
    df = df.copy()
    
    # Auto-detect barometer columns if not provided
    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    
    if not baro_cols:
        print("remove_drift_ema: No barometer columns found.")
        return df, {}
    
    # Initialize alpha_override if not provided
    if alpha_override is None:
        alpha_override = {}
    
    print(f"\n=== EMA Drift Removal ===")
    print(f"Default alpha = {alpha:.6f} (time constant ≈ {1/alpha:.0f} samples)")
    if alpha_override:
        print(f"Sensor-specific overrides: {alpha_override}")
    print(f"Zero at start: {zero_at_start}")
    
    ema_trends = {}
    
    for col in baro_cols:
        if col not in df.columns:
            continue
        
        # Get alpha for this sensor (use override if specified)
        alpha_use = alpha_override.get(col, alpha)
        
        P = df[col].to_numpy()
        n = len(P)
        
        # Check for valid data
        valid_mask = np.isfinite(P)
        if not valid_mask.any():
            print(f"  {col}: no valid data, skipping")
            continue
        
        # Initialize EMA with first valid value
        first_valid_idx = np.where(valid_mask)[0][0]
        initial_value = P[first_valid_idx]
        
        # Compute EMA (causal, real-time compatible)
        ema = np.zeros(n)
        ema[first_valid_idx] = initial_value
        
        for i in range(first_valid_idx + 1, n):
            if np.isfinite(P[i]):
                ema[i] = alpha_use * P[i] + (1 - alpha_use) * ema[i-1]
            else:
                ema[i] = ema[i-1]  # Hold last value for NaN
        
        # Store trend for diagnostics
        ema_trends[col] = ema.copy()
        
        # Apply high-pass filter: remove EMA trend
        P_corrected = P - ema
        
        # Optionally zero at start
        if zero_at_start:
            P_corrected = P_corrected - P_corrected[first_valid_idx]
        
        # Calculate statistics
        drift_removed = ema[-1] - ema[first_valid_idx]
        
        df[col] = P_corrected
        
        alpha_msg = f" (α={alpha_use:.6f})" if alpha_use != alpha else ""
        print(f"  {col}: drift removed = {drift_removed:+.3f} hPa{alpha_msg}")
    
    return df, ema_trends


# ================ METHOD B: LINEAR TEMPERATURE COMPENSATION ================

def remove_drift_temperature_linear(df, baro_cols=None, temp_cols=None, 
                                    skip_initial_fraction=0.05, 
                                    zero_at_start=True):
    """
    Remove drift using linear temperature compensation.
    
    Fits a linear relationship P = a + b*T for each sensor, then corrects
    pressure by removing temperature-dependent component.
    
    Algorithm (can be made real-time with calibration):
    - Fit: P = a + b*T using stable data (after warmup)
    - Reference temperature: T_ref = first stable temperature
    - Correction: P_corrected = P - b*(T - T_ref)
    
    This is more physically meaningful than polynomial and more robust.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with barometer (b1..b6) and temperature (t1..t6) columns
    baro_cols : list of str, optional
        List of barometer column names. If None, use b1..b6
    temp_cols : list of str, optional
        List of temperature column names. If None, use t1..t6
    skip_initial_fraction : float
        Fraction of data to skip when fitting (to avoid warmup transients)
    zero_at_start : bool
        If True, subtract the initial corrected value to start at zero
    
    Returns:
    --------
    df : pd.DataFrame
        Modified dataframe with temperature-corrected barometers
    coefficients : dict
        Dictionary with 'slope' and 'intercept' for each barometer
    """
    df = df.copy()
    
    # Auto-detect columns
    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    if temp_cols is None:
        temp_cols = [f"t{i}" for i in range(1, 7) if f"t{i}" in df.columns]
    
    if not baro_cols:
        print("remove_drift_temperature_linear: No barometer columns found.")
        return df, {}
    
    if len(temp_cols) < len(baro_cols):
        print(f"remove_drift_temperature_linear: Missing temperature columns.")
        print(f"  Found {len(temp_cols)} temp columns for {len(baro_cols)} barometers")
        return df, {}
    
    print(f"\n=== Linear Temperature Compensation ===")
    print(f"Skip initial: {skip_initial_fraction*100:.1f}%")
    print(f"Zero at start: {zero_at_start}")
    
    n_total = len(df)
    n_skip = int(n_total * skip_initial_fraction)
    
    coefficients = {}
    
    for i, (bcol, tcol) in enumerate(zip(baro_cols, temp_cols)):
        if bcol not in df.columns or tcol not in df.columns:
            continue
        
        P = df[bcol].to_numpy()
        T = df[tcol].to_numpy()
        
        # Use stable data for fitting (skip warmup)
        P_stable = P[n_skip:]
        T_stable = T[n_skip:]
        
        # Filter valid data
        valid_mask = np.isfinite(P_stable) & np.isfinite(T_stable)
        
        if valid_mask.sum() < 10:
            print(f"  {bcol}: insufficient valid data, skipping")
            continue
        
        P_fit = P_stable[valid_mask]
        T_fit = T_stable[valid_mask]
        
        # Check temperature variation
        T_range = T_fit.max() - T_fit.min()
        
        if T_range < 0.1:  # Less than 0.1°C variation
            print(f"  {bcol}: temperature too stable ({T_range:.3f}°C), skipping")
            continue
        
        # Fit linear model: P = a + b*T
        # Using numpy polyfit (degree 1 = linear)
        coeffs = np.polyfit(T_fit, P_fit, deg=1)
        slope = coeffs[0]  # dP/dT [hPa/°C]
        intercept = coeffs[1]  # P at T=0
        
        # Reference temperature (first stable value)
        T_ref = T_stable[valid_mask][0]
        
        # Apply correction to ALL data (including warmup)
        # P_corrected = P - slope*(T - T_ref)
        # This removes the temperature-dependent drift component
        correction = slope * (T - T_ref)
        P_corrected = P - correction
        
        # Optionally zero at start
        if zero_at_start:
            first_valid = np.where(np.isfinite(P_corrected))[0]
            if len(first_valid) > 0:
                P_corrected = P_corrected - P_corrected[first_valid[0]]
        
        # Store results
        df[bcol] = P_corrected
        coefficients[bcol] = {
            'slope': slope,
            'intercept': intercept,
            'T_ref': T_ref,
            'T_range': T_range,
            'max_correction': np.abs(correction).max()
        }
        
        print(f"  {bcol}: slope = {slope:+.3f} hPa/°C, T_range = {T_range:.2f}°C, "
              f"max_correction = {np.abs(correction).max():.2f} hPa")
    
    return df, coefficients


# ==================== FIXED ZEROING FUNCTION ====================

def zero_barometers_fixed(df, baro_cols=None, reference_index=0):
    """
    Zero barometers at a specific index (default: first valid sample).
    
    This replaces the old zero_barometers() function which had issues
    after warmup removal.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with barometer columns
    baro_cols : list of str, optional
        List of barometer columns. If None, auto-detect b1..b6
    reference_index : int or 'first_valid'
        Index to use as zero reference
        - 0: use first row (default)
        - 'first_valid': use first row with valid data
        - N: use row N
    
    Returns:
    --------
    df : pd.DataFrame
        Modified dataframe with zeroed barometers
    """
    df = df.copy()
    
    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    
    if not baro_cols:
        print("zero_barometers_fixed: No barometer columns found.")
        return df
    
    print(f"\n=== Zeroing Barometers ===")
    
    for col in baro_cols:
        if col not in df.columns:
            continue
        
        # Find reference value
        if reference_index == 'first_valid':
            valid_idx = np.where(np.isfinite(df[col]))[0]
            if len(valid_idx) == 0:
                print(f"  {col}: no valid data, skipping")
                continue
            ref_idx = valid_idx[0]
        else:
            ref_idx = reference_index
        
        if ref_idx >= len(df):
            print(f"  {col}: reference index {ref_idx} out of range, using 0")
            ref_idx = 0
        
        ref_value = df[col].iloc[ref_idx]
        
        if not np.isfinite(ref_value):
            print(f"  {col}: reference value is NaN, skipping")
            continue
        
        df[col] = df[col] - ref_value
        
        print(f"  {col}: zeroed at index {ref_idx}, value = {ref_value:.3f} hPa")
    
    return df


# ==================== DIAGNOSTIC PLOTTING ====================

def plot_step_removal_comparison(df_original, df_corrected, step_info, 
                                 baro_col='b1', save_path=None):
    """
    Plot before/after comparison for step removal on a single barometer.
    
    Shows:
    - Original data with detected plateaus highlighted
    - Corrected data
    - Step locations marked
    """
    if baro_col not in df_original.columns or baro_col not in df_corrected.columns:
        print(f"Column {baro_col} not found in data")
        return None
    
    if baro_col not in step_info or 'plateaus' not in step_info[baro_col]:
        print(f"No step info available for {baro_col}")
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    t = df_original['time'].to_numpy()
    P_orig = df_original[baro_col].to_numpy()
    P_corr = df_corrected[baro_col].to_numpy()
    
    plateaus = step_info[baro_col]['plateaus']
    
    # Top plot: Original with plateaus highlighted
    ax1.plot(t, P_orig, 'b-', linewidth=1.0, alpha=0.6, label='Original')
    
    # Highlight plateaus
    colors = plt.cm.Set3(np.linspace(0, 1, len(plateaus)))
    for i, plateau in enumerate(plateaus):
        start_idx = plateau.get('start', plateau.get('start_idx', 0))
        end_idx = plateau.get('end', len(t))
        ax1.axvspan(t[start_idx], t[end_idx], alpha=0.3, color=colors[i],
                   label=f"Plateau {i+1}: {plateau['mean']:.1f} hPa")
    
    ax1.set_ylabel(f"{baro_col} [hPa]")
    ax1.set_title("Original Data with Detected Plateaus")
    ax1.grid(alpha=0.3)
    ax1.legend(loc='best', fontsize=8, ncol=2)
    
    # Bottom plot: Corrected
    ax2.plot(t, P_corr, 'g-', linewidth=1.0, label='Corrected')
    
    # Mark step removal points
    if 'steps' in step_info[baro_col]:
        for step in step_info[baro_col]['steps']:
            step_idx = step.get('position', step.get('index', 0))
            if step_idx < len(t):
                ax2.axvline(t[step_idx], color='r', linestyle='--', alpha=0.5,
                           label=f"Step: {step['size']:+.1f} hPa")
    
    ax2.set_ylabel(f"{baro_col} [hPa]")
    ax2.set_xlabel("Time [s]")
    ax2.set_title("Corrected Data (Steps Removed)")
    ax2.grid(alpha=0.3)
    # Remove duplicate labels in legend
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)
    
    total_offset = step_info[baro_col].get('total_offset', 0)
    fig.suptitle(f"Plateau-Based Step Removal: {baro_col} "
                f"(Total offset removed: {total_offset:+.2f} hPa)",
                fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved step removal plot: {save_path}")
    
    return fig


def plot_drift_comparison(df_original, df_ema, df_temp, 
                          ema_trends=None, temp_coeffs=None,
                          save_path=None):
    """
    Plot comparison of original vs drift-removed data.
    
    Shows 6 barometers in subplots with:
    - Original data (gray)
    - EMA-corrected (blue)
    - Temperature-corrected (orange)
    - EMA trend (dashed, if available)
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    baro_cols = [f"b{i}" for i in range(1, 7)]
    
    for i, (ax, col) in enumerate(zip(axes, baro_cols)):
        if col not in df_original.columns:
            continue
        
        t = df_original['time'].to_numpy()
        
        # Plot original data (faint)
        ax.plot(t, df_original[col], color='gray', alpha=0.3, 
                linewidth=1.0, label='Original')
        
        # Plot EMA-corrected
        if col in df_ema.columns:
            ax.plot(t, df_ema[col], color='#1f77b4', 
                    linewidth=1.2, label='EMA-corrected')
        
        # Plot temperature-corrected
        if col in df_temp.columns:
            ax.plot(t, df_temp[col], color='#ff7f0e', alpha=0.8,
                    linewidth=1.2, label='Temp-corrected')
        
        # Plot EMA trend (dashed)
        if ema_trends and col in ema_trends:
            ax.plot(t, ema_trends[col], color='red', alpha=0.5,
                    linestyle='--', linewidth=1.0, label='EMA trend')
        
        ax.set_ylabel(f"{col} [hPa]")
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        # Add temperature coefficient to title if available
        if temp_coeffs and col in temp_coeffs:
            slope = temp_coeffs[col]['slope']
            ax.set_title(f"{col} (temp coeff: {slope:+.3f} hPa/°C)", fontsize=9)
        else:
            ax.set_title(f"{col}", fontsize=9)
    
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Drift Removal Comparison", fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved comparison plot: {save_path}")
    
    return fig


def plot_temperature_vs_pressure(df, baro_cols=None, temp_cols=None, 
                                 temp_coeffs=None, save_path=None):
    """
    Plot pressure vs temperature scatter plots to visualize correlation.
    
    Helps diagnose if temperature compensation is appropriate.
    """
    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    if temp_cols is None:
        temp_cols = [f"t{i}" for i in range(1, 7) if f"t{i}" in df.columns]
    
    if len(temp_cols) < len(baro_cols):
        print("plot_temperature_vs_pressure: Missing temperature columns")
        return None
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ax, bcol, tcol) in enumerate(zip(axes, baro_cols, temp_cols)):
        if bcol not in df.columns or tcol not in df.columns:
            continue
        
        T = df[tcol].to_numpy()
        P = df[bcol].to_numpy()
        
        valid = np.isfinite(T) & np.isfinite(P)
        
        if not valid.any():
            continue
        
        # Scatter plot
        ax.scatter(T[valid], P[valid], s=1, alpha=0.3, color='#1f77b4')
        
        # Add fitted line if coefficients available
        if temp_coeffs and bcol in temp_coeffs:
            slope = temp_coeffs[bcol]['slope']
            intercept = temp_coeffs[bcol]['intercept']
            T_range = [T[valid].min(), T[valid].max()]
            P_fit = [intercept + slope*t for t in T_range]
            ax.plot(T_range, P_fit, 'r-', linewidth=2, 
                   label=f'{slope:+.3f} hPa/°C')
            ax.legend(loc='best', fontsize=8)
        
        ax.set_xlabel(f"{tcol} [°C]")
        ax.set_ylabel(f"{bcol} [hPa]")
        ax.set_title(f"{bcol} vs {tcol}")
        ax.grid(alpha=0.3)
    
    fig.suptitle("Temperature vs Pressure Correlation", fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved temperature correlation plot: {save_path}")
    
    return fig


# ==================== REAL-TIME PROCESSOR CLASS ====================

class RealtimeBarometerProcessor:
    """
    Real-time barometer drift removal processor.
    
    Can use plateau step removal, EMA, or temperature-based correction.
    Processes samples one at a time (causal, streaming-compatible).
    
    Example usage:
    -------------
    processor = RealtimeBarometerProcessor(
        method='plateau+ema',
        plateau_params={'stability_window': 100, 'min_step_size': 5.0},
        alpha=0.001
    )
    
    for pressure, temperature in data_stream:
        corrected = processor.process_sample(pressure, temperature)
        # Use corrected value...
    """
    
    def __init__(self, method='ema', alpha=0.001, temp_slope=None, temp_ref=None,
                 plateau_params=None):
        """
        Initialize processor.
        
        Parameters:
        -----------
        method : str
            'plateau' for plateau-based step removal only
            'ema' for EMA high-pass filter only
            'temperature' for temperature compensation only
            'plateau+ema' for step removal then EMA
            'plateau+temperature' for step removal then temperature
        alpha : float
            EMA smoothing factor (only for method containing 'ema')
        temp_slope : float
            Temperature coefficient dP/dT (only for method containing 'temperature')
        temp_ref : float
            Reference temperature (only for method containing 'temperature')
        plateau_params : dict, optional
            Parameters for plateau detector (if method contains 'plateau')
            Keys: stability_window, stability_threshold, min_step_size, etc.
        """
        self.method = method
        self.alpha = alpha
        self.temp_slope = temp_slope
        self.temp_ref = temp_ref
        
        # Initialize plateau detector if needed
        if 'plateau' in method:
            if plateau_params is None:
                plateau_params = {}
            self.plateau_detector = RealtimePlateauDetector(**plateau_params)
        else:
            self.plateau_detector = None
        
        # State variables for EMA
        self.ema = None
        self.baseline = None
        self.initialized = False
    
    def process_sample(self, pressure, temperature=None):
        """
        Process a single sample.
        
        Parameters:
        -----------
        pressure : float
            Current pressure reading [hPa]
        temperature : float, optional
            Current temperature reading [°C] (required for temperature methods)
        
        Returns:
        --------
        corrected_pressure : float
            Drift-corrected pressure
        """
        if not np.isfinite(pressure):
            return np.nan
        
        # Step 1: Plateau-based step removal (if enabled)
        if self.plateau_detector is not None:
            pressure = self.plateau_detector.process_sample(pressure)
        
        # Initialize on first sample (for EMA/temp methods)
        if not self.initialized:
            self.ema = pressure
            self.baseline = pressure
            if temperature is not None and self.temp_ref is None:
                self.temp_ref = temperature
            self.initialized = True
            return 0.0  # First sample is zero
        
        # Step 2: Apply drift removal method
        if 'ema' in self.method:
            # Update EMA
            self.ema = self.alpha * pressure + (1 - self.alpha) * self.ema
            # High-pass output
            corrected = pressure - self.ema
            # Zero at baseline
            corrected = corrected - (self.baseline - self.ema)
            
        elif 'temperature' in self.method:
            if temperature is None:
                raise ValueError("Temperature required for temperature-based methods")
            if self.temp_slope is None:
                raise ValueError("temp_slope must be provided for temperature methods")
            
            # Apply temperature correction
            correction = self.temp_slope * (temperature - self.temp_ref)
            corrected = pressure - correction - self.baseline
        
        elif self.method == 'plateau':
            # Plateau removal only (already applied above)
            corrected = pressure - self.baseline
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return corrected
    
    def reset(self):
        """Reset processor state."""
        if self.plateau_detector is not None:
            self.plateau_detector.reset()
        self.ema = None
        self.baseline = None
        self.initialized = False
    
    def get_state(self):
        """Get current processor state (for diagnostics)."""
        state = {
            'method': self.method,
            'ema': self.ema,
            'baseline': self.baseline,
            'initialized': self.initialized
        }
        if self.plateau_detector is not None:
            state['plateau_detector'] = self.plateau_detector.get_state()
        return state


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("Barometer Drift Removal Library - Enhanced with Plateau Detection")
    print("=" * 70)
    print()
    print("Available functions:")
    print("  - remove_plateau_steps(): Plateau-based step removal (NEW)")
    print("  - remove_plateau_steps_robust(): Robust plateau detection (NEW)")
    print("  - remove_drift_ema(): EMA high-pass filter")
    print("  - remove_drift_temperature_linear(): Temperature compensation")
    print("  - zero_barometers_fixed(): Fixed zeroing function")
    print("  - plot_step_removal_comparison(): Visualize step removal (NEW)")
    print("  - plot_drift_comparison(): Visualization")
    print("  - RealtimeBarometerProcessor: Real-time processing class")
    print("  - RealtimePlateauDetector: Real-time plateau detection class (NEW)")
    print()
    print("Import this module and use these functions in your main script.")