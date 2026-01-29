#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Barometer Drift Removal - Real-time Compatible

Methods implemented:
1. EMA High-Pass Filter: Removes slow drift regardless of cause
2. Linear Temperature Compensation: Removes temperature-dependent drift

All methods are real-time compatible (causal, sample-by-sample processing).

Author: Enhanced by Claude
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================== STEP DETECTION & LEVELING ====================

def remove_steps_robust(df, baro_cols=None, threshold=5.0, alpha_baseline=0.01, window_size=10, settling_samples=5):
    """
    Real-time compatible step removal with a settling guard.
    Wait for 'settling_samples' after a jump before calculating the new offset
    to avoid 'hook' or 'tail' artifacts.
    """
    df = df.copy()
    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]

    print(f"\n=== Robust Step Leveling with Settling Guard ===")

    for col in baro_cols:
        y_raw = df[col].to_numpy()
        n = len(y_raw)
        y_corrected = np.zeros(n)
        
        active_offset = 0.0
        running_baseline = 0.0
        initialized = False
        
        # Guard variables
        wait_counter = 0
        
        for i in range(n):
            current_raw = y_raw[i]
            
            if not np.isfinite(current_raw):
                y_corrected[i] = np.nan
                continue
            
            # 1. INITIALIZATION
            if not initialized:
                if i >= window_size:
                    active_offset = np.median(y_raw[max(0, i-window_size):i+1])
                    running_baseline = 0.0
                    initialized = True
                y_corrected[i] = 0.0
                continue

            # 2. APPLY OFFSET
            current_leveled = current_raw - active_offset
            innovation = current_leveled - running_baseline
            
            # 3. STEP DETECTION & SETTLING GUARD
            if np.abs(innovation) > threshold and wait_counter == 0:
                # We detected a jump! Start the waiting period
                wait_counter = settling_samples
            
            if wait_counter > 0:
                wait_counter -= 1
                if wait_counter == 0:
                    # After waiting, we are on the new stable plateau.
                    # Recalculate the offset based on the CURRENT stable value.
                    # This removes the "hook".
                    new_jump_innovation = current_leveled - running_baseline
                    active_offset += new_jump_innovation
                    current_leveled = current_raw - active_offset
                    print(f"  {col}: Step leveled at index {i} after settling.")

            # 4. UPDATE BASELINE
            running_baseline = (alpha_baseline * current_leveled) + ((1 - alpha_baseline) * running_baseline)
            y_corrected[i] = current_leveled
            
        df[col] = y_corrected
            
    return df


# ==================== EMA DRIFT REMOVAL ====================

def remove_drift_ema(df, baro_cols=None, alpha=0.0001, alpha_override=None, zero_at_start=True):
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


# ==================== TEMPERATURE COMPENSATION (LEGACY) ====================

def remove_drift_temperature_linear(df, baro_cols=None, temp_cols=None, 
                                    skip_initial_fraction=0.05, 
                                    zero_at_start=True):
    """
    Remove drift using linear temperature compensation.
    
    Fits a linear relationship P = a + b*T for each sensor, then corrects
    pressure by removing temperature-dependent component.
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
        return df, {}
    
    print(f"\n=== Linear Temperature Compensation ===")
    print(f"Skip initial: {skip_initial_fraction*100:.1f}%")
    
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
        
        if T_range < 0.1:
            print(f"  {bcol}: temperature too stable ({T_range:.3f}°C), skipping")
            continue
        
        # Fit linear model: P = a + b*T
        coeffs = np.polyfit(T_fit, P_fit, deg=1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Reference temperature
        T_ref = T_stable[valid_mask][0]
        
        # Apply correction
        correction = slope * (T - T_ref)
        P_corrected = P - correction
        
        if zero_at_start:
            first_valid = np.where(np.isfinite(P_corrected))[0]
            if len(first_valid) > 0:
                P_corrected = P_corrected - P_corrected[first_valid[0]]
        
        df[bcol] = P_corrected
        coefficients[bcol] = {'slope': slope, 'intercept': intercept, 'T_ref': T_ref}
        
        print(f"  {bcol}: slope = {slope:+.3f} hPa/°C")
    
    return df, coefficients


# ==================== ZEROING FUNCTION ====================

def zero_barometers_fixed(df, baro_cols=None, reference_index='first_valid'):
    """Zero barometers at a specific index."""
    df = df.copy()
    
    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    
    print(f"\n=== Zeroing Barometers ===")
    
    for col in baro_cols:
        if col not in df.columns:
            continue
        
        if reference_index == 'first_valid':
            valid_idx = np.where(np.isfinite(df[col]))[0]
            if len(valid_idx) == 0:
                continue
            ref_idx = valid_idx[0]
        else:
            ref_idx = reference_index
        
        ref_value = df[col].iloc[ref_idx]
        
        if not np.isfinite(ref_value):
            continue
        
        df[col] = df[col] - ref_value
        print(f"  {col}: zeroed at index {ref_idx}")
    
    return df


# ==================== PLOTTING ====================

def plot_drift_comparison(df_original, df_ema, df_temp, 
                          ema_trends=None, temp_coeffs=None,
                          save_path=None):
    """Plot comparison of drift removal methods."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    for i, col in enumerate([f"b{j}" for j in range(1, 7)]):
        if col not in df_original.columns:
            continue
        
        ax = axes[i]
        t = df_original['time'].to_numpy()
        
        ax.plot(t, df_original[col], 'gray', alpha=0.3, label='Original')
        
        if col in df_ema.columns:
            ax.plot(t, df_ema[col], 'b', linewidth=1.2, label='EMA')
        
        if col in df_temp.columns:
            ax.plot(t, df_temp[col], 'orange', alpha=0.8, linewidth=1.2, label='Temp')
        
        ax.set_ylabel(f"{col} [hPa]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Drift Removal Comparison")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig


def plot_temperature_vs_pressure(df, baro_cols=None, temp_cols=None, 
                                 temp_coeffs=None, save_path=None):
    """Plot pressure vs temperature correlation."""
    if baro_cols is None:
        baro_cols = [f"b{i}" for i in range(1, 7) if f"b{i}" in df.columns]
    if temp_cols is None:
        temp_cols = [f"t{i}" for i in range(1, 7) if f"t{i}" in df.columns]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (bcol, tcol) in enumerate(zip(baro_cols, temp_cols)):
        if bcol not in df.columns or tcol not in df.columns:
            continue
        
        ax = axes[i]
        T = df[tcol].to_numpy()
        P = df[bcol].to_numpy()
        
        valid = np.isfinite(T) & np.isfinite(P)
        if not valid.any():
            continue
        
        ax.scatter(T[valid], P[valid], s=1, alpha=0.3)
        
        if temp_coeffs and bcol in temp_coeffs:
            slope = temp_coeffs[bcol]['slope']
            ax.set_title(f"{bcol} ({slope:+.3f} hPa/°C)")
        else:
            ax.set_title(bcol)
        
        ax.set_xlabel(f"{tcol} [°C]")
        ax.set_ylabel(f"{bcol} [hPa]")
        ax.grid(alpha=0.3)
    
    fig.suptitle("Temperature vs Pressure Correlation")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Simple Barometer Processing Library")
    print("=" * 50)
    print("Functions:")
    print("  - remove_drift_ema()")
    print("  - remove_drift_temperature_linear()")
    print("  - zero_barometers_fixed()")
    print("  - SimpleRealtimeProcessor class")
