#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Prediction Script (per-target features & scalers)
- Loads specialized RF models where each target stores:
  {'model': <RF>, 'scaler': <StandardScaler>, 'features': [b? ...]}
- Predicts on new data using each target's own feature subset and scaler.
- Backward compatible: if the models file is old (no dict), it will use the
  legacy single scaler file x_scaler_rf_v4.pkl and all features.
"""

import os
import pandas as pd
import numpy as np
import pickle
from collections.abc import Mapping

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
MODEL_DIR   = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\models parameters\random forest"
MODEL_FILE  = "specialized_rf_models_v4.pkl"   # new format (per-target)
SCALER_FILE = "x_scaler_rf_v4.pkl"             # legacy single-scaler (fallback)

NEW_DATA_DIR  = r"C:\Users\aurir\OneDrive - epfl.ch\Thesis- Biorobotics Lab\test data"
NEW_DATA_FILE = r"test 4102 - sensor v4\synchronized_events_4102.csv"

# Number of random samples to predict
N_SAMPLES = 100
RANDOM_SEED = 42

# Preferred display/order for outputs if present
PREFERRED_TARGET_ORDER = ['x', 'y', 'fx', 'fy', 'fz']

# =============================================================================
# HELPERS
# =============================================================================

def is_new_models_format(models_obj):
    """
    New format: models is a dict mapping target -> {'model','scaler','features'}.
    Old format: models is a dict mapping target -> sklearn estimator only.
    """
    if not isinstance(models_obj, Mapping):
        return False
    # peek one entry
    k = next(iter(models_obj.keys()))
    v = models_obj[k]
    return isinstance(v, Mapping) and all(key in v for key in ('model', 'scaler', 'features'))

def ordered_targets(keys, preferred_order):
    keys_set = set(keys)
    ordered = [t for t in preferred_order if t in keys_set]
    # append any extra keys in deterministic order
    ordered += [t for t in sorted(keys) if t not in ordered]
    return ordered

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    print("="*70)
    print("LOADING TRAINED MODELS")
    print("="*70)

    models_path = os.path.join(MODEL_DIR, MODEL_FILE)
    with open(models_path, 'rb') as f:
        models = pickle.load(f)

    new_format = is_new_models_format(models)
    print(f"✓ Loaded models from: {models_path}")
    print(f"✓ Detected format: {'per-target dict (new)' if new_format else 'single-estimators (legacy)'}")

    legacy_scaler = None
    if not new_format:
        # Legacy path: load the single global scaler (all features)
        scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)
        with open(scaler_path, 'rb') as f:
            legacy_scaler = pickle.load(f)
        print(f"✓ Loaded legacy scaler from: {scaler_path}")

    print("\n" + "="*70)
    print("LOADING NEW DATA")
    print("="*70)

    new_data_path = os.path.join(NEW_DATA_DIR, NEW_DATA_FILE)
    df_new = pd.read_csv(new_data_path)
    df_new.columns = df_new.columns.str.strip()
    print(f"✓ Loaded {len(df_new)} samples from: {NEW_DATA_FILE}\n")

    # Select random samples
    if N_SAMPLES > len(df_new):
        N_SAMPLES = len(df_new)
        print(f"⚠️  Dataset has only {len(df_new)} samples, using all of them")
    random_indices = np.random.choice(len(df_new), size=N_SAMPLES, replace=False)
    df_sample = df_new.iloc[random_indices].copy()
    print(f"Selected {N_SAMPLES} random samples\n")

    # Determine target order
    OUTPUT_TARGETS = ordered_targets(list(models.keys()), PREFERRED_TARGET_ORDER)
    print(f"Targets: {OUTPUT_TARGETS}\n")

    # Prepare results frame with barometers always included for reference
    all_b_cols = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    for b in all_b_cols:
        if b not in df_sample.columns:
            raise ValueError(f"Missing barometer column '{b}' in input CSV.")
    results = pd.DataFrame({
        'Sample_Index': random_indices,
        **{b: df_sample[b].values for b in all_b_cols}
    })

    # Make predictions per target, using each target's own feature subset & scaler
    print("="*70)
    print("MAKING PREDICTIONS")
    print("="*70)
    predictions = {}

    for target in OUTPUT_TARGETS:
        if new_format:
            # New: each target carries (model, scaler, features)
            entry = models[target]
            model   = entry['model']
            scaler  = entry['scaler']
            feats   = entry['features']
            # sanity check on features presence
            for c in feats:
                if c not in df_sample.columns:
                    raise ValueError(f"Target '{target}' expects feature '{c}' not found in input CSV.")
            X_new = df_sample[feats].values
            X_scaled = scaler.transform(X_new)
            pred = model.predict(X_scaled)
        else:
            # Legacy: single global scaler, all barometers
            feats = all_b_cols
            model = models[target]
            X_new = df_sample[feats].values
            X_scaled = legacy_scaler.transform(X_new)
            pred = model.predict(X_scaled)

        predictions[target] = pred
        results[f'{target}_pred'] = pred

    # Add ground truth (if present)
    for target in OUTPUT_TARGETS:
        if target in df_sample.columns:
            true_vals = df_sample[target].values
            results[f'{target}_true'] = true_vals
            results[f'{target}_error'] = true_vals - results[f'{target}_pred'].values

    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)

    have_truth = any(f'{t}_true' in results.columns for t in OUTPUT_TARGETS)
    if have_truth:
        print("\n📊 PREDICTIONS vs ACTUAL VALUES:\n")
        display_cols = ['Sample_Index']
        for t in OUTPUT_TARGETS:
            display_cols.extend([f'{t}_pred', f'{t}_true', f'{t}_error'])
        print(results[display_cols].to_string(index=False))

        # Error stats
        print("\n" + "="*70)
        print("ERROR STATISTICS")
        print("="*70)
        for t in OUTPUT_TARGETS:
            if f'{t}_error' in results.columns:
                e = results[f'{t}_error'].values
                mae = np.abs(e).mean()
                rmse = np.sqrt((e**2).mean())
                unit = "mm" if t in ['x', 'y'] else "N"
                print(f"\n{t.upper()}:")
                print(f"  MAE:  {mae:.4f} {unit}")
                print(f"  RMSE: {rmse:.4f} {unit}")
    else:
        print("\n📊 PREDICTIONS (no ground truth available):\n")
        display_cols = ['Sample_Index'] + [f'{t}_pred' for t in OUTPUT_TARGETS]
        print(results[display_cols].to_string(index=False))

    # Save predictions to CSV
    output_filename = NEW_DATA_FILE.replace('\\', '_').replace('/', '_').replace('.csv', '_predictions.csv')
    output_path = os.path.join(MODEL_DIR, output_filename)
    results.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions to: {output_path}")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
