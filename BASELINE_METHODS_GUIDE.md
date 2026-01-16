# Baseline Zeroing Methods - User Guide

This guide explains the 3 baseline selection methods implemented in `data_organization - New.py`.

---

## Quick Start

Edit these settings at the top of the file:

```python
BASELINE_METHOD = 'auto'        # Choose: 'auto', 'fixed', or 'zero_fz'
BASELINE_DURATION_S = 2         # Window size for baseline (in seconds)
BASELINE_WARMUP_S = 2.0         # For 'fixed' method only
BASELINE_FZ_THRESHOLD = 0.1     # For 'zero_fz' method only (in Newtons)
```

---

## Method 1: AUTO (Automatic Stable Period Detection) ⭐ RECOMMENDED

### Configuration:
```python
BASELINE_METHOD = 'auto'
BASELINE_DURATION_S = 2  # Window size to search for
```

### How it works:
- Scans through your entire dataset
- Finds the 2-second window with **lowest total variance** across all barometers
- Automatically avoids unstable warm-up periods
- Creates a diagnostic plot showing the variance scan

### When to use:
- ✅ **Offline processing** (analyzing recorded data)
- ✅ When you're not sure when the sensor stabilizes
- ✅ When data has unpredictable warm-up behavior
- ❌ **NOT for real-time** (needs to look ahead in time)

### Output example:
```
[Method: AUTO] Finding most stable 2.0s period:
  ✓ Most stable period: t=3.456-5.456s (indices 1234-1678)
    Total variance: 0.0234

  Subtracting baseline from 6 barometers:
    b1: baseline= 850.23 hPa, std=0.123 hPa  ✓ excellent
    b2: baseline= 970.45 hPa, std=0.156 hPa  ✓ excellent
    ...
  Diagnostic plot saved: baseline_diagnostic.png
```

### Quality indicators:
- `std < 0.5 hPa` → ✓ **excellent** baseline
- `std < 1.0 hPa` → ✓ **good** baseline
- `std < 2.0 hPa` → ⚠ **fair** baseline (consider longer duration)
- `std > 2.0 hPa` → ✗ **poor** baseline (try different method)

---

## Method 2: FIXED (Fixed Time Window) 🕐

### Configuration:
```python
BASELINE_METHOD = 'fixed'
BASELINE_DURATION_S = 1.0      # Baseline window duration
BASELINE_WARMUP_S = 2.0        # Skip first N seconds (warmup)
```

### How it works:
- Waits `BASELINE_WARMUP_S` seconds for sensor warm-up
- Uses the next `BASELINE_DURATION_S` seconds for baseline
- Example: With settings above, uses t=2.0-3.0s for baseline

### When to use:
- ✅ **Real-time processing** (works online, causal)
- ✅ When sensor warm-up time is consistent and known
- ✅ When you need predictable, deterministic behavior
- ❌ When warm-up time varies between experiments

### Output example:
```
[Method: FIXED] Using t=2.0-3.0s
  (skipping first 2.0s warmup)
  Baseline samples: 234

  Subtracting baseline from 6 barometers:
    b1: baseline= 850.45 hPa, std=0.234 hPa  ✓ excellent
    ...
```

### Tips:
- Set `BASELINE_WARMUP_S` based on your sensor's warm-up time (typically 1-3 seconds)
- Increase `BASELINE_DURATION_S` for more stable averaging (but delays real-time start)
- If `std > 1.0 hPa`, try increasing `BASELINE_WARMUP_S`

---

## Method 3: ZERO_FZ (Force-Based Baseline) 🎯

### Configuration:
```python
BASELINE_METHOD = 'zero_fz'
BASELINE_DURATION_S = 1.0      # Minimum zero-force window duration
BASELINE_FZ_THRESHOLD = 0.1    # |Fz| threshold (Newtons)
```

### How it works:
- Finds periods where `|Fz| < BASELINE_FZ_THRESHOLD` (no contact)
- Selects the most stable zero-force period ≥ `BASELINE_DURATION_S`
- Uses that period for baseline calculation

### When to use:
- ✅ When you have clear "no contact" periods at the beginning
- ✅ When warm-up time is unpredictable but you know when Fz=0
- ✅ As a fallback if 'auto' picks a bad period
- ❌ When experiment starts with immediate contact
- ❌ When force sensor is noisy (may not find clean Fz≈0 periods)

### Output example:
```
[Method: ZERO_FZ] Finding baseline from zero-force periods (|Fz| < 0.1N):
  ✓ Best zero-force period: t=1.234-3.456s (indices 567-1234)
    Variance: 0.0345, Duration: 2.2s

  Subtracting baseline from 6 barometers:
    b1: baseline= 850.12 hPa, std=0.178 hPa  ✓ excellent
    ...
```

### Automatic fallback:
If no suitable zero-force periods are found, it automatically falls back to 'fixed' method with your `BASELINE_WARMUP_S` settings.

---

## Diagnostic Plot (AUTO method only)

When using `BASELINE_METHOD = 'auto'`, a diagnostic plot is saved as `baseline_diagnostic.png` with 3 panels:

1. **Variance Scan**: Shows variance across time, with green region = selected baseline
2. **Barometer Traces**: Full time series with baseline region highlighted
3. **Baseline Region (Zoomed)**: Close-up of the selected baseline period

Use this plot to validate that the automatic selection makes sense!

---

## Real-Time Configuration Example

For real-time processing, use this configuration:

```python
# === REAL-TIME SETTINGS ===
BASELINE_METHOD = 'fixed'       # Can't search future data
BASELINE_DURATION_S = 1.0       # Use t=2.0-3.0s for baseline
BASELINE_WARMUP_S = 2.0         # Wait 2s for sensor warmup

# Dynamic re-zeroing to handle drift over time
ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ = True
FZ_ZERO_THRESHOLD = 0.2
MIN_ZERO_DURATION_S = 0.01
MIN_ZERO_SAMPLES = 5
```

**Why this works for real-time:**
- 'fixed' method only needs past data (causal)
- Dynamic re-zeroing adapts to drift during operation
- Together, they provide robust real-time performance

---

## Offline Configuration Example (Best Quality)

For offline analysis (your current use case), use:

```python
# === OFFLINE SETTINGS (BEST QUALITY) ===
BASELINE_METHOD = 'auto'        # Automatically find best period
BASELINE_DURATION_S = 2         # Longer window = more stable

# Temperature drift compensation
DRIFT_REMOVAL_METHOD = 'temperature'
DRIFT_POLY_ORDER = 2

# Dynamic re-zeroing for long experiments
ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ = True
FZ_ZERO_THRESHOLD = 0.2
```

---

## Troubleshooting

### Problem: Large spikes in final plot (like your case)

**Diagnosis**: Baseline was calculated during unstable warm-up period

**Solutions** (in order of preference):
1. ✅ Use `BASELINE_METHOD = 'auto'` with `BASELINE_DURATION_S = 2`
2. ✅ Use `BASELINE_METHOD = 'fixed'` with larger `BASELINE_WARMUP_S`
3. ✅ Try `BASELINE_METHOD = 'zero_fz'` if you have clean Fz≈0 periods

### Problem: "✗ poor" quality indicators (std > 2.0 hPa)

**Solutions**:
- Increase `BASELINE_DURATION_S` (try 2, 3, or even 5 seconds)
- For 'fixed' method: increase `BASELINE_WARMUP_S`
- Check the diagnostic plot to see if baseline region looks noisy

### Problem: Method fails to find suitable period

**AUTO**: May happen if entire dataset is unstable
- Solution: Try 'zero_fz' method instead

**ZERO_FZ**: No zero-force periods found
- Solution: Decrease `BASELINE_FZ_THRESHOLD` (try 0.2 or 0.3 N)
- Or use 'fixed' method with known warmup time

---

## Comparison Table

| Feature | AUTO | FIXED | ZERO_FZ |
|---------|------|-------|---------|
| **Real-time capable** | ❌ No | ✅ Yes | ✅ Yes |
| **Best for offline** | ✅ Yes | ❌ No | ⚠️ Maybe |
| **Needs force data** | ❌ No | ❌ No | ✅ Yes |
| **Automatic** | ✅ Yes | ❌ No | ⚠️ Semi |
| **Predictable** | ❌ No | ✅ Yes | ❌ No |
| **Diagnostic plot** | ✅ Yes | ❌ No | ❌ No |
| **Quality check** | ✅ Yes | ✅ Yes | ✅ Yes |

---

## Recommendations by Use Case

### 🎯 Your current case (offline, fixing large spikes):
```python
BASELINE_METHOD = 'auto'
BASELINE_DURATION_S = 2
```

### 🔴 Real-time application:
```python
BASELINE_METHOD = 'fixed'
BASELINE_WARMUP_S = 2.0
BASELINE_DURATION_S = 1.0
ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ = True
```

### 🧪 Experiments with touch/release cycles:
```python
BASELINE_METHOD = 'zero_fz'
BASELINE_FZ_THRESHOLD = 0.1
ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ = True
```

---

## Summary

**For your plots to look better NOW:**
1. Set `BASELINE_METHOD = 'auto'` (already configured!)
2. Set `BASELINE_DURATION_S = 2` (already configured!)
3. Run your script
4. Check the quality indicators in the output
5. Look at `baseline_diagnostic.png` to validate

**For future real-time work:**
- Use `BASELINE_METHOD = 'fixed'` with appropriate `BASELINE_WARMUP_S`
- Keep `ENABLE_DYNAMIC_REZERO_ON_ZERO_FZ = True` for drift compensation
