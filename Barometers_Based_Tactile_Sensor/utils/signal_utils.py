"""
Signal processing utilities for barometer time-series data.

Functions
---------
maybe_denoise          -- Optional rolling-mean smoothing on barometer channels.
convert_sentinel_to_nan -- Replace sentinel values with NaN in target columns.
"""

import numpy as np
import pandas as pd


def maybe_denoise(df, baro_cols, apply_denoising=True, denoise_window=5):
    """
    Optionally apply rolling-mean denoising on barometer channels.

    Parameters
    ----------
    df : pd.DataFrame
    baro_cols : list of str
        Barometer column names to smooth.
    apply_denoising : bool
        If False, return df unchanged (no copy made).
    denoise_window : int
        Rolling window size (odd number recommended: 3, 5, 7, …).

    Returns
    -------
    df : pd.DataFrame
        A *copy* of the input if denoising was applied; the original otherwise.
    """
    if not apply_denoising:
        return df
    df = df.copy()
    for col in baro_cols:
        df[col] = df[col].rolling(denoise_window, center=True).mean().bfill().ffill()
    return df


def convert_sentinel_to_nan(df, target_cols, sentinel=-999.0):
    """
    Replace sentinel values with NaN in the specified target columns.

    During data collection, rows without contact had their target columns set
    to a sentinel value (default -999).  Converting these to NaN allows
    downstream code to drop or ignore them cleanly.

    Parameters
    ----------
    df : pd.DataFrame
    target_cols : list of str
    sentinel : float
        The value to replace with NaN.

    Returns
    -------
    df : pd.DataFrame  (copy)
    """
    df = df.copy()
    n_converted = 0
    for col in target_cols:
        if col in df.columns:
            mask = df[col] == sentinel
            n_col = mask.sum()
            if n_col > 0:
                df.loc[mask, col] = np.nan
                n_converted += n_col

    if n_converted > 0:
        pct = 100 * n_converted / (len(df) * len(target_cols))
        print(
            f"\nConverted {n_converted} sentinel values ({sentinel}) to NaN "
            f"({pct:.2f}% of target values)"
        )
        print("  This teaches the model to recognize no-contact barometer patterns")

    return df
