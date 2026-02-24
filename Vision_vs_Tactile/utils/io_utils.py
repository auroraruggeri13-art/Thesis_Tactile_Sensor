"""
I/O utilities for the Vision_vs_Tactile pipeline.

Shared helpers for loading ATI F/T data and barometer CSVs that are used
by both direct_jpeg_extractor.py and forces_for_free.py.
"""

import os
import pandas as pd


def load_tabular_csv(path: str, expected_cols: list | None = None) -> pd.DataFrame:
    """
    Load a CSV file, strip whitespace from column names, and optionally
    validate that expected columns are present.

    Parameters
    ----------
    path : str
    expected_cols : list of str, optional

    Returns
    -------
    df : pd.DataFrame

    Raises
    ------
    FileNotFoundError
    ValueError  (if expected columns are missing)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"File '{os.path.basename(path)}' is missing columns: {missing}"
            )
    return df
