"""
I/O utilities for loading and validating tabular CSV data files.

Functions
---------
load_tabular_csv -- Load a CSV, strip whitespace from column names,
                    and validate that expected columns are present.
"""

import os
import pandas as pd


def load_tabular_csv(path: str, expected_cols: list) -> pd.DataFrame:
    """
    Load a CSV file, normalise column names, and validate required columns.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.
    expected_cols : list of str
        Column names that must be present after loading.

    Returns
    -------
    df : pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If any column in *expected_cols* is absent from the loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"File '{os.path.basename(path)}' is missing columns: {missing}"
        )
    return df
