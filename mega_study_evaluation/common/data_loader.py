"""Common data loading utilities for mega study evaluation scripts."""

import pandas as pd
import numpy as np


def load_standard_data(human_path, twin_path, skiprows_human=[1,2], skiprows_twin=[1,2]):
    """
    Standard CSV loading with skiprows handling.
    
    Args:
        human_path: Path to human data CSV
        twin_path: Path to twin data CSV
        skiprows_human: Rows to skip for human data
        skiprows_twin: Rows to skip for twin data
    
    Returns:
        tuple: (df_human, df_twin)
    """
    df_human = pd.read_csv(human_path, header=0, skiprows=skiprows_human)
    df_twin = pd.read_csv(twin_path, header=0, skiprows=skiprows_twin)
    return df_human, df_twin


def merge_twin_data(df_human, df_twin, merge_key="TWIN_ID", suffixes=("_human", "_twin")):
    """
    Standard merge operation with suffixes.
    
    Args:
        df_human: Human data DataFrame
        df_twin: Twin data DataFrame
        merge_key: Column(s) to merge on
        suffixes: Suffixes for overlapping columns
    
    Returns:
        Merged DataFrame
    """
    # Handle both string and list merge keys
    if isinstance(merge_key, str):
        merge_key = [merge_key]
    
    return pd.merge(df_human, df_twin, on=merge_key, suffixes=suffixes)


def prepare_data_for_analysis(df, DV_vars, raw_vars=None):
    """
    Standard data preparation: fix dtypes for analysis variables.
    
    Args:
        df: Merged DataFrame
        DV_vars: List of dependent variables
        raw_vars: List of raw variables (optional)
    
    Returns:
        DataFrame with corrected dtypes
    """
    all_vars = DV_vars
    if raw_vars:
        all_vars = raw_vars + DV_vars
    
    for var in all_vars:
        if f"{var}_human" in df.columns:
            df[f"{var}_human"] = pd.to_numeric(df[f"{var}_human"], errors="coerce")
        if f"{var}_twin" in df.columns:
            df[f"{var}_twin"] = pd.to_numeric(df[f"{var}_twin"], errors="coerce")
    
    return df