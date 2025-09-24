#!/usr/bin/env python3
"""Intelligent formatting for ML predictions with automatic integer conversion detection.

This module provides functions to:
1. Analyze ground truth data to determine which columns should be integers
2. Convert ML predictions to appropriate integer values when applicable
3. Format predictions for evaluation
"""

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def analyze_column_types(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Dict[str, any]]:
    """Analyze numeric columns to determine if they should be integers.

    Args:
        df: DataFrame with ground truth data
        numeric_columns: List of numeric column names to analyze

    Returns:
        Dictionary with column analysis results
    """
    column_info = {}

    for col in numeric_columns:
        if col not in df.columns:
            continue

        # Convert to numeric
        series = pd.to_numeric(df[col], errors="coerce")
        valid_values = series.dropna()

        if len(valid_values) == 0:
            continue

        # Check if all values are integers
        is_all_integers = np.all(valid_values == valid_values.astype(int))

        # Get unique values
        unique_values = sorted(valid_values.unique())
        n_unique = len(unique_values)

        # Determine if this looks like a discrete scale
        is_discrete_scale = False
        scale_range = None

        if is_all_integers and n_unique <= 10:
            # Check for common scales (1-5, 1-7, 0-10, etc.)
            int_values = [int(v) for v in unique_values]
            if int_values == list(range(min(int_values), max(int_values) + 1)):
                is_discrete_scale = True
                scale_range = (min(int_values), max(int_values))

        # Check if binary (1,2 or 0,1)
        is_binary = n_unique == 2 and is_all_integers

        column_info[col] = {
            "is_integer": is_all_integers,
            "is_discrete_scale": is_discrete_scale,
            "is_binary": is_binary,
            "unique_values": (
                unique_values.tolist()
                if n_unique <= 20 and hasattr(unique_values, "tolist")
                else list(unique_values) if n_unique <= 20 else None
            ),
            "n_unique": n_unique,
            "scale_range": scale_range,
            "min": float(valid_values.min()),
            "max": float(valid_values.max()),
            "mean": float(valid_values.mean()),
            "std": float(valid_values.std()),
        }

    return column_info


def intelligent_round_predictions(
    predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame, numeric_columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Intelligently round predictions based on ground truth data characteristics.

    Args:
        predictions_df: DataFrame with ML predictions
        ground_truth_df: DataFrame with ground truth data
        numeric_columns: List of numeric columns to process

    Returns:
        Tuple of (formatted_predictions_df, formatting_log)
    """
    # Analyze ground truth columns
    column_info = analyze_column_types(ground_truth_df, numeric_columns)

    # Copy predictions to avoid modifying original
    formatted_df = predictions_df.copy()
    formatting_log = {}

    for col in numeric_columns:
        if col not in predictions_df.columns or col not in column_info:
            continue

        info = column_info[col]
        pred_series = pd.to_numeric(formatted_df[col], errors="coerce")

        if pred_series.isna().all():
            continue

        # Apply appropriate formatting based on column type
        if info["is_integer"]:
            if info["is_discrete_scale"] and info["scale_range"]:
                # Round and clip to scale range
                min_val, max_val = info["scale_range"]
                formatted_series = pred_series.round().clip(min_val, max_val).astype(int)
                formatting_log[col] = f"Rounded and clipped to scale [{min_val}, {max_val}]"

            elif info["is_binary"]:
                # For binary, round to nearest valid value
                valid_values = info["unique_values"]
                if valid_values == [1, 2] or valid_values == [1.0, 2.0]:
                    # 1-2 binary: round to nearest
                    formatted_series = pred_series.round().clip(1, 2).astype(int)
                    formatting_log[col] = "Rounded to binary values [1, 2]"
                elif valid_values == [0, 1] or valid_values == [0.0, 1.0]:
                    # 0-1 binary: round to nearest
                    formatted_series = pred_series.round().clip(0, 1).astype(int)
                    formatting_log[col] = "Rounded to binary values [0, 1]"
                else:
                    # Other binary: map to closest values
                    v1, v2 = valid_values[0], valid_values[1]
                    midpoint = (v1 + v2) / 2
                    formatted_series = pred_series.apply(
                        lambda x: v1 if x < midpoint else v2
                    ).astype(int)
                    formatting_log[col] = f"Mapped to binary values {valid_values}"

            else:
                # General integer column: just round
                formatted_series = pred_series.round().astype(int)
                formatting_log[col] = "Rounded to integer"

            # Update the column
            formatted_df[col] = formatted_series

        else:
            # Continuous variable - keep as is
            formatting_log[col] = "Kept as continuous (not integer in ground truth)"

    return formatted_df, formatting_log


def format_predictions_for_study(
    predictions_file: str, ground_truth_file: str, output_file: str, study_name: str
) -> Dict[str, any]:
    """Format predictions for a specific study with intelligent integer conversion.

    Args:
        predictions_file: Path to ML predictions CSV
        ground_truth_file: Path to ground truth CSV
        output_file: Path for formatted output CSV
        study_name: Name of the study

    Returns:
        Dictionary with formatting results
    """
    logger.info(f"Formatting predictions for {study_name}")

    # Load data
    pred_df = pd.read_csv(predictions_file)
    gt_df = pd.read_csv(ground_truth_file)

    logger.info(f"Loaded {len(pred_df)} predictions and {len(gt_df)} ground truth samples")

    # Identify numeric columns (excluding metadata)
    exclude_patterns = ["Date", "Time", "ID", "Duration", "Progress", "Status"]
    numeric_cols = []

    for col in pred_df.columns:
        if any(pattern in col for pattern in exclude_patterns):
            continue
        if col == "TWIN_ID":
            continue

        # Check if column has numeric data
        try:
            numeric_series = pd.to_numeric(pred_df[col], errors="coerce")
            if numeric_series.notna().sum() > 0:
                numeric_cols.append(col)
        except:
            pass

    logger.info(f"Found {len(numeric_cols)} numeric columns to process")

    # Apply intelligent formatting
    formatted_df, formatting_log = intelligent_round_predictions(pred_df, gt_df, numeric_cols)

    # Save formatted predictions
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    formatted_df.to_csv(output_file, index=False)

    # Prepare results
    results = {
        "study_name": study_name,
        "input_file": predictions_file,
        "output_file": output_file,
        "total_columns": len(pred_df.columns),
        "numeric_columns_processed": len(numeric_cols),
        "formatting_applied": formatting_log,
    }

    # Log formatting summary
    logger.info(f"Formatting complete for {study_name}:")
    integer_cols = [col for col, log in formatting_log.items() if "integer" in log.lower()]
    scale_cols = [col for col, log in formatting_log.items() if "scale" in log.lower()]
    binary_cols = [col for col, log in formatting_log.items() if "binary" in log.lower()]

    logger.info(f"  - Converted to integer: {len(integer_cols)} columns")
    logger.info(f"  - Discrete scales: {len(scale_cols)} columns")
    logger.info(f"  - Binary variables: {len(binary_cols)} columns")

    return results


def main():
    """Example usage of intelligent formatting."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Intelligently format ML predictions based on ground truth data types"
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth CSV")
    parser.add_argument("--output", required=True, help="Path for formatted output CSV")
    parser.add_argument("--study-name", default="study", help="Name of the study")

    args = parser.parse_args()

    results = format_predictions_for_study(
        args.predictions, args.ground_truth, args.output, args.study_name
    )

    # Print detailed results
    print(f"\nFormatting Results for {results['study_name']}:")
    print(f"Total columns: {results['total_columns']}")
    print(f"Numeric columns processed: {results['numeric_columns_processed']}")

    if results["formatting_applied"]:
        print("\nFormatting applied:")
        for col, action in sorted(results["formatting_applied"].items()):
            print(f"  {col}: {action}")


if __name__ == "__main__":
    main()
