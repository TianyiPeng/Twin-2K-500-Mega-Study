#!/usr/bin/env python3
"""
Meta-analysis script for Junk Fees
Refactored to use common modules while preserving all study-specific logic.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.args_parser import create_base_parser, handle_file_discovery
from common.data_loader import load_standard_data, merge_twin_data, prepare_data_for_analysis
from common.stats_analysis import compute_standard_stats
from common.results_processor import create_results_dict, make_long_format, save_standard_outputs
from common.variable_mapper import create_domain_maps, build_min_max_maps, add_min_max_columns, check_condition_variable


def create_junk_fees_variables(df_human, df_twin):
    """
    Create study-specific junk fees variables with category-based processing.
    This is unique to junk_fees study.
    """
    categories = ["Hotel", "Car", "Ticket", "Food", "Apart", "Health", "CC"]

    for cat in categories:
        # Find MCQ columns for this category
        mcq_cols = [col for col in df_human.columns if col.startswith(f"{cat}.MCQ")]
        if not mcq_cols:
            raise ValueError(f"No {cat}.MCQ column found")
        mcq_col = mcq_cols[0]
        correct_col = f"{cat}.correct"
        
        # Create correct/incorrect indicators
        df_human[correct_col] = np.where(
            df_human[mcq_col].isna(), np.nan, (df_human[mcq_col] == 1).astype(int)
        )
        df_twin[correct_col] = np.where(
            df_twin[mcq_col].isna(), np.nan, (df_twin[mcq_col] == 1).astype(int)
        )

        # Find fairness columns
        fair_cols = [col for col in df_human.columns if col.startswith(f"{cat}.Fair")]
        if not fair_cols:
            raise ValueError(f"No {cat}.Fair column found")
        fair_col = fair_cols[0]
        rating_col = f"{cat}.fairness"

        # Create fairness ratings
        df_human[rating_col] = df_human[fair_col]
        df_twin[rating_col] = df_twin[fair_col]

    # Validate all categories have data
    correct_cols = [f"{cat}.correct" for cat in categories]
    fairness_cols = [f"{cat}.fairness" for cat in categories]
    
    # Check correct columns
    non_nan_counts = df_human[correct_cols].notna().sum(axis=1)
    if not (non_nan_counts == 6).all():
        bad_rows = non_nan_counts[non_nan_counts != 6]
        raise ValueError(
            f"Expected exactly 6 non-NaN .correct columns per row, but found counts:\n{bad_rows}"
        )
    
    non_nan_counts = df_twin[correct_cols].notna().sum(axis=1)
    if not (non_nan_counts == 6).all():
        bad_rows = non_nan_counts[non_nan_counts != 6]
        raise ValueError(
            f"Expected exactly 6 non-NaN .correct columns per row, but found counts:\n{bad_rows}"
        )
    
    # Check fairness columns
    non_nan_counts = df_human[fairness_cols].notna().sum(axis=1)
    if not (non_nan_counts == 6).all():
        bad = non_nan_counts[non_nan_counts != 6]
        raise ValueError(f"Expected exactly 6 non-NaN fairness ratings per row, but found:\n{bad}")
    
    non_nan_counts = df_twin[fairness_cols].notna().sum(axis=1)
    if not (non_nan_counts == 6).all():
        bad = non_nan_counts[non_nan_counts != 6]
        raise ValueError(f"Expected exactly 6 non-NaN fairness ratings per row, but found:\n{bad}")

    # Create aggregate variables
    df_human["percent_correct"] = df_human[correct_cols].sum(axis=1) / 6 * 100
    df_human["fairness_average"] = df_human[fairness_cols].mean(axis=1)
    df_twin["percent_correct"] = df_twin[correct_cols].sum(axis=1) / 6 * 100
    df_twin["fairness_average"] = df_twin[fairness_cols].mean(axis=1)

    # Create regulation support variable
    reg_cols = ["Regulation", "Support.Reg"]
    df_human["reg_support"] = df_human[reg_cols].mean(axis=1)
    df_twin["reg_support"] = df_twin[reg_cols].mean(axis=1)

    return df_human, df_twin


def save_processed_data(df_human, df_twin, output_path):
    """
    Save processed data files as in original study.
    """
    human_file = output_path / "human data values anonymized processed.csv"
    twin_file = output_path / "twins data values anonymized processed.csv"
    df_twin.to_csv(twin_file, index=False)
    df_human.to_csv(human_file, index=False)
    return human_file, twin_file


def make_long_format_with_index_reset(df, respondent_type, DV_vars, study_name, specification_name):
    """
    Custom long format function that handles index reset for junk_fees.
    """
    # Reset index to make TWIN_ID a regular column (junk_fees sets index earlier)
    df_reset = df.reset_index() if df.index.name == 'TWIN_ID' else df
    
    # Pick off TWIN_ID + the DVs, then melt
    long = df_reset[["TWIN_ID"] + DV_vars].melt(
        id_vars="TWIN_ID", value_vars=DV_vars, var_name="variable_name", value_name="value"
    )

    long["respondent_type"] = respondent_type
    long["study_name"] = study_name
    long["specification_name"] = specification_name
    return long


def main():
    # Parse arguments
    parser = create_base_parser("Junk Fees")
    args = parser.parse_args()

    # Handle file discovery
    paths = handle_file_discovery(args)

    if args.verbose:
        print(f"Loading data from:")
        print(f"  Human: {paths['human_data']}")
        print(f"  Twin: {paths['twin_data']}")
        print(f"  Output: {paths['output_dir']}")

    # Load data
    df_human, df_twin = load_standard_data(paths['human_data'], paths['twin_data'])

    # Create junk fees variables (unique to this study)
    df_human, df_twin = create_junk_fees_variables(df_human, df_twin)

    # Save processed data files (preserving original behavior)
    save_processed_data(df_human, df_twin, paths['output_path'])

    # Set index as in original (needed for later processing)
    df_human = df_human.set_index("TWIN_ID")
    df_twin = df_twin.set_index("TWIN_ID")

    # Define study variables
    DV_vars = ["percent_correct", "fairness_average", "reg_support"]
    DV_vars_min = [0, 1, 1]
    DV_vars_max = [100, 7, 7]  # Fixed range on 07/25/25

    # Domain classifications
    social = [0, 1, 1]
    cognitive = [1, 0, 0]
    known = [0, 0, 0]
    pref = [0, 1, 1]
    stim = [1, 1, 1]
    know = [1, 0, 0]
    politics = [0, 0, 1]

    # Create domain maps
    domain_maps = create_domain_maps(DV_vars, social, cognitive, known, pref, stim, know, politics)

    # Check condition variables (none for this study)
    condition_vars = [""]
    cond_exists, cond, cond_h, cond_t = check_condition_variable(condition_vars)

    # Build min/max maps and merge data
    min_map, max_map = build_min_max_maps(DV_vars, DV_vars_min, DV_vars_max)
    df = merge_twin_data(df_human, df_twin, merge_key=["TWIN_ID"])
    df = prepare_data_for_analysis(df, DV_vars)
    df = add_min_max_columns(df, min_map, max_map)

    # Extract specification info for results
    from parse_specification import extract_specification_from_path, parse_specification_name
    specification_name = extract_specification_from_path(args.results_dir)
    if specification_name:
        specification_type, ran_date = parse_specification_name(specification_name)
    else:
        specification_name = "unknown"
        specification_type = "unknown"
        ran_date = None

    spec_info = {
        'type': specification_type,
        'name': specification_name,
        'date': ran_date
    }

    # Compute results for all variables
    results = []
    for var in DV_vars:
        stats_dict = compute_standard_stats(
            df, var, cond_exists, cond_h, cond_t,
            min_map.get(var), max_map.get(var)
        )
        
        result = create_results_dict(
            "Junk Fees", var, stats_dict, 
            domain_maps, spec_info
        )
        results.append(result)

    # Create results DataFrame
    corr_df = pd.DataFrame(results)
    
    if args.verbose:
        print(corr_df)

    # Create long format data using custom function
    long_h = make_long_format_with_index_reset(df_human, "human", DV_vars, "Junk Fees", specification_name)
    long_t = make_long_format_with_index_reset(df_twin, "twin", DV_vars, "Junk Fees", specification_name)
    df_long = pd.concat([long_h, long_t], ignore_index=True)

    # Save outputs
    meta_analysis_file, individual_file = save_standard_outputs(
        corr_df, df_long, paths['output_path'], args.verbose
    )

    return corr_df, df_long


if __name__ == "__main__":
    main()