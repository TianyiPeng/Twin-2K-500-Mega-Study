#!/usr/bin/env python3
"""
Meta-analysis script for Story Beliefs
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


def create_story_beliefs_variables(df_human, df_twin):
    """
    Create study-specific story beliefs variables with weighted calculations.
    This includes complex weighted sum calculations across multiple columns.
    This is unique to story_beliefs study.
    """
    # Define weights once (1-5 scale)
    weights = np.arange(1, 6)

    # For each dataframe...
    for df in (df_human, df_twin):
        # Loop over A vs B, chap1 vs chap2, val vs aro
        for prefix in ["A", "B"]:
            for chap in [1, 2]:
                for metric in ["val", "aro"]:
                    # Build the five source columns
                    cols = [f"{prefix}chap{chap}_{metric}_{i}" for i in range(1, 6)]

                    # New column name, e.g. "Achap1_val" or "Bchap2_aro"
                    newcol = f"{prefix}chap{chap}_{metric}"

                    # Compute denominator (sum of counts)
                    denom = df[cols].sum(axis=1)

                    # Weighted sum divided by total count
                    df[newcol] = df[cols].dot(weights) / denom

                    # Set to NaN where there were no counts
                    df.loc[denom == 0, newcol] = np.nan

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


def analyze_story_beliefs_variables(df_human, df_twin, computed_vars, output_path, spec_info):
    """
    Analyze story beliefs variables with custom logic preserving original calculations.
    """
    # Set index as in original
    df_human = df_human.set_index("TWIN_ID")
    df_twin = df_twin.set_index("TWIN_ID")

    # Variable ranges
    computed_vars_min = [1] * 8
    computed_vars_max = [5] * 8

    # Domain classifications
    social = [0] * 8
    cognitive = [0] * 8
    known = [0] * 8

    domain_maps = create_domain_maps(computed_vars, social, cognitive, known, [], [], [], [])

    # Convert both dataframes to only have the relevant columns
    df_human = df_human[computed_vars]
    df_twin = df_twin[computed_vars]

    # Check if indices match
    if not df_human.index.equals(df_twin.index):
        # Use intersection
        common_index = df_human.index.intersection(df_twin.index)
        df_human = df_human.loc[common_index]
        df_twin = df_twin.loc[common_index]

    # Merge dataframes
    df_all = pd.merge(
        df_human, df_twin, left_index=True, right_index=True, suffixes=("_human", "_twin")
    )

    # Analyze each variable with custom logic preserving original calculations
    all_results = []
    for var in computed_vars:
        var_h = f"{var}_human"
        var_t = f"{var}_twin"

        # Get the data
        human_data = df_all[var_h].dropna()
        twin_data = df_all[var_t].dropna()

        # Get common indices
        common_idx = human_data.index.intersection(twin_data.index)
        human_clean = human_data.loc[common_idx]
        twin_clean = twin_data.loc[common_idx]

        n = len(human_clean)

        if n > 1:
            # Use compute_standard_stats to get all calculations
            # First prepare a temporary dataframe for the function
            temp_df = pd.DataFrame({
                f"{var}_human": human_clean,
                f"{var}_twin": twin_clean
            })
            
            # Get min/max values
            var_idx = computed_vars.index(var)
            min_val = computed_vars_min[var_idx]
            max_val = computed_vars_max[var_idx]
            
            stats_dict = compute_standard_stats(
                temp_df, var, False, None, None, min_val, max_val
            )
            
            result = create_results_dict(
                "Story Beliefs", var, stats_dict, domain_maps, spec_info,
                # Additional study-specific fields
                **{
                    "preference measure?": 0,
                    "stimuli dependent?": 1,
                    "knowledge question?": 0,
                    "political question?": 0,
                }
            )
            all_results.append(result)

    return all_results


def main():
    # Parse arguments
    parser = create_base_parser("Story Beliefs")
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

    # Create story beliefs variables (unique to this study)
    df_human, df_twin = create_story_beliefs_variables(df_human, df_twin)

    # Save processed data files (preserving original behavior)
    save_processed_data(df_human, df_twin, paths['output_path'])

    # Extract specification info for results
    from parse_specification import extract_specification_from_path, parse_specification_name
    specification_name = extract_specification_from_path(args.results_dir)
    if specification_name:
        specification_type, ran_date = parse_specification_name(specification_name)
    else:
        specification_name = "default persona"
        specification_type = "default persona"
        ran_date = None

    spec_info = {
        'type': specification_type,
        'name': specification_name,
        'date': ran_date
    }

    # Define computed response columns
    computed_vars = [
        f"{prefix}chap{chap}_{metric}"
        for prefix in ["A", "B"]
        for chap in [1, 2]
        for metric in ["val", "aro"]
    ]

    # Analyze variables with custom logic
    all_results = analyze_story_beliefs_variables(
        df_human, df_twin, computed_vars, paths['output_path'], spec_info
    )

    # Create results dataframe
    corr_df = pd.DataFrame(all_results)
    
    if args.verbose:
        print(corr_df)

    # Create long format data
    df_long = make_long_format(df_human, df_twin, computed_vars, "Story Beliefs", specification_name)

    # Save standard outputs
    meta_analysis_file, individual_file = save_standard_outputs(
        corr_df, df_long, paths['output_path'], args.verbose
    )

    return corr_df, df_long


if __name__ == "__main__":
    main()