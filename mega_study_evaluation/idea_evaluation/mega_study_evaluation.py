#!/usr/bin/env python3
"""
Meta-analysis script for Idea Evaluation
Converted from: Idea Evaluation meta analysis.ipynb
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
from common.variable_mapper import create_domain_maps, add_min_max_columns, check_condition_variable, build_min_max_maps
from parse_specification import extract_specification_from_path, parse_specification_name


def process_idea_evaluation_data(df_human, df_twin, output_path):
    """
    Study-specific data processing for idea evaluation.
    Creates condition variables and rating columns.
    """
    # create new relevant columns:
    df_human = df_human.set_index("TWIN_ID")
    df_twin = df_twin.set_index("TWIN_ID")
    
    # define the six sets of columns
    baseline_human = [f"{i}_Q11" for i in range(1, 201)]
    baseline_ai = [f"{i}_Q23" for i in range(1, 201)]
    partial_human = [f"{i}_Q13" for i in range(1, 201)]
    partial_ai = [f"{i}_Q24" for i in range(1, 201)]
    full_human = [f"{i}_Q14" for i in range(1, 201)]
    full_ai = [f"{i}_Q25" for i in range(1, 201)]
    
    # assign the 'condition' based on which block has any non‐missing
    df_human["condition"] = np.where(
        df_human[baseline_human].notna().any(axis=1),
        "human ideas - baseline",
        np.where(
            df_human[baseline_ai].notna().any(axis=1),
            "AI ideas - baseline",
            np.where(
                df_human[partial_human].notna().any(axis=1),
                "human ideas - partial",
                np.where(
                    df_human[partial_ai].notna().any(axis=1),
                    "AI ideas - partial",
                    np.where(
                        df_human[full_human].notna().any(axis=1),
                        "human ideas - full",
                        np.where(df_human[full_ai].notna().any(axis=1), "AI ideas - full", None),
                    ),
                ),
            ),
        ),
    )
    df_twin["condition"] = np.where(
        df_twin[baseline_human].notna().any(axis=1),
        "human ideas - baseline",
        np.where(
            df_twin[baseline_ai].notna().any(axis=1),
            "AI ideas - baseline",
            np.where(
                df_twin[partial_human].notna().any(axis=1),
                "human ideas - partial",
                np.where(
                    df_twin[partial_ai].notna().any(axis=1),
                    "AI ideas - partial",
                    np.where(
                        df_twin[full_human].notna().any(axis=1),
                        "human ideas - full",
                        np.where(df_twin[full_ai].notna().any(axis=1), "AI ideas - full", None),
                    ),
                ),
            ),
        ),
    )
    
    # verify that condition assignment is the same for humans and twins:
    # find the intersection of IDs just to be safe
    common_ids = df_human.index.intersection(df_twin.index)
    # a boolean Series saying whether they match for each ID
    matches = df_human.loc[common_ids, "condition"].eq(df_twin.loc[common_ids, "condition"])
    # check if *all* match
    all_match = matches.all()
    print("All conditions agree:", all_match)
    # if you want to see which ones don't match:
    mismatches = matches[~matches]
    print("Mismatched IDs and their human vs twin conditions:")
    for twin_id in mismatches.index:
        print(
            twin_id,
            "human →",
            df_human.at[twin_id, "condition"],
            "| twin →",
            df_twin.at[twin_id, "condition"],
        )
    
    # count total non‐missing across *all* 6 blocks
    all_cols = baseline_human + baseline_ai + partial_human + partial_ai + full_human + full_ai
    df_human["nratings"] = df_twin[all_cols].notna().sum(axis=1)
    df_twin["nratings"] = df_twin[all_cols].notna().sum(axis=1)
    
    # pull out the 20 non‐missing values in order into rating1…rating20
    #    (we assume each row has exactly 20 non‐nulls; if not, extra will be NaN)
    rating_lists = df_human[all_cols].apply(lambda row: row.dropna().tolist(), axis=1)
    rating_df = pd.DataFrame(
        rating_lists.tolist(), index=df_human.index, columns=[f"rating{i + 1}" for i in range(20)]
    )
    df_human = pd.concat([df_human, rating_df], axis=1)
    rating_lists = df_twin[all_cols].apply(lambda row: row.dropna().tolist(), axis=1)
    rating_df = pd.DataFrame(
        rating_lists.tolist(), index=df_twin.index, columns=[f"rating{i + 1}" for i in range(20)]
    )
    df_twin = pd.concat([df_twin, rating_df], axis=1)
    
    # sanity‐check: nratings should be 20 everywhere
    assert (df_human["nratings"] == 20).all(), "Some rows do not have exactly 20 ratings!"
    assert (df_twin["nratings"] == 20).all(), "Some rows do not have exactly 20 ratings!"
    
    # Save processed files to output directory
    df_twin.to_csv(output_path / "idea_evaluation_twins_processed.csv", index=False)
    df_human.to_csv(output_path / "idea_evaluation_human_processed.csv", index=False)
    
    return df_human, df_twin


def main():
    # Parse arguments
    parser = create_base_parser("Idea Evaluation")
    args = parser.parse_args()
    
    # Handle file discovery
    paths = handle_file_discovery(args)
    
    if args.verbose:
        print(f"Loading data from:")
        print(f"  Human: {paths['human_data']}")
        print(f"  Twin: {paths['twin_data']}")
        print(f"  Output: {paths['output_dir']}")

    # Study configuration
    study_name = "idea evaluation"
    
    # Extract specification info
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

    # Load data with custom processing
    df_human, df_twin = load_standard_data(paths['human_data'], paths['twin_data'])
    df_human, df_twin = process_idea_evaluation_data(df_human, df_twin, paths['output_path'])

    # Define study variables
    condition_vars = ["condition"]
    DV_vars = [f"rating{i}" for i in range(1, 21)]
    DV_vars_min = [1] * 20
    DV_vars_max = [5] * 20
    
    # Domain classifications
    DV_vars_social = [1] * 20
    DV_vars_cognitive = [0] * 20
    DV_vars_known = [0] * 20
    DV_vars_pref = [1] * 20
    DV_vars_stim = [1] * 20
    DV_vars_know = [0] * 20
    DV_vars_politics = [0] * 20

    # Merge and prepare data
    df = merge_twin_data(df_human, df_twin)
    df = prepare_data_for_analysis(df, DV_vars)
    
    # Create domain mappings
    domain_maps = create_domain_maps(
        DV_vars, DV_vars_social, DV_vars_cognitive, DV_vars_known, 
        DV_vars_pref, DV_vars_stim, DV_vars_know, DV_vars_politics
    )
    
    # Add min/max columns
    min_map, max_map = build_min_max_maps(DV_vars, DV_vars_min, DV_vars_max)
    df = add_min_max_columns(df, min_map, max_map)
    
    # Check condition variables
    cond_exists, cond, cond_h, cond_t = check_condition_variable(condition_vars)

    # Compute results
    results = []
    for var in DV_vars:
        # Calculate statistics using common function
        stats_dict = compute_standard_stats(
            df, var, cond_exists, cond_h, cond_t, min_map[var], max_map[var]
        )
        
        # Create result dictionary
        result = create_results_dict(
            study_name, var, stats_dict, domain_maps, spec_info
        )
        results.append(result)

    # Convert to DataFrame
    corr_df = pd.DataFrame(results)
    print(corr_df)

    # Create long format data (need to reset index for make_long_format)
    df_human.reset_index(inplace=True)
    df_twin.reset_index(inplace=True)
    df_long = make_long_format(df_human, df_twin, DV_vars, study_name, specification_name)
    print(df_long.head())

    # Save outputs
    save_standard_outputs(corr_df, df_long, paths['output_path'], args.verbose)
    print("done")

    return corr_df, df_long


if __name__ == "__main__":
    main()
