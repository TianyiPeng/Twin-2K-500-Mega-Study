#!/usr/bin/env python3
"""
Meta-analysis script for Recommendation Algorithms
Converted from: recommendation algorithms meta analysis.ipynb
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


def process_recommendation_algorithms_data(df_human, df_twin):
    """
    Study-specific data processing for recommendation algorithms.
    Creates derived variables and handles exclusions.
    """
    # in df_human, rename the second column called 1_Q16_1 as 1_Q16_1.1 if needed
    # locate all positions of columns named '1_Q16_1'
    dup_idxs = [i for i, name in enumerate(df_human.columns) if name == "1_Q16_1"]
    # if there are at least two, rename the second one
    if len(dup_idxs) >= 2:
        second_idx = dup_idxs[1]
        df_human.columns.values[second_idx] = "1_Q16_1.1"
        print("renamed duplicate columns in df_human")
    else:
        print("No duplicate found to rename.")

    # add new columns with relevant variables coded
    usage_map = {
        "usage_TikTok": "1_Q15",
        "usage_Netflix": "2_Q15",
        "algo_knowledge_TikTok": "1_Q16_4",
        "algo_knowledge_Netflix": "2_Q16_4",
        "strategization_type1_TikTok": "1_Q16_1.1",
        "strategization_type1_Netflix": "2_Q16_1.1",
    }
    for df in (df_human, df_twin):
        for new_col, src_col in usage_map.items():
            df[new_col] = df[src_col].fillna(0)

    for df in (df_human, df_twin):
        df["strategization_type2_TikTok"] = ((df["1_Q18_5"] == 1) | (df["1_Q18_7"] == 1)).astype(
            int
        )
    for df in (df_human, df_twin):
        df["strategization_type2_Netflix"] = ((df["2_Q18_5"] == 1) | (df["2_Q18_7"] == 1)).astype(
            int
        )

    for df in (df_human, df_twin):
        # TikTok: Q20_2.1 … Q20_6.1 plus Q20_7
        tiktok_cols = [f"1_Q20_{i}.1" for i in range(2, 7)] + ["1_Q20_7"]
        df["pref_control_TikTok"] = df[tiktok_cols].fillna(0).sum(axis=1)
        # Netflix: Q20_2.1 … Q20_6.1 plus Q20_7 (but with "2_" prefix)
        netflix_cols = [f"2_Q20_{i}.1" for i in range(2, 7)] + ["2_Q20_7"]
        df["pref_control_Netflix"] = df[netflix_cols].fillna(0).sum(axis=1)

    # list all of the new columns you've created
    new_cols = [
        "usage_TikTok",
        "usage_Netflix",
        "algo_knowledge_TikTok",
        "algo_knowledge_Netflix",
        "strategization_type1_TikTok",
        "strategization_type1_Netflix",
        "strategization_type2_TikTok",
        "strategization_type2_Netflix",
        "pref_control_TikTok",
        "pref_control_Netflix",
    ]

    # split out TikTok‑ vs. Netflix‑specific columns
    tik_tok_cols = [c for c in new_cols if "TikTok" in c]
    netflix_cols = [c for c in new_cols if "Netflix" in c]

    for df in (df_human, df_twin):
        if "exclusion" not in df.columns:
            df["exclusion"] = np.nan

    # if Q13 ≠ 5 (failed attention check) in either human or twin, null out *all* new cols for that TWIN_ID
    mask_q13 = (df_human["Q13"] != 5) | (df_twin["Q13"] != 5)
    for df in (df_human, df_twin):
        df.loc[mask_q13, new_cols] = np.nan
        df.loc[mask_q13, "exclusion"] = "failed attention check"

    # TikTok missing → mask_use6
    mask_use6 = df_human["Platform Use_6"].isna() | df_twin["Platform Use_6"].isna()
    for df in (df_human, df_twin):
        # null out the TikTok‐cols
        df.loc[mask_use6, tik_tok_cols] = np.nan
        # grab the existing text (or empty string)
        existing = df["exclusion"].fillna("")
        # build the new text: if there was already something, prepend "; "
        suffix = np.where(existing != "", existing + "; no TikTok usage", "no TikTok usage")
        # assign back, but only on the mask rows
        df.loc[mask_use6, "exclusion"] = suffix[mask_use6]

    # Netflix missing → mask_use4
    mask_use4 = df_human["Platform Use_4"].isna() | df_twin["Platform Use_4"].isna()
    for df in (df_human, df_twin):
        df.loc[mask_use4, netflix_cols] = np.nan
        existing = df["exclusion"].fillna("")
        suffix = np.where(existing != "", existing + "; no Netflix usage", "no Netflix usage")
        df.loc[mask_use4, "exclusion"] = suffix[mask_use4]

    return df_human, df_twin


def main():
    # Parse arguments
    parser = create_base_parser("Recommendation Algorithms")
    args = parser.parse_args()
    
    # Handle file discovery
    paths = handle_file_discovery(args)
    
    if args.verbose:
        print(f"Loading data from:")
        print(f"  Human: {paths['human_data']}")
        print(f"  Twin: {paths['twin_data']}")
        print(f"  Output: {paths['output_dir']}")

    # Study configuration
    study_name = "recommendation systems"
    
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

    # Load and process data
    df_human, df_twin = load_standard_data(paths['human_data'], paths['twin_data'])
    df_human, df_twin = process_recommendation_algorithms_data(df_human, df_twin)

    # Define study variables
    condition_vars = [""]  # No conditions
    DV_vars = [
        "usage_TikTok",
        "usage_Netflix",
        "algo_knowledge_TikTok",
        "algo_knowledge_Netflix",
        "strategization_type1_TikTok",
        "strategization_type1_Netflix",
        "strategization_type2_TikTok",
        "strategization_type2_Netflix",
        "pref_control_TikTok",
        "pref_control_Netflix",
    ]
    DV_vars_min = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    DV_vars_max = [5, 5, 1, 1, 1, 1, 1, 1, 6, 6]
    
    # Domain classifications
    DV_vars_social = [0] * 10
    DV_vars_cognitive = [0] * 10
    DV_vars_known = [0] * 10
    DV_vars_pref = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    DV_vars_stim = [0] * 10
    DV_vars_know = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    DV_vars_politics = [0] * 10

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

    # Create long format data
    df_long = make_long_format(df_human, df_twin, DV_vars, study_name, specification_name)
    print(df_long.head())

    # Save outputs
    save_standard_outputs(corr_df, df_long, paths['output_path'], args.verbose)
    print("done")

    return corr_df, df_long


if __name__ == "__main__":
    main()
