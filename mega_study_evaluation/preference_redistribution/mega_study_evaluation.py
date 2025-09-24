#!/usr/bin/env python3
"""
Meta-analysis script for Preference Redistribution
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


def apply_special_filtering(df_human, df_twin, verbose=False):
    """
    Apply study-specific filtering for response time and attention check.
    This is unique to preference_redistribution study.
    """
    # Response time filtering - remove observations with completion time < 50% of median
    median_human = df_human["Duration (in seconds)"].median()
    threshold_human = 0.5 * median_human
    median_twin = df_twin["Duration (in seconds)"].median()
    threshold_twin = 0.5 * median_twin
    
    # Identify which TWIN_IDs would be removed
    removed_human = df_human.loc[df_human["Duration (in seconds)"] < threshold_human, "TWIN_ID"]
    removed_twin = df_twin.loc[df_twin["Duration (in seconds)"] < threshold_twin, "TWIN_ID"]
    set_human = set(removed_human)
    set_twin = set(removed_twin)
    
    if verbose:
        print(f"Human-sample removals due to response time ({len(set_human)} IDs):\n", set_human)
        print(f"Twin-sample removals due to response time ({len(set_twin)} IDs):\n", set_twin)
        print("Sets equal? ", set_human == set_twin)
        print("In human only:", set_human - set_twin)
        print("In twin only: ", set_twin - set_human)
    
    # Actually filter out those rows
    df_human_filtered = df_human[df_human["Duration (in seconds)"] >= threshold_human].copy()
    df_twin_filtered = df_twin[df_twin["Duration (in seconds)"] >= threshold_twin].copy()
    
    if verbose:
        print("After filtering:")
        print(" df_human:", df_human.shape, "→", df_human_filtered.shape)
        print(" df_twin: ", df_twin.shape, "→", df_twin_filtered.shape)
    
    # Attention check filtering - remove those failing attention check
    # The attention check asks users to select "Don't know" (with curly quote)
    # Those who fail are those who did NOT select "Don't know"
    fail_human = df_human_filtered.loc[
        df_human_filtered["ATTENTION_CHECK"] != "Don’t know", "TWIN_ID"
    ]
    fail_twin = df_twin_filtered.loc[
        df_twin_filtered["ATTENTION_CHECK"] != "Don’t know", "TWIN_ID"
    ]
    
    set_fail_human = set(fail_human)
    set_fail_twin = set(fail_twin)
    
    if verbose:
        print("Failed attention in human (n={}):".format(len(set_fail_human)), set_fail_human)
        print("Failed attention in twin  (n={}):".format(len(set_fail_twin)), set_fail_twin)
    
    # Remove everyone who failed in the *human* sample, from both dfs
    ids_to_remove = set_fail_human
    df_human_clean = df_human_filtered.loc[~df_human_filtered["TWIN_ID"].isin(ids_to_remove)].copy()
    df_twin_clean = df_twin_filtered.loc[~df_twin_filtered["TWIN_ID"].isin(ids_to_remove)].copy()
    
    if verbose:
        print("\nAfter attention‐check filtering:")
        print(" df_human: {} → {}".format(df_human_filtered.shape, df_human_clean.shape))
        print(" df_twin:  {} → {}".format(df_twin_filtered.shape, df_twin_clean.shape))
    
    return df_human_clean, df_twin_clean


def create_redistribution_variables(df_human, df_twin, verbose=False):
    """
    Create study-specific redistribution variables using mapping dictionaries.
    This is unique to preference_redistribution study.
    """
    redistribution1_map = {
        "1 - I strongly agree the government should improve living standards": 5,
        "2": 4,
        "3 - I agree with both answers": 3,
        "4": 2,
        "5 - I strongly agree that people should take care of themselves": 1,
        # any other value (e.g. "Don't know") will become NaN
    }

    redistribution2_map = {
        "Very willing": 5,
        "Fairly willing": 4,
        "Neither willing nor unwilling": 3,
        "Fairly unwilling": 2,
        "Very unwilling": 1,
        # e.g. "Donâ€™t know/Can't choose" → NaN
    }

    fairness_map = {
        "Would take advantage of you": 3,
        "It depends": 2,
        "Would try to be fair": 1,
        # "Don't know" → NaN
    }

    workluck_map = {
        "Hard work most important": 1,
        "Hard work and luck are equally important": 2,
        "Luck or help from other people most important": 3,
        # "Don't know" → NaN
    }

    father_educ_map = {
        "Less than high school": 10,
        "High school graduate": 12,
        "Some college, no degree": 13,
        "Associate's degree": 14,
        "College graduate / some postgraduate": 16,
        "Postgraduate degree": 18,
    }

    trust_map = {
        "Can trust": 3,
        "Depends": 2,
        "Can't be too careful": 1,
        # "Don't know" → NaN
    }

    for df in (df_human, df_twin):
        df["Redistribution 1"] = df["GOVT_RESPONSIBILITY"].map(redistribution1_map)
        df["Redistribution 2"] = df["HLTHTAX"].map(redistribution2_map)
        df["Fairness"] = df["FAIR"].map(fairness_map)
        df["Work vs Luck"] = df["GETAHEAD"].map(workluck_map)
        df["Father education"] = df["FAEDUC"].map(father_educ_map)
        df["Trust"] = df["TRUST_GSS"].map(trust_map)

    # Check results if verbose
    if verbose:
        pairs = [
            ("GOVT_RESPONSIBILITY", "Redistribution 1"),
            ("HLTHTAX", "Redistribution 2"),
            ("FAIR", "Fairness"),
            ("GETAHEAD", "Work vs Luck"),
            ("FAEDUC", "Father education"),
            ("TRUST_GSS", "Trust"),
        ]

        for df_name, df in [("Human", df_human), ("Twin", df_twin)]:
            print(f"\n==== {df_name} sample ====\n")
            for orig, new in pairs:
                print(f"– {orig} vs {new} –")
                ct = pd.crosstab(df[orig], df[new], dropna=False)
                print(ct)
                print()

    return df_human, df_twin


def main():
    # Parse arguments
    parser = create_base_parser("Preference Redistribution")
    args = parser.parse_args()

    # Handle file discovery with study-specific configuration
    study_config = {
        'human_at_study_level': True,
        'human_filename': 'preferences for redistribution human data labels anonymized.csv',
        'twin_filename': 'consolidated_llm_labels.csv'
    }
    paths = handle_file_discovery(args, study_config)

    if args.verbose:
        print(f"Loading data from:")
        print(f"  Human: {paths['human_data']}")
        print(f"  Twin: {paths['twin_data']}")
        print(f"  Output: {paths['output_dir']}")

    # Load data
    df_human, df_twin = load_standard_data(paths['human_data'], paths['twin_data'])

    # Apply special filtering (unique to this study)
    df_human, df_twin = apply_special_filtering(df_human, df_twin, args.verbose)

    # Create redistribution variables (unique to this study)
    df_human, df_twin = create_redistribution_variables(df_human, df_twin, args.verbose)

    # Define study variables
    DV_vars = [
        "Redistribution 1",
        "Redistribution 2", 
        "Fairness",
        "Work vs Luck",
        "Father education",
        "Trust",
    ]
    DV_vars_min = [1, 1, 1, 1, 10, 1]
    DV_vars_max = [5, 5, 3, 3, 18, 3]

    # Domain classifications
    social = [1, 1, 1, 1, 0, 1]
    cognitive = [0] * 6
    known = [0] * 6
    pref = [1, 1, 0, 0, 0, 0]
    stim = [0] * 6
    know = [0] * 6
    politics = [1, 1, 0, 0, 0, 0]

    # Create domain maps
    domain_maps = create_domain_maps(DV_vars, social, cognitive, known, pref, stim, know, politics)

    # Check condition variables
    condition_vars = [""]
    cond_exists, cond, cond_h, cond_t = check_condition_variable(condition_vars)

    # Build min/max maps and merge data
    min_map, max_map = build_min_max_maps(DV_vars, DV_vars_min, DV_vars_max)
    df = merge_twin_data(df_human, df_twin)
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
            "Preferences for redistribution", var, stats_dict, 
            domain_maps, spec_info
        )
        results.append(result)

    # Create results DataFrame
    corr_df = pd.DataFrame(results)
    
    if args.verbose:
        print(corr_df)

    # Create long format data
    df_long = make_long_format(df_human, df_twin, DV_vars, "Preferences for redistribution", specification_name)

    # Save outputs
    meta_analysis_file, individual_file = save_standard_outputs(
        corr_df, df_long, paths['output_path'], args.verbose
    )

    return corr_df, df_long


if __name__ == "__main__":
    main()