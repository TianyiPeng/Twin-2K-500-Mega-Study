#!/usr/bin/env python3
"""
Meta-analysis script for Affective Priming
Converted from: Affective Primes meta analysis.ipynb
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

# Import common modules
from common.args_parser import create_base_parser, handle_file_discovery
from common.data_loader import load_standard_data, merge_twin_data, prepare_data_for_analysis
from common.stats_analysis import compute_standard_stats
from common.results_processor import create_results_dict, make_long_format, save_standard_outputs
from common.variable_mapper import create_domain_maps, add_min_max_columns, check_condition_variable, build_min_max_maps
from parse_specification import extract_specification_from_path, parse_specification_name


def main():
    # Parse arguments
    args = create_base_parser("Affective Primes").parse_args()
    
    # Handle file discovery
    study_config = {
        'human_filename': 'consolidated_original_answers_values.csv',
        'twin_filename': 'consolidated_llm_values.csv',
        'human_at_study_level': False
    }
    paths = handle_file_discovery(args, study_config)
    
    output_path = paths['output_path']
    
    if args.verbose:
        print(f"Loading data from:")
        print(f"  Human: {paths['human_data']}")
        print(f"  Twin: {paths['twin_data']}")
        print(f"  Output: {paths['output_dir']}")
    
    # Load data
    study_name = "Affective Primes"
    
    # Extract specification name from path
    specification_name = extract_specification_from_path(args.results_dir)
    if specification_name:
        specification_type, ran_date = parse_specification_name(specification_name)
    else:
        # Fallback if path parsing fails
        specification_name = "unknown"
        specification_type = "unknown"
        ran_date = None
    
    # Load data files
    df_human, df_twin = load_standard_data(
        paths['human_data'], 
        paths['twin_data'],
        skiprows_human=[1, 2],
        skiprows_twin=[1, 2]
    )
    
    # STUDY-SPECIFIC DATA PROCESSING
    # Create composite measures in both human and twin dfs
    for df in (df_human, df_twin):
        # Gratitude manipulation check: average of g_manip_check_1–3
        df["gratitude_manipulation_check"] = df[
            ["g_manip_check_1", "g_manip_check_2", "g_manip_check_3"]
        ].mean(axis=1)

        # Elevation scale: average of elevation_1–6
        df["elevation_scale"] = df[[f"elevation_{i}" for i in range(1, 7)]].mean(axis=1)

        # Empathic concern measure: average of DV1_1 and DV1_2
        df["empathic_concern_measure"] = df[["DV1_1", "DV1_2"]].mean(axis=1)

        # Lack of control manipulation check: average of c_manip_check_1–3
        df["lack_of_control_manipulation_check"] = df[
            ["c_manip_check_1", "c_manip_check_2", "c_manip_check_3"]
        ].mean(axis=1)

        # Fatigue scale: reverse‐code items 3, 7, and 8 (8 – value), then average
        fat_items = [f"fatigue_{i}" for i in range(1, 10)]
        fat_rev = df[fat_items].copy()
        for i in (3, 7, 8):
            fat_rev[f"fatigue_{i}"] = 8 - fat_rev[f"fatigue_{i}"]
        df["fatigue_scale"] = fat_rev.mean(axis=1)

        # Desire for predictability measure: reverse‐code items 1, 4, and 5, then average 1–8
        desire_items = [f"desire_predict_{i}" for i in range(1, 9)]
        desire_rev = df[desire_items].copy()
        for i in (1, 4, 5):
            desire_rev[f"desire_predict_{i}"] = 8 - desire_rev[f"desire_predict_{i}"]
        df["desire_for_predictability_measure"] = desire_rev.mean(axis=1)

    for df in (df_human, df_twin):
        # Overall gratitude = mean of elevation_scale and empathic_concern_measure
        df["gratitude"] = df[["elevation_scale", "empathic_concern_measure"]].mean(axis=1)

        # Overall lack of control = mean of fatigue_scale and desire_for_predictability_measure
        df["lack of control"] = df[["fatigue_scale", "desire_for_predictability_measure"]].mean(
            axis=1
        )

    # Save processed data
    out_file = output_path / f"Affective Primes {specification_name} human data processed.csv"
    df_human.to_csv(out_file, index=False)
    out_file = output_path / f"Affective Primes {specification_name} twins data processed.csv"
    df_twin.to_csv(out_file, index=False)
    
    # Set index
    df_human = df_human.set_index("TWIN_ID")
    df_twin = df_twin.set_index("TWIN_ID")
    
    # define relevant columns:
    # condition variable names:
    condition_vars = ["cond"]
    cond_exists, cond, cond_h, cond_t = check_condition_variable(condition_vars)
    
    # DVs:
    DV_vars = [
        "gratitude",
        "lack of control",
        "gratitude_manipulation_check",
        "lack_of_control_manipulation_check",
    ]
    DV_vars_min = [1, 1, 1, 1]
    DV_vars_max = [7, 7, 7, 7]
    
    # Create domain maps
    domain_maps = create_domain_maps(
        DV_vars,
        social=[1, 1, 1, 1],      # domain=social?
        cognitive=[0, 0, 0, 0],   # domain=cognitive?
        known=[0, 0, 0, 0],       # replicating know human bias?
        pref=[0, 0, 0, 0],        # preference measure?
        stim=[1, 1, 1, 1],        # stimuli dependent?
        know=[0, 0, 0, 0],        # knowledge question?
        politics=[0, 0, 0, 0]     # political question?
    )
    
    # merging key
    merge_key = ["TWIN_ID"]
    
    # Merge on TWIN_ID
    df = merge_twin_data(df_human, df_twin, merge_key)
    
    # Debug: check columns after merge
    if args.verbose:
        print(f"Columns after merge: {sorted([c for c in df.columns if 'Q47' in c or 'Q49' in c or 'Q50' in c or 'Q52' in c])}")
    
    # Fix dtypes
    df = prepare_data_for_analysis(df, DV_vars)
    
    # build min/max maps
    min_map, max_map = build_min_max_maps(DV_vars, DV_vars_min, DV_vars_max)
    
    # add _min and _max columns
    df = add_min_max_columns(df, min_map, max_map)
    
    # Compute results
    results = []
    
    for var in DV_vars:
        # Compute all statistics using common function
        if cond_exists:
            stats = compute_standard_stats(df, var, True, cond_h, cond_t)
        else:
            stats = compute_standard_stats(df, var, False, None, None)
        
        # Create result dictionary
        spec_info = {
            'type': specification_type,
            'name': specification_name,
            'date': ran_date
        }
        
        result = create_results_dict(
            study_name, var, stats, domain_maps, spec_info
        )
        results.append(result)
    
    # results DataFrame
    corr_df = pd.DataFrame(results)
    print(corr_df)
    
    # Reset index before creating long format
    df_human.reset_index(inplace=True)
    df_twin.reset_index(inplace=True)
    
    # Create long format data
    df_long = make_long_format(df_human, df_twin, DV_vars, study_name, specification_name)
    print(df_long.head())
    
    # Save outputs
    save_standard_outputs(corr_df, df_long, output_path, args.verbose)
    
    print("done")
    
    return corr_df, df_long


if __name__ == "__main__":
    main()