#!/usr/bin/env python3
"""
Meta-analysis script for Infotainment
Converted from: infotainment news meta analysis.ipynb
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
    args = create_base_parser("Infotainment News Sharing").parse_args()
    
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
    
    # Extract specification name from path
    specification_name = extract_specification_from_path(args.results_dir)
    if specification_name:
        specification_type, ran_date = parse_specification_name(specification_name)
    else:
        # Fallback if path parsing fails
        specification_name = "default persona"
        specification_type = "default persona"
        ran_date = None

    # Load data
    study_name = "infotainment news sharing"
    
    # Load data files
    df_human, df_twin = load_standard_data(
        paths['human_data'], 
        paths['twin_data'],
        skiprows_human=[1, 2],
        skiprows_twin=[1, 2]
    )


    # define relevant columns:
    # condition variable names:
    condition_vars = [""]
    cond_exists, cond, cond_h, cond_t = check_condition_variable(condition_vars)


    # DVs:
    DV_vars = [f"Profile{i}" for i in range(1, 19)]
    DV_vars_min = [1] * 18
    DV_vars_max = [8] * 18
    
    # Create domain maps
    domain_maps = create_domain_maps(
        DV_vars,
        social=[1] * 18,      # domain=social?
        cognitive=[0] * 18,   # domain=cognitive?
        known=[0] * 18,       # replicating know human bias?
        pref=[1] * 18,        # preference measure?
        stim=[0] * 18,        # stimuli dependent?
        know=[0] * 18,        # knowledge question?
        politics=[0] * 18     # political question?
    )


    # merging key
    merge_key = ["TWIN_ID"]
    
    # Merge on TWIN_ID
    df = merge_twin_data(df_human, df_twin, merge_key)
    
    print("Merged df columns:", df.columns.tolist())
    
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
    
    # Create long format data - handle non-NaN filtering
    df_long = make_long_format(df_human, df_twin, DV_vars, study_name, specification_name)
    df_long = df_long[df_long["value"].notna()]
    print(df_long.head())
    
    # Save outputs
    save_standard_outputs(corr_df, df_long, output_path, args.verbose)
    
    print("done")
    
    return corr_df, df_long


if __name__ == "__main__":
    main()
