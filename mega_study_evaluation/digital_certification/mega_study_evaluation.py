#!/usr/bin/env python3
"""
Meta-analysis script for Digital Certification
Converted from: Digital Certifications for Luxury Consumption meta analysis.ipynb
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
    args = create_base_parser("Digital Certificates for Luxury Consumption").parse_args()
    
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
    study_name = "Digital certificates for luxury consumption"
    
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
    # DVs:
    df_human["DV1"] = df_human[["DV1_1", "DV1_2", "DV1_3"]].mean(axis=1)
    df_twin["DV1"] = df_twin[["DV1_1", "DV1_2", "DV1_3"]].mean(axis=1)
    # Add small epsilon to handle zero WTP values
    epsilon = 1e-10
    df_human["log_WTP"] = np.log(epsilon + df_human["WTP"])
    df_twin["log_WTP"] = np.log(epsilon + df_twin["WTP"])
    
    # Set index
    df_human = df_human.set_index("TWIN_ID")
    df_twin = df_twin.set_index("TWIN_ID")
    
    # define relevant columns:
    # condition variable names:
    condition_vars = ["item"]
    cond_exists, cond, cond_h, cond_t = check_condition_variable(condition_vars)
    
    # DVs:
    DV_vars = ["DV1", "log_WTP"]
    DV_vars_min = [3, np.nan]
    DV_vars_max = [21, np.nan]
    
    # Create domain maps
    domain_maps = create_domain_maps(
        DV_vars,
        social=[1, 0],      # domain=social?
        cognitive=[0, 0],   # domain=cognitive?
        known=[0, 0],       # replicating know human bias?
        pref=[0, 1],        # preference measure?
        stim=[1, 1],        # stimuli dependent?
        know=[0, 0],        # knowledge question?
        politics=[0, 0]     # political question?
    )
    
    # merging key
    merge_key = ["TWIN_ID"]
    
    # Merge on TWIN_ID
    df = merge_twin_data(df_human, df_twin, merge_key)
    
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