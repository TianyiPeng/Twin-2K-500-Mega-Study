#!/usr/bin/env python3
"""
Meta-analysis script for Targeting Fairness
Converted from: Targeting fairness meta analysis.ipynb
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


def main():
    # Parse arguments
    parser = create_base_parser("Targeting Fairness")
    args = parser.parse_args()
    
    # Handle file discovery
    paths = handle_file_discovery(args)
    
    if args.verbose:
        print(f"Loading data from:")
        print(f"  Human: {paths['human_data']}")
        print(f"  Twin: {paths['twin_data']}")
        print(f"  Output: {paths['output_dir']}")

    # Study configuration
    study_name = "Targeting fairness"
    
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

    # Define study variables
    condition_vars = ["Segment"]
    DV_vars = ["fair1"]
    DV_vars_min = [1]
    DV_vars_max = [9]
    
    # Domain classifications
    DV_vars_social = [1]
    DV_vars_cognitive = [0]
    DV_vars_known = [1]
    DV_vars_pref = [1]
    DV_vars_stim = [1]
    DV_vars_know = [0]
    DV_vars_politics = [1]

    # Load and prepare data
    df_human, df_twin = load_standard_data(paths['human_data'], paths['twin_data'])
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
