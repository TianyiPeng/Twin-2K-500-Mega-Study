#!/usr/bin/env python3
"""
Meta-analysis script for Privacy
Converted from: Privacy Preferences meta analysis.ipynb
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
    args = create_base_parser("Privacy").parse_args()
    
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
    study_name = "Privacy"
    
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
    # In df_human, rename the second column called Q2_1 as Q2_1.1 if needed
    # locate all positions of columns named 'Q2_1'
    dup_idxs = [i for i, name in enumerate(df_human.columns) if name == "Q2_1"]
    # if there are at least two, rename the second one
    if len(dup_idxs) >= 2:
        second_idx = dup_idxs[1]
        df_human.columns.values[second_idx] = "Q2_1.1"
        print("renamed duplicate 'Q2_1'")
    else:
        print("No duplicate 'Q2_1' found to rename.")
    
    # create new variable PPV
    # Define a regex to pick up exactly Q11_1, Q138_1, Q140_1, Q142_1, Q2_1 and Q2_1.1 columns
    ppv_pattern = r"^(?:Q11_1|Q138_1|Q140_1|Q142_1|Q2_1|Q2_1\.1)$"
    
    for df in (df_human, df_twin):
        # 1) identify the six PPV columns (this will pick up both copies of Q2_1 if they exist)
        ppv_cols = df.filter(regex=ppv_pattern).columns.tolist()
        if len(ppv_cols) != 6:
            raise ValueError(f"Expected 6 PPV columns, found {len(ppv_cols)}: {ppv_cols}")
        
        # 2) check that exactly one is non-NaN per row
        non_na_counts = df[ppv_cols].notna().sum(axis=1)
        bad = non_na_counts != 1
        if bad.any():
            # report which rows violate the rule
            idx = df.index[bad].tolist()
            counts = non_na_counts[bad].tolist()
            raise ValueError(
                f"Rows {idx} have non-NaN counts {counts} (must be exactly 1) among {ppv_cols}"
            )
        
        # 3) create PPV as the sum (only the single non-NaN will contribute)
        df["PPV"] = df[ppv_cols].sum(axis=1, skipna=True)
    
    # define relevant columns:
    # condition variable names:
    condition_vars = ["Group"]
    cond_exists, cond, cond_h, cond_t = check_condition_variable(condition_vars)
    
    # DVs:
    DV_vars = ["PPV"]
    DV_vars_min = [1]
    DV_vars_max = [7]
    
    # Create domain maps
    domain_maps = create_domain_maps(
        DV_vars,
        social=[1],      # domain=social?
        cognitive=[0],   # domain=cognitive?
        known=[0],       # replicating know human bias?
        pref=[1],        # preference measure?
        stim=[1],        # stimuli dependent?
        know=[0],        # knowledge question?
        politics=[0]     # political question?
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
        stats = compute_standard_stats(df, var, cond_exists, cond_h, cond_t)
        
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
    
    # Create long format data
    df_long = make_long_format(df_human, df_twin, DV_vars, study_name, specification_name)
    print(df_long.head())
    
    # Save outputs
    save_standard_outputs(corr_df, df_long, output_path, args.verbose)
    
    print("done")
    
    return corr_df, df_long


if __name__ == "__main__":
    main()