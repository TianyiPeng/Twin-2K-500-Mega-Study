#!/usr/bin/env python3
"""
Meta-analysis script for Hiring Algorithms
Converted from: hiring algorithms meta analysis.ipynb
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


def compute_ranking_metrics(df_human, df_twin):
    """Compute ranking similarity metrics for hiring algorithms study"""
    from scipy.spatial.distance import cosine, euclidean
    from itertools import permutations
    
    # Extract ranking data
    ranking_cols = ['Q44', 'Q45', 'Q46', 'Q47']
    df_human_rankings = df_human[['TWIN_ID'] + ranking_cols].set_index('TWIN_ID')
    df_twin_rankings = df_twin[['TWIN_ID'] + ranking_cols].set_index('TWIN_ID')
    
    # Join on TWIN_ID
    df_rankings = df_human_rankings.join(df_twin_rankings, lsuffix='_human', rsuffix='_twin')
    
    # Compute metrics
    results = []
    for idx in df_rankings.index:
        human_ranks = df_rankings.loc[idx, [f'{col}_human' for col in ranking_cols]].values
        twin_ranks = df_rankings.loc[idx, [f'{col}_twin' for col in ranking_cols]].values
        
        # Skip if any NaN
        if pd.isna(human_ranks).any() or pd.isna(twin_ranks).any():
            continue
            
        # Compute similarity metrics
        cos_sim = 1 - cosine(human_ranks, twin_ranks)
        eucl_dist = euclidean(human_ranks, twin_ranks)
        
        # Check if same ranking
        same_ranking = np.array_equal(human_ranks, twin_ranks)
        
        results.append({
            'TWIN_ID': idx,
            'cosine_similarity': cos_sim,
            'euclidean_distance': eucl_dist,
            'same_ranking': int(same_ranking)
        })
    
    return pd.DataFrame(results)


def main():
    # Parse arguments
    args = create_base_parser("hiring algorithms").parse_args()
    
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
    study_name = "hiring algorithms"
    
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
    # ─── NEW 08/04/25: Recode Q9 and Q14 ─────────────────────────────────────────────────────
    # Mapping for Q9: 3→1, 4→2, 2→3, 1→4
    recode_Q9 = {3: 1, 4: 2, 2: 3, 1: 4}
    
    # Mapping for Q14: 1→1, 3→2, 4→3, 2→4
    recode_Q14 = {1: 1, 3: 2, 4: 3, 2: 4}
    
    for df in (df_human, df_twin):
        # ensure numeric before mapping (optional)
        df['Q9']  = pd.to_numeric(df['Q9'], errors='coerce').map(recode_Q9)
        df['Q14'] = pd.to_numeric(df['Q14'], errors='coerce').map(recode_Q14)
    # ────────────────────────────────────────────────────────
    
    # Create condition columns
    for df in (df_human, df_twin):
        # build the masks *on this* df
        conds = [
            df["FL_7_DO_AlgorithmicHiring"] == 1,
            df["FL_7_DO_HiringTeam"] == 1,
        ]
        # choices for each cond_job*
        # for cond_job1 & 3: algo if algohire, else team
        df["cond_job1"] = np.select(conds, ["algo", "team"], default="")
        df["cond_job3"] = np.select(conds, ["algo", "team"], default="")
        # for cond_job2 & 4: team if algohire, else algo
        df["cond_job2"] = np.select(conds, ["team", "algo"], default="")
        df["cond_job4"] = np.select(conds, ["team", "algo"], default="")
    
    # ratings for each phase 2 job:
    # mapping for each job under each flag
    ht_map = {1: "Q22", 2: "Q24", 3: "Q25", 4: "Q26"}
    algo_map = {1: "Q27", 2: "Q28", 3: "Q29", 4: "Q30"}
    
    for df in (df_human, df_twin):
        # initialize all new columns to NaN
        for i in range(1, 5):
            for k in range(1, 9):
                df[f"job{i}_item{k}"] = np.nan
        
        # fill in for HiringTeam
        mask_ht = df["FL_7_DO_HiringTeam"] == 1
        for i, base in ht_map.items():
            for k in range(1, 9):
                df.loc[mask_ht, f"job{i}_item{k}"] = df.loc[mask_ht, f"{base}_{k}"]
        
        # fill in for AlgorithmicHiring
        mask_algo = df["FL_7_DO_AlgorithmicHiring"] == 1
        for i, base in algo_map.items():
            for k in range(1, 9):
                df.loc[mask_algo, f"job{i}_item{k}"] = df.loc[mask_algo, f"{base}_{k}"]
    
    # Initialize results list
    results = []
    
    # PART 1: DVs without conditions
    # DVs:
    DV_vars_no_cond = ["Q6", "Q8", "Q9", "Q10", "Q13", "Q14", "Q15", "Q16"]
    DV_vars_min_no_cond = [1] * 8
    DV_vars_max_no_cond = [5, 5, 4, 5, 5, 4, 5, 5]
    
    # Create domain maps for no-condition DVs
    domain_maps_no_cond = create_domain_maps(
        DV_vars_no_cond,
        social=[1] * 8,      # domain=social?
        cognitive=[0] * 8,   # domain=cognitive?
        known=[0] * 8,       # replicating know human bias?
        pref=[1] * 8,        # preference measure?
        stim=[0] * 8,        # stimuli dependent?
        know=[0] * 8,        # knowledge question?
        politics=[0] * 8     # political question?
    )
    
    # merging key
    merge_key = ["TWIN_ID"]
    
    # Merge on TWIN_ID
    df = merge_twin_data(df_human, df_twin, merge_key)
    
    # Fix dtypes
    df = prepare_data_for_analysis(df, DV_vars_no_cond)
    
    # build min/max maps
    min_map, max_map = build_min_max_maps(DV_vars_no_cond, DV_vars_min_no_cond, DV_vars_max_no_cond)
    
    # add _min and _max columns
    df = add_min_max_columns(df, min_map, max_map)
    
    # Process DVs without conditions
    for var in DV_vars_no_cond:
        # Compute all statistics using common function
        stats = compute_standard_stats(df, var, False, None, None)
        
        # Create result dictionary
        spec_info = {
            'type': specification_type,
            'name': specification_name,
            'date': ran_date
        }
        
        result = create_results_dict(
            study_name, var, stats, domain_maps_no_cond, spec_info
        )
        results.append(result)
    
    # PART 2: DVs with job-specific conditions
    # different condition assignments for each DV (unique to this study)
    dv_to_cond = {f"job{i}_item{k}": f"cond_job{i}" for i in range(1, 5) for k in range(1, 9)}
    
    # DVs:
    DV_vars_with_cond = [f"job{i}_item{k}" for i in range(1, 5) for k in range(1, 9)]
    DV_vars_min_with_cond = [1] * 32
    DV_vars_max_with_cond = [7] * 32
    
    # Create domain maps for condition DVs
    domain_maps_with_cond = create_domain_maps(
        DV_vars_with_cond,
        social=[1] * 32,      # domain=social?
        cognitive=[0] * 32,   # domain=cognitive?
        known=[0] * 32,       # replicating know human bias?
        pref=[1] * 32,        # preference measure?
        stim=[1] * 32,        # stimuli dependent?
        know=[0] * 32,        # knowledge question?
        politics=[0] * 32     # political question?
    )
    
    # Prepare data for condition DVs
    df = prepare_data_for_analysis(df, DV_vars_with_cond)
    
    # build min/max maps
    min_map_cond, max_map_cond = build_min_max_maps(DV_vars_with_cond, DV_vars_min_with_cond, DV_vars_max_with_cond)
    
    # add _min and _max columns
    df = add_min_max_columns(df, min_map_cond, max_map_cond)
    
    # Process DVs with conditions
    for var in DV_vars_with_cond:
        # Look up the right condition for this DV
        cond = dv_to_cond[var]  # e.g. 'cond_job1'
        cond_h = f"{cond}_human"
        cond_t = f"{cond}_twin"
        
        # Compute all statistics using common function with specific condition variables
        stats = compute_standard_stats(df, var, True, cond_h, cond_t)
        
        # Create result dictionary
        spec_info = {
            'type': specification_type,
            'name': specification_name,
            'date': ran_date
        }
        
        result = create_results_dict(
            study_name, var, stats, domain_maps_with_cond, spec_info
        )
        results.append(result)
    
    # Combine all DVs for long format
    all_DV_vars = DV_vars_no_cond + DV_vars_with_cond
    
    # results DataFrame
    corr_df = pd.DataFrame(results)
    print(corr_df)
    
    # Create long format data
    df_long = make_long_format(df_human, df_twin, all_DV_vars, study_name, specification_name)
    print(df_long.head())
    
    # Save outputs
    save_standard_outputs(corr_df, df_long, output_path, args.verbose)
    
    # Compute and save ranking metrics (study-specific)
    # NOTE: Q44-Q47 not present in data, skip ranking metrics
    # if args.verbose:
    #     print("Computing ranking similarity metrics...")
    # ranking_metrics = compute_ranking_metrics(df_human, df_twin)
    # if not ranking_metrics.empty:
    #     ranking_file = output_path / "ranking_similarity_metrics.csv"
    #     ranking_metrics.to_csv(ranking_file, index=False)
    #     if args.verbose:
    #         print(f"Saved ranking metrics to {ranking_file}")
    #         print(f"Mean cosine similarity: {ranking_metrics['cosine_similarity'].mean():.3f}")
    #         print(f"Mean Euclidean distance: {ranking_metrics['euclidean_distance'].mean():.3f}")
    #         print(f"Proportion with same ranking: {ranking_metrics['same_ranking'].mean():.3f}")
    
    print("done")
    
    return corr_df, df_long


if __name__ == "__main__":
    main()