#!/usr/bin/env python3
"""
Meta-analysis script for Idea Generation
Converted from: Idea Generation meta analysis.ipynb
"""

import re
import sys
import warnings
from itertools import permutations
from pathlib import Path

import gensim.downloader as api
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine, euclidean, pdist
from scipy.stats import f, norm, pearsonr, ttest_rel

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


def f_test(x, y):
    """Two-sample F-test for equality of variances."""
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    F = var_x / var_y
    df1 = len(x) - 1
    df2 = len(y) - 1
    # Two-tailed p-value
    p_value = 2 * min(f.cdf(F, df1, df2), 1 - f.cdf(F, df1, df2))
    return F, p_value


def process_idea_generation_data(df_human, df_twin, paths):
    """
    Study-specific data processing for idea generation.
    Includes Word2Vec analysis for DAT and SSPT metrics.
    """
    # Ensure TWIN_ID is numeric for both dataframes
    for df in [df_human, df_twin]:
        if "TWIN_ID" in df.columns:
            df["TWIN_ID"] = pd.to_numeric(df["TWIN_ID"], errors="coerce")
            df.dropna(subset=["TWIN_ID"], inplace=True)
            df["TWIN_ID"] = df["TWIN_ID"].astype(int)
    
    # Load creativity ratings if available
    df_human, df_twin = load_creativity_ratings(df_human, df_twin, paths)
    
    # Load wave scores if available
    df_human, df_twin = load_wave_scores(df_human, df_twin)
    
    # Load Word2Vec model and compute metrics
    print("Loading word2vec model... (this may take a while)")
    model = api.load("word2vec-google-news-300")
    print("Model loaded.")
    
    # Compute DAT performance
    compute_dat_performance(df_human, df_twin, model)
    
    # Compute SSPT metrics
    compute_sspt_metrics(df_human, df_twin, model)
    
    return df_human, df_twin


def load_creativity_ratings(df_human, df_twin, paths):
    """Load creativity rating files if available."""
    results_path = Path(paths['output_dir'])
    human_creativity_file = results_path / "idea generation human data creativity ratings.csv"
    twin_creativity_file = results_path / "idea generation twins data creativity ratings.csv"
    
    if human_creativity_file.exists() and twin_creativity_file.exists():
        print("Found creativity rating files - will include creativity metrics")
        creativity_human = pd.read_csv(human_creativity_file, header=0)
        creativity_twin = pd.read_csv(twin_creativity_file, header=0)
        
        creat_cols = [
            "TWIN_ID",
            "creativity_rating_byhuman",
            "creativity_rating_byhuman_partial", 
            "creativity_rating_byhuman_full",
            "creativity_rating_bytwins",
        ]
        
        df_human = df_human.merge(creativity_human[creat_cols], on="TWIN_ID", how="left")
        df_twin = df_twin.merge(creativity_twin[creat_cols], on="TWIN_ID", how="left")
        return df_human, df_twin
    else:
        print("Creativity rating files not found - will skip creativity metrics")
        return df_human, df_twin


def load_wave_scores(df_human, df_twin):
    """Load wave score files if available."""
    project_root = Path(__file__).parent.parent.parent
    wave_data_dir = project_root / "data" / "mega_persona_summary_csv"
    
    wave_files = [wave_data_dir / f"wave {i} scores.csv" for i in range(1, 4)]
    
    if all(f.exists() for f in wave_files):
        wave2_human = pd.read_csv(wave_files[1], header=0)
        creat_col = wave2_human[["TWIN_ID", "score_forwardflow"]]
        
        df_human = df_human.merge(creat_col, on="TWIN_ID", how="left")
        df_twin = df_twin.merge(creat_col, on="TWIN_ID", how="left")
        return df_human, df_twin
    else:
        print("Wave score files not found - will skip forward flow score metrics")
        return df_human, df_twin


def compute_dat_performance(df_human, df_twin, model):
    """Compute DAT performance using Word2Vec embeddings."""
    text_cols = [f"Q74_{i}" for i in range(1, 11)]
    
    def compute_dat_perf_from_cols(row):
        # Grab the 10 entries, lowercase and drop any NaN/empty
        tokens = [
            str(row[col]).lower().strip()
            for col in text_cols
            if pd.notna(row[col]) and str(row[col]).strip() != ""
        ]
        # Look up embeddings (skip tokens not in vocab)
        vecs = [model[w] for w in tokens if w in model.key_to_index]
        # Must have at least 7 embeddings
        if len(vecs) < 7:
            return np.nan
        # Stack the first 7 into a matrix
        mat = np.vstack(vecs[:7])
        # Compute all pairwise cosine distances and return the mean
        dists = pdist(mat, metric="cosine")
        return dists.mean()
    
    # Apply to both dataframes
    df_human["DAT_perf"] = df_human[text_cols].apply(compute_dat_perf_from_cols, axis=1)
    df_twin["DAT_perf"] = df_twin[text_cols].apply(compute_dat_perf_from_cols, axis=1)


def compute_sspt_metrics(df_human, df_twin, model):
    """Compute SSPT metrics using Word2Vec embeddings."""
    def compute_circuitousness_euclidean(words, model):
        """Euclidean-based circuitousness over 5 word embeddings."""
        if len(words) != len(set(words)):
            return np.nan
        embs = []
        for w in words:
            toks = [t for t in str(w).lower().split() if t]
            vecs = [model[t] for t in toks if t in model.key_to_index]
            if not vecs:
                return np.nan
            embs.append(np.mean(vecs, axis=0))
        if len(embs) != 5:
            return np.nan
        # Original path
        orig = sum(euclidean(embs[i], embs[i + 1]) for i in range(4))
        best = np.inf
        for perm in permutations(embs[1:4]):
            path = [embs[0]] + list(perm) + [embs[4]]
            length = sum(euclidean(path[i], path[i + 1]) for i in range(4))
            best = min(best, length)
        return orig / best if best > 0 else np.nan
    
    def compute_circuitousness_cosine(words, model):
        """Cosine-based circuitousness over 5 word embeddings."""
        if len(words) != len(set(words)):
            return np.nan
        embs = []
        for w in words:
            toks = [t for t in str(w).lower().split() if t]
            vecs = [model[t] for t in toks if t in model.key_to_index]
            if not vecs:
                return np.nan
            embs.append(np.mean(vecs, axis=0))
        if len(embs) != 5:
            return np.nan
        orig = sum(cosine(embs[i], embs[i + 1]) for i in range(4))
        best = np.inf
        for perm in permutations(embs[1:4]):
            path = [embs[0]] + list(perm) + [embs[4]]
            length = sum(cosine(path[i], path[i + 1]) for i in range(4))
            best = min(best, length)
        return orig / best if best > 0 else np.nan
    
    def add_sspt_all(df, model):
        prefixes = [1, 2, 4, 5, 7]
        eu_cols, co_cols = [], []
        
        # Compute raw circuitousness for each metric & task
        for p in prefixes:
            cols = [f"{p}_Q30_{i}" for i in range(1, 6)]
            df[cols] = df[cols].applymap(lambda x: str(x).lower())
            
            e_col = f"circuitousness_task{p}_euclid"
            c_col = f"circuitousness_task{p}_cosine"
            eu_cols.append(e_col)
            co_cols.append(c_col)
            
            df[e_col] = df[cols].apply(
                lambda r: compute_circuitousness_euclidean(r.tolist(), model), axis=1
            )
            df[c_col] = df[cols].apply(
                lambda r: compute_circuitousness_cosine(r.tolist(), model), axis=1
            )
        
        # Raw SSPT
        df["SSPT"] = df[eu_cols].mean(axis=1)
        df["SSPT_cosine"] = df[co_cols].mean(axis=1)
        
        # Standardize per task
        stats_eu = df[eu_cols].agg(["mean", "std"])
        stats_co = df[co_cols].agg(["mean", "std"])
        
        z_eu, z_co = [], []
        for col in eu_cols:
            zc = f"{col}_z"
            df[zc] = (df[col] - stats_eu.loc["mean", col]) / stats_eu.loc["std", col]
            z_eu.append(zc)
        for col in co_cols:
            zc = f"{col}_z"
            df[zc] = (df[col] - stats_co.loc["mean", col]) / stats_co.loc["std", col]
            z_co.append(zc)
        
        # Standardized SSPT
        df["SSPT_standardized"] = df[z_eu].mean(axis=1)
        df["SSPT_cosine_standardized"] = df[z_co].mean(axis=1)
    
    # Apply to both dataframes
    add_sspt_all(df_human, model)
    add_sspt_all(df_twin, model)


def main():
    # Parse arguments - use special configuration for this study
    parser = create_base_parser("Idea Generation")
    args = parser.parse_args()
    
    # Special file handling for idea generation
    study_config = {
        'human_at_study_level': True,
        'human_filename': 'idea generation human data labels anonymized.csv',
        'twin_filename': 'consolidated_llm_labels.csv',
        'skiprows_twin': [1]  # Different skiprows for twin data
    }
    
    # Handle file discovery
    paths = handle_file_discovery(args, study_config)
    
    if args.verbose:
        print(f"Loading data from:")
        print(f"  Human: {paths['human_data']}")
        print(f"  Twin: {paths['twin_data']}")
        print(f"  Output: {paths['output_dir']}")

    # Study configuration
    study_name = "idea generation"
    
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
    df_human, df_twin = load_standard_data(paths['human_data'], paths['twin_data'], 
                                          skiprows_twin=study_config['skiprows_twin'])
    df_human, df_twin = process_idea_generation_data(df_human, df_twin, paths)

    # Define study variables  
    condition_vars = [""]  # No conditions
    # Only include the core variables that are in the test output
    DV_vars = ["DAT_perf", "SSPT"]
    DV_vars_min = [np.nan, np.nan]  # Use NaN for unbounded variables like backup
    DV_vars_max = [np.nan, np.nan]
    
    # Add only the main creativity variable if available
    if "creativity_rating_byhuman" in df_human.columns:
        DV_vars.append("creativity_rating_byhuman")
        DV_vars_min.append(1)  # creativity rating has explicit 1-5 range
        DV_vars_max.append(5)
    
    # Set n_vars after determining final DV_vars
    n_vars = len(DV_vars)
    
    # Domain classifications
    DV_vars_social = [0] * n_vars
    DV_vars_cognitive = [1] * n_vars  # All are cognitive measures
    DV_vars_known = [0] * n_vars
    DV_vars_pref = [0] * n_vars
    DV_vars_stim = [0] * n_vars  # Not stimuli dependent
    DV_vars_know = [0] * n_vars
    DV_vars_politics = [0] * n_vars

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