"""Common results processing utilities for mega study evaluation scripts."""

import pandas as pd
import numpy as np


def create_results_dict(study_name, var_name, stats_dict, domain_maps, spec_info, **kwargs):
    """
    Create standard results dictionary.
    
    Args:
        study_name: Name of the study
        var_name: Variable name
        stats_dict: Dictionary with all statistics (from compute_standard_stats)
        domain_maps: Domain classification mappings
        spec_info: Dictionary with specification information
            - 'type': specification type (e.g., 'full_persona_without_reasoning')
            - 'name': full specification name
            - 'date': ran date
        **kwargs: Additional study-specific fields
    
    Returns:
        Dictionary ready for DataFrame
    """
    # Start with basic fields (matching original script structure)
    result = {
        "study name": study_name,
        "persona specification": spec_info['type'],  # Use specification type from original scripts
        "specification_name_full": spec_info['name'],  # Full specification name
        "ran_date": spec_info['date'],  # Date from specification
        "variable name": var_name,
    }
    
    # Add correlation and related stats
    result.update({
        "correlation between the responses from humans vs. their twins": stats_dict.get('r', np.nan),
        "CI_lower": stats_dict.get('CI_lower', np.nan),
        "CI_upper": stats_dict.get('CI_upper', np.nan),
        "z-score for correlation between humans vs. their twins": stats_dict.get('z_score', np.nan),
        "accuracy between humans vs. their twins": stats_dict.get('accuracy', np.nan),
    })
    
    # Add means and test statistics
    result.update({
        "mean_human": stats_dict.get('mean_human', np.nan),
        "mean_twin": stats_dict.get('mean_twin', np.nan),
        "paired t-test t-stat": stats_dict.get('paired_t_stat', np.nan),
        "paired t-test p-value": stats_dict.get('paired_p_value', np.nan),
        "std_human": stats_dict.get('std_human', np.nan),
        "std_twin": stats_dict.get('std_twin', np.nan),
        "variance test F-stat": stats_dict.get('variance_f_stat', np.nan),
        "variance test p-value": stats_dict.get('variance_p_value', np.nan),
    })
    
    # Add effect sizes
    result.update({
        "effect size based on human": stats_dict.get('effect_size_human', np.nan),
        "effect size based on twin": stats_dict.get('effect_size_twin', np.nan),
    })
    
    # Add domain classifications
    for domain_type in ['social', 'cognitive', 'known', 'pref', 'stim', 'know', 'politics']:
        # Determine the key format
        if domain_type == 'social':
            key = "domain=social?"
        elif domain_type == 'cognitive':
            key = "domain=cognitive?"
        elif domain_type == 'known':
            key = "replicating know human bias?"
        elif domain_type == 'pref':
            key = "preference measure?"
        elif domain_type == 'stim':
            key = "stimuli dependent?"
        elif domain_type == 'know':
            key = "knowledge question?"
        elif domain_type == 'politics':
            key = "political question?"
        
        # Get the map for this domain
        map_key = f"DV_vars_{domain_type}_map"
        if map_key in domain_maps:
            result[key] = domain_maps[map_key].get(var_name, np.nan)
        else:
            result[key] = np.nan
    
    # Add sample size
    result["sample size"] = stats_dict.get('sample_size', stats_dict.get('n', np.nan))
    
    # Add any study-specific fields
    result.update(kwargs)
    
    return result


def make_long_format(df_human, df_twin, DV_vars, study_name, specification_name):
    """
    Standard long format transformation.
    
    Args:
        df_human: Human data DataFrame
        df_twin: Twin data DataFrame
        DV_vars: List of dependent variables
        study_name: Name of the study
        specification_name: Specification name
    
    Returns:
        DataFrame in long format
    """
    def make_long(df, respondent_type):
        # Ensure TWIN_ID is available
        if 'TWIN_ID' not in df.columns:
            # If TWIN_ID is the index, reset it
            if df.index.name == 'TWIN_ID':
                df = df.reset_index()
            else:
                raise ValueError("TWIN_ID not found in DataFrame")
        
        # Select only columns that exist
        available_vars = [var for var in DV_vars if var in df.columns]
        if not available_vars:
            raise ValueError(f"None of the DV_vars {DV_vars} found in DataFrame columns")
        
        # Melt the DataFrame
        long = df[["TWIN_ID"] + available_vars].melt(
            id_vars="TWIN_ID", 
            value_vars=available_vars, 
            var_name="variable_name", 
            value_name="value"
        )
        
        long["respondent_type"] = respondent_type
        long["study_name"] = study_name
        long["specification_name"] = specification_name
        
        return long
    
    # Create long format for each
    long_h = make_long(df_human, "human")
    long_t = make_long(df_twin, "twin")
    
    # Stack them
    df_long = pd.concat([long_h, long_t], ignore_index=True)
    
    return df_long


def save_standard_outputs(corr_df, df_long, output_path, verbose=False):
    """
    Save standard output files.
    
    Args:
        corr_df: Correlation DataFrame
        df_long: Long format DataFrame
        output_path: Path object for output directory
        verbose: Whether to print verbose output
    
    Returns:
        Tuple of (meta_analysis_file, individual_file)
    """
    # Save meta-analysis results
    meta_analysis_file = output_path / "meta analysis.csv"
    corr_df.to_csv(meta_analysis_file, index=False)
    
    # Save individual-level data
    individual_file = output_path / "meta analysis individual level.csv"
    df_long.to_csv(individual_file, index=False)
    
    if verbose:
        print(f"\nSaved meta-analysis results to: {meta_analysis_file}")
        print(f"Number of variables analyzed: {len(corr_df)}")
        if "correlation between the responses from humans vs. their twins" in corr_df.columns:
            mean_corr = corr_df[
                "correlation between the responses from humans vs. their twins"
            ].mean()
            print(f"Mean correlation: {mean_corr:.3f}")
        print(f"\nSaved individual-level data to: {individual_file}")
        print(f"Number of individual observations: {len(df_long)}")
        print("\nAnalysis complete!")
    
    return meta_analysis_file, individual_file