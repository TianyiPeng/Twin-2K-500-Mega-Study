"""Common statistical analysis functions for mega study evaluation scripts."""

import numpy as np
import pandas as pd
from scipy.stats import f, norm, pearsonr, ttest_rel


def calculate_correlation_stats(pair_data, col_h, col_t, min_n=4):
    """
    Calculate correlation, CI, and z-score.
    Preserves exact calculation from original scripts.
    
    Args:
        pair_data: DataFrame with paired observations
        col_h: Human column name
        col_t: Twin column name
        min_n: Minimum sample size required for calculation
    
    Returns:
        dict with r, CI_lower, CI_upper, z_score, n
    """
    n = len(pair_data)
    
    if n >= min_n:
        r, _ = pearsonr(pair_data[col_h], pair_data[col_t])
        z_f = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = norm.ppf(0.975)
        lo_z, hi_z = z_f - z_crit * se, z_f + z_crit * se
        lo_r, hi_r = np.tanh(lo_z), np.tanh(hi_z)
        z_score = z_f / se
        
        return {
            'r': r,
            'CI_lower': lo_r,
            'CI_upper': hi_r,
            'z_score': z_score,
            'n': n
        }
    else:
        return {
            'r': np.nan,
            'CI_lower': np.nan,
            'CI_upper': np.nan,
            'z_score': np.nan,
            'n': n
        }


def calculate_accuracy(pair_data, col_h, col_t, min_val, max_val):
    """
    Standard accuracy calculation: 1 - MAD/range.
    
    Args:
        pair_data: DataFrame with paired observations
        col_h: Human column name
        col_t: Twin column name
        min_val: Minimum value for range
        max_val: Maximum value for range
    
    Returns:
        Accuracy score or np.nan
    """
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return np.nan
    else:
        abs_diff = np.abs(pair_data[col_h] - pair_data[col_t])
        mean_abs_diff = abs_diff.mean()
        return 1 - mean_abs_diff / (max_val - min_val)


def perform_paired_tests(pair_data, col_h, col_t):
    """
    Paired t-test and F-test calculations.
    
    Args:
        pair_data: DataFrame with paired observations
        col_h: Human column name
        col_t: Twin column name
    
    Returns:
        dict with test statistics and p-values
    """
    n = len(pair_data)
    
    # Means
    mean_h = pair_data[col_h].mean()
    mean_t = pair_data[col_t].mean()
    
    # Standard deviations
    std_h = pair_data[col_h].std(ddof=1)
    std_t = pair_data[col_t].std(ddof=1)
    
    # Paired t-test
    if n > 1:
        t_stat, p_val = ttest_rel(pair_data[col_h], pair_data[col_t])
    else:
        t_stat, p_val = np.nan, np.nan
    
    # F-test for equal variances
    if n > 1 and std_t > 0:
        df1 = df2 = n - 1
        f_stat = (std_h**2 / std_t**2)
        # Two-tailed p-value
        p_f = 2 * min(f.cdf(f_stat, df1, df2), 1 - f.cdf(f_stat, df1, df2))
    else:
        f_stat = np.nan
        p_f = np.nan
    
    return {
        'mean_human': mean_h,
        'mean_twin': mean_t,
        'paired_t_stat': t_stat,
        'paired_p_value': p_val,
        'std_human': std_h,
        'std_twin': std_t,
        'variance_f_stat': f_stat,
        'variance_p_value': p_f
    }


def calculate_effect_sizes(pair_data, col_h, col_t, cond_h=None, cond_t=None, cond_exists=False):
    """
    Cohen's d calculations for conditions.
    
    Args:
        pair_data: DataFrame with paired observations
        col_h: Human column name
        col_t: Twin column name
        cond_h: Condition column for humans (optional)
        cond_t: Condition column for twins (optional)
        cond_exists: Whether conditions exist
    
    Returns:
        dict with d_human and d_twin
    """
    d_human = np.nan
    d_twin = np.nan
    
    if cond_exists and cond_h and cond_t and len(pair_data) > 3:
        # For humans
        if cond_h in pair_data.columns:
            levels_h = pair_data[cond_h].unique()
            if len(levels_h) == 2:
                g1 = pair_data.loc[pair_data[cond_h] == levels_h[0], col_h]
                g2 = pair_data.loc[pair_data[cond_h] == levels_h[1], col_h]
                n1, n2 = len(g1), len(g2)
                if n1 > 0 and n2 > 0:
                    # Pooled standard deviation
                    s_pool = np.sqrt(
                        ((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2)
                    )
                    d_human = (g1.mean() - g2.mean()) / s_pool if s_pool > 0 else np.nan
        
        # For twins
        if cond_t in pair_data.columns:
            levels_t = pair_data[cond_t].unique()
            if len(levels_t) == 2:
                g1 = pair_data.loc[pair_data[cond_t] == levels_t[0], col_t]
                g2 = pair_data.loc[pair_data[cond_t] == levels_t[1], col_t]
                n1, n2 = len(g1), len(g2)
                if n1 > 0 and n2 > 0:
                    s_pool = np.sqrt(
                        ((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2)
                    )
                    d_twin = (g1.mean() - g2.mean()) / s_pool if s_pool > 0 else np.nan
    
    return {
        'effect_size_human': d_human,
        'effect_size_twin': d_twin
    }


def compute_standard_stats(df, var, cond_exists=False, cond_h=None, cond_t=None, min_val=None, max_val=None):
    """
    Compute all standard statistics for a variable.
    This is a convenience function that combines all the individual stat functions.
    
    Args:
        df: Merged DataFrame
        var: Variable name
        cond_exists: Whether conditions exist
        cond_h: Condition column for humans
        cond_t: Condition column for twins
        min_val: Minimum value for accuracy calculation
        max_val: Maximum value for accuracy calculation
    
    Returns:
        Dictionary with all statistics
    """
    col_h = f"{var}_human"
    col_t = f"{var}_twin"
    min_col = f"{var}_min" if min_val is None else None
    max_col = f"{var}_max" if max_val is None else None
    
    # Build columns list
    cols = [col_h, col_t]
    if cond_exists and cond_h and cond_t:
        cols.extend([cond_h, cond_t])
    if min_col and min_col in df.columns:
        cols.append(min_col)
    if max_col and max_col in df.columns:
        cols.append(max_col)
    
    # Get paired data
    pair = df[cols].dropna(subset=[col_h, col_t])
    
    # Get min/max values
    if min_val is None and min_col and min_col in pair.columns and len(pair) > 0:
        min_val = pair[min_col].iloc[0]
    if max_val is None and max_col and max_col in pair.columns and len(pair) > 0:
        max_val = pair[max_col].iloc[0]
    
    n = len(pair)
    
    # Calculate all statistics
    results = {}
    
    # Correlation stats
    corr_stats = calculate_correlation_stats(pair, col_h, col_t)
    results.update(corr_stats)
    
    # Accuracy
    results['accuracy'] = calculate_accuracy(pair, col_h, col_t, min_val, max_val)
    
    # Paired tests
    test_stats = perform_paired_tests(pair, col_h, col_t)
    results.update(test_stats)
    
    # Effect sizes
    effect_stats = calculate_effect_sizes(pair, col_h, col_t, cond_h, cond_t, cond_exists)
    results.update(effect_stats)
    
    # Sample size
    results['sample_size'] = n
    
    return results