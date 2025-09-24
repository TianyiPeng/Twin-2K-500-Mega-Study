#!/usr/bin/env python3
"""
Demonstrate better accuracy metrics that are more discriminative than current accuracy.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')


def calculate_rank_correlation_accuracy(human_vec, twin_vec):
    """Calculate accuracy based on rank correlation (Spearman)."""
    if len(human_vec) < 4:
        return np.nan
    try:
        rho, _ = spearmanr(human_vec, twin_vec)
        return abs(rho)  # Use absolute value so higher is better
    except:
        return np.nan


def calculate_percentile_accuracy(human_vec, twin_vec):
    """Calculate accuracy based on percentile rank similarity."""
    if len(human_vec) < 4:
        return np.nan
    
    # Convert to percentile ranks
    from scipy.stats import rankdata
    human_ranks = rankdata(human_vec, method='average') / len(human_vec)
    twin_ranks = rankdata(twin_vec, method='average') / len(twin_vec)
    
    # Calculate mean absolute difference in percentile ranks
    mad_percentiles = np.mean(np.abs(human_ranks - twin_ranks))
    return 1 - mad_percentiles  # Higher is better


def calculate_directional_accuracy(human_vec, twin_vec):
    """Calculate accuracy based on directional agreement relative to median."""
    if len(human_vec) < 4:
        return np.nan
    
    human_median = np.median(human_vec)
    twin_median = np.median(twin_vec)
    
    # Direction relative to median
    human_direction = human_vec > human_median
    twin_direction = twin_vec > twin_median
    
    # Proportion of agreements
    return np.mean(human_direction == twin_direction)


def calculate_distribution_similarity(human_vec, twin_vec):
    """Calculate distribution similarity using Wasserstein distance."""
    if len(human_vec) < 4:
        return np.nan
    
    try:
        # Normalize to [0,1] range for fair comparison
        if np.std(human_vec) > 0 and np.std(twin_vec) > 0:
            h_norm = (human_vec - np.min(human_vec)) / (np.max(human_vec) - np.min(human_vec))
            t_norm = (twin_vec - np.min(twin_vec)) / (np.max(twin_vec) - np.min(twin_vec))
            
            # Wasserstein distance (lower is better, so we return 1 - distance)
            distance = wasserstein_distance(h_norm, t_norm)
            return max(0, 1 - distance)  # Ensure non-negative
        else:
            return 0.0
    except:
        return np.nan


def calculate_weighted_accuracy(human_vec, twin_vec):
    """Calculate accuracy weighted by difficulty (human variance)."""
    if len(human_vec) < 4:
        return np.nan
    
    # Weight by inverse of human variance (harder questions get more weight)
    human_var = np.var(human_vec)
    if human_var == 0:
        return 1.0  # Perfect accuracy if no variance
    
    # Standard accuracy calculation
    if np.max(human_vec) > np.min(human_vec):
        mad = np.mean(np.abs(human_vec - twin_vec))
        range_val = np.max(human_vec) - np.min(human_vec)
        accuracy = 1 - mad / range_val
        
        # Weight by difficulty (higher variance = more weight)
        weight = human_var / (human_var + 1)  # Normalize weight
        return accuracy * weight + (1 - weight) * 0.5  # Blend with neutral score
    else:
        return 1.0


def process_individual_file_with_better_metrics(file_path):
    """Process individual level file with better accuracy metrics."""
    try:
        df = pd.read_csv(file_path)
    except:
        return []
    
    if len(df) == 0:
        return []
    
    study_name = df['study_name'].iloc[0]
    specification_name = df['specification_name'].iloc[0]
    variables = df['variable_name'].unique()
    
    results = []
    
    for var in variables:
        var_data = df[df['variable_name'] == var]
        
        # Get human and twin vectors
        human_data = var_data[var_data['respondent_type'] == 'human']
        twin_data = var_data[var_data['respondent_type'] == 'twin']
        
        # Merge on TWIN_ID
        merged = pd.merge(
            human_data[['TWIN_ID', 'value']], 
            twin_data[['TWIN_ID', 'value']], 
            on='TWIN_ID', 
            suffixes=('_human', '_twin')
        )
        
        if len(merged) == 0:
            continue
            
        # Remove NaN values
        mask = ~(merged['value_human'].isna() | merged['value_twin'].isna())
        clean_data = merged[mask]
        
        if len(clean_data) < 4:
            continue
            
        human_vec = clean_data['value_human'].values
        twin_vec = clean_data['value_twin'].values
        
        # Calculate all better accuracy metrics
        rank_accuracy = calculate_rank_correlation_accuracy(human_vec, twin_vec)
        percentile_accuracy = calculate_percentile_accuracy(human_vec, twin_vec)
        directional_accuracy = calculate_directional_accuracy(human_vec, twin_vec)
        distribution_similarity = calculate_distribution_similarity(human_vec, twin_vec)
        weighted_accuracy = calculate_weighted_accuracy(human_vec, twin_vec)
        
        # Extract specification type
        import re
        spec_name = specification_name.replace("resoning", "reasoning")
        date_pattern = r"_\d{4}-\d{2}-\d{2}$"
        spec_type = re.sub(date_pattern, "", spec_name)
        
        result = {
            'study_name': study_name,
            'specification_type': spec_type,
            'variable_name': var,
            'rank_accuracy': rank_accuracy,
            'percentile_accuracy': percentile_accuracy,
            'directional_accuracy': directional_accuracy,
            'distribution_similarity': distribution_similarity,
            'weighted_accuracy': weighted_accuracy,
            'n_points': len(clean_data)
        }
        
        results.append(result)
    
    return results


def main():
    """Demonstrate better accuracy metrics."""
    print("=== Testing Better Accuracy Metrics ===")
    print()
    
    # Process a few sample files to demonstrate
    from pathlib import Path
    results_dir = Path("results")
    
    sample_files = []
    for file_path in results_dir.rglob("meta analysis individual level.csv"):
        sample_files.append(file_path)
        if len(sample_files) >= 10:  # Just process 10 files for demo
            break
    
    all_results = []
    
    print(f"Processing {len(sample_files)} sample files...")
    for file_path in sample_files:
        results = process_individual_file_with_better_metrics(file_path)
        all_results.extend(results)
    
    if not all_results:
        print("No results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate averages by specification
    print("\n=== Better Accuracy Metrics by Specification ===")
    print()
    
    metrics = ['rank_accuracy', 'percentile_accuracy', 'directional_accuracy', 
               'distribution_similarity', 'weighted_accuracy']
    
    avg_metrics = df.groupby('specification_type')[metrics].mean()
    
    print("Specification".ljust(25) + "".join([m.replace('_', ' ').title()[:8].rjust(10) for m in metrics]))
    print("-" * 75)
    
    for spec, row in avg_metrics.iterrows():
        values = [f"{row[m]:.3f}" for m in metrics]
        print(f"{spec[:24]:25s}" + "".join([v.rjust(10) for v in values]))
    
    print()
    print("=== Discriminative Power Analysis ===")
    print()
    
    for metric in metrics:
        values = avg_metrics[metric].values
        range_val = values.max() - values.min()
        cv = values.std() / values.mean() if values.mean() > 0 else 0
        relative_range = range_val / values.mean() * 100 if values.mean() > 0 else 0
        
        print(f"{metric.replace('_', ' ').title()}:")
        print(f"  Range: {range_val:.4f} ({relative_range:.1f}% of mean)")
        print(f"  CV: {cv:.4f}")
        print()


if __name__ == "__main__":
    main()




