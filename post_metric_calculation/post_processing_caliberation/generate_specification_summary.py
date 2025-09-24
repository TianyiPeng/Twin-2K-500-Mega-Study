#!/usr/bin/env python3
"""
Generate specification-style summary for calibration results.
This mimics the format of average_metrics_by_specification.csv
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from compute_vector_metrics import fisher_z_average, fisher_z_std


def generate_calibration_summary():
    """Generate a summary similar to average_metrics_by_specification.csv for calibration results."""
    
    # Load calibration results
    df = pd.read_csv('calibration_results.csv')
    
    print("=== Generating Calibration Summary ===")
    print(f"Processing {len(df)} test columns...")
    
    # Define the two "specification types" we want to compare
    specification_types = ['original', 'calibrated']
    
    # Prepare data for analysis
    results = []
    
    for spec_type in specification_types:
        print(f"Processing {spec_type} metrics...")
        
        # Get the appropriate columns based on specification type
        if spec_type == 'original':
            metric_cols = {
                'correlation': 'original_correlation',
                'current_accuracy': 'original_current_accuracy', 
                'standardized_accuracy': 'original_standardized_accuracy',
                'wasserstein_distance': 'original_wasserstein_distance',
                'std_ratio': 'original_std_ratio',
                'mean_difference': 'original_mean_difference',
                'cohens_d': 'original_cohens_d',
                'human_std': 'original_human_std',
                'n_points': 'original_n_points'
            }
        else:  # calibrated
            metric_cols = {
                'correlation': 'calibrated_correlation',
                'current_accuracy': 'calibrated_current_accuracy',
                'standardized_accuracy': 'calibrated_standardized_accuracy', 
                'wasserstein_distance': 'calibrated_wasserstein_distance',
                'std_ratio': 'calibrated_std_ratio',
                'mean_difference': 'calibrated_mean_difference',
                'cohens_d': 'calibrated_cohens_d',
                'human_std': 'calibrated_human_std',
                'n_points': 'calibrated_n_points'
            }
        
        # Extract data for this specification type
        spec_data = {}
        for metric, col_name in metric_cols.items():
            if col_name in df.columns:
                spec_data[metric] = df[col_name].values
            else:
                print(f"Warning: Column {col_name} not found")
                spec_data[metric] = np.full(len(df), np.nan)
        
        # Calculate statistics for non-correlation metrics
        other_metrics = ['current_accuracy', 'standardized_accuracy', 'wasserstein_distance',
                        'std_ratio', 'mean_difference', 'cohens_d', 'human_std']
        
        result = {'specification_type': spec_type}
        
        for metric in other_metrics:
            if metric in spec_data:
                values = spec_data[metric]
                # Remove NaN values for calculations
                clean_values = values[~np.isnan(values)]
                
                result[f'{metric}_mean'] = np.mean(clean_values) if len(clean_values) > 0 else np.nan
                result[f'{metric}_std'] = np.std(clean_values, ddof=1) if len(clean_values) > 1 else np.nan
                result[f'{metric}_count'] = len(clean_values)
        
        # Handle correlation separately with Fisher z-transformation
        correlations = spec_data['correlation']
        n_points = spec_data['n_points']
        
        # Fisher z-transform averaging for correlation
        fisher_mean = fisher_z_average(correlations, weights=n_points)
        fisher_std = fisher_z_std(correlations, weights=n_points)
        count = len(correlations[~np.isnan(correlations)])
        
        result['correlation_mean'] = fisher_mean
        result['correlation_std'] = fisher_std
        result['correlation_count'] = count
        
        # Add total observations
        result['total_observations'] = len(df)
        
        results.append(result)
    
    # Create DataFrame and format similar to original
    summary_df = pd.DataFrame(results)
    
    # Reorder columns to match original format
    column_order = ['specification_type']
    
    # Add other metrics in the same order as original
    for metric in ['current_accuracy', 'standardized_accuracy', 'wasserstein_distance',
                   'std_ratio', 'mean_difference', 'cohens_d', 'human_std']:
        column_order.extend([f'{metric}_mean', f'{metric}_std', f'{metric}_count'])
    
    # Add correlation metrics
    column_order.extend(['correlation_mean', 'correlation_std', 'correlation_count'])
    column_order.append('total_observations')
    
    # Reorder columns
    summary_df = summary_df[column_order]
    
    # Round to 4 decimal places like the original
    numeric_cols = [col for col in summary_df.columns if col != 'specification_type']
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    # Sort by correlation_mean (descending) like the original
    summary_df = summary_df.sort_values('correlation_mean', ascending=False)
    
    # Save the summary
    output_path = 'calibration_metrics_by_specification.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary saved to: {output_path}")
    
    # Print the summary
    print("\n=== Calibration Metrics by Specification Type ===")
    print(summary_df.to_string(index=False))
    
    # Calculate and display improvements
    if len(summary_df) == 2:
        calibrated_row = summary_df[summary_df['specification_type'] == 'calibrated'].iloc[0]
        original_row = summary_df[summary_df['specification_type'] == 'original'].iloc[0]
        
        print(f"\n=== Improvements Summary ===")
        print(f"Correlation improvement: {calibrated_row['correlation_mean'] - original_row['correlation_mean']:.4f}")
        print(f"Current accuracy improvement: {calibrated_row['current_accuracy_mean'] - original_row['current_accuracy_mean']:.4f}")
        print(f"Standardized accuracy improvement: {calibrated_row['standardized_accuracy_mean'] - original_row['standardized_accuracy_mean']:.4f}")
        print(f"Wasserstein distance change: {calibrated_row['wasserstein_distance_mean'] - original_row['wasserstein_distance_mean']:.4f}")
        print(f"Std ratio change: {calibrated_row['std_ratio_mean'] - original_row['std_ratio_mean']:.4f}")
        print(f"Mean difference change: {calibrated_row['mean_difference_mean'] - original_row['mean_difference_mean']:.4f}")
        print(f"Cohen's d change: {calibrated_row['cohens_d_mean'] - original_row['cohens_d_mean']:.4f}")
    
    return summary_df


if __name__ == "__main__":
    summary_df = generate_calibration_summary()

