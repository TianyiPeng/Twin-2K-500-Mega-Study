#!/usr/bin/env python3
"""
Compare calibration results with the original mega study specifications.
"""

import pandas as pd
import numpy as np


def compare_with_mega_study():
    """Compare calibration results with original mega study performance."""
    
    # Load our calibration summary
    calib_df = pd.read_csv('calibration_metrics_by_specification.csv')
    
    # Load original mega study results
    mega_df = pd.read_csv('../average_metrics_by_specification.csv')
    
    print("=== Calibration vs Mega Study Comparison ===")
    
    # Extract key metrics for comparison
    print("\n=== Calibrated Results vs Best Mega Study Specifications ===")
    
    # Get our calibrated results
    calibrated_results = calib_df[calib_df['specification_type'] == 'calibrated'].iloc[0]
    original_results = calib_df[calib_df['specification_type'] == 'original'].iloc[0]
    
    # Get top performing mega study specifications
    mega_top_corr = mega_df.iloc[0]  # Best correlation (already sorted)
    mega_top_acc = mega_df.loc[mega_df['current_accuracy_mean'].idxmax()]  # Best accuracy
    
    print(f"Our Calibrated Results:")
    print(f"  Correlation (Fisher z-avg): {calibrated_results['correlation_mean']:.4f}")
    print(f"  Current Accuracy: {calibrated_results['current_accuracy_mean']:.4f}")
    print(f"  Standardized Accuracy: {calibrated_results['standardized_accuracy_mean']:.4f}")
    
    print(f"\nOur Original LLM Results:")
    print(f"  Correlation (Fisher z-avg): {original_results['correlation_mean']:.4f}")
    print(f"  Current Accuracy: {original_results['current_accuracy_mean']:.4f}")
    print(f"  Standardized Accuracy: {original_results['standardized_accuracy_mean']:.4f}")
    
    print(f"\nBest Mega Study Specification (by correlation - {mega_top_corr['specification_type']}):")
    print(f"  Correlation (Fisher z-avg): {mega_top_corr['correlation_mean']:.4f}")
    print(f"  Current Accuracy: {mega_top_corr['current_accuracy_mean']:.4f}")
    print(f"  Standardized Accuracy: {mega_top_corr['standardized_accuracy_mean']:.4f}")
    
    print(f"\nBest Mega Study Specification (by accuracy - {mega_top_acc['specification_type']}):")
    print(f"  Correlation (Fisher z-avg): {mega_top_acc['correlation_mean']:.4f}")
    print(f"  Current Accuracy: {mega_top_acc['current_accuracy_mean']:.4f}")
    print(f"  Standardized Accuracy: {mega_top_acc['standardized_accuracy_mean']:.4f}")
    
    # Performance comparisons
    print(f"\n=== Performance Ranking ===")
    
    # Correlation ranking
    print(f"\nCorrelation Performance:")
    calib_corr = calibrated_results['correlation_mean']
    orig_corr = original_results['correlation_mean']
    mega_corrs = mega_df['correlation_mean'].values
    
    # Count how many mega study specs our calibrated results beat
    better_than_count = (calib_corr > mega_corrs).sum()
    total_specs = len(mega_corrs)
    
    print(f"  Our Calibrated: {calib_corr:.4f} (beats {better_than_count}/{total_specs} mega study specs)")
    print(f"  Our Original: {orig_corr:.4f} (beats {(orig_corr > mega_corrs).sum()}/{total_specs} mega study specs)")
    print(f"  Best Mega Study: {mega_corrs.max():.4f}")
    print(f"  Worst Mega Study: {mega_corrs.min():.4f}")
    
    # Accuracy ranking  
    print(f"\nCurrent Accuracy Performance:")
    calib_acc = calibrated_results['current_accuracy_mean']
    orig_acc = original_results['current_accuracy_mean']
    mega_accs = mega_df['current_accuracy_mean'].values
    
    better_acc_count = (calib_acc > mega_accs).sum()
    
    print(f"  Our Calibrated: {calib_acc:.4f} (beats {better_acc_count}/{total_specs} mega study specs)")
    print(f"  Our Original: {orig_acc:.4f} (beats {(orig_acc > mega_accs).sum()}/{total_specs} mega study specs)")
    print(f"  Best Mega Study: {mega_accs.max():.4f}")
    print(f"  Worst Mega Study: {mega_accs.min():.4f}")
    
    # Create combined ranking table
    print(f"\n=== Combined Results Table ===")
    
    # Create comparison dataframe
    comparison_data = []
    
    # Add our results
    comparison_data.append({
        'specification_type': 'calibrated_ours',
        'correlation_mean': calibrated_results['correlation_mean'],
        'current_accuracy_mean': calibrated_results['current_accuracy_mean'],
        'cohens_d_mean': calibrated_results['cohens_d_mean'],
        'std_ratio_mean': calibrated_results['std_ratio_mean'],
        'total_observations': calibrated_results['total_observations']
    })
    
    comparison_data.append({
        'specification_type': 'original_ours',
        'correlation_mean': original_results['correlation_mean'],
        'current_accuracy_mean': original_results['current_accuracy_mean'],
        'cohens_d_mean': original_results['cohens_d_mean'],
        'std_ratio_mean': original_results['std_ratio_mean'],
        'total_observations': original_results['total_observations']
    })
    
    # Add top 5 mega study results
    top_5_mega = mega_df.head(5)
    for _, row in top_5_mega.iterrows():
        comparison_data.append({
            'specification_type': f"mega_{row['specification_type']}",
            'correlation_mean': row['correlation_mean'],
            'current_accuracy_mean': row['current_accuracy_mean'], 
            'cohens_d_mean': row['cohens_d_mean'],
            'std_ratio_mean': row['std_ratio_mean'],
            'total_observations': row['total_observations']
        })
    
    # Create comparison DataFrame and sort by correlation
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('correlation_mean', ascending=False)
    
    # Display the table
    print(comparison_df[['specification_type', 'correlation_mean', 'current_accuracy_mean', 
                        'cohens_d_mean', 'total_observations']].to_string(index=False, float_format='%.4f'))
    
    # Save the comparison
    comparison_df.to_csv('calibration_vs_mega_study_comparison.csv', index=False)
    print(f"\nComparison saved to: calibration_vs_mega_study_comparison.csv")
    
    # Statistical significance insights
    print(f"\n=== Key Insights ===")
    
    improvement = calib_corr - orig_corr
    print(f"1. Calibration improved correlation by {improvement:.4f} ({improvement/orig_corr*100:.1f}% relative improvement)")
    
    rank_among_mega = (calib_corr >= mega_corrs).sum()
    print(f"2. Our calibrated method ranks #{rank_among_mega} out of {total_specs + 1} total specifications")
    
    if calib_corr > mega_corrs.max():
        print(f"3. 🎉 Our calibrated method BEATS ALL mega study specifications!")
    elif calib_corr > mega_corrs[0]:  # Better than best
        print(f"3. 🎉 Our calibrated method beats the best mega study specification!")
    else:
        print(f"3. Our calibrated method performs better than {better_than_count} out of {total_specs} mega study specifications")
    
    return comparison_df


if __name__ == "__main__":
    comparison_df = compare_with_mega_study()

