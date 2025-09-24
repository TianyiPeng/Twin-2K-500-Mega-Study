#!/usr/bin/env python3
"""
Analysis script to visualize and summarize the vector metrics results.
"""

import pandas as pd
import numpy as np

def main():
    print("=== Vector Metrics Analysis ===\n")
    
    # Load the joint metrics
    joint_df = pd.read_csv("joint_vector_metrics.csv")
    
    # Load the averages
    avg_df = pd.read_csv("average_metrics_by_specification.csv")
    
    print(f"Total observations: {len(joint_df)}")
    print(f"Studies: {joint_df['study_name'].nunique()}")
    print(f"Specifications: {joint_df['specification_type'].nunique()}")
    print(f"Variables per specification: {len(joint_df) / joint_df['specification_type'].nunique():.1f}")
    
    print("\n=== Specification Performance Ranking ===")
    
    # Rank by each metric
    metrics = ['correlation', 'current_accuracy', 'standardized_l1_norm', 'standardized_l2_norm', 'absolute_accuracy']
    
    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        col_name = f"{metric}_mean"
        
        if 'norm' in metric:  # Lower is better for norms
            ranking = avg_df.sort_values(col_name)
        else:  # Higher is better for correlation, accuracy
            ranking = avg_df.sort_values(col_name, ascending=False)
        
        for i, (_, row) in enumerate(ranking.iterrows(), 1):
            spec = row['specification_type']
            value = row[col_name]
            std_value = row[f"{metric}_std"]
            print(f"  {i:2d}. {spec:35s} {value:.4f} (±{std_value:.4f})")
    
    print("\n=== Comparison with Current Meta-Analysis Accuracy ===")
    
    # Load the existing meta-analysis results
    try:
        existing_df = pd.read_csv("mega_study_evaluation/meta_analysis_results/combined_all_specifications_meta_analysis.csv")
        
        # Group by specification type and get mean accuracy
        existing_acc = existing_df.groupby('persona specification')['accuracy between humans vs. their twins'].mean()
        
        print("\nComparison of accuracy metrics:")
        print(f"{'Specification':35s} {'Current Def':>12s} {'New Def':>12s} {'Difference':>12s}")
        print("-" * 75)
        
        for spec_type in sorted(avg_df['specification_type'].unique()):
            new_acc = avg_df[avg_df['specification_type'] == spec_type]['current_accuracy_mean'].iloc[0]
            
            if spec_type in existing_acc.index:
                old_acc = existing_acc[spec_type]
                diff = new_acc - old_acc
                print(f"{spec_type:35s} {old_acc:12.4f} {new_acc:12.4f} {diff:12.4f}")
            else:
                print(f"{spec_type:35s} {'N/A':>12s} {new_acc:12.4f} {'N/A':>12s}")
                
    except FileNotFoundError:
        print("Could not load existing meta-analysis results for comparison")
    
    print("\n=== Standard Error Analysis ===")
    
    # Calculate coefficient of variation for each metric
    print("\nCoefficient of variation (std/mean) for each metric:")
    for metric in ['correlation', 'current_accuracy', 'standardized_l1_norm', 'absolute_accuracy']:
        col_mean = f"{metric}_mean"
        col_std = f"{metric}_std"
        
        cv_values = avg_df[col_std] / avg_df[col_mean]
        avg_cv = cv_values.mean()
        
        print(f"  {metric.replace('_', ' ').title():25s}: {avg_cv:.3f}")
    
    print("\n=== Top Performing Specifications Summary ===")
    
    # Find top 3 in each main metric
    top_corr = avg_df.nlargest(3, 'correlation_mean')['specification_type'].tolist()
    top_acc = avg_df.nlargest(3, 'current_accuracy_mean')['specification_type'].tolist()
    low_l1 = avg_df.nsmallest(3, 'standardized_l1_norm_mean')['specification_type'].tolist()
    
    print(f"Top 3 correlation:     {', '.join(top_corr)}")
    print(f"Top 3 accuracy:        {', '.join(top_acc)}")
    print(f"Top 3 low L1 norm:     {', '.join(low_l1)}")
    
    # Find consistent top performers
    all_top = set(top_corr + top_acc + low_l1)
    top_counts = {spec: (spec in top_corr) + (spec in top_acc) + (spec in low_l1) for spec in all_top}
    consistent_tops = [spec for spec, count in top_counts.items() if count >= 2]
    
    print(f"Consistent performers: {', '.join(consistent_tops) if consistent_tops else 'None'}")

if __name__ == "__main__":
    main()




