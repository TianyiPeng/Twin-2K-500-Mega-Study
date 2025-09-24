#!/usr/bin/env python3
"""
Analysis script for calibration results.
"""

import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def analyze_calibration_results():
    """Analyze the calibration results."""
    # Load results
    df = pd.read_csv('calibration_results.csv')
    
    print("=== Calibration Results Analysis ===")
    print(f"Total test columns processed: {len(df)}")
    
    # Calculate improvements
    df['correlation_improvement'] = df['calibrated_correlation'] - df['original_correlation']
    df['accuracy_improvement'] = df['calibrated_current_accuracy'] - df['original_current_accuracy']
    
    # Summary statistics
    print("\n=== Overall Performance ===")
    print(f"Average correlation improvement: {df['correlation_improvement'].mean():.4f}")
    print(f"Average accuracy improvement: {df['accuracy_improvement'].mean():.4f}")
    
    print(f"Median correlation improvement: {df['correlation_improvement'].median():.4f}")
    print(f"Median accuracy improvement: {df['accuracy_improvement'].median():.4f}")
    
    # Count improvements
    corr_better = (df['correlation_improvement'] > 0).sum()
    acc_better = (df['accuracy_improvement'] > 0).sum()
    
    print(f"\nColumns with improved correlation: {corr_better}/{len(df)} ({corr_better/len(df)*100:.1f}%)")
    print(f"Columns with improved accuracy: {acc_better}/{len(df)} ({acc_better/len(df)*100:.1f}%)")
    
    # Significant improvements
    significant_corr = (df['correlation_improvement'] > 0.1).sum()
    significant_acc = (df['accuracy_improvement'] > 0.05).sum()
    
    print(f"\nColumns with significant correlation improvement (>0.1): {significant_corr}/{len(df)} ({significant_corr/len(df)*100:.1f}%)")
    print(f"Columns with significant accuracy improvement (>0.05): {significant_acc}/{len(df)} ({significant_acc/len(df)*100:.1f}%)")
    
    # Best and worst performers
    print("\n=== Top 10 Correlation Improvements ===")
    top_corr = df.nlargest(10, 'correlation_improvement')[['test_column', 'original_correlation', 'calibrated_correlation', 'correlation_improvement']]
    for _, row in top_corr.iterrows():
        print(f"{row['test_column']:20} | Original: {row['original_correlation']:.3f} → Calibrated: {row['calibrated_correlation']:.3f} (Δ: +{row['correlation_improvement']:.3f})")
    
    print("\n=== Top 10 Accuracy Improvements ===")
    top_acc = df.nlargest(10, 'accuracy_improvement')[['test_column', 'original_current_accuracy', 'calibrated_current_accuracy', 'accuracy_improvement']]
    for _, row in top_acc.iterrows():
        print(f"{row['test_column']:20} | Original: {row['original_current_accuracy']:.3f} → Calibrated: {row['calibrated_current_accuracy']:.3f} (Δ: +{row['accuracy_improvement']:.3f})")
    
    print("\n=== Worst 5 Correlation Changes ===")
    worst_corr = df.nsmallest(5, 'correlation_improvement')[['test_column', 'original_correlation', 'calibrated_correlation', 'correlation_improvement']]
    for _, row in worst_corr.iterrows():
        print(f"{row['test_column']:20} | Original: {row['original_correlation']:.3f} → Calibrated: {row['calibrated_correlation']:.3f} (Δ: {row['correlation_improvement']:.3f})")
    
    # Distribution analysis
    print(f"\n=== Distribution Analysis ===")
    print(f"Correlation improvement - Min: {df['correlation_improvement'].min():.3f}, Max: {df['correlation_improvement'].max():.3f}")
    print(f"Accuracy improvement - Min: {df['accuracy_improvement'].min():.3f}, Max: {df['accuracy_improvement'].max():.3f}")
    
    # Original performance analysis
    print(f"\n=== Original Performance Context ===")
    print(f"Original correlation - Mean: {df['original_correlation'].mean():.3f}, Std: {df['original_correlation'].std():.3f}")
    print(f"Original accuracy - Mean: {df['original_current_accuracy'].mean():.3f}, Std: {df['original_current_accuracy'].std():.3f}")
    
    print(f"Calibrated correlation - Mean: {df['calibrated_correlation'].mean():.3f}, Std: {df['calibrated_correlation'].std():.3f}")
    print(f"Calibrated accuracy - Mean: {df['calibrated_current_accuracy'].mean():.3f}, Std: {df['calibrated_current_accuracy'].std():.3f}")
    
    # Create simple visualizations if matplotlib is available
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(15, 5))
        
        # Correlation comparison
        plt.subplot(1, 3, 1)
        plt.scatter(df['original_correlation'], df['calibrated_correlation'], alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        plt.xlabel('Original Correlation')
        plt.ylabel('Calibrated Correlation')
        plt.title('Correlation: Original vs Calibrated')
        plt.grid(True, alpha=0.3)
        
        # Accuracy comparison
        plt.subplot(1, 3, 2)
        plt.scatter(df['original_current_accuracy'], df['calibrated_current_accuracy'], alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        plt.xlabel('Original Accuracy')
        plt.ylabel('Calibrated Accuracy')
        plt.title('Accuracy: Original vs Calibrated')
        plt.grid(True, alpha=0.3)
        
        # Improvement distribution
        plt.subplot(1, 3, 3)
        plt.hist(df['correlation_improvement'], bins=20, alpha=0.7, label='Correlation', density=True)
        plt.hist(df['accuracy_improvement'], bins=20, alpha=0.7, label='Accuracy', density=True)
        plt.axvline(0, color='red', linestyle='--', alpha=0.8)
        plt.xlabel('Improvement')
        plt.ylabel('Density')
        plt.title('Distribution of Improvements')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('calibration_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved as: calibration_analysis.png")
    else:
        print("\nMatplotlib not available - skipping visualizations")
    
    return df

if __name__ == "__main__":
    results_df = analyze_calibration_results()
