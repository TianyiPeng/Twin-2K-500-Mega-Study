#!/usr/bin/env python3
"""
Analyze the relationship between linear regression fit quality and calibration improvement.
"""

import pandas as pd
import numpy as np
from scipy import stats
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def analyze_fit_quality_correlation():
    """Analyze correlation between fit quality and calibration improvement."""
    
    # Load results
    df = pd.read_csv('calibration_results.csv')
    
    print("=== Fit Quality vs Calibration Improvement Analysis ===")
    print(f"Total test columns: {len(df)}")
    
    # Calculate improvement metrics
    df['correlation_improvement'] = df['calibrated_correlation'] - df['original_correlation']
    df['accuracy_improvement'] = df['calibrated_current_accuracy'] - df['original_current_accuracy']
    df['standardized_accuracy_improvement'] = df['calibrated_standardized_accuracy'] - df['original_standardized_accuracy']
    
    # Fit quality metrics to analyze
    fit_metrics = [col for col in df.columns if col.startswith('fit_')]
    improvement_metrics = ['correlation_improvement', 'accuracy_improvement', 'standardized_accuracy_improvement']
    
    print(f"\nFit quality metrics available: {fit_metrics}")
    print(f"Improvement metrics: {improvement_metrics}")
    
    # Correlation analysis
    print(f"\n=== Correlation Analysis ===")
    correlation_results = []
    
    for fit_metric in fit_metrics:
        for imp_metric in improvement_metrics:
            # Remove NaN values for correlation calculation
            valid_mask = ~(np.isnan(df[fit_metric]) | np.isnan(df[imp_metric]))
            
            if valid_mask.sum() < 10:
                continue
                
            fit_values = df[fit_metric][valid_mask]
            imp_values = df[imp_metric][valid_mask]
            
            # Calculate Pearson correlation
            corr_coef, p_value = stats.pearsonr(fit_values, imp_values)
            
            # Calculate Spearman correlation (rank-based, more robust)
            spear_coef, spear_p = stats.spearmanr(fit_values, imp_values)
            
            correlation_results.append({
                'fit_metric': fit_metric,
                'improvement_metric': imp_metric,
                'pearson_r': corr_coef,
                'pearson_p': p_value,
                'spearman_r': spear_coef,
                'spearman_p': spear_p,
                'n_samples': valid_mask.sum()
            })
    
    # Convert to DataFrame and sort by absolute correlation
    corr_df = pd.DataFrame(correlation_results)
    corr_df['abs_pearson_r'] = np.abs(corr_df['pearson_r'])
    corr_df = corr_df.sort_values('abs_pearson_r', ascending=False)
    
    # Display correlation results
    print(f"\nTop correlations between fit quality and calibration improvement:")
    print(f"{'Fit Metric':<20} {'Improvement':<25} {'Pearson r':<10} {'p-value':<10} {'Spearman r':<12} {'n':<8}")
    print("-" * 95)
    
    for _, row in corr_df.head(15).iterrows():
        significance = "***" if row['pearson_p'] < 0.001 else "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else ""
        print(f"{row['fit_metric']:<20} {row['improvement_metric']:<25} {row['pearson_r']:<10.3f} {row['pearson_p']:<10.3f} {row['spearman_r']:<12.3f} {row['n_samples']:<8.0f} {significance}")
    
    # Summary statistics by fit quality quartiles
    print(f"\n=== Improvement by Fit Quality Quartiles ===")
    
    # Focus on R² as primary fit quality metric
    if 'fit_r2' in df.columns:
        print(f"\nAnalyzing improvement by R² quartiles:")
        
        # Remove NaN values
        valid_r2_mask = ~np.isnan(df['fit_r2'])
        df_valid = df[valid_r2_mask].copy()
        
        # Create quartiles
        df_valid['r2_quartile'] = pd.qcut(df_valid['fit_r2'], 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        
        # Calculate mean improvements by quartile
        quartile_stats = df_valid.groupby('r2_quartile')[improvement_metrics + ['fit_r2']].agg(['mean', 'std', 'count'])
        
        print(f"R² Quartile Analysis:")
        for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
            if quartile in quartile_stats.index:
                corr_imp = quartile_stats.loc[quartile, ('correlation_improvement', 'mean')]
                acc_imp = quartile_stats.loc[quartile, ('accuracy_improvement', 'mean')]
                r2_mean = quartile_stats.loc[quartile, ('fit_r2', 'mean')]
                n_count = quartile_stats.loc[quartile, ('correlation_improvement', 'count')]
                
                print(f"  {quartile}: R²={r2_mean:.3f}, Corr_imp={corr_imp:.3f}, Acc_imp={acc_imp:.3f} (n={n_count})")
    
    # Statistical tests
    print(f"\n=== Statistical Tests ===")
    
    if 'fit_r2' in df.columns and len(df_valid) > 0:
        # Test if high R² models have better improvements
        high_r2_mask = df_valid['fit_r2'] > df_valid['fit_r2'].median()
        low_r2_mask = ~high_r2_mask
        
        high_r2_corr_imp = df_valid[high_r2_mask]['correlation_improvement']
        low_r2_corr_imp = df_valid[low_r2_mask]['correlation_improvement']
        
        # T-test
        t_stat, t_p = stats.ttest_ind(high_r2_corr_imp.dropna(), low_r2_corr_imp.dropna())
        
        print(f"T-test: High R² vs Low R² correlation improvement")
        print(f"  High R² mean: {high_r2_corr_imp.mean():.4f} (n={len(high_r2_corr_imp)})")
        print(f"  Low R² mean: {low_r2_corr_imp.mean():.4f} (n={len(low_r2_corr_imp)})")
        print(f"  t-statistic: {t_stat:.3f}, p-value: {t_p:.3f}")
        
        significance = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
        print(f"  Significance: {significance}")
    
    # Key insights
    print(f"\n=== Key Insights ===")
    
    # Find strongest correlations
    strongest_corr = corr_df.iloc[0] if len(corr_df) > 0 else None
    
    if strongest_corr is not None:
        print(f"1. Strongest correlation: {strongest_corr['fit_metric']} vs {strongest_corr['improvement_metric']}")
        print(f"   Pearson r = {strongest_corr['pearson_r']:.3f} (p = {strongest_corr['pearson_p']:.3f})")
        
        if abs(strongest_corr['pearson_r']) > 0.3:
            print(f"   This suggests a moderate-to-strong relationship between fit quality and improvement!")
        elif abs(strongest_corr['pearson_r']) > 0.1:
            print(f"   This suggests a weak-to-moderate relationship between fit quality and improvement.")
        else:
            print(f"   This suggests a weak relationship between fit quality and improvement.")
    
    # Count significant correlations
    sig_correlations = corr_df[corr_df['pearson_p'] < 0.05]
    print(f"\n2. Significant correlations (p < 0.05): {len(sig_correlations)} out of {len(corr_df)}")
    
    if len(sig_correlations) > 0:
        print(f"   Most significant: {sig_correlations.iloc[0]['fit_metric']} vs {sig_correlations.iloc[0]['improvement_metric']}")
        print(f"   (r = {sig_correlations.iloc[0]['pearson_r']:.3f}, p = {sig_correlations.iloc[0]['pearson_p']:.3f})")
    
    # Visualization
    if HAS_PLOTTING and 'fit_r2' in df.columns:
        print(f"\n=== Creating Visualizations ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # R² vs correlation improvement
        axes[0, 0].scatter(df['fit_r2'], df['correlation_improvement'], alpha=0.6)
        axes[0, 0].set_xlabel('Linear Regression R²')
        axes[0, 0].set_ylabel('Correlation Improvement')
        axes[0, 0].set_title('Fit Quality vs Correlation Improvement')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line
        valid_mask = ~(np.isnan(df['fit_r2']) | np.isnan(df['correlation_improvement']))
        if valid_mask.sum() > 2:
            x_valid = df['fit_r2'][valid_mask]
            y_valid = df['correlation_improvement'][valid_mask]
            z = np.polyfit(x_valid, y_valid, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(x_valid, p(x_valid), "r--", alpha=0.8)
        
        # R² vs accuracy improvement
        axes[0, 1].scatter(df['fit_r2'], df['accuracy_improvement'], alpha=0.6, color='orange')
        axes[0, 1].set_xlabel('Linear Regression R²')
        axes[0, 1].set_ylabel('Accuracy Improvement')
        axes[0, 1].set_title('Fit Quality vs Accuracy Improvement')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F-statistic vs correlation improvement
        if 'fit_f_stat' in df.columns:
            axes[1, 0].scatter(df['fit_f_stat'], df['correlation_improvement'], alpha=0.6, color='green')
            axes[1, 0].set_xlabel('F-statistic')
            axes[1, 0].set_ylabel('Correlation Improvement')
            axes[1, 0].set_title('F-statistic vs Correlation Improvement')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Adjusted R² vs correlation improvement
        if 'fit_adj_r2' in df.columns:
            axes[1, 1].scatter(df['fit_adj_r2'], df['correlation_improvement'], alpha=0.6, color='purple')
            axes[1, 1].set_xlabel('Adjusted R²')
            axes[1, 1].set_ylabel('Correlation Improvement')
            axes[1, 1].set_title('Adjusted R² vs Correlation Improvement')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fit_quality_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Visualizations saved as: fit_quality_analysis.png")
    
    # Save detailed results
    corr_df.to_csv('fit_quality_correlations.csv', index=False)
    print(f"\nDetailed correlation results saved to: fit_quality_correlations.csv")
    
    return corr_df


if __name__ == "__main__":
    results = analyze_fit_quality_correlation()

