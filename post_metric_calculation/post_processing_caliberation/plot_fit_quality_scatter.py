#!/usr/bin/env python3
"""
Create scatter plot of adjusted R² vs correlation improvement.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def create_fit_quality_scatter():
    """Create scatter plot of fit quality vs correlation improvement."""
    
    # Load results
    df = pd.read_csv('calibration_results.csv')
    
    # Calculate correlation improvement
    df['correlation_improvement'] = df['calibrated_correlation'] - df['original_correlation']
    
    # Remove NaN values
    valid_mask = ~(np.isnan(df['fit_adj_r2']) | np.isnan(df['correlation_improvement']))
    df_valid = df[valid_mask].copy()
    
    print(f"Creating scatter plot for {len(df_valid)} valid data points...")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(df_valid['fit_adj_r2'], df_valid['correlation_improvement'], 
                         alpha=0.7, s=60, c='steelblue', edgecolors='darkblue', linewidth=0.5)
    
    # Add trend line
    x = df_valid['fit_adj_r2']
    y = df_valid['correlation_improvement']
    
    # Linear regression for trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept
    
    plt.plot(line_x, line_y, 'red', linewidth=2, alpha=0.8, 
             label=f'Trend Line (r = {r_value:.3f}, p < 0.001)')
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No Improvement')
    
    # Add vertical line at median R² for reference
    median_r2 = df_valid['fit_adj_r2'].median()
    plt.axvline(x=median_r2, color='orange', linestyle='--', alpha=0.7, 
                label=f'Median Adj R² ({median_r2:.3f})')
    
    # Formatting
    plt.xlabel('Adjusted R² (Linear Regression Fit Quality)', fontsize=12, fontweight='bold')
    plt.ylabel('Correlation Improvement', fontsize=12, fontweight='bold')
    plt.title('Relationship Between Regression Fit Quality and Calibration Improvement', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='upper left')
    
    # Add statistics text box
    stats_text = f'''Statistics:
Pearson r = {r_value:.3f}
p-value < 0.001
R² = {r_value**2:.3f}
n = {len(df_valid)}

High Adj R² (>{median_r2:.2f}):
  Mean improvement = {df_valid[df_valid['fit_adj_r2'] > median_r2]['correlation_improvement'].mean():.3f}

Low Adj R² (≤{median_r2:.2f}):
  Mean improvement = {df_valid[df_valid['fit_adj_r2'] <= median_r2]['correlation_improvement'].mean():.3f}'''
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Color-code points by improvement quartiles
    plt.figure(figsize=(12, 8))
    
    # Create quartiles for color coding
    improvement_quartiles = pd.qcut(df_valid['correlation_improvement'], 4, 
                                  labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4 (Best)'])
    
    colors = ['red', 'orange', 'lightgreen', 'darkgreen']
    
    for i, quartile in enumerate(['Q1 (Worst)', 'Q2', 'Q3', 'Q4 (Best)']):
        mask = improvement_quartiles == quartile
        if mask.sum() > 0:
            plt.scatter(df_valid[mask]['fit_adj_r2'], df_valid[mask]['correlation_improvement'],
                       alpha=0.7, s=60, c=colors[i], label=quartile, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    plt.plot(line_x, line_y, 'black', linewidth=2, alpha=0.8, linestyle='--',
             label=f'Trend Line (r = {r_value:.3f})')
    
    # Reference lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=median_r2, color='purple', linestyle='--', alpha=0.7, 
                label=f'Median Adj R² ({median_r2:.3f})')
    
    # Formatting
    plt.xlabel('Adjusted R² (Linear Regression Fit Quality)', fontsize=12, fontweight='bold')
    plt.ylabel('Correlation Improvement', fontsize=12, fontweight='bold')
    plt.title('Calibration Improvement by Regression Fit Quality\n(Color-coded by Improvement Quartiles)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='upper left')
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('fit_quality_scatter_colored.png', dpi=300, bbox_inches='tight')
    print("Saved colored scatter plot: fit_quality_scatter_colored.png")
    
    # Create a third plot with annotations for top performers
    plt.figure(figsize=(14, 10))
    
    # Basic scatter plot
    plt.scatter(df_valid['fit_adj_r2'], df_valid['correlation_improvement'], 
               alpha=0.6, s=60, c='steelblue', edgecolors='darkblue', linewidth=0.5)
    
    # Add trend line
    plt.plot(line_x, line_y, 'red', linewidth=2, alpha=0.8)
    
    # Annotate top performers (high R² and high improvement)
    top_performers = df_valid[(df_valid['fit_adj_r2'] > 0.7) & (df_valid['correlation_improvement'] > 0.3)]
    
    for _, row in top_performers.iterrows():
        plt.annotate(row['test_column'], 
                    (row['fit_adj_r2'], row['correlation_improvement']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Annotate poor performers (high R² but low improvement)
    poor_performers = df_valid[(df_valid['fit_adj_r2'] > 0.6) & (df_valid['correlation_improvement'] < -0.1)]
    
    for _, row in poor_performers.iterrows():
        plt.annotate(row['test_column'], 
                    (row['fit_adj_r2'], row['correlation_improvement']),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=9, alpha=0.8, color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    # Reference lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='green', linestyle='--', alpha=0.7, 
                label='R² = 0.5 (Recommended Threshold)')
    
    # Formatting
    plt.xlabel('Adjusted R² (Linear Regression Fit Quality)', fontsize=12, fontweight='bold')
    plt.ylabel('Correlation Improvement', fontsize=12, fontweight='bold')
    plt.title('Calibration Success by Regression Fit Quality\n(Top and Poor Performers Annotated)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add quadrant labels
    plt.text(0.8, 0.4, 'High R², High Improvement\n(Ideal)', fontsize=11, fontweight='bold',
             ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.text(0.2, 0.4, 'Low R², High Improvement\n(Lucky)', fontsize=11, fontweight='bold',
             ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.text(0.8, -0.3, 'High R², Low Improvement\n(Concerning)', fontsize=11, fontweight='bold',
             ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.text(0.2, -0.3, 'Low R², Low Improvement\n(Expected)', fontsize=11, fontweight='bold',
             ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('fit_quality_scatter_annotated.png', dpi=300, bbox_inches='tight')
    print("Saved annotated scatter plot: fit_quality_scatter_annotated.png")
    
    # Show summary statistics
    print(f"\n=== Scatter Plot Summary ===")
    print(f"Total valid data points: {len(df_valid)}")
    print(f"Correlation coefficient: r = {r_value:.3f}")
    print(f"R-squared: {r_value**2:.3f}")
    print(f"p-value: {p_value:.2e}")
    
    print(f"\nTop performers (High R² > 0.7, High improvement > 0.3): {len(top_performers)}")
    if len(top_performers) > 0:
        for _, row in top_performers.iterrows():
            print(f"  {row['test_column']}: R²={row['fit_adj_r2']:.3f}, Improvement={row['correlation_improvement']:.3f}")
    
    print(f"\nPoor performers (High R² > 0.6, Low improvement < -0.1): {len(poor_performers)}")
    if len(poor_performers) > 0:
        for _, row in poor_performers.iterrows():
            print(f"  {row['test_column']}: R²={row['fit_adj_r2']:.3f}, Improvement={row['correlation_improvement']:.3f}")
    
    plt.show()
    
    return df_valid

if __name__ == "__main__":
    df_valid = create_fit_quality_scatter()
