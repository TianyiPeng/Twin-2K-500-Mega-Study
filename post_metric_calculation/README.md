# Post-Metric Calculation

This folder contains scripts and data files related to calculating and analyzing metrics for comparing human and digital twin responses.

## Files

### Python Scripts

- **`compute_vector_metrics.py`**: Main script that computes 7 different metrics between human and twin response vectors for each study/specification/DV combination. Also generates random benchmark specifications for comparison.

- **`analyze_vector_metrics.py`**: Analysis script to visualize and summarize the vector metrics results. Reads the computed metrics and provides rankings and performance comparisons.

- **`better_accuracy_metrics.py`**: Demonstrates alternative accuracy metrics that are more discriminative than the current accuracy metric, including rank correlation and percentile-based measures.

- **`demonstrate_fisher_z_difference.py`**: Shows the difference between arithmetic mean and Fisher z-transformation for averaging correlations.

### Data Files

- **`joint_vector_metrics.csv`**: All computed metrics for each study/specification/DV combination (including random benchmark)

- **`average_metrics_by_specification.csv`**: Average metrics aggregated by specification type (including random benchmark)

## Metrics Computed

The scripts compute the following 7 metrics:
1. Correlation of human and twin vectors
2. Current accuracy: 1 - |x-y| / (max(human)-min(human)) for raw vectors
3. Standardized accuracy: 1 - |x-y| / (max(human)-min(human)) for standardized vectors
4. Wasserstein distance between human and twin distributions (normalized by range)
5. Standard deviation ratio: std(twin) / std(human)
6. Mean difference: |mean(twin) - mean(human)| / (max(human)-min(human))
7. Cohen's d for mean comparison (absolute value)

## Usage

1. Run `compute_vector_metrics.py` to generate the metric files
2. Run `analyze_vector_metrics.py` to get summary statistics and rankings
3. Use `better_accuracy_metrics.py` and `demonstrate_fisher_z_difference.py` for additional analysis insights
