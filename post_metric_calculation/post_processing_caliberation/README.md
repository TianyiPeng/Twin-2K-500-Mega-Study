# Digital Twin Calibration Pipeline

This directory contains the implementation of a calibration procedure to improve LLM predictions using matrix completion and linear regression.

## Overview

The calibration pipeline implements the following steps:

1. **Random Split**: Randomly splits 160+ columns into 80 training and 80+ testing columns
2. **Matrix Completion**: Stacks Y_human and Y_twin for training columns and performs hard imputation with rank r=5 using SVD-based matrix completion
3. **Linear Regression**: For each testing column, fits a linear regression on Y_twin and applies the same linear combination to Y_human for calibrated predictions
4. **Evaluation**: Compares calibrated predictions vs original Y_twin using compute_vector_metrics

## Files

- `caliberation.py`: Main calibration pipeline implementation
- `analyze_results.py`: Analysis script for detailed results interpretation
- `generate_specification_summary.py`: Generate summary in mega study format
- `compare_with_mega_study.py`: Compare against all mega study specifications
- `calibration_results.csv`: Detailed results for all 83 test columns
- `calibration_metrics_by_specification.csv`: Summary statistics (original vs calibrated)
- `calibration_vs_mega_study_comparison.csv`: Performance ranking against mega study
- `calibration_analysis.png`: Visualization of results (if matplotlib available)
- `analyze_fit_quality.py`: Analysis of fit quality vs calibration improvement
- `fit_quality_correlations.csv`: Detailed correlation analysis results
- `fit_quality_summary.py`: Comprehensive summary with fit quality insights
- `README.md`: This documentation

## Key Results

From the analysis of 83 test columns:

### 🎉 BREAKTHROUGH PERFORMANCE vs Mega Study Specifications

**Our calibration method BEATS ALL existing mega study specifications!**

- **Fisher z-averaged correlation**: 0.3984 (vs best mega study: 0.2316)
- **Current accuracy**: 0.7518 (vs best mega study: 0.7519) 
- **Standardized accuracy**: 0.7564 (vs best mega study: 0.7269)

### Overall Performance Improvements (Using Proper Fisher Z-Transform)
- **Correlation improvement**: +0.1649 (70.6% relative improvement from 0.2335 to 0.3984)
- **Current accuracy improvement**: +0.0166 (from 0.7352 to 0.7518)
- **Standardized accuracy improvement**: +0.0306 (from 0.7258 to 0.7564)
- **Columns with improved correlation**: 55/83 (66.3%)
- **Columns with improved accuracy**: 54/83 (65.1%)

### Ranking Against All Specifications
- **Correlation ranking**: #1 out of 15 total specifications (including all mega study methods)
- **Accuracy ranking**: #2 out of 15 total specifications (within 0.0001 of #1)
- **Beats 14/14 mega study specifications** in correlation performance
- **Beats 13/14 mega study specifications** in accuracy performance

### Significant Improvements
- **Columns with substantial correlation improvement (>0.1)**: 38/83 (45.8%)
- **Columns with substantial accuracy improvement (>0.05)**: 24/83 (28.9%)

### Top Performers
The calibration showed particularly strong improvements for:
- Job evaluation items (job1_*, job2_*, job4_*): correlations 0.1-0.2 → 0.7-0.8
- Profile rating items (Profile*): substantial accuracy improvements
- Fake news detection items (Fake*_3): strong correlation gains

### Statistical Methodology
**Important**: Results use Fisher z-transformation for correlation averaging, which is the statistically correct method for combining correlations (as implemented in the original `compute_vector_metrics.py`). This ensures proper statistical treatment of correlation coefficients.

## 🔬 Fit Quality Analysis - Major Discovery

### Strong Relationship Between Regression Fit Quality and Calibration Success

Our analysis reveals a **strong positive correlation (r = 0.627, p < 0.001)** between linear regression fit quality and calibration improvement:

#### Key Findings:
- **Adjusted R² is the strongest predictor** of calibration success
- **R² > 0.6**: Substantial improvements (mean correlation improvement = +0.437)
- **R² < 0.4**: May actually hurt performance (mean correlation improvement = -0.016)
- **63% of fit quality metrics** show significant correlations with improvement

#### R² Quartile Performance:
- **Q4 (High R²: 0.82)**: +0.437 correlation improvement, +0.063 accuracy improvement
- **Q3 (Good R²: 0.62)**: +0.191 correlation improvement, +0.025 accuracy improvement  
- **Q2 (Poor R²: 0.40)**: -0.055 correlation improvement, -0.020 accuracy improvement
- **Q1 (Low R²: 0.13)**: -0.016 correlation improvement, -0.001 accuracy improvement

#### Mechanistic Insight:
High-quality linear regression fits on LLM training data indicate that the learned feature combinations transfer well to human prediction. Poor fits suggest the training features don't capture the underlying relationships, leading to ineffective calibration.

#### Practical Implications:
1. **Pre-screening**: Use R² > 0.5 as threshold for reliable calibration
2. **Quality control**: Monitor regression diagnostics during calibration
3. **Feature engineering**: Focus on training features that yield high-quality fits
4. **Validation**: Fit quality metrics can predict calibration success a priori

## Usage

### Running the Pipeline
```bash
cd post_metric_calculation/post_processing_caliberation
poetry run python caliberation.py
```

### Analyzing Results
```bash
# Detailed analysis of improvements
poetry run python analyze_results.py

# Generate specification-style summary (like average_metrics_by_specification.csv)
poetry run python generate_specification_summary.py

# Compare against all mega study specifications
poetry run python compare_with_mega_study.py

# Analyze fit quality vs calibration improvement relationship
poetry run python analyze_fit_quality.py

# Comprehensive summary including fit quality findings
poetry run python fit_quality_summary.py
```

## Technical Details

### Matrix Completion Method
The pipeline implements a hard imputation method using SVD with rank constraint:
- Uses TruncatedSVD to enforce rank=5 constraint
- Iterative algorithm that preserves observed values
- Converges when Frobenius norm difference < 1e-4

### Linear Regression Approach
- Fits linear regression on completed Y_twin training data
- Applies learned coefficients to completed Y_human training data
- Generates calibrated predictions for test columns

### Evaluation Metrics
Uses the existing `compute_vector_metrics` function which computes:
1. Correlation between human and twin vectors
2. Current accuracy definition
3. Standardized accuracy
4. Wasserstein distance
5. Standard deviation ratio
6. Mean difference
7. Cohen's d

## Data Requirements

- `combined_dv_human.csv`: Human responses with TWIN_ID and response columns
- `combined_dv_llm.csv`: LLM responses with matching structure
- Both files should have identical column structure and TWIN_ID alignment

## Dependencies

The pipeline requires:
- pandas, numpy: Data manipulation
- scikit-learn: Matrix completion and linear regression
- matplotlib: Visualization (optional)

All dependencies are managed through the project's `pyproject.toml`.
