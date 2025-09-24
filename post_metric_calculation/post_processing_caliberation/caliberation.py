#!/usr/bin/env python3
"""
Calibration pipeline for Digital Twin Mega Study.

This script implements a calibration procedure to improve LLM predictions by:
1. Random split of 160 columns into 80 training and 80 testing
2. Stack Y_human and Y_twin for training columns and perform matrix completion with r=5
3. For each testing column, run linear regression on Y_twin and apply to Y_human
4. Compare calibrated predictions vs original Y_twin using compute_vector_metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the compute_vector_metrics function
import sys
sys.path.append('../')
from compute_vector_metrics import compute_vector_metrics
from causaltensor.matlib import SVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

def hard_impute_svd(X, rank=5, max_iter=100, tol=1e-4):
    """
    Hard impute using SVD with rank constraint.
    
    Args:
        X: Input matrix with missing values (NaN)
        rank: Target rank for matrix completion
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        Completed matrix
    """
    # Initialize missing values with column means
    X_filled = X.copy()
    col_means = np.nanmean(X, axis=0)
    
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X_filled[mask, j] = col_means[j]
    
    # Handle any remaining NaNs (columns with all NaN)
    X_filled = np.nan_to_num(X_filled, nan=0)
    
    prev_X = X_filled.copy()
    
    for iteration in range(max_iter):
        # SVD decomposition with rank constraint
        X_filled = SVD(X_filled, min(min(X_filled.shape), rank))
        
        # Keep original observed values
        mask_observed = ~np.isnan(X)
        X_filled[mask_observed] = X[mask_observed]
        
        # Check convergence
        diff = np.linalg.norm(X_filled - prev_X, 'fro') / np.linalg.norm(prev_X, 'fro')
        #print(f"Iteration {iteration + 1}, Diff: {diff}")
        if diff < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
            
        prev_X = X_filled.copy()
    
    return X_filled

def load_data():
    """Load the human and LLM data."""
    human_path = Path("combined_dv_human.csv")
    llm_path = Path("combined_dv_llm.csv")
    
    df_human = pd.read_csv(human_path)
    df_llm = pd.read_csv(llm_path)
    
    print(f"Loaded human data: {df_human.shape}")
    print(f"Loaded LLM data: {df_llm.shape}")
    
    return df_human, df_llm


def random_split_columns(df, train_size=80, random_state=42):
    """
    Randomly split columns into training and testing sets.
    
    Args:
        df: DataFrame to split
        train_size: Number of columns for training
        random_state: Random seed
    
    Returns:
        train_cols, test_cols: Lists of column names
    """
    # Exclude TWIN_ID from splitting
    feature_cols = [col for col in df.columns if col != 'TWIN_ID']
    
    if len(feature_cols) < train_size * 2:
        raise ValueError(f"Not enough columns. Have {len(feature_cols)}, need at least {train_size * 2}")
    
    np.random.seed(random_state)
    train_cols = np.random.choice(feature_cols, size=train_size, replace=False).tolist()
    test_cols = [col for col in feature_cols if col not in train_cols]
    
    print(f"Training columns: {len(train_cols)}")
    print(f"Testing columns: {len(test_cols)}")
    
    return train_cols, test_cols


def stack_and_impute(df_human, df_llm, train_cols, rank=5, use_cv=True):
    """
    Stack human and LLM data and perform matrix completion with optional cross-validation for rank selection.
    
    Args:
        df_human: Human responses DataFrame
        df_llm: LLM responses DataFrame  
        train_cols: Training column names
        rank: Default rank for matrix completion (used if use_cv=False)
        use_cv: Whether to use cross-validation for rank selection
    
    Returns:
        Completed matrices for human and LLM, optimal rank used
    """
    # Extract training data
    Y_human_train = df_human[train_cols].values
    Y_llm_train = df_llm[train_cols].values
    
    print(f"Y_human_train shape: {Y_human_train.shape}")
    print(f"Y_llm_train shape: {Y_llm_train.shape}")
    
    # Stack vertically
    Y_stacked = np.vstack([Y_human_train, Y_llm_train])
    print(f"Stacked matrix shape: {Y_stacked.shape}")
    
    # Check missing values
    n_missing = np.isnan(Y_stacked).sum()
    print(f"Missing values in stacked matrix: {n_missing} ({n_missing / Y_stacked.size * 100:.2f}%)")
    
    # Determine optimal rank
    if use_cv:
        # Define rank range: 1 to min(n,m)/5
        max_rank = min(Y_stacked.shape) // 5
        rank_range = list(range(1, max_rank + 1, 4))
        
        print(f"Matrix dimensions: {Y_stacked.shape}")
        print(f"Testing ranks from 1 to {max_rank} (min(n,m)/5 = {max_rank})")
        

        mask = (np.random.rand(Y_stacked.shape[0], Y_stacked.shape[1]) < 0.2) & (np.isnan(Y_stacked) == False)
        Y_stacked_observed = Y_stacked.copy()
        Y_stacked_observed[mask] = np.nan

        best_error = np.inf
        best_rank = None

        for rank in rank_range:
            Y_completed = hard_impute_svd(Y_stacked_observed, rank=rank)
            error = np.linalg.norm(Y_completed[mask] - Y_stacked[mask]) / np.linalg.norm(Y_stacked[mask])
            if error < best_error:
                best_error = error
                best_rank = rank
            print(f"Rank: {rank}, Error: {error}")

        print(f"Using optimal rank: {best_rank}")
        print(f"Best error: {best_error}")
    else:
        best_rank = rank
        print(f"Using specified rank: {best_rank}")
    
    # Perform matrix completion with optimal rank
    print(f"Performing matrix completion with rank={best_rank}...")
    Y_completed = hard_impute_svd(Y_stacked, rank=best_rank)
    
    # Split back into human and LLM parts
    n_human = Y_human_train.shape[0]
    Y_human_completed = Y_completed[:n_human, :]
    Y_llm_completed = Y_completed[n_human:, :]
    
    return Y_human_completed, Y_llm_completed, best_rank


def calculate_fit_quality(reg, X, y):
    """
    Calculate various metrics for regression fit quality.
    
    Args:
        reg: Fitted LinearRegression model
        X: Feature matrix
        y: Target vector
    
    Returns:
        Dictionary with fit quality metrics
    """
    y_pred = reg.predict(X)
    n, p = X.shape
    
    # Basic metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Adjusted R²
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
    
    # F-statistic for overall model significance
    if r2 > 0 and n > p + 1:
        f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
        f_pvalue = 1 - stats.f.cdf(f_stat, p, n - p - 1)
    else:
        f_stat = np.nan
        f_pvalue = np.nan
    
    # Residual analysis
    residuals = y - y_pred
    residual_std = np.std(residuals)
    
    # Coefficient statistics
    n_significant_coefs = 0
    if hasattr(reg, 'coef_'):
        # Estimate coefficient standard errors (simplified)
        try:
            residual_var = np.sum(residuals**2) / (n - p - 1)
            X_centered = X - np.mean(X, axis=0)
            cov_matrix = residual_var * np.linalg.inv(X_centered.T @ X_centered)
            coef_std_errors = np.sqrt(np.diag(cov_matrix))
            t_stats = reg.coef_ / coef_std_errors
            t_pvalues = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
            n_significant_coefs = np.sum(t_pvalues < 0.05)
        except:
            n_significant_coefs = np.nan
    
    return {
        'r2': r2,
        'adj_r2': adj_r2,
        'mse': mse,
        'rmse': rmse,
        'f_stat': f_stat,
        'f_pvalue': f_pvalue,
        'residual_std': residual_std,
        'n_significant_coefs': n_significant_coefs,
        'n_features': p,
        'n_samples': n
    }


def calibrate_predictions(Y_human_completed, Y_llm_completed, df_human, df_llm, test_cols):
    """
    Calibrate predictions using linear regression with fit quality assessment.
    
    Args:
        Y_human_completed: Completed human training matrix
        Y_llm_completed: Completed LLM training matrix
        df_human: Original human DataFrame
        df_llm: Original LLM DataFrame
        test_cols: Testing column names
    
    Returns:
        calibrated_predictions: Dictionary with calibrated predictions and fit quality for each test column
        original_predictions: Dictionary with original LLM predictions for each test column
    """
    calibrated_predictions = {}
    original_predictions = {}
    
    print(f"Calibrating predictions for {len(test_cols)} test columns...")
    
    for i, test_col in enumerate(test_cols):
        if i % 10 == 0:
            print(f"Processing column {i+1}/{len(test_cols)}: {test_col}")
        
        # Get test column data from LLM
        y_test_llm = df_llm[test_col].values
        
        # Remove rows with NaN in test column
        valid_mask = ~np.isnan(y_test_llm)
        
        if valid_mask.sum() < 10:  # Need enough data points
            print(f"Skipping {test_col}: insufficient valid data ({valid_mask.sum()} points)")
            continue
        
        y_test_llm_valid = y_test_llm[valid_mask]
        X_train_llm = Y_llm_completed[valid_mask]
        X_train_human = Y_human_completed[valid_mask]
        
        # Fit linear regression on LLM data
        reg = LinearRegression()
        reg.fit(X_train_llm, y_test_llm_valid)
        
        # Calculate fit quality metrics
        fit_quality = calculate_fit_quality(reg, X_train_llm, y_test_llm_valid)
        
        # Apply same linear combination to human data for calibration
        y_calibrated = reg.predict(X_train_human)
        
        # Store results with fit quality
        calibrated_predictions[test_col] = {
            'predictions': y_calibrated,
            'mask': valid_mask,
            'model': reg,
            'fit_quality': fit_quality
        }
        
        original_predictions[test_col] = {
            'predictions': y_test_llm_valid,
            'mask': valid_mask
        }
    
    return calibrated_predictions, original_predictions


def evaluate_calibration(calibrated_predictions, original_predictions, df_human, test_cols):
    """
    Evaluate calibration performance using compute_vector_metrics.
    
    Args:
        calibrated_predictions: Calibrated predictions
        original_predictions: Original LLM predictions
        df_human: Human data for ground truth
        test_cols: Test column names
    
    Returns:
        results: DataFrame with evaluation metrics
    """
    results = []
    
    print("Evaluating calibration performance...")
    
    for test_col in test_cols:
        if test_col not in calibrated_predictions:
            continue
            
        # Get human ground truth
        y_human = df_human[test_col].values
        valid_mask = calibrated_predictions[test_col]['mask']
        y_human_valid = y_human[valid_mask]
        
        # Get predictions
        y_calibrated = calibrated_predictions[test_col]['predictions']
        y_original = original_predictions[test_col]['predictions']
        
        # Compute metrics for calibrated predictions
        metrics_calibrated = compute_vector_metrics(y_human_valid, y_calibrated)
        metrics_calibrated = {f"calibrated_{k}": v for k, v in metrics_calibrated.items()}
        
        # Compute metrics for original predictions
        metrics_original = compute_vector_metrics(y_human_valid, y_original)
        metrics_original = {f"original_{k}": v for k, v in metrics_original.items()}
        
        # Get fit quality metrics
        fit_quality = calibrated_predictions[test_col]['fit_quality']
        fit_quality_prefixed = {f"fit_{k}": v for k, v in fit_quality.items()}
        
        # Combine results
        result = {
            'test_column': test_col,
            'n_valid_points': len(y_human_valid),
            **metrics_calibrated,
            **metrics_original,
            **fit_quality_prefixed
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def main():
    """Main calibration pipeline."""
    print("=== Digital Twin Calibration Pipeline ===")
    
    # Load data
    print("\n1. Loading data...")
    df_human, df_llm = load_data()
    
    # Random split columns
    print("\n2. Random split of columns...")
    train_cols, test_cols = random_split_columns(df_human, train_size=80, random_state=42)
    
    # Stack and impute
    print("\n3. Stacking and matrix completion with cross-validation...")
    Y_human_completed, Y_llm_completed, optimal_rank = stack_and_impute(df_human, df_llm, train_cols,  use_cv=True)
    
    # Calibrate predictions
    print("\n4. Calibrating predictions...")
    calibrated_predictions, original_predictions = calibrate_predictions(
        Y_human_completed, Y_llm_completed, df_human, df_llm, test_cols
    )
    
    # Evaluate performance
    print("\n5. Evaluating performance...")
    results_df = evaluate_calibration(calibrated_predictions, original_predictions, df_human, test_cols)
    
    # Save results
    output_path = "calibration_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Processed {len(results_df)} test columns")
    print(f"Optimal rank selected via cross-validation: {optimal_rank}")
    
    # Compare average performance
    calibrated_corr = results_df['calibrated_correlation'].mean()
    original_corr = results_df['original_correlation'].mean()
    
    calibrated_acc = results_df['calibrated_current_accuracy'].mean()
    original_acc = results_df['original_current_accuracy'].mean()
    
    print(f"\nAverage Correlation:")
    print(f"  Original: {original_corr:.4f}")
    print(f"  Calibrated: {calibrated_corr:.4f}")
    print(f"  Improvement: {calibrated_corr - original_corr:.4f}")
    
    print(f"\nAverage Current Accuracy:")
    print(f"  Original: {original_acc:.4f}")
    print(f"  Calibrated: {calibrated_acc:.4f}")
    print(f"  Improvement: {calibrated_acc - original_acc:.4f}")
    
    # Count improvements
    corr_improvements = (results_df['calibrated_correlation'] > results_df['original_correlation']).sum()
    acc_improvements = (results_df['calibrated_current_accuracy'] > results_df['original_current_accuracy']).sum()
    
    print(f"\nColumns with improved correlation: {corr_improvements}/{len(results_df)} ({corr_improvements/len(results_df)*100:.1f}%)")
    print(f"Columns with improved accuracy: {acc_improvements}/{len(results_df)} ({acc_improvements/len(results_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
