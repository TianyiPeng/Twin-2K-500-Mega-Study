#!/usr/bin/env python3
"""
Script to compute 4 metrics between human and twin response vectors for each study/specification/DV combination.
Also generates a random benchmark specification for comparison.

For each DV variable in each study/specification, computes:
1. Correlation of human and twin vectors
2. Current accuracy definition: 1 - |x-y| / (predefined_max - predefined_min) for raw vectors
3. Standard deviation ratio: std(twin) / std(human)
4. Glass's delta for mean comparison (absolute value)

Additionally generates a 'random_benchmark' specification with random uniform responses
for each study/variable combination to serve as a baseline comparison.

Outputs:
- joint_vector_metrics.csv: All metrics for each study/specification/DV (including random benchmark)
- average_metrics_by_specification.csv: Average metrics by specification type (including random benchmark)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load DV variables summary for min/max lookup
DV_VARIABLES_FILE = Path(__file__).parent / "dv_variables_summary.csv"
_dv_lookup = None

def load_dv_variables_lookup():
    """Load DV variables summary and create lookup dictionary."""
    global _dv_lookup
    if _dv_lookup is None:
        try:
            df = pd.read_csv(DV_VARIABLES_FILE)
            # Create lookup: (study_name, variable_name) -> (min, max)
            _dv_lookup = {}
            for _, row in df.iterrows():
                key = (row['study_name'], row['DV_variable'])
                dv_min = row['DV_min']
                dv_max = row['DV_max']
                # Handle 'nan' strings and actual NaN values
                if pd.isna(dv_min) or str(dv_min).lower() == 'nan':
                    dv_min = np.nan
                else:
                    dv_min = float(dv_min)
                if pd.isna(dv_max) or str(dv_max).lower() == 'nan':
                    dv_max = np.nan
                else:
                    dv_max = float(dv_max)
                _dv_lookup[key] = (dv_min, dv_max)
        except Exception as e:
            print(f"Warning: Could not load DV variables file: {e}")
            _dv_lookup = {}
    return _dv_lookup

def get_dv_range(study_name, variable_name):
    """Get the predefined min/max range for a DV variable."""
    lookup = load_dv_variables_lookup()
    return lookup.get((study_name, variable_name), (np.nan, np.nan))


def fisher_z_average(correlations, weights=None):
    """
    Average correlations using Fisher z-transformation.
    
    Args:
        correlations: Array of correlation values
        weights: Optional weights for weighted averaging (e.g., sample sizes)
    
    Returns:
        Average correlation value (transformed back from z-space)
    """
    # Remove NaN values
    mask = ~np.isnan(correlations)
    clean_corrs = correlations[mask]
    
    if len(clean_corrs) == 0:
        return np.nan
    
    # Handle weights
    if weights is not None:
        clean_weights = weights[mask]
        if len(clean_weights) != len(clean_corrs):
            clean_weights = None  # Fall back to unweighted if mismatch
    else:
        clean_weights = None
    
    # Clamp correlations to valid range to avoid arctanh issues
    clean_corrs = np.clip(clean_corrs, -0.9999, 0.9999)
    
    # Transform to Fisher z
    z_values = np.arctanh(clean_corrs)
    
    # Average in z-space
    if clean_weights is not None and len(clean_weights) == len(z_values):
        avg_z = np.average(z_values, weights=clean_weights)
    else:
        avg_z = np.mean(z_values)
    
    # Transform back to correlation space
    avg_r = np.tanh(avg_z)
    
    return avg_r


def fisher_z_std(correlations, weights=None):
    """
    Calculate standard deviation of correlations in Fisher z-space, then transform back.
    
    Args:
        correlations: Array of correlation values
        weights: Optional weights for weighted standard deviation
    
    Returns:
        Standard deviation (transformed back from z-space)
    """
    # Remove NaN values
    mask = ~np.isnan(correlations)
    clean_corrs = correlations[mask]
    
    if len(clean_corrs) <= 1:
        return np.nan
    
    # Clamp correlations to valid range
    clean_corrs = np.clip(clean_corrs, -0.9999, 0.9999)
    
    # Transform to Fisher z
    z_values = np.arctanh(clean_corrs)
    
    # Calculate std in z-space
    if weights is not None:
        clean_weights = weights[mask]
        if len(clean_weights) == len(z_values):
            # Weighted standard deviation
            avg_z = np.average(z_values, weights=clean_weights)
            variance = np.average((z_values - avg_z)**2, weights=clean_weights)
            std_z = np.sqrt(variance)
        else:
            std_z = np.std(z_values, ddof=1)
    else:
        std_z = np.std(z_values, ddof=1)
    
    # Note: For interpretation, we return the std in z-space
    # because transforming back doesn't have a clear interpretation
    return std_z


def compute_vector_metrics(human_vec, twin_vec, variable_name=None, study_name=None):
    """
    Compute all 4 metrics between human and twin vectors, plus random benchmark.
    
    Args:
        human_vec: numpy array of human responses
        twin_vec: numpy array of twin responses
        variable_name: name of the variable being processed (for conditional metric calculation)
        study_name: name of the study (for looking up predefined min/max values)
    
    Returns:
        dict with all 4 metrics plus random benchmark metrics
    """
    # Remove NaN values - keep only paired observations
    mask = ~(np.isnan(human_vec) | np.isnan(twin_vec))
    h = human_vec[mask]
    t = twin_vec[mask]
    
    if len(h) < 2:  # Need at least 2 points for correlation
        return {
            'correlation': np.nan,
            'current_accuracy': np.nan,
            'std_ratio': np.nan,
            'Glass_delta': np.nan,
            'n_points': len(h)
        }
    
    # 1. Correlation - set to 0 if either vector is constant
    if np.std(h) == 0 or np.std(t) == 0:
        correlation = 0.0  # No linear relationship possible with constant vector
    else:
        correlation, _ = pearsonr(h, t)
    
    # 2. Current accuracy definition: |x-y| / (predefined_max - predefined_min)
    # Skip current accuracy for variables with undefined ranges
    accuracy_excluded_variables = ['log_WTP', 'DAT_perf', 'SSPT']
    if variable_name in accuracy_excluded_variables:
        current_accuracy = np.nan
    else:
        # Get predefined min/max from DV variables summary
        if study_name and variable_name:
            dv_min, dv_max = get_dv_range(study_name, variable_name)
            if not (np.isnan(dv_min) or np.isnan(dv_max)) and dv_max > dv_min:
                current_accuracy = 1 - np.mean(np.abs(h - t)) / (dv_max - dv_min)
            else:
                current_accuracy = np.nan
        else:
            # Fallback to human data range if study/variable info not available
            if np.max(h) > np.min(h):
                current_accuracy = 1 - np.mean(np.abs(h - t)) / (np.max(h) - np.min(h))
            else:
                current_accuracy = np.nan
    
    # 3. Standard deviation ratio: std(twin) / std(human)
    std_h = np.std(h)
    std_t = np.std(t)
    if std_h > 0:
        std_ratio = std_t / std_h
    else:
        std_ratio = np.nan
    
    # 4. Glass's delta for mean comparison
    if std_h > 0:
        Glass_delta = (np.mean(t) - np.mean(h)) / std_h
    else:
        Glass_delta = np.nan

    # human std 
    human_std = 1 / np.sqrt(len(h)) #np.std(h) / (np.max(h) - np.min(h)) / np.sqrt(len(h))
    
    return {
        'correlation': correlation,
        'current_accuracy': current_accuracy,
        'std_ratio': std_ratio,
        'Glass_delta': np.abs(Glass_delta),
        'n_points': len(h),
        'human_std': human_std
    }


def generate_random_benchmark_results(df_human, variables, study_name):
    """
    Generate random benchmark results for comparison.
    
    Args:
        df_human: DataFrame with human responses
        variables: List of variable names
        study_name: Name of the study
        
    Returns:
        List of dictionaries with random benchmark metrics
    """
    np.random.seed(42)  # For reproducible results
    results = []
    
    # Filter out excluded variables
    excluded_variables = ['creativity_rating_byhuman']
    variables = [var for var in variables if var not in excluded_variables]
    
    for var in variables:
        var_data = df_human[df_human['variable_name'] == var]
        human_data = var_data[var_data['respondent_type'] == 'human']
        
        if len(human_data) == 0:
            continue
            
        # Remove NaN values
        human_values = human_data['value'].dropna().values
        if len(human_values) < 2:
            continue
            
        
        # Generate random responses using predefined DV range or human range as fallback
        undefined_range_variables = ['log_WTP', 'DAT_perf', 'SSPT']
        
        if var in undefined_range_variables:
            # Use human data range for variables with undefined theoretical ranges
            random_responses = np.random.uniform(np.min(human_values), np.max(human_values), size=len(human_values))
        else:
            # Use predefined DV range from CSV file
            dv_min, dv_max = get_dv_range(study_name, var)
            if not (np.isnan(dv_min) or np.isnan(dv_max)) and dv_max > dv_min:
                random_responses = np.random.uniform(dv_min, dv_max, size=len(human_values))
            else:
                # Fallback to human data range if predefined range is not available
                random_responses = np.random.uniform(np.min(human_values), np.max(human_values), size=len(human_values))
        
        # Compute metrics for random vs human
        metrics = compute_vector_metrics(human_values, random_responses, var, study_name)
        
        result = {
            'study_name': study_name,
            'specification_name': 'random_benchmark',
            'specification_type': 'random_benchmark',
            'variable_name': var,
            **metrics
        }
        
        results.append(result)
    
    return results


def process_individual_level_file(file_path):
    """
    Process a single meta analysis individual level.csv file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of dictionaries with metrics for each DV variable
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    # Extract study and specification info from the first row
    if len(df) == 0:
        return []
    
    study_name = df['study_name'].iloc[0]
    specification_name = df['specification_name'].iloc[0]
    
    # Get unique variables
    variables = df['variable_name'].unique()
    
    # Filter out excluded variables
    excluded_variables = ['creativity_rating_byhuman']
    variables = [var for var in variables if var not in excluded_variables]
    
    results = []
    
    for var in variables:
        var_data = df[df['variable_name'] == var]
        
        # Get human and twin vectors
        human_data = var_data[var_data['respondent_type'] == 'human']
        twin_data = var_data[var_data['respondent_type'] == 'twin']
        
        # Merge on TWIN_ID to ensure proper pairing
        merged = pd.merge(
            human_data[['TWIN_ID', 'value']], 
            twin_data[['TWIN_ID', 'value']], 
            on='TWIN_ID', 
            suffixes=('_human', '_twin')
        )
        
        if len(merged) == 0:
            continue
            
        human_vec = merged['value_human'].values
        twin_vec = merged['value_twin'].values
        
        # Compute metrics
        metrics = compute_vector_metrics(human_vec, twin_vec, var, study_name)
        
        # Add study/specification/variable info
        result = {
            'study_name': study_name,
            'specification_name': specification_name,
            'variable_name': var,
            **metrics
        }
        
        results.append(result)
    
    return results


def find_all_individual_level_files():
    """Find all meta analysis individual level.csv files in the results directory."""
    results_dir = Path("../results")
    
    files = []
    for file_path in results_dir.rglob("meta analysis individual level.csv"):
        files.append(file_path)
    
    return sorted(files)


def main():
    print("Computing vector metrics for all study/specification/DV combinations...")
    
    # Find all individual level files
    files = find_all_individual_level_files()
    print(f"Found {len(files)} individual level files")
    
    # Process all files
    all_results = []
    random_benchmarks_generated = set()  # Track studies for which we've generated random benchmarks
    
    for i, file_path in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {file_path}")
        
        results = process_individual_level_file(file_path)
        all_results.extend(results)
        
        # Generate random benchmark only once per study (only if we got results)
        if results:
            try:
                df = pd.read_csv(file_path)
                study_name = df['study_name'].iloc[0]
                
                # Only generate random benchmark if we haven't done it for this study yet
                if study_name not in random_benchmarks_generated:
                    variables = df['variable_name'].unique()
                    random_results = generate_random_benchmark_results(df, variables, study_name)
                    all_results.extend(random_results)
                    random_benchmarks_generated.add(study_name)
                    print(f"  Generated random benchmark for study: {study_name}")
                    
            except Exception as e:
                print(f"Error generating random benchmark for {file_path}: {e}")
    
    if not all_results:
        print("No results found!")
        return
    
    # Create joint CSV
    joint_df = pd.DataFrame(all_results)
    
    # Add specification type (extracted from specification_name)
    def extract_spec_type(spec_name):
        """Extract specification type from full specification name."""
        # Remove date pattern YYYY-MM-DD from the end
        import re
        spec_name = spec_name.replace("resoning", "reasoning")  # Fix typo
        date_pattern = r"_\d{4}-\d{2}-\d{2}$"
        spec_type = re.sub(date_pattern, "", spec_name)
        return spec_type
    
    joint_df['specification_type'] = joint_df['specification_name'].apply(extract_spec_type)
    
    # Reorder columns (new order: 1=correlation, 2=current_accuracy, 3=std_ratio, 4=Glass_delta)
    cols = ['study_name', 'specification_name', 'specification_type', 'variable_name'] + \
           ['correlation', 'current_accuracy', 'std_ratio', 'Glass_delta', 'n_points', 'human_std']
    joint_df = joint_df[cols]
    
    # Sort by correlation (descending)
    joint_df = joint_df.sort_values('correlation', ascending=False)
    
    # Save joint CSV
    joint_output = "joint_vector_metrics.csv"
    joint_df.to_csv(joint_output, index=False)
    print(f"\nSaved joint metrics to: {joint_output}")
    print(f"Total rows: {len(joint_df)}")
    
    # Compute averages by specification type
    print("\nComputing averages by specification type...")
    
    # Separate correlation from other metrics (correlation needs Fisher z-transform)
    other_metric_cols = ['current_accuracy', 'std_ratio', 'Glass_delta', 'human_std']
    
    # Group by specification type and compute means for non-correlation metrics
    avg_metrics = joint_df.groupby('specification_type')[other_metric_cols].agg(['mean', 'std', 'count']).round(4)
    
    # Handle correlation separately with Fisher z-transformation
    corr_results = []
    for spec_type in joint_df['specification_type'].unique():
        spec_data = joint_df[joint_df['specification_type'] == spec_type]
        correlations = spec_data['correlation'].values
        n_points = spec_data['n_points'].values  # Use sample sizes as weights
        
        # Fisher z-transform averaging for correlation
        fisher_mean = fisher_z_average(correlations, weights=n_points)
        fisher_std = fisher_z_std(correlations, weights=n_points)
        count = len(correlations[~np.isnan(correlations)])
        
        corr_results.append({
            'specification_type': spec_type,
            'correlation_mean': fisher_mean,
            'correlation_std': fisher_std,
            'correlation_count': count
        })
    
    # Convert correlation results to DataFrame
    corr_df = pd.DataFrame(corr_results).set_index('specification_type')
    
    # Combine correlation results with other metrics
    avg_metrics = pd.concat([avg_metrics, corr_df], axis=1)
    
    # Flatten column names for the multi-level columns from agg(), keep others as-is
    new_columns = []
    for col in avg_metrics.columns:
        if isinstance(col, tuple):
            new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(col)
    avg_metrics.columns = new_columns
    
    # Add total observations count
    avg_metrics['total_observations'] = joint_df.groupby('specification_type').size()
    
    # Sort by correlation_mean (descending)
    avg_metrics = avg_metrics.sort_values('correlation_mean', ascending=False)
    
    # Save averages CSV
    avg_output = "average_metrics_by_specification.csv"
    avg_metrics.to_csv(avg_output)
    print(f"Saved average metrics to: {avg_output}")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Studies found: {joint_df['study_name'].nunique()}")
    print(f"Specifications found: {joint_df['specification_type'].nunique()}")
    print(f"Total DV variables: {len(joint_df)}")
    
    print(f"\nSpecification types:")
    for spec in sorted(joint_df['specification_type'].unique()):
        count = len(joint_df[joint_df['specification_type'] == spec])
        print(f"  - {spec}: {count} variables")
    
    print(f"\nAverage correlation by specification (Fisher z-transformed):")
    # Use the Fisher z-transformed correlation averages we computed
    corr_avg = avg_metrics['correlation_mean'].sort_values(ascending=False)
    for spec, corr in corr_avg.items():
        print(f"  - {spec}: {corr:.4f}")
    
    print(f"\nAverage current_accuracy by specification:")
    acc_avg = joint_df.groupby('specification_type')['current_accuracy'].mean().sort_values(ascending=False)
    for spec, acc in acc_avg.items():
        print(f"  - {spec}: {acc:.4f}")


if __name__ == "__main__":
    main()
