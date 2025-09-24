#!/usr/bin/env python3
"""Run ML predictions for all studies based on their ml_simulation configurations.
This script iterates through all study configs and runs XGBoost predictions
using the existing predict_answer_xgboost functions.
"""

import argparse
import glob
import logging
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from existing predict_answer_xgboost module
# Import formatting module
from ml_prediction.format_predictions import intelligent_round_predictions
from ml_prediction.predict_answer_xgboost import train_xgboost_predictions

# Import evaluation modules

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default columns to exclude from prediction targets
DEFAULT_EXCLUDE_COLUMNS = [
    # Timing and Progress Metadata
    "StartDate",
    "EndDate",
    "Progress",
    "Duration (in seconds)",
    "Finished",
    "RecordedDate",
    # Browser and System Information
    "Browser_Browser",
    "Browser_Version",
    "Browser_Operating System",
    "Browser_Resolution",
    # Response and Distribution Metadata
    "ResponseID",
    "IPAddress",
    "LocationLatitude",
    "LocationLongitude",
    "DistributionChannel",
    "UserLanguage",
    "Status",
    # Participant Information
    "RecipientLastName",
    "RecipientFirstName",
    "RecipientEmail",
    "ExternalReference",
    "mTurkCode",
    "gc",
    # Survey ID (always exclude)
    "TWIN_ID",
    # Wildcard patterns for systematic exclusions
    "*_RT_*",  # Response time columns
    "*_DO_*",  # Display order columns
    "*_First Click",
    "*_Last Click",
    "*_Page Submit",
    "*_Click Count",
    # Common metadata patterns
    "Comments",
    "Condition",
    "Random",
    "att_check*",  # Attention check questions
]


def extract_study_specific_data(
    wave_files: List[str],
    values_file: str,
    additional_columns: List[str] = None,
    exclude_columns: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract training features from wave files and prediction targets from values file.

    Args:
        wave_files: List of wave CSV files for training features
        values_file: Path to study-specific values CSV file
        additional_columns: Additional columns from values file to use as features
        exclude_columns: Columns to exclude from prediction targets

    Returns:
        Tuple of (training_df, targets_df)
    """
    logger.info(f"Extracting data from wave files and {values_file}")

    # Load wave files and concatenate
    wave_dfs = []
    for wave_file in wave_files:
        if os.path.exists(wave_file):
            # Skip the first 2 rows after header (Qualtrics metadata rows)
            df = pd.read_csv(wave_file, skiprows=[1, 2])
            wave_dfs.append(df)
            logger.info(f"Loaded {len(df)} rows from {wave_file} (skipped 2 metadata rows)")
        else:
            logger.warning(f"Wave file not found: {wave_file}")

    if not wave_dfs:
        raise ValueError("No wave files found")

    # Merge wave data on TWIN_ID
    training_df = wave_dfs[0]
    for df in wave_dfs[1:]:
        # Merge on TWIN_ID, keeping all columns except duplicates
        training_df = pd.merge(training_df, df, on="TWIN_ID", how="outer", suffixes=("", "_dup"))
        # Drop duplicate columns
        dup_cols = [col for col in training_df.columns if col.endswith("_dup")]
        training_df.drop(columns=dup_cols, inplace=True)

    # Load values file
    if not os.path.exists(values_file):
        raise ValueError(f"Values file not found: {values_file}")

    # Skip the first 2 rows after header (Qualtrics metadata rows)
    values_df = pd.read_csv(values_file, skiprows=[1, 2])
    logger.info(f"Loaded {len(values_df)} rows from values file (skipped 2 metadata rows)")

    # Extract additional features if specified
    if additional_columns:
        additional_features = values_df[["TWIN_ID"] + additional_columns].copy()
        training_df = pd.merge(training_df, additional_features, on="TWIN_ID", how="left")
        logger.info(f"Added {len(additional_columns)} additional features")

    # Prepare prediction targets
    if exclude_columns is None:
        exclude_columns = []

    # Merge default exclusions with user-specified exclusions
    all_exclude_columns = DEFAULT_EXCLUDE_COLUMNS + exclude_columns
    logger.info(
        f"Total exclusion patterns: {len(DEFAULT_EXCLUDE_COLUMNS)} default + {len(exclude_columns)} from config"
    )

    # Process exclusion patterns
    actual_exclude = set()  # TWIN_ID is already in DEFAULT_EXCLUDE_COLUMNS

    for pattern in all_exclude_columns:
        if "*" in pattern:
            # Handle wildcard patterns
            import fnmatch

            matching_cols = [col for col in values_df.columns if fnmatch.fnmatch(col, pattern)]
            actual_exclude.update(matching_cols)
        else:
            actual_exclude.add(pattern)

    # Get target columns
    target_columns = [col for col in values_df.columns if col not in actual_exclude]
    targets_df = values_df[["TWIN_ID"] + target_columns].copy()

    # Check for duplicates between additional features and targets
    if additional_columns:
        duplicates = set(additional_columns) & set(target_columns)
        if duplicates:
            warnings.warn(
                f"WARNING: Duplicate columns found between additional_features and "
                f"prediction_targets: {duplicates}. These columns will only be used "
                f"as features, not as prediction targets."
            )
            targets_df.drop(columns=list(duplicates), inplace=True)

    logger.info(f"Training features: {len(training_df.columns) - 1} columns")
    logger.info(f"Prediction targets: {len(targets_df.columns) - 1} columns")

    return training_df, targets_df


def format_predictions_to_match_values_csv(
    predictions_df: pd.DataFrame,
    values_csv_path: str,
    output_path: str,
    include_metadata_rows: bool = True,
    apply_intelligent_formatting: bool = True,
) -> pd.DataFrame:
    """Format predictions to match the structure of the original values CSV file.

    Args:
        predictions_df: DataFrame with predictions (must have TWIN_ID column)
        values_csv_path: Path to the original values CSV file
        output_path: Path to save the formatted CSV
        include_metadata_rows: If True, includes the 2 metadata rows from original file
        apply_intelligent_formatting: If True, applies intelligent integer conversion

    Returns:
        Formatted DataFrame
    """
    logger.info(f"Formatting predictions to match {values_csv_path}")

    # First, read just the headers and metadata rows if needed
    if include_metadata_rows:
        # Read the first 3 rows (header + 2 metadata rows)
        metadata_df = pd.read_csv(values_csv_path, nrows=2, header=None, skiprows=0)
        # Get headers separately
        values_df = pd.read_csv(values_csv_path, nrows=0)
        original_columns = list(values_df.columns)
    else:
        # Just get headers
        values_df = pd.read_csv(values_csv_path, nrows=0)
        original_columns = list(values_df.columns)

    # Create a new DataFrame with the same structure
    formatted_df = pd.DataFrame(columns=original_columns)

    # Apply intelligent formatting if requested
    if apply_intelligent_formatting:
        # Load the ground truth data (skip metadata rows)
        ground_truth_df = pd.read_csv(values_csv_path, skiprows=[1, 2])

        # Identify numeric columns for formatting
        numeric_cols = []
        for col in predictions_df.columns:
            if col == "TWIN_ID":
                continue
            try:
                pd.to_numeric(predictions_df[col], errors="coerce")
                numeric_cols.append(col)
            except:
                pass

        # Apply intelligent formatting
        predictions_df, formatting_log = intelligent_round_predictions(
            predictions_df, ground_truth_df, numeric_cols
        )

        # Log formatting results
        logger.info(f"Applied intelligent formatting to {len(formatting_log)} columns")
        for col, action in list(formatting_log.items())[:5]:  # Show first 5
            logger.debug(f"  {col}: {action}")

    # Get TWIN_IDs from predictions
    if "TWIN_ID" not in predictions_df.columns:
        raise ValueError("Predictions DataFrame must have TWIN_ID column")

    twin_ids = predictions_df["TWIN_ID"].values
    formatted_df["TWIN_ID"] = twin_ids

    # Fill in prediction columns that exist in both DataFrames
    prediction_columns = set(predictions_df.columns) - {"TWIN_ID"}
    values_columns = set(original_columns) - {"TWIN_ID"}

    # Columns that have predictions
    common_columns = prediction_columns & values_columns
    for col in common_columns:
        formatted_df[col] = predictions_df[col].values

    # Fill metadata columns with empty strings or defaults
    metadata_columns = [
        "StartDate",
        "EndDate",
        "Progress",
        "Duration (in seconds)",
        "Finished",
        "RecordedDate",
        "Browser_Browser",
        "Browser_Version",
        "Browser_Operating System",
        "Browser_Resolution",
    ]

    for col in metadata_columns:
        if col in original_columns and col not in formatted_df.columns:
            if col == "Progress":
                formatted_df[col] = 100  # Assume completed
            elif col == "Finished":
                formatted_df[col] = "True"
            else:
                formatted_df[col] = ""

    # Load the ground truth data to copy values for excluded columns
    if include_metadata_rows:
        ground_truth_full = pd.read_csv(values_csv_path, skiprows=[1, 2])
    else:
        ground_truth_full = pd.read_csv(values_csv_path)

    # Merge with ground truth to get values for excluded columns
    if "TWIN_ID" in ground_truth_full.columns:
        # Ensure TWIN_ID is string type for merging
        twin_ids_str = pd.Series(twin_ids).astype(str)
        ground_truth_full["TWIN_ID"] = ground_truth_full["TWIN_ID"].astype(str)

        # Get the subset of ground truth that matches our predictions
        merged_data = pd.merge(
            pd.DataFrame({"TWIN_ID": twin_ids_str}), ground_truth_full, on="TWIN_ID", how="left"
        )

        # Fill any remaining columns from ground truth
        for col in original_columns:
            if col not in formatted_df.columns and col in merged_data.columns:
                formatted_df[col] = merged_data[col].values
            elif col not in formatted_df.columns:
                formatted_df[col] = np.nan
    else:
        # If no TWIN_ID in ground truth, fill with NaN
        for col in original_columns:
            if col not in formatted_df.columns:
                formatted_df[col] = np.nan

    # Ensure column order matches original
    formatted_df = formatted_df[original_columns]

    # Save formatted CSV
    if include_metadata_rows:
        # Write the file with metadata rows
        with open(output_path, "w") as f:
            # Write header
            f.write(",".join(original_columns) + "\n")
            # Write metadata rows
            for idx in range(len(metadata_df)):
                # metadata_df has no header, so columns are just numbers
                row_values = []
                for col in original_columns:
                    col_idx = original_columns.index(col)
                    if col_idx < len(metadata_df.columns):
                        value = str(metadata_df.iloc[idx, col_idx])
                        # Escape commas and quotes in metadata
                        if "," in value or '"' in value or "\n" in value:
                            value = '"' + value.replace('"', '""') + '"'
                        row_values.append(value)
                    else:
                        row_values.append("")
                f.write(",".join(row_values) + "\n")

        # Append the predictions data without header
        formatted_df.to_csv(output_path, mode="a", header=False, index=False)
        logger.info(f"Saved formatted predictions with metadata rows to {output_path}")
    else:
        formatted_df.to_csv(output_path, index=False)
        logger.info(f"Saved formatted predictions to {output_path}")

    # Log summary
    logger.info("Formatted predictions summary:")
    logger.info(f"  - Total rows: {len(formatted_df)}")
    logger.info(f"  - Total columns: {len(original_columns)}")
    logger.info(f"  - Predicted columns: {len(common_columns)}")
    logger.info(f"  - Missing columns filled: {len(values_columns - prediction_columns)}")

    return formatted_df


def run_study_ml_prediction(
    study_name: str,
    config_path: str,
    test_mode: bool = False,
    max_personas: Optional[int] = None,
    override_cv_folds: Optional[int] = None,
    override_inner_cv_folds: Optional[int] = None,
    override_n_iter: Optional[int] = None,
    no_tuning: bool = False,
    include_metadata_rows: bool = True,
    apply_intelligent_formatting: bool = True,
) -> Dict:
    """Run ML predictions for a single study based on its configuration.

    Args:
        study_name: Name of the study
        config_path: Path to the study's configuration file
        test_mode: If True, use reduced cross-validation for testing
        max_personas: Optional limit on number of personas to process

    Returns:
        Dictionary with prediction results
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing study: {study_name}")
    logger.info(f"Config: {config_path}")

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check if ml_simulation section exists
    if "ml_simulation" not in config:
        logger.warning(f"No ml_simulation section found in {config_path}")
        return None

    ml_config = config["ml_simulation"]

    # Extract configuration parameters
    wave_files = ml_config["training_data"]["wave_files"]
    values_file = ml_config["prediction_targets"]["source"]
    additional_features = ml_config["training_data"]["additional_features"]
    additional_columns = additional_features.get("columns", [])
    exclude_columns = ml_config["prediction_targets"].get("exclude_columns", [])

    # ML configuration
    ml_params = ml_config["ml_config"]
    cv_folds = ml_params.get("cv_folds", 3)
    use_nested_cv = ml_params.get("use_nested_cv", True)
    inner_cv_folds = 3  # default
    n_iter = 10  # default

    # Override for test mode
    if test_mode:
        logger.info("Test mode: Using reduced cross-validation")
        cv_folds = 2
        use_nested_cv = False
        n_iter = 5

    # Apply command line overrides
    if override_cv_folds is not None:
        cv_folds = override_cv_folds
        logger.info(f"Overriding CV folds to {cv_folds}")

    if override_inner_cv_folds is not None:
        inner_cv_folds = override_inner_cv_folds
        logger.info(f"Overriding inner CV folds to {inner_cv_folds}")

    if override_n_iter is not None:
        n_iter = override_n_iter
        logger.info(f"Overriding hyperparameter search iterations to {n_iter}")

    if no_tuning:
        use_nested_cv = False
        logger.info("Disabling hyperparameter tuning")

    # Output configuration
    output_config = ml_config["output"]
    base_dir = output_config["base_dir"]
    os.makedirs(base_dir, exist_ok=True)

    try:
        # Extract data
        training_df, targets_df = extract_study_specific_data(
            wave_files=wave_files,
            values_file=values_file,
            additional_columns=additional_columns if additional_columns else None,
            exclude_columns=exclude_columns,
        )

        # Limit personas if specified
        if max_personas and max_personas > 0:
            logger.info(f"Limiting to {max_personas} personas")
            # Get common TWIN_IDs
            common_ids = set(training_df["TWIN_ID"]) & set(targets_df["TWIN_ID"])
            selected_ids = list(common_ids)[:max_personas]
            training_df = training_df[training_df["TWIN_ID"].isin(selected_ids)]
            targets_df = targets_df[targets_df["TWIN_ID"].isin(selected_ids)]

        # Run predictions using existing function
        results = train_xgboost_predictions(
            training_df=training_df,
            label_df=targets_df,
            output_dir=base_dir,
            cv_folds=cv_folds,
            use_nested_cv=use_nested_cv,
            inner_cv_folds=inner_cv_folds,
            n_iter=n_iter,
        )

        # Format predictions to match original values CSV structure
        if results and "predictions" in results:
            predictions_df = results["predictions"]
            formatted_csv_path = os.path.join(base_dir, "xgboost_predictions_formatted.csv")

            try:
                format_predictions_to_match_values_csv(
                    predictions_df=predictions_df,
                    values_csv_path=values_file,
                    output_path=formatted_csv_path,
                    include_metadata_rows=include_metadata_rows,
                    apply_intelligent_formatting=apply_intelligent_formatting,
                )
            except Exception as e:
                logger.warning(f"Could not format predictions: {e}")

        # Save study-specific metadata
        metadata = {
            "study_name": study_name,
            "config_path": config_path,
            "wave_files": wave_files,
            "values_file": values_file,
            "additional_features": additional_columns,
            "excluded_columns": exclude_columns,
            "test_mode": test_mode,
        }

        import json

        metadata_path = os.path.join(base_dir, "study_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Successfully completed predictions for {study_name}")
        return results

    except Exception as e:
        import traceback

        logger.error(f"Error processing {study_name}: {e!s}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def find_all_study_configs(configs_dir: str = "configs") -> Dict[str, List[str]]:
    """Find all study ML configuration files (*_ml.yaml).

    Returns:
        Dictionary mapping study names to list of config paths
    """
    study_configs = {}

    # Search specifically for *_ml.yaml files in configs directory
    for config_path in glob.glob(os.path.join(configs_dir, "**/*_ml.yaml"), recursive=True):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Verify it has ml_simulation section
            if "ml_simulation" in config:
                # Extract study name from path
                relative_path = os.path.relpath(config_path, configs_dir)
                study_name = relative_path.split("/")[0]

                if study_name not in study_configs:
                    study_configs[study_name] = []
                study_configs[study_name].append(config_path)
                logger.debug(f"Found ML config for {study_name}: {config_path}")

        except Exception as e:
            logger.warning(f"Error reading {config_path}: {e}")

    return study_configs


def main():
    parser = argparse.ArgumentParser(
        description="Run ML predictions for all studies",
        epilog="""
Examples:
  # Run all studies in test mode:
  python %(prog)s --test
  
  # Run specific studies:
  python %(prog)s --studies accuracy_nudges story_beliefs
  
  # Run with limited personas for testing:
  python %(prog)s --max-personas 100 --test
  
  # Run with comprehensive hyperparameter search:
  python %(prog)s --cv-folds 5 --n-iter 30 --inner-cv-folds 5
  
  # Run from specific config directory:
  python %(prog)s --configs-dir ../configs
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--studies", nargs="+", help="Specific studies to run (default: all)")
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="configs",
        help="Directory containing study configs (default: configs)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with reduced cross-validation"
    )
    parser.add_argument("--max-personas", type=int, help="Limit number of personas per study")
    parser.add_argument("--cv-folds", type=int, help="Override number of CV folds from config")
    parser.add_argument(
        "--inner-cv-folds", type=int, help="Override number of inner CV folds (default: 3)"
    )
    parser.add_argument(
        "--n-iter", type=int, help="Override number of hyperparameter search iterations"
    )
    parser.add_argument(
        "--no-tuning", action="store_true", help="Disable hyperparameter tuning completely"
    )
    parser.add_argument(
        "--no-metadata-rows",
        action="store_true",
        help="Exclude metadata rows when formatting predictions",
    )
    parser.add_argument(
        "--no-intelligent-formatting",
        action="store_true",
        help="Disable intelligent integer conversion for predictions",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="ml_prediction/all_studies_summary.json",
        help="Path for summary output file",
    )

    args = parser.parse_args()

    # Find all study configs
    logger.info(f"Searching for configs in {args.configs_dir}")
    study_configs = find_all_study_configs(args.configs_dir)

    if not study_configs:
        logger.error("No study configurations found with ml_simulation sections")
        return

    logger.info(f"Found {len(study_configs)} studies with ML configurations (*_ml.yaml files)")

    # Filter studies if specified
    if args.studies:
        study_configs = {k: v for k, v in study_configs.items() if k in args.studies}
        logger.info(f"Filtered to {len(study_configs)} studies")

    # Run predictions for each study
    all_results = {}
    successful_studies = []
    failed_studies = []

    for study_name, config_paths in study_configs.items():
        # Use the first config if multiple exist
        config_path = config_paths[0]

        try:
            results = run_study_ml_prediction(
                study_name=study_name,
                config_path=config_path,
                test_mode=args.test,
                max_personas=args.max_personas,
                override_cv_folds=args.cv_folds,
                override_inner_cv_folds=args.inner_cv_folds,
                override_n_iter=args.n_iter,
                no_tuning=args.no_tuning,
                include_metadata_rows=not args.no_metadata_rows,
                apply_intelligent_formatting=not args.no_intelligent_formatting,
            )

            if results:
                all_results[study_name] = {
                    "config_path": config_path,
                    "summary": results["results"]["summary"],
                    "metrics_summary": {
                        "total_metrics": len(results["results"]["metrics"]),
                        "sample_metrics": dict(list(results["results"]["metrics"].items())[:5]),
                    },
                }
                successful_studies.append(study_name)
            else:
                failed_studies.append(study_name)

        except Exception as e:
            logger.error(f"Failed to process {study_name}: {e}")
            failed_studies.append(study_name)

    # Save summary
    summary = {
        "total_studies": len(study_configs),
        "successful": len(successful_studies),
        "failed": len(failed_studies),
        "successful_studies": successful_studies,
        "failed_studies": failed_studies,
        "study_results": all_results,
        "test_mode": args.test,
        "max_personas": args.max_personas,
    }

    import json

    os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)
    with open(args.output_summary, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total studies processed: {summary['total_studies']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")

    if successful_studies:
        logger.info("\nSuccessful studies:")
        for study in successful_studies:
            logger.info(f"  - {study}")

    if failed_studies:
        logger.info("\nFailed studies:")
        for study in failed_studies:
            logger.info(f"  - {study}")

    logger.info(f"\nSummary saved to: {args.output_summary}")


if __name__ == "__main__":
    main()
