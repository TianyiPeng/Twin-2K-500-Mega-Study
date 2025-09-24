#!/usr/bin/env python3
"""Test all mega_study_evaluation.py scripts with appropriate data"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Base paths - detect project root
if Path.cwd().name == "mega_study_evaluation":
    PROJECT_ROOT = Path.cwd().parent
else:
    PROJECT_ROOT = Path.cwd()

BASE_DIR = PROJECT_ROOT / "mega_study_evaluation"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_OUTPUT_DIR = BASE_DIR / "test_outputs"
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Load study mapping
MAPPING_FILE = BASE_DIR / "study_folder_mapping.json"
with open(MAPPING_FILE, "r") as f:
    STUDY_MAPPING = json.load(f)

# Skip the _comments entry
STUDIES = [k for k in STUDY_MAPPING.keys() if k not in ["_comments"]]

# Map studies to their available specifications
STUDY_SPECS = {
    "accuracy_nudges": "full_persona_without_reasoning_2025-06-18",  # Added
    "affective_priming": "full_persona_without_reasoning_2025-06-18",
    "consumer_minimalism": "full_persona_without_reasoning_2025-06-26",
    "context_effects": "full_persona_without_reasoning_2025-06-18",
    "default_eric": "full_persona_without_reasoning_2025-06-18",
    "digital_certification": "full_persona_without_reasoning_2025-06-23",
    "hiring_algorithms": "full_persona_without_reasoning_2025-07-03",
    "idea_evaluation": "full_persona_without_reasoning_2025-07-03",
    "idea_generation": "full_persona_without_reasoning_2025-06-12",
    "infotainment": "full_persona_without_reasoning_2025-06-20",  # Updated to correct date
    "junk_fees": "full_persona_without_reasoning_2025-07-04",
    "obedient_twins": "full_persona_without_reasoning_2025-06-26",
    "preference_redistribution": "full_persona_without_resoning_2025-07-06",  # Note typo in dir name
    "privacy": "full_persona_without_reasoning_2025-06-18",
    "promiscuous_donors": "full_persona_without_reasoning_2025-07-04",
    "quantitative_intuition": "full_persona_without_reasoning_2025-06-24",
    "recommendation_algorithms": "full_persona_without_reasoning_2025-07-22",
    "story_beliefs": "full_persona_without_reasoning_2025-06-20",
    "targeting_fairness": "full_persona_without_reasoning_2025-06-23",
}


def compare_csv_files(expected_file, actual_file, tolerance=1e-10, ignore_columns=None, allow_extra_columns=True):
    """Compare two CSV files for equality.

    Args:
        expected_file: Path to expected CSV (from notebook)
        actual_file: Path to actual CSV (from script)
        tolerance: Numerical tolerance for float comparisons
        ignore_columns: List of column names to ignore in comparison
        allow_extra_columns: If True, allow actual to have more columns than expected

    Returns:
        tuple: (is_equal, message)
    """
    if ignore_columns is None:
        ignore_columns = []
    try:
        expected_df = pd.read_csv(expected_file)
        actual_df = pd.read_csv(actual_file)
    except Exception as e:
        return False, f"Error reading files: {e}"

    # Store original shape for reporting
    original_actual_shape = actual_df.shape
    
    # Remove ignored columns
    for col in ignore_columns:
        if col in expected_df.columns:
            expected_df = expected_df.drop(columns=[col])
        if col in actual_df.columns:
            actual_df = actual_df.drop(columns=[col])

    # Check columns
    missing_in_actual = set(expected_df.columns) - set(actual_df.columns)
    extra_in_actual = set(actual_df.columns) - set(expected_df.columns)
    
    # Missing columns in actual is always an error
    if missing_in_actual:
        return False, f"Missing required columns in actual: {missing_in_actual}"
    
    # Extra columns in actual is OK if allow_extra_columns is True
    if extra_in_actual and not allow_extra_columns:
        return False, f"Extra columns not allowed: {extra_in_actual}"
    
    # For comparison, only use columns that exist in expected
    actual_df = actual_df[expected_df.columns]
    
    # Check shape after column alignment
    if expected_df.shape[0] != actual_df.shape[0]:
        return False, f"Row count mismatch: expected {expected_df.shape[0]}, got {actual_df.shape[0]}"
    
    # Report shape difference if there are extra columns (informational)
    shape_msg = ""
    if extra_in_actual:
        shape_msg = f" (actual has {len(extra_in_actual)} extra columns: {', '.join(sorted(extra_in_actual))})"

    # Check values column by column
    mismatches = []
    for col in expected_df.columns:
        expected_col = expected_df[col]
        actual_col = actual_df[col]

        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(expected_col):
            # Convert to float for comparison
            expected_vals = pd.to_numeric(expected_col, errors="coerce")
            actual_vals = pd.to_numeric(actual_col, errors="coerce")

            # Check for NaN mismatches
            if not (expected_vals.isna() == actual_vals.isna()).all():
                mismatches.append(f"NaN mismatch in column '{col}'")
                continue

            # Compare non-NaN values
            mask = ~expected_vals.isna()
            if mask.any():
                if not np.allclose(
                    expected_vals[mask], actual_vals[mask], rtol=tolerance, atol=tolerance
                ):
                    max_diff = np.max(np.abs(expected_vals[mask] - actual_vals[mask]))
                    mismatches.append(f"Numeric mismatch in column '{col}' (max diff: {max_diff})")

        # Handle string/object columns
        else:
            # Convert to string for comparison
            expected_str = expected_col.astype(str)
            actual_str = actual_col.astype(str)

            if not expected_str.equals(actual_str):
                # Find first mismatch
                for i, (e, a) in enumerate(zip(expected_str, actual_str)):
                    if e != a:
                        mismatches.append(
                            f"String mismatch in column '{col}' at row {i}: '{e}' != '{a}'"
                        )
                        break

    if mismatches:
        return False, "; ".join(mismatches[:5]) + shape_msg  # Show first 5 mismatches

    return True, "Files match" + shape_msg


def compare_outputs(expected_files, actual_files, tolerance=1e-5):
    """Compare output files between test_files_original and test_files_new.

    Args:
        expected_files: Dict of test_files_original paths
        actual_files: Dict of test_files_new paths
        tolerance: Numerical tolerance for comparisons

    Returns:
        dict: Results for each file comparison
    """
    results = {}
    meta_analysis_match = None

    # Compare each file type
    for file_type, expected_path in expected_files.items():
        if file_type not in actual_files:
            results[file_type] = {
                "match": False,
                "message": f"No corresponding file in test_files_new",
                "is_meta_analysis": file_type == "meta_analysis",
            }
            continue

        actual_path = actual_files[file_type]
        expected_file = PROJECT_ROOT / expected_path
        actual_file = PROJECT_ROOT / actual_path

        # Check if files exist
        if not expected_file.exists():
            results[file_type] = {
                "match": False,
                "message": f"Expected file not found: {expected_path}",
                "is_meta_analysis": file_type == "meta_analysis",
            }
            continue

        if not actual_file.exists():
            results[file_type] = {
                "match": False,
                "message": f"Actual file not found: {actual_path}",
                "is_meta_analysis": file_type == "meta_analysis",
            }
            continue

        # For meta analysis files, ignore persona specification columns
        is_meta_analysis = file_type == "meta_analysis"
        is_individual_level = file_type == "meta_analysis_individual_level" or file_type == "individual_level"

        # Columns added on July 31 that aren't in the test files (notebooks)
        # These should be ignored as they were added after test files were created
        extra_cols_to_ignore = ["specification_name_full", "ran_date"]
        
        # The "persona specification" column changed from hardcoded "default persona"
        # to dynamic extraction from path on July 31
        if is_meta_analysis:
            extra_cols_to_ignore.append("persona specification")
        
        # For individual level, also handle specification_name differences
        # The notebooks have "default persona" while scripts have the actual specification
        if is_individual_level:
            # We'll ignore this column's values since it's a known change
            # Notebooks have "default persona", scripts have actual specification name
            extra_cols_to_ignore.append("specification_name")
        
        # Allow extra columns that were added after test files were created
        # This handles the change from 26 to 28 columns on July 31
        is_equal, message = compare_csv_files(
            expected_file,
            actual_file,
            tolerance,
            ignore_columns=extra_cols_to_ignore,
            allow_extra_columns=True  # Allow the extra columns from July 31 update
        )

        # For meta analysis files, check if values are close
        if (is_meta_analysis or is_individual_level) and not is_equal:
            # Try with a slightly larger tolerance
            is_close, _ = compare_csv_files(
                expected_file,
                actual_file,
                tolerance=1e-6,
                ignore_columns=extra_cols_to_ignore,
                allow_extra_columns=True  # Allow the extra columns from July 31 update
            )
            if is_close:
                message = f"Values are close but not identical (within 1e-6): {message}"
                results[file_type] = {
                    "match": "close",
                    "message": message,
                    "is_meta_analysis": is_meta_analysis,
                }
                if is_meta_analysis:
                    meta_analysis_match = "close"
            else:
                results[file_type] = {
                    "match": False,
                    "message": message,
                    "is_meta_analysis": is_meta_analysis,
                }
                if is_meta_analysis:
                    meta_analysis_match = False
        else:
            results[file_type] = {
                "match": is_equal,
                "message": message,
                "is_meta_analysis": is_meta_analysis,
            }
            if is_meta_analysis:
                meta_analysis_match = is_equal

    # Determine overall test status
    if meta_analysis_match is None:
        test_status = "NO_META_ANALYSIS"
    elif meta_analysis_match == True:
        test_status = "PASSED"
    elif meta_analysis_match == "close":
        test_status = "PASSED_WITH_WARNING"
    else:
        test_status = "FAILED"

    results["_summary"] = {
        "test_status": test_status,
        "meta_analysis_match": meta_analysis_match,
        "total_files": len(expected_files),
        "matching_files": sum(
            1 for r in results.values() if isinstance(r, dict) and r.get("match") == True
        ),
    }

    return results


def run_evaluation(study):
    """Run evaluation for a single study"""
    print(f"\n{'=' * 60}")
    print(f"Testing: {study}")
    print(f"{'=' * 60}")

    # Get specification from STUDY_SPECS
    spec = STUDY_SPECS.get(study)
    if not spec:
        print(f"ERROR: No specification found for {study}")
        return {"study": study, "status": "no_spec", "error": "No specification found"}

    # Get study mapping info
    study_info = STUDY_MAPPING.get(study, {})
    if not study_info:
        print(f"ERROR: No study mapping found for {study}")
        return {"study": study, "status": "no_mapping", "error": "No study mapping found"}

    # Build paths
    results_path = RESULTS_DIR / study / spec

    # Check for input files using mapping
    input_files = study_info.get("input_files", {})

    # Get the correct file paths from mapping
    # Extract just the filename from the full path template
    human_generic = input_files.get("human_generic", "")
    twin_generic = input_files.get("twin_generic", "")

    # Extract filename from path (after the last /)
    human_filename = (
        human_generic.split("/")[-1]
        if human_generic
        else "consolidated_original_answers_values.csv"
    )
    twin_filename = twin_generic.split("/")[-1] if twin_generic else "consolidated_llm_values.csv"

    # Build full paths
    # Check if human_generic path contains {specification} - if not, it's at study level
    if human_generic and "{specification}" not in human_generic:
        # Human data is at study level
        human_data = RESULTS_DIR / study / human_filename
    else:
        # Human data is in specification folder
        human_data = results_path / human_filename

    # Twin data is always in specification folder
    twin_data = results_path / twin_filename

    # Check if files exist
    if not human_data.exists():
        print(f"ERROR: Human data not found: {human_data}")
        return {"study": study, "status": "missing_data", "error": "Human data not found"}

    if not twin_data.exists():
        print(f"ERROR: Twin data not found: {twin_data}")
        return {"study": study, "status": "missing_data", "error": "Twin data not found"}

    # Run the evaluation script
    script_path = BASE_DIR / study / "mega_study_evaluation.py"
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return {"study": study, "status": "no_script", "error": "Script not found"}

    # Use poetry run from project root
    # Scripts now use --results-dir as both input and output directory
    cmd = [
        "poetry",
        "run",
        "python",
        str(script_path),
        "--results-dir",
        str(results_path),
    ]

    try:
        # Run from project root instead of study directory
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.returncode == 0:
            print(f"✓ Script execution successful!")

            # Save stdout for later
            script_output = result.stdout

            # Check if output files were created
            meta_analysis = results_path / "meta analysis.csv"
            if meta_analysis.exists():
                print(f"  - Created meta analysis.csv")

            # Perform detailed comparison with test files
            # Compare test_files_original (expected) with test_files_new (actual)
            study_info = STUDY_MAPPING.get(study, {})
            test_files_original = study_info.get("test_files_original", {})
            test_files_new = study_info.get("test_files_new", {})

            if test_files_original and test_files_new:
                print(f"\n  Comparing outputs...")
                print(f"  - Expected: test_files_original (notebook outputs)")
                print(f"  - Actual: test_files_new (script outputs)")

                comparison = compare_outputs(test_files_original, test_files_new, tolerance=1e-5)

                if "_summary" in comparison:
                    summary = comparison["_summary"]
                    test_status = summary.get("test_status", "UNKNOWN")

                    # Display comparison results
                    if test_status == "PASSED":
                        print(f"  ✓ COMPARISON PASSED - Meta analysis file matches exactly")
                    elif test_status == "PASSED_WITH_WARNING":
                        print(
                            f"  ⚠ COMPARISON PASSED WITH WARNING - Meta analysis file is close but not identical"
                        )
                    elif test_status == "FAILED":
                        print(f"  ✗ COMPARISON FAILED - Meta analysis file does not match")
                    else:
                        print(f"  ✗ COMPARISON ERROR - {test_status}")

                    # Show details for each file
                    for file_type, result in comparison.items():
                        if file_type.startswith("_"):
                            continue
                        if result["match"] == True:
                            print(f"  ✓ {file_type}: matches")
                        elif result["match"] == "close":
                            print(f"  ⚠ {file_type}: {result['message']}")
                        else:
                            print(f"  ✗ {file_type}: {result['message']}")

                    return {
                        "study": study,
                        "status": "success",
                        "test_status": test_status,
                        "comparison": comparison,
                        "output": script_output,
                    }
                else:
                    print(f"  - No comparison summary available")
                    return {
                        "study": study,
                        "status": "success",
                        "test_status": "NO_COMPARISON",
                        "output": script_output,
                    }
            else:
                print(f"  - No test files configured for comparison")
                return {
                    "study": study,
                    "status": "success",
                    "test_status": "NO_TEST_CONFIG",
                    "output": script_output,
                }
        else:
            print(f"✗ Failed with return code {result.returncode}")
            print(f"Error: {result.stderr[:500]}...")  # First 500 chars of error
            return {"study": study, "status": "failed", "error": result.stderr}

    except Exception as e:
        print(f"✗ Exception: {e}")
        return {"study": study, "status": "exception", "error": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test mega_study_evaluation.py scripts")
    parser.add_argument("--study", help="Test a specific study only")
    args = parser.parse_args()
    
    results = []
    
    # Determine which studies to test
    if args.study:
        if args.study not in STUDIES:
            print(f"ERROR: Study '{args.study}' not found in STUDIES list")
            sys.exit(1)
        studies_to_test = [args.study]
    else:
        studies_to_test = STUDIES

    # Run evaluations
    for study in studies_to_test:
        result = run_evaluation(study)
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    # Count different statuses
    success_count = sum(1 for r in results if r["status"] == "success")
    test_passed = sum(1 for r in results if r.get("test_status") == "PASSED")
    test_passed_warning = sum(1 for r in results if r.get("test_status") == "PASSED_WITH_WARNING")
    test_failed = sum(1 for r in results if r.get("test_status") == "FAILED")

    print(f"\nTotal studies: {len(STUDIES)}")
    print(f"Script execution successful: {success_count}")
    print(f"Script execution failed: {len(STUDIES) - success_count}")

    if success_count > 0:
        print(f"\nOf the successful executions:")
        print(f"  - Test comparisons passed: {test_passed}")
        print(f"  - Test comparisons passed with warnings: {test_passed_warning}")
        print(f"  - Test comparisons failed: {test_failed}")
        print(
            f"  - No test comparison available: {success_count - test_passed - test_passed_warning - test_failed}"
        )

    # Group by status
    by_status = {}
    for r in results:
        status = r["status"]
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(r["study"])

    for status, studies_list in by_status.items():
        print(f"\n{status.upper()} ({len(studies_list)}):")
        for study in studies_list:
            # Find the result for this study to show test status
            study_result = next((r for r in results if r["study"] == study), None)
            test_status = study_result.get("test_status", "N/A") if study_result else "N/A"
            if test_status != "N/A" and test_status != "NO_TEST_DIR":
                print(f"  - {study} (comparison: {test_status})")
            else:
                print(f"  - {study}")

    # Save results
    results_file = TEST_OUTPUT_DIR / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
