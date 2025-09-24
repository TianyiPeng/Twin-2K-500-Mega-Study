#!/usr/bin/env python3
"""Run all mega_study_evaluation scripts for all available study/specification combinations to generate meta-analyses."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Base paths
PROJECT_ROOT = Path.cwd() if Path.cwd().name != "mega_study_evaluation" else Path.cwd().parent
BASE_DIR = PROJECT_ROOT / "mega_study_evaluation"
RESULTS_DIR = PROJECT_ROOT / "results"

# Load study mapping to understand which studies exist
MAPPING_FILE = BASE_DIR / "study_folder_mapping.json"
with open(MAPPING_FILE, "r") as f:
    STUDY_MAPPING = json.load(f)

# Get list of studies (excluding _comments)
STUDIES = [k for k in STUDY_MAPPING.keys() if k not in ["_comments"]]


def find_all_specifications():
    """Find all study/specification combinations in the results directory."""
    study_specs = {}

    print("Scanning results directory for available specifications...")

    for study in STUDIES:
        study_dir = RESULTS_DIR / study
        if not study_dir.exists():
            continue

        specs = []
        for spec_dir in study_dir.iterdir():
            if spec_dir.is_dir() and not spec_dir.name.startswith("."):
                # Check if data files exist
                has_data = False

                # Check for various data file patterns based on study mapping
                study_info = STUDY_MAPPING.get(study, {})
                input_files = study_info.get("input_files", {})

                # Get expected filenames
                human_generic = input_files.get("human_generic", "")
                twin_generic = input_files.get("twin_generic", "")

                # Extract filenames
                if "{specification}" not in human_generic:
                    # Human data at study level
                    human_filename = human_generic.split("/")[-1] if human_generic else ""
                    human_path = study_dir / human_filename
                else:
                    # Human data in spec folder
                    human_filename = (
                        human_generic.split("/")[-1]
                        if human_generic
                        else "consolidated_original_answers_values.csv"
                    )
                    human_path = spec_dir / human_filename

                twin_filename = (
                    twin_generic.split("/")[-1] if twin_generic else "consolidated_llm_values.csv"
                )
                twin_path = spec_dir / twin_filename

                # Check if both files exist
                if human_path.exists() and twin_path.exists():
                    has_data = True
                    specs.append(spec_dir.name)
                    print(f"  Found: {study}/{spec_dir.name}")
                elif not human_path.exists():
                    print(f"  Missing human data: {study}/{spec_dir.name} - {human_path.name}")
                elif not twin_path.exists():
                    print(f"  Missing twin data: {study}/{spec_dir.name} - {twin_path.name}")

        if specs:
            study_specs[study] = specs

    return study_specs


def run_evaluation(study, specification, force=False):
    """Run evaluation for a single study/specification combination.

    Args:
        study: Study name
        specification: Specification name
        force: If True, rerun even if meta analysis already exists
    """
    script_path = BASE_DIR / study / "mega_study_evaluation.py"
    results_path = RESULTS_DIR / study / specification

    if not script_path.exists():
        return {
            "study": study,
            "specification": specification,
            "status": "error",
            "error": f"Script not found: {script_path}",
        }

    # Check if meta analysis already exists
    meta_analysis_file = results_path / "meta analysis.csv"
    if meta_analysis_file.exists() and not force:
        return {
            "study": study,
            "specification": specification,
            "status": "skipped",
            "message": "Meta analysis already exists",
        }

    # Run the evaluation
    cmd = ["poetry", "run", "python", str(script_path), "--results-dir", str(results_path)]

    try:
        print(f"  Running: {' '.join(cmd[-3:])}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.returncode == 0:
            # Check if output was created
            if meta_analysis_file.exists():
                return {
                    "study": study,
                    "specification": specification,
                    "status": "success",
                    "message": "Meta analysis created successfully",
                }
            else:
                return {
                    "study": study,
                    "specification": specification,
                    "status": "warning",
                    "message": "Script completed but no meta analysis file created",
                }
        else:
            return {
                "study": study,
                "specification": specification,
                "status": "failed",
                "error": result.stderr[:500],  # First 500 chars of error
            }

    except Exception as e:
        return {
            "study": study,
            "specification": specification,
            "status": "exception",
            "error": str(e),
        }


def main():
    """Main function to run all evaluations."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run all mega study evaluations to generate meta-analyses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force rerun even if meta analysis files already exist",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Running all mega study evaluations")
    if args.force:
        print("Mode: FORCE RERUN (will overwrite existing meta analysis files)")
    print("=" * 80)

    # Find all available specifications
    study_specs = find_all_specifications()

    if not study_specs:
        print("\nNo study/specification combinations found!")
        return

    # Count total combinations
    total_combinations = sum(len(specs) for specs in study_specs.values())
    print(
        f"\nFound {total_combinations} study/specification combinations across {len(study_specs)} studies"
    )

    # Run evaluations
    results = []
    completed = 0

    print("\nRunning evaluations...")
    print("-" * 80)

    for study, specifications in sorted(study_specs.items()):
        print(f"\n{study}:")
        for spec in sorted(specifications):
            completed += 1
            print(f"\n[{completed}/{total_combinations}] {study}/{spec}")

            result = run_evaluation(study, spec, force=args.force)
            results.append(result)

            # Print result
            if result["status"] == "success":
                print(f"  ✓ Success: {result['message']}")
            elif result["status"] == "skipped":
                print(f"  - Skipped: {result['message']}")
            elif result["status"] == "warning":
                print(f"  ⚠ Warning: {result['message']}")
            elif result["status"] == "failed":
                print(f"  ✗ Failed: {result['error']}")
            else:
                print(f"  ✗ Error: {result.get('error', 'Unknown error')}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Count by status
    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"\nTotal combinations processed: {len(results)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Show failed evaluations
    failed = [r for r in results if r["status"] in ["failed", "error", "exception"]]
    if failed:
        print(f"\nFailed evaluations ({len(failed)}):")
        for r in failed:
            print(
                f"  - {r['study']}/{r['specification']}: {r.get('error', 'Unknown error')[:100]}..."
            )

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = BASE_DIR / f"run_all_meta_analyses_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    # Suggest next step
    if status_counts.get("success", 0) > 0:
        print("\n" + "=" * 80)
        print("NEXT STEP")
        print("=" * 80)
        print("\nTo combine all meta-analysis results, run:")
        print("poetry run python mega_study_evaluation/combine_all_meta_analyses.py")


if __name__ == "__main__":
    main()
