#!/usr/bin/env python3
"""Master script to update all meta-analysis files after running evaluations.

This script orchestrates the complete update process:
1. Runs meta-analyses for all study/specification combinations
2. Combines all results into a single CSV
3. Creates summary tables with various aggregations
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Success")
        if result.stdout:
            print(f"Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with error code {e.returncode}")
        if e.stdout:
            print(f"Stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")
        return False


def main():
    """Run the complete update process."""
    # Get the project root
    project_root = Path.cwd()

    # Verify we're in the right directory
    if not (project_root / "mega_study_evaluation").exists():
        print("Error: This script must be run from the project root directory")
        print("Current directory:", project_root)
        sys.exit(1)

    print(
        f"Starting meta-analysis update process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Project root: {project_root}")

    # Track overall success
    all_successful = True

    # Step 1: Run all individual meta-analyses
    cmd = [sys.executable, "mega_study_evaluation/run_all_meta_analyses.py"]
    success = run_command(
        cmd, "Step 1: Running individual meta-analyses for all study/specification combinations"
    )
    all_successful = all_successful and success

    if not success:
        print("\nWarning: Some individual meta-analyses may have failed.")
        print("Continuing with available results...")

    # Step 2: Combine all meta-analysis results
    cmd = [sys.executable, "mega_study_evaluation/combine_all_meta_analyses.py"]
    success = run_command(cmd, "Step 2: Combining all meta-analysis results into a single CSV")
    all_successful = all_successful and success

    if not success:
        print("\nError: Failed to combine meta-analysis results.")
        print("Cannot proceed with summary table creation.")
        sys.exit(1)

    # Step 3: Create summary tables
    cmd = [sys.executable, "mega_study_evaluation/create_summary_table.py"]
    success = run_command(
        cmd, "Step 3: Creating summary tables with persona specifications as rows"
    )
    all_successful = all_successful and success

    # Final summary
    print(f"\n{'=' * 60}")
    print("Update Process Complete")
    print(f"{'=' * 60}")

    # Check what files were created
    output_dir = project_root / "mega_study_evaluation" / "meta_analysis_results"
    if output_dir.exists():
        print("\nFiles in meta_analysis_results directory:")
        files = sorted(output_dir.glob("*.csv"))
        for f in files:
            size = f.stat().st_size / 1024  # KB
            print(f"  - {f.name} ({size:.1f} KB)")

    # Save a timestamp of the last update
    timestamp_file = output_dir / "last_update.json"
    timestamp_data = {
        "last_update": datetime.now().isoformat(),
        "all_successful": all_successful,
        "files_generated": [f.name for f in files] if output_dir.exists() else [],
    }

    with open(timestamp_file, "w") as f:
        json.dump(timestamp_data, f, indent=2)

    print(f"\nLast update timestamp saved to: {timestamp_file}")

    if all_successful:
        print("\n✓ All steps completed successfully!")
    else:
        print("\n⚠ Some steps had warnings or errors. Check the output above.")

    return 0 if all_successful else 1


if __name__ == "__main__":
    sys.exit(main())
