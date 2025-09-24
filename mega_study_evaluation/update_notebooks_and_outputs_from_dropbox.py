#!/usr/bin/env python3
"""Update notebooks and their output CSV files from mega_study_data to mega_study_evaluation folders.

This script:
1. Scans mega_study_data folders for Jupyter notebooks
2. Copies notebooks to corresponding mega_study_evaluation folders
3. Identifies and copies CSV output files to test/ subdirectories
4. Checks for existence of mega_study_evaluation.py scripts
5. Generates a summary report with output file mappings
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Define expected output CSV patterns for each study
STUDY_OUTPUT_PATTERNS = {
    "accuracy_nudges": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "affective_priming": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "consumer_minimalism": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "context_effects": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "default_eric": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "digital_certification": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "hiring_algorithms": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "idea_evaluation": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "idea_generation": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "infotainment": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "junk_fees": [
        "*twins data values anonymized processed.csv",
        "*human data values anonymized processed.csv",
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "obedient_twins": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "preference_redistribution": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "privacy": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "promiscuous_donors": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "quantitative_intuition": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "recommendation_algorithms": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "story_beliefs": [
        "*twins data values anonymized processed.csv",
        "*human data values anonymized processed.csv",
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
    "targeting_fairness": [
        "*default persona meta analysis.csv",
        "*default persona meta analysis individual level.csv",
    ],
}


def load_mapping(mapping_file: str) -> Dict[str, str]:
    """Load the study folder mapping from JSON file."""
    with open(mapping_file, "r") as f:
        return json.load(f)


def load_extended_mapping(mapping_file: str) -> Dict:
    """Load the extended mapping with input/output file specifications."""
    extended_file = mapping_file.parent / "study_folder_mapping_extended.json"
    if extended_file.exists():
        with open(extended_file, "r") as f:
            return json.load(f)
    return {}


def save_extended_mapping(mapping_file: str, extended_mapping: Dict) -> None:
    """Save the extended mapping with output files."""
    output_file = mapping_file.parent / "study_folder_mapping_with_outputs.json"
    with open(output_file, "w") as f:
        json.dump(extended_mapping, f, indent=2)
    print(f"\nExtended mapping saved to: {output_file}")


def find_notebooks(folder_path: Path) -> List[Path]:
    """Find all Jupyter notebooks directly in the folder (not in subfolders).

    If multiple notebooks are found, prioritize those with 'meta analysis' in the name.
    """
    notebooks = []
    for path in folder_path.glob("*.ipynb"):
        # Skip checkpoint notebooks
        if ".ipynb_checkpoints" not in str(path):
            notebooks.append(path)

    # If multiple notebooks found, prioritize meta analysis notebooks
    if len(notebooks) > 1:
        meta_analysis_notebooks = [nb for nb in notebooks if "meta analysis" in nb.name.lower()]
        if meta_analysis_notebooks:
            return meta_analysis_notebooks

    return notebooks


def find_csv_outputs(data_path: Path, patterns: List[str]) -> List[Path]:
    """Find CSV files matching the given patterns in the data folder."""
    csv_files = []
    for pattern in patterns:
        # Search in main folder
        csv_files.extend(data_path.glob(pattern))
        # Also search in evaluation subfolder if it exists
        eval_subfolder = data_path / "evaluation"
        if eval_subfolder.exists():
            csv_files.extend(eval_subfolder.glob(pattern))

    # Remove duplicates and return
    return list(set(csv_files))


def copy_notebook(source: Path, destination_folder: Path) -> Path:
    """Copy a notebook to the destination folder."""
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination = destination_folder / source.name
    shutil.copy2(source, destination)
    return destination


def copy_csv_files(csv_files: List[Path], test_folder: Path) -> List[str]:
    """Copy CSV files to the test folder and return list of copied filenames."""
    test_folder.mkdir(parents=True, exist_ok=True)
    copied_files = []

    for csv_file in csv_files:
        try:
            destination = test_folder / csv_file.name
            shutil.copy2(csv_file, destination)
            copied_files.append(csv_file.name)
            print(f"    Copied CSV: {csv_file.name}")
        except Exception as e:
            print(f"    ERROR copying CSV {csv_file.name}: {e}")

    return copied_files


def copy_input_files(
    data_path: Path, eval_path: Path, input_specs: Dict, results_path: Path = None
) -> Dict[str, str]:
    """Copy original input data files to evaluation folder and optionally human data to results folder."""
    copied_files = {}

    # Copy human data file
    if "human_original" in input_specs:
        human_file = input_specs["human_original"]
        source = data_path / human_file
        if source.exists():
            try:
                # Copy to evaluation folder
                destination = eval_path / human_file
                shutil.copy2(source, destination)
                copied_files["human_original"] = human_file
                print(f"    Copied human data to eval: {human_file}")

                # Also copy to results folder if specified (only human data)
                if results_path:
                    # Make sure the results directory exists
                    results_path.mkdir(parents=True, exist_ok=True)
                    results_dest = results_path / human_file
                    shutil.copy2(source, results_dest)
                    print(f"    Copied human data to results: {human_file}")
            except Exception as e:
                print(f"    ERROR copying human data {human_file}: {e}")
        else:
            print(f"    WARNING: Human data file not found: {human_file}")

    # Copy twin data file (only to evaluation folder, not to results)
    if "twin_original" in input_specs:
        twin_file = input_specs["twin_original"]
        source = data_path / twin_file
        if source.exists():
            try:
                # Copy to evaluation folder only
                destination = eval_path / twin_file
                shutil.copy2(source, destination)
                copied_files["twin_original"] = twin_file
                print(f"    Copied twin data to eval: {twin_file}")
            except Exception as e:
                print(f"    ERROR copying twin data {twin_file}: {e}")
        else:
            print(f"    WARNING: Twin data file not found: {twin_file}")

    # Copy additional files if specified (e.g., creativity ratings for idea_generation)
    # Note: Additional files are only copied to evaluation folder, not to results
    if "additional_files" in input_specs:
        for additional_file in input_specs["additional_files"]:
            source = data_path / additional_file
            if source.exists():
                try:
                    # Copy to evaluation folder only
                    destination = eval_path / additional_file
                    shutil.copy2(source, destination)
                    print(f"    Copied additional file to eval: {additional_file}")
                except Exception as e:
                    print(f"    ERROR copying additional file {additional_file}: {e}")
            else:
                print(f"    WARNING: Additional file not found: {additional_file}")

    return copied_files


def check_python_script_exists(folder: Path) -> bool:
    """Check if mega_study_evaluation.py exists in the folder."""
    script_path = folder / "mega_study_evaluation.py"
    return script_path.exists()


def generate_summary(results: List[Dict]) -> pd.DataFrame:
    """Generate a summary DataFrame from the results."""
    return pd.DataFrame(results)


def main():
    # Set up paths
    project_root = Path(__file__).parent.parent
    mega_study_data = project_root / "mega_study_data"
    mega_study_evaluation = project_root / "mega_study_evaluation"
    mapping_file = mega_study_evaluation / "study_folder_mapping.json"

    # Load full mapping
    with open(mapping_file, "r") as f:
        full_mapping = json.load(f)

    # Results list for summary
    results = []
    extended_mapping = {}

    # Process each study
    for eval_folder, study_info in full_mapping.items():
        # Skip _comments entry if present
        if eval_folder == "_comments":
            continue
        print(f"\nProcessing {eval_folder}...")

        # Extract data folder from study info
        data_folder = study_info.get("data_folder", "")
        if not data_folder:
            print(f"  WARNING: No data_folder found for {eval_folder}")
            continue

        # Paths
        data_path = mega_study_data / data_folder
        eval_path = mega_study_evaluation / eval_folder
        test_path = eval_path / "test"

        # Initialize result entry
        result = {
            "study": eval_folder,
            "data_folder": data_folder,
            "notebooks_found": 0,
            "notebooks_copied": 0,
            "notebook_names": [],
            "csv_files_found": 0,
            "csv_files_copied": 0,
            "csv_file_names": [],
            "python_script_exists": False,
            "test_status": "Not tested",
        }

        # Initialize extended mapping entry
        extended_mapping[eval_folder] = {
            "data_folder": data_folder,
            "notebooks": [],
            "output_files": [],
        }

        # Check if data folder exists
        if not data_path.exists():
            print(f"  WARNING: Data folder not found: {data_path}")
            result["test_status"] = "Data folder not found"
            results.append(result)
            continue

        # Find and copy notebooks
        notebooks = find_notebooks(data_path)
        result["notebooks_found"] = len(notebooks)

        if notebooks:
            print(f"  Found {len(notebooks)} notebook(s)")
            for notebook in notebooks:
                # Copy notebook
                try:
                    dest = copy_notebook(notebook, eval_path)
                    result["notebooks_copied"] += 1
                    result["notebook_names"].append(notebook.name)
                    extended_mapping[eval_folder]["notebooks"].append(notebook.name)
                    print(f"  Copied: {notebook.name}")
                except Exception as e:
                    print(f"  ERROR copying {notebook.name}: {e}")
        else:
            print(f"  No notebooks found")

        # Copy input data files if we have extended info
        if "input_files" in study_info:
            print(f"  Copying input data files...")
            input_specs = study_info["input_files"]

            # Find the results directory for this study (not specification subdirs)
            results_study_dir = project_root / "results" / eval_folder

            # Copy human data to study-level results directory
            copied_input_files = copy_input_files(
                data_path, eval_path, input_specs, results_study_dir
            )

            if copied_input_files:
                extended_mapping[eval_folder]["input_files"] = copied_input_files

        # Find and copy CSV output files
        patterns = STUDY_OUTPUT_PATTERNS.get(eval_folder, [])
        if patterns:
            print(f"  Looking for CSV output files...")
            csv_files = find_csv_outputs(data_path, patterns)
            result["csv_files_found"] = len(csv_files)

            if csv_files:
                print(f"  Found {len(csv_files)} CSV file(s)")
                copied_files = copy_csv_files(csv_files, test_path)
                result["csv_files_copied"] = len(copied_files)
                result["csv_file_names"] = copied_files
                extended_mapping[eval_folder]["output_files"] = copied_files
            else:
                print(f"  No CSV output files found")

        # Check for Python script
        result["python_script_exists"] = check_python_script_exists(eval_path)
        if result["python_script_exists"]:
            print(f"  ✓ mega_study_evaluation.py exists")
        else:
            print(f"  ✗ mega_study_evaluation.py not found")

        results.append(result)

    # Save extended mapping
    save_extended_mapping(mapping_file, extended_mapping)

    # Generate summary
    summary_df = generate_summary(results)

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = mega_study_evaluation / f"notebook_and_output_update_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal studies processed: {len(results)}")
    print(f"Studies with notebooks: {sum(1 for r in results if r['notebooks_found'] > 0)}")
    print(f"Studies with CSV outputs: {sum(1 for r in results if r['csv_files_found'] > 0)}")
    print(f"Studies with Python scripts: {sum(1 for r in results if r['python_script_exists'])}")
    print(f"\nDetailed summary saved to: {summary_file}")

    # Print detailed table
    print("\n" + "-" * 100)
    print(f"{'Study':<25} {'Notebooks':<12} {'CSV Files':<12} {'Script':<10} {'Status':<20}")
    print("-" * 100)
    for result in results:
        print(
            f"{result['study']:<25} "
            f"{result['notebooks_found']:<12} "
            f"{result['csv_files_found']:<12} "
            f"{'Yes' if result['python_script_exists'] else 'No':<10} "
            f"{result['test_status']:<20}"
        )

    return summary_df


if __name__ == "__main__":
    summary = main()
