#!/usr/bin/env python3
"""Simple script to combine all meta-analysis results across studies and specifications."""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_specification_name(spec_name):
    """Parse specification name to extract type and date."""
    # Handle typo in preference_redistribution
    spec_name = spec_name.replace("resoning", "reasoning")

    # Try to extract date pattern YYYY-MM-DD at the end
    date_pattern = r"(\d{4}-\d{2}-\d{2})$"
    date_match = re.search(date_pattern, spec_name)

    if date_match:
        ran_date = date_match.group(1)
        spec_type = spec_name[: date_match.start()].rstrip("_")
    else:
        # No date found
        ran_date = None
        spec_type = spec_name

    return spec_type, ran_date


# Find all meta analysis.csv files in results directory
project_root = Path.cwd()
results_dir = project_root / "results"

all_data = []

print("Searching for meta analysis files...")
for study_dir in results_dir.iterdir():
    if study_dir.is_dir() and not study_dir.name.startswith("."):
        for spec_dir in study_dir.iterdir():
            if spec_dir.is_dir() and not spec_dir.name.startswith("."):
                meta_file = spec_dir / "meta analysis.csv"
                if meta_file.exists():
                    try:
                        df = pd.read_csv(meta_file)

                        # Add study name if not present
                        if "study_name" not in df.columns and "study name" not in df.columns:
                            df["study_name"] = study_dir.name

                        # Add specification info
                        spec_type, ran_date = parse_specification_name(spec_dir.name)

                        # If the dataframe doesn't have these columns, add them
                        if "persona specification" not in df.columns:
                            df["persona specification"] = spec_type
                        if "specification_name_full" not in df.columns:
                            df["specification_name_full"] = spec_dir.name
                        if "ran_date" not in df.columns:
                            df["ran_date"] = ran_date

                        all_data.append(df)
                        print(f"  Found: {study_dir.name}/{spec_dir.name} ({len(df)} rows)")
                    except Exception as e:
                        print(f"  Error reading {meta_file}: {e}")

if all_data:
    # Combine all data
    print(f"\nCombining {len(all_data)} files...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save without timestamp
    output_dir = project_root / "mega_study_evaluation" / "meta_analysis_results"
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / "combined_all_specifications_meta_analysis.csv"
    combined_df.to_csv(output_file, index=False)

    print(f"\nSaved combined data to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"\nColumns: {', '.join(combined_df.columns[:10])}...")

    # Summary by specification type
    if "specification_type" in combined_df.columns:
        print("\nSummary by specification type:")
        spec_summary = combined_df.groupby("specification_type").size()
        for spec_type, count in spec_summary.items():
            print(f"  {spec_type}: {count} rows")
else:
    print("No meta analysis files found!")
