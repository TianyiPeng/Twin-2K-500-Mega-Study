#!/usr/bin/env python3
"""Clean up ML config files by removing default exclude columns."""

import glob

import yaml

# Default columns that are now handled in the script
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


def clean_ml_config(file_path):
    """Remove default exclude columns from ML config file."""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    if "ml_simulation" not in config:
        return False

    ml_sim = config["ml_simulation"]
    if "prediction_targets" not in ml_sim:
        return False

    pred_targets = ml_sim["prediction_targets"]
    if "exclude_columns" not in pred_targets:
        return False

    # Get current exclude columns
    current_excludes = pred_targets["exclude_columns"]
    if not isinstance(current_excludes, list):
        return False

    # Filter out default columns, keeping only study-specific ones
    new_excludes = []
    for col in current_excludes:
        if col not in DEFAULT_EXCLUDE_COLUMNS:
            new_excludes.append(col)

    # Update the config
    if len(new_excludes) != len(current_excludes):
        if new_excludes:
            pred_targets["exclude_columns"] = new_excludes
        else:
            # If no study-specific exclusions remain, remove the key
            del pred_targets["exclude_columns"]

        # Write back to file
        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(
            f"Cleaned {file_path}: removed {len(current_excludes) - len(new_excludes)} default exclusions"
        )
        return True

    return False


def main():
    # Find all *_ml.yaml files
    ml_configs = glob.glob("configs/**/*_ml.yaml", recursive=True)

    cleaned_count = 0
    for config_path in ml_configs:
        if clean_ml_config(config_path):
            cleaned_count += 1

    print(f"\nTotal files cleaned: {cleaned_count}/{len(ml_configs)}")


if __name__ == "__main__":
    main()
