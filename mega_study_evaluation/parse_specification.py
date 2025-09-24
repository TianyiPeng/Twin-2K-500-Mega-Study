#!/usr/bin/env python3
"""Utility function to parse specification names consistently across all scripts."""

import re
from pathlib import Path


def parse_specification_name(spec_name):
    """Parse specification name to extract type and date.

    Args:
        spec_name: Specification directory name (e.g., "full_persona_without_reasoning_2025-06-18")

    Returns:
        tuple: (spec_type, ran_date)
            - spec_type: The specification type without date (e.g., "full_persona_without_reasoning")
            - ran_date: The date part (e.g., "2025-06-18") or None if no date found
    """
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


def extract_specification_from_path(results_dir_path):
    """Extract specification name from results directory path.

    Args:
        results_dir_path: Path to results directory (e.g., "results/study_name/specification_name")

    Returns:
        str: The specification directory name, or None if cannot be determined
    """
    if not results_dir_path:
        return None

    path_parts = Path(results_dir_path).parts

    # Standard structure is results/{study_name}/{specification_name}
    if len(path_parts) >= 3 and path_parts[-3] == "results":
        return path_parts[-1]

    # Sometimes the path might be relative to the study
    if len(path_parts) >= 1:
        return path_parts[-1]

    return None
