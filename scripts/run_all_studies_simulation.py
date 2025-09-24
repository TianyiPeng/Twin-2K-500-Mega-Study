#!/usr/bin/env python3
"""Run Snakemake for All Studies.

This script runs snakemake for studies in configs/ using configurable suffix patterns.

Usage:
    python run_all_studies_simulation.py --suffix demographics_only [--cores 3]
    python run_all_studies_simulation.py --suffix empty_persona [--cores 8]
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path
import time


# List of all studies (subfolder names in configs/)
STUDIES = [
    "accuracy_nudges",
    "affective_priming",
    "consumer_minimalism",
    "context_effects",
    "default_eric",
    "digital_certification",
    "hiring_algorithms",
    "idea_evaluation",
    "idea_generation",
    "infotainment",
    "junk_fees",
    "obedient_twins",
    "preference_redistribution",
    "privacy",
    "promiscuous_donors",
    "quantitative_intuition",
    "recommendation_algorithms",
    "story_beliefs",
    "targeting_fairness",
]


def run_snakemake_for_study(study: str, suffix: str, cores: int, dry_run: bool = False) -> bool:
    """Run snakemake for a specific study with the given suffix.

    Args:
        study: Study name (subfolder in configs/)
        suffix: Config file suffix (e.g., "demographics_only", "empty_persona")
        cores: Number of cores to use
        dry_run: If True, only show what would be run

    Returns:
        True if successful, False otherwise
    """
    config_file = f"configs/{study}/{study}_{suffix}.yaml"

    # Check if config file exists
    if not Path(config_file).exists():
        print(f"⚠ Skipping {study} - config file not found: {config_file}")
        return None  # Neither success nor failure, just skipped

    cmd = [
        "poetry",
        "run",
        "snakemake",
        "--configfile",
        config_file,
        "--cores",
        str(cores),
        "--forceall",
        "--rerun-incomplete",
    ]

    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return True

    print(f"Running: {' '.join(cmd)}")

    try:
        start_time = time.time()
        _ = subprocess.run(cmd, check=True)
        end_time = time.time()

        duration = end_time - start_time
        print(f"✓ Completed {study} in {duration:.1f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed {study}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted while running {study}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run snakemake for all studies with configurable suffix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all demographics_only configs with 3 cores
    python run_all_studies_simulation.py --suffix demographics_only

    # Run all empty_persona configs with 8 cores
    python run_all_studies_simulation.py --suffix empty_persona --cores 8

    # Dry run to see what would be executed
    python run_all_studies_simulation.py --suffix demographics_only --dry-run
        """,
    )

    parser.add_argument(
        "--suffix",
        required=True,
        help='Config file suffix (e.g., "demographics_only", "empty_persona", "ml")',
    )

    parser.add_argument("--cores", type=int, default=3, help="Number of cores to use (default: 3)")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands that would be run without executing them",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running other studies even if one fails",
    )

    parser.add_argument(
        "--studies", nargs="*", help="Specific studies to run (default: all studies)"
    )

    args = parser.parse_args()

    # Determine which studies to run
    studies_to_run = args.studies if args.studies else STUDIES

    print(
        f"Running snakemake for {len(studies_to_run)} studies with suffix '{args.suffix}' using {args.cores} cores..."
    )
    if args.dry_run:
        print("=== DRY RUN MODE ===")
    print("=" * 50)

    # Run snakemake for each study
    successful = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for i, study in enumerate(studies_to_run, 1):
        print(f"\n[{i}/{len(studies_to_run)}] Processing {study}...")

        result = run_snakemake_for_study(study, args.suffix, args.cores, args.dry_run)

        if result is True:
            successful += 1
        elif result is False:
            failed += 1
            if not args.continue_on_error and not args.dry_run:
                response = input("Continue with remaining studies? (y/n): ").lower().strip()
                if response not in ["y", "yes"]:
                    print("Stopping execution.")
                    break
        else:  # result is None (skipped)
            skipped += 1

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  ⚠ Skipped: {skipped}")
    print(f"  Total time: {total_time:.1f} seconds")
    print("=" * 50)

    if failed > 0 and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
