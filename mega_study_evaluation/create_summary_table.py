#!/usr/bin/env python3
"""Create summary tables with persona specifications in rows and performance metrics in columns."""

from pathlib import Path

import numpy as np
import pandas as pd


def create_summary_tables():
    """Create summary tables from the combined meta-analysis CSV."""
    # Load the combined data
    project_root = Path.cwd()
    input_file = (
        project_root
        / "mega_study_evaluation"
        / "meta_analysis_results"
        / "combined_all_specifications_meta_analysis.csv"
    )

    if not input_file.exists():
        print(f"Error: Combined file not found at {input_file}")
        print("Please run combine_all_meta_analyses.py first.")
        return

    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)

    # Output directory
    output_dir = project_root / "mega_study_evaluation" / "meta_analysis_results"

    # Key metrics to include in the summary
    metrics = [
        "correlation between the responses from humans vs. their twins",
        "accuracy between humans vs. their twins",
        "paired t-test t-stat",
        "paired t-test p-value",
        "variance test F-stat",
        "variance test p-value",
    ]

    # 1. Create a wide summary table: Average metrics by persona specification
    print("\nCreating summary table by persona specification...")

    # Group by persona specification and calculate mean for each metric
    summary_avg = df.groupby("persona specification")[metrics].mean()

    # Add count of observations
    summary_avg["n_observations"] = df.groupby("persona specification").size()

    # Round for readability
    summary_avg = summary_avg.round(4)

    # Save the average summary
    avg_output_file = output_dir / "summary_by_persona_specification_avg.csv"
    summary_avg.to_csv(avg_output_file)
    print(f"Saved average summary to: {avg_output_file}")

    # 2. Create a more detailed summary with study breakdown
    print("\nCreating detailed summary table by persona specification and study...")

    # Pivot table with studies as columns for correlation metric
    correlation_by_study = df.pivot_table(
        values="correlation between the responses from humans vs. their twins",
        index="persona specification",
        columns="study name",
        aggfunc="mean",
    ).round(4)

    # Save the correlation by study table
    corr_output_file = output_dir / "correlation_by_persona_and_study.csv"
    correlation_by_study.to_csv(corr_output_file)
    print(f"Saved correlation by study to: {corr_output_file}")

    # 3. Create a summary with median values (more robust to outliers)
    print("\nCreating median summary table...")

    summary_median = df.groupby("persona specification")[metrics].median()
    summary_median["n_observations"] = df.groupby("persona specification").size()
    summary_median = summary_median.round(4)

    median_output_file = output_dir / "summary_by_persona_specification_median.csv"
    summary_median.to_csv(median_output_file)
    print(f"Saved median summary to: {median_output_file}")

    # 4. Create a comparison table showing which specification performs best
    print("\nCreating performance comparison table...")

    # For each metric, find which persona specification has the best value
    comparison_data = []

    for metric in metrics:
        metric_data = {"metric": metric}

        # Get mean values for each persona specification
        means = df.groupby("persona specification")[metric].mean()

        # Determine if higher or lower is better
        if "p-value" in metric:
            # For p-values, we typically want them to be low (significant)
            # But for our comparison, we'll just report the values
            best_spec = means.idxmin()
            metric_data["best_specification"] = best_spec
            metric_data["interpretation"] = "lowest value"
        elif "correlation" in metric or "accuracy" in metric:
            # Higher correlation/accuracy is better
            best_spec = means.idxmax()
            metric_data["best_specification"] = best_spec
            metric_data["interpretation"] = "highest value"
        else:
            # For other metrics, just report which has the highest absolute value
            best_spec = means.abs().idxmax()
            metric_data["best_specification"] = best_spec
            metric_data["interpretation"] = "highest absolute value"

        # Add the actual values for each specification
        for spec in means.index:
            metric_data[f"{spec}_value"] = round(means[spec], 4)

        comparison_data.append(metric_data)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_output_file = output_dir / "performance_comparison_by_metric.csv"
    comparison_df.to_csv(comparison_output_file, index=False)
    print(f"Saved performance comparison to: {comparison_output_file}")

    # Print some insights
    print("\n=== Summary Statistics ===")
    print(f"\nPersona specifications found: {sorted(df['persona specification'].unique())}")
    print(f"\nNumber of rows per specification:")
    print(df.groupby("persona specification").size().sort_values(ascending=False))

    print("\n=== Average Correlation by Specification ===")
    avg_corr = df.groupby("persona specification")[
        "correlation between the responses from humans vs. their twins"
    ].mean()
    print(avg_corr.sort_values(ascending=False).round(4))

    # Answer the specific questions
    print("\n=== Answers to Your Questions ===")
    print("\n1. Mapping between persona specification and specification_name_full:")
    mapping = df[["persona specification", "specification_name_full"]].drop_duplicates()
    print(f"   - There are {len(mapping)} unique combinations")
    print(
        f"   - This is {'NOT ' if len(mapping) > len(df['persona specification'].unique()) else ''}a 1-to-1 mapping"
    )

    if len(mapping) > len(df["persona specification"].unique()):
        print("\n   Multiple specification_name_full values per persona specification:")
        for spec in sorted(df["persona specification"].unique()):
            full_names = df[df["persona specification"] == spec]["specification_name_full"].unique()
            if len(full_names) > 1:
                print(f"   - {spec}: {full_names}")

    print("\n2. Number of rows per persona specification:")
    rows_per_spec = df.groupby("persona specification").size()
    print(f"   - Same number of rows: {'Yes' if rows_per_spec.nunique() == 1 else 'No'}")
    print(f"   - Row counts: {dict(rows_per_spec)}")

    return summary_avg, correlation_by_study, summary_median, comparison_df


if __name__ == "__main__":
    create_summary_tables()
