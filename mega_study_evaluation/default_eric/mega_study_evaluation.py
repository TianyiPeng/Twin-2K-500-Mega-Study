#!/usr/bin/env python
# coding: utf-8
"""
Meta-analysis script for Defaults (default_eric)
Converted from: Defaults meta analysis.ipynb
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.patches import Patch
from scipy.stats import f, norm, pearsonr, ttest_rel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import common modules
from common.args_parser import create_base_parser
from common.data_loader import load_standard_data, merge_twin_data
from common.results_processor import create_results_dict, make_long_format
from common.stats_analysis import (
    calculate_accuracy,
    calculate_correlation_stats,
    calculate_effect_sizes,
    perform_paired_tests,
)
from common.variable_mapper import add_min_max_columns, create_domain_maps
from parse_specification import extract_specification_from_path, parse_specification_name


def run_mixedlm_split_contrast(df_human, df_twin, dv, cond_col):
    """Study-specific mixed effects model"""
    # 1) Stack human and twin into one long DataFrame
    df_h = df_human[["TWIN_ID", dv, cond_col]].assign(dataset_dummy=0)
    df_t = df_twin[["TWIN_ID", dv, cond_col]].assign(dataset_dummy=1)
    long = pd.concat([df_h, df_t], ignore_index=True)

    # 2) Code the default contrast: OPTOUT=1, OPTIN=-1
    long["default_contrast"] = long[cond_col].map({"opt-out": 1, "opt-in": -1})
    long = long.dropna(subset=[dv, "default_contrast"])

    # 3) Create separate interaction terms
    long["default_contrast_human"] = long["default_contrast"] * (1 - long["dataset_dummy"])
    long["default_contrast_twin"] = long["default_contrast"] * long["dataset_dummy"]

    # 4) Fit mixed‑effects model
    md = smf.mixedlm(
        f"{dv} ~ default_contrast_human + default_contrast_twin + dataset_dummy",
        long,
        groups=long["TWIN_ID"],
        re_formula="1",
    )
    mdf = md.fit()
    print(f"\n--- MixedLM results for {dv} ---")
    print(mdf.summary())


def main():
    # 1. ARGUMENT PARSING (structure preserved, using common function)
    args = create_base_parser("default_eric").parse_args()

    # 2. FILE DISCOVERY (structure preserved)
    # Handle automatic file discovery if --results-dir is used
    if args.results_dir:
        results_path = Path(args.results_dir)

        # Study: default_eric uses 'labels' format with human data at study level
        # Check for human data at study level first
        human_file_study = results_path.parent / "defaults human data labels anonymized.csv"
        if human_file_study.exists():
            args.human_data = str(human_file_study)
        else:
            # Fallback to generic name in specification folder
            human_file_spec = results_path / "defaults human data labels anonymized.csv"
            if human_file_spec.exists():
                args.human_data = str(human_file_spec)
            else:
                raise FileNotFoundError(
                    f"Could not find human data file in {results_path.parent} or {results_path}"
                )

        # Twin data in specification folder - uses consolidated_llm_labels format
        twin_file = results_path / "consolidated_llm_labels.csv"
        if not twin_file.exists():
            # Fallback to old naming convention
            twin_file = results_path / "defaults twins data labels anonymized.csv"
            if not twin_file.exists():
                raise FileNotFoundError(f"Could not find twin data file: {twin_file}")
        args.twin_data = str(twin_file)

        # Set output directory to results path
        args.output_dir = str(results_path)

        # Extract specification info from path
        specification_name = extract_specification_from_path(str(results_path))
        if specification_name:
            specification_type, ran_date = parse_specification_name(specification_name)
        else:
            # Fallback if path parsing fails
            specification_name = "default persona"
            specification_type = "default"
            ran_date = ""
    else:
        # Manual specification
        if not all([args.human_data, args.twin_data, args.output_dir]):
            raise ValueError(
                "Must specify either --results-dir OR all of (--human-data, --twin-data, --output-dir)"
            )
        specification_name = "custom_run"
        specification_type = "custom"
        ran_date = ""

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for visualizations
    figures_path = output_path / "figures"
    tables_path = output_path / "tables"
    figures_path.mkdir(exist_ok=True)
    tables_path.mkdir(exist_ok=True)

    # 3. DATA LOADING (structure preserved, using common function)
    study_name = "Defaults"
    print(f"Loading human data from: {args.human_data}")
    print(f"Loading twin data from: {args.twin_data}")

    df_human, df_twin = load_standard_data(
        args.human_data,
        args.twin_data,
        skiprows_human=[1, 2],  # Study-specific: skip rows for both
        skiprows_twin=[1, 2],  # Study-specific: skip rows for both
    )

    # 4. STUDY-SPECIFIC DATA PROCESSING
    # add new columns with relevant variables coded
    # chose target behavior green:
    for df in (df_human, df_twin):
        df["target_behavior_green"] = (
            (df["OPTOUT_Green"] == "Stay with GreenGrid") | (df["OPTIN_Green"] == "Switch to GreenGrid")
        ).astype(int)
    # chose target behavior organs:
    for df in (df_human, df_twin):
        df["target_behavior_organ"] = (
            (
                df["OPTOUT_Organ"]
                == "You are therefore currently a potential donor. If this is acceptable, click here:"
            )
            | (df["OPTIN_Organ"] == "If you wish to change your status, click here:")
        ).astype(int)

    # conditon for green
    conditions = [
        df_human["OPTOUT_Green"].notna() & df_human["OPTOUT_Green"].str.strip().ne(""),
        df_human["OPTIN_Green"].notna() & df_human["OPTIN_Green"].str.strip().ne(""),
    ]
    choices = ["opt-out", "opt-in"]
    for df in (df_human, df_twin):
        df["condition_green"] = np.select(conditions, choices, default=None)

    # conditon for organ
    conditions = [
        df_human["OPTOUT_Organ"].notna() & df_human["OPTOUT_Organ"].str.strip().ne(""),
        df_human["OPTIN_Organ"].notna() & df_human["OPTIN_Organ"].str.strip().ne(""),
    ]
    choices = ["opt-out", "opt-in"]
    for df in (df_human, df_twin):
        df["condition_organ"] = np.select(conditions, choices, default=None)

    # check results:
    pairs = [
        ("OPTOUT_Green", "target_behavior_green"),
        ("OPTIN_Green", "target_behavior_green"),
        ("OPTOUT_Organ", "target_behavior_organ"),
        ("OPTIN_Organ", "target_behavior_organ"),
        ("OPTOUT_Green", "condition_green"),
        ("OPTIN_Green", "condition_green"),
        ("OPTOUT_Organ", "condition_organ"),
        ("OPTIN_Organ", "condition_organ"),
    ]

    for df_name, df in [("Human", df_human), ("Twin", df_twin)]:
        print(f"\n==== {df_name} sample ====\n")
        for orig, new in pairs:
            print(f"– {orig} vs {new} –")
            ct = pd.crosstab(df[orig], df[new], dropna=False)
            print(ct)
            print()

    # 5. VARIABLE DEFINITIONS (structure preserved)
    # different condition assignments for each DV (unique to this study)
    dv_to_cond = {
        "target_behavior_green": "condition_green",
        "target_behavior_organ": "condition_organ",
    }

    # raw responses:
    raw_vars = []

    # DVs:
    DV_vars = ["target_behavior_green", "target_behavior_organ"]
    DV_vars_min = [0] * 2
    DV_vars_max = [1] * 2

    # Create domain maps using helper
    domain_maps = create_domain_maps(
        DV_vars,
        social=[0] * 2,  # Study-specific values
        cognitive=[0] * 2,
        known=[1] * 2,
        pref=[1] * 2,
        stim=[1] * 2,
        know=[0, 0],
        politics=[0, 0],
    )

    # 6. MERGE DATA (structure preserved, using common function)
    # merging key
    merge_key = ["TWIN_ID"]

    # Merge on TWIN_ID
    df = merge_twin_data(df_human, df_twin, merge_key=merge_key)

    # Fix dtypes
    for var in raw_vars + DV_vars:
        df[f"{var}_human"] = pd.to_numeric(df[f"{var}_human"], errors="coerce")
        df[f"{var}_twin"] = pd.to_numeric(df[f"{var}_twin"], errors="coerce")

    # Add min/max columns
    min_map = dict(zip(DV_vars, DV_vars_min))
    max_map = dict(zip(DV_vars, DV_vars_max))
    df = add_min_max_columns(df, min_map, max_map)

    # 7. COMPUTE RESULTS (loop structure preserved exactly)
    results = []

    for var in DV_vars:
        # Study-specific: condition assignment different for each DV
        # look up the right condition for this DV
        cond = dv_to_cond[var]  # e.g. 'condition_green'
        cond_h = f"{cond}_human"  # 'condition_green_human'
        cond_t = f"{cond}_twin"  # 'condition_green_twin'
        col_h = f"{var}_human"
        col_t = f"{var}_twin"
        min_col = f"{var}_min"
        max_col = f"{var}_max"
        # always include the matching condition columns
        cols = [col_h, col_t, cond_h, cond_t, min_col, max_col]
        pair = df[cols].dropna(subset=[col_h, col_t, cond_h, cond_t])

        min_val = pair[min_col].iloc[0] if len(pair) > 0 else np.nan
        max_val = pair[max_col].iloc[0] if len(pair) > 0 else np.nan
        n = len(pair)

        # Use common functions for calculations
        stats = calculate_correlation_stats(pair, col_h, col_t)
        stats["accuracy"] = calculate_accuracy(pair, col_h, col_t, min_val, max_val)

        test_stats = perform_paired_tests(pair, col_h, col_t)
        effect_sizes = calculate_effect_sizes(pair, col_h, col_t, cond_h, cond_t, cond_exists=True)

        # Get means and stds for compatibility
        if n >= 4:
            mean_h = pair[col_h].mean()
            mean_t = pair[col_t].mean()
            std_h = pair[col_h].std(ddof=1)
            std_t = pair[col_t].std(ddof=1)
        else:
            mean_h = mean_t = std_h = std_t = np.nan

        # Combine all stats
        all_stats = {
            **stats,
            **test_stats,
            **effect_sizes,
            "mean_human": mean_h,
            "mean_twin": mean_t,
            "std_human": std_h,
            "std_twin": std_t,
            "sample size": n,
        }

        # Create result dict using common function
        result = create_results_dict(
            study_name,
            var,
            all_stats,
            domain_maps,
            {"name": specification_name, "type": specification_type, "date": ran_date},
        )
        results.append(result)

    # 8. SAVE OUTPUTS (structure preserved exactly)
    corr_df = pd.DataFrame(results)
    print(corr_df)

    # save output as csv
    out_file = output_path / "meta analysis.csv"
    corr_df.to_csv(out_file, index=False)

    # 9. CREATE LONG FORMAT (using common function)
    df_long = make_long_format(df_human, df_twin, DV_vars, study_name, specification_name)

    print(df_long.head())
    # save output as csv - unit of observation is TWIN_ID:
    out_file = output_path / "meta analysis individual level.csv"
    df_long.to_csv(out_file, index=False)

    print("done")

    # 10. STUDY-SPECIFIC ADDITIONAL ANALYSES
    # Visualization
    groups = {
        "Human": df_human,
        "Twin": df_twin,
    }

    # calculate standard error of the mean
    def compute_sem(series):
        return series.std() / np.sqrt(len(series))

    conditions = ["opt-in", "opt-out"]
    color_map = {"opt-in": "blue", "opt-out": "orange"}

    labels = []
    means = []
    sems = []
    colors = []

    # First 4 bars: target_behavior_green by condition_green
    for grp_name, df in groups.items():
        for cond in conditions:
            subset = df[df["condition_green"] == cond]
            labels.append(f"{grp_name} Green {cond}")
            means.append(subset["target_behavior_green"].mean())
            sems.append(compute_sem(subset["target_behavior_green"]))
            colors.append(color_map[cond])

    # Next 4 bars: target_behavior_organ by condition_organ
    for grp_name, df in groups.items():
        for cond in conditions:
            subset = df[df["condition_organ"] == cond]
            labels.append(f"{grp_name} Organ {cond}")
            means.append(subset["target_behavior_organ"].mean())
            sems.append(compute_sem(subset["target_behavior_organ"]))
            colors.append(color_map[cond])

    # Plot
    x = np.arange(len(means))
    fig, ax = plt.subplots()
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors)

    # Annotate each bar with its mean
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.2f}",
            ha="center",
            va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean target behavior")

    # Legend for opt‐in vs. opt‐out
    legend_handles = [Patch(facecolor=color_map[c], label=c.capitalize()) for c in conditions]
    ax.legend(handles=legend_handles, title="Condition")

    plt.tight_layout()
    # Save figure instead
    plt.savefig(figures_path / "condition_comparison.png")
    plt.close()

    # Mixed effects models
    run_mixedlm_split_contrast(df_human, df_twin, dv="target_behavior_green", cond_col="condition_green")
    run_mixedlm_split_contrast(df_human, df_twin, dv="target_behavior_organ", cond_col="condition_organ")

    return corr_df, df_long


if __name__ == "__main__":
    main()