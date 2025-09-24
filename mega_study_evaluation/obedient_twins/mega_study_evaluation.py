#!/usr/bin/env python3
"""
Meta-analysis script for Obedient Twins
Refactored to use common modules while preserving all study-specific logic.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.args_parser import create_base_parser, handle_file_discovery
from common.data_loader import load_standard_data, merge_twin_data, prepare_data_for_analysis
from common.stats_analysis import compute_standard_stats
from common.results_processor import create_results_dict, make_long_format, save_standard_outputs
from common.variable_mapper import create_domain_maps, build_min_max_maps, add_min_max_columns, check_condition_variable


def create_obedient_variables(df_human, df_twin):
    """
    Create study-specific obedient twins variables.
    This is unique to obedient_twins study.
    """
    # Task 1 - attitude change
    for df in (df_human, df_twin):
        df["attitude_1"] = pd.to_numeric(df["attitude_1"], errors="coerce")
        df["new_att_1"] = pd.to_numeric(df["new_att_1"], errors="coerce")

        conditions = [df["attitude_1"] > 50, df["attitude_1"] < 50, df["attitude_1"] == 50]
        choices = [
            df["attitude_1"] - df["new_att_1"],
            df["new_att_1"] - df["attitude_1"],
            (df["new_att_1"] - df["attitude_1"]).abs(),
        ]

        df["task 1 - attitude_change"] = np.select(conditions, choices, default=np.nan)

    # Task 2 scenarios
    for df in (df_human, df_twin):
        df["task 2 - S1"] = (8 - df["i1_1"].astype("Int64") + 8 - df["i1_2"].astype("Int64")) / 2
        df["task 2 - S2"] = (df["i2_1"].astype("Int64") + 8 - df["i2_2"].astype("Int64")) / 2
        df["task 2 - S3"] = (df["i3_1"].astype("Int64") + df["i3_2"].astype("Int64")) / 2

    # Task 3 scenarios 
    for df in (df_human, df_twin):
        df["task 3 - S1"] = df["a1_1"].astype("Int64")
        df["task 3 - S2"] = df["a2_1"].astype("Int64")
        df["task 3 - S3"] = df["a3_1"].astype("Int64")

    return df_human, df_twin


def analyze_variables_no_condition(df, DV_vars, domain_maps, spec_info, paths):
    """
    Analyze variables without conditions (task 1).
    """
    # Variables without conditions
    vars_no_cond = ["task 1 - attitude_change"]
    vars_min = [-100]
    vars_max = [100]
    
    # Domain classifications for task 1
    social = [1]
    cognitive = [0]
    known = [1]
    pref = [1]
    stim = [1]
    know = [0]
    politics = [1]
    
    domain_maps_no_cond = create_domain_maps(vars_no_cond, social, cognitive, known, pref, stim, know, politics)
    min_map, max_map = build_min_max_maps(vars_no_cond, vars_min, vars_max)
    df = add_min_max_columns(df, min_map, max_map)
    
    results = []
    for var in vars_no_cond:
        stats_dict = compute_standard_stats(
            df, var, False, None, None,
            min_map.get(var), max_map.get(var)
        )
        
        result = create_results_dict(
            "Obedient twins", var, stats_dict, 
            domain_maps_no_cond, spec_info
        )
        results.append(result)
    
    return results


def analyze_variables_with_conditions(df, paths, spec_info):
    """
    Analyze variables with condition-specific mappings (tasks 2 and 3).
    This preserves the unique dv_to_cond mapping logic.
    """
    # Variables with conditions
    DV_vars = [
        "task 2 - S1",
        "task 2 - S2", 
        "task 2 - S3",
        "task 3 - S1",
        "task 3 - S2",
        "task 3 - S3",
    ]
    DV_vars_min = [1, 1, 1, 1, 1, 1]
    DV_vars_max = [7, 7, 7, 7, 7, 7]
    
    # Domain classifications
    social = [1] * 6
    cognitive = [0] * 6
    known = [1] * 6
    pref = [1] * 6
    stim = [1] * 6
    know = [0] * 6
    politics = [0, 0, 0, 0, 0, 0]
    
    domain_maps = create_domain_maps(DV_vars, social, cognitive, known, pref, stim, know, politics)
    
    # Different condition assignments for each DV (unique to this study)
    dv_to_cond = {
        "task 2 - S1": "s_1",
        "task 2 - S2": "s_2", 
        "task 2 - S3": "s_3",
        "task 3 - S1": "absurd_1",
        "task 3 - S2": "absurd_2",
        "task 3 - S3": "absurd_3",
    }
    
    min_map, max_map = build_min_max_maps(DV_vars, DV_vars_min, DV_vars_max)
    df = add_min_max_columns(df, min_map, max_map)
    
    results = []
    for var in DV_vars:
        # Look up the right condition for this DV
        cond = dv_to_cond[var]
        cond_h = f"{cond}_human"
        cond_t = f"{cond}_twin"
        
        # Use the compute_standard_stats but we need to handle condition differently
        # since each variable has its own condition
        col_h = f"{var}_human"
        col_t = f"{var}_twin"
        min_col = f"{var}_min"
        max_col = f"{var}_max"
        
        # Build columns list with specific condition
        cols = [col_h, col_t, cond_h, cond_t, min_col, max_col]
        pair = df[cols].dropna(subset=[col_h, col_t, cond_h, cond_t])
        
        # Get min/max values
        if len(pair) > 0:
            min_val = pair[min_col].iloc[0]
            max_val = pair[max_col].iloc[0]
        else:
            min_val = min_map.get(var)
            max_val = max_map.get(var)
        
        # Compute stats using the specific condition variables
        stats_dict = compute_standard_stats(
            df, var, True, cond_h, cond_t,
            min_val, max_val
        )
        
        result = create_results_dict(
            "Obedient twins", var, stats_dict, 
            domain_maps, spec_info
        )
        results.append(result)
    
    return results


def run_task_analyses(df_human, df_twin, paths):
    """
    Run the additional task-specific analyses (preserving original logic).
    """
    # TASK 1 analysis
    df_human_model = (
        df_human[["TWIN_ID", "task 1 - attitude_change", "attitude_1"]]
        .rename(columns={"task 1 - attitude_change": "attitude_change"})
        .assign(twin_type="human")
    )
    df_twin_model = (
        df_twin[["TWIN_ID", "task 1 - attitude_change", "attitude_1"]]
        .rename(columns={"task 1 - attitude_change": "attitude_change"})
        .assign(twin_type="digital")
    )
    df_model = pd.concat([df_human_model, df_twin_model], ignore_index=True)

    # Cast to float and clean
    df_model["attitude_change"] = df_model["attitude_change"].astype(float)
    df_model["attitude_1"] = df_model["attitude_1"].astype(float)
    df_model = df_model.dropna(subset=["attitude_change", "attitude_1"])
    df_model["twin_type"] = df_model["twin_type"].astype("category")

    # Fit mixed-effects model
    md = smf.mixedlm(
        "attitude_change ~ C(twin_type) + attitude_1",
        df_model,
        groups=df_model["TWIN_ID"],
        missing="drop",
    )
    mdf = md.fit(reml=False)
    print(mdf.summary())

    # TASK 2 analysis
    def stack_scenarios(df, twin_label):
        long = df.melt(
            id_vars=["TWIN_ID", "s_1", "s_2", "s_3"],
            value_vars=["task 2 - S1", "task 2 - S2", "task 2 - S3"],
            var_name="scenario",
            value_name="score",
        )
        long["scenario"] = long["scenario"].str[-1].astype(int)
        long["manipulation"] = long.apply(lambda r: r[f"s_{r.scenario}"], axis=1)
        long["twin_type"] = twin_label
        return long

    long_h = stack_scenarios(df_human, "human")
    long_t = stack_scenarios(df_twin, "digital")
    df_long = pd.concat([long_h, long_t], ignore_index=True)

    # Clean and prepare
    df_long["score"] = pd.to_numeric(df_long["score"], errors="coerce")
    df_long["manipulation"] = df_long["manipulation"].astype("category")
    df_long["twin_type"] = df_long["twin_type"].astype("category")
    df_long = df_long.dropna(subset=["score", "manipulation"])

    # Create scenario dummy variables
    df_long["sc2"] = (df_long["scenario"] == 2).astype(float)
    df_long["sc3"] = (df_long["scenario"] == 3).astype(float)

    # Fit mixed-effects model for task 2
    md = smf.mixedlm(
        "score ~ C(manipulation) * C(twin_type)",
        df_long,
        groups=df_long["TWIN_ID"],
        vc_formula={"sc2": "0 + sc2", "sc3": "0 + sc3"},
    )
    mdf = md.fit(reml=False)
    print(mdf.summary())

    # Summary stats
    stats = (
        df_long.groupby(["twin_type", "manipulation"])["score"]
        .agg(avg_score="mean", sd_score="std")
        .reset_index()
    )
    print(stats)

    # TASK 3 analysis and visualization
    def stack_task3(df, twin_label):
        long = df.melt(
            id_vars=["TWIN_ID", "absurd_1", "absurd_2", "absurd_3"],
            value_vars=["task 3 - S1", "task 3 - S2", "task 3 - S3"],
            var_name="scenario",
            value_name="score",
        )
        long["scenario"] = long["scenario"].str[-1].astype(int)
        long["absurd_cond"] = long.apply(lambda r: r[f"absurd_{r.scenario}"], axis=1)
        long["twin_type"] = twin_label
        return long

    long_human = stack_task3(df_human, "human")
    long_twin = stack_task3(df_twin, "twin")
    df_long3 = pd.concat([long_twin, long_human], ignore_index=True)

    df_long3["score"] = pd.to_numeric(df_long3["score"], errors="coerce")
    df_long3 = df_long3.dropna(subset=["score", "absurd_cond"])

    # Compute means & SEMs for visualization
    grouped = df_long3.groupby(["scenario", "twin_type", "absurd_cond"])["score"]
    stats = grouped.agg(mean="mean", sem="sem").reset_index()

    # Create visualization
    orders = []
    for sc in [1, 2, 3]:
        for tt in ["twin", "human"]:
            for cond in ["A", "B"]:
                orders.append((sc, tt, cond))

    means = [
        stats.loc[
            (stats["scenario"] == sc) & (stats["twin_type"] == tt) & (stats["absurd_cond"] == cond),
            "mean",
        ].values[0]
        for sc, tt, cond in orders
    ]
    sems = [
        stats.loc[
            (stats["scenario"] == sc) & (stats["twin_type"] == tt) & (stats["absurd_cond"] == cond),
            "sem",
        ].values[0]
        for sc, tt, cond in orders
    ]

    color_map = {
        ("twin", "A"): "C0",
        ("twin", "B"): "C1", 
        ("human", "A"): "C2",
        ("human", "B"): "C3",
    }

    # Create and save plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(orders))

    for i, (sc, tt, cond) in enumerate(orders):
        ax.bar(x[i], means[i], yerr=sems[i], capsize=5, color=color_map[(tt, cond)])
        ax.text(
            x[i],
            means[i] + sems[i] + 0.05,
            f"{means[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add separators and labels
    for sep in [4, 8]:
        ax.axvline(sep - 0.5, color="gray", linewidth=1)

    ax.set_xticks([1.5, 5.5, 9.5])
    ax.set_xticklabels(["Task 3 S1", "Task 3 S2", "Task 3 S3"])
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean Task 3 Score")
    ax.set_title("Task 3 by Twin Type and Absurd Condition")

    # Custom legend
    handles = [
        mpatches.Patch(color=color_map[(tt, cond)], label=f"{tt.capitalize()} {cond}")
        for tt, cond in [("twin", "A"), ("twin", "B"), ("human", "A"), ("human", "B")]
    ]
    ax.legend(handles=handles, title="Group")

    plt.tight_layout()
    fig_path = paths['figures_path'] / "task3_by_twin_type_and_absurd_condition.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig_path}")


def main():
    # Parse arguments
    parser = create_base_parser("Obedient Twins")
    args = parser.parse_args()

    # Handle file discovery
    paths = handle_file_discovery(args)

    if args.verbose:
        print(f"Loading data from:")
        print(f"  Human: {paths['human_data']}")
        print(f"  Twin: {paths['twin_data']}")
        print(f"  Output: {paths['output_dir']}")

    # Load data
    df_human, df_twin = load_standard_data(paths['human_data'], paths['twin_data'])

    # Create obedient twins variables (unique to this study)
    df_human, df_twin = create_obedient_variables(df_human, df_twin)

    # Extract specification info for results
    from parse_specification import extract_specification_from_path, parse_specification_name
    specification_name = extract_specification_from_path(args.results_dir)
    if specification_name:
        specification_type, ran_date = parse_specification_name(specification_name)
    else:
        specification_name = "unknown"
        specification_type = "unknown"
        ran_date = None

    spec_info = {
        'type': specification_type,
        'name': specification_name,
        'date': ran_date
    }

    # Merge data for analysis
    df = merge_twin_data(df_human, df_twin)
    
    # Analyze variables without conditions (task 1)
    results_no_cond = analyze_variables_no_condition(df, None, None, spec_info, paths)
    
    # Analyze variables with conditions (tasks 2 and 3)  
    results_with_cond = analyze_variables_with_conditions(df, paths, spec_info)
    
    # Combine all results
    all_results = results_no_cond + results_with_cond
    corr_df = pd.DataFrame(all_results)
    
    if args.verbose:
        print(corr_df)

    # Create long format data for all DVs
    all_DV_vars = [
        "task 1 - attitude_change",
        "task 2 - S1", 
        "task 2 - S2",
        "task 2 - S3",
        "task 3 - S1",
        "task 3 - S2", 
        "task 3 - S3",
    ]
    
    df_long = make_long_format(df_human, df_twin, all_DV_vars, "Obedient twins", specification_name)
    # Filter out NaN values as in original
    df_long = df_long[df_long["value"].notna()]

    # Save standard outputs
    meta_analysis_file, individual_file = save_standard_outputs(
        corr_df, df_long, paths['output_path'], args.verbose
    )

    # Run additional task analyses
    run_task_analyses(df_human, df_twin, paths)

    return corr_df, df_long


if __name__ == "__main__":
    main()