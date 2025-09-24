#!/usr/bin/env python3
"""
Meta-analysis script for Context Effects
Refactored to use common modules while preserving all study-specific logic.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
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


def create_context_effects_variables(df_human, df_twin, verbose=False):
    """
    Create study-specific context effects variables.
    This includes choice target variables and category variables.
    This is unique to context_effects study.
    """
    # Choice target in binary choice
    for df in (df_human, df_twin):
        df["choice_target_binary"] = (
            (df["Bin.Printer_TC"] == "1000AX")
            | (df["Bin.TV_TC"] == "HD-558")
            | (df["Bin.Cell_TC"] == "X-3A")
        ).astype(int)
    
    # Choice target in attraction choice
    for df in (df_human, df_twin):
        df["choice_target_attraction"] = (
            (df["Attr_Printer"] == "1000AX")
            | (df["Attr_TV"] == "HD-558")
            | (df["Attr_Cell"] == "X-3A")
        ).astype(int)
    
    # Choice target in compromise choice
    for df in (df_human, df_twin):
        df["choice_target_compromise"] = (
            (df["Comp_Printer"] == "1000AX")
            | (df["Comp_TV"] == "HD-558")
            | (df["Comp_Cell"] == "X-3A")
        ).astype(int)

    # Category for binary choice
    for df in (df_human, df_twin):
        conditions = [
            df["Bin.Printer_TC"].notna() & (df["Bin.Printer_TC"].astype(str).str.strip() != ""),
            df["Bin.Cell_TC"].notna() & (df["Bin.Cell_TC"].astype(str).str.strip() != ""),
            df["Bin.TV_TC"].notna() & (df["Bin.TV_TC"].astype(str).str.strip() != ""),
        ]
        choices = ["printer", "cell", "TV"]
        df["category_binary"] = np.select(conditions, choices, default=None)

    # Category for attraction choice
    for df in (df_human, df_twin):
        conditions = [
            df["Attr_Printer"].notna() & (df["Attr_Printer"].astype(str).str.strip() != ""),
            df["Attr_Cell"].notna() & (df["Attr_Cell"].astype(str).str.strip() != ""),
            df["Attr_TV"].notna() & (df["Attr_TV"].astype(str).str.strip() != ""),
        ]
        choices = ["printer", "cell", "TV"]
        df["category_attraction"] = np.select(conditions, choices, default=None)

    # Category for compromise choice
    for df in (df_human, df_twin):
        conditions = [
            df["Comp_Printer"].notna() & (df["Comp_Printer"].astype(str).str.strip() != ""),
            df["Comp_Cell"].notna() & (df["Comp_Cell"].astype(str).str.strip() != ""),
            df["Comp_TV"].notna() & (df["Comp_TV"].astype(str).str.strip() != ""),
        ]
        choices = ["printer", "cell", "TV"]
        df["category_compromise"] = np.select(conditions, choices, default=None)

    # Check results if verbose
    if verbose:
        pairs = [
            ("Bin.Printer_TC", "choice_target_binary"),
            ("Bin.TV_TC", "choice_target_binary"),
            ("Bin.Cell_TC", "choice_target_binary"),
            ("Attr_Printer", "choice_target_attraction"),
            ("Attr_TV", "choice_target_attraction"),
            ("Attr_Cell", "choice_target_attraction"),
            ("Comp_Printer", "choice_target_compromise"),
            ("Comp_TV", "choice_target_compromise"),
            ("Comp_Cell", "choice_target_compromise"),
            ("Bin.Printer_TC", "category_binary"),
            ("Bin.TV_TC", "category_binary"),
            ("Bin.Cell_TC", "category_binary"),
            ("Attr_Printer", "category_attraction"),
            ("Attr_TV", "category_attraction"),
            ("Attr_Cell", "category_attraction"),
            ("Comp_Printer", "category_compromise"),
            ("Comp_TV", "category_compromise"),
            ("Comp_Cell", "category_compromise"),
        ]

        for df_name, df in [("Human", df_human), ("Twin", df_twin)]:
            print(f"\n==== {df_name} sample ====\n")
            for orig, new in pairs:
                print(f"– {orig} vs {new} –")
                ct = pd.crosstab(df[orig], df[new], dropna=False)
                print(ct)
                print()

    return df_human, df_twin


def run_context_effects_regression(df_human, df_twin, verbose=False):
    """
    Run the context effects regression analysis with data melting/reshaping.
    This preserves the complex data transformation logic from the original.
    """
    # Tag each source and stack into one long dataset
    df_human["dataset"] = "human"
    df_twin["dataset"] = "twin"

    df_comb = pd.concat([df_human, df_twin], ignore_index=True)

    # Pivot each scenario/trial into three rows: binary, attraction, compromise
    n = len(df_comb)
    df_long = pd.DataFrame(
        {
            "TWIN_ID": np.repeat(df_comb["TWIN_ID"].values, 3),
            "dataset": np.repeat(df_comb["dataset"].values, 3),
            "trial_type": np.tile(["binary", "attraction", "compromise"], n),
            "choice_target": np.concatenate(
                [
                    df_comb["choice_target_binary"].values,
                    df_comb["choice_target_attraction"].values,
                    df_comb["choice_target_compromise"].values,
                ]
            ),
            "category": np.concatenate(
                [
                    df_comb["category_binary"].values,
                    df_comb["category_attraction"].values,
                    df_comb["category_compromise"].values,
                ]
            ),
        }
    )

    # Define the two custom contrasts
    df_long["attraction_contrast"] = df_long["trial_type"].map(
        {"attraction": 1, "binary": -1, "compromise": 0}
    )
    df_long["compromise_contrast"] = df_long["trial_type"].map(
        {"compromise": 1, "binary": -1, "attraction": 0}
    )

    # Make sure key predictors are categoricals with the right reference levels
    df_long["dataset"] = pd.Categorical(df_long["dataset"], categories=["human", "twin"])
    df_long["category"] = pd.Categorical(df_long["category"], categories=["cell", "printer", "TV"])

    if verbose:
        print("Context effects regression analysis:")

    # Attraction contrast analysis
    fe_formula = (
        "choice_target ~ "
        "C(dataset, Treatment(reference='human')) + "
        "C(category, Treatment(reference='cell')) + "
        "attraction_contrast:C(dataset)"
    )

    ols_model = smf.ols(fe_formula, data=df_long).fit(
        cov_type="cluster", cov_kwds={"groups": df_long["TWIN_ID"]}
    )

    if verbose:
        print("Attraction contrast model:")
        print(ols_model.summary())

    # Compromise contrast analysis
    fe_formula = (
        "choice_target ~ "
        "C(dataset, Treatment(reference='human')) + "
        "C(category, Treatment(reference='cell')) + "
        "compromise_contrast:C(dataset)"
    )

    ols_model = smf.ols(fe_formula, data=df_long).fit(
        cov_type="cluster", cov_kwds={"groups": df_long["TWIN_ID"]}
    )

    if verbose:
        print("Compromise contrast model:")
        print(ols_model.summary())

    return df_long


def main():
    # Parse arguments
    parser = create_base_parser("Context Effects")
    args = parser.parse_args()

    # Handle file discovery with study-specific configuration
    study_config = {
        'human_at_study_level': True,
        'human_filename': 'context effects human data labels anonymized.csv',
        'twin_filename': 'consolidated_llm_labels.csv'
    }
    paths = handle_file_discovery(args, study_config)

    if args.verbose:
        print(f"Loading data from:")
        print(f"  Human: {paths['human_data']}")
        print(f"  Twin: {paths['twin_data']}")
        print(f"  Output: {paths['output_dir']}")

    # Load data
    df_human, df_twin = load_standard_data(paths['human_data'], paths['twin_data'])

    # Create context effects variables (unique to this study)
    df_human, df_twin = create_context_effects_variables(df_human, df_twin, args.verbose)

    # Save processed human data (preserving original behavior)
    human_file = paths['output_path'] / "human data values anonymized processed.csv"
    df_human.to_csv(human_file, index=False)

    # Define study variables
    DV_vars = ["choice_target_binary", "choice_target_attraction", "choice_target_compromise"]
    DV_vars_min = [0] * 3
    DV_vars_max = [1] * 3

    # Domain classifications
    social = [0] * 3
    cognitive = [0] * 3
    known = [1] * 3
    pref = [1] * 3
    stim = [1] * 3
    know = [0, 0, 0]
    politics = [0, 0, 0]

    # Create domain maps
    domain_maps = create_domain_maps(DV_vars, social, cognitive, known, pref, stim, know, politics)

    # Check condition variables (none for this study)
    condition_vars = [""]
    cond_exists, cond, cond_h, cond_t = check_condition_variable(condition_vars)

    # Build min/max maps and merge data
    min_map, max_map = build_min_max_maps(DV_vars, DV_vars_min, DV_vars_max)
    df = merge_twin_data(df_human, df_twin)
    df = prepare_data_for_analysis(df, DV_vars)
    df = add_min_max_columns(df, min_map, max_map)

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

    # Compute results for all variables
    results = []
    for var in DV_vars:
        stats_dict = compute_standard_stats(
            df, var, cond_exists, cond_h, cond_t,
            min_map.get(var), max_map.get(var)
        )
        
        result = create_results_dict(
            "context effects", var, stats_dict, 
            domain_maps, spec_info
        )
        results.append(result)

    # Create results DataFrame
    corr_df = pd.DataFrame(results)
    
    if args.verbose:
        print(corr_df)

    # Create long format data
    df_long = make_long_format(df_human, df_twin, DV_vars, "context effects", specification_name)

    # Save standard outputs
    meta_analysis_file, individual_file = save_standard_outputs(
        corr_df, df_long, paths['output_path'], args.verbose
    )

    # Run the context effects regression analysis (preserving original complex logic)
    df_long_regression = run_context_effects_regression(df_human, df_twin, args.verbose)

    return corr_df, df_long


if __name__ == "__main__":
    main()