# Diff Summary: Jupyter Notebook to Python Script Conversion

This document shows the complete differences between the direct `jupyter nbconvert` output and our modified Python scripts. The changes include the initial conversion updates and the recent path fixes to enable running from project root.

## Common Pattern Applied to All Scripts

### 1. Added Imports and Argument Parsing

All scripts received the following additions at the beginning:

```diff
+#!/usr/bin/env python3
+"""
+Meta-analysis script for [STUDY_NAME]
+Converted from: [NOTEBOOK_NAME].ipynb
+"""
+
+import argparse
+import os
+import sys
+from pathlib import Path
 import pandas as pd
 import numpy as np
 from scipy.stats import pearsonr, norm, ttest_rel, f
 import matplotlib.pyplot as plt
 import seaborn as sns
 import warnings
 warnings.filterwarnings('ignore')
+
+def parse_arguments():
+    parser = argparse.ArgumentParser(
+        description='Run meta-analysis for [STUDY_NAME]',
+        formatter_class=argparse.ArgumentDefaultsHelpFormatter
+    )
+    
+    # Option 1: Specify results directory (automatic file discovery)
+    parser.add_argument('--results-dir', 
+                       help='Path to results/{study}/{specification} directory')
+    
+    # Option 2: Specify individual files (for custom paths)
+    parser.add_argument('--human-data', 
+                       help='Path to human data CSV file')
+    parser.add_argument('--twin-data',
+                       help='Path to twin data CSV file')
+    parser.add_argument('--output-dir',
+                       help='Output directory for results')
+    
+    parser.add_argument('--verbose', action='store_true',
+                       help='Print detailed progress information')
+    
+    return parser.parse_args()
+
+def main():
+    # Parse arguments
+    args = parse_arguments()
```

### 2. Path Handling Updates (Recent Fix for Project Root)

The path handling was updated to correctly locate files based on study_folder_mapping.json configuration:

#### For studies with "values" format and human data in specification directory:
```diff
+    # Handle automatic file discovery if --results-dir is used
+    if args.results_dir:
+        results_path = Path(args.results_dir)
+        
+        # Determine file paths based on study configuration
+        # Study: [STUDY_NAME]
+        # Uses values format
+        # Human data at specification level
+        
+        # Both files are in the specification directory
+        args.human_data = str(results_path / "consolidated_original_answers_values.csv")
+        args.twin_data = str(results_path / "consolidated_llm_values.csv")
+        # Only set output_dir if not explicitly provided
+        if not args.output_dir:
+            args.output_dir = str(results_path)
```

#### For studies with "labels" format and human data at study level:
```diff
+    # Handle automatic file discovery if --results-dir is used
+    if args.results_dir:
+        results_path = Path(args.results_dir)
+        
+        # Determine file paths based on study configuration
+        # Study: [STUDY_NAME]
+        # Uses labels format
+        # Human data at study level
+        
+        # Human data is at the study level (parent of specification)
+        study_path = results_path.parent
+        args.human_data = str(study_path / "[HUMAN_DATA_FILENAME].csv")
+        args.twin_data = str(results_path / "consolidated_llm_labels.csv")
+        # Only set output_dir if not explicitly provided
+        if not args.output_dir:
+            args.output_dir = str(results_path)
```

### 3. Output Directory and Subdirectory Creation

```diff
+    output_path = Path(args.output_dir)
+    output_path.mkdir(parents=True, exist_ok=True)
+    
+    # Create subdirectories for visualizations
+    figures_path = output_path / "figures"
+    tables_path = output_path / "tables"
+    figures_path.mkdir(exist_ok=True)
+    tables_path.mkdir(exist_ok=True)
+    
+    if args.verbose:
+        print(f"Loading data from:")
+        print(f"  Human: {args.human_data}")
+        print(f"  Twin: {args.twin_data}")
+        print(f"  Output: {args.output_dir}")
```

### 4. Data Loading Updates

```diff
-# Load data
-study_name = "[STUDY_NAME]"
-specification_name = "default persona"
-human_file = f"{study_name} human data values anonymized.csv"
-twin_file = f"{study_name} twins data values anonymized.csv"
-df_human = pd.read_csv(human_file, header=0, skiprows=[1,2])
-df_twin = pd.read_csv(twin_file, header=0, skiprows=[1,2])
+    # Load data
+    study_name = "[STUDY_NAME]"
+    specification_name = "default persona"  # Default value
+    
+    # Extract specification name from path if using --results-dir
+    if args.results_dir:
+        results_path = Path(args.results_dir)
+        path_parts = results_path.parts
+        if len(path_parts) >= 2 and path_parts[-2] == "[STUDY_FOLDER]":
+            specification_name = path_parts[-1]
+        elif len(path_parts) >= 3 and path_parts[-3] == "results":
+            specification_name = path_parts[-1]
+    
+    df_human = pd.read_csv(args.human_data, header=0, skiprows=[1,2])
+    df_twin = pd.read_csv(args.twin_data, header=0, skiprows=[1,2])
```

### 5. Output File Path Updates

```diff
-# save output as csv - unit of observation is comparison between humans and twins:
-out_file = f"{study_name} {specification_name} meta analysis.csv"
-corr_df.to_csv(out_file, index=False)
+    # save output as csv - unit of observation is comparison between humans and twins:
+    meta_analysis_file = output_path / "meta analysis.csv"
+    corr_df.to_csv(meta_analysis_file, index=False)

-# save output as csv - unit of observation is TWIN_ID:
-out_file = f"{study_name} {specification_name} meta analysis individual level.csv"
-df_long.to_csv(out_file, index=False)
+    # save output as csv - unit of observation is TWIN_ID:
+    individual_file = output_path / "meta analysis individual level.csv"
+    df_long.to_csv(individual_file, index=False)
```

### 6. Figure Saving Updates

```diff
-plt.show()
+    fig_path = figures_path / "[DESCRIPTIVE_NAME].png"
+    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
+    plt.close()
+    if args.verbose:
+        print(f"Saved figure: {fig_path}")
```

### 7. Main Function Entry Point

```diff
+if __name__ == "__main__":
+    main()
```

### 8. Removed IPython Magic Commands

```diff
-get_ipython().run_line_magic('matplotlib', 'inline')
```

## Study-Specific Path Configurations

Based on study_folder_mapping.json, here are the specific path configurations for each study:

### Studies using "values" format with human data at specification level:
- accuracy_nudges
- affective_priming
- consumer_minimalism
- digital_certification
- hiring_algorithms
- idea_evaluation
- infotainment
- junk_fees
- obedient_twins
- privacy
- promiscuous_donors
- recommendation_algorithms
- story_beliefs
- targeting_fairness

### Studies using "labels" format with human data at study level:
- context_effects (human: "context effects human data labels anonymized.csv")
- default_eric (human: "defaults human data labels anonymized.csv")
- idea_generation (human: "idea generation human data labels anonymized.csv")
- preference_redistribution (human: "preferences for redistribution human data labels anonymized.csv")
- quantitative_intuition (human: "quantitative intuition human data labels anonymized.csv")

## Study-Specific Code Fixes

### context_effects: String Operations on NaN Values

```diff
-# category for binary choice
-conditions = [
-    df_human['Bin.Printer_TC'].notna() & df_human['Bin.Printer_TC'].str.strip().ne(''),
-    df_human['Bin.Cell_TC'].   notna() & df_human['Bin.Cell_TC'].   str.strip().ne(''),
-    df_human['Bin.TV_TC'].     notna() & df_human['Bin.TV_TC'].     str.strip().ne('')
-]
-choices = ['printer', 'cell', 'TV']
-for df in (df_human, df_twin):
-    df['category_binary'] = np.select(conditions, choices, default=np.nan)
+    # category for binary choice
+    for df in (df_human, df_twin):
+        conditions = [
+            df['Bin.Printer_TC'].notna() & (df['Bin.Printer_TC'].astype(str).str.strip() != ''),
+            df['Bin.Cell_TC'].notna() & (df['Bin.Cell_TC'].astype(str).str.strip() != ''),
+            df['Bin.TV_TC'].notna() & (df['Bin.TV_TC'].astype(str).str.strip() != '')
+        ]
+        choices = ['printer', 'cell', 'TV']
+        df['category_binary'] = np.select(conditions, choices, default=None)
```

### preference_redistribution: Empty DataFrame Handling

```diff
-pair = (
-df[cols]
-  .dropna(subset=[col_h, col_t])
-)
-min_val = pair[min_col].iloc[0]
-max_val = pair[max_col].iloc[0]
-n    = len(pair)
+        pair = (
+        df[cols]
+          .dropna(subset=[col_h, col_t])
+        )
+        n = len(pair)
+        if n == 0:
+            # Handle empty dataframe case
+            min_val = np.nan
+            max_val = np.nan
+        else:
+            min_val = pair[min_col].iloc[0]
+            max_val = pair[max_col].iloc[0]
```

### junk_fees and story_beliefs: Additional Output Files

Both scripts create processed data files in addition to the standard meta-analysis outputs:

```diff
+    # Save processed data files
+    human_processed_file = output_path / "human data values anonymized processed.csv"
+    df_human.to_csv(human_processed_file, index=False)
+    
+    twin_processed_file = output_path / "twins data values anonymized processed.csv"
+    df_twin.to_csv(twin_processed_file, index=False)
```

### Output Directory Override Fix (Applied to All Scripts)

The recent fix ensures that when --results-dir is provided, the script doesn't override an explicitly provided --output-dir:

```diff
-        args.output_dir = str(results_path)
+        # Only set output_dir if not explicitly provided
+        if not args.output_dir:
+            args.output_dir = str(results_path)
```

## Figure Generation Scripts

The following scripts generate matplotlib figures and save them to the figures/ subdirectory:

### digital_certification
- log_WTP_by_respondent_and_item.png
- status_perception_by_respondent_and_item.png

### hiring_algorithms
- algorithmic_vs_team_item_1.png through algorithmic_vs_team_item_8.png

### obedient_twins
- task3_by_twin_type_and_absurd_condition.png

### promiscuous_donors
- regression_coefficients_comparison.png

### story_beliefs
- Story_chapters_correlation_heatmap.png (in notebook but removed in script conversion)

## Summary of Changes

1. **Command-line interface**: All scripts now accept command-line arguments for flexible path configuration
2. **Project root execution**: Scripts can be run from the project root directory with proper path resolution
3. **Path configuration**: Automatic file discovery based on study_folder_mapping.json specifications
4. **Output organization**: All outputs are saved to specified directory with figures/ and tables/ subdirectories
5. **Robustness**: Added error handling for missing files and empty dataframes
6. **Consistency**: Standardized output filenames across all studies
7. **Memory management**: Matplotlib figures are saved and closed to prevent memory issues

These changes maintain the exact analysis logic while making the scripts more flexible, robust, and suitable for automated pipelines.