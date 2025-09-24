# Mega Study Evaluation Refactoring Documentation

## Overview

This document describes the refactoring of 19 mega_study_evaluation scripts to reduce code duplication through shared common modules. The refactoring extracted ~60% of duplicated code into reusable modules while maintaining backward compatibility.

## Refactoring Approach

### Goals
1. Extract common patterns into shared modules
2. Maintain exact output compatibility
3. Preserve study-specific logic
4. Improve maintainability and reduce duplication

### Common Modules Created

The refactoring created five common modules in `mega_study_evaluation/common/`:

#### 1. `args_parser.py`
- **Purpose**: Standardize command-line argument parsing
- **Key Functions**:
  - `create_base_parser()`: Creates base argument parser with standard options
  - `handle_file_discovery()`: Discovers input files based on patterns
- **Used by**: All 19 scripts

#### 2. `data_loader.py`
- **Purpose**: Centralize data loading and merging logic
- **Key Functions**:
  - `load_standard_data()`: Loads human and twin data files
  - `merge_twin_data()`: Merges twin data with human IDs
  - `prepare_data_for_analysis()`: Prepares merged data for analysis
- **Used by**: All scripts except those with special data requirements

#### 3. `stats_analysis.py`
- **Purpose**: Standardize statistical calculations
- **Key Functions**:
  - `compute_standard_stats()`: Computes correlations, t-tests, F-tests
  - `calculate_accuracy()`: Calculates accuracy metrics
  - `compute_fisher_ci()`: Computes Fisher z-transformed confidence intervals
- **Used by**: All 19 scripts

#### 4. `results_processor.py`
- **Purpose**: Handle result formatting and export
- **Key Functions**:
  - `create_meta_analysis_df()`: Creates standardized meta-analysis dataframe
  - `export_results()`: Exports results to CSV files
  - `format_for_output()`: Formats data for display
- **Used by**: All 19 scripts

#### 5. `variable_mapper.py`
- **Purpose**: Map variables between human and twin datasets
- **Key Functions**:
  - `get_variable_mappings()`: Returns study-specific variable mappings
  - `map_columns()`: Maps column names between datasets
- **Used by**: Scripts with variable name differences

## Refactoring Procedure

### Step 1: Analysis
1. Identify common patterns across all scripts
2. Document variable mappings and data flows
3. Create refactoring plan with phases

### Step 2: Create Common Infrastructure
```bash
# Create common directory
mkdir mega_study_evaluation/common

# Create common modules
touch mega_study_evaluation/common/__init__.py
touch mega_study_evaluation/common/args_parser.py
touch mega_study_evaluation/common/data_loader.py
touch mega_study_evaluation/common/stats_analysis.py
touch mega_study_evaluation/common/results_processor.py
touch mega_study_evaluation/common/variable_mapper.py
```

### Step 3: Refactor Scripts
For each script:
1. Create backup: `cp script.py script.py.backup`
2. Import common modules
3. Replace duplicated code with common functions
4. Test output matches original
5. Fix any issues

### Step 4: Testing
Run comprehensive tests to ensure compatibility:
```bash
# Test all scripts against notebook outputs
poetry run python mega_study_evaluation/test_all_evaluations.py

# Test full pipeline
poetry run python mega_study_evaluation/run_all_meta_analyses.py -f -y
poetry run python mega_study_evaluation/combine_all_meta_analyses.py

# Run master test script
poetry run python mega_study_evaluation/test_refactoring.py
```

## Testing Procedure

### Master Test Script
The `test_refactoring.py` script runs three comprehensive tests:

1. **Notebook Compatibility Test**
   - Runs all refactored scripts
   - Compares outputs with original notebook results
   - Validates meta-analysis and individual-level files

2. **Pipeline Compatibility Test**
   - Regenerates all meta-analyses
   - Combines results
   - Compares with git version

3. **Common Module Test**
   - Verifies all common modules exist
   - Checks that scripts import common modules

### Expected Differences
After refactoring, the following differences are acceptable:
- Floating-point differences in confidence intervals (<0.000002)
- Sample size changes from bug fixes
- New columns added (but no missing columns)
- NaN handling improvements

### Running Tests
```bash
# Full test suite
poetry run python mega_study_evaluation/test_refactoring.py

# Quick test (skip regeneration)
poetry run python mega_study_evaluation/test_refactoring.py --quick

# Individual script test
poetry run python mega_study_evaluation/test_all_evaluations.py
```

## File Structure

```
mega_study_evaluation/
├── common/                      # Shared modules
│   ├── __init__.py
│   ├── args_parser.py          # Argument parsing
│   ├── data_loader.py          # Data loading
│   ├── stats_analysis.py       # Statistical calculations
│   ├── results_processor.py    # Results formatting
│   └── variable_mapper.py      # Variable mapping
├── [study_name]/               # Individual study directories (19 total)
│   └── mega_study_evaluation.py # Refactored script
├── test_all_evaluations.py     # Notebook compatibility test
├── test_refactoring.py         # Master test script
├── run_all_meta_analyses.py    # Batch processing
├── combine_all_meta_analyses.py # Results combination
└── README.md                    # Main documentation

```

## Maintenance Guidelines

### Adding New Studies
1. Copy template from similar existing study
2. Import common modules
3. Implement study-specific logic only
4. Add to test configuration
5. Run full test suite

### Modifying Common Modules
1. Make changes in common module
2. Run full test suite to verify compatibility
3. Document any API changes
4. Update affected scripts if needed

### Debugging Issues
1. Check test_all_evaluations.py output for specific failures
2. Compare dataframe shapes and columns
3. Look for Unicode/encoding issues
4. Verify file paths and naming conventions

