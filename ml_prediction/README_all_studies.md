# ML Predictions for All Studies

This document describes how to run ML predictions across all studies using the `run_ml_predictions_all_studies.py` script.

## Prerequisites

The `mega_study_data` directory should be created as a symbolic link to your actual data location:

```bash
# Create symlink to mega study data
ln -s /path/to/your/mega_study_data mega_study_data
```

This allows the pipeline to access study-specific data files without hardcoding absolute paths.

## Overview

The script automatically:
1. Scans all config files for `ml_simulation` sections
2. Extracts training features from wave 1-3 CSVs
3. Identifies prediction targets from study-specific values CSVs
4. Trains XGBoost models for each study
5. Generates a summary report

## Configuration

Each study config must have an `ml_simulation` section:

```yaml
ml_simulation:
  training_data:
    wave_files:
      - data/wave_csv/wave_1_numbers_anonymized.csv
      - data/wave_csv/wave_2_numbers_anonymized.csv
      - data/wave_csv/wave_3_numbers_anonymized.csv
    additional_features:
      source: mega_study_data/{study_folder}/{study}_values_anonymized.csv
      columns: []  # Optional study-specific features
  
  prediction_targets:
    source: mega_study_data/{study_folder}/{study}_values_anonymized.csv
    columns: auto  # 'auto' uses all non-metadata columns
    exclude_columns:
      - StartDate
      - EndDate
      - Progress
      - Duration (in seconds)
      - TWIN_ID
      - "*_RT_*"  # Response time columns
      - "*_DO_*"  # Display order columns
  
  ml_config:
    use_nested_cv: true
    cv_folds: 3
    max_personas: -1
    model_name: XGBoost
  
  output:
    base_dir: ml_prediction/output/{study_name}
    save_predictions: true
    save_model: true
    generate_mad_evaluation: false
```

## Usage

### Quick Test Mode

Test with reduced cross-validation and limited personas:

```bash
# Minimal test - 2 CV folds, no tuning, 100 personas
poetry run python ml_prediction/run_ml_predictions_all_studies.py \
    --test \
    --max-personas 100
```

### Standard Run with Reasonable Hyperparameters

For all studies with default settings (3 CV folds, 10 iterations):

```bash
# Run all studies with standard hyperparameter search
poetry run python ml_prediction/run_ml_predictions_all_studies.py
```

For specific studies:

```bash
# Run specific studies with standard settings
poetry run python ml_prediction/run_ml_predictions_all_studies.py \
    --studies accuracy_nudges story_beliefs privacy
```

### Comprehensive Hyperparameter Search

For publication-quality results with extensive hyperparameter tuning:

```bash
# Comprehensive search with 5 outer CV folds, 5 inner folds, 30 iterations
poetry run python ml_prediction/run_ml_predictions_all_studies.py \
    --cv-folds 5 \
    --inner-cv-folds 5 \
    --n-iter 30

# Run specific studies with comprehensive search
poetry run python ml_prediction/run_ml_predictions_all_studies.py \
    --studies accuracy_nudges story_beliefs \
    --cv-folds 5 \
    --inner-cv-folds 5 \
    --n-iter 30

# Run without hyperparameter tuning (faster, uses default XGBoost params)
poetry run python ml_prediction/run_ml_predictions_all_studies.py \
    --no-tuning \
    --cv-folds 5
```

### Command Line Parameters

- `--cv-folds`: Override number of outer CV folds (default: from config)
- `--inner-cv-folds`: Override number of inner CV folds for tuning (default: 3)
- `--n-iter`: Override number of hyperparameter search iterations (default: 10)
- `--no-tuning`: Disable hyperparameter tuning completely (uses defaults)
- `--test`: Enable test mode (2 CV folds, no tuning, 5 iterations)
- `--max-personas`: Limit number of personas per study
- `--studies`: Run specific studies only

### Test Mode vs Production Runs

```bash
# Quick test (2 folds, no tuning, limited data)
poetry run python ml_prediction/run_ml_predictions_all_studies.py \
    --test \
    --max-personas 200

# Standard production run (3 folds, nested CV, all data)
poetry run python ml_prediction/run_ml_predictions_all_studies.py

# Run with more personas for better results
poetry run python ml_prediction/run_ml_predictions_all_studies.py \
    --max-personas 1000
```

### Custom Configuration Directory

```bash
poetry run python ml_prediction/run_ml_predictions_all_studies.py \
    --configs-dir path/to/configs
```

### Monitoring Progress

The script provides detailed logging. To save logs:

```bash
# Save logs to file
poetry run python ml_prediction/run_ml_predictions_all_studies.py \
    2>&1 | tee ml_prediction_run_$(date +%Y%m%d_%H%M%S).log
```

## Output Structure

Each study generates:
```
ml_prediction/output/{study_name}/
├── xgboost_predictions.csv          # Model predictions (compact format)
├── xgboost_predictions_formatted.csv # Predictions matching original CSV structure
├── xgboost_results.json            # Metrics and hyperparameters
└── study_metadata.json             # Study-specific configuration
```

Summary report:
```
ml_prediction/all_studies_summary.json
```

### Formatted Output

The script generates two versions of predictions:
1. **xgboost_predictions.csv**: Compact format with only TWIN_ID and predicted columns
2. **xgboost_predictions_formatted.csv**: Full format matching the original values CSV structure
   - Includes all original columns (filled with defaults for non-predicted columns)
   - Maintains column order from original file
   - Useful for downstream analysis expecting the original format

## Test Mode

When using `--test`:
- CV folds reduced to 2
- Nested CV disabled (no hyperparameter tuning)
- Faster execution for development/testing

## Handling Warnings

The script will warn about:
- Duplicate columns between training features and prediction targets
- Missing wave files
- Non-numeric targets
- Insufficient samples for prediction

## Integration with Existing Pipeline

This script reuses functions from `predict_answer_xgboost.py`:
- `train_xgboost_predictions()` - Core XGBoost training
- `NumpyEncoder` - JSON serialization

The original single-study workflow remains unchanged:
```bash
poetry run python ml_prediction/predict_answer_xgboost.py \
    --config ml_prediction/ml_prediction_config.yaml
```

## Adding New Studies

1. Create study folder in `mega_study_data/`
2. Add values CSV file with consistent naming
3. Create config in `configs/{study_name}/`
4. Add `ml_simulation` section to config
5. Run the script

## Troubleshooting

**No studies found**: Check that configs have `ml_simulation` sections

**Path errors**: Verify that values CSV files match the paths in configs

**Memory issues**: Use `--max-personas` to limit data size

**Slow execution**: Use `--test` mode or reduce `cv_folds` in configs