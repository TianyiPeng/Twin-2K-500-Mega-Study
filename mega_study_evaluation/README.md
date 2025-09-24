# Mega Study Evaluation Scripts

Meta-analysis scripts for comparing human responses with AI twin responses across 19 behavioral studies.

## Quick Start: Update All Analyses

```bash
# Run all meta-analyses and update combined files
poetry run python mega_study_evaluation/update_all_analyses.py

# Or run individual steps:
# 1. Run all individual meta-analyses
poetry run python mega_study_evaluation/run_all_meta_analyses.py

# 2. Combine all results
poetry run python mega_study_evaluation/combine_all_meta_analyses.py

# 3. Create summary tables
poetry run python mega_study_evaluation/create_summary_table.py
```

## Running Individual Studies

```bash
# Using results directory (recommended)
poetry run python mega_study_evaluation/[study_name]/mega_study_evaluation.py \
    --results-dir results/[study_name]/[specification]

# Example
poetry run python mega_study_evaluation/story_beliefs/mega_study_evaluation.py \
    --results-dir results/story_beliefs/full_persona_without_reasoning_2025-06-20
```

## Testing Framework

```bash
# Test all converted scripts
poetry run python mega_study_evaluation/test_all_evaluations.py

# Quick compatibility test after refactoring
poetry run python mega_study_evaluation/test_refactoring.py --quick
```

## Study Status (19 Studies - All Functional)

All studies successfully converted and tested:
- accuracy_nudges, affective_priming, consumer_minimalism, context_effects
- default_eric, digital_certification, hiring_algorithms, idea_evaluation
- idea_generation*, infotainment, junk_fees, obedient_twins
- preference_redistribution, privacy, promiscuous_donors, quantitative_intuition
- recommendation_algorithms, story_beliefs, targeting_fairness

*idea_generation requires Word2Vec model (~1.5GB) and optionally uses creativity rating files

## Key Infrastructure

### Core Modules
- **parse_specification.py**: Parses specification names and dates
- **run_all_meta_analyses.py**: Batch processes all study/specification combinations
- **combine_all_meta_analyses.py**: Aggregates results across studies
- **create_summary_table.py**: Generates summary tables
- **update_all_analyses.py**: Orchestrates the full pipeline

### Common Modules (in `common/`)
- **args_parser.py**: Command-line argument handling
- **data_loader.py**: Data loading and preprocessing
- **stats_analysis.py**: Statistical computations
- **results_processor.py**: Output formatting
- **variable_mapper.py**: Variable mapping and domain classification

## Output Files

Each study generates:
- `meta analysis.csv` - Main results with correlations and statistics
- `meta analysis individual level.csv` - Individual-level data
- Additional study-specific outputs (figures, tables)

Combined outputs in `meta_analysis_results/`:
- `combined_all_specifications_meta_analysis.csv` - All data combined
- `summary_by_persona_specification_*.csv` - Summary statistics
- `correlation_by_persona_and_study.csv` - Correlation matrix
- `performance_comparison_by_metric.csv` - Best performers

## Data Requirements

Standard files in results directory:
- `consolidated_original_answers_values.csv` (human data)
- `consolidated_llm_values.csv` (AI twin data)

Some studies use alternative formats (see `study_folder_mapping.json`)

## Batch Processing Options

```bash
# Force rerun even if outputs exist
poetry run python mega_study_evaluation/run_all_meta_analyses.py --force

# Process specific studies
for study in affective_priming consumer_minimalism; do
    poetry run python mega_study_evaluation/$study/mega_study_evaluation.py \
        --results-dir results/$study/full_persona_without_reasoning_2025-06-20
done
```

## Requirements

All dependencies managed via Poetry:
```bash
poetry install
```

Key dependencies: Python 3.11.7+, pandas, numpy, scipy, matplotlib, seaborn

## Contributing

When converting notebooks:
1. Preserve data processing logic exactly
2. Add command-line argument parsing
3. Test against original outputs
4. Document special requirements
5. Update `study_folder_mapping.json` if needed