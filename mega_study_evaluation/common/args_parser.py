"""Common argument parsing utilities for mega study evaluation scripts."""

import argparse
from pathlib import Path


def create_base_parser(study_name, description=None):
    """
    Create standard argument parser used by all studies.
    
    Args:
        study_name: Name of the study
        description: Optional custom description
    
    Returns:
        ArgumentParser with standard arguments
    """
    parser = argparse.ArgumentParser(
        description=description or f"Run meta-analysis for {study_name}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Option 1: Specify results directory (automatic file discovery)
    parser.add_argument("--results-dir", 
                       help="Path to results/{study}/{specification} directory")
    
    # Option 2: Specify individual files (for custom paths)
    parser.add_argument("--human-data", 
                       help="Path to human data CSV file")
    parser.add_argument("--twin-data", 
                       help="Path to twin data CSV file")
    parser.add_argument("--output-dir", 
                       help="Output directory for results")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress information")
    
    return parser


def handle_file_discovery(args, study_config=None):
    """
    Standard file path resolution logic.
    
    Args:
        args: Parsed arguments
        study_config: Dictionary with study-specific configuration
            - human_at_study_level: bool, whether human data is at study level
            - human_filename: str, filename for human data
            - twin_filename: str, filename for twin data
            - skiprows_twin: list, rows to skip for twin data (optional)
    
    Returns:
        Dictionary with resolved paths and configuration
    """
    paths = {}
    
    if args.results_dir:
        results_path = Path(args.results_dir)
        
        # Get configuration with defaults
        config = study_config or {}
        human_at_study_level = config.get('human_at_study_level', False)
        human_filename = config.get('human_filename', 'consolidated_original_answers_values.csv')
        twin_filename = config.get('twin_filename', 'consolidated_llm_values.csv')
        
        # Determine human data path
        if human_at_study_level:
            # Human data is at study level (parent of specification)
            study_path = results_path.parent
            paths['human_data'] = str(study_path / human_filename)
        else:
            # Human data is in specification folder
            paths['human_data'] = str(results_path / human_filename)
        
        # Twin data is always in specification folder
        paths['twin_data'] = str(results_path / twin_filename)
        
        # Output directory
        paths['output_dir'] = str(results_path)
        
        # Verify files exist
        if not Path(paths['human_data']).exists():
            raise FileNotFoundError(f"Human data not found: {paths['human_data']}")
        if not Path(paths['twin_data']).exists():
            raise FileNotFoundError(f"Twin data not found: {paths['twin_data']}")
    else:
        # Use individual file paths
        if not all([args.human_data, args.twin_data, args.output_dir]):
            raise ValueError(
                "Must specify either --results-dir OR all of --human-data, --twin-data, --output-dir"
            )
        paths['human_data'] = args.human_data
        paths['twin_data'] = args.twin_data
        paths['output_dir'] = args.output_dir
    
    # Create output directories
    output_path = Path(paths['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for visualizations
    figures_path = output_path / "figures"
    tables_path = output_path / "tables"
    figures_path.mkdir(exist_ok=True)
    tables_path.mkdir(exist_ok=True)
    
    paths['output_path'] = output_path
    paths['figures_path'] = figures_path
    paths['tables_path'] = tables_path
    
    return paths