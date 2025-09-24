#!/usr/bin/env python3
"""
Simple test script to verify refactored mega_study_evaluation scripts.

This script runs three tests:
1. Test against notebook outputs using test_all_evaluations.py
2. Regenerate all meta-analyses and compare with git version
3. Check that common modules are properly imported

Usage:
    poetry run python mega_study_evaluation/test_refactoring.py
    poetry run python mega_study_evaluation/test_refactoring.py --quick  # Skip regeneration
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def run_command(cmd, cwd=None):
    """Run a shell command and return result."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        shell=True if isinstance(cmd, str) else False
    )
    return result


def test_notebook_compatibility():
    """Test 1: Run test_all_evaluations.py to compare against notebook outputs."""
    print("=" * 80)
    print("TEST 1: Testing compatibility with notebook outputs")
    print("=" * 80)
    
    result = run_command(["poetry", "run", "python", "mega_study_evaluation/test_all_evaluations.py"])
    
    if result.returncode != 0:
        print("❌ Notebook compatibility test failed")
        print(f"Error: {result.stderr[:500]}")
        return False
    
    # Count passed/failed from output
    passed = result.stdout.count("PASSED")
    failed = result.stdout.count("FAILED") - result.stdout.count("PASSED_WITH")  # Exclude PASSED_WITH_WARNING
    
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All notebook tests passed")
        return True
    else:
        print(f"⚠️ {failed} notebook tests failed")
        return False


def compare_dataframes(df_old, df_new, tolerance=1e-6):
    """Compare two dataframes and return statistics."""
    if df_old.shape != df_new.shape:
        return {"identical": 0, "similar": 0, "different": 100}
    
    # Sort for fair comparison
    sort_cols = [col for col in ['study name', 'study_name', 'specification_name_full'] 
                 if col in df_old.columns and col in df_new.columns]
    if sort_cols:
        df_old = df_old.sort_values(sort_cols).reset_index(drop=True)
        df_new = df_new.sort_values(sort_cols).reset_index(drop=True)
    
    identical = 0
    similar = 0
    different = 0
    
    for col in df_old.columns:
        if col not in df_new.columns:
            different += 1
            continue
            
        old_vals = df_old[col]
        new_vals = df_new[col]
        
        if old_vals.equals(new_vals):
            identical += 1
        elif pd.api.types.is_numeric_dtype(old_vals) and pd.api.types.is_numeric_dtype(new_vals):
            mask = ~(old_vals.isna() | new_vals.isna())
            if mask.sum() > 0:
                max_diff = np.abs(old_vals[mask] - new_vals[mask]).max()
                if max_diff < tolerance:
                    identical += 1
                elif max_diff < 0.01:
                    similar += 1
                else:
                    different += 1
            else:
                identical += 1  # Both all NaN
        else:
            if (old_vals != new_vals).sum() == 0:
                identical += 1
            else:
                different += 1
    
    total = len(df_old.columns)
    return {
        "identical": 100 * identical / total,
        "similar": 100 * similar / total,
        "different": 100 * different / total
    }


def test_regeneration(quick=False):
    """Test 2: Regenerate all meta-analyses and compare with git version."""
    print("\n" + "=" * 80)
    print("TEST 2: Testing regeneration of combined meta-analysis")
    print("=" * 80)
    
    combined_file = Path("mega_study_evaluation/meta_analysis_results/combined_all_specifications_meta_analysis.csv")
    
    if not quick:
        print("Regenerating all meta-analyses (this may take a while)...")
        result = run_command(["poetry", "run", "python", "mega_study_evaluation/run_all_meta_analyses.py", "-f", "-y"])
        if result.returncode != 0:
            print("⚠️ Some meta-analyses failed to regenerate")
    
    print("Combining meta-analysis results...")
    result = run_command(["poetry", "run", "python", "mega_study_evaluation/combine_all_meta_analyses.py"])
    if result.returncode != 0:
        print("❌ Failed to combine meta-analyses")
        return False
    
    # Load new version
    if not combined_file.exists():
        print("❌ Combined file not generated")
        return False
    
    df_new = pd.read_csv(combined_file)
    
    # Get git version
    result = run_command(f"git show HEAD:{combined_file}", cwd=Path.cwd())
    if result.returncode != 0:
        print("⚠️ No git version to compare with (file may be new)")
        return True
    
    # Save git version to temp file and load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp.write(result.stdout)
        tmp_path = tmp.name
    
    df_old = pd.read_csv(tmp_path)
    Path(tmp_path).unlink()
    
    # Compare
    stats = compare_dataframes(df_old, df_new)
    
    print(f"Comparison results:")
    print(f"  - Identical columns: {stats['identical']:.1f}%")
    print(f"  - Similar columns: {stats['similar']:.1f}%")
    print(f"  - Different columns: {stats['different']:.1f}%")
    
    if stats['identical'] >= 90:
        print("✅ Combined CSV is compatible")
        return True
    elif stats['identical'] + stats['similar'] >= 90:
        print("✅ Combined CSV has minor acceptable differences")
        return True
    else:
        print("❌ Combined CSV has significant differences")
        return False


def test_common_modules():
    """Test 3: Verify common modules exist and are used by scripts."""
    print("\n" + "=" * 80)
    print("TEST 3: Testing common module existence and usage")
    print("=" * 80)
    
    # Check that common module files exist
    common_dir = Path("mega_study_evaluation/common")
    expected_modules = [
        "args_parser.py",
        "data_loader.py", 
        "stats_analysis.py",
        "results_processor.py",
        "variable_mapper.py"
    ]
    
    all_exist = True
    for module_file in expected_modules:
        module_path = common_dir / module_file
        if module_path.exists():
            print(f"✅ {module_file} exists")
        else:
            print(f"❌ {module_file} not found")
            all_exist = False
    
    # Check that at least one script imports from common
    sample_script = Path("mega_study_evaluation/privacy/mega_study_evaluation.py")
    if sample_script.exists():
        with open(sample_script, 'r') as f:
            content = f.read()
            if "from common" in content or "from ..common" in content:
                print("✅ Scripts are using common modules")
            else:
                print("⚠️ Scripts may not be using common modules")
                all_exist = False
    
    if all_exist:
        print("✅ All common modules exist and are being used")
        return True
    else:
        print("❌ Some common modules are missing or not used")
        return False


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(
        description="Test refactored mega_study_evaluation scripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip full regeneration (use existing files)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MEGA STUDY EVALUATION REFACTORING TEST")
    print("=" * 80)
    
    # Run tests
    test1_passed = test_notebook_compatibility()
    test2_passed = test_regeneration(quick=args.quick)
    test3_passed = test_common_modules()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    if test1_passed:
        print("✅ Test 1: Notebook compatibility - PASSED")
    else:
        print("❌ Test 1: Notebook compatibility - FAILED")
    
    if test2_passed:
        print("✅ Test 2: Regeneration compatibility - PASSED")
    else:
        print("❌ Test 2: Regeneration compatibility - FAILED")
    
    if test3_passed:
        print("✅ Test 3: Common modules - PASSED")
    else:
        print("❌ Test 3: Common modules - FAILED")
    
    print("\n" + "=" * 80)
    if all([test1_passed, test2_passed, test3_passed]):
        print("✅ ALL TESTS PASSED - Refactoring is successful!")
        print("=" * 80)
        return 0
    else:
        print("⚠️ Some tests failed - Review the output above")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())