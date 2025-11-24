#!/usr/bin/env python3
"""
Analyze remaining test failures to categorize issues and prioritize fixes.
"""

import json
import sys
import os
import tempfile
import subprocess
import re
from collections import defaultdict

def categorize_error(error_msg: str) -> str:
    """Categorize error message into issue type."""

    if 'simulator' in error_msg.lower() and 'not defined' in error_msg.lower():
        return 'SIMULATOR_NOT_DEFINED'

    if 'No statevector for experiment' in error_msg:
        return 'STATEVECTOR_NOT_SAVED'

    if 'Expected' in error_msg and 'got' in error_msg:
        # Check if it's array ordering issue
        if re.search(r'Expected \[[^\]]+\], got \[[^\]]+\]', error_msg):
            # Check for dtype mismatch (int vs complex)
            if '+0.j' in error_msg or '+0j' in error_msg:
                return 'TYPE_MISMATCH_COMPLEX'
            # Check for value ordering issues
            return 'ARRAY_MISMATCH'

    if "has no attribute 'c_if'" in error_msg:
        return 'DEPRECATED_C_IF'

    if "has no attribute 'r1'" in error_msg:
        return 'DEPRECATED_R1'

    if 'Sum of amplitudes-squared is not 1' in error_msg:
        return 'NORMALIZATION_ERROR'

    if 'invalid literal for int()' in error_msg:
        return 'PARSE_ERROR'

    if 'is not iterable' in error_msg:
        return 'TYPE_ERROR'

    if 'Incorrect' in error_msg or 'should' in error_msg.lower():
        return 'LOGIC_ERROR'

    return 'OTHER'

def analyze_failures(jsonl_path: str):
    """Analyze all failures and categorize them."""

    print("=" * 80)
    print("Analyzing Remaining Failures")
    print("=" * 80)

    failure_categories = defaultdict(list)
    total_tests = 0
    failed_tests = 0

    venv_python = '.venv/bin/python3'
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable

    with open(jsonl_path, 'r') as f:
        entries = [json.loads(line) for line in f]

    print(f"\nAnalyzing {len(entries)} entries...\n")

    for i, entry in enumerate(entries, 1):
        total_tests += 1
        task_id = entry['task_id']

        # Create test code
        test_code = f"""
{entry['canonical_solution']}

{entry['test']}

if __name__ == '__main__':
    try:
        test_{entry['entry_point']}()
        print('PASS')
    except Exception as e:
        print(f'FAIL: {{e}}')
        import traceback
        traceback.print_exc()
"""

        # Write and execute
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_file = f.name

            result = subprocess.run(
                [python_exec, temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )

            os.unlink(temp_file)

            if 'PASS' not in result.stdout:
                failed_tests += 1
                error_msg = result.stdout + result.stderr
                category = categorize_error(error_msg)

                # Extract brief error
                if 'FAIL:' in error_msg:
                    brief_error = error_msg.split('FAIL:')[1].split('\n')[0][:100]
                else:
                    brief_error = error_msg[:100]

                failure_categories[category].append((task_id, brief_error))

                if i % 50 == 0:
                    print(f"Progress: {i}/{len(entries)}")

        except subprocess.TimeoutExpired:
            failed_tests += 1
            failure_categories['TIMEOUT'].append((task_id, 'Test timed out'))
            os.unlink(temp_file)
        except Exception as e:
            failed_tests += 1
            failure_categories['EXECUTION_ERROR'].append((task_id, str(e)))

    # Print summary
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal tests: {total_tests}")
    print(f"Failed tests: {failed_tests}")
    print(f"Pass rate: {((total_tests - failed_tests) / total_tests * 100):.1f}%")

    print("\n" + "-" * 80)
    print("FAILURE CATEGORIES")
    print("-" * 80)

    # Sort by count
    sorted_categories = sorted(failure_categories.items(), key=lambda x: len(x[1]), reverse=True)

    for category, failures in sorted_categories:
        print(f"\n{category}: {len(failures)} failures")
        print("-" * 40)

        # Show first 3 examples
        for task_id, error in failures[:3]:
            print(f"  {task_id}")
            print(f"    Error: {error}")

        if len(failures) > 3:
            print(f"  ... and {len(failures) - 3} more")

    # Write detailed report
    report_path = 'failure_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("Detailed Failure Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        for category, failures in sorted_categories:
            f.write(f"\n{category}: {len(failures)} failures\n")
            f.write("-" * 80 + "\n")
            for task_id, error in failures:
                f.write(f"{task_id}: {error}\n")
            f.write("\n")

    print(f"\n\nDetailed report written to: {report_path}")

    return failure_categories

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze test failures')
    parser.add_argument('--file', default='quantum_katas_dataset.jsonl',
                        help='JSONL dataset file')

    args = parser.parse_args()

    categories = analyze_failures(args.file)

    print(f"\n✓ Analysis complete!")
