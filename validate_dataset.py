#!/usr/bin/env python3
"""
Comprehensive validation script for the Qiskit Quantum Katas Dataset.
Tests each canonical solution against its test cases and reports issues.
"""

import json
import sys
import traceback
import os
from typing import Dict, List, Tuple
import tempfile
import subprocess

def test_entry(entry: Dict, verbose: bool = False) -> Tuple[bool, str]:
    """
    Test a single dataset entry by executing its canonical solution with tests.

    Args:
        entry: Dictionary containing task_id, prompt, canonical_solution, test
        verbose: Print detailed output

    Returns:
        (success: bool, message: str)
    """
    task_id = entry['task_id']

    try:
        # Create a complete test file
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

        # Write to temp file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_file = f.name

        # Run the test using venv python
        venv_python = '.venv/bin/python3'
        python_exec = venv_python if os.path.exists(venv_python) else sys.executable

        result = subprocess.run(
            [python_exec, temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Clean up
        os.unlink(temp_file)

        if 'PASS' in result.stdout:
            if verbose:
                print(f"✓ {task_id}")
            return True, "PASS"
        else:
            error_msg = result.stdout + result.stderr
            return False, error_msg

    except subprocess.TimeoutExpired:
        return False, "Test timed out (>30s)"
    except Exception as e:
        return False, f"Exception: {str(e)}\n{traceback.format_exc()}"

def validate_dataset(jsonl_path: str, verbose: bool = False, stop_on_error: bool = False):
    """
    Validate all entries in the dataset.

    Args:
        jsonl_path: Path to the JSONL dataset file
        verbose: Print detailed output for each test
        stop_on_error: Stop on first error
    """
    print("=" * 80)
    print("Qiskit Quantum Katas Dataset Validation")
    print("=" * 80)

    passed = 0
    failed = 0
    errors = []

    with open(jsonl_path, 'r') as f:
        entries = [json.loads(line) for line in f]

    print(f"\nTotal entries to validate: {len(entries)}\n")

    for i, entry in enumerate(entries, 1):
        task_id = entry['task_id']

        if not verbose:
            # Show progress
            if i % 10 == 0:
                print(f"Progress: {i}/{len(entries)} ({passed} passed, {failed} failed)")

        success, message = test_entry(entry, verbose)

        if success:
            passed += 1
        else:
            failed += 1
            errors.append({
                'task_id': task_id,
                'message': message
            })
            if not verbose:
                print(f"✗ {task_id}: FAILED")
            else:
                print(f"✗ {task_id}")
                print(f"  Error: {message[:200]}...")

            if stop_on_error:
                print("\nStopping on first error (use --continue to test all)")
                break

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(entries)}")
    print(f"Passed: {passed} ({100*passed/len(entries):.1f}%)")
    print(f"Failed: {failed} ({100*failed/len(entries) if len(entries) > 0 else 0:.1f}%)")

    if errors:
        print("\n" + "=" * 80)
        print("FAILED TASKS")
        print("=" * 80)
        for error in errors:
            print(f"\n{error['task_id']}:")
            print(f"  {error['message'][:300]}")

    return passed, failed, errors

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate Qiskit Quantum Katas Dataset')
    parser.add_argument('--file', default='quantum_katas_dataset.jsonl', help='Path to JSONL file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--continue', dest='continue_on_error', action='store_true',
                        help='Continue testing after errors')

    args = parser.parse_args()

    passed, failed, errors = validate_dataset(
        args.file,
        verbose=args.verbose,
        stop_on_error=not args.continue_on_error
    )

    sys.exit(0 if failed == 0 else 1)
