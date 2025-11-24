#!/usr/bin/env python3
"""
Comprehensive automated fix script for Qiskit Quantum Katas Dataset.
Handles all known API compatibility issues with Qiskit 2.x.
"""

import json
import re
from typing import Dict, List, Tuple

def fix_test_code_comprehensive(test_code: str) -> Tuple[str, List[str]]:
    """
    Comprehensively fix test code to use modern Qiskit API.
    """
    changes = []
    fixed = test_code

    # Fix 1: Replace all variations of the simulator + get_statevector pattern
    # This includes cases where simulator is defined separately

    # Pattern A: All-in-one (simulator defined + used)
    pattern_a = re.compile(
        r'(\s+)simulator = AerSimulator\((?:method=[\'"]statevector[\'"]\s*)?\)\s*\n'
        r'\s+job(\d*) = simulator\.run\(([^)]+)\)\s*\n'
        r'\s+result\2 = job\2\.result\(\)\s*\n'
        r'\s+statevector(\2) = result\2\.get_statevector\(\)',
        re.MULTILINE
    )

    def replace_pattern(match):
        """Replace function for statevector patterns."""
        indent = match.group(1)
        var_num = match.group(2)
        circuit = match.group(3).strip().split(',')[0].strip()
        return f'{indent}statevector{var_num} = Statevector.from_instruction({circuit})'

    match_count = len(pattern_a.findall(fixed))
    if match_count > 0:
        changes.append(f"Fixed {match_count} simulator+run patterns")
        fixed = pattern_a.sub(replace_pattern, fixed)

    # Pattern B: Simulator defined elsewhere, just job = simulator.run() part
    pattern_b = re.compile(
        r'(\s+)job(\d*) = simulator\.run\(([^)]+)\)\s*\n'
        r'\s+result\2 = job\2\.result\(\)\s*\n'
        r'\s+statevector(\2) = result\2\.get_statevector\(\)',
        re.MULTILINE
    )

    match_count_b = len(pattern_b.findall(fixed))
    if match_count_b > 0:
        changes.append(f"Fixed {match_count_b} job+result patterns")
        fixed = pattern_b.sub(replace_pattern, fixed)

    # Pattern C: result.get_statevector() with save_statevector()
    pattern_c = re.compile(
        r'(\s+)qc(\d*)\.save_statevector\(\)\s*\n'
        r'\s+(?:simulator = AerSimulator\(.*?\)\s*\n\s+)?'
        r'job(\d*) = simulator\.run\(([^)]+)\)\s*\n'
        r'\s+result\3 = job\3\.result\(\)\s*\n'
        r'\s+statevector(\d*) = result\3\.get_statevector\(\)',
        re.MULTILINE
    )

    def replace_c(match):
        indent = match.group(1)
        qc_num = match.group(2)
        var_num = match.group(5)
        circuit = match.group(4).strip().split(',')[0].strip()
        return f'{indent}statevector{var_num} = Statevector.from_instruction({circuit})'

    if pattern_c.search(fixed):
        changes.append("Fixed save_statevector + get_statevector patterns")
        fixed = pattern_c.sub(replace_c, fixed)

    # Fix 2: Remove orphaned simulator definitions
    if 'simulator.run(' not in fixed:
        simulator_defs = re.findall(r'\s*simulator = AerSimulator\([^)]*\)\s*\n', fixed)
        if simulator_defs:
            for sim_def in simulator_defs:
                fixed = fixed.replace(sim_def, '\n', 1)
            changes.append(f"Removed {len(simulator_defs)} orphaned simulator definitions")

    # Fix 3: Ensure Statevector import
    if 'Statevector.from_instruction' in fixed and 'from qiskit.quantum_info import Statevector' not in fixed:
        quantum_info_import = re.search(r'from qiskit\.quantum_info import ([^\n]+)', fixed)
        if quantum_info_import:
            existing = quantum_info_import.group(1)
            if 'Statevector' not in existing:
                new_imports = existing.rstrip() + ', Statevector'
                fixed = fixed.replace(
                    f'from qiskit.quantum_info import {existing}',
                    f'from qiskit.quantum_info import {new_imports}'
                )
                changes.append("Added Statevector to quantum_info import")
        else:
            import_match = re.search(r'^((?:import .*\n|from .* import .*\n)+)', fixed, re.MULTILINE)
            if import_match:
                last_import = import_match.end()
                fixed = fixed[:last_import] + 'from qiskit.quantum_info import Statevector\n' + fixed[last_import:]
            else:
                fixed = 'from qiskit.quantum_info import Statevector\n' + fixed
            changes.append("Added Statevector import")

    # Fix 4: Remove unused AerSimulator imports
    if 'AerSimulator' not in fixed or 'simulator' not in fixed:
        if 'from qiskit_aer import AerSimulator' in fixed:
            fixed = re.sub(r'from qiskit_aer import AerSimulator\n?', '', fixed)
            fixed = re.sub(r',\s*AerSimulator', '', fixed)
            fixed = re.sub(r'AerSimulator\s*,\s*', '', fixed)
            changes.append("Removed AerSimulator import")

    # Fix 5: Clean up multiple blank lines
    fixed = re.sub(r'\n\n\n+', '\n\n', fixed)

    return fixed, changes

def fix_canonical_solution(solution_code: str) -> Tuple[str, List[str]]:
    """
    Fix canonical solution code if needed.
    Most canonical solutions shouldn't need fixes, but check for deprecated APIs.
    """
    changes = []
    fixed = solution_code

    # Fix deprecated .c_if() calls - replaced with if_test() in Qiskit 2.x
    if '.c_if(' in fixed:
        # This is a breaking change - c_if is removed in Qiskit 2.x
        # We need to update to use the new control flow
        changes.append("WARNING: Uses deprecated .c_if() - needs manual review")

    return fixed, changes

def fix_dataset_entry(entry: Dict) -> Tuple[Dict, List[str]]:
    """Fix a single dataset entry."""
    changes = []
    fixed_entry = entry.copy()

    # Fix test code
    if 'test' in entry:
        fixed_test, test_changes = fix_test_code_comprehensive(entry['test'])
        if test_changes:
            fixed_entry['test'] = fixed_test
            changes.extend([f"Test: {c}" for c in test_changes])

    # Check canonical solution
    if 'canonical_solution' in entry:
        fixed_solution, solution_changes = fix_canonical_solution(entry['canonical_solution'])
        if solution_changes:
            fixed_entry['canonical_solution'] = fixed_solution
            changes.extend([f"Solution: {c}" for c in solution_changes])

    return fixed_entry, changes

def fix_dataset_file(input_path: str, output_path: str):
    """Fix entire dataset file."""
    print("=" * 80)
    print("Qiskit Quantum Katas Dataset - Comprehensive Fix")
    print("=" * 80)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}\n")

    entries_fixed = 0
    total_entries = 0
    all_changes = []
    warnings = []

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            total_entries += 1
            entry = json.loads(line)
            task_id = entry.get('task_id', f'line_{line_num}')

            fixed_entry, changes = fix_dataset_entry(entry)

            if changes:
                entries_fixed += 1
                print(f"✓ Fixed {task_id}")
                for change in changes:
                    print(f"  - {change}")
                    if 'WARNING' in change:
                        warnings.append((task_id, change))
                all_changes.append((task_id, changes))

            # Write fixed entry
            f_out.write(json.dumps(fixed_entry) + '\n')

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total entries: {total_entries}")
    print(f"Entries fixed: {entries_fixed}")
    print(f"Entries unchanged: {total_entries - entries_fixed}")

    if warnings:
        print(f"\n⚠️  Warnings: {len(warnings)} entries need manual review")
        for task_id, warning in warnings:
            print(f"  - {task_id}: {warning}")

    print(f"\nFixed dataset written to: {output_path}")

    # Write change log
    log_path = output_path.replace('.jsonl', '_changes.log')
    with open(log_path, 'w') as f:
        f.write("Dataset Comprehensive Fix Change Log\n")
        f.write("=" * 80 + "\n\n")
        for task_id, changes in all_changes:
            f.write(f"{task_id}:\n")
            for change in changes:
                f.write(f"  - {change}\n")
            f.write("\n")

        if warnings:
            f.write("\n" + "=" * 80 + "\n")
            f.write("WARNINGS - Manual Review Required\n")
            f.write("=" * 80 + "\n\n")
            for task_id, warning in warnings:
                f.write(f"{task_id}: {warning}\n")

    print(f"Change log written to: {log_path}")

    return entries_fixed, total_entries, len(warnings)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive fix for Qiskit API issues')
    parser.add_argument('--input', default='quantum_katas_dataset.jsonl',
                        help='Input JSONL file')
    parser.add_argument('--output', default='quantum_katas_dataset_fixed.jsonl',
                        help='Output JSONL file')

    args = parser.parse_args()

    # Run the comprehensive fix
    fixed, total, warn_count = fix_dataset_file(args.input, args.output)

    print(f"\n✓ Complete! {fixed}/{total} entries fixed")
    if warn_count > 0:
        print(f"⚠️  {warn_count} entries flagged for manual review")
