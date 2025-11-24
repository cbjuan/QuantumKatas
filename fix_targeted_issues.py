#!/usr/bin/env python3
"""
Targeted fixes for specific, easily fixable failure categories.
Focuses on high-confidence fixes with minimal risk.
"""

import json
import re
import sys
from pathlib import Path

def fix_simulator_not_defined(test_code: str) -> tuple[str, bool]:
    """
    Fix tests that reference 'simulator' variable that doesn't exist.
    These are leftover lines from old patterns where simulator.run() was used
    but the result wasn't actually used.
    """
    modified = False

    # Pattern: job/result/state_before = simulator.run() that's not used
    # These lines compute something but don't use it, and can be removed
    patterns = [
        # Pattern: job_before = simulator.run(qc.copy())
        # followed by: state_before = job_before.result().get_statevector()
        # This is followed by Statevector.from_instruction() so the simulator line is unused
        (r'\n\s+job_before = simulator\.run\([^)]+\)\s*\n\s+state_before = job_before\.result\(\)\.get_statevector\(\)\s*\n',
         '\n'),

        # Pattern: job_orig = simulator.run(qc.copy())
        # followed by: original_state = job_orig.result().get_statevector()
        (r'\n\s+job_orig = simulator\.run\([^)]+\)\s*\n\s+original_state = job_orig\.result\(\)\.get_statevector\(\)\s*\n',
         '\n'),

        # Pattern: standalone job = simulator.run() that's never used
        (r'\n\s+job\d* = simulator\.run\([^)]+\)\s*\n',
         '\n'),
    ]

    for pattern, replacement in patterns:
        new_code = re.sub(pattern, replacement, test_code)
        if new_code != test_code:
            test_code = new_code
            modified = True

    return test_code, modified

def fix_statevector_not_saved(test_code: str) -> tuple[str, bool]:
    """
    Fix tests that use result.get_statevector() which doesn't work in Qiskit 2.x
    Should use Statevector.from_instruction() instead.
    """
    modified = False

    # Pattern 1: result = simulator.run(qc).result()
    #            statevector = result.get_statevector()
    pattern1 = re.compile(
        r'(\s+)(\w+) = simulator\.run\(([^)]+)\)\.result\(\)\s*\n'
        r'\s+(\w+) = \2\.get_statevector\(\)',
        re.MULTILINE
    )

    def replace_match1(match):
        indent = match.group(1)
        circuit = match.group(3).strip()
        # Remove .copy() if present since from_instruction doesn't need it
        circuit = circuit.replace('.copy()', '')
        statevector_var = match.group(4)
        return f'{indent}{statevector_var} = Statevector.from_instruction({circuit})'

    new_code = pattern1.sub(replace_match1, test_code)
    if new_code != test_code:
        test_code = new_code
        modified = True

    # Pattern 2: Two-line version with job variable
    #   job_before = simulator.run(qc.copy())
    #   state_before = job_before.result().get_statevector()
    # Use .+? to handle nested parentheses
    pattern2 = re.compile(
        r'(\s+)(\w+) = simulator\.run\((.+?)\)\s*\n'
        r'\s+(\w+) = \2\.result\(\)\.get_statevector\(\)',
        re.MULTILINE | re.DOTALL
    )

    def replace_match2(match):
        indent = match.group(1)
        circuit = match.group(3).strip()
        # Remove .copy() if present
        circuit = circuit.replace('.copy()', '')
        statevector_var = match.group(4)
        return f'{indent}{statevector_var} = Statevector.from_instruction({circuit})'

    new_code2 = pattern2.sub(replace_match2, test_code)
    if new_code2 != test_code:
        test_code = new_code2
        modified = True

    # Pattern 3: job variables that don't exist
    # Pattern: statevectorN = jobN.result().get_statevector()
    # where jobN is never defined, and we have qcN available
    pattern3 = re.compile(
        r'(\s+)statevector(\d+) = job\2\.result\(\)\.get_statevector\(\)',
        re.MULTILINE
    )

    def replace_match3(match):
        indent = match.group(1)
        num = match.group(2)
        return f'{indent}statevector{num} = Statevector.from_instruction(qc{num})'

    new_code3 = pattern3.sub(replace_match3, test_code)
    if new_code3 != test_code:
        test_code = new_code3
        modified = True

    # Pattern 4: Generic job.result().get_statevector().data
    # Look for a circuit variable in context
    pattern4 = re.compile(
        r'(\s+)(\w+) = job\.result\(\)\.get_statevector\(\)\.data',
        re.MULTILINE
    )

    # This one is tricky - need to find the circuit name from context
    # Look for qc or qc_test or similar in preceding lines
    def replace_match4(match):
        indent = match.group(1)
        var_name = match.group(2)
        # Try to find circuit name - look for recent circuit references
        # For now, assume qc_test based on the context we saw
        # This is a heuristic that might need manual review
        # We'll look for the most recent circuit variable assignment before this line
        lines_before = test_code[:match.start()].split('\n')
        circuit_name = 'qc'  # default
        for line in reversed(lines_before[-10:]):  # Check last 10 lines
            if '= QuantumCircuit(' in line or 'qc_' in line or ' qc' in line:
                # Extract circuit name
                if ' qc_test' in line or 'qc_test = ' in line:
                    circuit_name = 'qc_test'
                    break
                elif ' qc' in line:
                    # Try to extract variable name
                    import re as re_inner
                    match_qc = re_inner.search(r'(qc\w*)', line)
                    if match_qc:
                        circuit_name = match_qc.group(1)
                        break
        return f'{indent}{var_name} = Statevector.from_instruction({circuit_name}).data'

    new_code4 = pattern4.sub(replace_match4, test_code)
    if new_code4 != test_code:
        test_code = new_code4
        modified = True

    # Pattern 5: sim = AerSimulator(...); result = sim.run(qc).result().get_statevector()
    # Replace with just Statevector.from_instruction(qc)
    pattern5 = re.compile(
        r'(\s+)sim = AerSimulator\([^)]*\)\s*\n'
        r'\s+(\w+) = sim\.run\((\w+)\)\.result\(\)\.get_statevector\(\)',
        re.MULTILINE
    )

    def replace_match5(match):
        indent = match.group(1)
        result_var = match.group(2)
        circuit = match.group(3)
        return f'{indent}{result_var} = Statevector.from_instruction({circuit}).data'

    new_code5 = pattern5.sub(replace_match5, test_code)
    if new_code5 != test_code:
        test_code = new_code5
        modified = True

    if modified:
        # Make sure Statevector is imported
        if 'from qiskit.quantum_info import Statevector' not in test_code:
            # Add after qiskit imports
            test_code = test_code.replace(
                'from qiskit import QuantumCircuit',
                'from qiskit import QuantumCircuit\nfrom qiskit.quantum_info import Statevector'
            )

    return test_code, modified

def fix_aer_simulator_import(test_code: str, canonical: str) -> tuple[str, str, bool]:
    """
    Fix references to AerSimulator that shouldn't be there.
    """
    modified = False

    # Check if test has 'AerSimulator' but canonical doesn't - it's a leftover
    if 'AerSimulator' in test_code and 'AerSimulator' not in canonical:
        # Remove the import
        test_code = re.sub(r'from qiskit_aer import AerSimulator\s*\n', '', test_code)

        # Remove any simulator = AerSimulator() lines
        test_code = re.sub(r'\s+simulator = AerSimulator\([^)]*\)\s*\n', '\n', test_code)

        modified = True

    return test_code, canonical, modified

def fix_deprecated_c_if(test_code: str) -> tuple[str, bool]:
    """
    Fix deprecated .c_if() usage.
    In Qiskit 2.x, use .if_test() or measurement operations directly.
    """
    modified = False

    # This is complex and context-dependent, so we'll handle specific case
    # Pattern: measure_result = qc.measure(qubit, clbit)
    #          measure_result.c_if(clbit, value)
    # Replace with: qc.measure(qubit, clbit)
    #               with qc.if_test((clbit, value)):

    if '.c_if(' in test_code:
        # For now, just flag it - this needs manual review
        # The actual fix depends on what follows
        pass

    return test_code, modified

def apply_fixes(entry: dict) -> tuple[dict, list[str]]:
    """Apply all fixes to an entry."""
    fixes_applied = []
    modified_entry = entry.copy()

    test_code = entry['test']
    canonical = entry['canonical_solution']

    # Fix 1: Simulator not defined
    test_code, mod1 = fix_simulator_not_defined(test_code)
    if mod1:
        fixes_applied.append('SIMULATOR_NOT_DEFINED')

    # Fix 2: Statevector not saved
    test_code, mod2 = fix_statevector_not_saved(test_code)
    if mod2:
        fixes_applied.append('STATEVECTOR_NOT_SAVED')

    # Fix 3: AerSimulator import
    test_code, canonical, mod3 = fix_aer_simulator_import(test_code, canonical)
    if mod3:
        fixes_applied.append('AER_SIMULATOR_IMPORT')

    # Fix 4: Deprecated c_if
    test_code, mod4 = fix_deprecated_c_if(test_code)
    if mod4:
        fixes_applied.append('DEPRECATED_C_IF')

    if fixes_applied:
        modified_entry['test'] = test_code
        modified_entry['canonical_solution'] = canonical

    return modified_entry, fixes_applied

def main():
    input_file = 'quantum_katas_dataset.jsonl'
    output_file = 'quantum_katas_dataset_targeted.jsonl'

    print("=" * 80)
    print("Targeted Fixes for Specific Failure Categories")
    print("=" * 80)
    print()

    entries = []
    with open(input_file, 'r') as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries")
    print()

    # Track fixes
    fix_counts = {}
    modified_entries = 0

    # Target specific failing entries
    target_ids = {
        # SIMULATOR_NOT_DEFINED
        'BasicGates/1.2',
        'DeutschJozsa/1.1',
        'GroversAlgorithm/1.4',
        'GroversAlgorithm/2.2',
        'RippleCarryAdder/1.7',
        'KeyDistribution_BB84/1.2',
        # STATEVECTOR_NOT_SAVED
        'SimonsAlgorithm/1.1',
        'SimonsAlgorithm/1.2',
        'Teleportation/1.3',
        'Teleportation/1.4',
    }

    fixed_entries = []
    for entry in entries:
        if entry['task_id'] in target_ids:
            modified_entry, fixes = apply_fixes(entry)
            if fixes:
                print(f"✓ {entry['task_id']}: {', '.join(fixes)}")
                modified_entries += 1
                for fix in fixes:
                    fix_counts[fix] = fix_counts.get(fix, 0) + 1
                fixed_entries.append(modified_entry)
            else:
                fixed_entries.append(entry)
        else:
            fixed_entries.append(entry)

    # Write output
    with open(output_file, 'w') as f:
        for entry in fixed_entries:
            f.write(json.dumps(entry) + '\n')

    print()
    print("=" * 80)
    print("Fix Summary")
    print("=" * 80)
    print(f"Modified entries: {modified_entries}")
    print()
    print("Fixes applied:")
    for fix, count in sorted(fix_counts.items()):
        print(f"  {fix}: {count}")
    print()
    print(f"Output written to: {output_file}")

if __name__ == '__main__':
    main()
