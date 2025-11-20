#!/usr/bin/env python3
"""
Additional targeted fixes for quantum katas dataset.
Focus on conservative, high-confidence fixes.
"""

import json
import re
from typing import Dict, List

def load_dataset(filename: str) -> List[Dict]:
    """Load JSONL dataset."""
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

def save_dataset(filename: str, entries: List[Dict]):
    """Save JSONL dataset."""
    with open(filename, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

def fix_ripplecarryadder_aersimulator(entry: Dict) -> bool:
    """Fix RippleCarryAdder tests to use Statevector instead of AerSimulator."""
    if not entry['task_id'].startswith('RippleCarryAdder/'):
        return False

    test = entry['test']

    # Pattern: sim = AerSimulator(method='statevector')
    #          result = sim.run(qc).result().get_statevector()
    # Replace with: statevector = Statevector.from_instruction(qc)

    if 'AerSimulator' in test and 'get_statevector()' in test:
        # Replace the old pattern
        test = re.sub(
            r'sim = AerSimulator\(method=[\'"]statevector[\'"]\)\s*\n\s*result = sim\.run\(qc\)\.result\(\)\.get_statevector\(\)',
            'statevector = Statevector.from_instruction(qc)',
            test
        )

        # Update variable references from result[x] to statevector.data[x]
        test = re.sub(
            r'assert abs\(result\[(\d+)\]\)',
            r'assert abs(statevector.data[\1])',
            test
        )

        # Add Statevector import if not present
        if 'from qiskit.quantum_info import Statevector' not in test:
            # Add after other imports
            lines = test.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('from qiskit import'):
                    indent = len(line) - len(line.lstrip())
                    lines.insert(i + 1, ' ' * indent + 'from qiskit.quantum_info import Statevector')
                    break
            test = '\n'.join(lines)

        entry['test'] = test
        return True

    return False

def fix_type_mismatch_dtype(entry: Dict) -> bool:
    """Fix type mismatches by ensuring test arrays use dtype=complex."""
    test = entry['test']

    # Look for np.array comparisons without dtype=complex
    if 'np.array([' in test and 'dtype=complex' not in test:
        # Check if this is likely a comparison with statevector
        if 'statevector' in test.lower() and 'np.allclose' in test:
            # Add dtype=complex to numpy arrays
            test = re.sub(
                r'np\.array\(\[([^\]]+)\]\)(?!\s*,\s*dtype)',
                r'np.array([\1], dtype=complex)',
                test
            )
            entry['test'] = test
            return True

    return False

def main():
    """Main fix routine."""
    print("=" * 80)
    print("Additional Targeted Fixes - Pass 2")
    print("=" * 80)

    # Load dataset
    entries = load_dataset('quantum_katas_dataset.jsonl')
    print(f"\nLoaded {len(entries)} entries")

    # Track fixes
    fixes = {
        'ripplecarryadder_aersimulator': 0,
        'type_mismatch_dtype': 0
    }

    # Apply fixes
    for entry in entries:
        if fix_ripplecarryadder_aersimulator(entry):
            fixes['ripplecarryadder_aersimulator'] += 1
        if fix_type_mismatch_dtype(entry):
            fixes['type_mismatch_dtype'] += 1

    # Report
    print("\nFixes Applied:")
    print("=" * 80)
    total_fixes = sum(fixes.values())
    for fix_type, count in fixes.items():
        if count > 0:
            print(f"  {fix_type.replace('_', ' ').title()}: {count} entries")
    print(f"\nTotal entries modified: {total_fixes}")

    if total_fixes > 0:
        # Save updated dataset
        save_dataset('quantum_katas_dataset.jsonl', entries)
        print("\n✅ Dataset updated successfully!")
    else:
        print("\n⚠️  No additional fixes applied")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
