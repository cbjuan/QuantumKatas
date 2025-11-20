#!/usr/bin/env python3
"""
Fix critical, high-impact issues in the Qiskit Quantum Katas Dataset.
Focus on automatable fixes with high success probability.

Priority fixes:
1. RippleCarryAdder indentation errors (23 entries) - HIGH IMPACT
2. Missing AerSimulator imports (6+ entries)
3. Normalization errors in initialize() (2 entries)
4. Other easily fixable issues
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

def fix_ripplecarryadder_indentation(entry: Dict) -> bool:
    """Fix indentation errors in RippleCarryAdder tests."""
    if not entry['task_id'].startswith('RippleCarryAdder/'):
        return False

    test = entry['test']
    fixed = False

    # Pattern 1: Fix "    import" that should be at same level as previous line
    # Look for: "from qiskit import QuantumCircuit\n        import"
    # Fix to: "from qiskit import QuantumCircuit\n    import"
    test = re.sub(
        r'(from qiskit import [^\n]+\n)        (import|qc = |from )',
        r'\1    \2',
        test
    )

    # Pattern 2: Fix general over-indentation after imports
    # Any line that has 8 spaces at start when it should have 4
    lines = test.split('\n')
    new_lines = []
    in_function = False

    for line in lines:
        if line.strip().startswith('def test_'):
            in_function = True
            new_lines.append(line)
        elif in_function and line.startswith('        ') and not line.strip().startswith('#'):
            # Check if this is over-indented (8 spaces when should be 4)
            # But NOT if it's inside a nested block
            prev_line = new_lines[-1] if new_lines else ''
            if 'def ' not in prev_line and ':' not in prev_line.rstrip():
                # This is likely over-indented
                new_lines.append('    ' + line.lstrip())
                fixed = True
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    if fixed:
        entry['test'] = '\n'.join(new_lines)
        return True

    # Simpler approach: look for specific pattern
    original_test = entry['test']
    test = entry['test']

    # Fix pattern: "from qiskit import X\n        Y" -> "from qiskit import X\n    Y"
    test = re.sub(
        r'(from qiskit[^\n]+\n)(        )([^\s])',
        r'\1    \3',
        test
    )

    if test != original_test:
        entry['test'] = test
        return True

    return False

def fix_missing_aersimulator_import(entry: Dict) -> bool:
    """Add missing AerSimulator import where needed."""
    test = entry['test']

    # Check if AerSimulator is used but not imported
    if 'AerSimulator' in test and 'from qiskit_aer import AerSimulator' not in test:
        # Add the import after other qiskit imports
        lines = test.split('\n')
        new_lines = []
        import_added = False

        for i, line in enumerate(lines):
            new_lines.append(line)
            # Add after the last qiskit import in the function
            if not import_added and line.strip().startswith('from qiskit import'):
                # Check if next line is not an import
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith(('from ', 'import ')):
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + 'from qiskit_aer import AerSimulator')
                    import_added = True

        if import_added:
            entry['test'] = '\n'.join(new_lines)
            return True

    return False

def fix_normalize_parameter(entry: Dict) -> bool:
    """Add normalize=True to qc.initialize() calls that are missing it."""
    canonical = entry['canonical_solution']

    # Check if initialize is used without normalize parameter
    if 'qc.initialize(' in canonical:
        # Check if normalize is already specified
        if re.search(r'qc\.initialize\([^)]*normalize\s*=', canonical):
            return False  # Already has normalize parameter

        # Add normalize=True
        canonical = re.sub(
            r'qc\.initialize\(([^)]+)\)',
            r'qc.initialize(\1, normalize=True)',
            canonical
        )

        entry['canonical_solution'] = canonical
        return True

    return False

def fix_statevector_issues(entry: Dict) -> bool:
    """Fix statevector-related issues."""
    canonical = entry['canonical_solution']
    test = entry['test']
    fixed = False

    # Pattern 1: result.get_statevector() -> Statevector.from_instruction()
    if 'result.get_statevector()' in canonical:
        canonical = canonical.replace(
            'result.get_statevector()',
            'Statevector.from_instruction(qc)'
        )
        if 'from qiskit.quantum_info import Statevector' not in canonical:
            # Add import at the top
            lines = canonical.split('\n')
            # Find where to insert (after existing imports)
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith(('from ', 'import ')):
                    insert_idx = i + 1
            lines.insert(insert_idx, 'from qiskit.quantum_info import Statevector')
            canonical = '\n'.join(lines)
        entry['canonical_solution'] = canonical
        fixed = True

    # Pattern 2: AerSimulator().run().result().get_statevector() pattern
    if 'save_statevector()' not in canonical and 'save_statevector()' not in test:
        # Check if using old simulator pattern for statevector
        pattern = r'AerSimulator\([^)]*\)\.run\([^)]+\)\.result\(\)\.get_statevector\(\)'
        if re.search(pattern, canonical):
            # This needs to use Statevector.from_instruction instead
            canonical = re.sub(
                pattern,
                'Statevector.from_instruction(qc)',
                canonical
            )
            if 'from qiskit.quantum_info import Statevector' not in canonical:
                lines = canonical.split('\n')
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith(('from ', 'import ')):
                        insert_idx = i + 1
                lines.insert(insert_idx, 'from qiskit.quantum_info import Statevector')
                canonical = '\n'.join(lines)
            entry['canonical_solution'] = canonical
            fixed = True

    return fixed

def main():
    """Main fix routine."""
    print("=" * 80)
    print("Critical Dataset Fixes - Targeted High-Impact Issues")
    print("Target: Fix 20-30 entries (71.4% -> 77-80%)")
    print("=" * 80)

    # Load dataset
    entries = load_dataset('quantum_katas_dataset.jsonl')
    print(f"\nLoaded {len(entries)} entries")

    # Track fixes
    fixes = {
        'ripplecarryadder_indentation': 0,
        'missing_aersimulator_import': 0,
        'normalize_parameter': 0,
        'statevector_issues': 0
    }

    # Apply fixes
    for entry in entries:
        if fix_ripplecarryadder_indentation(entry):
            fixes['ripplecarryadder_indentation'] += 1
        if fix_missing_aersimulator_import(entry):
            fixes['missing_aersimulator_import'] += 1
        if fix_normalize_parameter(entry):
            fixes['normalize_parameter'] += 1
        if fix_statevector_issues(entry):
            fixes['statevector_issues'] += 1

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
        print("\nNext: Run validation to verify fixes")
        print("  python3 validate_dataset.py --file quantum_katas_dataset.jsonl --continue")
    else:
        print("\n⚠️  No automatic fixes applied")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
