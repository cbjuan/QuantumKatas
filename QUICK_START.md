# Quick Start Guide - Quantum Katas Qiskit Dataset

## 1. Installation

```bash
pip install qiskit>=2.2 qiskit-aer>=0.13 numpy>=1.20
```

## 2. Load the Dataset

```python
import json

# Load verified dataset (recommended)
with open('quantum_katas_dataset_verified.jsonl') as f:
    dataset = [json.loads(line) for line in f]

print(f"Loaded {len(dataset)} verified tasks")
```

## 3. Example Task

```python
# Get first task
task = dataset[0]

print(f"Task ID: {task['task_id']}")
print(f"Prompt: {task['prompt'][:100]}...")
print(f"Entry point: {task['entry_point']}")
```

## 4. Evaluate a Solution

```python
def evaluate_solution(task, llm_generated_code):
    """
    Evaluate an LLM-generated solution.

    Args:
        task: Dictionary with task info
        llm_generated_code: String containing the generated function

    Returns:
        bool: True if test passes, False otherwise
    """
    # Combine solution with test
    full_code = llm_generated_code + '\n\n' + task['test']

    try:
        # Execute the test
        exec(full_code, {})
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

# Example: Test with canonical solution
task = dataset[0]
result = evaluate_solution(task, task['canonical_solution'])
print(f"Test passed: {result}")
```

## 5. Batch Evaluation

```python
def evaluate_llm(dataset, llm_generate_fn):
    """
    Evaluate an LLM on the entire dataset.

    Args:
        dataset: List of task dictionaries
        llm_generate_fn: Function that takes a prompt and returns code

    Returns:
        dict: Results with pass rate and details
    """
    results = {
        'total': len(dataset),
        'passed': 0,
        'failed': 0,
        'details': []
    }

    for task in dataset:
        # Generate solution from LLM
        generated_code = llm_generate_fn(task['prompt'])

        # Evaluate
        passed = evaluate_solution(task, generated_code)

        if passed:
            results['passed'] += 1
        else:
            results['failed'] += 1

        results['details'].append({
            'task_id': task['task_id'],
            'passed': passed
        })

    results['pass_rate'] = results['passed'] / results['total']
    return results

# Example usage (you need to implement your_llm_generate)
# results = evaluate_llm(dataset, your_llm_generate)
# print(f"Pass rate: {results['pass_rate']*100:.1f}%")
```

## 6. Using the Validation Script

```bash
# Validate the entire dataset
python3 validate_dataset.py --file quantum_katas_dataset_verified.jsonl

# Continue validation even if errors occur
python3 validate_dataset.py --file quantum_katas_dataset_verified.jsonl --continue
```

## 7. Check Task Categories

```python
from collections import Counter

# Count tasks by category
categories = [task['task_id'].split('/')[0] for task in dataset]
category_counts = Counter(categories)

print("Tasks by category:")
for category, count in sorted(category_counts.items()):
    print(f"  {category}: {count}")
```

## 8. Filter by Difficulty

```python
# Categories roughly ordered by difficulty
beginner = ['BasicGates', 'Superposition', 'Measurements']
intermediate = ['DeutschJozsa', 'Teleportation', 'QFT']
advanced = ['GroversAlgorithm', 'SimonsAlgorithm', 'CHSHGame']

# Get beginner tasks
beginner_tasks = [
    t for t in dataset
    if t['task_id'].split('/')[0] in beginner
]

print(f"Beginner tasks: {len(beginner_tasks)}")
```

## 9. Understanding Test Format

```python
task = dataset[0]

# The test expects your solution to be defined
print("Your function should be named:", task['entry_point'])

# The test will call your function with appropriate arguments
print("\nTest code:")
print(task['test'][:200], "...")

# The canonical solution shows the expected implementation
print("\nReference solution:")
print(task['canonical_solution'][:200], "...")
```

## 10. Common Patterns

### Pattern 1: Circuit Modification
```python
def task_name(qc: QuantumCircuit, ...) -> QuantumCircuit:
    # Modify the circuit
    qc.h(0)
    qc.cx(0, 1)
    return qc
```

### Pattern 2: State Preparation
```python
def prepare_state(qc: QuantumCircuit, qubits: list) -> QuantumCircuit:
    # Prepare specific quantum state
    for q in qubits:
        qc.h(q)
    return qc
```

### Pattern 3: Measurement
```python
def measure_basis(qc: QuantumCircuit, basis: str) -> QuantumCircuit:
    # Apply basis transformation and measure
    if basis == 'X':
        qc.h(0)
    qc.measure_all()
    return qc
```

## Tips for LLM Prompting

1. **Include Qiskit version**: "Using Qiskit 2.2+"
2. **Specify imports**: The tests import necessary modules
3. **Return type**: Always return the QuantumCircuit
4. **Function signature**: Match the entry_point name exactly

Example enhanced prompt:
```python
enhanced_prompt = f"""
Using Qiskit 2.2+, implement the following function:

{task['prompt']}

Function name: {task['entry_point']}

Return the modified QuantumCircuit.
"""
```

## Troubleshooting

**Issue**: ImportError for Qiskit modules
```bash
pip install --upgrade qiskit qiskit-aer
```

**Issue**: Tests failing with "name not defined"
- Ensure your function name matches `entry_point`
- Check that all required imports are in the test code

**Issue**: Statevector errors
- Use `Statevector.from_instruction(qc)` not deprecated methods
- Ensure numpy arrays use `dtype=complex` when needed

## Next Steps

- Read [DATASET_README.md](DATASET_README.md) for full documentation
- Check [dataset_metadata.json](dataset_metadata.json) for excluded tasks
- Review canonical solutions for implementation examples
- Compare your LLM's performance against the 78.6% baseline

Happy quantum computing! 🚀
