# Benchmarking Guide - Qiskit Code Assistant

This guide explains how to benchmark the Qiskit Code Assistant on the Quantum Katas dataset.

## Overview

The benchmark evaluates the Qiskit Code Assistant's ability to generate correct Qiskit code for quantum computing tasks. It uses the OpenAI-compatible Completions API to generate solutions and tests them against the dataset.

## Quick Start

### 1. Get IBM Quantum API Token

1. Visit [IBM Quantum](https://quantum.ibm.com/)
2. Sign in to your account
3. Go to Account Settings → API Token
4. Copy your API token

### 2. Set Up Environment

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set your API token
export IBM_QUANTUM_TOKEN='your-api-token-here'
```

### 3. Run Benchmark

```bash
# Benchmark on verified dataset (275 tasks)
uv run python evaluate_qiskit_assistant.py

# Or with custom options
uv run python evaluate_qiskit_assistant.py \
  --dataset quantum_katas_dataset_verified.jsonl \
  --model mistral-small-3.2-24b-qiskit \
  --temperature 0.2 \
  --output results.json
```

## Running via GitHub Actions

The easiest way to run benchmarks is through GitHub Actions.

### Setup

1. **Add Secret to GitHub Repository**
   - Go to your repository on GitHub
   - Navigate to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `IBM_QUANTUM_TOKEN`
   - Value: Your IBM Quantum API token
   - Click "Add secret"

2. **Run Workflow**
   - Go to Actions tab in your repository
   - Select "Benchmark Qiskit Code Assistant"
   - Click "Run workflow"
   - Configure options:
     - **Dataset**: Choose verified (275 tasks) or complete (350 tasks)
     - **Max tasks**: Leave empty for all, or specify a number for testing
     - **Model**: Default is `mistral-small-3.2-24b-qiskit`
     - **Temperature**: 0.2 (lower = more deterministic)
   - Click "Run workflow"

3. **View Results**
   - Wait for the workflow to complete (may take 1-3 hours for full dataset)
   - Check the workflow summary for pass rate
   - Download the detailed results JSON from Artifacts

## Command Line Options

```bash
python evaluate_qiskit_assistant.py [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset file to use | `quantum_katas_dataset_verified.jsonl` |
| `--output` | Output file for results | `results_TIMESTAMP.json` |
| `--api-key` | IBM Quantum API token | From `IBM_QUANTUM_TOKEN` env var |
| `--base-url` | API base URL | `https://qiskit-code-assistant.quantum.ibm.com/v1` |
| `--model` | Model name | `mistral-small-3.2-24b-qiskit` |
| `--max-tasks` | Max tasks to evaluate | All tasks |
| `--temperature` | Sampling temperature | `0.2` |

### Examples

**Quick test with 10 tasks:**
```bash
uv run python evaluate_qiskit_assistant.py --max-tasks 10
```

**Full benchmark on verified dataset:**
```bash
uv run python evaluate_qiskit_assistant.py \
  --dataset quantum_katas_dataset_verified.jsonl \
  --output results_verified.json
```

**Full benchmark on complete dataset:**
```bash
uv run python evaluate_qiskit_assistant.py \
  --dataset quantum_katas_dataset.jsonl \
  --output results_complete.json
```

**Custom temperature:**
```bash
uv run python evaluate_qiskit_assistant.py --temperature 0.5
```

## Understanding Results

### Output Format

Results are saved as JSON with the following structure:

```json
{
  "model": "mistral-small-3.2-24b-qiskit",
  "base_url": "https://qiskit-code-assistant.quantum.ibm.com/v1",
  "dataset": "quantum_katas_dataset_verified.jsonl",
  "timestamp": "2025-01-15T10:30:00",
  "total_tasks": 275,
  "passed": 180,
  "failed": 95,
  "pass_rate": "65.45%",
  "results": [
    {
      "task_id": "BasicGates/1.1",
      "passed": true,
      "error": null,
      "solution": "def solve(...)..."
    },
    ...
  ]
}
```

### Metrics

- **Pass Rate**: Percentage of tasks that generated correct code
- **Total Tasks**: Number of tasks evaluated
- **Passed**: Number of tasks with correct solutions
- **Failed**: Number of tasks with incorrect or failing solutions

### Per-Task Results

Each task result includes:
- `task_id`: Unique identifier
- `passed`: Boolean indicating success
- `error`: Error message if failed (null if passed)
- `solution`: Generated code

## Comparing Results

### Baseline Performance

The canonical solutions have a **78.6% pass rate** on the verified dataset (275/350 tasks from complete dataset).

This represents the maximum achievable score, as some test expectations have known issues (see `dataset_metadata.json`).

### Expected LLM Performance

Typical LLM performance ranges:
- **Good**: 50-70% on verified dataset
- **Very Good**: 70-80% on verified dataset
- **Excellent**: >80% on verified dataset

## API Details

### Qiskit Code Assistant

- **Base URL**: `https://qiskit-code-assistant.quantum.ibm.com/v1`
- **Model**: `mistral-small-3.2-24b-qiskit`
- **API Type**: OpenAI-compatible Completions API (not Chat)
- **Documentation**: [IBM Quantum Docs](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-code-assistant-openai-api)

### Authentication

The API requires an IBM Quantum API token:
- Set as `IBM_QUANTUM_TOKEN` environment variable
- Or pass via `--api-key` argument
- Or set in `.env` file

### Rate Limiting

The script includes:
- Exponential backoff on failures
- 0.5s delay between requests
- 3 retry attempts per task

## Troubleshooting

### Error: "IBM_QUANTUM_TOKEN not found"

**Solution**: Set your API token
```bash
export IBM_QUANTUM_TOKEN='your-token-here'
```

Or create a `.env` file:
```bash
echo "IBM_QUANTUM_TOKEN=your-token-here" > .env
```

### Error: API connection failed

**Check:**
1. Your API token is valid
2. You have internet connection
3. The Qiskit Code Assistant service is available

### Error: Import errors

**Solution**: Install dependencies
```bash
uv sync
```

### Low pass rate

**Possible causes:**
1. Temperature too high (try 0.1-0.3)
2. Using complete dataset (has known issues)
3. Model performance baseline

**Try:**
- Use verified dataset: `--dataset quantum_katas_dataset_verified.jsonl`
- Lower temperature: `--temperature 0.1`
- Check example failures in results JSON

## Advanced Usage

### Custom Prompting

Edit the `create_prompt()` method in `evaluate_qiskit_assistant.py` to customize how tasks are presented to the model.

### Different Datasets

You can benchmark on:
- **Verified dataset** (275 tasks): Reliable evaluation, no test issues
- **Complete dataset** (350 tasks): Includes challenging/edge cases

### Batch Processing

For large-scale evaluation:
```bash
# Split into batches
for i in {0..4}; do
  START=$((i * 50))
  uv run python evaluate_qiskit_assistant.py \
    --max-tasks 50 \
    --output "results_batch_${i}.json" &
done
wait
```

### CI/CD Integration

Add the workflow to run on schedule:

```yaml
# In .github/workflows/benchmark-qiskit-assistant.yml
on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday
```

## Interpreting Results

### Category Analysis

To see performance by category:

```python
import json
from collections import defaultdict

with open('results.json') as f:
    data = json.load(f)

by_category = defaultdict(lambda: {'passed': 0, 'total': 0})

for result in data['results']:
    category = result['task_id'].split('/')[0]
    by_category[category]['total'] += 1
    if result['passed']:
        by_category[category]['passed'] += 1

for cat, stats in sorted(by_category.items()):
    rate = stats['passed'] / stats['total'] * 100
    print(f"{cat}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
```

### Common Failure Patterns

Check failed tasks for patterns:
```python
failures = [r for r in data['results'] if not r['passed']]

# Group by error type
error_types = defaultdict(list)
for f in failures:
    error = f['error'].split(':')[0] if f['error'] else 'Unknown'
    error_types[error].append(f['task_id'])

for error, tasks in error_types.items():
    print(f"{error}: {len(tasks)} tasks")
```

## Support

- **Issues**: [GitHub Issues](https://github.com/cbjuan/QuantumKatas/issues)
- **Documentation**: [DATASET_README.md](DATASET_README.md)
- **Qiskit Code Assistant**: [IBM Quantum Docs](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-code-assistant-openai-api)

## Contributing

Found issues or improvements? Please:
1. Test your changes locally
2. Update this documentation
3. Submit a pull request

---

**Note**: The Qiskit Code Assistant is in active development. Performance may improve over time as the model is updated.
