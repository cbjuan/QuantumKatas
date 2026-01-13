# Quantum Katas Benchmark

Benchmark LLMs on quantum computing tasks using the Quantum Katas dataset translated to Qiskit.

## Dataset

The dataset contains **350 quantum computing tasks** across 26 categories, derived from Microsoft's [QuantumKatas](https://github.com/microsoft/QuantumKatas) and translated to Qiskit.

### Categories

| Category | Tasks | Description |
|----------|-------|-------------|
| BasicGates | 16 | Fundamental quantum gates |
| Superposition | 21 | Superposition state preparation |
| Measurements | 18 | Quantum measurements |
| DeutschJozsa | 15 | Deutsch-Jozsa algorithm |
| GroversAlgorithm | 8 | Grover's search algorithm |
| QFT | 16 | Quantum Fourier Transform |
| PhaseEstimation | 7 | Quantum Phase Estimation |
| ... | ... | And 19 more categories |

## Installation

```bash
# Using pip
pip install -e .

# Using uv (recommended)
uv pip install -e .
```

### Dependencies

- Python 3.10+
- qiskit >= 1.0.0
- qiskit-aer >= 0.15.0
- anthropic >= 0.40.0 (for Claude models)
- openai >= 1.0.0 (for GPT/vLLM models)

## Usage

### Command Line

```bash
# Run benchmark with Claude Sonnet
export ANTHROPIC_API_KEY="your-key"
qk-benchmark --model claude-sonnet --verbose

# Run benchmark with GPT-4o
export OPENAI_API_KEY="your-key"
qk-benchmark --model gpt-4o --verbose

# Filter by category
qk-benchmark --model claude-sonnet --categories BasicGates Superposition

# Filter by task IDs
qk-benchmark --model claude-sonnet --task-ids "BasicGates/1" "BasicGates/2"
```

### Python API

```python
from benchmark import BenchmarkRunner, ModelConfig, ProviderType

# Using predefined model
from benchmark import get_model_config

config = get_model_config("claude-sonnet")
runner = BenchmarkRunner(model_config=config)
results = runner.run(verbose=True)
results.save("results/claude-sonnet.json")

# Custom model configuration
config = ModelConfig(
    provider=ProviderType.OPENAI,
    model_id="gpt-4o",
    temperature=0.0,
)
runner = BenchmarkRunner(model_config=config)
results = runner.run()
```

### Using vLLM

```python
from benchmark import BenchmarkRunner, ModelConfig, ProviderType

config = ModelConfig(
    provider=ProviderType.VLLM,
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    base_url="http://localhost:8000/v1",
)
runner = BenchmarkRunner(model_config=config)
results = runner.run(verbose=True)
```

### Using Qiskit Code Assistant

```python
from benchmark import BenchmarkRunner, ModelConfig, ProviderType

config = ModelConfig(
    provider=ProviderType.QISKIT_ASSISTANT,
    model_id="mistral-small-3.2-24b-qiskit",
    api_key="your-ibm-token",
)
runner = BenchmarkRunner(model_config=config)
results = runner.run(verbose=True)
```

## Available Models

### Anthropic
- `claude-opus` - Claude Opus 4
- `claude-sonnet` - Claude Sonnet 4
- `claude-haiku` - Claude Haiku 4

### OpenAI
- `gpt-4o` - GPT-4o
- `gpt-4o-mini` - GPT-4o Mini
- `o1` - o1
- `o1-mini` - o1 Mini

### Qiskit Code Assistant
- `mistral-qiskit` - Mistral Small 3.2 24B (Qiskit-tuned)

## Results

Results are saved in JSON format with:
- Pass/fail status for each task
- Generated code
- Evaluation details (syntax errors, test failures)
- Token usage and latency metrics

### Generate Report

```python
from benchmark import load_results, generate_report, format_markdown_report

results = load_results("results/claude-sonnet.json")
report = generate_report(results)
print(format_markdown_report(report))
```

## Dataset Format

Each task in `dataset/quantum_katas.jsonl` has:

```json
{
  "task_id": "BasicGates/1",
  "prompt": "# Task 1. State flip\n# Input: A qubit in state |ψ⟩ = α|0⟩ + β|1⟩\n# Goal: Change the state to α|1⟩ + β|0⟩...",
  "canonical_solution": "from qiskit import QuantumCircuit\n\ndef state_flip(qc, q):\n    qc.x(q)\n    return qc",
  "test": "def test_state_flip():\n    qc = QuantumCircuit(1)\n    ...",
  "entry_point": "state_flip"
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

Based on Microsoft's [QuantumKatas](https://github.com/microsoft/QuantumKatas) (MIT License).
