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
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Dependencies

- Python 3.10+
- qiskit >= 1.0.0
- qiskit-aer >= 0.15.0
- anthropic >= 0.40.0 (for Claude models)
- openai >= 1.0.0 (for GPT/vLLM models)
- google-generativeai >= 0.8.0 (for Gemini models)

## Usage

### Command Line

```bash
# Run benchmark with a specific model
uv run qk-benchmark --model claude-sonnet

# List available models
uv run qk-benchmark --list-models

# Filter by category
uv run qk-benchmark --model claude-sonnet --categories BasicGates Superposition

# Filter by specific task IDs
uv run qk-benchmark --model claude-sonnet --task-ids "BasicGates/1.1" "BasicGates/1.2"

# Save results to custom directory
uv run qk-benchmark --model claude-sonnet --output results/experiment1
```

### Parallel Execution

Run multiple models simultaneously:

```bash
# Run all configured models in parallel
uv run qk-benchmark --all --parallel

# Limit parallel workers (useful for rate limits)
uv run qk-benchmark --all --parallel 2

# Parallel with specific models from config
uv run qk-benchmark --model claude-sonnet --model gpt-4o --parallel
```

### Debug & Quiet Modes

```bash
# Debug mode: show raw model responses for failed tasks
uv run qk-benchmark --model claude-sonnet --debug

# Quiet mode: suppress task-by-task output (useful for scripts)
uv run qk-benchmark --model claude-sonnet --quiet

# Combine: quiet overall but debug failures
uv run qk-benchmark --model claude-sonnet -q -d
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
- `o1` - o1 (reasoning model)
- `o1-mini` - o1 Mini
- `o3-mini` - o3 Mini (latest reasoning model)

### Google Gemini
- `gemini-2.5-pro` - Gemini 2.5 Pro (state-of-the-art reasoning)
- `gemini-2.5-flash` - Gemini 2.5 Flash (fast, balanced)

### Qiskit Code Assistant
- `mistral-qiskit` - Mistral Small 3.2 24B (Qiskit-tuned)

### Open Source (via vLLM)
You can also benchmark open-source models via vLLM:
- DeepSeek-V3.2 / DeepSeek-Coder
- Qwen2.5-Coder / Qwen3-Coder
- Code Llama
- Any model with OpenAI-compatible API

### LiteLLM Proxy
Access any model through a LiteLLM proxy for unified access:
- Any model supported by LiteLLM
- Useful for accessing multiple providers through a single endpoint

## JSON Configuration

Model configurations are defined in `models.json` at the project root. The CLI automatically loads this file if present.

### Config File Format

```json
{
  "my-model": {
    "provider": "anthropic",
    "model_id": "claude-sonnet-4-20250514",
    "max_tokens": 4096,
    "temperature": 0.0
  },
  "my-vllm-model": {
    "provider": "vllm",
    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "base_url": "http://localhost:8000/v1",
    "max_tokens": 4096,
    "temperature": 0.0
  }
}
```

### Using JSON Config

```python
from benchmark import load_models_from_json, BenchmarkRunner

# Load models from JSON file
models = load_models_from_json("models.json")

# Use a model from the config
runner = BenchmarkRunner(model_config=models["my-model"])
results = runner.run()
```

### CLI with Custom Config File

```bash
# Uses models.json by default if present
qk-benchmark --model claude-sonnet

# Or specify a custom config file
qk-benchmark --model my-model --config custom_models.json
```

### Environment Variables

API keys and URLs are configured via environment variables. Create a `.env` file from the template:

```bash
cp .env.example .env
# Edit .env with your credentials
```

Available variables:
- `ANTHROPIC_API_KEY` - Anthropic Claude
- `OPENAI_API_KEY` - OpenAI GPT
- `GOOGLE_API_KEY` - Google Gemini
- `LITELLM_API_KEY` - LiteLLM proxy API key
- `LITELLM_BASE_URL` - LiteLLM base URL (default: http://localhost:4000/v1)
- `VLLM_API_KEY` - vLLM API key (optional, defaults to "dummy")
- `VLLM_BASE_URL` - vLLM base URL (e.g., http://localhost:8000/v1)
- `QISKIT_ASSISTANT_TOKEN` - IBM Qiskit Code Assistant

The `.env` file is automatically loaded when running the benchmark.

## Statistical Analysis & Multiple Runs

For statistically rigorous evaluation, the benchmark supports multiple runs per task with various aggregation methods.

### Recommended Settings

| Use Case | `--num-runs` | `--aggregate` | Notes |
|----------|--------------|---------------|-------|
| Quick exploration | 1 | - | Fast, good for initial testing |
| Published results | 3-5 | `majority` | Recommended for papers |
| High confidence | 5-10 | `majority` | Tighter confidence intervals |
| pass@k metrics | 10+ | `any` | For computing pass@1, pass@5, etc. |

### Multiple Runs

```bash
# Run each task 3 times, pass if majority succeed
uv run qk-benchmark --model claude-sonnet --num-runs 3 --aggregate majority

# Run 5 times, pass if any attempt succeeds (for pass@k estimation)
uv run qk-benchmark --model claude-sonnet --num-runs 5 --aggregate any

# Run 3 times, pass only if all attempts succeed (strict)
uv run qk-benchmark --model claude-sonnet --num-runs 3 --aggregate all
```

**Aggregation methods:**
- `majority` (default): Pass if >50% of runs succeed. Best for robust estimates.
- `any`: Pass if at least one run succeeds. Use for pass@k metrics.
- `all`: Pass only if every run succeeds. Very strict evaluation.

### Prompting Strategies

The benchmark supports different prompting approaches:

```bash
# Zero-shot (default)
uv run qk-benchmark --model claude-sonnet --prompt-strategy zero_shot

# Few-shot with 1, 3, or 5 examples
uv run qk-benchmark --model claude-sonnet --prompt-strategy few_shot_3

# Chain-of-thought prompting
uv run qk-benchmark --model claude-sonnet --prompt-strategy chain_of_thought
```

### System Prompt Variants

```bash
# Default: Balanced instructions
uv run qk-benchmark --model claude-sonnet --system-prompt default

# Minimal: Brief instructions
uv run qk-benchmark --model claude-sonnet --system-prompt minimal

# Detailed: Comprehensive Qiskit guidance
uv run qk-benchmark --model claude-sonnet --system-prompt detailed
```

### Ablation Studies

Run all prompting strategy combinations automatically:

```bash
# Full ablation study (7 configurations)
uv run qk-benchmark --model claude-sonnet --ablation

# Ablation with multiple runs for statistical rigor
uv run qk-benchmark --model claude-sonnet --ablation --num-runs 3

# Filter to specific strategies
uv run qk-benchmark --model claude-sonnet --ablation --ablation-strategies zero_shot few_shot_3

# Multi-model ablation
uv run qk-benchmark --all --ablation
```

The ablation study runs these configurations:
1. Zero-shot + default prompt
2. Zero-shot + minimal prompt
3. Zero-shot + detailed prompt
4. 1-shot + default prompt
5. 3-shot + default prompt
6. 5-shot + default prompt
7. Chain-of-thought + CoT prompt

Results are saved with descriptive filenames and a summary JSON is generated.

### Confidence Intervals

Results include 95% Wilson score confidence intervals:

```python
from benchmark import load_results, generate_report

results = load_results("results/claude-sonnet.json")
report = generate_report(results)

print(f"Pass rate: {report.pass_rate:.1%}")
print(f"95% CI: [{report.stats.ci_lower:.1%}, {report.stats.ci_upper:.1%}]")
```

### Statistical Comparison

Compare models with statistical significance testing:

```python
from benchmark import load_all_results, generate_report, format_statistical_comparison

all_results = load_all_results("results")
reports = [generate_report(data) for _, data in all_results]

# Generate comparison with CIs and significance info
print(format_statistical_comparison(reports))
```

## Results

Results are saved in JSON format with:
- Pass/fail status for each task
- Generated code
- Evaluation details (syntax errors, test failures)
- Token usage and latency metrics

### Comparing Multiple Runs

After running benchmarks with different models, compare all results:

```bash
# List all available results
uv run qk-compare --list

# Generate comparison table
uv run qk-compare

# Save comparison to file
uv run qk-compare --output results/comparison.md

# Compare without category breakdown
uv run qk-compare --no-categories
```

Or use the Python API:

```python
from benchmark import load_all_results, compare_results_from_dir

# Quick comparison
print(compare_results_from_dir("results"))

# Load individual results for custom analysis
all_results = load_all_results("results")
for path, data in all_results:
    print(f"{data['model_id']}: {data['pass_rate']:.1%}")
```

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
