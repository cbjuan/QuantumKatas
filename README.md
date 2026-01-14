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

## Quick Start

```bash
# Run benchmark with a model
uv run qk-benchmark --model claude-sonnet

# List available models
uv run qk-benchmark --list-models
```

## Configuration

### Environment Variables

API keys are configured via environment variables. Create a `.env` file:

```bash
cp .env.example .env
# Edit .env with your credentials
```

| Variable | Provider |
|----------|----------|
| `ANTHROPIC_API_KEY` | Anthropic Claude |
| `OPENAI_API_KEY` | OpenAI GPT |
| `GOOGLE_API_KEY` | Google Gemini |
| `VLLM_API_KEY` | vLLM (optional, defaults to "dummy") |
| `LITELLM_API_KEY` | LiteLLM proxy |
| `QISKIT_ASSISTANT_TOKEN` | IBM Qiskit Code Assistant |

The `.env` file is automatically loaded when running the benchmark.

### JSON Configuration

Models can be configured in `models.json` at the project root. The CLI loads this file automatically.

```json
{
  "my-claude": {
    "provider": "anthropic",
    "model_id": "claude-sonnet-4-20250514",
    "max_tokens": 4096,
    "temperature": 0.0
  },
  "my-vllm": {
    "provider": "vllm",
    "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "base_url": "http://localhost:8000/v1",
    "max_tokens": 4096,
    "temperature": 0.0
  }
}
```

Supported providers: `anthropic`, `openai`, `google`, `vllm`, `litellm`, `qiskit_assistant`

### Custom Headers

Some endpoints require custom HTTP headers for authentication. Header values can reference environment variables:

```json
{
  "my-model": {
    "provider": "vllm",
    "model_id": "ibm-granite/granite-4.0-h-small",
    "base_url": "https://inference.example.com/granite-4-h-small/v1",
    "max_tokens": 4096,
    "temperature": 0.0,
    "headers": {
      "API_KEY": "VLLM_API_KEY"
    }
  }
}
```

The `VLLM_API_KEY` environment variable is resolved at runtime and sent as the `API_KEY` header.

**vLLM Example:**

```json
{
  "granite-4.0-h-small": {
    "provider": "vllm",
    "model_id": "ibm-granite/granite-4.0-h-small",
    "base_url": "https://<vllm-url>/granite-4-h-small/v1",
    "max_tokens": 4096,
    "temperature": 0.0,
    "headers": {
      "API_KEY": "VLLM_API_KEY"
    }
  }
}
```

**Note:** Base URLs for OpenAI-compatible endpoints must end with `/v1`.

## Available Models

### Built-in Models

| Provider | Model Key | Description |
|----------|-----------|-------------|
| Anthropic | `claude-opus` | Claude Opus 4 |
| Anthropic | `claude-sonnet` | Claude Sonnet 4 |
| Anthropic | `claude-haiku` | Claude Haiku 4 |
| OpenAI | `gpt-4o` | GPT-4o |
| OpenAI | `gpt-4o-mini` | GPT-4o Mini |
| OpenAI | `o1`, `o1-mini`, `o3-mini` | Reasoning models |
| Google | `gemini-2.5-pro` | Gemini 2.5 Pro |
| Google | `gemini-2.5-flash` | Gemini 2.5 Flash |
| IBM | `mistral-qiskit` | Mistral Small 3.2 24B (Qiskit-tuned) |

### Open Source via vLLM

Any model with an OpenAI-compatible API:
- DeepSeek-V3 / DeepSeek-Coder
- Qwen2.5-Coder / Qwen3-Coder
- Llama 4, Code Llama
- Granite, Mistral, Phi-4

### LiteLLM Proxy

Access 100+ models through a unified LiteLLM proxy endpoint.

## Usage

### Command Line

```bash
# Basic usage
uv run qk-benchmark --model claude-sonnet

# Filter by category
uv run qk-benchmark --model claude-sonnet --categories BasicGates Superposition

# Filter by task IDs
uv run qk-benchmark --model claude-sonnet --task-ids "BasicGates/1.1" "BasicGates/1.2"

# Custom output directory
uv run qk-benchmark --model claude-sonnet --output results/experiment1

# Use custom config file
uv run qk-benchmark --model my-model --config custom_models.json
```

### Debug & Quiet Modes

```bash
# Debug: show raw responses for failed tasks
uv run qk-benchmark --model claude-sonnet --debug

# Quiet: suppress task-by-task output
uv run qk-benchmark --model claude-sonnet --quiet

# Combine both
uv run qk-benchmark --model claude-sonnet -q -d
```

### Parallel Execution

```bash
# Run all models in parallel
uv run qk-benchmark --all --parallel

# Limit workers (useful for rate limits)
uv run qk-benchmark --all --parallel 2

# Parallel with ablation study (each model runs all 7 configs in its own worker)
uv run qk-benchmark --all --parallel --ablation

# Parallel ablation with worker limit and multiple runs
uv run qk-benchmark --all --parallel 4 --ablation --num-runs 3
```

### Python API

```python
from benchmark import BenchmarkRunner, ModelConfig, ProviderType, get_model_config

# Using a built-in model
config = get_model_config("claude-sonnet")
runner = BenchmarkRunner(model_config=config)
results = runner.run(verbose=True)
results.save("results/claude-sonnet.json")

# Custom configuration
config = ModelConfig(
    provider=ProviderType.OPENAI,
    model_id="gpt-4o",
    temperature=0.0,
)
runner = BenchmarkRunner(model_config=config)
results = runner.run()

# vLLM with custom headers
config = ModelConfig(
    provider=ProviderType.VLLM,
    model_id="ibm-granite/granite-4.0-h-small",
    base_url="https://inference.example.com/granite-4-h-small/v1",
    headers={"API_KEY": "VLLM_API_KEY"},
)

# Load from JSON config
from benchmark import load_models_from_json
models = load_models_from_json("models.json")
runner = BenchmarkRunner(model_config=models["my-model"])
```

## Statistical Analysis

### Multiple Runs

For statistically rigorous evaluation, run each task multiple times:

```bash
# Majority voting (recommended)
uv run qk-benchmark --model claude-sonnet --num-runs 3 --aggregate majority

# Any pass (for pass@k metrics)
uv run qk-benchmark --model claude-sonnet --num-runs 5 --aggregate any

# All must pass (strict)
uv run qk-benchmark --model claude-sonnet --num-runs 3 --aggregate all
```

| Aggregation | Behavior | Use Case |
|-------------|----------|----------|
| `majority` | Pass if >50% succeed | Robust estimates |
| `any` | Pass if any succeeds | pass@k metrics |
| `all` | Pass only if all succeed | Strict evaluation |

### Recommended Settings

| Use Case | `--num-runs` | `--aggregate` |
|----------|--------------|---------------|
| Quick exploration | 1 | - |
| Published results | 3-5 | `majority` |
| pass@k metrics | 10+ | `any` |

### Confidence Intervals

Results include 95% Wilson score confidence intervals:

```python
from benchmark import load_results, generate_report

results = load_results("results/claude-sonnet.json")
report = generate_report(results)

print(f"Pass rate: {report.pass_rate:.1%}")
print(f"95% CI: [{report.stats.ci_lower:.1%}, {report.stats.ci_upper:.1%}]")
```

## Prompting Strategies

### Strategy Options

```bash
# Zero-shot (default)
uv run qk-benchmark --model claude-sonnet --prompt-strategy zero_shot

# Few-shot with examples
uv run qk-benchmark --model claude-sonnet --prompt-strategy few_shot_3

# Chain-of-thought
uv run qk-benchmark --model claude-sonnet --prompt-strategy chain_of_thought
```

### System Prompts

```bash
# Default: balanced instructions
uv run qk-benchmark --model claude-sonnet --system-prompt default

# Minimal: brief instructions
uv run qk-benchmark --model claude-sonnet --system-prompt minimal

# Detailed: comprehensive Qiskit guidance
uv run qk-benchmark --model claude-sonnet --system-prompt detailed
```

### Ablation Studies

Run all prompting combinations automatically:

```bash
# Full ablation (7 configurations)
uv run qk-benchmark --model claude-sonnet --ablation

# With multiple runs
uv run qk-benchmark --model claude-sonnet --ablation --num-runs 3

# Specific strategies only
uv run qk-benchmark --model claude-sonnet --ablation --ablation-strategies zero_shot few_shot_3

# Parallel ablation across all models
uv run qk-benchmark --all --parallel --ablation

# Parallel ablation with worker limit
uv run qk-benchmark --all --parallel 4 --ablation --num-runs 3
```

Configurations tested:
1. Zero-shot + default/minimal/detailed prompts
2. Few-shot (1, 3, 5 examples) + default prompt
3. Chain-of-thought + CoT prompt

When using `--parallel` with `--ablation`, each model runs its full ablation study in a separate worker process.

## Results & Reporting

Results are saved as JSON with pass/fail status, generated code, evaluation details, and metrics.

### Compare Models

```bash
# List available results
uv run qk-compare --list

# Generate comparison table
uv run qk-compare

# Save to file
uv run qk-compare --output results/comparison.md
```

### Generate Reports

```python
from benchmark import (
    load_results,
    load_all_results,
    generate_report,
    format_markdown_report,
    format_statistical_comparison,
)

# Single model report
results = load_results("results/claude-sonnet.json")
report = generate_report(results)
print(format_markdown_report(report))

# Compare all models
all_results = load_all_results("results")
reports = [generate_report(data) for _, data in all_results]
print(format_statistical_comparison(reports))
```

## Dataset Format

Each task in `dataset/quantum_katas.jsonl`:

```json
{
  "task_id": "BasicGates/1.1",
  "prompt": "# Task: State flip\n# Input: A qubit in state |ψ⟩ = α|0⟩ + β|1⟩\n# Goal: Change the state to α|1⟩ + β|0⟩...",
  "canonical_solution": "def state_flip(qc, q):\n    qc.x(q)\n    return qc",
  "test": "def test_state_flip():\n    qc = QuantumCircuit(1)\n    ...",
  "entry_point": "state_flip"
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

Based on Microsoft's [QuantumKatas](https://github.com/microsoft/QuantumKatas) (MIT License).
