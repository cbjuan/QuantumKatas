"""Configuration management for quantum katas benchmark."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import os


class ProviderType(Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    VLLM = "vllm"
    QISKIT_ASSISTANT = "qiskit_assistant"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: ProviderType
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.0

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            env_var_map = {
                ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
                ProviderType.OPENAI: "OPENAI_API_KEY",
                ProviderType.VLLM: "VLLM_API_KEY",
                ProviderType.QISKIT_ASSISTANT: "QISKIT_ASSISTANT_TOKEN",
            }
            env_var = env_var_map.get(self.provider)
            if env_var:
                self.api_key = os.environ.get(env_var)


@dataclass
class EvaluationConfig:
    """Configuration for code evaluation."""

    check_syntax: bool = True
    run_tests: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 3


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""

    model: ModelConfig
    dataset_path: Path = field(default_factory=lambda: Path("dataset/quantum_katas.jsonl"))
    output_dir: Path = field(default_factory=lambda: Path("results"))
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint_interval: int = 10
    task_ids: Optional[list[str]] = None
    categories: Optional[list[str]] = None
    verbose: bool = False

    def __post_init__(self):
        """Convert paths if needed."""
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


# Predefined model configurations
MODELS = {
    # Anthropic models
    "claude-opus": ModelConfig(
        provider=ProviderType.ANTHROPIC,
        model_id="claude-opus-4-20250514",
    ),
    "claude-sonnet": ModelConfig(
        provider=ProviderType.ANTHROPIC,
        model_id="claude-sonnet-4-20250514",
    ),
    "claude-haiku": ModelConfig(
        provider=ProviderType.ANTHROPIC,
        model_id="claude-haiku-4-20250514",
    ),
    # OpenAI models
    "gpt-4o": ModelConfig(
        provider=ProviderType.OPENAI,
        model_id="gpt-4o",
    ),
    "gpt-4o-mini": ModelConfig(
        provider=ProviderType.OPENAI,
        model_id="gpt-4o-mini",
    ),
    "o1": ModelConfig(
        provider=ProviderType.OPENAI,
        model_id="o1",
    ),
    "o1-mini": ModelConfig(
        provider=ProviderType.OPENAI,
        model_id="o1-mini",
    ),
    # Qiskit Code Assistant
    "mistral-qiskit": ModelConfig(
        provider=ProviderType.QISKIT_ASSISTANT,
        model_id="mistral-small-3.2-24b-qiskit",
        base_url="https://qiskit-code-assistant.quantum.ibm.com",
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get a predefined model configuration by name."""
    if model_name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODELS[model_name]
