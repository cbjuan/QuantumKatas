"""Quantum Katas Benchmark - Evaluate LLMs on quantum computing tasks."""

from .config import (
    BenchmarkConfig,
    EvaluationConfig,
    ModelConfig,
    ProviderType,
    get_model_config,
    MODELS,
    load_models_from_json,
    save_models_to_json,
    load_config_file,
)
from .models import (
    Provider,
    AnthropicProvider,
    OpenAIProvider,
    GoogleProvider,
    LiteLLMProvider,
    VLLMProvider,
    QiskitAssistantProvider,
    GenerationResult,
    create_provider,
)
from .runner import (
    BenchmarkRunner,
    BenchmarkResults,
    TaskResult,
    Task,
)
from .evaluator import (
    EvaluationResult,
    evaluate_solution,
    extract_code_from_response,
)
from .reporter import (
    BenchmarkReport,
    CategoryStats,
    generate_report,
    format_markdown_report,
    print_summary,
    load_results,
    load_all_results,
    compare_reports,
    compare_results_from_dir,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "BenchmarkConfig",
    "EvaluationConfig",
    "ModelConfig",
    "ProviderType",
    "get_model_config",
    "MODELS",
    "load_models_from_json",
    "save_models_to_json",
    "load_config_file",
    # Models
    "Provider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "LiteLLMProvider",
    "VLLMProvider",
    "QiskitAssistantProvider",
    "GenerationResult",
    "create_provider",
    # Runner
    "BenchmarkRunner",
    "BenchmarkResults",
    "TaskResult",
    "Task",
    # Evaluator
    "EvaluationResult",
    "evaluate_solution",
    "extract_code_from_response",
    # Reporter
    "BenchmarkReport",
    "CategoryStats",
    "generate_report",
    "format_markdown_report",
    "print_summary",
    "load_results",
    "load_all_results",
    "compare_reports",
    "compare_results_from_dir",
]
