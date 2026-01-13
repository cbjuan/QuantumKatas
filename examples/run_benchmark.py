#!/usr/bin/env python3
"""Example: Run quantum katas benchmark with different providers."""

import os
import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import (
    BenchmarkRunner,
    ModelConfig,
    ProviderType,
    get_model_config,
    generate_report,
    format_markdown_report,
    print_summary,
)


def run_with_anthropic():
    """Run benchmark with Claude."""
    print("\n" + "=" * 60)
    print("Running benchmark with Claude Sonnet")
    print("=" * 60)

    config = get_model_config("claude-sonnet")
    runner = BenchmarkRunner(
        model_config=config,
        dataset_path=Path("dataset/quantum_katas.jsonl"),
    )

    # Run on a subset for demo
    runner.config.categories = ["BasicGates"]
    results = runner.run(verbose=True)

    # Save results
    output_path = Path("results/claude-sonnet-demo.json")
    results.save(output_path)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print_summary(results.to_dict())

    return results


def run_with_openai():
    """Run benchmark with GPT-4o."""
    print("\n" + "=" * 60)
    print("Running benchmark with GPT-4o")
    print("=" * 60)

    config = get_model_config("gpt-4o")
    runner = BenchmarkRunner(
        model_config=config,
        dataset_path=Path("dataset/quantum_katas.jsonl"),
    )

    # Run on a subset for demo
    runner.config.categories = ["BasicGates"]
    results = runner.run(verbose=True)

    output_path = Path("results/gpt-4o-demo.json")
    results.save(output_path)
    print(f"\nResults saved to: {output_path}")

    print_summary(results.to_dict())

    return results


def run_with_vllm(base_url: str = "http://localhost:8000/v1", model_id: str = "local-model"):
    """Run benchmark with vLLM deployment."""
    print("\n" + "=" * 60)
    print(f"Running benchmark with vLLM ({model_id})")
    print("=" * 60)

    config = ModelConfig(
        provider=ProviderType.VLLM,
        model_id=model_id,
        base_url=base_url,
    )
    runner = BenchmarkRunner(
        model_config=config,
        dataset_path=Path("dataset/quantum_katas.jsonl"),
    )

    # Run on a subset for demo
    runner.config.categories = ["BasicGates"]
    results = runner.run(verbose=True)

    output_path = Path(f"results/vllm-{model_id.replace('/', '-')}-demo.json")
    results.save(output_path)
    print(f"\nResults saved to: {output_path}")

    print_summary(results.to_dict())

    return results


def run_with_qiskit_assistant():
    """Run benchmark with Qiskit Code Assistant."""
    print("\n" + "=" * 60)
    print("Running benchmark with Qiskit Code Assistant")
    print("=" * 60)

    config = get_model_config("granite-8b-qiskit")
    runner = BenchmarkRunner(
        model_config=config,
        dataset_path=Path("dataset/quantum_katas.jsonl"),
    )

    # Run on a subset for demo
    runner.config.categories = ["BasicGates"]
    results = runner.run(verbose=True)

    output_path = Path("results/granite-8b-demo.json")
    results.save(output_path)
    print(f"\nResults saved to: {output_path}")

    print_summary(results.to_dict())

    return results


def run_single_task(model_name: str, task_id: str):
    """Run a single task for debugging."""
    print(f"\n Running single task: {task_id} with {model_name}")

    config = get_model_config(model_name)
    runner = BenchmarkRunner(
        model_config=config,
        dataset_path=Path("dataset/quantum_katas.jsonl"),
    )

    result = runner.run_single(task_id)

    print(f"\nTask: {result.task_id}")
    print(f"Passed: {result.evaluation.passed}")
    if result.evaluation.error_message:
        print(f"Error: {result.evaluation.error_message}")
    print(f"\nGenerated code:\n{result.generation.content[:500]}...")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run quantum katas benchmark examples")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "vllm", "qiskit"],
        default="anthropic",
        help="Provider to use",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM base URL")
    parser.add_argument("--vllm-model", default="local-model", help="vLLM model ID")
    parser.add_argument("--task", help="Run single task by ID (e.g., BasicGates/1)")

    args = parser.parse_args()

    # Check for API keys
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    elif args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    elif args.provider == "qiskit" and not os.environ.get("QISKIT_ASSISTANT_TOKEN"):
        print("Error: QISKIT_ASSISTANT_TOKEN environment variable not set")
        sys.exit(1)

    if args.task:
        model_map = {
            "anthropic": "claude-sonnet",
            "openai": "gpt-4o",
            "qiskit": "granite-8b-qiskit",
        }
        if args.provider == "vllm":
            print("Single task mode not supported with vLLM in this example")
            sys.exit(1)
        run_single_task(model_map[args.provider], args.task)
    elif args.provider == "anthropic":
        run_with_anthropic()
    elif args.provider == "openai":
        run_with_openai()
    elif args.provider == "vllm":
        run_with_vllm(args.vllm_url, args.vllm_model)
    elif args.provider == "qiskit":
        run_with_qiskit_assistant()
