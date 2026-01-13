"""Benchmark runner for quantum katas dataset."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

from .config import BenchmarkConfig, ModelConfig, get_model_config
from .models import Provider, create_provider, GenerationResult
from .evaluator import evaluate_solution, EvaluationResult


SYSTEM_PROMPT = """You are an expert quantum computing programmer specializing in Qiskit.
Your task is to implement quantum computing functions using Qiskit.
Provide ONLY the Python code implementation, no explanations.
The code should be complete and ready to execute."""


@dataclass
class TaskResult:
    """Result for a single benchmark task."""

    task_id: str
    category: str
    entry_point: str
    generation: GenerationResult
    evaluation: EvaluationResult
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "entry_point": self.entry_point,
            "generation": asdict(self.generation),
            "evaluation": asdict(self.evaluation),
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkResults:
    """Collection of benchmark results."""

    model_id: str
    provider: str
    total_tasks: int
    completed_tasks: int
    passed_tasks: int
    failed_tasks: int
    error_tasks: int
    results: list[TaskResult]
    start_time: str
    end_time: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "passed_tasks": self.passed_tasks,
            "failed_tasks": self.failed_tasks,
            "error_tasks": self.error_tasks,
            "pass_rate": self.passed_tasks / self.completed_tasks if self.completed_tasks > 0 else 0,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Path):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @property
    def pass_rate(self) -> float:
        if self.completed_tasks == 0:
            return 0.0
        return self.passed_tasks / self.completed_tasks


@dataclass
class Task:
    """A benchmark task from the dataset."""

    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str

    @property
    def category(self) -> str:
        return self.task_id.split("/")[0]


class BenchmarkRunner:
    """Runs benchmarks against LLM providers."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        provider: Optional[Provider] = None,
        model_config: Optional[ModelConfig] = None,
        dataset_path: Optional[Path] = None,
    ):
        if provider:
            self.provider = provider
            self.model_config = provider.config
        elif model_config:
            self.model_config = model_config
            self.provider = create_provider(model_config)
        elif config:
            self.model_config = config.model
            self.provider = create_provider(config.model)
        else:
            raise ValueError("Must provide config, provider, or model_config")

        self.config = config or BenchmarkConfig(model=self.model_config)
        if dataset_path:
            self.config.dataset_path = Path(dataset_path)

        self.tasks: list[Task] = []
        self._load_dataset()

    def _load_dataset(self):
        """Load tasks from JSONL dataset."""
        with open(self.config.dataset_path) as f:
            for line in f:
                data = json.loads(line)
                task = Task(
                    task_id=data["task_id"],
                    prompt=data["prompt"],
                    canonical_solution=data["canonical_solution"],
                    test=data["test"],
                    entry_point=data["entry_point"],
                )
                # Filter by category if specified
                if self.config.categories:
                    if task.category not in self.config.categories:
                        continue
                # Filter by task_id if specified
                if self.config.task_ids:
                    if task.task_id not in self.config.task_ids:
                        continue
                self.tasks.append(task)

    def run(self, verbose: Optional[bool] = None) -> BenchmarkResults:
        """Run the benchmark on all tasks."""
        verbose = verbose if verbose is not None else self.config.verbose
        results = BenchmarkResults(
            model_id=self.model_config.model_id,
            provider=self.model_config.provider.value,
            total_tasks=len(self.tasks),
            completed_tasks=0,
            passed_tasks=0,
            failed_tasks=0,
            error_tasks=0,
            results=[],
            start_time=datetime.now().isoformat(),
        )

        checkpoint_path = self.config.output_dir / f"checkpoint_{self.model_config.model_id}.json"

        for i, task in enumerate(self.tasks):
            if verbose:
                print(f"[{i+1}/{len(self.tasks)}] {task.task_id}...", end=" ", flush=True)

            # Generate solution
            generation = self.provider.generate(task.prompt, SYSTEM_PROMPT)

            # Evaluate solution
            if generation.error:
                evaluation = EvaluationResult(
                    passed=False,
                    syntax_valid=False,
                    test_passed=False,
                    error_message=f"Generation error: {generation.error}",
                )
            else:
                evaluation = evaluate_solution(
                    code=generation.content,
                    test_code=task.test,
                    entry_point=task.entry_point,
                    timeout=self.config.evaluation.timeout_seconds,
                )

            task_result = TaskResult(
                task_id=task.task_id,
                category=task.category,
                entry_point=task.entry_point,
                generation=generation,
                evaluation=evaluation,
            )
            results.results.append(task_result)
            results.completed_tasks += 1

            if evaluation.passed:
                results.passed_tasks += 1
                if verbose:
                    print("PASS")
            elif generation.error or evaluation.error_message:
                results.error_tasks += 1
                if verbose:
                    print(f"ERROR: {evaluation.error_message[:50]}...")
            else:
                results.failed_tasks += 1
                if verbose:
                    print("FAIL")

            # Checkpoint
            if (i + 1) % self.config.checkpoint_interval == 0:
                results.save(checkpoint_path)

        results.end_time = datetime.now().isoformat()
        return results

    def run_single(self, task_id: str, verbose: bool = True) -> TaskResult:
        """Run benchmark on a single task."""
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        if verbose:
            print(f"Running {task_id}...")

        generation = self.provider.generate(task.prompt, SYSTEM_PROMPT)
        evaluation = evaluate_solution(
            code=generation.content,
            test_code=task.test,
            entry_point=task.entry_point,
            timeout=self.config.evaluation.timeout_seconds,
        )

        return TaskResult(
            task_id=task.task_id,
            category=task.category,
            entry_point=task.entry_point,
            generation=generation,
            evaluation=evaluation,
        )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run quantum katas benchmark")
    parser.add_argument("--model", required=True, help="Model name (e.g., claude-sonnet, gpt-4o)")
    parser.add_argument("--dataset", default="dataset/quantum_katas.jsonl", help="Dataset path")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--categories", nargs="+", help="Filter by categories")
    parser.add_argument("--task-ids", nargs="+", help="Filter by task IDs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        model_config = get_model_config(args.model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    config = BenchmarkConfig(
        model=model_config,
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output),
        categories=args.categories,
        task_ids=args.task_ids,
        verbose=args.verbose,
    )

    runner = BenchmarkRunner(config=config)
    print(f"Running benchmark with {model_config.model_id}...")
    print(f"Tasks: {len(runner.tasks)}")

    results = runner.run()

    output_file = config.output_dir / f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results.save(output_file)

    print(f"\nResults saved to: {output_file}")
    print(f"Pass rate: {results.pass_rate:.1%} ({results.passed_tasks}/{results.completed_tasks})")


if __name__ == "__main__":
    main()
