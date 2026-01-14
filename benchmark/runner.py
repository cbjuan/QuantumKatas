"""Benchmark runner for quantum katas dataset."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

from .config import (
    BenchmarkConfig,
    ModelConfig,
    get_model_config,
    load_models_from_json,
    MODELS,
    PromptStrategy,
    PromptConfig,
    RunConfig,
    SYSTEM_PROMPTS,
)
from .models import Provider, create_provider, GenerationResult
from .evaluator import evaluate_solution, EvaluationResult


def load_few_shot_examples(dataset_path: Path, n: int, exclude_ids: list[str] = None) -> list[dict]:
    """Load n few-shot examples from the dataset.

    Args:
        dataset_path: Path to the JSONL dataset
        n: Number of examples to load
        exclude_ids: Task IDs to exclude (e.g., the current task being evaluated)

    Returns:
        List of {prompt, solution} dicts
    """
    exclude_ids = exclude_ids or []
    examples = []

    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            if data["task_id"] in exclude_ids:
                continue
            # Prefer simpler tasks for examples
            if data["task_id"].startswith(("BasicGates/", "Superposition/")):
                examples.append({
                    "prompt": data["prompt"],
                    "solution": data["canonical_solution"],
                })
            if len(examples) >= n:
                break

    return examples


def format_few_shot_prompt(task_prompt: str, examples: list[dict]) -> str:
    """Format a prompt with few-shot examples."""
    parts = ["Here are some examples of quantum computing implementations:\n"]

    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Task: {ex['prompt'][:200]}...")
        parts.append(f"Solution:\n```python\n{ex['solution']}\n```\n")

    parts.append("Now implement the following:\n")
    parts.append(task_prompt)

    return "\n".join(parts)


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

    def _prepare_prompt(self, task: Task) -> tuple[str, str]:
        """Prepare the prompt and system prompt for a task.

        Returns:
            (task_prompt, system_prompt)
        """
        system_prompt = self.config.prompt.get_system_prompt()
        task_prompt = task.prompt

        # Handle few-shot prompting
        strategy = self.config.prompt.strategy
        if strategy in (PromptStrategy.FEW_SHOT_1, PromptStrategy.FEW_SHOT_3, PromptStrategy.FEW_SHOT_5):
            n_examples = int(strategy.value.split("_")[-1])
            examples = self.config.prompt.few_shot_examples
            if not examples:
                # Load examples from dataset
                examples = load_few_shot_examples(
                    self.config.dataset_path,
                    n=n_examples,
                    exclude_ids=[task.task_id],
                )
            task_prompt = format_few_shot_prompt(task.prompt, examples[:n_examples])

        return task_prompt, system_prompt

    def _run_single_task(self, task: Task, run_idx: int = 0) -> tuple[GenerationResult, EvaluationResult]:
        """Run a single task and return generation and evaluation results."""
        task_prompt, system_prompt = self._prepare_prompt(task)

        # Adjust temperature for multiple runs
        original_temp = self.model_config.temperature
        if self.config.runs.num_runs > 1 and run_idx > 0:
            # Use slightly different temperatures for variance
            temps = self.config.runs.temperatures
            if len(temps) > 1:
                self.model_config.temperature = temps[run_idx % len(temps)]
            else:
                # Add small variance for multiple runs
                self.model_config.temperature = min(0.3, 0.1 * run_idx)

        generation = self.provider.generate(task_prompt, system_prompt)

        # Restore temperature
        self.model_config.temperature = original_temp

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

        return generation, evaluation

    def run(
        self,
        verbose: Optional[bool] = None,
        progress_callback: Optional[callable] = None,
        model_name: Optional[str] = None,
        debug: bool = False,
    ) -> BenchmarkResults:
        """Run the benchmark on all tasks.

        Args:
            verbose: Print progress to stdout
            progress_callback: Called after each task with (model_name, completed, total, passed, status)
            model_name: Model name for progress reporting (defaults to model_id)
            debug: Print raw model responses for failed tasks
        """
        verbose = verbose if verbose is not None else self.config.verbose
        model_name = model_name or self.model_config.model_id
        num_runs = self.config.runs.num_runs

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

        # Store run metadata
        results_metadata = {
            "num_runs": num_runs,
            "prompt_strategy": self.config.prompt.strategy.value,
            "system_prompt": self.config.prompt.system_prompt,
        }

        checkpoint_path = self.config.output_dir / f"checkpoint_{self.model_config.model_id}.json"

        for i, task in enumerate(self.tasks):
            if verbose:
                print(f"[{i+1}/{len(self.tasks)}] {task.task_id}...", end=" ", flush=True)

            # Run task (potentially multiple times)
            all_runs = []
            for run_idx in range(num_runs):
                generation, evaluation = self._run_single_task(task, run_idx)
                all_runs.append((generation, evaluation))

            # Aggregate results from multiple runs
            if num_runs > 1:
                passed_runs = sum(1 for _, ev in all_runs if ev.passed)
                aggregate = self.config.runs.aggregate_method

                if aggregate == "majority":
                    final_passed = passed_runs > num_runs // 2
                elif aggregate == "any":
                    final_passed = passed_runs > 0
                elif aggregate == "all":
                    final_passed = passed_runs == num_runs
                else:
                    final_passed = all_runs[0][1].passed

                # Use the first passing run, or the first run if none passed
                best_idx = next((i for i, (_, ev) in enumerate(all_runs) if ev.passed), 0)
                generation, evaluation = all_runs[best_idx]

                # Override passed status based on aggregation
                if final_passed != evaluation.passed:
                    evaluation = EvaluationResult(
                        passed=final_passed,
                        syntax_valid=evaluation.syntax_valid,
                        test_passed=final_passed,
                        error_message=evaluation.error_message,
                        error_type=evaluation.error_type,
                        stdout=evaluation.stdout,
                    )
            else:
                generation, evaluation = all_runs[0]

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
                status = "PASS"
                if verbose:
                    print("PASS")
            elif generation.error or evaluation.error_message:
                results.error_tasks += 1
                status = "ERROR"
                if verbose:
                    error_msg = generation.error or evaluation.error_message or "Unknown error"
                    print(f"ERROR: {error_msg[:50]}...")
                if debug:
                    print(f"\n--- DEBUG: {task.task_id} ---")
                    print(f"Error: {generation.error or evaluation.error_message}")
                    print(f"Response ({len(generation.content)} chars):")
                    print(generation.content[:1000] if generation.content else "(empty)")
                    if len(generation.content) > 1000:
                        print(f"... ({len(generation.content) - 1000} more chars)")
                    print("--- END DEBUG ---\n")
            else:
                results.failed_tasks += 1
                status = "FAIL"
                if verbose:
                    print("FAIL")
                if debug:
                    print(f"\n--- DEBUG: {task.task_id} ---")
                    print(f"Response ({len(generation.content)} chars):")
                    print(generation.content[:1000] if generation.content else "(empty)")
                    if len(generation.content) > 1000:
                        print(f"... ({len(generation.content) - 1000} more chars)")
                    print("--- END DEBUG ---\n")

            # Progress callback for parallel monitoring
            if progress_callback:
                progress_callback(
                    model_name,
                    results.completed_tasks,
                    len(self.tasks),
                    results.passed_tasks,
                    status,
                )

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

        generation, evaluation = self._run_single_task(task)

        return TaskResult(
            task_id=task.task_id,
            category=task.category,
            entry_point=task.entry_point,
            generation=generation,
            evaluation=evaluation,
        )


def run_single_model_benchmark(
    model_name: str,
    model_config: ModelConfig,
    dataset_path: Path,
    output_dir: Path,
    categories: Optional[list[str]],
    task_ids: Optional[list[str]],
    verbose: bool,
    progress_queue=None,
    prompt_config: Optional[PromptConfig] = None,
    run_config: Optional[RunConfig] = None,
) -> tuple[str, Path, float]:
    """Run benchmark for a single model. Used for parallel execution."""
    from .config import PromptConfig, RunConfig

    # When using progress queue, don't print verbose output (use queue instead)
    use_verbose = verbose and not progress_queue

    config = BenchmarkConfig(
        model=model_config,
        dataset_path=dataset_path,
        output_dir=output_dir,
        categories=categories,
        task_ids=task_ids,
        verbose=use_verbose,
        prompt=prompt_config or PromptConfig(),
        runs=run_config or RunConfig(),
    )

    runner = BenchmarkRunner(config=config)

    # Send initial progress
    if progress_queue:
        progress_queue.put((model_name, 0, len(runner.tasks), 0, "STARTED"))

    # Create progress callback that sends to queue
    def progress_callback(name, completed, total, passed, status):
        if progress_queue:
            progress_queue.put((name, completed, total, passed, status))

    results = runner.run(
        verbose=use_verbose,
        progress_callback=progress_callback if progress_queue else None,
        model_name=model_name,
    )

    output_file = output_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results.save(output_file)

    # Send completion
    if progress_queue:
        progress_queue.put((model_name, results.completed_tasks, len(runner.tasks), results.passed_tasks, "DONE"))

    return model_name, output_file, results.pass_rate


# Ablation study configurations
ABLATION_CONFIGS = [
    # (strategy, system_prompt, description)
    ("zero_shot", "default", "Zero-shot with default prompt"),
    ("zero_shot", "minimal", "Zero-shot with minimal prompt"),
    ("zero_shot", "detailed", "Zero-shot with detailed prompt"),
    ("few_shot_1", "default", "1-shot with default prompt"),
    ("few_shot_3", "default", "3-shot with default prompt"),
    ("few_shot_5", "default", "5-shot with default prompt"),
    ("chain_of_thought", "chain_of_thought", "Chain-of-thought prompting"),
]


def run_single_model_ablation(
    model_name: str,
    model_config: ModelConfig,
    configs: list[tuple[str, str, str]],
    dataset_path: Path,
    output_dir: Path,
    categories: Optional[list[str]],
    task_ids: Optional[list[str]],
    verbose: bool,
    num_runs: int,
    aggregate_method: str,
    progress_queue=None,
) -> dict:
    """Run ablation study for a single model. Used for parallel execution.

    Args:
        model_name: Name of the model
        model_config: Model configuration
        configs: List of (strategy, system_prompt, description) tuples
        dataset_path: Path to dataset
        output_dir: Output directory for results
        categories: Optional category filter
        task_ids: Optional task ID filter
        verbose: Whether to print progress
        num_runs: Number of runs per task
        aggregate_method: How to aggregate multiple runs
        progress_queue: Optional queue for progress updates

    Returns:
        Dictionary with model results for all configurations
    """
    run_config = RunConfig(
        num_runs=num_runs,
        aggregate_method=aggregate_method,
    )

    model_results = []

    if progress_queue:
        progress_queue.put((model_name, "STARTED", 0, len(configs), None))

    for config_idx, (strategy, system_prompt, desc) in enumerate(configs):
        if progress_queue:
            progress_queue.put((model_name, "CONFIG", config_idx, len(configs), desc))

        prompt_config = PromptConfig(
            strategy=PromptStrategy(strategy),
            system_prompt=system_prompt,
        )

        config = BenchmarkConfig(
            model=model_config,
            dataset_path=dataset_path,
            output_dir=output_dir,
            categories=categories,
            task_ids=task_ids,
            verbose=verbose and not progress_queue,
            prompt=prompt_config,
            runs=run_config,
        )

        runner = BenchmarkRunner(config=config)
        results = runner.run()

        # Save with descriptive filename
        safe_strategy = strategy.replace("_", "-")
        output_file = output_dir / f"{model_name}_{safe_strategy}_{system_prompt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results.save(output_file)

        model_results.append({
            "strategy": strategy,
            "system_prompt": system_prompt,
            "description": desc,
            "pass_rate": results.pass_rate,
            "passed": results.passed_tasks,
            "total": results.completed_tasks,
            "output_file": str(output_file),
        })

        if progress_queue:
            progress_queue.put((
                model_name, "RESULT", config_idx + 1, len(configs),
                f"{desc}: {results.pass_rate:.1%}"
            ))

    if progress_queue:
        progress_queue.put((model_name, "DONE", len(configs), len(configs), None))

    return {
        "model": model_name,
        "model_id": model_config.model_id,
        "results": model_results,
    }


def run_ablation_study(
    models_to_run: list[str],
    models: dict[str, ModelConfig],
    args,
    output_dir: Path,
):
    """Run ablation study across all prompting configurations.

    Runs each model with each prompting strategy combination and produces
    a summary comparison at the end. Supports parallel execution with --parallel.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Filter configs if specific strategies requested
    if args.ablation_strategies:
        configs = [
            (s, p, d) for s, p, d in ABLATION_CONFIGS
            if s in args.ablation_strategies
        ]
    else:
        configs = ABLATION_CONFIGS

    total_runs = len(models_to_run) * len(configs)
    parallel = args.parallel is not None and len(models_to_run) > 1

    print(f"\n{'='*70}")
    print("ABLATION STUDY" + (" (PARALLEL)" if parallel else ""))
    print(f"{'='*70}")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Configurations: {len(configs)}")
    print(f"Total benchmark runs: {total_runs}")
    if parallel:
        max_workers = args.parallel if args.parallel > 0 else len(models_to_run)
        print(f"Parallel workers: {max_workers}")
    print(f"{'='*70}\n")

    # Print configurations to run
    print("Configurations:")
    for i, (strategy, system_prompt, desc) in enumerate(configs, 1):
        print(f"  {i}. {desc} ({strategy} + {system_prompt})")
    print()

    all_results = []

    if parallel:
        # Parallel execution across models
        from multiprocessing import Manager
        import threading

        max_workers = args.parallel if args.parallel > 0 else len(models_to_run)

        manager = Manager()
        progress_queue = manager.Queue()
        stop_progress = threading.Event()

        # Track progress per model
        model_progress = {name: {"current": 0, "total": len(configs), "status": "waiting"} for name in models_to_run}

        def progress_monitor(queue):
            """Monitor progress queue and update display."""
            while not stop_progress.is_set():
                try:
                    msg = queue.get(timeout=0.5)
                    if msg is None:
                        break
                    model_name, status, current, total, info = msg

                    if status == "STARTED":
                        print(f"  [{model_name}] Started ablation study ({total} configs)", flush=True)
                    elif status == "CONFIG":
                        print(f"  [{model_name}] Running config {current + 1}/{total}: {info}", flush=True)
                    elif status == "RESULT":
                        print(f"  [{model_name}] Completed {current}/{total} - {info}", flush=True)
                    elif status == "DONE":
                        print(f"  [{model_name}] Completed all configurations", flush=True)
                except Exception:
                    pass

        # Start progress monitor thread
        monitor_thread = threading.Thread(target=progress_monitor, args=(progress_queue,), daemon=True)
        monitor_thread.start()

        print(f"Starting parallel ablation with {max_workers} workers...\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_model_ablation,
                    model_name,
                    models[model_name],
                    configs,
                    Path(args.dataset),
                    output_dir,
                    args.categories,
                    args.task_ids,
                    not args.quiet,
                    args.num_runs,
                    args.aggregate,
                    progress_queue,
                ): model_name
                for model_name in models_to_run
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    best_rate = max(r["pass_rate"] for r in result["results"])
                    print(f"\n[{model_name}] COMPLETED - Best: {best_rate:.1%}")
                except Exception as e:
                    print(f"\n[{model_name}] FAILED: {e}")

        # Stop progress monitor
        stop_progress.set()
        progress_queue.put(None)
        monitor_thread.join(timeout=1.0)

    else:
        # Sequential execution
        run_count = 0
        for model_name in models_to_run:
            model_config = models[model_name]
            model_results = []

            print(f"\n{'='*60}")
            print(f"Model: {model_name} ({model_config.model_id})")
            print(f"{'='*60}")

            for strategy, system_prompt, desc in configs:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] {desc}...")

                prompt_config = PromptConfig(
                    strategy=PromptStrategy(strategy),
                    system_prompt=system_prompt,
                )

                run_config = RunConfig(
                    num_runs=args.num_runs,
                    aggregate_method=args.aggregate,
                )

                config = BenchmarkConfig(
                    model=model_config,
                    dataset_path=Path(args.dataset),
                    output_dir=output_dir,
                    categories=args.categories,
                    task_ids=args.task_ids,
                    verbose=not args.quiet,
                    prompt=prompt_config,
                    runs=run_config,
                )

                runner = BenchmarkRunner(config=config)
                results = runner.run(debug=args.debug)

                # Save with descriptive filename
                safe_strategy = strategy.replace("_", "-")
                output_file = output_dir / f"{model_name}_{safe_strategy}_{system_prompt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                results.save(output_file)

                model_results.append({
                    "strategy": strategy,
                    "system_prompt": system_prompt,
                    "description": desc,
                    "pass_rate": results.pass_rate,
                    "passed": results.passed_tasks,
                    "total": results.completed_tasks,
                    "output_file": str(output_file),
                })

                print(f"  Pass rate: {results.pass_rate:.1%} ({results.passed_tasks}/{results.completed_tasks})")
                print(f"  Saved to: {output_file}")

            all_results.append({
                "model": model_name,
                "model_id": model_config.model_id,
                "results": model_results,
            })

    # Print summary
    print(f"\n\n{'='*70}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*70}\n")

    # Create summary table
    header = "| Model | " + " | ".join(c[0][:8] for c in configs) + " |"
    separator = "|" + "---|" * (len(configs) + 1)
    print(header)
    print(separator)

    for model_data in sorted(all_results, key=lambda x: x["model"]):
        row = f"| {model_data['model'][:15]} |"
        for result in model_data["results"]:
            row += f" {result['pass_rate']:.0%} |"
        print(row)

    print()

    # Save ablation summary
    summary_file = output_dir / f"ablation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "models": models_to_run,
            "configurations": [{"strategy": s, "system_prompt": p, "description": d} for s, p, d in configs],
            "results": all_results,
        }, f, indent=2)
    print(f"Summary saved to: {summary_file}")


def main():
    """CLI entry point."""
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Run quantum katas benchmark")
    parser.add_argument("--model", help="Model name (e.g., claude-sonnet, gpt-4o)")
    parser.add_argument("--config", help="JSON config file with model definitions")
    parser.add_argument("--dataset", default="dataset/quantum_katas.jsonl", help="Dataset path")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--categories", nargs="+", help="Filter by categories")
    parser.add_argument("--task-ids", nargs="+", help="Filter by task IDs")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress task-by-task output")
    parser.add_argument("--debug", "-d", action="store_true", help="Show raw model responses for failed tasks")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--all", action="store_true", help="Run benchmark on all configured models")
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        nargs="?",
        const=0,
        default=None,
        help="Run models in parallel (optionally specify max workers, default: number of models)",
    )

    # Prompting options
    parser.add_argument(
        "--prompt-strategy",
        choices=["zero_shot", "few_shot_1", "few_shot_3", "few_shot_5", "chain_of_thought"],
        default="zero_shot",
        help="Prompting strategy (default: zero_shot)",
    )
    parser.add_argument(
        "--system-prompt",
        choices=["default", "minimal", "detailed", "chain_of_thought"],
        default="default",
        help="System prompt variant (default: default)",
    )

    # Multiple runs options
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs per task for statistical analysis (default: 1)",
    )
    parser.add_argument(
        "--aggregate",
        choices=["majority", "any", "all"],
        default="majority",
        help="How to aggregate multiple runs: majority, any, all (default: majority)",
    )

    # Ablation study options
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run all prompting strategy combinations (ignores --prompt-strategy and --system-prompt)",
    )
    parser.add_argument(
        "--ablation-strategies",
        nargs="+",
        choices=["zero_shot", "few_shot_1", "few_shot_3", "few_shot_5", "chain_of_thought"],
        default=None,
        help="Specific strategies to include in ablation (default: all)",
    )

    args = parser.parse_args()

    # Load models from config file or use built-in
    # Priority: --config flag > models.json in cwd > built-in MODELS
    # When using JSON config, only those models are available (no merging with built-ins)
    if args.config:
        models = load_models_from_json(args.config)
    elif Path("models.json").exists():
        models = load_models_from_json("models.json")
    else:
        models = MODELS

    # List models if requested
    if args.list_models:
        print("Available models:")
        for name in sorted(models.keys()):
            cfg = models[name]
            print(f"  {name}: {cfg.provider.value}/{cfg.model_id}")
        sys.exit(0)

    # Determine which models to run
    if args.all:
        models_to_run = list(models.keys())
    elif args.model:
        if args.model not in models:
            available = ", ".join(sorted(models.keys()))
            print(f"Error: Unknown model '{args.model}'. Available: {available}", file=sys.stderr)
            sys.exit(1)
        models_to_run = [args.model]
    else:
        print("Error: --model or --all is required", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle ablation mode - run all prompting combinations
    if args.ablation:
        run_ablation_study(
            models_to_run=models_to_run,
            models=models,
            args=args,
            output_dir=output_dir,
        )
        sys.exit(0)

    # Create prompt and run configs from CLI args (shared across all models)
    prompt_config = PromptConfig(
        strategy=PromptStrategy(args.prompt_strategy),
        system_prompt=args.system_prompt,
    )
    run_config = RunConfig(
        num_runs=args.num_runs,
        aggregate_method=args.aggregate,
    )

    # Run benchmark for each model
    if args.parallel is not None and len(models_to_run) > 1:
        # Parallel execution with progress tracking
        from multiprocessing import Manager
        import threading

        max_workers = args.parallel if args.parallel > 0 else len(models_to_run)
        print(f"\nRunning {len(models_to_run)} models in parallel (max workers: {max_workers})...")
        print(f"Strategy: {args.prompt_strategy}, System prompt: {args.system_prompt}")
        if args.num_runs > 1:
            print(f"Runs per task: {args.num_runs}, Aggregate: {args.aggregate}")

        stop_progress = threading.Event()

        def progress_monitor(queue):
            """Monitor progress queue and update display."""
            while not stop_progress.is_set():
                try:
                    # Non-blocking get with timeout
                    msg = queue.get(timeout=0.5)
                    if msg is None:
                        break
                    name, completed, total, passed, status = msg
                    # Print progress update for each task completion
                    if status in ("PASS", "FAIL", "ERROR"):
                        pass_rate = passed / completed * 100 if completed > 0 else 0
                        print(f"  [{name}] {completed}/{total} - {passed} passed ({pass_rate:.0f}%) - {status}", flush=True)
                    elif status == "STARTED":
                        print(f"  [{name}] Started ({total} tasks)", flush=True)
                except Exception:
                    # Queue.get timeout or other error
                    pass

        # Create multiprocessing manager and queue
        manager = Manager()
        progress_queue = manager.Queue()

        # Start progress monitor thread
        monitor_thread = threading.Thread(target=progress_monitor, args=(progress_queue,), daemon=True)
        monitor_thread.start()

        completed_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_model_benchmark,
                    model_name,
                    models[model_name],
                    Path(args.dataset),
                    output_dir,
                    args.categories,
                    args.task_ids,
                    not args.quiet,
                    progress_queue,
                    prompt_config,
                    run_config,
                ): model_name
                for model_name in models_to_run
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    name, output_file, pass_rate = future.result()
                    completed_results.append((name, output_file, pass_rate))
                    print(f"\n[{name}] COMPLETED: {pass_rate:.1%} -> {output_file}")
                except Exception as e:
                    print(f"\n[{model_name}] FAILED: {e}", file=sys.stderr)

        # Stop progress monitor
        stop_progress.set()
        progress_queue.put(None)
        monitor_thread.join(timeout=1.0)

        # Print summary
        print(f"\n{'='*60}")
        print("Parallel Benchmark Summary")
        print(f"{'='*60}")
        for name, output_file, pass_rate in sorted(completed_results, key=lambda x: -x[2]):
            print(f"  {name}: {pass_rate:.1%}")
    else:
        # Sequential execution
        for model_name in models_to_run:
            model_config = models[model_name]

            config = BenchmarkConfig(
                model=model_config,
                dataset_path=Path(args.dataset),
                output_dir=output_dir,
                categories=args.categories,
                task_ids=args.task_ids,
                verbose=not args.quiet,
                prompt=prompt_config,
                runs=run_config,
            )

            runner = BenchmarkRunner(config=config)
            print(f"\nRunning benchmark with {model_name} ({model_config.model_id})...")
            print(f"Tasks: {len(runner.tasks)}")
            print(f"Strategy: {args.prompt_strategy}, System prompt: {args.system_prompt}")
            if args.num_runs > 1:
                print(f"Runs per task: {args.num_runs}, Aggregate: {args.aggregate}")

            results = runner.run(debug=args.debug)

            output_file = output_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            results.save(output_file)

            print(f"\nResults saved to: {output_file}")
            print(f"Pass rate: {results.pass_rate:.1%} ({results.passed_tasks}/{results.completed_tasks})")


if __name__ == "__main__":
    main()
