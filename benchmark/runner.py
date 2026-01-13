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

from .config import BenchmarkConfig, ModelConfig, get_model_config, load_models_from_json, MODELS
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


def run_single_model_benchmark(
    model_name: str,
    model_config: ModelConfig,
    dataset_path: Path,
    output_dir: Path,
    categories: Optional[list[str]],
    task_ids: Optional[list[str]],
    verbose: bool,
    progress_queue=None,
) -> tuple[str, Path, float]:
    """Run benchmark for a single model. Used for parallel execution."""
    # When using progress queue, don't print verbose output (use queue instead)
    use_verbose = verbose and not progress_queue

    config = BenchmarkConfig(
        model=model_config,
        dataset_path=dataset_path,
        output_dir=output_dir,
        categories=categories,
        task_ids=task_ids,
        verbose=use_verbose,
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

    # Run benchmark for each model
    if args.parallel is not None and len(models_to_run) > 1:
        # Parallel execution with progress tracking
        from multiprocessing import Manager
        import threading

        max_workers = args.parallel if args.parallel > 0 else len(models_to_run)
        print(f"\nRunning {len(models_to_run)} models in parallel (max workers: {max_workers})...")

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
            )

            runner = BenchmarkRunner(config=config)
            print(f"\nRunning benchmark with {model_name} ({model_config.model_id})...")
            print(f"Tasks: {len(runner.tasks)}")

            results = runner.run(debug=args.debug)

            output_file = output_dir / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            results.save(output_file)

            print(f"\nResults saved to: {output_file}")
            print(f"Pass rate: {results.pass_rate:.1%} ({results.passed_tasks}/{results.completed_tasks})")


if __name__ == "__main__":
    main()
