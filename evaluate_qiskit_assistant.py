#!/usr/bin/env python3
"""
Evaluate Qiskit Code Assistant on the Quantum Katas dataset.

This script uses the OpenAI-compatible Completions API (not Chat API)
to benchmark the Qiskit Code Assistant service.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QiskitCodeAssistantEvaluator:
    """Evaluator for Qiskit Code Assistant using OpenAI Completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://qiskit-code-assistant.quantum.ibm.com/v1",
        model: str = "mistral-small-3.2-24b-qiskit",
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ):
        """
        Initialize the evaluator.

        Args:
            api_key: IBM Quantum API token (or set IBM_QUANTUM_TOKEN env var)
            base_url: Base URL for the Qiskit Code Assistant API
            model: Model name to use
            max_tokens: Maximum tokens for completion
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.api_key = api_key or os.getenv("IBM_QUANTUM_TOKEN")
        if not self.api_key:
            raise ValueError(
                "IBM_QUANTUM_TOKEN not found. Set it as environment variable or pass api_key."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load the JSONL dataset."""
        with open(dataset_path, "r") as f:
            return [json.loads(line) for line in f]

    def create_prompt(self, task: Dict) -> str:
        """
        Create a completion prompt from a task.

        The Completions API expects a prompt string, not messages.
        We format it to encourage the model to generate just the function.
        """
        prompt = f"""# Task: {task['task_id']}

{task['prompt']}

# Generate a Python function using Qiskit 2.2+
# Function name: {task['entry_point']}

def {task['entry_point']}"""

        return prompt

    def generate_solution(self, task: Dict, retry_count: int = 3) -> Optional[str]:
        """
        Generate a solution using the Qiskit Code Assistant.

        Args:
            task: Task dictionary with prompt
            retry_count: Number of retries on failure

        Returns:
            Generated code or None if failed
        """
        prompt = self.create_prompt(task)

        for attempt in range(retry_count):
            try:
                # Use Completions API (not Chat API)
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=["def ", "\n\n\n", "# Task:"],  # Stop at next function or task
                )

                # Extract the completion
                completion = response.choices[0].text

                # Reconstruct the full function
                # The prompt already includes "def entry_point", so we add the completion
                full_function = f"def {task['entry_point']}" + completion

                return full_function

            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    print(f"Error generating solution for {task['task_id']}: {e}")
                    return None

    def evaluate_solution(self, task: Dict, solution: str) -> Dict:
        """
        Evaluate a generated solution against the test.

        Args:
            task: Task dictionary with test
            solution: Generated solution code

        Returns:
            Dictionary with evaluation results
        """
        result = {
            "task_id": task["task_id"],
            "passed": False,
            "error": None,
            "solution": solution,
        }

        if solution is None:
            result["error"] = "Generation failed"
            return result

        # Combine solution with test
        test_code = solution + "\n\n" + task["test"]

        try:
            # Execute the test
            exec(test_code, {})
            result["passed"] = True
        except Exception as e:
            result["passed"] = False
            result["error"] = f"{type(e).__name__}: {str(e)[:200]}"

        return result

    def evaluate_dataset(
        self,
        dataset_path: str = "quantum_katas_dataset_verified.jsonl",
        output_path: Optional[str] = None,
        max_tasks: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate the model on the entire dataset.

        Args:
            dataset_path: Path to the JSONL dataset
            output_path: Path to save detailed results (optional)
            max_tasks: Maximum number of tasks to evaluate (None = all)

        Returns:
            Dictionary with evaluation results
        """
        print(f"Loading dataset from {dataset_path}...")
        tasks = self.load_dataset(dataset_path)

        if max_tasks:
            tasks = tasks[:max_tasks]
            print(f"Evaluating first {max_tasks} tasks")

        print(f"Evaluating {len(tasks)} tasks with model: {self.model}")
        print(f"Base URL: {self.client.base_url}")
        print()

        results = []
        passed_count = 0
        failed_count = 0

        for task in tqdm(tasks, desc="Evaluating"):
            # Generate solution
            solution = self.generate_solution(task)

            # Evaluate
            result = self.evaluate_solution(task, solution)

            if result["passed"]:
                passed_count += 1
            else:
                failed_count += 1

            results.append(result)

            # Rate limiting - be nice to the API
            time.sleep(0.5)

        # Compute statistics
        total = len(tasks)
        pass_rate = (passed_count / total) * 100 if total > 0 else 0

        summary = {
            "model": self.model,
            "base_url": str(self.client.base_url),
            "dataset": dataset_path,
            "timestamp": datetime.now().isoformat(),
            "total_tasks": total,
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": f"{pass_rate:.2f}%",
            "results": results,
        }

        # Save detailed results if output path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nDetailed results saved to: {output_path}")

        return summary

    def print_summary(self, summary: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Model: {summary['model']}")
        print(f"Dataset: {summary['dataset']}")
        print(f"Total tasks: {summary['total_tasks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass rate: {summary['pass_rate']}")
        print("=" * 80)

        # Show some example failures
        failures = [r for r in summary["results"] if not r["passed"]]
        if failures:
            print("\nExample failures:")
            for i, failure in enumerate(failures[:5], 1):
                print(f"\n{i}. {failure['task_id']}")
                if failure["error"]:
                    print(f"   Error: {failure['error'][:100]}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Qiskit Code Assistant on Quantum Katas dataset"
    )
    parser.add_argument(
        "--dataset",
        default="quantum_katas_dataset_verified.jsonl",
        help="Path to dataset file (default: quantum_katas_dataset_verified.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save detailed results JSON (default: results_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="IBM Quantum API token (or set IBM_QUANTUM_TOKEN env var)",
    )
    parser.add_argument(
        "--base-url",
        default="https://qiskit-code-assistant.quantum.ibm.com/v1",
        help="Base URL for API (default: Qiskit Code Assistant)",
    )
    parser.add_argument(
        "--model",
        default="mistral-small-3.2-24b-qiskit",
        help="Model name (default: mistral-small-3.2-24b-qiskit)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate (default: all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )

    args = parser.parse_args()

    # Generate default output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_{timestamp}.json"

    # Create evaluator
    try:
        evaluator = QiskitCodeAssistantEvaluator(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo get your IBM Quantum API token:")
        print("1. Visit https://quantum.ibm.com/")
        print("2. Go to your account settings")
        print("3. Copy your API token")
        print("4. Set it as: export IBM_QUANTUM_TOKEN='your-token-here'")
        sys.exit(1)

    # Run evaluation
    summary = evaluator.evaluate_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        max_tasks=args.max_tasks,
    )

    # Print summary
    evaluator.print_summary(summary)


if __name__ == "__main__":
    main()
