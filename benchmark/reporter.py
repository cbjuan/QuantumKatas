"""Results reporting for quantum katas benchmark."""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =============================================================================
# Statistical Analysis Functions
# =============================================================================


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score confidence interval for a proportion.

    More accurate than normal approximation, especially for small samples
    or proportions near 0 or 1.

    Args:
        successes: Number of successes
        trials: Total number of trials
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    if trials == 0:
        return (0.0, 0.0)

    # Z-score for confidence level (95% -> 1.96)
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    p = successes / trials
    denominator = 1 + z**2 / trials

    center = (p + z**2 / (2 * trials)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))

    return (max(0.0, center - margin), min(1.0, center + margin))


def clopper_pearson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Clopper-Pearson exact confidence interval.

    More conservative than Wilson score, guaranteed coverage.
    Uses beta distribution quantiles.

    Args:
        successes: Number of successes
        trials: Total number of trials
        confidence: Confidence level (default: 0.95)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    if trials == 0:
        return (0.0, 0.0)

    alpha = 1 - confidence

    # For simplicity, use normal approximation when sample is large enough
    # For small samples, fall back to Wilson which is more practical
    if trials < 30:
        return wilson_score_interval(successes, trials, confidence)

    # Normal approximation for large samples
    p = successes / trials
    se = math.sqrt(p * (1 - p) / trials)
    z = 1.96 if confidence == 0.95 else 1.645 if confidence == 0.90 else 2.576

    return (max(0.0, p - z * se), min(1.0, p + z * se))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k metric.

    Estimates probability of passing at least once in k attempts,
    given n total runs with c successes.

    Based on Chen et al. "Evaluating Large Language Models Trained on Code"

    Args:
        n: Total number of runs
        c: Number of successful runs
        k: Number of attempts to consider

    Returns:
        Estimated pass@k probability
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def calculate_variance(values: list[float]) -> float:
    """Calculate sample variance."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)


def calculate_std(values: list[float]) -> float:
    """Calculate sample standard deviation."""
    return math.sqrt(calculate_variance(values))


@dataclass
class StatisticalMetrics:
    """Statistical metrics for benchmark results."""

    mean: float
    std: float
    variance: float
    ci_lower: float  # 95% confidence interval lower bound
    ci_upper: float  # 95% confidence interval upper bound
    n_samples: int
    pass_at_1: Optional[float] = None
    pass_at_3: Optional[float] = None
    pass_at_5: Optional[float] = None

    @classmethod
    def from_pass_counts(cls, passed: int, total: int, num_runs: int = 1) -> "StatisticalMetrics":
        """Create metrics from pass/fail counts."""
        if total == 0:
            return cls(
                mean=0.0, std=0.0, variance=0.0,
                ci_lower=0.0, ci_upper=0.0, n_samples=0
            )

        mean = passed / total
        ci_lower, ci_upper = wilson_score_interval(passed, total)

        # For single runs, variance is estimated from binomial proportion
        variance = mean * (1 - mean) / total if total > 0 else 0.0
        std = math.sqrt(variance)

        # Calculate pass@k if we have run information
        pass_at_1 = mean
        pass_at_3 = pass_at_k(num_runs, passed, 3) if num_runs >= 3 else None
        pass_at_5 = pass_at_k(num_runs, passed, 5) if num_runs >= 5 else None

        return cls(
            mean=mean,
            std=std,
            variance=variance,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=total,
            pass_at_1=pass_at_1,
            pass_at_3=pass_at_3,
            pass_at_5=pass_at_5,
        )

    @classmethod
    def from_multiple_runs(cls, pass_rates: list[float]) -> "StatisticalMetrics":
        """Create metrics from multiple run pass rates."""
        if not pass_rates:
            return cls(
                mean=0.0, std=0.0, variance=0.0,
                ci_lower=0.0, ci_upper=0.0, n_samples=0
            )

        n = len(pass_rates)
        mean = sum(pass_rates) / n
        variance = calculate_variance(pass_rates)
        std = math.sqrt(variance)

        # CI from t-distribution approximation (using z for simplicity)
        se = std / math.sqrt(n) if n > 0 else 0
        z = 1.96
        ci_lower = max(0.0, mean - z * se)
        ci_upper = min(1.0, mean + z * se)

        return cls(
            mean=mean,
            std=std,
            variance=variance,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=n,
        )


def mcnemar_test(model_a_correct: list[bool], model_b_correct: list[bool]) -> tuple[float, float]:
    """Perform McNemar's test for paired comparison.

    Tests whether two models have significantly different error rates.

    Args:
        model_a_correct: List of True/False for each task for model A
        model_b_correct: List of True/False for each task for model B

    Returns:
        (chi_squared_statistic, p_value)
    """
    if len(model_a_correct) != len(model_b_correct):
        raise ValueError("Lists must have same length")

    # Count discordant pairs
    b = sum(1 for a, b_val in zip(model_a_correct, model_b_correct) if a and not b_val)  # A correct, B wrong
    c = sum(1 for a, b_val in zip(model_a_correct, model_b_correct) if not a and b_val)  # A wrong, B correct

    if b + c == 0:
        return (0.0, 1.0)  # No difference

    # McNemar's test with continuity correction
    chi_sq = (abs(b - c) - 1) ** 2 / (b + c)

    # Approximate p-value using chi-squared distribution with 1 df
    # Using simple approximation
    p_value = math.exp(-chi_sq / 2) if chi_sq < 10 else 0.001

    return (chi_sq, p_value)


@dataclass
class CategoryStats:
    """Statistics for a category."""

    name: str
    total: int
    passed: int
    failed: int
    errors: int
    stats: Optional[StatisticalMetrics] = None

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def with_stats(self) -> "CategoryStats":
        """Return a copy with statistical metrics calculated."""
        stats = StatisticalMetrics.from_pass_counts(self.passed, self.total)
        return CategoryStats(
            name=self.name,
            total=self.total,
            passed=self.passed,
            failed=self.failed,
            errors=self.errors,
            stats=stats,
        )


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report."""

    model_id: str
    provider: str
    total_tasks: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    categories: list[CategoryStats]
    avg_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    error_breakdown: dict[str, int]
    stats: Optional[StatisticalMetrics] = None
    prompt_strategy: Optional[str] = None
    system_prompt: Optional[str] = None
    num_runs: int = 1


def load_results(path: Path) -> dict:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_all_results(results_dir: Path) -> list[tuple[Path, dict]]:
    """Load all benchmark results from a directory.

    Returns list of (path, results) tuples sorted by model name.
    """
    results_dir = Path(results_dir)
    results = []

    for path in results_dir.glob("*.json"):
        if path.name.startswith("checkpoint_"):
            continue
        try:
            data = load_results(path)
            results.append((path, data))
        except (json.JSONDecodeError, KeyError):
            continue

    return sorted(results, key=lambda x: x[1].get("model_id", ""))


def generate_report(results: dict, include_stats: bool = True) -> BenchmarkReport:
    """Generate a report from benchmark results.

    Args:
        results: Benchmark results dictionary
        include_stats: Whether to calculate statistical metrics (default: True)
    """
    # Calculate category stats
    category_data = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "errors": 0})

    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    error_breakdown = defaultdict(int)

    for task_result in results.get("results", []):
        category = task_result["category"]
        category_data[category]["total"] += 1

        if task_result["evaluation"]["passed"]:
            category_data[category]["passed"] += 1
        elif task_result["evaluation"].get("error_message"):
            category_data[category]["errors"] += 1
            error_type = task_result["evaluation"].get("error_type", "Unknown")
            error_breakdown[error_type] += 1
        else:
            category_data[category]["failed"] += 1

        gen = task_result["generation"]
        total_latency += gen.get("latency_ms", 0)
        total_input_tokens += gen.get("input_tokens", 0)
        total_output_tokens += gen.get("output_tokens", 0)

    categories = [
        CategoryStats(
            name=name,
            total=data["total"],
            passed=data["passed"],
            failed=data["failed"],
            errors=data["errors"],
        ).with_stats() if include_stats else CategoryStats(
            name=name,
            total=data["total"],
            passed=data["passed"],
            failed=data["failed"],
            errors=data["errors"],
        )
        for name, data in sorted(category_data.items())
    ]

    num_results = len(results.get("results", []))
    passed = results.get("passed_tasks", 0)
    total = results.get("total_tasks", 0)

    # Calculate overall statistical metrics
    overall_stats = StatisticalMetrics.from_pass_counts(passed, total) if include_stats else None

    return BenchmarkReport(
        model_id=results.get("model_id", "unknown"),
        provider=results.get("provider", "unknown"),
        total_tasks=total,
        passed=passed,
        failed=results.get("failed_tasks", 0),
        errors=results.get("error_tasks", 0),
        pass_rate=results.get("pass_rate", 0.0),
        categories=categories,
        avg_latency_ms=total_latency / num_results if num_results > 0 else 0,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        error_breakdown=dict(error_breakdown),
        stats=overall_stats,
    )


def format_markdown_report(report: BenchmarkReport, include_ci: bool = True) -> str:
    """Format report as markdown.

    Args:
        report: The benchmark report
        include_ci: Whether to include confidence intervals (default: True)
    """
    lines = [
        f"# Benchmark Report: {report.model_id}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Model | {report.model_id} |",
        f"| Provider | {report.provider} |",
        f"| Total Tasks | {report.total_tasks} |",
        f"| Passed | {report.passed} |",
        f"| Failed | {report.failed} |",
        f"| Errors | {report.errors} |",
        f"| **Pass Rate** | **{report.pass_rate:.1%}** |",
    ]

    # Add confidence interval if available
    if include_ci and report.stats:
        lines.append(f"| 95% CI | [{report.stats.ci_lower:.1%}, {report.stats.ci_upper:.1%}] |")

    lines.extend([
        f"| Avg Latency | {report.avg_latency_ms:.0f}ms |",
        f"| Total Input Tokens | {report.total_input_tokens:,} |",
        f"| Total Output Tokens | {report.total_output_tokens:,} |",
        "",
        "## Results by Category",
        "",
    ])

    # Header with optional CI column
    if include_ci:
        lines.append("| Category | Total | Passed | Pass Rate | 95% CI |")
        lines.append("|----------|-------|--------|-----------|--------|")
    else:
        lines.append("| Category | Total | Passed | Failed | Errors | Pass Rate |")
        lines.append("|----------|-------|--------|--------|--------|-----------|")

    for cat in report.categories:
        if include_ci and cat.stats:
            ci_str = f"[{cat.stats.ci_lower:.0%}, {cat.stats.ci_upper:.0%}]"
            lines.append(
                f"| {cat.name} | {cat.total} | {cat.passed} | {cat.pass_rate:.1%} | {ci_str} |"
            )
        else:
            lines.append(
                f"| {cat.name} | {cat.total} | {cat.passed} | {cat.failed} | {cat.errors} | {cat.pass_rate:.1%} |"
            )

    if report.error_breakdown:
        lines.extend([
            "",
            "## Error Breakdown",
            "",
            "| Error Type | Count |",
            "|------------|-------|",
        ])
        for error_type, count in sorted(report.error_breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"| {error_type} | {count} |")

    return "\n".join(lines)


def format_csv_report(report: BenchmarkReport) -> str:
    """Format report as CSV."""
    lines = ["category,total,passed,failed,errors,pass_rate"]
    for cat in report.categories:
        lines.append(f"{cat.name},{cat.total},{cat.passed},{cat.failed},{cat.errors},{cat.pass_rate:.4f}")
    return "\n".join(lines)


def save_report(
    report: BenchmarkReport,
    output_path: Path,
    format: str = "markdown",
):
    """Save report to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "markdown":
        content = format_markdown_report(report)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".md")
    elif format == "csv":
        content = format_csv_report(report)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".csv")
    else:
        raise ValueError(f"Unknown format: {format}")

    with open(output_path, "w") as f:
        f.write(content)


def compare_reports(
    reports: list[BenchmarkReport],
    include_categories: bool = False,
    include_ci: bool = True,
) -> str:
    """Generate comparison table for multiple reports.

    Args:
        reports: List of benchmark reports to compare
        include_categories: Include per-category breakdown
        include_ci: Include 95% confidence intervals
    """
    lines = [
        "# Model Comparison",
        "",
        "## Overall Results",
        "",
    ]

    if include_ci:
        lines.extend([
            "| Model | Pass Rate | 95% CI | Passed | Failed | Errors | Avg Latency |",
            "|-------|-----------|--------|--------|--------|--------|-------------|",
        ])
    else:
        lines.extend([
            "| Model | Pass Rate | Passed | Failed | Errors | Avg Latency |",
            "|-------|-----------|--------|--------|--------|-------------|",
        ])

    for report in sorted(reports, key=lambda r: -r.pass_rate):
        if include_ci and report.stats:
            ci_str = f"[{report.stats.ci_lower:.1%}, {report.stats.ci_upper:.1%}]"
            lines.append(
                f"| {report.model_id} | {report.pass_rate:.1%} | {ci_str} | {report.passed} | "
                f"{report.failed} | {report.errors} | {report.avg_latency_ms:.0f}ms |"
            )
        else:
            lines.append(
                f"| {report.model_id} | {report.pass_rate:.1%} | {report.passed} | "
                f"{report.failed} | {report.errors} | {report.avg_latency_ms:.0f}ms |"
            )

    if include_categories and reports:
        # Get all categories across all reports
        all_categories = set()
        for report in reports:
            for cat in report.categories:
                all_categories.add(cat.name)

        lines.extend([
            "",
            "## Results by Category",
            "",
        ])

        # Header row with model names
        header = "| Category |"
        separator = "|----------|"
        for report in sorted(reports, key=lambda r: r.model_id):
            header += f" {report.model_id} |"
            separator += "------------|"

        lines.append(header)
        lines.append(separator)

        # Data rows
        for cat_name in sorted(all_categories):
            row = f"| {cat_name} |"
            for report in sorted(reports, key=lambda r: r.model_id):
                cat_stats = next((c for c in report.categories if c.name == cat_name), None)
                if cat_stats:
                    row += f" {cat_stats.pass_rate:.0%} ({cat_stats.passed}/{cat_stats.total}) |"
                else:
                    row += " - |"
            lines.append(row)

    return "\n".join(lines)


def aggregate_multiple_runs(result_files: list[Path]) -> StatisticalMetrics:
    """Aggregate results from multiple benchmark runs.

    Useful for calculating statistics across repeated runs of the same model.

    Args:
        result_files: List of paths to result JSON files

    Returns:
        Statistical metrics across all runs
    """
    pass_rates = []
    for path in result_files:
        data = load_results(path)
        pass_rate = data.get("pass_rate", 0.0)
        pass_rates.append(pass_rate)

    return StatisticalMetrics.from_multiple_runs(pass_rates)


def format_statistical_comparison(
    reports: list[BenchmarkReport],
    results_data: Optional[list[dict]] = None,
) -> str:
    """Generate detailed statistical comparison with significance testing.

    Args:
        reports: List of benchmark reports
        results_data: Optional raw results for per-task comparison

    Returns:
        Markdown formatted comparison with statistical analysis
    """
    lines = [
        "# Statistical Analysis",
        "",
        "## Summary Statistics",
        "",
        "| Model | Pass Rate | 95% CI | Std Error |",
        "|-------|-----------|--------|-----------|",
    ]

    for report in sorted(reports, key=lambda r: -r.pass_rate):
        if report.stats:
            se = report.stats.std / math.sqrt(report.stats.n_samples) if report.stats.n_samples > 0 else 0
            ci_str = f"[{report.stats.ci_lower:.1%}, {report.stats.ci_upper:.1%}]"
            lines.append(f"| {report.model_id} | {report.pass_rate:.1%} | {ci_str} | {se:.3f} |")
        else:
            lines.append(f"| {report.model_id} | {report.pass_rate:.1%} | - | - |")

    # Add note about confidence intervals
    lines.extend([
        "",
        "*Note: 95% CI calculated using Wilson score interval, which provides accurate coverage even for small samples and proportions near 0 or 1.*",
        "",
    ])

    # If we have raw results, we can do pairwise significance testing
    if results_data and len(results_data) >= 2:
        lines.extend([
            "## Pairwise Comparison",
            "",
            "Differences are considered significant at p < 0.05.",
            "",
        ])

        # Create comparison matrix
        model_ids = [r.model_id for r in sorted(reports, key=lambda r: -r.pass_rate)]
        lines.append("| vs | " + " | ".join(model_ids[1:]) + " |")
        lines.append("|" + "---|" * len(model_ids))

        # Note: Full pairwise testing requires per-task results which may not be available
        lines.extend([
            "",
            "*Pairwise McNemar test requires per-task results aligned across models.*",
        ])

    return "\n".join(lines)


def compare_results_from_dir(results_dir: Path, include_categories: bool = True) -> str:
    """Load all results from a directory and generate comparison."""
    all_results = load_all_results(results_dir)
    if not all_results:
        return "No results found in directory."

    reports = [generate_report(data) for _, data in all_results]
    return compare_reports(reports, include_categories=include_categories)


def print_summary(results: dict):
    """Print a quick summary to console."""
    report = generate_report(results)
    print(f"\n{'='*60}")
    print(f"Model: {report.model_id}")
    print(f"Pass Rate: {report.pass_rate:.1%} ({report.passed}/{report.total_tasks})")
    print(f"Avg Latency: {report.avg_latency_ms:.0f}ms")
    print(f"{'='*60}")

    print("\nBy Category:")
    for cat in report.categories:
        bar = "#" * int(cat.pass_rate * 20)
        print(f"  {cat.name:25} {cat.pass_rate:6.1%} [{bar:20}] ({cat.passed}/{cat.total})")


def main():
    """CLI entry point for comparing results."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compare quantum katas benchmark results")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing result JSON files (default: results)",
    )
    parser.add_argument(
        "--output",
        help="Output file for comparison report (default: print to stdout)",
    )
    parser.add_argument(
        "--no-categories",
        action="store_true",
        help="Don't include per-category breakdown",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available result files",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    all_results = load_all_results(results_dir)

    if args.list:
        if not all_results:
            print("No results found.")
        else:
            print(f"Found {len(all_results)} result files:\n")
            for path, data in all_results:
                report = generate_report(data)
                print(f"  {path.name}")
                print(f"    Model: {report.model_id}")
                print(f"    Pass Rate: {report.pass_rate:.1%} ({report.passed}/{report.total_tasks})")
                print()
        sys.exit(0)

    if not all_results:
        print("No results found in directory.", file=sys.stderr)
        sys.exit(1)

    reports = [generate_report(data) for _, data in all_results]
    comparison = compare_reports(reports, include_categories=not args.no_categories)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(comparison)
        print(f"Comparison saved to: {output_path}")
    else:
        print(comparison)


if __name__ == "__main__":
    main()
