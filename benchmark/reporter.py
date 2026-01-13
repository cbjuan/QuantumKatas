"""Results reporting for quantum katas benchmark."""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CategoryStats:
    """Statistics for a category."""

    name: str
    total: int
    passed: int
    failed: int
    errors: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


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


def generate_report(results: dict) -> BenchmarkReport:
    """Generate a report from benchmark results."""
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
        )
        for name, data in sorted(category_data.items())
    ]

    num_results = len(results.get("results", []))

    return BenchmarkReport(
        model_id=results.get("model_id", "unknown"),
        provider=results.get("provider", "unknown"),
        total_tasks=results.get("total_tasks", 0),
        passed=results.get("passed_tasks", 0),
        failed=results.get("failed_tasks", 0),
        errors=results.get("error_tasks", 0),
        pass_rate=results.get("pass_rate", 0.0),
        categories=categories,
        avg_latency_ms=total_latency / num_results if num_results > 0 else 0,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        error_breakdown=dict(error_breakdown),
    )


def format_markdown_report(report: BenchmarkReport) -> str:
    """Format report as markdown."""
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
        f"| Avg Latency | {report.avg_latency_ms:.0f}ms |",
        f"| Total Input Tokens | {report.total_input_tokens:,} |",
        f"| Total Output Tokens | {report.total_output_tokens:,} |",
        "",
        "## Results by Category",
        "",
        "| Category | Total | Passed | Failed | Errors | Pass Rate |",
        "|----------|-------|--------|--------|--------|-----------|",
    ]

    for cat in report.categories:
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


def compare_reports(reports: list[BenchmarkReport], include_categories: bool = False) -> str:
    """Generate comparison table for multiple reports."""
    lines = [
        "# Model Comparison",
        "",
        "## Overall Results",
        "",
        "| Model | Pass Rate | Passed | Failed | Errors | Avg Latency |",
        "|-------|-----------|--------|--------|--------|-------------|",
    ]

    for report in sorted(reports, key=lambda r: -r.pass_rate):
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
