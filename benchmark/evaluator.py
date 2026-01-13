"""Code evaluation for quantum katas benchmark."""

import ast
import re
import traceback
from dataclasses import dataclass
from typing import Optional
import multiprocessing
import sys
import io


@dataclass
class EvaluationResult:
    """Result of evaluating a generated solution."""

    passed: bool
    syntax_valid: bool
    test_passed: bool
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stdout: Optional[str] = None


def extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response.

    Handles markdown code blocks and raw code.
    """
    # Try to extract from markdown code block
    patterns = [
        r"```python\n(.*?)```",
        r"```py\n(.*?)```",
        r"```\n(.*?)```",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no code block, return as-is (might be raw code)
    return response.strip()


def check_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Check if code has valid Python syntax."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _run_test_worker(code: str, test_code: str, entry_point: str, result_queue):
    """Worker function for running tests in subprocess."""
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture

    try:
        # Create execution namespace
        namespace = {}

        # Execute the solution code
        exec(code, namespace)

        # Check if entry point exists
        if entry_point not in namespace:
            result_queue.put({
                "passed": False,
                "error_message": f"Entry point '{entry_point}' not found in solution",
                "error_type": "MissingEntryPoint",
                "stdout": stdout_capture.getvalue(),
            })
            return

        # Execute test code
        test_namespace = namespace.copy()
        exec(test_code, test_namespace)

        # Find and run test function
        test_func_name = f"test_{entry_point}"
        if test_func_name not in test_namespace:
            # Try to find any test function
            test_funcs = [k for k in test_namespace if k.startswith("test_")]
            if test_funcs:
                test_func_name = test_funcs[0]
            else:
                result_queue.put({
                    "passed": False,
                    "error_message": "No test function found",
                    "error_type": "MissingTest",
                    "stdout": stdout_capture.getvalue(),
                })
                return

        # Run the test
        test_namespace[test_func_name]()

        result_queue.put({
            "passed": True,
            "error_message": None,
            "error_type": None,
            "stdout": stdout_capture.getvalue(),
        })

    except AssertionError as e:
        result_queue.put({
            "passed": False,
            "error_message": f"AssertionError: {str(e)}",
            "error_type": "AssertionError",
            "stdout": stdout_capture.getvalue(),
        })
    except Exception as e:
        result_queue.put({
            "passed": False,
            "error_message": f"{type(e).__name__}: {str(e)}",
            "error_type": type(e).__name__,
            "stdout": stdout_capture.getvalue(),
        })
    finally:
        sys.stdout = old_stdout


def run_test(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: float = 30.0,
) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """Run test code against solution with timeout.

    Returns: (passed, error_message, error_type, stdout)
    """
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=_run_test_worker,
        args=(code, test_code, entry_point, result_queue),
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False, f"Timeout after {timeout}s", "Timeout", None

    if result_queue.empty():
        return False, "Process ended without result", "ProcessError", None

    result = result_queue.get()
    return (
        result["passed"],
        result["error_message"],
        result["error_type"],
        result["stdout"],
    )


def evaluate_solution(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: float = 30.0,
) -> EvaluationResult:
    """Evaluate a generated solution.

    Args:
        code: The generated code from the LLM
        test_code: The test code to run
        entry_point: The expected function name
        timeout: Maximum execution time in seconds

    Returns:
        EvaluationResult with pass/fail status and details
    """
    # Extract code from response
    extracted_code = extract_code_from_response(code)

    # Check syntax
    syntax_valid, syntax_error = check_syntax(extracted_code)
    if not syntax_valid:
        return EvaluationResult(
            passed=False,
            syntax_valid=False,
            test_passed=False,
            error_message=syntax_error,
            error_type="SyntaxError",
        )

    # Run tests
    test_passed, error_message, error_type, stdout = run_test(
        extracted_code,
        test_code,
        entry_point,
        timeout,
    )

    return EvaluationResult(
        passed=test_passed,
        syntax_valid=True,
        test_passed=test_passed,
        error_message=error_message,
        error_type=error_type,
        stdout=stdout,
    )


def validate_dataset_solution(
    solution: str,
    test_code: str,
    entry_point: str,
) -> bool:
    """Quick validation of a dataset solution (for testing dataset quality)."""
    result = evaluate_solution(solution, test_code, entry_point)
    return result.passed
