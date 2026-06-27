"""Execution runners for supported HumanEval-X target languages."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    """Execution outcome for one candidate."""

    available: bool
    passed: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


def select_test(test: str, example_test: str, budget: str) -> str:
    """Select a test body for an execution budget."""
    if budget == "none":
        return ""
    if budget == "example":
        return example_test
    return test


def run_candidate(language: str, candidate_code: str, test: str, timeout_seconds: int) -> ExecutionResult:
    """Run candidate code plus test for a supported target language."""
    if not test:
        return ExecutionResult(False, False, None, "", "execution budget is none", False)
    language = language.lower()
    if language == "python":
        return _run_python(candidate_code, test, timeout_seconds)
    if language == "js":
        return _run_js(candidate_code, test, timeout_seconds)
    if language == "java":
        return _run_java(candidate_code, test, timeout_seconds)
    return ExecutionResult(False, False, None, "", f"unsupported language: {language}", False)


def _run_subprocess(cmd: list[str], cwd: Path, timeout_seconds: int) -> ExecutionResult:
    """Run a subprocess and capture outcome."""
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult(False, False, None, exc.stdout or "", exc.stderr or "", True)
    return ExecutionResult(
        available=True,
        passed=completed.returncode == 0,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
    )


def _run_python(candidate_code: str, test: str, timeout_seconds: int) -> ExecutionResult:
    """Run Python candidate."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(candidate_code + "\n\n" + test, encoding="utf-8")
        return _run_subprocess(["python", str(path)], Path(tmp), timeout_seconds)


def _run_js(candidate_code: str, test: str, timeout_seconds: int) -> ExecutionResult:
    """Run JavaScript candidate."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.js"
        path.write_text(candidate_code + "\n\n" + test, encoding="utf-8")
        return _run_subprocess(["node", str(path)], Path(tmp), timeout_seconds)


def _run_java(candidate_code: str, test: str, timeout_seconds: int) -> ExecutionResult:
    """Compile and run Java candidate."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "Main.java"
        path.write_text(candidate_code + "\n\n" + test, encoding="utf-8")
        compile_result = _run_subprocess(["javac", "Main.java"], root, timeout_seconds)
        if not compile_result.passed:
            return compile_result
        return _run_subprocess(["java", "Main"], root, timeout_seconds)
