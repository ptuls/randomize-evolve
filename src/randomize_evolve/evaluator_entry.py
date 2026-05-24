"""Shared helpers for OpenEvolve evaluator entry points."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Generic, Sequence, TypeVar

from openevolve.evaluation_result import EvaluationResult

ResultT = TypeVar("ResultT")


def score_to_reward(score: float) -> float:
    """Converts a non-negative loss into a bounded reward."""
    return 0.0 if not math.isfinite(score) else 1.0 / (1.0 + max(score, 0.0))


def run_with_timeout(
    func: Callable[..., ResultT],
    *args,
    timeout_seconds: float,
    **kwargs,
) -> ResultT:
    """Executes ``func`` with a wall-clock timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:  # pragma: no cover - best effort
            future.cancel()
            raise TimeoutError(f"evaluation exceeded {timeout_seconds}s wall-clock limit") from exc


def load_candidate_factory(
    program_path: str,
    exported_names: Sequence[str] = ("candidate_factory", "build_candidate"),
) -> Callable[..., object]:
    """Loads a candidate factory from a Python module on disk."""
    path = Path(program_path)
    if not path.exists():
        raise FileNotFoundError(f"program path {path} does not exist")

    spec = importlib.util.spec_from_file_location("candidate_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]

    factory = extract_exported_callable(module, exported_names)
    if not callable(factory):
        raise TypeError(f"{exported_names[0]} must be callable")
    return factory


def extract_exported_callable(
    module: ModuleType,
    exported_names: Sequence[str],
) -> Callable[..., object]:
    """Returns the first supported exported callable from ``module``."""
    for name in exported_names:
        if hasattr(module, name):
            return getattr(module, name)  # type: ignore[return-value]
    joined_names = " or ".join(f"`{name}`" for name in exported_names)
    raise AttributeError(f"candidate module must expose {joined_names}")


@dataclass(frozen=True)
class EvaluationEntryPoint(Generic[ResultT]):
    """Coordinates the common evaluator entry-point flow."""

    evaluator_factory: Callable[[], Callable[[Callable[..., object]], ResultT]]
    timeout_seconds: float
    load_error_suggestion: str
    timeout_suggestion: str
    success_result_builder: Callable[[ResultT], EvaluationResult]
    error_result_builder: Callable[[str, dict[str, Any]], EvaluationResult]
    unexpected_error_suggestion: str = "Unexpected evaluator failure; inspect the traceback."
    exported_names: Sequence[str] = ("candidate_factory", "build_candidate")

    def evaluate(self, program_path: str) -> EvaluationResult:
        """Loads a candidate module, evaluates it, and adapts the result."""
        try:
            factory = load_candidate_factory(program_path, self.exported_names)
        except Exception as exc:  # pragma: no cover - defensive
            artifacts = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "full_traceback": traceback.format_exc(),
                "suggestion": self.load_error_suggestion,
            }
            return self.error_result_builder(
                "failed to load candidate factory",
                artifacts,
            )

        evaluator = self.evaluator_factory()

        try:
            result = run_with_timeout(
                evaluator,
                factory,
                timeout_seconds=self.timeout_seconds,
            )
        except TimeoutError as exc:
            artifacts = {
                "error_type": "TimeoutError",
                "error_message": str(exc),
                "suggestion": self.timeout_suggestion,
            }
            return self.error_result_builder("evaluation timed out", artifacts)
        except Exception as exc:  # pragma: no cover - defensive
            artifacts = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "full_traceback": traceback.format_exc(),
                "suggestion": self.unexpected_error_suggestion,
            }
            return self.error_result_builder("evaluation failed", artifacts)

        return self.success_result_builder(result)
