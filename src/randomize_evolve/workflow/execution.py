import importlib
import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from randomize_evolve.evaluator_entry import (
    EvaluatorResult,
    _exec_registered_module,
    load_candidate_factory_from_source,
)


@dataclass
class LeviRunResult:
    """Stable result shape consumed by the repo's portfolio/reporting code."""

    best_program: str
    best_score: float
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    total_evaluations: int = 0
    total_cost: float = 0.0
    archive_size: int = 0
    runtime_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def code(self) -> str:
        return self.best_program

    @property
    def best_code(self) -> str:
        return self.best_program


class LeviScoreFunction:
    """Picklable score function adapter used by Levi worker processes."""

    def __init__(
        self,
        evaluate_factory: Callable[[Callable[..., object]], EvaluatorResult],
        evaluate_source: Callable[[str], EvaluatorResult] | None = None,
        score_metric: str = "combined_score",
    ) -> None:
        self._evaluate_factory = evaluate_factory
        self._evaluate_source = evaluate_source
        self._score_metric = score_metric

    def __call__(self, factory: Callable[..., object], inputs=None) -> dict[str, float]:
        source = _candidate_source(factory, inputs)
        if self._evaluate_source is not None:
            if source is None:
                return {
                    "score": 0.0,
                    "combined_score": 0.0,
                    "complexity_source_missing": 1.0,
                }
            result = self._evaluate_source(source)
        else:
            result = self._evaluate_factory(factory)
        metrics = result.metrics or {}
        score = metrics.get(self._score_metric, 0.0)
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            score = 0.0

        levi_metrics = {"score": float(score)}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                levi_metrics[key] = float(value)
        return levi_metrics


class LeviRunner:
    """Coordinates execution through Levi's evolve_code API."""

    def __init__(
        self,
        evolve_code: Callable[..., Any],
        evaluator_path: Path,
        problem_description: str,
        function_signature: str,
    ) -> None:
        self._evolve_code = evolve_code
        self._evaluator_path = evaluator_path
        self._problem_description = problem_description
        self._function_signature = function_signature
        self._evaluate_factory = self._load_evaluate_factory(evaluator_path)
        self._evaluate_source = self._load_evaluate_source(evaluator_path)

    def run(self, program_path: Path, config) -> LeviRunResult:
        seed_program = program_path.read_text(encoding="utf-8")
        score_fn = LeviScoreFunction(self._evaluate_factory, self._evaluate_source)
        kwargs = config.evolve_kwargs()
        kwargs["function_signature"] = self._function_signature or getattr(
            config, "function_signature", ""
        )
        kwargs["seed_program"] = seed_program
        kwargs["score_fn"] = score_fn

        result = self._evolve_code(
            getattr(config, "problem_description", None) or self._problem_description,
            **kwargs,
        )

        best_program = getattr(result, "best_program", "")
        evaluation = self._evaluate_best_program(best_program)
        metrics = evaluation.metrics
        artifacts = evaluation.artifacts
        metadata = {
            "levi_total_cost": getattr(result, "total_cost", 0.0),
            "levi_runtime_seconds": getattr(result, "runtime_seconds", 0.0),
        }

        return LeviRunResult(
            best_program=best_program,
            best_score=float(
                getattr(result, "best_score", metrics.get("combined_score", 0.0))
            ),
            metrics=metrics,
            artifacts=artifacts,
            total_evaluations=int(getattr(result, "total_evaluations", 0) or 0),
            total_cost=float(getattr(result, "total_cost", 0.0) or 0.0),
            archive_size=int(getattr(result, "archive_size", 0) or 0),
            runtime_seconds=float(getattr(result, "runtime_seconds", 0.0) or 0.0),
            metadata=metadata,
        )

    def _evaluate_best_program(self, source: str) -> EvaluatorResult:
        try:
            factory = load_candidate_factory_from_source(source)
        except Exception as exc:
            return EvaluatorResult(
                metrics={
                    "combined_score": 0.0,
                    "error": "failed to load Levi best program",
                },
                artifacts={"error_type": type(exc).__name__, "error_message": str(exc)},
            )
        if self._evaluate_source is not None:
            return self._evaluate_source(source)
        return self._evaluate_factory(factory)

    def _load_evaluate_factory(
        self, evaluator_path: Path
    ) -> Callable[[Callable[..., object]], EvaluatorResult]:
        module_name = _module_name_from_package_path(evaluator_path)
        if module_name is not None:
            module = importlib.import_module(module_name)
            evaluate_factory = getattr(module, "evaluate_factory", None)
            if callable(evaluate_factory):
                return evaluate_factory

        spec = importlib.util.spec_from_file_location(
            f"randomize_evolve_levi_evaluator_{abs(hash(evaluator_path.resolve()))}",
            evaluator_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load evaluator from {evaluator_path}")
        module = importlib.util.module_from_spec(spec)
        _exec_registered_module(module, lambda: spec.loader.exec_module(module))  # type: ignore[call-arg]
        evaluate_factory = getattr(module, "evaluate_factory", None)
        if not callable(evaluate_factory):
            raise AttributeError(
                f"{evaluator_path} must expose evaluate_factory(factory)"
            )
        return evaluate_factory

    def _load_evaluate_source(
        self, evaluator_path: Path
    ) -> Callable[[str], EvaluatorResult] | None:
        module_name = _module_name_from_package_path(evaluator_path)
        if module_name is not None:
            module = importlib.import_module(module_name)
            evaluate_source = getattr(module, "evaluate_source", None)
            return evaluate_source if callable(evaluate_source) else None

        spec = importlib.util.spec_from_file_location(
            f"randomize_evolve_levi_evaluator_{abs(hash(evaluator_path.resolve()))}",
            evaluator_path,
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        _exec_registered_module(module, lambda: spec.loader.exec_module(module))  # type: ignore[call-arg]
        evaluate_source = getattr(module, "evaluate_source", None)
        return evaluate_source if callable(evaluate_source) else None


def _candidate_source(factory: Callable[..., object], inputs: Any) -> str | None:
    """Recover the candidate source Levi attaches while evaluating code."""

    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, dict):
        for key in ("source", "code", "program", "candidate_source"):
            value = inputs.get(key)
            if isinstance(value, str):
                return value

    for attr in ("__source__", "source", "code"):
        value = getattr(factory, attr, None)
        if isinstance(value, str):
            return value

    globals_dict = getattr(factory, "__globals__", {})
    if isinstance(globals_dict, dict):
        value = globals_dict.get("__source_code__")
        if isinstance(value, str):
            return value
    return None


def _module_name_from_package_path(path: Path) -> str | None:
    """Return an importable module name for a file inside a Python package."""

    resolved = path.resolve()
    if resolved.suffix != ".py" or not resolved.exists():
        return None

    parts = [resolved.stem]
    parent = resolved.parent
    while (parent / "__init__.py").exists():
        parts.append(parent.name)
        parent = parent.parent

    if len(parts) == 1:
        return None
    return ".".join(reversed(parts))
