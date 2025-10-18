"""OpenEvolve entry points for the Bloom filter alternative evaluator.

The functions defined here adapt ``BloomAlternativeEvaluator`` so that the
OpenEvolve orchestration layer can invoke the evaluation with standard
timeouts, structured metrics, and cascade stages.
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import math
import traceback
from pathlib import Path
from types import ModuleType
from typing import Callable

from openevolve.evaluation_result import EvaluationResult

from randomize_evolve.evaluators.bloom_alternatives import (
    BloomAlternativeEvaluator,
    EvaluatorConfig,
    EvaluationResult as BloomEvaluationResult,
)

# Overall timeouts safeguard the evaluation entry points; the evaluator itself
# enforces per-stage build and query limits.
EVALUATION_TIMEOUT_S = 60
STAGE1_TIMEOUT_S = 20

# Configurations tuned to track the YAML defaults while providing a faster
# cascade stage for early filtering.
DEFAULT_CONFIG = EvaluatorConfig(
    key_bits=32,
    positives=5000,
    queries=10000,
    negative_fraction=0.5,
    seeds=(17, 23, 71, 89, 131),
    build_timeout_s=1.0,
    query_timeout_s=1.0,
    false_negative_penalty=1_000_000.0,
    false_positive_weight=180.0,
    memory_weight=0.05,
    latency_weight=8.0,
    max_memory_bytes=100 * 1024 * 1024,
)

STAGE1_CONFIG = EvaluatorConfig(
    key_bits=32,
    positives=2000,
    queries=4000,
    negative_fraction=0.5,
    seeds=(17, 23),
    build_timeout_s=0.75,
    query_timeout_s=0.75,
    false_negative_penalty=1_000_000.0,
    false_positive_weight=150.0,
    memory_weight=0.05,
    latency_weight=8.0,
    max_memory_bytes=100 * 1024 * 1024,
)


CandidateFactory = Callable[[int, int], object]


def run_with_timeout(
    func: Callable[..., BloomEvaluationResult],
    *args,
    timeout_seconds: float,
    **kwargs,
) -> BloomEvaluationResult:
    """Execute ``func`` with a wall-clock timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:  # pragma: no cover - best effort
            future.cancel()
            raise TimeoutError(
                f"evaluation exceeded {timeout_seconds}s wall-clock limit"
            ) from exc


def evaluate(program_path: str) -> EvaluationResult:
    """Main evaluation entry point used by OpenEvolve."""
    return _evaluate_with_config(
        program_path=program_path,
        config=DEFAULT_CONFIG,
        timeout_seconds=EVALUATION_TIMEOUT_S,
        stage_name="stage2",
    )


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Cascade stage 1 evaluation with a reduced workload."""
    return _evaluate_with_config(
        program_path=program_path,
        config=STAGE1_CONFIG,
        timeout_seconds=STAGE1_TIMEOUT_S,
        stage_name="stage1",
    )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Cascade stage 2 reuses the full evaluation."""
    return evaluate(program_path)


def _evaluate_with_config(
    program_path: str,
    *,
    config: EvaluatorConfig,
    timeout_seconds: float,
    stage_name: str,
) -> EvaluationResult:
    try:
        factory = _load_candidate_factory(program_path)
    except Exception as exc:  # pragma: no cover - defensive programming
        message = f"{stage_name}: failed to load candidate factory - {exc}"
        artifacts = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "full_traceback": traceback.format_exc(),
            "suggestion": (
                "Ensure the candidate module defines a callable named "
                "`candidate_factory(key_bits, capacity)`."
            ),
        }
        return _error_result(message, {**artifacts, "stage": stage_name})

    evaluator = BloomAlternativeEvaluator(config)

    try:
        bloom_result = run_with_timeout(
            evaluator, factory, timeout_seconds=timeout_seconds
        )
    except TimeoutError as exc:
        artifacts = {
            "error_type": "TimeoutError",
            "error_message": str(exc),
            "suggestion": (
                "Review build/query complexity or tighten the candidate's "
                "resource usage to satisfy evaluator limits."
            ),
        }
        return _error_result(
            f"{stage_name}: evaluation timed out",
            {**artifacts, "stage": stage_name},
        )
    except Exception as exc:  # pragma: no cover - defensive: capture evaluator bugs
        artifacts = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Unexpected evaluator failure; inspect traceback.",
        }
        return _error_result(
            f"{stage_name}: evaluation failed",
            {**artifacts, "stage": stage_name},
        )

    return _success_result(bloom_result, config, stage_name)


def _load_candidate_factory(program_path: str) -> CandidateFactory:
    path = Path(program_path)
    if not path.exists():
        raise FileNotFoundError(f"program path {path} does not exist")

    spec = importlib.util.spec_from_file_location("candidate_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    loader.exec_module(module)  # type: ignore[call-arg]

    factory = _extract_factory(module)
    if not callable(factory):
        raise TypeError("candidate_factory must be callable")
    return factory


def _extract_factory(module: ModuleType) -> CandidateFactory:
    if hasattr(module, "candidate_factory"):
        return getattr(module, "candidate_factory")  # type: ignore[return-value]
    if hasattr(module, "build_candidate"):
        return getattr(module, "build_candidate")  # type: ignore[return-value]
    raise AttributeError(
        "candidate module must expose `candidate_factory` or `build_candidate`"
    )


def _success_result(
    bloom_result: BloomEvaluationResult,
    config: EvaluatorConfig,
    stage_name: str,
) -> EvaluationResult:
    total_trials = len(config.seeds)
    successful_trials = len(bloom_result.trials)
    reliability = successful_trials / total_trials if total_trials else 0.0

    combined_score = _score_to_reward(bloom_result.score)
    if not bloom_result.success:
        combined_score *= 0.7  # penalise partial failures

    metrics = {
        "combined_score": combined_score,
        "reliability": reliability,
        "false_positive_rate": bloom_result.false_positive_rate,
        "false_negative_rate": bloom_result.false_negative_rate,
        "mean_build_time_ms": bloom_result.mean_build_time_ms,
        "mean_query_time_ms": bloom_result.mean_query_time_ms,
        "mean_peak_memory_bytes": bloom_result.mean_peak_memory_bytes,
    }

    artifacts = {
        "stage": stage_name,
        "trial_count": successful_trials,
        "errors": bloom_result.error or "",
        "score_breakdown": (
            f"fp_rate={bloom_result.false_positive_rate:.4f}, "
            f"fn_rate={bloom_result.false_negative_rate:.4f}, "
            f"memory={bloom_result.mean_peak_memory_bytes:.0f}B, "
            f"build={bloom_result.mean_build_time_ms:.2f}ms, "
            f"query={bloom_result.mean_query_time_ms:.2f}ms, "
            f"raw_score={bloom_result.score:.2f}"
        ),
    }

    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _error_result(message: str, artifacts: dict) -> EvaluationResult:
    metrics = {
        "combined_score": 0.0,
        "reliability": 0.0,
        "false_positive_rate": 1.0,
        "false_negative_rate": 1.0,
        "mean_build_time_ms": math.inf,
        "mean_query_time_ms": math.inf,
        "mean_peak_memory_bytes": math.inf,
        "error": message,
    }
    artifacts = {"stage": artifacts.get("stage", "error"), **artifacts}
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _score_to_reward(score: float) -> float:
    if not math.isfinite(score):
        return 0.0
    return 1.0 / (1.0 + max(score, 0.0))
