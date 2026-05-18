"""OpenEvolve evaluation entry point for packet-switching schedulers."""

import concurrent.futures
import importlib.util
import math
import traceback
from pathlib import Path
from types import ModuleType
from typing import Callable

from openevolve.evaluation_result import EvaluationResult

from randomize_evolve.evaluators.packet_switching import (
    PacketSwitchingEvaluation,
    PacketSwitchingEvaluator,
    PacketSwitchingEvaluatorConfig,
)

EVALUATION_TIMEOUT_S = 75

DEFAULT_CONFIG = PacketSwitchingEvaluatorConfig()

SchedulerFactory = Callable[[int], object]


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate a candidate module using the packet-switching evaluator."""
    try:
        factory = _load_candidate_factory(program_path)
    except Exception as exc:  # pragma: no cover - defensive
        artifacts = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "full_traceback": traceback.format_exc(),
            "suggestion": (
                "Ensure the module defines `candidate_factory(ports)` or "
                "`build_candidate(ports)` and returns an object implementing "
                "`select_matches(requests, time_slot, queue_lengths)`."
            ),
        }
        return _error_result("failed to load candidate factory", artifacts)

    evaluator = PacketSwitchingEvaluator(DEFAULT_CONFIG)

    try:
        packet_result = _run_with_timeout(
            evaluator,
            factory,
            timeout_seconds=EVALUATION_TIMEOUT_S,
        )
    except TimeoutError as exc:
        artifacts = {
            "error_type": "TimeoutError",
            "error_message": str(exc),
            "suggestion": "Inspect the scheduler for long-running matching logic.",
        }
        return _error_result("evaluation timed out", artifacts)
    except Exception as exc:  # pragma: no cover - defensive
        artifacts = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Unexpected evaluator failure; inspect the traceback.",
        }
        return _error_result("evaluation failed", artifacts)

    return _success_result(packet_result)


def _run_with_timeout(
    func: Callable[..., PacketSwitchingEvaluation],
    *args,
    timeout_seconds: float,
    **kwargs,
) -> PacketSwitchingEvaluation:
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


def _load_candidate_factory(program_path: str) -> SchedulerFactory:
    path = Path(program_path)
    if not path.exists():
        raise FileNotFoundError(f"program path {path} does not exist")

    spec = importlib.util.spec_from_file_location("candidate_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]

    factory = _extract_factory(module)
    if not callable(factory):
        raise TypeError("candidate_factory must be callable")
    return factory


def _extract_factory(module: ModuleType) -> SchedulerFactory:
    if hasattr(module, "candidate_factory"):
        return getattr(module, "candidate_factory")  # type: ignore[return-value]
    if hasattr(module, "build_candidate"):
        return getattr(module, "build_candidate")  # type: ignore[return-value]
    raise AttributeError("candidate module must expose `candidate_factory` or `build_candidate`")


def _success_result(packet_result: PacketSwitchingEvaluation) -> EvaluationResult:
    scenario_count = len(packet_result.scenario_results)
    total_scenarios = len(DEFAULT_CONFIG.scenarios)
    reliability = scenario_count / total_scenarios if total_scenarios else 0.0

    combined_score = _score_to_reward(packet_result.score)
    if not packet_result.success:
        combined_score *= 0.7

    mean_throughput = _average(
        result.metrics.throughput for result in packet_result.scenario_results
    )
    mean_input_fairness = _average(
        result.metrics.fairness_inputs for result in packet_result.scenario_results
    )
    mean_flow_fairness = _average(
        result.metrics.fairness_flows for result in packet_result.scenario_results
    )
    mean_drop_rate = _average(
        result.metrics.drop_rate for result in packet_result.scenario_results
    )
    mean_queue = _average(
        result.metrics.average_queue for result in packet_result.scenario_results
    )

    metrics = {
        "combined_score": combined_score,
        "reliability": reliability,
        "mean_throughput": mean_throughput,
        "mean_input_fairness": mean_input_fairness,
        "mean_flow_fairness": mean_flow_fairness,
        "mean_drop_rate": mean_drop_rate,
        "mean_average_queue": mean_queue,
    }

    artifacts = {
        "score_breakdown": (
            f"throughput={mean_throughput:.3f}, "
            f"input_fairness={mean_input_fairness:.3f}, "
            f"flow_fairness={mean_flow_fairness:.3f}, "
            f"drop_rate={mean_drop_rate:.3f}, "
            f"avg_queue={mean_queue:.2f}, "
            f"raw_score={packet_result.score:.4f}"
        ),
        "scenario_scores": {
            result.config.name: {
                "normalized_score": result.score,
                "throughput": result.metrics.throughput,
                "fairness_inputs": result.metrics.fairness_inputs,
                "fairness_flows": result.metrics.fairness_flows,
                "drop_rate": result.metrics.drop_rate,
                "average_queue": result.metrics.average_queue,
            }
            for result in packet_result.scenario_results
        },
    }

    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _error_result(message: str, artifacts: dict) -> EvaluationResult:
    metrics = {
        "combined_score": 0.0,
        "reliability": 0.0,
        "mean_throughput": 0.0,
        "mean_input_fairness": 0.0,
        "mean_flow_fairness": 0.0,
        "mean_drop_rate": 1.0,
        "mean_average_queue": math.inf,
        "error": message,
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _score_to_reward(score: float) -> float:
    return 0.0 if not math.isfinite(score) else 1.0 / (1.0 + max(score, 0.0))


def _average(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
