"""OpenEvolve evaluation entry point for packet-switching schedulers."""

import math

from openevolve.evaluation_result import EvaluationResult

from randomize_evolve.evaluator_entry import EvaluationEntryPoint, score_to_reward
from randomize_evolve.evaluators.packet_switching import (
    PacketSwitchingEvaluation,
    PacketSwitchingEvaluator,
    PacketSwitchingEvaluatorConfig,
)

EVALUATION_TIMEOUT_S = 75

DEFAULT_CONFIG = PacketSwitchingEvaluatorConfig()


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate a candidate module using the packet-switching evaluator."""
    return _ENTRY_POINT.evaluate(program_path)


def _success_result(packet_result: PacketSwitchingEvaluation) -> EvaluationResult:
    scenario_count = len(packet_result.scenario_results)
    total_scenarios = len(DEFAULT_CONFIG.scenarios)
    reliability = scenario_count / total_scenarios if total_scenarios else 0.0

    combined_score = score_to_reward(packet_result.score)
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


def _average(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


_ENTRY_POINT = EvaluationEntryPoint(
    evaluator_factory=lambda: PacketSwitchingEvaluator(DEFAULT_CONFIG),
    timeout_seconds=EVALUATION_TIMEOUT_S,
    load_error_suggestion=(
        "Ensure the module defines `candidate_factory(ports)` or "
        "`build_candidate(ports)` and returns an object implementing "
        "`select_matches(requests, time_slot, queue_lengths)`."
    ),
    timeout_suggestion="Inspect the scheduler for long-running matching logic.",
    success_result_builder=_success_result,
    error_result_builder=_error_result,
)
