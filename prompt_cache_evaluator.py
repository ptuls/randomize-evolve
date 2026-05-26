"""OpenEvolve evaluation entry point for prompt-cache compaction experiments."""

import math

from openevolve.evaluation_result import EvaluationResult

from randomize_evolve.evaluator_entry import EvaluationEntryPoint, score_to_reward
from randomize_evolve.evaluators.prompt_cache_compaction import (
    EvaluationResult as PromptCacheEvaluationResult,
)
from randomize_evolve.evaluators.prompt_cache_compaction import (
    Evaluator,
    EvaluatorConfig,
)

EVALUATION_TIMEOUT_S = 45

DEFAULT_CONFIG = EvaluatorConfig(
    seeds=(7, 19, 41),
    prompt_build_timeout_s=0.25,
    cache_ttl_s=300.0,
    cache_write_multiplier=1.25,
    cache_read_multiplier=0.10,
    required_coverage_penalty=100.0,
    cache_miss_weight=100.0,
    prefix_regret_weight=250.0,
    dynamic_regret_weight=150.0,
    token_weight=0.1,
    cached_cost_weight=1.0,
    latency_weight=2.0,
)


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate a candidate module using the prompt-cache compaction evaluator."""
    return _ENTRY_POINT.evaluate(program_path)


def _success_result(result: PromptCacheEvaluationResult) -> EvaluationResult:
    reliability = len(result.trials) / len(DEFAULT_CONFIG.seeds)
    combined_score = score_to_reward(result.score)
    if not result.success:
        combined_score *= 0.7

    metrics = {
        "combined_score": combined_score,
        "reliability": reliability,
        "required_coverage": result.required_coverage,
        "prefix_ratio": result.prefix_ratio,
        "oracle_prefix_ratio": result.oracle_prefix_ratio,
        "dynamic_offset_ratio": result.dynamic_offset_ratio,
        "oracle_dynamic_offset_ratio": result.oracle_dynamic_offset_ratio,
        "cache_hit_rate": result.cache_hit_rate,
        "naive_cache_hit_rate": result.naive_cache_hit_rate,
        "oracle_cache_hit_rate": result.oracle_cache_hit_rate,
        "avg_prompt_tokens": result.avg_prompt_tokens,
        "avg_naive_prompt_tokens": result.avg_naive_prompt_tokens,
        "avg_cached_input_cost": result.avg_cached_input_cost,
        "avg_naive_input_cost": result.avg_naive_input_cost,
        "avg_oracle_cached_input_cost": result.avg_oracle_cached_input_cost,
        "avg_oracle_prompt_tokens": result.avg_oracle_prompt_tokens,
        "oracle_score": result.oracle_score,
        "score_minus_oracle_score": result.score_minus_oracle_score,
        "avg_build_time_ms": result.avg_build_time_ms,
    }

    artifacts = {
        "errors": result.error or "",
        "score_breakdown": (
            f"coverage={result.required_coverage:.3f}, "
            f"prefix={result.prefix_ratio:.3f}/{result.oracle_prefix_ratio:.3f}, "
            f"dynamic_offset={result.dynamic_offset_ratio:.3f}/{result.oracle_dynamic_offset_ratio:.3f}, "
            f"cache_hit_rate={result.cache_hit_rate:.3f} vs naive {result.naive_cache_hit_rate:.3f} "
            f"(oracle {result.oracle_cache_hit_rate:.3f}), "
            f"cached_input_cost={result.avg_cached_input_cost:.1f} vs naive {result.avg_naive_input_cost:.1f} "
            f"(oracle {result.avg_oracle_cached_input_cost:.1f}), "
            f"score_delta_vs_oracle={result.score_minus_oracle_score:.2f}, "
            f"tokens={result.avg_prompt_tokens:.1f} vs naive {result.avg_naive_prompt_tokens:.1f}, "
            f"build={result.avg_build_time_ms:.3f}ms, "
            f"raw_score={result.score:.2f}"
        ),
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _error_result(message: str, artifacts: dict) -> EvaluationResult:
    metrics = {
        "combined_score": 0.0,
        "reliability": 0.0,
        "required_coverage": 0.0,
        "prefix_ratio": 0.0,
        "oracle_prefix_ratio": 1.0,
        "dynamic_offset_ratio": 0.0,
        "oracle_dynamic_offset_ratio": 1.0,
        "cache_hit_rate": 0.0,
        "naive_cache_hit_rate": 0.0,
        "oracle_cache_hit_rate": 1.0,
        "avg_prompt_tokens": math.inf,
        "avg_naive_prompt_tokens": math.inf,
        "avg_cached_input_cost": math.inf,
        "avg_naive_input_cost": math.inf,
        "avg_oracle_cached_input_cost": math.inf,
        "avg_oracle_prompt_tokens": math.inf,
        "oracle_score": math.inf,
        "score_minus_oracle_score": math.inf,
        "avg_build_time_ms": math.inf,
        "error": message,
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


_ENTRY_POINT = EvaluationEntryPoint(
    evaluator_factory=lambda: Evaluator(DEFAULT_CONFIG),
    timeout_seconds=EVALUATION_TIMEOUT_S,
    load_error_suggestion=(
        "Ensure the module defines `candidate_factory(key_bits, capacity)` "
        "optionally accepting `tasks` or `corpus_hint`, "
        "or `build_candidate(key_bits, capacity)` and returns an object "
        "implementing build_prompt(task) with optional `corpus_hint` support."
    ),
    timeout_suggestion="Inspect the candidate implementation for long-running prompt assembly.",
    success_result_builder=_success_result,
    error_result_builder=_error_result,
)
