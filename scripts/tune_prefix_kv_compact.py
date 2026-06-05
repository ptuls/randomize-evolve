"""Tune the compact deployable prefix KV-cache policy on validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import random

from randomize_evolve.evaluators.prefix_kv_cache import (
    EvaluationResult,
    EvaluatorConfig,
    PrefixKVCacheEvaluator,
)


@dataclass(frozen=True)
class Parameters:
    """Compact policy coefficients."""

    hit_weight: float
    admission_frequency: float
    admission_priority: float
    admission_descendant: float
    admission_deep_penalty: float
    admission_cost: float
    admission_cost_divisor: float
    admission_threshold: float
    admission_depth_threshold: float
    eviction_age: float
    eviction_frequency: float
    eviction_descendant: float
    eviction_priority: float
    frequency_half_life: float | None
    priority_half_life: float | None


DEFAULT_PARAMETERS = Parameters(
    hit_weight=2.5,
    admission_frequency=0.95,
    admission_priority=0.2,
    admission_descendant=0.35,
    admission_deep_penalty=0.2,
    admission_cost=1.5,
    admission_cost_divisor=96.0,
    admission_threshold=0.7,
    admission_depth_threshold=0.24,
    eviction_age=0.85,
    eviction_frequency=1.8,
    eviction_descendant=0.2,
    eviction_priority=0.55,
    frequency_half_life=12.0,
    priority_half_life=1.5,
)


class TunableCompactPolicy:
    """Parameter-search equivalent of the deployable compact policy."""

    def __init__(self, parameters: Parameters) -> None:
        self.parameters = parameters
        self._state: dict[int, tuple[float, float, int]] = {}
        self._current_priority = 0

    def on_request_start(self, request, now: int) -> None:
        self._current_priority = max(0, request.priority)

    def on_cache_hit(self, block, request, now: int) -> None:
        self._observe(block, self.parameters.hit_weight, now)

    def on_cache_miss(self, block, request, now: int) -> None:
        self._observe(block, 1.0, now)

    def score_admission(self, block, now: int) -> float:
        parameters = self.parameters
        frequency, priority = self._values(block.prefix_hash, now)
        structure = parameters.admission_descendant * math.log1p(block.descendant_count)
        if block.depth >= 5:
            structure -= parameters.admission_deep_penalty * (block.depth - 4)
        value = (
            parameters.admission_frequency * math.log1p(frequency)
            + parameters.admission_priority * priority
            + structure
            + parameters.admission_cost
            * math.log1p(
                block.estimated_recompute_cost / parameters.admission_cost_divisor
            )
        )
        return (
            value
            - parameters.admission_threshold
            - parameters.admission_depth_threshold * max(0, block.depth - 2)
        )

    def score_eviction(self, block, now: int) -> float:
        parameters = self.parameters
        frequency, priority = self._values(block.prefix_hash, now)
        return (
            parameters.eviction_age * math.log1p(max(0, now - block.last_accessed_at))
            - parameters.eviction_frequency * math.log1p(frequency + block.hit_count)
            - parameters.eviction_descendant * math.log1p(block.descendant_count)
            - parameters.eviction_priority * priority
        )

    def _observe(self, block, weight: float, now: int) -> None:
        key = block.prefix_hash
        frequency, priority = self._values(key, now)
        self._state[key] = (
            frequency + weight,
            max(priority, self._current_priority),
            now,
        )

    def _values(self, key: int, now: int) -> tuple[float, float]:
        parameters = self.parameters
        frequency, priority, observed_at = self._state.get(key, (0.0, 0.0, now))
        elapsed = max(0, now - observed_at)
        if parameters.frequency_half_life is not None:
            frequency *= 2.0 ** (-elapsed / parameters.frequency_half_life)
        if parameters.priority_half_life is not None:
            priority *= 2.0 ** (-elapsed / parameters.priority_half_life)
        self._state[key] = (frequency, priority, now)
        return frequency, priority


def _sample_parameters(rng: random.Random) -> Parameters:
    return Parameters(
        hit_weight=rng.choice((1.5, 2.0, 2.5, 3.0)),
        admission_frequency=rng.choice((0.7, 0.95, 1.2, 1.5)),
        admission_priority=rng.choice((0.2, 0.45, 0.7, 1.0)),
        admission_descendant=rng.choice((0.15, 0.35, 0.55, 0.8)),
        admission_deep_penalty=rng.choice((0.1, 0.2, 0.3, 0.4)),
        admission_cost=rng.choice((0.6, 0.9, 1.2, 1.5)),
        admission_cost_divisor=rng.choice((48.0, 64.0, 96.0, 128.0)),
        admission_threshold=rng.choice((0.7, 1.0, 1.3, 1.6)),
        admission_depth_threshold=rng.choice((0.12, 0.18, 0.24, 0.3)),
        eviction_age=rng.choice((0.5, 0.85, 1.2, 1.5)),
        eviction_frequency=rng.choice((0.6, 0.95, 1.3, 1.8)),
        eviction_descendant=rng.choice((0.2, 0.4, 0.7, 1.0)),
        eviction_priority=rng.choice((0.3, 0.55, 0.8, 1.2)),
        frequency_half_life=rng.choice((8.0, 10.0, 12.0, 16.0, 24.0)),
        priority_half_life=rng.choice((1.5, 3.0, 6.0, 12.0, 24.0)),
    )


def _evaluate_result(
    parameters: Parameters,
    config: EvaluatorConfig,
    *,
    splits: tuple[str, ...] = ("validation",),
) -> EvaluationResult:
    evaluator = PrefixKVCacheEvaluator(config, splits=splits)
    return evaluator(lambda *_: TunableCompactPolicy(parameters))


def _evaluate(parameters: Parameters, config: EvaluatorConfig) -> float:
    return _evaluate_result(parameters, config).combined_score


def _run_decay_ablation(
    frequency_half_life: float,
    priority_half_life: float,
) -> None:
    config = EvaluatorConfig(capacity_sweep_blocks=(24, 48))
    variants = {
        "no_decay": replace(
            DEFAULT_PARAMETERS,
            frequency_half_life=None,
            priority_half_life=None,
        ),
        "frequency_decay_only": replace(
            DEFAULT_PARAMETERS,
            frequency_half_life=frequency_half_life,
            priority_half_life=None,
        ),
        "priority_decay_only": replace(
            DEFAULT_PARAMETERS,
            frequency_half_life=None,
            priority_half_life=priority_half_life,
        ),
        "both_decays": replace(
            DEFAULT_PARAMETERS,
            frequency_half_life=frequency_half_life,
            priority_half_life=priority_half_life,
        ),
    }
    for name, parameters in variants.items():
        result = _evaluate_result(parameters, config)
        validation = result.split_metrics["validation"]
        print(
            json.dumps(
                {
                    "variant": name,
                    "combined_score_without_complexity": result.combined_score,
                    "token_hit_rate": validation["token_hit_rate"],
                    "worst_quarter_token_hit_rate": validation[
                        "worst_quarter_token_hit_rate"
                    ],
                    "wasted_admission_token_rate": validation[
                        "wasted_admission_token_rate"
                    ],
                    "admission_token_utility": validation["admission_token_utility"],
                    "avoidable_eviction_rate": validation["avoidable_eviction_rate"],
                    "cache_churn_per_1k": validation["cache_churn_per_1k"],
                    "frequency_half_life": parameters.frequency_half_life,
                    "priority_half_life": parameters.priority_half_life,
                },
                sort_keys=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--full-top", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--decay-ablation", action="store_true")
    parser.add_argument("--frequency-half-life", type=float, default=12.0)
    parser.add_argument("--priority-half-life", type=float, default=1.5)
    args = parser.parse_args()

    if args.decay_ablation:
        _run_decay_ablation(args.frequency_half_life, args.priority_half_life)
        return

    rng = random.Random(args.seed)
    quick_config = EvaluatorConfig(
        request_count=48,
        seeds=(3,),
        capacity_sweep_blocks=(24, 48),
    )
    sampled = [
        (_evaluate(parameters, quick_config), parameters)
        for parameters in (_sample_parameters(rng) for _ in range(args.samples))
    ]

    full_config = EvaluatorConfig(capacity_sweep_blocks=(24, 48))
    finalists = [
        (_evaluate(parameters, full_config), quick_score, parameters)
        for quick_score, parameters in sorted(
            sampled, key=lambda item: item[0], reverse=True
        )[: args.full_top]
    ]
    for full_score, quick_score, parameters in sorted(
        finalists, key=lambda item: item[0], reverse=True
    ):
        print(
            json.dumps(
                {
                    "full_score_without_complexity": full_score,
                    "quick_score_without_complexity": quick_score,
                    "parameters": asdict(parameters),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
