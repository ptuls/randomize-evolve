"""Initial prefix KV-cache scoring policy for Levi evolution."""

from __future__ import annotations

from randomize_evolve.evaluators.prefix_kv_cache import (
    EvaluatorConfig,
    PrefixBlockInfo,
    PrefixKVCacheEvaluator,
    RequestInfo,
)


class HybridPrefixPolicy:
    """Small stateless heuristic balancing reuse, depth, age, and recompute cost."""

    def __init__(
        self,
        capacity_blocks: int,
        block_size_tokens: int,
        seed: int | None = None,
    ) -> None:
        self.capacity_blocks = capacity_blocks
        self.block_size_tokens = block_size_tokens
        self.seed = seed
        self._current_priority = 0

    def on_request_start(self, request: RequestInfo, now: int) -> None:
        self._current_priority = request.priority

    def on_cache_hit(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        return None

    def on_cache_miss(self, block: PrefixBlockInfo, request: RequestInfo, now: int) -> None:
        return None

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        # EVOLVE-BLOCK-START admission_formula
        age = max(0, now - block.created_at)
        shallow_bonus = 1.0 / (1.0 + block.depth)
        reuse_hint = min(4.0, float(block.hit_count + block.descendant_count))
        return 0.35 + shallow_bonus + 0.08 * reuse_hint - 0.002 * age
        # EVOLVE-BLOCK-END

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        # EVOLVE-BLOCK-START eviction_formula
        age_since_access = max(0, now - block.last_accessed_at)
        reuse_pressure = 2.0 * block.hit_count + 0.75 * block.descendant_count
        recompute_pressure = 0.15 * block.estimated_recompute_cost
        depth_penalty = 0.2 * block.depth
        return age_since_access + depth_penalty - reuse_pressure - recompute_pressure
        # EVOLVE-BLOCK-END


def build_candidate(
    capacity_blocks: int,
    block_size_tokens: int,
    seed: int | None = None,
) -> HybridPrefixPolicy:
    return HybridPrefixPolicy(capacity_blocks, block_size_tokens, seed)


candidate_factory = build_candidate


def run_demo() -> None:
    config = EvaluatorConfig(request_count=32, seeds=(3,), capacity_blocks=16)
    evaluator = PrefixKVCacheEvaluator(config)
    result = evaluator(build_candidate, scoring_fn_complexity=0)
    print(f"combined_score={result.combined_score:.3f}")
    for split, metrics in result.split_metrics.items():
        print(
            f"{split}: token_hit_rate={metrics['token_hit_rate']:.3f}, "
            f"block_hit_rate={metrics['block_hit_rate']:.3f}, "
            f"p95_latency={metrics['p95_latency_proxy']:.2f}"
        )


if __name__ == "__main__":
    run_demo()
