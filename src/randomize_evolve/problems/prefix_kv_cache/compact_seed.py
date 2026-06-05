"""Compact deployable prefix KV-cache policy."""

import math

DEFAULT_FREQUENCY_HALF_LIFE = 12.0
DEFAULT_PRIORITY_HALF_LIFE = 1.5


class CompactReusePolicy:
    """Tracks time-decayed observed reuse and priority without future knowledge."""

    def __init__(
        self,
        capacity_blocks,
        block_size_tokens,
        seed=None,
        frequency_half_life=DEFAULT_FREQUENCY_HALF_LIFE,
        priority_half_life=DEFAULT_PRIORITY_HALF_LIFE,
    ):
        self._state = {}
        self._current_priority = 0
        self._frequency_half_life = frequency_half_life
        self._priority_half_life = priority_half_life

    def on_request_start(self, request, now):
        self._current_priority = max(0, request.priority)

    def on_cache_hit(self, block, request, now):
        self._observe(block, 2.5, now)

    def on_cache_miss(self, block, request, now):
        self._observe(block, 1.0, now)

    def score_admission(self, block, now):
        frequency, priority = self._values(block.prefix_hash, now)
        structure = 0.35 * math.log1p(block.descendant_count)
        if block.depth >= 5:
            structure -= 0.2 * (block.depth - 4)
        value = (
            0.95 * math.log1p(frequency)
            + 0.2 * priority
            + structure
            + 1.5 * math.log1p(block.estimated_recompute_cost / 96.0)
        )
        return value - 0.7 - 0.24 * max(0, block.depth - 2)

    def score_eviction(self, block, now):
        frequency, priority = self._values(block.prefix_hash, now)
        return (
            0.85 * math.log1p(max(0, now - block.last_accessed_at))
            - 1.8 * math.log1p(frequency + block.hit_count)
            - 0.2 * math.log1p(block.descendant_count)
            - 0.55 * priority
        )

    def _observe(self, block, weight, now):
        key = block.prefix_hash
        frequency, priority = self._values(key, now)
        self._state[key] = (
            frequency + weight,
            max(priority, self._current_priority),
            now,
        )

    def _values(self, key, now):
        frequency, priority, observed_at = self._state.get(key, (0.0, 0.0, now))
        elapsed = max(0, now - observed_at)
        if self._frequency_half_life is not None:
            frequency *= 2.0 ** (-elapsed / self._frequency_half_life)
        if self._priority_half_life is not None:
            priority *= 2.0 ** (-elapsed / self._priority_half_life)
        self._state[key] = (frequency, priority, now)
        return frequency, priority


def build_candidate(capacity_blocks, block_size_tokens, seed=None):
    return CompactReusePolicy(capacity_blocks, block_size_tokens, seed)


candidate_factory = build_candidate
