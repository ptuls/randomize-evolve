"""Tests for the compact prefix KV-cache seed policy."""

from types import SimpleNamespace

from randomize_evolve.problems.prefix_kv_cache.compact_seed import CompactReusePolicy
from scripts.tune_prefix_kv_compact import DEFAULT_PARAMETERS, TunableCompactPolicy


def _block() -> SimpleNamespace:
    return SimpleNamespace(
        prefix_hash=7,
        descendant_count=0,
        depth=2,
        estimated_recompute_cost=0.0,
        last_accessed_at=0,
        hit_count=0,
    )


def _request(priority: int) -> SimpleNamespace:
    return SimpleNamespace(priority=priority)


def test_compact_frequency_state_decays_by_half_life() -> None:
    policy = CompactReusePolicy(
        1,
        4,
        frequency_half_life=8.0,
        priority_half_life=None,
    )
    block = _block()

    policy.on_request_start(_request(priority=0), now=0)
    policy.on_cache_miss(block, _request(priority=0), now=0)

    assert policy._values(block.prefix_hash, 0)[0] == 1.0
    assert policy._values(block.prefix_hash, 8)[0] == 0.5


def test_compact_priority_maximum_decays_and_can_be_refreshed() -> None:
    policy = CompactReusePolicy(
        1,
        4,
        frequency_half_life=None,
        priority_half_life=8.0,
    )
    block = _block()

    policy.on_request_start(_request(priority=4), now=0)
    policy.on_cache_miss(block, _request(priority=4), now=0)
    assert policy._values(block.prefix_hash, 8)[1] == 2.0

    policy.on_request_start(_request(priority=3), now=8)
    policy.on_cache_hit(block, _request(priority=3), now=8)
    assert policy._values(block.prefix_hash, 8)[1] == 3.0


def test_compact_decay_terms_can_be_disabled_independently() -> None:
    policy = CompactReusePolicy(
        1,
        4,
        frequency_half_life=None,
        priority_half_life=None,
    )
    block = _block()

    policy.on_request_start(_request(priority=4), now=0)
    policy.on_cache_miss(block, _request(priority=4), now=0)

    assert policy._values(block.prefix_hash, 100) == (1.0, 4.0)


def test_tunable_default_matches_compact_seed() -> None:
    compact = CompactReusePolicy(1, 4)
    tunable = TunableCompactPolicy(DEFAULT_PARAMETERS)
    block = _block()

    for now, priority, callback in (
        (0, 4, "on_cache_miss"),
        (8, 0, "on_cache_hit"),
        (20, 2, "on_cache_miss"),
    ):
        request = _request(priority)
        compact.on_request_start(request, now)
        tunable.on_request_start(request, now)
        getattr(compact, callback)(block, request, now)
        getattr(tunable, callback)(block, request, now)

        assert compact.score_admission(block, now) == tunable.score_admission(
            block, now
        )
        assert compact.score_eviction(block, now) == tunable.score_eviction(block, now)
