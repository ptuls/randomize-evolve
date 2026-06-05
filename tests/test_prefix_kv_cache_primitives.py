"""Tests for composable prefix KV-cache policy primitives."""

import math

import pytest

from randomize_evolve.problems.prefix_kv_cache.primitives import (
    MAX_DECAY_TIMESCALES,
    MultiTimescaleDecay,
    decay_vector,
)


def test_decay_vector_is_pure_and_deterministic() -> None:
    values = (8.0, 4.0)
    half_lives = (2.0, 4.0)

    first = decay_vector(values, half_lives, elapsed=2)
    second = decay_vector(values, half_lives, elapsed=2)

    assert first == second
    assert first == pytest.approx((4.0, 4.0 / math.sqrt(2.0)))
    assert values == (8.0, 4.0)


def test_multi_timescale_decay_supports_one_or_many_timescales() -> None:
    single = MultiTimescaleDecay((2.0,))
    multiple = MultiTimescaleDecay((2.0, 4.0))

    assert single.observe("key", 4.0, now=0) == (4.0,)
    assert single.values("key", now=2) == pytest.approx((2.0,))
    assert multiple.observe("key", 4.0, now=0) == (4.0, 4.0)
    assert multiple.values("key", now=2) == pytest.approx((2.0, 4.0 / math.sqrt(2.0)))
    assert multiple.observe("key", 2.0, now=2) == pytest.approx(
        (4.0, 2.0 + 4.0 / math.sqrt(2.0))
    )
    assert multiple.combine("key", now=2, weights=(1.0, 0.5)) == pytest.approx(
        5.0 + 2.0 / math.sqrt(2.0)
    )


def test_multi_timescale_decay_state_is_bounded() -> None:
    decay = MultiTimescaleDecay((2.0, 8.0), max_keys=2)

    decay.observe("first", 1.0, now=0)
    decay.observe("second", 1.0, now=1)
    decay.observe("third", 1.0, now=2)

    assert decay.state_size == 2
    assert decay.values("first", now=3) == (0.0, 0.0)
    assert decay.state_size == 2
    assert all(len(values) == 2 for values, _ in decay._state.values())


@pytest.mark.parametrize(
    "half_lives",
    [
        (),
        (0.0,),
        (math.inf,),
        tuple(float(index + 1) for index in range(MAX_DECAY_TIMESCALES + 1)),
    ],
)
def test_multi_timescale_decay_rejects_invalid_half_lives(half_lives) -> None:
    with pytest.raises(ValueError):
        MultiTimescaleDecay(half_lives)
