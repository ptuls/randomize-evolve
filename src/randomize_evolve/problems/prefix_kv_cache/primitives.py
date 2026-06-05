"""Composable online primitives for evolved prefix KV-cache policies."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Iterable, Sequence
import math

MAX_DECAY_TIMESCALES = 8
DEFAULT_MAX_DECAY_KEYS = 1024


def decay_vector(
    values: Sequence[float],
    half_lives: Sequence[float],
    elapsed: int,
) -> tuple[float, ...]:
    """Return a deterministically decayed value vector without mutating state."""

    if len(values) != len(half_lives):
        raise ValueError("values and half_lives must have the same length")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("values must be finite")
    normalized_half_lives = _validate_half_lives(half_lives)
    elapsed = max(0, elapsed)
    return tuple(
        float(value) * 2.0 ** (-elapsed / half_life)
        for value, half_life in zip(values, normalized_half_lives, strict=True)
    )


class MultiTimescaleDecay:
    """Maintains bounded per-key exponential-decay vectors.

    Each observed amount is added to every configured timescale. State is
    bounded by ``max_keys`` and evicts the least-recently-touched key when the
    limit is exceeded.
    """

    def __init__(
        self,
        half_lives: Iterable[float],
        *,
        max_keys: int = DEFAULT_MAX_DECAY_KEYS,
    ) -> None:
        self.half_lives = _validate_half_lives(tuple(half_lives))
        if max_keys <= 0:
            raise ValueError("max_keys must be positive")
        self.max_keys = max_keys
        self._state: OrderedDict[Hashable, tuple[tuple[float, ...], int]] = (
            OrderedDict()
        )

    @property
    def timescale_count(self) -> int:
        """Return the fixed number of accumulators stored per key."""

        return len(self.half_lives)

    @property
    def state_size(self) -> int:
        """Return the number of keys currently retained."""

        return len(self._state)

    def observe(
        self,
        key: Hashable,
        amount: float,
        now: int,
    ) -> tuple[float, ...]:
        """Add ``amount`` to every timescale for ``key`` and return the vector."""

        if not math.isfinite(amount):
            raise ValueError("amount must be finite")
        values = self.values(key, now)
        updated = tuple(value + amount for value in values)
        self._put(key, updated, now)
        return updated

    def values(self, key: Hashable, now: int) -> tuple[float, ...]:
        """Return the online-decayed vector for ``key`` at logical step ``now``."""

        stored = self._state.get(key)
        if stored is None:
            values = (0.0,) * self.timescale_count
            observed_at = now
        else:
            values, observed_at = stored
        effective_now = max(now, observed_at)
        decayed = decay_vector(values, self.half_lives, effective_now - observed_at)
        self._put(key, decayed, effective_now)
        return decayed

    def combine(
        self,
        key: Hashable,
        now: int,
        weights: Sequence[float],
    ) -> float:
        """Return a weighted sum of the current per-timescale values."""

        if len(weights) != self.timescale_count:
            raise ValueError("weights must match the configured timescales")
        if any(not math.isfinite(weight) for weight in weights):
            raise ValueError("weights must be finite")
        return sum(
            value * weight
            for value, weight in zip(self.values(key, now), weights, strict=True)
        )

    def _put(self, key: Hashable, values: tuple[float, ...], now: int) -> None:
        self._state[key] = (values, now)
        self._state.move_to_end(key)
        while len(self._state) > self.max_keys:
            self._state.popitem(last=False)


def _validate_half_lives(half_lives: Sequence[float]) -> tuple[float, ...]:
    """Validate and normalize a bounded half-life vector."""

    normalized = tuple(float(half_life) for half_life in half_lives)
    if not normalized:
        raise ValueError("at least one half-life is required")
    if len(normalized) > MAX_DECAY_TIMESCALES:
        raise ValueError(f"at most {MAX_DECAY_TIMESCALES} half-lives are supported")
    if any(
        not math.isfinite(half_life) or half_life <= 0.0 for half_life in normalized
    ):
        raise ValueError("half-lives must be finite and positive")
    return normalized
