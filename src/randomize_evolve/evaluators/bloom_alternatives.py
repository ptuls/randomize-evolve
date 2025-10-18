"""Evaluator implementation for evolving alternatives to Bloom filters.

The evaluator is designed to be plugged into OpenEvolve. It expects the search
candidate to expose a callable that builds a data structure with ``add`` and
``query`` methods. The evaluator applies synthetic workloads across multiple
seeds and collapses the resulting metrics into a scalar fitness score that
rewards low false-positive rates, zero false negatives, modest memory usage,
and low latency.
"""

from __future__ import annotations

import dataclasses
import math
import random
import statistics
import time
import tracemalloc
from typing import Callable, Iterable, List, Optional, Protocol, Sequence


@dataclasses.dataclass(slots=True)
class EvaluatorConfig:
    """Configuration knobs for the Bloom filter alternative evaluator."""

    key_bits: int = 32
    positives: int = 5000
    queries: int = 10000
    negative_fraction: float = 0.5
    seeds: Sequence[int] = dataclasses.field(
        default_factory=lambda: (17, 23, 71, 89, 131)
    )
    build_timeout_s: float = 1.0
    query_timeout_s: float = 1.0
    false_negative_penalty: float = 1e6
    false_positive_weight: float = 200.0
    memory_weight: float = 0.05
    latency_weight: float = 10.0
    max_memory_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.negative_fraction < 1.0:
            raise ValueError("negative_fraction must be between 0 and 1")
        if self.positives <= 0:
            raise ValueError("positives must be > 0")
        if self.queries <= 0:
            raise ValueError("queries must be > 0")
        if self.key_bits <= 0:
            raise ValueError("key_bits must be > 0")


class Candidate(Protocol):
    """Minimal protocol the evolved structure must satisfy."""

    def add(self, item: int) -> None:
        ...

    def query(self, item: int) -> bool:
        ...


CandidateFactory = Callable[[int, int], Candidate]


@dataclasses.dataclass(slots=True)
class TrialMetrics:
    seed: int
    build_time_s: float
    query_time_s: float
    false_positives: int
    false_negatives: int
    total_positive_queries: int
    total_negative_queries: int
    peak_memory_bytes: int

    @property
    def false_positive_rate(self) -> float:
        if self.total_negative_queries == 0:
            return 0.0
        return self.false_positives / self.total_negative_queries

    @property
    def false_negative_rate(self) -> float:
        if self.total_positive_queries == 0:
            return 0.0
        return self.false_negatives / self.total_positive_queries


@dataclasses.dataclass(slots=True)
class EvaluationResult:
    score: float
    success: bool
    trials: List[TrialMetrics]
    false_positive_rate: float
    false_negative_rate: float
    mean_peak_memory_bytes: float
    mean_build_time_ms: float
    mean_query_time_ms: float
    error: Optional[str] = None


class BloomAlternativeEvaluator:
    """Callable wrapper suitable for OpenEvolve evaluator registrations."""

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        self.config = config or EvaluatorConfig()

    def __call__(self, factory: CandidateFactory) -> EvaluationResult:
        trials: List[TrialMetrics] = []
        errors: List[str] = []

        for seed in self.config.seeds:
            try:
                trial = self._run_trial(factory, seed)
            except Exception as exc:  # noqa: BLE001 - evolutionary search spews
                errors.append(f"seed {seed}: {exc!r}")
                continue

            trials.append(trial)

        if not trials:
            message = ", ".join(errors) if errors else "no successful trials"
            return EvaluationResult(
                score=math.inf,
                success=False,
                trials=[],
                false_positive_rate=1.0,
                false_negative_rate=1.0,
                mean_peak_memory_bytes=math.inf,
                mean_build_time_ms=math.inf,
                mean_query_time_ms=math.inf,
                error=message,
            )

        fp_rate = statistics.fmean(t.false_positive_rate for t in trials)
        fn_rate = statistics.fmean(t.false_negative_rate for t in trials)
        mean_mem = statistics.fmean(t.peak_memory_bytes for t in trials)
        mean_build_ms = statistics.fmean(t.build_time_s for t in trials) * 1e3
        mean_query_ms = statistics.fmean(t.query_time_s for t in trials) * 1e3

        score = self._score(fp_rate, fn_rate, mean_mem, mean_build_ms, mean_query_ms)

        message = ", ".join(errors) if errors else None
        return EvaluationResult(
            score=score,
            success=not errors,
            trials=trials,
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            mean_peak_memory_bytes=mean_mem,
            mean_build_time_ms=mean_build_ms,
            mean_query_time_ms=mean_query_ms,
            error=message,
        )

    def _run_trial(self, factory: CandidateFactory, seed: int) -> TrialMetrics:
        cfg = self.config
        rng = random.Random(seed)

        positives = rng.sample(range(1 << cfg.key_bits), cfg.positives)
        positive_set = set(positives)

        neg_needed = int(cfg.queries * cfg.negative_fraction)
        negatives = self._draw_negatives(rng, positive_set, neg_needed, 1 << cfg.key_bits)

        pos_queries = cfg.queries - neg_needed

        tracemalloc.start()
        build_start = time.perf_counter()
        candidate = factory(cfg.key_bits, cfg.positives)
        for value in positives:
            candidate.add(value)
        build_end = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        build_time = build_end - build_start
        if build_time > cfg.build_timeout_s:
            raise TimeoutError(f"build exceeded {cfg.build_timeout_s}s ({build_time:.3f}s)")

        if cfg.max_memory_bytes and peak_memory > cfg.max_memory_bytes:
            raise MemoryError(
                f"candidate used {peak_memory} bytes (> {cfg.max_memory_bytes})"
            )

        false_negatives = 0
        false_positives = 0

        query_start = time.perf_counter()

        # Positive queries can contain duplicates to mimic real workloads.
        for value in rng.choices(positives, k=pos_queries):
            if not candidate.query(value):
                false_negatives += 1

        for value in negatives:
            if candidate.query(value):
                false_positives += 1

        query_time = time.perf_counter() - query_start
        if query_time > cfg.query_timeout_s:
            raise TimeoutError(f"query exceeded {cfg.query_timeout_s}s ({query_time:.3f}s)")

        return TrialMetrics(
            seed=seed,
            build_time_s=build_time,
            query_time_s=query_time,
            false_positives=false_positives,
            false_negatives=false_negatives,
            total_positive_queries=pos_queries,
            total_negative_queries=neg_needed,
            peak_memory_bytes=peak_memory,
        )

    def _score(
        self,
        false_positive_rate: float,
        false_negative_rate: float,
        mean_peak_memory_bytes: float,
        mean_build_time_ms: float,
        mean_query_time_ms: float,
    ) -> float:
        cfg = self.config
        score = 0.0

        if false_negative_rate > 0.0:
            score += cfg.false_negative_penalty * false_negative_rate

        score += cfg.false_positive_weight * false_positive_rate
        score += cfg.memory_weight * mean_peak_memory_bytes
        score += cfg.latency_weight * (mean_build_time_ms + mean_query_time_ms)

        return score

    @staticmethod
    def _draw_negatives(
        rng: random.Random,
        positives: Iterable[int],
        needed: int,
        modulus: int,
    ) -> List[int]:
        positive_set = set(positives)
        negatives: List[int] = []
        while len(negatives) < needed:
            candidate = rng.randrange(modulus)
            if candidate in positive_set:
                continue
            negatives.append(candidate)
        return negatives


def baseline_bloom_filter(bits_per_item: int) -> CandidateFactory:
    """Return a candidate factory that wraps ``pybloomfiltermmap``-like APIs.

    The intent is to give you a quick way to validate the evaluator: hand the
    returned factory to :class:`BloomAlternativeEvaluator` and ensure the
    metrics look sane before launching a search. The implementation here is a
    lightweight pure-Python stand-in so the repository keeps zero external
    dependencies.
    """

    class _BloomSim:
        def __init__(self, key_bits: int, capacity: int) -> None:
            self._mask = (1 << key_bits) - 1
            table_size = max(1, capacity * bits_per_item)
            self._size = table_size
            self._bits = [False] * table_size
            self._hash_count = max(1, bits_per_item // 2)

        def _hash(self, item: int, salt: int) -> int:
            return ((item ^ (salt * 0x9E3779B97F4A7C15)) & self._mask) % self._size

        def add(self, item: int) -> None:
            for offset in range(self._hash_count):
                self._bits[self._hash(item, offset)] = True

        def query(self, item: int) -> bool:
            return all(self._bits[self._hash(item, offset)] for offset in range(self._hash_count))

    def _factory(key_bits: int, capacity: int) -> Candidate:
        return _BloomSim(key_bits, capacity)

    return _factory
