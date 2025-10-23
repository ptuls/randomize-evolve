"""Evaluator implementation for evolving alternatives to Bloom filters.

The evaluator is designed to be plugged into OpenEvolve. It expects the search
candidate to expose a callable that builds a data structure with ``add`` and
``query`` methods. The evaluator applies synthetic workloads across multiple
seeds and collapses the resulting metrics into a scalar fitness score that
rewards low false-positive rates, zero false negatives, modest memory usage,
and low latency.
"""

import dataclasses
import math
import random
import statistics
import time
import tracemalloc
from enum import Enum
from typing import Callable, Iterable, List, Optional, Protocol, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Distribution(str, Enum):
    """Distribution types for generating test items."""

    UNIFORM = "uniform"  # Random uniform across keyspace
    CLUSTERED = "clustered"  # Items grouped in clusters
    SEQUENTIAL = "sequential"  # Sequential IDs
    POWER_LAW = "power_law"  # Zipf/power-law distribution


class EvaluatorConfig(BaseModel):
    """Configuration knobs for the Bloom filter alternative evaluator."""

    model_config = ConfigDict(validate_assignment=True)

    key_bits: int = Field(default=32, gt=0)
    positives: int = Field(default=5000, gt=0)
    queries: int = Field(default=10000, gt=0)
    negative_fraction: float = Field(default=0.5)
    seeds: Sequence[int] = Field(default_factory=lambda: (17, 23, 71, 89, 131))
    build_timeout_s: float = Field(default=1.0, gt=0.0)
    query_timeout_s: float = Field(default=1.0, gt=0.0)
    false_negative_penalty: float = Field(default=1e6, gt=0.0)
    false_positive_weight: float = Field(default=200.0, ge=0.0)
    memory_weight: float = Field(default=0.05, ge=0.0)
    latency_weight: float = Field(default=10.0, ge=0.0)
    max_memory_bytes: Optional[int] = Field(default=None, gt=0)

    # Distribution configuration
    distribution: Distribution = Field(default=Distribution.UNIFORM)
    num_clusters: int = Field(default=10, gt=0)  # For clustered distribution
    cluster_radius: int = Field(default=1000, gt=0)  # For clustered distribution
    power_law_exponent: float = Field(default=1.5, gt=0.0)  # For power-law distribution

    @field_validator("negative_fraction")
    @classmethod
    def _check_negative_fraction(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("negative_fraction must be between 0 and 1")
        return value


class Candidate(Protocol):
    """Minimal protocol the evolved structure must satisfy."""

    def add(self, item: int) -> None: ...

    def query(self, item: int) -> bool: ...


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
    bits_per_item: float
    mean_build_time_ms: float
    mean_query_time_ms: float
    error: Optional[str] = None


class Evaluator:
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
                logger.exception("Candidate evaluation failed for seed {}", seed)
                continue

            trials.append(trial)

        if not trials:
            message = ", ".join(errors) if errors else "no successful trials"
            logger.error("Evaluator produced no successful trials: {}", message)
            return EvaluationResult(
                score=math.inf,
                success=False,
                trials=[],
                false_positive_rate=1.0,
                false_negative_rate=1.0,
                mean_peak_memory_bytes=math.inf,
                bits_per_item=math.inf,
                mean_build_time_ms=math.inf,
                mean_query_time_ms=math.inf,
                error=message,
            )

        fp_rate = statistics.fmean(t.false_positive_rate for t in trials)
        fn_rate = statistics.fmean(t.false_negative_rate for t in trials)
        mean_mem = statistics.fmean(t.peak_memory_bytes for t in trials)
        mean_build_ms = statistics.fmean(t.build_time_s for t in trials) * 1e3
        mean_query_ms = statistics.fmean(t.query_time_s for t in trials) * 1e3
        bits_per_item = (mean_mem * 8.0) / max(1, self.config.positives)

        score = self._score(fp_rate, fn_rate, mean_mem, mean_build_ms, mean_query_ms)

        message = ", ".join(errors) if errors else None
        if message:
            logger.warning("Evaluator encountered partial failures: {}", message)

        logger.debug(
            "Evaluation complete: fp_rate={:.4f}, fn_rate={:.4f}, memory={:.0f}B ({:.1f} bits/item), "
            "build={:.2f}ms, query={:.2f}ms, score={:.2f}",
            fp_rate,
            fn_rate,
            mean_mem,
            bits_per_item,
            mean_build_ms,
            mean_query_ms,
            score,
        )
        return EvaluationResult(
            score=score,
            success=not errors,
            trials=trials,
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            mean_peak_memory_bytes=mean_mem,
            bits_per_item=bits_per_item,
            mean_build_time_ms=mean_build_ms,
            mean_query_time_ms=mean_query_ms,
            error=message,
        )

    def _run_trial(self, factory: CandidateFactory, seed: int) -> TrialMetrics:
        cfg = self.config
        rng = random.Random(seed)

        # Generate items according to configured distribution
        keyspace_size = 1 << cfg.key_bits
        positives = self._generate_items(rng, cfg.positives, keyspace_size)
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
            raise MemoryError(f"candidate used {peak_memory} bytes (> {cfg.max_memory_bytes})")

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

        # Heavily penalize memory to force space-efficient structures
        # E.g., Cuckoo filters become competitive at < 1 byte/item
        bytes_per_item = mean_peak_memory_bytes / max(1, cfg.positives)
        score += cfg.memory_weight * mean_peak_memory_bytes * (1.0 + bytes_per_item)

        score += cfg.latency_weight * (mean_build_time_ms + mean_query_time_ms)

        return score

    def _generate_items(
        self,
        rng: random.Random,
        count: int,
        keyspace_size: int,
    ) -> List[int]:
        """Generate items according to the configured distribution."""
        cfg = self.config

        if cfg.distribution == Distribution.UNIFORM:
            return rng.sample(range(keyspace_size), count)

        elif cfg.distribution == Distribution.CLUSTERED:
            return self._generate_clustered(rng, count, keyspace_size)

        elif cfg.distribution == Distribution.SEQUENTIAL:
            start = rng.randrange(keyspace_size - count)
            return list(range(start, start + count))

        elif cfg.distribution == Distribution.POWER_LAW:
            return self._generate_power_law(rng, count, keyspace_size)

        else:
            # Fallback to uniform
            return rng.sample(range(keyspace_size), count)

    def _generate_clustered(
        self,
        rng: random.Random,
        count: int,
        keyspace_size: int,
    ) -> List[int]:
        """Generate items in clusters with locality."""
        cfg = self.config
        items: List[int] = []
        items_per_cluster = count // cfg.num_clusters
        remainder = count % cfg.num_clusters

        for cluster_idx in range(cfg.num_clusters):
            # Pick a random center for this cluster
            center = rng.randrange(keyspace_size)

            # Determine how many items in this cluster
            cluster_size = items_per_cluster
            if cluster_idx < remainder:
                cluster_size += 1

            # Generate items around the center
            for _ in range(cluster_size):
                offset = rng.randint(-cfg.cluster_radius, cfg.cluster_radius)
                item = (center + offset) % keyspace_size
                items.append(item)

        # Ensure uniqueness
        items = list(set(items))

        # If we lost items due to collisions, add more uniform random items
        while len(items) < count:
            candidate = rng.randrange(keyspace_size)
            if candidate not in items:
                items.append(candidate)

        return items[:count]

    def _generate_power_law(
        self,
        rng: random.Random,
        count: int,
        keyspace_size: int,
    ) -> List[int]:
        """Generate items with power-law (Zipf) distribution."""
        cfg = self.config
        items: List[int] = []

        # Generate items using inverse transform sampling for Zipf distribution
        # This creates a skewed distribution where some values are much more common
        for _ in range(count * 2):  # Generate extra to account for duplicates
            # Use rejection sampling to approximate Zipf
            u = rng.random()
            # Transform uniform random to power-law
            x = int((1 - u) ** (-1.0 / (cfg.power_law_exponent - 1)))
            item = x % keyspace_size
            items.append(item)

        # Remove duplicates and trim to desired count
        items = list(set(items))[:count]

        # If we don't have enough unique items, add uniform random
        while len(items) < count:
            candidate = rng.randrange(keyspace_size)
            if candidate not in items:
                items.append(candidate)

        return items[:count]

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
