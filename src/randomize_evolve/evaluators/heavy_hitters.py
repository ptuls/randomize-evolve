"""Evaluator for approximate heavy hitter / frequency estimation structures."""

from __future__ import annotations

import collections
import dataclasses
import math
import random
import statistics
import time
import tracemalloc
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class Candidate(Protocol):
    """Protocol a heavy-hitter candidate must satisfy."""

    def observe(self, item: int, weight: int = 1) -> None:
        """Consume a stream update for ``item`` with the provided ``weight``."""

    def estimate(self, item: int) -> int:
        """Return an estimated frequency for ``item``."""

    def top_k(self, k: int) -> List[Tuple[int, int]]:
        """Return the candidate's best guess of the ``k`` heaviest items."""


CandidateFactory = Callable[[int, int], Candidate]


class EvaluatorConfig(BaseModel):
    """Configuration for the heavy hitter evaluator."""

    model_config = ConfigDict(validate_assignment=True)

    key_bits: int = Field(default=20, gt=0)
    stream_length: int = Field(default=20000, gt=0)
    top_k: int = Field(default=10, gt=0)
    queries: int = Field(default=2048, gt=0)
    num_true_heavy_hitters: int = Field(default=12, gt=0)
    heavy_hitters_fraction: float = Field(default=0.65)
    max_update_weight: int = Field(default=5, gt=0)
    capacity_hint_factor: float = Field(default=6.0, gt=0.0)
    seeds: Sequence[int] = Field(default_factory=lambda: (11, 19, 43, 73))
    build_timeout_s: float = Field(default=1.5, gt=0.0)
    query_timeout_s: float = Field(default=1.0, gt=0.0)
    max_memory_bytes: Optional[int] = Field(default=50 * 1024 * 1024, gt=0)

    heavy_recall_weight: float = Field(default=900.0, ge=0.0)
    heavy_precision_weight: float = Field(default=450.0, ge=0.0)
    relative_error_weight: float = Field(default=260.0, ge=0.0)
    absolute_error_weight: float = Field(default=0.15, ge=0.0)
    zero_frequency_error_weight: float = Field(default=40.0, ge=0.0)
    memory_weight: float = Field(default=0.02, ge=0.0)
    latency_weight: float = Field(default=6.0, ge=0.0)

    @field_validator("heavy_hitters_fraction")
    @classmethod
    def _check_fraction(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("heavy_hitters_fraction must be between 0 and 1")
        return value

    @field_validator("queries")
    @classmethod
    def _check_queries(cls, value: int, info: ValidationInfo) -> int:  # type: ignore[override]
        top_k = info.data.get("top_k", 0) if info.data else 0
        if top_k and value < top_k:
            raise ValueError("queries must be >= top_k so the evaluator can score heavy hitters")
        return value


@dataclasses.dataclass(slots=True)
class TrialMetrics:
    seed: int
    build_time_s: float
    query_time_s: float
    peak_memory_bytes: int
    mean_absolute_error: float
    mean_relative_error: float
    zero_frequency_error: float
    heavy_precision: float
    heavy_recall: float


@dataclasses.dataclass(slots=True)
class EvaluationResult:
    score: float
    success: bool
    trials: List[TrialMetrics]
    mean_absolute_error: float
    mean_relative_error: float
    zero_frequency_error: float
    heavy_precision: float
    heavy_recall: float
    mean_peak_memory_bytes: float
    bits_per_observation: float
    mean_build_time_ms: float
    mean_query_time_ms: float
    error: Optional[str] = None


class Evaluator:
    """Run heavy hitter trials and aggregate metrics into a scalar score."""

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        self.config = config or EvaluatorConfig()

    def __call__(self, factory: CandidateFactory) -> EvaluationResult:
        trials: List[TrialMetrics] = []
        errors: List[str] = []

        for seed in self.config.seeds:
            try:
                trial = self._run_trial(factory, seed)
            except Exception as exc:  # noqa: BLE001 - evolutionary runs surface many errors
                logger.exception("Heavy hitter evaluator failed for seed {}", seed)
                errors.append(f"seed {seed}: {exc!r}")
                continue
            trials.append(trial)

        if not trials:
            message = ", ".join(errors) if errors else "no successful trials"
            logger.error("Heavy hitter evaluator produced no successful trials: {}", message)
            return EvaluationResult(
                score=math.inf,
                success=False,
                trials=[],
                mean_absolute_error=math.inf,
                mean_relative_error=math.inf,
                zero_frequency_error=math.inf,
                heavy_precision=0.0,
                heavy_recall=0.0,
                mean_peak_memory_bytes=math.inf,
                bits_per_observation=math.inf,
                mean_build_time_ms=math.inf,
                mean_query_time_ms=math.inf,
                error=message,
            )

        mean_abs_error = statistics.fmean(t.mean_absolute_error for t in trials)
        mean_rel_error = statistics.fmean(t.mean_relative_error for t in trials)
        zero_freq_error = statistics.fmean(t.zero_frequency_error for t in trials)
        precision = statistics.fmean(t.heavy_precision for t in trials)
        recall = statistics.fmean(t.heavy_recall for t in trials)
        mean_mem = statistics.fmean(t.peak_memory_bytes for t in trials)
        mean_build_ms = statistics.fmean(t.build_time_s for t in trials) * 1e3
        mean_query_ms = statistics.fmean(t.query_time_s for t in trials) * 1e3
        bits_per_observation = (mean_mem * 8.0) / max(1, self.config.stream_length)

        score = self._score(
            precision,
            recall,
            mean_rel_error,
            mean_abs_error,
            zero_freq_error,
            mean_mem,
            mean_build_ms,
            mean_query_ms,
        )

        message = ", ".join(errors) if errors else None
        if message:
            logger.warning("Heavy hitter evaluator encountered partial failures: {}", message)

        logger.debug(
            (
                "Heavy hitter evaluation: precision={:.3f}, recall={:.3f}, "
                "rel_err={:.4f}, abs_err={:.3f}, zero_err={:.3f}, memory={:.0f}B, "
                "build={:.2f}ms, query={:.2f}ms, score={:.2f}"
            ),
            precision,
            recall,
            mean_rel_error,
            mean_abs_error,
            zero_freq_error,
            mean_mem,
            mean_build_ms,
            mean_query_ms,
            score,
        )

        return EvaluationResult(
            score=score,
            success=not errors,
            trials=trials,
            mean_absolute_error=mean_abs_error,
            mean_relative_error=mean_rel_error,
            zero_frequency_error=zero_freq_error,
            heavy_precision=precision,
            heavy_recall=recall,
            mean_peak_memory_bytes=mean_mem,
            bits_per_observation=bits_per_observation,
            mean_build_time_ms=mean_build_ms,
            mean_query_time_ms=mean_query_ms,
            error=message,
        )

    def _run_trial(self, factory: CandidateFactory, seed: int) -> TrialMetrics:
        cfg = self.config
        rng = random.Random(seed)
        keyspace = 1 << cfg.key_bits

        heavy_keys = rng.sample(range(keyspace), cfg.num_true_heavy_hitters)
        heavy_set = set(heavy_keys)

        true_frequencies: Dict[int, int] = collections.defaultdict(int)
        updates: List[Tuple[int, int]] = []

        for _ in range(cfg.stream_length):
            if rng.random() < cfg.heavy_hitters_fraction:
                key = rng.choice(heavy_keys)
            else:
                key = rng.randrange(keyspace)
            weight = rng.randint(1, cfg.max_update_weight)
            updates.append((key, weight))
            true_frequencies[key] += weight

        tracemalloc.start()
        build_start = time.perf_counter()
        candidate_capacity = int(cfg.top_k * cfg.capacity_hint_factor)
        candidate = factory(cfg.key_bits, max(candidate_capacity, cfg.top_k))
        for key, weight in updates:
            candidate.observe(key, weight)
        build_end = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        build_time = build_end - build_start
        if build_time > cfg.build_timeout_s:
            raise TimeoutError(f"build exceeded {cfg.build_timeout_s}s ({build_time:.3f}s)")

        if cfg.max_memory_bytes and peak_memory > cfg.max_memory_bytes:
            raise MemoryError(f"candidate used {peak_memory} bytes (> {cfg.max_memory_bytes})")

        query_items = self._prepare_query_items(rng, heavy_set, keyspace)

        query_start = time.perf_counter()
        absolute_errors: List[float] = []
        relative_errors: List[float] = []
        zero_frequency_errors: List[float] = []

        for item in query_items:
            estimate = max(0, int(candidate.estimate(item)))
            truth = true_frequencies.get(item, 0)
            absolute_errors.append(abs(estimate - truth))
            if truth > 0:
                relative_errors.append(abs(estimate - truth) / truth)
            else:
                zero_frequency_errors.append(float(estimate))

        try:
            reported_top = candidate.top_k(cfg.top_k)
        except AttributeError as exc:
            raise TypeError("candidate must implement top_k(k) -> List[Tuple[int, int]]") from exc

        query_time = time.perf_counter() - query_start
        if query_time > cfg.query_timeout_s:
            raise TimeoutError(f"query exceeded {cfg.query_timeout_s}s ({query_time:.3f}s)")

        precision, recall = self._score_top_k(reported_top, true_frequencies, cfg.top_k)

        mean_abs_error = statistics.fmean(absolute_errors) if absolute_errors else 0.0
        mean_rel_error = statistics.fmean(relative_errors) if relative_errors else 0.0
        zero_freq_error = statistics.fmean(zero_frequency_errors) if zero_frequency_errors else 0.0

        return TrialMetrics(
            seed=seed,
            build_time_s=build_time,
            query_time_s=query_time,
            peak_memory_bytes=peak_memory,
            mean_absolute_error=mean_abs_error,
            mean_relative_error=mean_rel_error,
            zero_frequency_error=zero_freq_error,
            heavy_precision=precision,
            heavy_recall=recall,
        )

    def _score(
        self,
        precision: float,
        recall: float,
        mean_rel_error: float,
        mean_abs_error: float,
        zero_freq_error: float,
        mean_peak_memory_bytes: float,
        mean_build_time_ms: float,
        mean_query_time_ms: float,
    ) -> float:
        cfg = self.config

        score = 0.0
        score += cfg.heavy_recall_weight * max(0.0, 1.0 - recall)
        score += cfg.heavy_precision_weight * max(0.0, 1.0 - precision)
        score += cfg.relative_error_weight * mean_rel_error
        score += cfg.absolute_error_weight * mean_abs_error
        score += cfg.zero_frequency_error_weight * zero_freq_error

        bytes_per_observation = mean_peak_memory_bytes / max(1, cfg.stream_length)
        score += cfg.memory_weight * mean_peak_memory_bytes * (1.0 + bytes_per_observation)
        score += cfg.latency_weight * (mean_build_time_ms + mean_query_time_ms)
        return score

    def _prepare_query_items(
        self, rng: random.Random, heavy_set: Iterable[int], keyspace: int
    ) -> List[int]:
        cfg = self.config
        query_items: List[int] = list(heavy_set)

        while len(query_items) < cfg.queries:
            candidate = rng.randrange(keyspace)
            query_items.append(candidate)

        rng.shuffle(query_items)
        return query_items[: cfg.queries]

    def _score_top_k(
        self,
        reported: Sequence[Tuple[int, int]],
        true_frequencies: Dict[int, int],
        k: int,
    ) -> Tuple[float, float]:
        if not reported:
            return 0.0, 0.0

        sorted_truth = sorted(true_frequencies.items(), key=lambda kv: kv[1], reverse=True)[:k]
        truth_keys = {item for item, _ in sorted_truth}

        reported_keys = [item for item, _ in reported[:k]]
        hits = sum(1 for item in reported_keys if item in truth_keys)

        precision = hits / max(1, min(len(reported_keys), k))
        recall = hits / max(1, len(truth_keys))
        return precision, recall


def baseline_count_min_sketch(
    depth: int = 4,
    width: int = 512,
    heavy_store_factor: float = 4.0,
) -> CandidateFactory:
    """Return a simple Count-Min Sketch style baseline factory."""

    if depth <= 0 or width <= 0:
        raise ValueError("depth and width must be positive")
    if heavy_store_factor <= 1.0:
        raise ValueError("heavy_store_factor must exceed 1.0 so we can track extras")

    class _CountMinSketchCandidate:
        def __init__(self, key_bits: int, capacity: int) -> None:
            self._mask = (1 << key_bits) - 1
            self._depth = depth
            self._width = width
            self._tables = [[0] * width for _ in range(depth)]
            self._heavy_capacity = max(capacity, int(capacity * heavy_store_factor))
            self._heavy_counts: Dict[int, int] = {}

        def observe(self, item: int, weight: int = 1) -> None:
            if weight <= 0:
                return
            weight = int(weight)
            value = self._normalize_item(item)
            for row in range(self._depth):
                index = self._hash(value, row)
                self._tables[row][index] += weight

            self._heavy_counts[value] = self._heavy_counts.get(value, 0) + weight
            if len(self._heavy_counts) > self._heavy_capacity:
                # Evict the lightest tracked item to keep dictionary bounded
                lightest_key = min(self._heavy_counts, key=self._heavy_counts.get)
                self._heavy_counts.pop(lightest_key, None)

        def estimate(self, item: int) -> int:
            value = self._normalize_item(item)
            estimates = [self._tables[row][self._hash(value, row)] for row in range(self._depth)]
            return min(estimates)

        def top_k(self, k: int) -> List[Tuple[int, int]]:
            items = list(self._heavy_counts.items())
            items.sort(key=lambda kv: (kv[1], kv[0]), reverse=True)
            return items[:k]

        def _hash(self, value: int, salt_idx: int) -> int:
            # Use multiplicative hashing on the raw integer to avoid allocations
            salt = salt_idx * 0x9E3779B97F4A7C15
            hashed = (value ^ salt) * 0x9E3779B185EBCA87
            return (hashed & ((1 << 64) - 1)) % self._width

        def _normalize_item(self, item: int) -> int:
            if self._mask:
                return item & self._mask
            return item

    def _factory(key_bits: int, capacity: int) -> Candidate:
        return _CountMinSketchCandidate(key_bits, capacity)

    return _factory
