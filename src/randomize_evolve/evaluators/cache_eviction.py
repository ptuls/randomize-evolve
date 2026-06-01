"""Evaluator for fixed-capacity cache eviction/admission policies.

The evaluator owns the authoritative resident set and asks candidates only for
the behavioral choices that matter: update metadata, admit a miss or skip it,
and choose a victim when the cache is full.
"""

from __future__ import annotations

import dataclasses
import math
import random
import statistics
import time
import tracemalloc
from bisect import bisect_left
from collections import OrderedDict
from typing import Callable, List, Optional, Protocol, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Candidate(Protocol):
    """Minimal policy contract for cache admission/eviction candidates."""

    def on_access(self, key: int, hit: bool) -> None:
        """Update policy metadata after every access."""

    def should_admit(self, key: int) -> bool:
        """Return whether a missed key should enter the cache."""

    def pick_victim(self) -> int:
        """Return the resident key to evict when the cache is full."""


CandidateFactory = Callable[[int, int], Candidate]


class EvaluatorConfig(BaseModel):
    """Configuration for randomized cache eviction workloads."""

    model_config = ConfigDict(validate_assignment=True)

    key_bits: int = Field(default=24, gt=0)
    capacity: int = Field(default=512, gt=0)
    trace_length: int = Field(default=50000, gt=0)
    working_set_size: int = Field(default=8192, gt=1)
    zipf_exponent: float = Field(default=1.15, gt=0.0)
    scan_burst_every: int = Field(default=1200, gt=0)
    scan_burst_length: int = Field(default=384, ge=0)
    drift_interval: int = Field(default=10000, gt=0)
    drift_stride: int = Field(default=2048, gt=0)
    seeds: Sequence[int] = Field(default_factory=lambda: (7, 19, 43, 89, 131))
    trace_timeout_s: float = Field(default=3.0, gt=0.0)
    max_memory_bytes: Optional[int] = Field(default=80 * 1024 * 1024, gt=0)

    miss_weight: float = Field(default=1000.0, ge=0.0)
    metadata_weight: float = Field(default=1.5, ge=0.0)
    latency_weight: float = Field(default=8.0, ge=0.0)
    work_weight: float = Field(default=0.02, ge=0.0)

    @field_validator("zipf_exponent")
    @classmethod
    def _check_zipf_exponent(cls, value: float) -> float:
        if value <= 1.0:
            raise ValueError("zipf_exponent must exceed 1.0 for finite Zipf weights")
        return value

    @model_validator(mode="after")
    def _check_capacity_vs_working_set(self) -> "EvaluatorConfig":
        if self.capacity >= self.working_set_size:
            raise ValueError("capacity must stay below working_set_size")
        if self.working_set_size >= (1 << self.key_bits):
            raise ValueError("working_set_size must fit within key_bits")
        return self


@dataclasses.dataclass(slots=True)
class TrialMetrics:
    seed: int
    accesses: int
    hits: int
    misses: int
    admits: int
    evictions: int
    skipped_admissions: int
    trace_time_s: float
    peak_memory_bytes: int
    metadata_bytes: int
    operation_count: int

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.accesses)

    @property
    def miss_rate(self) -> float:
        return self.misses / max(1, self.accesses)


@dataclasses.dataclass(slots=True)
class EvaluationResult:
    score: float
    success: bool
    trials: List[TrialMetrics]
    hit_rate: float
    miss_rate: float
    admission_rate: float
    eviction_rate: float
    metadata_bytes_per_cached_item: float
    mean_peak_memory_bytes: float
    mean_trace_time_ms: float
    mean_access_time_us: float
    mean_operation_count: float
    error: Optional[str] = None


class Evaluator:
    """Run cache policy trials and aggregate them into a scalar loss."""

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        self.config = config or EvaluatorConfig()

    def __call__(self, factory: CandidateFactory) -> EvaluationResult:
        trials: List[TrialMetrics] = []
        errors: List[str] = []

        for seed in self.config.seeds:
            try:
                trace = self._generate_trace(seed)
                trial = self._run_trial(factory, seed, trace)
            except Exception as exc:  # noqa: BLE001 - evolutionary runs surface many errors
                logger.exception("Cache eviction evaluator failed for seed {}", seed)
                errors.append(f"seed {seed}: {exc!r}")
                continue
            trials.append(trial)

        if not trials:
            message = ", ".join(errors) if errors else "no successful trials"
            return EvaluationResult(
                score=math.inf,
                success=False,
                trials=[],
                hit_rate=0.0,
                miss_rate=1.0,
                admission_rate=0.0,
                eviction_rate=0.0,
                metadata_bytes_per_cached_item=math.inf,
                mean_peak_memory_bytes=math.inf,
                mean_trace_time_ms=math.inf,
                mean_access_time_us=math.inf,
                mean_operation_count=math.inf,
                error=message,
            )

        hit_rate = statistics.fmean(t.hit_rate for t in trials)
        miss_rate = statistics.fmean(t.miss_rate for t in trials)
        admission_rate = statistics.fmean(t.admits / max(1, t.misses) for t in trials)
        eviction_rate = statistics.fmean(t.evictions / max(1, t.accesses) for t in trials)
        metadata_bytes_per_cached_item = statistics.fmean(
            t.metadata_bytes / self.config.capacity for t in trials
        )
        mean_peak_memory = statistics.fmean(t.peak_memory_bytes for t in trials)
        mean_trace_ms = statistics.fmean(t.trace_time_s for t in trials) * 1e3
        mean_access_us = statistics.fmean(
            (t.trace_time_s / max(1, t.accesses)) * 1e6 for t in trials
        )
        mean_operation_count = statistics.fmean(t.operation_count for t in trials)

        score = self._score(
            miss_rate,
            metadata_bytes_per_cached_item,
            mean_access_us,
            mean_operation_count,
        )

        message = ", ".join(errors) if errors else None
        if message:
            logger.warning("Cache eviction evaluator encountered partial failures: {}", message)

        return EvaluationResult(
            score=score,
            success=not errors,
            trials=trials,
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            admission_rate=admission_rate,
            eviction_rate=eviction_rate,
            metadata_bytes_per_cached_item=metadata_bytes_per_cached_item,
            mean_peak_memory_bytes=mean_peak_memory,
            mean_trace_time_ms=mean_trace_ms,
            mean_access_time_us=mean_access_us,
            mean_operation_count=mean_operation_count,
            error=message,
        )

    def _run_trial(
        self,
        factory: CandidateFactory,
        seed: int,
        trace: Sequence[int],
    ) -> TrialMetrics:
        cfg = self.config
        resident: set[int] = set()

        tracemalloc.start()
        start = time.perf_counter()
        policy = factory(cfg.key_bits, cfg.capacity)

        hits = 0
        misses = 0
        admits = 0
        evictions = 0
        skipped = 0

        for key in trace:
            hit = key in resident
            policy.on_access(key, hit)
            if hit:
                hits += 1
                continue

            misses += 1
            if not policy.should_admit(key):
                skipped += 1
                continue

            admits += 1
            if len(resident) >= cfg.capacity:
                victim = policy.pick_victim()
                if victim not in resident:
                    raise ValueError(f"pick_victim returned non-resident key {victim!r}")
                resident.remove(victim)
                evictions += 1
                if victim == key:
                    continue
            resident.add(key)
            if len(resident) > cfg.capacity:
                raise AssertionError("framework capacity enforcement failed")

        elapsed = time.perf_counter() - start
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if elapsed > cfg.trace_timeout_s:
            raise TimeoutError(f"trace exceeded {cfg.trace_timeout_s}s ({elapsed:.3f}s)")
        if cfg.max_memory_bytes and peak_memory > cfg.max_memory_bytes:
            raise MemoryError(f"candidate used {peak_memory} bytes (> {cfg.max_memory_bytes})")

        metadata_bytes = self._metadata_bytes(policy, len(resident))
        operation_count = self._operation_count(policy)

        return TrialMetrics(
            seed=seed,
            accesses=len(trace),
            hits=hits,
            misses=misses,
            admits=admits,
            evictions=evictions,
            skipped_admissions=skipped,
            trace_time_s=elapsed,
            peak_memory_bytes=peak_memory,
            metadata_bytes=metadata_bytes,
            operation_count=operation_count,
        )

    def _generate_trace(self, seed: int) -> List[int]:
        """Build a trace where frequency, recency, scans, and drift disagree."""
        cfg = self.config
        rng = random.Random(seed)
        keyspace = 1 << cfg.key_bits
        trace: List[int] = []
        zipf_cdf = self._zipf_cdf(cfg.working_set_size, cfg.zipf_exponent)
        scan_base = keyspace - cfg.trace_length - cfg.scan_burst_length - 1
        scan_cursor = 0

        while len(trace) < cfg.trace_length:
            if cfg.scan_burst_length and len(trace) > 0 and len(trace) % cfg.scan_burst_every == 0:
                for _ in range(cfg.scan_burst_length):
                    if len(trace) >= cfg.trace_length:
                        break
                    trace.append((scan_base + scan_cursor) % keyspace)
                    scan_cursor += 1
                continue

            phase = len(trace) // cfg.drift_interval
            hot_start = (phase * cfg.drift_stride) % (keyspace - cfg.working_set_size)
            rank = self._sample_zipf_rank(rng, zipf_cdf)
            trace.append(hot_start + rank)

        return trace

    @staticmethod
    def _zipf_cdf(n: int, exponent: float) -> List[float]:
        cumulative: List[float] = []
        running = 0.0
        for rank in range(1, n + 1):
            running += rank ** (-exponent)
            cumulative.append(running)
        return [value / running for value in cumulative]

    @staticmethod
    def _sample_zipf_rank(rng: random.Random, cdf: Sequence[float]) -> int:
        return bisect_left(cdf, rng.random())

    def _score(
        self,
        miss_rate: float,
        metadata_bytes_per_cached_item: float,
        mean_access_time_us: float,
        mean_operation_count: float,
    ) -> float:
        cfg = self.config
        score = cfg.miss_weight * miss_rate
        score += cfg.metadata_weight * metadata_bytes_per_cached_item
        score += cfg.latency_weight * mean_access_time_us
        score += cfg.work_weight * (mean_operation_count / max(1, cfg.trace_length))
        return score

    def _metadata_bytes(self, policy: Candidate, resident_count: int) -> int:
        reporter = getattr(policy, "metadata_bytes", None)
        if callable(reporter):
            value = reporter()
            if isinstance(value, (int, float)) and math.isfinite(value):
                return max(0, int(value))
        # Conservative fallback for evolved candidates that do not report their
        # modeled metadata: charge Python peak-ish object overhead per resident.
        return max(1, resident_count) * 64

    @staticmethod
    def _operation_count(policy: Candidate) -> int:
        reporter = getattr(policy, "operation_count", None)
        if callable(reporter):
            value = reporter()
            if isinstance(value, (int, float)) and math.isfinite(value):
                return max(0, int(value))
        return 0


class _RandomReplacementPolicy:
    def __init__(self, key_bits: int, capacity: int, seed: int = 0) -> None:
        del key_bits
        self._capacity = capacity
        self._rng = random.Random(seed or capacity)
        self._resident: list[int] = []
        self._positions: dict[int, int] = {}

    def on_access(self, key: int, hit: bool) -> None:
        del key, hit

    def should_admit(self, key: int) -> bool:
        if key in self._positions:
            return False
        self._positions[key] = len(self._resident)
        self._resident.append(key)
        return True

    def pick_victim(self) -> int:
        if not self._resident:
            raise RuntimeError("no resident keys to evict")
        index = self._rng.randrange(len(self._resident) - 1)
        victim = self._resident[index]
        last = self._resident.pop()
        if index < len(self._resident):
            self._resident[index] = last
            self._positions[last] = index
        del self._positions[victim]
        return victim

    def metadata_bytes(self) -> int:
        return self._capacity * 16


class _LRUPolicy:
    def __init__(self, key_bits: int, capacity: int) -> None:
        del key_bits
        self._capacity = capacity
        self._order: OrderedDict[int, None] = OrderedDict()

    def on_access(self, key: int, hit: bool) -> None:
        if hit and key in self._order:
            self._order.move_to_end(key)

    def should_admit(self, key: int) -> bool:
        self._order[key] = None
        self._order.move_to_end(key)
        return True

    def pick_victim(self) -> int:
        victim, _ = self._order.popitem(last=False)
        return victim

    def metadata_bytes(self) -> int:
        return self._capacity * 24


class _ExactLFUPolicy:
    def __init__(self, key_bits: int, capacity: int) -> None:
        del key_bits
        self._capacity = capacity
        self._clock = 0
        self._freq: dict[int, int] = {}
        self._resident_age: dict[int, int] = {}
        self._pending: int | None = None

    def on_access(self, key: int, hit: bool) -> None:
        self._clock += 1
        self._freq[key] = self._freq.get(key, 0) + 1
        if hit:
            self._resident_age[key] = self._clock

    def should_admit(self, key: int) -> bool:
        if len(self._resident_age) < self._capacity:
            self._resident_age[key] = self._clock
            return True
        victim = self._least_frequent_resident()
        if victim is None:
            return False
        if self._freq.get(key, 0) < self._freq.get(victim, 0):
            return False
        self._pending = key
        self._resident_age[key] = self._clock
        return True

    def pick_victim(self) -> int:
        victim = self._least_frequent_resident(exclude=self._pending)
        if victim is None:
            victim = self._pending
        if victim is None:
            raise RuntimeError("no resident keys to evict")
        self._resident_age.pop(victim, None)
        self._pending = None
        return victim

    def _least_frequent_resident(self, exclude: int | None = None) -> int | None:
        candidates = (key for key in self._resident_age if exclude is None or key != exclude)
        return min(
            candidates,
            key=lambda key: (self._freq.get(key, 0), self._resident_age[key]),
            default=None,
        )

    def metadata_bytes(self) -> int:
        return len(self._freq) * 12 + self._capacity * 24


class _AgingFrequencySketch:
    def __init__(self, width: int, depth: int = 4, sample_size: int = 10000) -> None:
        self._width = max(1, width)
        self._depth = max(1, depth)
        self._sample_size = max(1, sample_size)
        self._size = 0
        self._tables = [[0] * self._width for _ in range(self._depth)]

    def increment(self, key: int) -> None:
        for row in range(self._depth):
            index = self._hash(key, row)
            if self._tables[row][index] < 255:
                self._tables[row][index] += 1
        self._size += 1
        if self._size >= self._sample_size:
            self._age()

    def estimate(self, key: int) -> int:
        return min(self._tables[row][self._hash(key, row)] for row in range(self._depth))

    def metadata_bytes(self) -> int:
        return self._width * self._depth

    def _age(self) -> None:
        for row in self._tables:
            for idx, value in enumerate(row):
                row[idx] = value // 2
        self._size //= 2

    def _hash(self, key: int, row: int) -> int:
        hashed = (key ^ (row * 0x9E3779B97F4A7C15)) * 0xBF58476D1CE4E5B9
        return (hashed & ((1 << 64) - 1)) % self._width


class _WindowTinyLFUPolicy:
    def __init__(
        self,
        key_bits: int,
        capacity: int,
        *,
        window_fraction: float = 0.2,
        sketch_width_factor: int = 4,
    ) -> None:
        del key_bits
        self._capacity = capacity
        self._window_capacity = max(1, min(capacity, int(round(capacity * window_fraction))))
        self._main_capacity = max(0, capacity - self._window_capacity)
        width = max(64, capacity * sketch_width_factor)
        self._sketch = _AgingFrequencySketch(width, sample_size=max(1000, capacity * 20))
        self._window: OrderedDict[int, None] = OrderedDict()
        self._main: OrderedDict[int, None] = OrderedDict()
        self._ops = 0

    def on_access(self, key: int, hit: bool) -> None:
        self._ops += 1
        self._sketch.increment(key)
        if hit:
            if key in self._window:
                self._window.move_to_end(key)
            elif key in self._main:
                self._main.move_to_end(key)

    def should_admit(self, key: int) -> bool:
        if key in self._window or key in self._main:
            return False
        self._window[key] = None
        self._window.move_to_end(key)
        return True

    def pick_victim(self) -> int:
        if len(self._window) <= self._window_capacity:
            victim, _ = self._main.popitem(last=False)
            return victim

        candidate, _ = self._window.popitem(last=False)
        if self._main_capacity <= 0:
            return candidate
        if len(self._main) < self._main_capacity:
            self._main[candidate] = None
            return self._window.popitem(last=False)[0]

        victim = next(iter(self._main))
        self._ops += 2
        if self._sketch.estimate(candidate) >= self._sketch.estimate(victim):
            self._main.popitem(last=False)
            self._main[candidate] = None
            return victim
        return candidate

    def metadata_bytes(self) -> int:
        return self._sketch.metadata_bytes() + self._capacity * 24

    def operation_count(self) -> int:
        return self._ops


def random_replacement_policy() -> CandidateFactory:
    """Return a random-replacement baseline factory."""

    return lambda key_bits, capacity: _RandomReplacementPolicy(key_bits, capacity)


def lru_policy() -> CandidateFactory:
    """Return a standard LRU baseline factory."""

    return lambda key_bits, capacity: _LRUPolicy(key_bits, capacity)


def lfu_policy() -> CandidateFactory:
    """Return an exact LFU-without-aging baseline factory."""

    return lambda key_bits, capacity: _ExactLFUPolicy(key_bits, capacity)


def window_tinylfu_policy(
    *,
    window_fraction: float = 0.2,
    sketch_width_factor: int = 4,
) -> CandidateFactory:
    """Return a compact W-TinyLFU-style baseline factory."""

    def _factory(key_bits: int, capacity: int) -> Candidate:
        return _WindowTinyLFUPolicy(
            key_bits,
            capacity,
            window_fraction=window_fraction,
            sketch_width_factor=sketch_width_factor,
        )

    return _factory
