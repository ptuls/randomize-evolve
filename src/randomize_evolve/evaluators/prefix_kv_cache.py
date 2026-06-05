"""Deterministic prefix KV-cache evaluator for scoring eviction heuristics."""

from __future__ import annotations

import ast
from collections import deque
import hashlib
import inspect
import math
import random
import tracemalloc
from dataclasses import dataclass, field
from statistics import mean, median, pstdev
from typing import Callable, Iterable, Protocol


@dataclass(frozen=True)
class RequestInfo:
    """Candidate-visible request metadata."""

    request_id: int
    tenant_id: int
    session_id: int
    prompt_length: int
    priority: int
    request_type: str
    prompt_tokens: tuple[int, ...]
    predicted_output_length: int | None = None


@dataclass(frozen=True)
class PrefixBlockInfo:
    """Candidate-visible prefix block metadata."""

    block_id: int
    prefix_hash: int
    parent_hash: int | None
    depth: int
    start_token: int
    end_token: int
    token_count: int
    tenant_id: int
    created_at: int
    last_accessed_at: int
    hit_count: int
    descendant_count: int
    active_ref_count: int
    estimated_recompute_cost: float
    estimated_future_reuse: float | None = None
    estimated_next_reuse_distance: float | None = None


class PrefixKVPolicy(Protocol):
    """Scoring-only policy interface used by the simulator."""

    def on_request_start(self, request: RequestInfo, now: int) -> None: ...

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float: ...

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float: ...

    def on_cache_hit(
        self, block: PrefixBlockInfo, request: RequestInfo, now: int
    ) -> None: ...

    def on_cache_miss(
        self, block: PrefixBlockInfo, request: RequestInfo, now: int
    ) -> None: ...


PolicyFactory = Callable[[int, int, int | None], PrefixKVPolicy]

_DEPTH_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("depth_1_2", 1, 2),
    ("depth_3_4", 3, 4),
    ("depth_5_8", 5, 8),
    ("depth_9_plus", 9, None),
)
_HIGH_DESCENDANT_MIN_COUNT = 2
_COLD_DEEP_MIN_DEPTH = 5
_SHORT_REUSE_DISTANCE_STEPS = 8
_TEMPORAL_WINDOWS = 4
_PREFIX_ROLES = ("system", "developer", "user")
_TOKEN_PREFIX_ROLES: dict[int, str] = {}


@dataclass(frozen=True)
class WorkloadRequest:
    """Simulator-internal request, including true output length."""

    info: RequestInfo
    true_output_length: int
    prompt_tokens: tuple[int, ...] = ()
    arrival_step: int | None = None


def _request_arrival_steps(requests: tuple[WorkloadRequest, ...]) -> tuple[int, ...]:
    """Returns monotonic logical arrival times, preserving sequential defaults."""

    arrival_steps = []
    previous_step = -1
    for request_index, request in enumerate(requests):
        arrival_step = (
            request_index if request.arrival_step is None else request.arrival_step
        )
        if arrival_step < previous_step:
            raise ValueError("workload arrival steps must be monotonic")
        arrival_steps.append(arrival_step)
        previous_step = arrival_step
    return tuple(arrival_steps)


@dataclass
class WorkloadConfig:
    """Configures one workload family inside one split."""

    family: str
    split: str
    request_count: int = 96
    seed_offset: int = 0


@dataclass
class EvaluatorConfig:
    """Configuration for prefix KV-cache evaluation and scoring."""

    capacity_blocks: int = 24
    capacity_sweep_blocks: tuple[int, ...] = ()
    block_size_tokens: int = 8
    seeds: tuple[int, ...] = (11, 23, 37)
    train_families: tuple[str, ...] = (
        "shared_system_prompt",
        "rag_template_reuse",
        "long_context_mixed",
        "session_continuation_growth",
    )
    validation_families: tuple[str, ...] = (
        "agent_trace_branching",
        "phase_shift_prompts",
        "multi_tenant_skew",
        "hotset_cold_scan",
        "cyclic_working_set_pressure",
        "concurrent_long_generation",
        "stochastic_serving_mix",
        "rolling_template_versions",
        "heavy_tailed_prefix_lengths",
        "priority_burst_recovery",
        "priority_one_off_noise",
        "tenant_phase_shift_cycles",
    )
    hidden_families: tuple[str, ...] = (
        "adversarial_unique_prompts",
        "cross_family_mixture",
        "tenant_session_reentry",
        "stochastic_serving_mix_shifted",
        "rolling_template_versions_shifted",
        "heavy_tailed_prefix_lengths_shifted",
        "priority_burst_recovery_shifted",
        "cyclic_working_set_pressure_shifted",
        "priority_one_off_noise_shifted",
        "tenant_phase_shift_cycles_shifted",
    )
    request_count: int = 96
    family_request_multipliers: dict[str, int] = field(
        default_factory=lambda: {
            "tenant_phase_shift_cycles": 3,
            "tenant_phase_shift_cycles_shifted": 4,
        }
    )
    prefill_cost_per_token: float = 1.0
    lookup_cost_per_block: float = 0.035
    eviction_cost_per_block: float = 0.2
    active_tokens_per_step: int = 64
    w_avg_tok: float = 80.0
    w_avg_blk: float = 60.0
    min_workload_weight: float = 0.5
    min_seed_weight: float = 0.15
    request_tail_weight: float = 12.0
    worst_window_weight: float = 12.0
    priority_hit_weight: float = 8.0
    wasted_admission_weight: float = 6.0
    admission_utility_weight: float = 1.0
    avoidable_eviction_weight: float = 8.0
    latency_norm: float = 0.0
    latency_weight: float = 35.0
    latency_cap: float = 40.0
    churn_weight: float = 0.015
    churn_cap: float = 25.0
    fairness_weight: float = 80.0
    fairness_cap: float = 30.0
    k_complex: float = 0.065
    complexity_exponent: float = 0.75
    v_min: float = -1_000.0
    invalid_surcharge: float = 1_000.0
    timeout_s: float = 30.0
    max_memory_bytes: int = 64 * 1024 * 1024

    def effective_capacity_blocks(self) -> tuple[int, ...]:
        """Returns the capacities evaluated for each workload and seed."""

        values = self.capacity_sweep_blocks or (self.capacity_blocks,)
        capacities: list[int] = []
        for value in values:
            capacity = int(value)
            if capacity <= 0:
                raise ValueError("capacity blocks must be positive")
            if capacity not in capacities:
                capacities.append(capacity)
        return tuple(capacities)

    def workload_configs(self, splits: Iterable[str]) -> tuple[WorkloadConfig, ...]:
        configs: list[WorkloadConfig] = []
        for split in splits:
            families = {
                "train": self.train_families,
                "validation": self.validation_families,
                "hidden": self.hidden_families,
            }[split]
            for index, family in enumerate(families):
                multiplier = int(self.family_request_multipliers.get(family, 1))
                if multiplier <= 0:
                    raise ValueError("family request multipliers must be positive")
                configs.append(
                    WorkloadConfig(
                        family=family,
                        split=split,
                        request_count=self.request_count * multiplier,
                        seed_offset=1000 * (index + 1),
                    )
                )
        return tuple(configs)


@dataclass
class TrialMetrics:
    """Metrics for one workload family and random seed."""

    split: str
    workload: str
    seed: int
    capacity_blocks: int = 0
    block_hit_rate: float = 0.0
    token_hit_rate: float = 0.0
    priority_weighted_token_hit_rate: float = 0.0
    high_priority_token_hit_rate: float = 0.0
    low_priority_token_hit_rate: float = 0.0
    priority_request_fraction: float = 0.0
    request_token_hit_rate_p10: float = 0.0
    request_token_hit_rate_p50: float = 0.0
    high_priority_request_token_hit_rate_p10: float = 0.0
    worst_quarter_token_hit_rate: float = 0.0
    final_quarter_token_hit_rate: float = 0.0
    quarter_token_hit_rate_stddev: float = 0.0
    prefill_tokens_saved: float = 0.0
    recompute_tokens: float = 0.0
    recompute_cost: float = 0.0
    lookup_block_count: int = 0
    lookup_blocks_per_request: float = 0.0
    eviction_count: int = 0
    admission_count: int = 0
    admission_score_count: int = 0
    admission_rejection_count: int = 0
    admission_rate: float = 0.0
    useful_admission_count: int = 0
    useful_admission_rate: float = 0.0
    wasted_admission_count: int = 0
    wasted_admission_rate: float = 0.0
    admitted_token_count: int = 0
    useful_admission_token_count: int = 0
    useful_admission_token_rate: float = 0.0
    wasted_admission_token_count: int = 0
    wasted_admission_token_rate: float = 0.0
    admission_saved_tokens: int = 0
    admission_saved_tokens_per_admission: float = 0.0
    admission_token_utility: float = 0.0
    evicted_without_hit_count: int = 0
    evicted_without_hit_rate: float = 0.0
    policy_bypass_tokens: int = 0
    policy_bypass_token_rate: float = 0.0
    cache_churn_per_1k: float = 0.0
    forced_bypass_count: int = 0
    forced_bypass_tokens: int = 0
    forced_bypass_token_rate: float = 0.0
    short_reuse_after_eviction_missed_tokens: int = 0
    short_reuse_after_eviction_missed_token_rate: float = 0.0
    eviction_reuse_distance_p50: float = 0.0
    eviction_reuse_distance_p95: float = 0.0
    avoidable_eviction_count: int = 0
    avoidable_eviction_rate: float = 0.0
    avoidable_short_reuse_eviction_count: int = 0
    avoidable_short_reuse_eviction_rate: float = 0.0
    tenant_count: int = 0
    tenant_fairness_penalty: float = 0.0
    tenant_token_hit_rate_p10: float = 0.0
    tenant_jain_fairness: float = 1.0
    p50_latency_proxy: float = 0.0
    p95_latency_proxy: float = 0.0
    p99_latency_proxy: float = 0.0
    high_priority_p95_latency_proxy: float = 0.0
    high_priority_p99_latency_proxy: float = 0.0
    p95_recompute_cost: float = 0.0
    recovery_request_count: int = 0
    recovery_token_hit_rate: float = 0.0
    recovery_p95_latency_proxy: float = 0.0
    recovery_phase_count: int = 0
    worst_recovery_phase_token_hit_rate: float = 0.0
    final_recovery_phase_token_hit_rate: float = 0.0
    worst_recovery_phase_p95_latency_proxy: float = 0.0
    memory_occupancy_mean: float = 0.0
    memory_occupancy_peak: int = 0
    arrival_span_steps: int = 0
    active_request_count_peak: int = 0
    max_prefill_cost: float = 0.0
    scoring_fn_complexity: int = 0
    invalid: bool = False
    invalid_reason: str = ""
    matched_lengths: tuple[int, ...] = ()
    structural_metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | bool | str]:
        metrics: dict[str, float | int | bool | str] = {
            "block_hit_rate": self.block_hit_rate,
            "capacity_blocks": self.capacity_blocks,
            "token_hit_rate": self.token_hit_rate,
            "priority_weighted_token_hit_rate": self.priority_weighted_token_hit_rate,
            "high_priority_token_hit_rate": self.high_priority_token_hit_rate,
            "low_priority_token_hit_rate": self.low_priority_token_hit_rate,
            "priority_request_fraction": self.priority_request_fraction,
            "request_token_hit_rate_p10": self.request_token_hit_rate_p10,
            "request_token_hit_rate_p50": self.request_token_hit_rate_p50,
            "high_priority_request_token_hit_rate_p10": (
                self.high_priority_request_token_hit_rate_p10
            ),
            "worst_quarter_token_hit_rate": self.worst_quarter_token_hit_rate,
            "final_quarter_token_hit_rate": self.final_quarter_token_hit_rate,
            "quarter_token_hit_rate_stddev": self.quarter_token_hit_rate_stddev,
            "prefill_tokens_saved": self.prefill_tokens_saved,
            "recompute_tokens": self.recompute_tokens,
            "recompute_cost": self.recompute_cost,
            "lookup_block_count": self.lookup_block_count,
            "lookup_blocks_per_request": self.lookup_blocks_per_request,
            "eviction_count": self.eviction_count,
            "admission_count": self.admission_count,
            "admission_score_count": self.admission_score_count,
            "admission_rejection_count": self.admission_rejection_count,
            "admission_rate": self.admission_rate,
            "useful_admission_count": self.useful_admission_count,
            "useful_admission_rate": self.useful_admission_rate,
            "wasted_admission_count": self.wasted_admission_count,
            "wasted_admission_rate": self.wasted_admission_rate,
            "admitted_token_count": self.admitted_token_count,
            "useful_admission_token_count": self.useful_admission_token_count,
            "useful_admission_token_rate": self.useful_admission_token_rate,
            "wasted_admission_token_count": self.wasted_admission_token_count,
            "wasted_admission_token_rate": self.wasted_admission_token_rate,
            "admission_saved_tokens": self.admission_saved_tokens,
            "admission_saved_tokens_per_admission": (
                self.admission_saved_tokens_per_admission
            ),
            "admission_token_utility": self.admission_token_utility,
            "evicted_without_hit_count": self.evicted_without_hit_count,
            "evicted_without_hit_rate": self.evicted_without_hit_rate,
            "policy_bypass_tokens": self.policy_bypass_tokens,
            "policy_bypass_token_rate": self.policy_bypass_token_rate,
            "cache_churn_per_1k": self.cache_churn_per_1k,
            "forced_bypass_count": self.forced_bypass_count,
            "forced_bypass_tokens": self.forced_bypass_tokens,
            "forced_bypass_token_rate": self.forced_bypass_token_rate,
            "short_reuse_after_eviction_missed_tokens": (
                self.short_reuse_after_eviction_missed_tokens
            ),
            "short_reuse_after_eviction_missed_token_rate": (
                self.short_reuse_after_eviction_missed_token_rate
            ),
            "eviction_reuse_distance_p50": self.eviction_reuse_distance_p50,
            "eviction_reuse_distance_p95": self.eviction_reuse_distance_p95,
            "avoidable_eviction_count": self.avoidable_eviction_count,
            "avoidable_eviction_rate": self.avoidable_eviction_rate,
            "avoidable_short_reuse_eviction_count": (
                self.avoidable_short_reuse_eviction_count
            ),
            "avoidable_short_reuse_eviction_rate": (
                self.avoidable_short_reuse_eviction_rate
            ),
            "tenant_count": self.tenant_count,
            "tenant_fairness_penalty": self.tenant_fairness_penalty,
            "tenant_token_hit_rate_p10": self.tenant_token_hit_rate_p10,
            "tenant_jain_fairness": self.tenant_jain_fairness,
            "p50_latency_proxy": self.p50_latency_proxy,
            "p95_latency_proxy": self.p95_latency_proxy,
            "p99_latency_proxy": self.p99_latency_proxy,
            "high_priority_p95_latency_proxy": self.high_priority_p95_latency_proxy,
            "high_priority_p99_latency_proxy": self.high_priority_p99_latency_proxy,
            "p95_recompute_cost": self.p95_recompute_cost,
            "recovery_request_count": self.recovery_request_count,
            "recovery_token_hit_rate": self.recovery_token_hit_rate,
            "recovery_p95_latency_proxy": self.recovery_p95_latency_proxy,
            "recovery_phase_count": self.recovery_phase_count,
            "worst_recovery_phase_token_hit_rate": (
                self.worst_recovery_phase_token_hit_rate
            ),
            "final_recovery_phase_token_hit_rate": (
                self.final_recovery_phase_token_hit_rate
            ),
            "worst_recovery_phase_p95_latency_proxy": (
                self.worst_recovery_phase_p95_latency_proxy
            ),
            "memory_occupancy_mean": self.memory_occupancy_mean,
            "memory_occupancy_peak": self.memory_occupancy_peak,
            "arrival_span_steps": self.arrival_span_steps,
            "active_request_count_peak": self.active_request_count_peak,
            "max_prefill_cost": self.max_prefill_cost,
            "scoring_fn_complexity": self.scoring_fn_complexity,
            "invalid": self.invalid,
            "invalid_reason": self.invalid_reason,
        }
        metrics.update(self.structural_metrics)
        return metrics


@dataclass
class EvaluationResult:
    """Aggregated evaluator result."""

    combined_score: float
    success: bool
    invalid_fraction: float
    split_metrics: dict[str, dict[str, float | int | bool | str]]
    workload_metrics: dict[str, dict[str, float | int | bool | str]]
    capacity_metrics: dict[str, dict[str, float | int | bool | str]]
    candidate_metadata: dict[str, float | int | bool | str]
    score_breakdown: dict[str, float]
    trials: tuple[TrialMetrics, ...] = ()


@dataclass
class _BlockState:
    prefix_hash: int
    parent_hash: int | None
    depth: int
    start_token: int
    end_token: int
    token_count: int
    prefix_role: str
    tenant_id: int
    created_at: int
    last_accessed_at: int
    hit_count: int = 0
    active_ref_count: int = 0
    resident: bool = False
    admission_tracked: bool = False
    resident_hit_count: int = 0
    resident_children: set[int] = field(default_factory=set)
    known_children: set[int] = field(default_factory=set)

    @property
    def block_id(self) -> int:
        return self.prefix_hash


@dataclass
class _AdmissionAccounting:
    """Finalized utility for successful admission residency intervals."""

    useful_count: int = 0
    wasted_count: int = 0
    admitted_tokens: int = 0
    useful_tokens: int = 0
    wasted_tokens: int = 0
    saved_tokens: int = 0
    evicted_without_hit_count: int = 0

    def record(self, block: _BlockState, *, evicted: bool) -> None:
        """Record one tracked interval before its state is reset."""

        if not block.admission_tracked:
            return
        self.admitted_tokens += block.token_count
        self.saved_tokens += block.resident_hit_count * block.token_count
        if block.resident_hit_count > 0:
            self.useful_count += 1
            self.useful_tokens += block.token_count
            return
        self.wasted_count += 1
        self.wasted_tokens += block.token_count
        if evicted:
            self.evicted_without_hit_count += 1


class InvalidCandidateError(RuntimeError):
    """Raised when a candidate returns an invalid score or crashes."""


class _FutureReuseTracker:
    """Live future-reuse metadata for oracle-style reporting baselines."""

    def __init__(
        self,
        requests: tuple[WorkloadRequest, ...],
        *,
        block_size_tokens: int,
        enabled: bool,
    ) -> None:
        self.enabled = enabled
        self._remaining_counts: dict[int, int] = {}
        self._future_positions: dict[int, deque[int]] = {}
        if not enabled:
            return

        arrival_steps = _request_arrival_steps(requests)
        for now, request in zip(arrival_steps, requests, strict=True):
            for prefix_hash in _request_prefix_hashes(request, block_size_tokens):
                self._remaining_counts[prefix_hash] = (
                    self._remaining_counts.get(prefix_hash, 0) + 1
                )
                self._future_positions.setdefault(prefix_hash, deque()).append(now)

    def advance(self, blocks: list[_BlockState], now: int) -> None:
        if not self.enabled:
            return
        for block in blocks:
            prefix_hash = block.prefix_hash
            self._remaining_counts[prefix_hash] = max(
                0,
                self._remaining_counts.get(prefix_hash, 0) - 1,
            )
            positions = self._future_positions.get(prefix_hash)
            if positions:
                positions.popleft()

    def remaining_count(self, prefix_hash: int) -> float | None:
        if not self.enabled:
            return None
        return float(self._remaining_counts.get(prefix_hash, 0))

    def next_distance(self, prefix_hash: int, now: int) -> float | None:
        if not self.enabled:
            return None
        positions = self._future_positions.get(prefix_hash) or ()
        if not positions:
            return math.inf
        return float(max(0, positions[0] - now))


class PrefixKVCacheSimulator:
    """Owns cache state and applies scoring-only candidate policies."""

    def __init__(
        self,
        *,
        capacity_blocks: int,
        block_size_tokens: int,
        prefill_cost_per_token: float,
        lookup_cost_per_block: float,
        eviction_cost_per_block: float,
        active_tokens_per_step: int = 64,
        expose_future_reuse: bool = False,
        max_memory_bytes: int | None = None,
    ) -> None:
        self.capacity_blocks = capacity_blocks
        self.block_size_tokens = block_size_tokens
        self.prefill_cost_per_token = prefill_cost_per_token
        self.lookup_cost_per_block = lookup_cost_per_block
        self.eviction_cost_per_block = eviction_cost_per_block
        self.active_tokens_per_step = active_tokens_per_step
        self.expose_future_reuse = expose_future_reuse
        self.max_memory_bytes = max_memory_bytes
        self.blocks: dict[int, _BlockState] = {}
        self._release_events: dict[int, list[int]] = {}
        self._resident_hashes: set[int] = set()
        self._leaf_hashes: set[int] = set()
        self._descendant_counts: dict[int, int] = {}
        self._evicted_hashes: set[int] = set()
        self._last_evicted_at: dict[int, int] = {}

    def run(
        self,
        policy: PrefixKVPolicy,
        requests: tuple[WorkloadRequest, ...],
        *,
        split: str,
        workload: str,
        seed: int,
        scoring_fn_complexity: int = 0,
    ) -> TrialMetrics:
        total_blocks = 0
        total_tokens = 0
        hit_blocks = 0
        hit_tokens = 0
        priority_weighted_total_tokens = 0
        priority_weighted_hit_tokens = 0
        high_priority_tokens = 0
        high_priority_hit_tokens = 0
        high_priority_request_count = 0
        low_priority_tokens = 0
        low_priority_hit_tokens = 0
        recompute_tokens = 0
        recompute_cost = 0.0
        lookup_block_count = 0
        admission_count = 0
        admission_score_count = 0
        admission_rejection_count = 0
        policy_bypass_tokens = 0
        eviction_count = 0
        forced_bypass_count = 0
        forced_bypass_tokens = 0
        latencies: list[float] = []
        high_priority_latencies: list[float] = []
        high_priority_request_token_hit_rates: list[float] = []
        recompute_costs: list[float] = []
        request_token_hit_rates: list[float] = []
        request_hit_records: list[tuple[int, int]] = []
        recovery_hit_tokens = 0
        recovery_tokens = 0
        recovery_latencies: list[float] = []
        recovery_request_count = 0
        recovery_phase_records: list[list[tuple[int, int, float]]] = []
        previous_was_recovery = False
        occupancies: list[int] = []
        active_request_releases: dict[int, int] = {}
        active_request_count = 0
        active_request_count_peak = 0
        max_prefill_cost = 0.0
        matched_lengths: list[int] = []
        tenant_hits: dict[int, int] = {}
        tenant_tokens: dict[int, int] = {}
        depth_total_blocks: dict[str, int] = {}
        depth_hit_blocks: dict[str, int] = {}
        depth_total_tokens: dict[str, int] = {}
        depth_hit_tokens: dict[str, int] = {}
        prefix_role_hit_tokens: dict[str, int] = {
            "system": 0,
            "developer": 0,
            "user": 0,
        }
        high_descendant_evictions = 0
        cold_deep_admission_opportunities = 0
        cold_deep_admissions = 0
        reuse_after_eviction_missed_blocks = 0
        reuse_after_eviction_missed_tokens = 0
        short_reuse_after_eviction_missed_tokens = 0
        eviction_reuse_distances: list[float] = []
        admission_accounting = _AdmissionAccounting()
        avoidable_eviction_count = 0
        avoidable_short_reuse_eviction_count = 0

        try:
            self._validate_policy(policy)
            self._check_memory_limit()
            future_reuse = _FutureReuseTracker(
                requests,
                block_size_tokens=self.block_size_tokens,
                enabled=self.expose_future_reuse,
            )
            audit_future_reuse = _FutureReuseTracker(
                requests,
                block_size_tokens=self.block_size_tokens,
                enabled=True,
            )
            arrival_steps = _request_arrival_steps(requests)
            for now, request in zip(arrival_steps, requests, strict=True):
                self._release_expired(now)
                for release_at in sorted(
                    step for step in active_request_releases if step <= now
                ):
                    active_request_count -= active_request_releases.pop(release_at)
                request_blocks = self._materialize_chain(request, now)
                future_reuse.advance(request_blocks, now)
                audit_future_reuse.advance(request_blocks, now)
                max_prefill_cost = max(
                    max_prefill_cost,
                    sum(
                        self._estimated_recompute_cost(block)
                        for block in request_blocks
                    ),
                )
                total_blocks += len(request_blocks)
                total_tokens += request.info.prompt_length
                for block in request_blocks:
                    band = _depth_band(block.depth)
                    depth_total_blocks[band] = depth_total_blocks.get(band, 0) + 1
                    depth_total_tokens[band] = (
                        depth_total_tokens.get(band, 0) + block.token_count
                    )
                tenant_tokens[request.info.tenant_id] = (
                    tenant_tokens.get(request.info.tenant_id, 0)
                    + request.info.prompt_length
                )

                self._call_hook(policy.on_request_start, request.info, now)
                matched_len = self.match_resident_prefix(request_blocks)
                lookup_blocks = matched_len + int(matched_len < len(request_blocks))
                lookup_block_count += lookup_blocks
                matched_lengths.append(matched_len)
                per_request_evictions = 0
                hit_blocks += matched_len
                tokens_hit = sum(
                    block.token_count for block in request_blocks[:matched_len]
                )
                hit_tokens += tokens_hit
                priority_weight = 1 + max(0, request.info.priority)
                priority_weighted_total_tokens += (
                    request.info.prompt_length * priority_weight
                )
                priority_weighted_hit_tokens += tokens_hit * priority_weight
                if request.info.priority > 0:
                    high_priority_tokens += request.info.prompt_length
                    high_priority_hit_tokens += tokens_hit
                else:
                    low_priority_tokens += request.info.prompt_length
                    low_priority_hit_tokens += tokens_hit
                request_token_hit_rates.append(
                    tokens_hit / max(1, request.info.prompt_length)
                )
                request_hit_records.append((tokens_hit, request.info.prompt_length))
                is_recovery_request = "recovery" in request.info.request_type
                if is_recovery_request:
                    if not previous_was_recovery:
                        recovery_phase_records.append([])
                    recovery_request_count += 1
                    recovery_hit_tokens += tokens_hit
                    recovery_tokens += request.info.prompt_length
                if request.info.priority > 0:
                    high_priority_request_count += 1
                    high_priority_request_token_hit_rates.append(
                        tokens_hit / max(1, request.info.prompt_length)
                    )
                tenant_hits[request.info.tenant_id] = (
                    tenant_hits.get(request.info.tenant_id, 0) + tokens_hit
                )

                duration = max(
                    1,
                    math.ceil(request.true_output_length / self.active_tokens_per_step),
                )
                active_request_count += 1
                active_request_releases[now + duration] = (
                    active_request_releases.get(now + duration, 0) + 1
                )
                active_request_count_peak = max(
                    active_request_count_peak, active_request_count
                )
                for block in request_blocks[:matched_len]:
                    band = _depth_band(block.depth)
                    depth_hit_blocks[band] = depth_hit_blocks.get(band, 0) + 1
                    depth_hit_tokens[band] = (
                        depth_hit_tokens.get(band, 0) + block.token_count
                    )
                    if block.prefix_role in prefix_role_hit_tokens:
                        prefix_role_hit_tokens[block.prefix_role] += block.token_count
                    block.last_accessed_at = now
                    block.hit_count += 1
                    block.resident_hit_count += 1
                    self._pin(block, now + duration)
                    self._call_hook(
                        policy.on_cache_hit,
                        self._info(block, now, future_reuse),
                        request.info,
                        now,
                    )

                admission_blocked = False
                forced_bypass_active = False
                for block in request_blocks[matched_len:]:
                    recompute_tokens += block.token_count
                    recompute_cost += self._estimated_recompute_cost(block)
                    if block.prefix_hash in self._evicted_hashes:
                        reuse_after_eviction_missed_blocks += 1
                        reuse_after_eviction_missed_tokens += block.token_count
                        reuse_distance = max(
                            0,
                            now - self._last_evicted_at.get(block.prefix_hash, now),
                        )
                        eviction_reuse_distances.append(float(reuse_distance))
                        if reuse_distance <= _SHORT_REUSE_DISTANCE_STEPS:
                            short_reuse_after_eviction_missed_tokens += (
                                block.token_count
                            )
                    self._call_hook(
                        policy.on_cache_miss,
                        self._info(block, now, future_reuse),
                        request.info,
                        now,
                    )
                    is_cold_deep = (
                        block.depth >= _COLD_DEEP_MIN_DEPTH and block.hit_count == 0
                    )
                    if is_cold_deep:
                        cold_deep_admission_opportunities += 1
                    if admission_blocked:
                        if forced_bypass_active:
                            forced_bypass_tokens += block.token_count
                        else:
                            policy_bypass_tokens += block.token_count
                        continue
                    admission_score_count += 1
                    score = self._score(
                        policy.score_admission,
                        self._info(block, now, future_reuse),
                        now,
                    )
                    if score <= 0.0:
                        admission_rejection_count += 1
                        policy_bypass_tokens += block.token_count
                        admission_blocked = True
                        continue
                    (
                        admitted,
                        evictions,
                        high_descendant_victims,
                        avoidable_evictions,
                        avoidable_short_reuse_evictions,
                    ) = self._admit_block(
                        policy,
                        block,
                        now,
                        duration,
                        future_reuse,
                        audit_future_reuse,
                        admission_accounting,
                    )
                    per_request_evictions += evictions
                    eviction_count += evictions
                    high_descendant_evictions += high_descendant_victims
                    avoidable_eviction_count += avoidable_evictions
                    avoidable_short_reuse_eviction_count += (
                        avoidable_short_reuse_evictions
                    )
                    if admitted:
                        admission_count += 1
                        if is_cold_deep:
                            cold_deep_admissions += 1
                    else:
                        forced_bypass_count += 1
                        forced_bypass_tokens += block.token_count
                        forced_bypass_active = True
                        admission_blocked = True

                uncached_cost = sum(
                    self._estimated_recompute_cost(block)
                    for block in request_blocks[matched_len:]
                )
                latency = (
                    uncached_cost
                    + lookup_blocks * self.lookup_cost_per_block
                    + per_request_evictions * self.eviction_cost_per_block
                )
                latencies.append(latency)
                recompute_costs.append(uncached_cost)
                if request.info.priority > 0:
                    high_priority_latencies.append(latency)
                if is_recovery_request:
                    recovery_latencies.append(latency)
                    recovery_phase_records[-1].append(
                        (tokens_hit, request.info.prompt_length, latency)
                    )
                previous_was_recovery = is_recovery_request
                occupancies.append(self.resident_count)
        except InvalidCandidateError as exc:
            return TrialMetrics(
                split=split,
                workload=workload,
                seed=seed,
                capacity_blocks=self.capacity_blocks,
                scoring_fn_complexity=scoring_fn_complexity,
                invalid=True,
                invalid_reason=str(exc),
                matched_lengths=tuple(matched_lengths),
            )

        request_count = max(len(requests), 1)
        tenant_rates = self._tenant_hit_rates(tenant_hits, tenant_tokens)
        fairness_penalty = (
            max(tenant_rates) - min(tenant_rates) if tenant_rates else 0.0
        )
        resident_admissions = [
            block
            for block in self.blocks.values()
            if block.resident and block.admission_tracked
        ]
        for block in resident_admissions:
            admission_accounting.record(block, evicted=False)
        quarter_hit_rates = _window_token_hit_rates(
            request_hit_records,
            window_count=_TEMPORAL_WINDOWS,
        )
        recovery_phase_hit_rates = [
            sum(hit_tokens for hit_tokens, _, _ in records)
            / max(1, sum(tokens for _, tokens, _ in records))
            for records in recovery_phase_records
        ]
        recovery_phase_p95_latencies = [
            _percentile([latency for _, _, latency in records], 95)
            for records in recovery_phase_records
        ]
        structural_metrics = _structural_metrics(
            depth_total_blocks=depth_total_blocks,
            depth_hit_blocks=depth_hit_blocks,
            depth_total_tokens=depth_total_tokens,
            depth_hit_tokens=depth_hit_tokens,
            prefix_role_hit_tokens=prefix_role_hit_tokens,
            total_hit_tokens=hit_tokens,
            high_descendant_evictions=high_descendant_evictions,
            eviction_count=eviction_count,
            cold_deep_admission_opportunities=cold_deep_admission_opportunities,
            cold_deep_admissions=cold_deep_admissions,
            reuse_after_eviction_missed_blocks=reuse_after_eviction_missed_blocks,
            reuse_after_eviction_missed_tokens=reuse_after_eviction_missed_tokens,
            recompute_tokens=recompute_tokens,
        )
        return TrialMetrics(
            split=split,
            workload=workload,
            seed=seed,
            capacity_blocks=self.capacity_blocks,
            block_hit_rate=hit_blocks / total_blocks if total_blocks else 0.0,
            token_hit_rate=hit_tokens / total_tokens if total_tokens else 0.0,
            priority_weighted_token_hit_rate=(
                priority_weighted_hit_tokens / priority_weighted_total_tokens
                if priority_weighted_total_tokens
                else 0.0
            ),
            high_priority_token_hit_rate=(
                high_priority_hit_tokens / high_priority_tokens
                if high_priority_tokens
                else 0.0
            ),
            low_priority_token_hit_rate=(
                low_priority_hit_tokens / low_priority_tokens
                if low_priority_tokens
                else 0.0
            ),
            priority_request_fraction=high_priority_request_count / request_count,
            request_token_hit_rate_p10=_percentile(request_token_hit_rates, 10),
            request_token_hit_rate_p50=_percentile(request_token_hit_rates, 50),
            high_priority_request_token_hit_rate_p10=_percentile(
                high_priority_request_token_hit_rates,
                10,
            ),
            worst_quarter_token_hit_rate=min(quarter_hit_rates, default=0.0),
            final_quarter_token_hit_rate=quarter_hit_rates[-1]
            if quarter_hit_rates
            else 0.0,
            quarter_token_hit_rate_stddev=pstdev(quarter_hit_rates)
            if len(quarter_hit_rates) > 1
            else 0.0,
            prefill_tokens_saved=hit_tokens,
            recompute_tokens=recompute_tokens,
            recompute_cost=recompute_cost,
            lookup_block_count=lookup_block_count,
            lookup_blocks_per_request=lookup_block_count / request_count,
            eviction_count=eviction_count,
            admission_count=admission_count,
            admission_score_count=admission_score_count,
            admission_rejection_count=admission_rejection_count,
            admission_rate=admission_count / max(1, admission_score_count),
            useful_admission_count=admission_accounting.useful_count,
            useful_admission_rate=(
                admission_accounting.useful_count / max(1, admission_count)
            ),
            wasted_admission_count=admission_accounting.wasted_count,
            wasted_admission_rate=(
                admission_accounting.wasted_count / max(1, admission_count)
            ),
            admitted_token_count=admission_accounting.admitted_tokens,
            useful_admission_token_count=admission_accounting.useful_tokens,
            useful_admission_token_rate=(
                admission_accounting.useful_tokens
                / max(1, admission_accounting.admitted_tokens)
            ),
            wasted_admission_token_count=admission_accounting.wasted_tokens,
            wasted_admission_token_rate=(
                admission_accounting.wasted_tokens
                / max(1, admission_accounting.admitted_tokens)
            ),
            admission_saved_tokens=admission_accounting.saved_tokens,
            admission_saved_tokens_per_admission=(
                admission_accounting.saved_tokens / max(1, admission_count)
            ),
            admission_token_utility=(
                admission_accounting.saved_tokens
                / max(1, admission_count * self.block_size_tokens)
            ),
            evicted_without_hit_count=admission_accounting.evicted_without_hit_count,
            evicted_without_hit_rate=(
                admission_accounting.evicted_without_hit_count / max(1, eviction_count)
            ),
            policy_bypass_tokens=policy_bypass_tokens,
            policy_bypass_token_rate=policy_bypass_tokens / max(1, total_tokens),
            cache_churn_per_1k=eviction_count * 1000.0 / request_count,
            forced_bypass_count=forced_bypass_count,
            forced_bypass_tokens=forced_bypass_tokens,
            forced_bypass_token_rate=forced_bypass_tokens / max(1, total_tokens),
            short_reuse_after_eviction_missed_tokens=(
                short_reuse_after_eviction_missed_tokens
            ),
            short_reuse_after_eviction_missed_token_rate=(
                short_reuse_after_eviction_missed_tokens / max(1, recompute_tokens)
            ),
            eviction_reuse_distance_p50=_percentile(eviction_reuse_distances, 50),
            eviction_reuse_distance_p95=_percentile(eviction_reuse_distances, 95),
            avoidable_eviction_count=avoidable_eviction_count,
            avoidable_eviction_rate=avoidable_eviction_count / max(1, eviction_count),
            avoidable_short_reuse_eviction_count=(avoidable_short_reuse_eviction_count),
            avoidable_short_reuse_eviction_rate=(
                avoidable_short_reuse_eviction_count / max(1, eviction_count)
            ),
            tenant_count=len(tenant_rates),
            tenant_fairness_penalty=fairness_penalty,
            tenant_token_hit_rate_p10=_percentile(tenant_rates, 10),
            tenant_jain_fairness=_jain_fairness(tenant_rates),
            p50_latency_proxy=_percentile(latencies, 50),
            p95_latency_proxy=_percentile(latencies, 95),
            p99_latency_proxy=_percentile(latencies, 99),
            high_priority_p95_latency_proxy=_percentile(high_priority_latencies, 95),
            high_priority_p99_latency_proxy=_percentile(high_priority_latencies, 99),
            p95_recompute_cost=_percentile(recompute_costs, 95),
            recovery_request_count=recovery_request_count,
            recovery_token_hit_rate=recovery_hit_tokens / max(1, recovery_tokens),
            recovery_p95_latency_proxy=_percentile(recovery_latencies, 95),
            recovery_phase_count=len(recovery_phase_records),
            worst_recovery_phase_token_hit_rate=min(
                recovery_phase_hit_rates,
                default=0.0,
            ),
            final_recovery_phase_token_hit_rate=(
                recovery_phase_hit_rates[-1] if recovery_phase_hit_rates else 0.0
            ),
            worst_recovery_phase_p95_latency_proxy=max(
                recovery_phase_p95_latencies,
                default=0.0,
            ),
            memory_occupancy_mean=mean(occupancies) if occupancies else 0.0,
            memory_occupancy_peak=max(occupancies) if occupancies else 0,
            arrival_span_steps=(
                arrival_steps[-1] - arrival_steps[0] + 1 if arrival_steps else 0
            ),
            active_request_count_peak=active_request_count_peak,
            max_prefill_cost=max_prefill_cost,
            scoring_fn_complexity=scoring_fn_complexity,
            matched_lengths=tuple(matched_lengths),
            structural_metrics=structural_metrics,
        )

    @property
    def resident_count(self) -> int:
        return len(self._resident_hashes)

    def match_resident_prefix(self, blocks: list[_BlockState]) -> int:
        """Return the largest root-contiguous resident prefix length."""

        matched = 0
        for block in blocks:
            if not block.resident:
                break
            matched += 1
        return matched

    def evict_block(self, prefix_hash: int) -> None:
        """Evict a resident leaf block; useful for direct simulator tests."""

        block = self.blocks[prefix_hash]
        if block.active_ref_count or block.resident_children:
            raise ValueError("only inactive resident leaves can be evicted")
        self._remove_resident(block)

    def _admit_block(
        self,
        policy: PrefixKVPolicy,
        block: _BlockState,
        now: int,
        duration: int,
        future_reuse: _FutureReuseTracker,
        audit_future_reuse: _FutureReuseTracker,
        admission_accounting: _AdmissionAccounting,
    ) -> tuple[bool, int, int, int, int]:
        if block.resident:
            self._pin(block, now + duration)
            return True, 0, 0, 0, 0

        if block.parent_hash is not None:
            parent = self.blocks.get(block.parent_hash)
            if parent is None or not parent.resident:
                return False, 0, 0, 0, 0

        self._make_resident(block)
        block.last_accessed_at = now
        release_at = now + duration
        self._pin(block, release_at)
        evictions = 0
        high_descendant_evictions = 0
        avoidable_evictions = 0
        avoidable_short_reuse_evictions = 0
        while self.resident_count > self.capacity_blocks:
            evictable = self._evictable_blocks()
            if not evictable:
                self._unpin(block)
                self._cancel_release(block, release_at)
                self._remove_resident(block)
                return (
                    False,
                    evictions,
                    high_descendant_evictions,
                    avoidable_evictions,
                    avoidable_short_reuse_evictions,
                )
            scored = [
                (
                    self._score(
                        policy.score_eviction,
                        self._info(candidate, now, future_reuse),
                        now,
                    ),
                    candidate.prefix_hash,
                    candidate,
                )
                for candidate in evictable
            ]
            _, _, victim = max(scored)
            victim_next_reuse = audit_future_reuse.next_distance(
                victim.prefix_hash,
                now,
            )
            alternative_next_reuse = [
                audit_future_reuse.next_distance(candidate.prefix_hash, now)
                for candidate in evictable
                if candidate.prefix_hash != victim.prefix_hash
            ]
            furthest_alternative_reuse = max(
                (
                    distance
                    for distance in alternative_next_reuse
                    if distance is not None
                ),
                default=None,
            )
            if (
                victim_next_reuse is not None
                and furthest_alternative_reuse is not None
                and victim_next_reuse < furthest_alternative_reuse
            ):
                avoidable_evictions += 1
                if (
                    victim_next_reuse <= _SHORT_REUSE_DISTANCE_STEPS
                    and furthest_alternative_reuse > _SHORT_REUSE_DISTANCE_STEPS
                ):
                    avoidable_short_reuse_evictions += 1
            if (
                self._descendant_counts.get(victim.prefix_hash, 0)
                >= _HIGH_DESCENDANT_MIN_COUNT
            ):
                high_descendant_evictions += 1
            admission_accounting.record(victim, evicted=True)
            self._evicted_hashes.add(victim.prefix_hash)
            self._last_evicted_at[victim.prefix_hash] = now
            self._remove_resident(victim)
            evictions += 1
        block.admission_tracked = True
        return (
            True,
            evictions,
            high_descendant_evictions,
            avoidable_evictions,
            avoidable_short_reuse_evictions,
        )

    def _evictable_blocks(self) -> list[_BlockState]:
        return [
            self.blocks[prefix_hash]
            for prefix_hash in self._leaf_hashes
            if self.blocks[prefix_hash].active_ref_count == 0
        ]

    def _materialize_chain(
        self,
        request: WorkloadRequest,
        now: int,
    ) -> list[_BlockState]:
        blocks: list[_BlockState] = []
        prefix_tokens: list[int] = []
        tokens = request.prompt_tokens or request.info.prompt_tokens
        for depth, start in enumerate(
            range(0, len(tokens), self.block_size_tokens), start=1
        ):
            chunk = tokens[start : start + self.block_size_tokens]
            prefix_tokens.extend(chunk)
            prefix_hash = _stable_hash((request.info.tenant_id, tuple(prefix_tokens)))
            parent_hash = blocks[-1].prefix_hash if blocks else None
            if prefix_hash not in self.blocks:
                self.blocks[prefix_hash] = _BlockState(
                    prefix_hash=prefix_hash,
                    parent_hash=parent_hash,
                    depth=depth,
                    start_token=start,
                    end_token=start + len(chunk),
                    token_count=len(chunk),
                    prefix_role=_prefix_role(chunk),
                    tenant_id=request.info.tenant_id,
                    created_at=now,
                    last_accessed_at=now,
                )
                if parent_hash is not None:
                    self.blocks[parent_hash].known_children.add(prefix_hash)
                    ancestor_hash = parent_hash
                    while ancestor_hash is not None:
                        self._descendant_counts[ancestor_hash] = (
                            self._descendant_counts.get(ancestor_hash, 0) + 1
                        )
                        ancestor_hash = self.blocks[ancestor_hash].parent_hash
            blocks.append(self.blocks[prefix_hash])
        return blocks

    def _make_resident(self, block: _BlockState) -> None:
        if block.resident:
            return
        block.resident = True
        block.admission_tracked = False
        block.resident_hit_count = 0
        self._resident_hashes.add(block.prefix_hash)
        self._leaf_hashes.add(block.prefix_hash)
        if block.parent_hash is not None and block.parent_hash in self.blocks:
            parent = self.blocks[block.parent_hash]
            if parent.resident:
                parent.resident_children.add(block.prefix_hash)
                self._leaf_hashes.discard(parent.prefix_hash)

    def _remove_resident(self, block: _BlockState) -> None:
        if not block.resident:
            return
        if block.parent_hash is not None and block.parent_hash in self.blocks:
            parent = self.blocks[block.parent_hash]
            parent.resident_children.discard(block.prefix_hash)
            if parent.resident and not parent.resident_children:
                self._leaf_hashes.add(parent.prefix_hash)
        block.resident = False
        block.admission_tracked = False
        block.resident_hit_count = 0
        self._resident_hashes.discard(block.prefix_hash)
        self._leaf_hashes.discard(block.prefix_hash)
        block.resident_children.clear()

    def _pin(self, block: _BlockState, release_at: int) -> None:
        block.active_ref_count += 1
        self._release_events.setdefault(release_at, []).append(block.prefix_hash)

    def _unpin(self, block: _BlockState) -> None:
        block.active_ref_count = max(0, block.active_ref_count - 1)

    def _cancel_release(self, block: _BlockState, release_at: int) -> None:
        events = self._release_events.get(release_at)
        if not events:
            return
        try:
            events.remove(block.prefix_hash)
        except ValueError:
            return
        if not events:
            self._release_events.pop(release_at, None)

    def _release_expired(self, now: int) -> None:
        for release_at in sorted([key for key in self._release_events if key <= now]):
            for prefix_hash in self._release_events.pop(release_at):
                block = self.blocks.get(prefix_hash)
                if block is not None:
                    self._unpin(block)

    def _info(
        self,
        block: _BlockState,
        now: int,
        future_reuse: _FutureReuseTracker,
    ) -> PrefixBlockInfo:
        return PrefixBlockInfo(
            block_id=block.block_id,
            prefix_hash=block.prefix_hash,
            parent_hash=block.parent_hash,
            depth=block.depth,
            start_token=block.start_token,
            end_token=block.end_token,
            token_count=block.token_count,
            tenant_id=block.tenant_id,
            created_at=block.created_at,
            last_accessed_at=block.last_accessed_at,
            hit_count=block.hit_count,
            descendant_count=self._descendant_counts.get(block.prefix_hash, 0),
            active_ref_count=block.active_ref_count,
            estimated_recompute_cost=self._estimated_recompute_cost(block),
            estimated_future_reuse=future_reuse.remaining_count(block.prefix_hash),
            estimated_next_reuse_distance=future_reuse.next_distance(
                block.prefix_hash, now
            ),
        )

    def _estimated_recompute_cost(self, block: _BlockState) -> float:
        return block.end_token * self.prefill_cost_per_token

    def _score(self, func: Callable[[PrefixBlockInfo, int], float], *args) -> float:
        try:
            score = func(*args)
        except Exception as exc:  # pragma: no cover - exercised by tests
            raise InvalidCandidateError(
                f"{func.__name__} raised {type(exc).__name__}"
            ) from exc
        if isinstance(score, bool) or not isinstance(score, (float, int)):
            raise InvalidCandidateError(f"{func.__name__} returned non-numeric score")
        score = float(score)
        if not math.isfinite(score):
            raise InvalidCandidateError(f"{func.__name__} returned non-finite score")
        self._check_memory_limit()
        return score

    def _call_hook(self, func: Callable, *args) -> None:
        try:
            func(*args)
        except Exception as exc:  # pragma: no cover - defensive
            raise InvalidCandidateError(
                f"{func.__name__} raised {type(exc).__name__}"
            ) from exc
        self._check_memory_limit()

    def _check_memory_limit(self) -> None:
        if not self.max_memory_bytes or not tracemalloc.is_tracing():
            return
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        if peak_memory_bytes > self.max_memory_bytes:
            raise InvalidCandidateError(
                f"candidate used {peak_memory_bytes} bytes (> {self.max_memory_bytes})"
            )

    @staticmethod
    def _validate_policy(policy: PrefixKVPolicy) -> None:
        """Ensures missing hooks become structured invalid-candidate results."""

        required_methods = (
            "on_request_start",
            "score_admission",
            "score_eviction",
            "on_cache_hit",
            "on_cache_miss",
        )
        for method_name in required_methods:
            if not callable(getattr(policy, method_name, None)):
                raise InvalidCandidateError(f"policy must implement {method_name}()")

    @staticmethod
    def _tenant_hit_rates(
        tenant_hits: dict[int, int],
        tenant_tokens: dict[int, int],
    ) -> list[float]:
        return [
            tenant_hits.get(tenant, 0) / tokens
            for tenant, tokens in tenant_tokens.items()
            if tokens > 0
        ]


class PrefixKVCacheEvaluator:
    """Callable evaluator compatible with Levi-style candidate factories."""

    def __init__(
        self,
        config: EvaluatorConfig | None = None,
        *,
        splits: tuple[str, ...] = ("train", "validation"),
        expose_future_reuse: bool = False,
    ) -> None:
        self.config = config or EvaluatorConfig()
        self.splits = splits
        self.expose_future_reuse = expose_future_reuse

    def __call__(
        self,
        factory: Callable[..., PrefixKVPolicy] | None = None,
        *,
        scoring_fn_complexity: int = 0,
    ) -> EvaluationResult:
        factory = factory or baseline_lru_blocks
        trials: list[TrialMetrics] = []
        capacity_blocks_values = self.config.effective_capacity_blocks()
        for workload in self.config.workload_configs(self.splits):
            for capacity_blocks in capacity_blocks_values:
                for seed in self.config.seeds:
                    actual_seed = seed + workload.seed_offset
                    requests = build_workload(
                        workload.family,
                        request_count=workload.request_count,
                        block_size_tokens=self.config.block_size_tokens,
                        seed=actual_seed,
                    )
                    trials.append(
                        self._run_trial(
                            factory,
                            requests,
                            split=workload.split,
                            workload=workload.family,
                            seed=actual_seed,
                            capacity_blocks=capacity_blocks,
                            scoring_fn_complexity=scoring_fn_complexity,
                        )
                    )

        return self._result_from_trials(trials, scoring_fn_complexity)

    def evaluate_requests(
        self,
        factory: Callable[..., PrefixKVPolicy] | None,
        requests: Iterable[WorkloadRequest],
        *,
        workload: str = "trace_replay",
        split: str = "validation",
        seed: int = 0,
        scoring_fn_complexity: int = 0,
    ) -> EvaluationResult:
        """Evaluate a fixed metadata-derived request sequence."""

        factory = factory or baseline_lru_blocks
        request_tuple = tuple(requests)
        trials = [
            self._run_trial(
                factory,
                request_tuple,
                split=split,
                workload=workload,
                seed=seed,
                capacity_blocks=capacity_blocks,
                scoring_fn_complexity=scoring_fn_complexity,
            )
            for capacity_blocks in self.config.effective_capacity_blocks()
        ]
        return self._result_from_trials(trials, scoring_fn_complexity)

    def rescore_trials(
        self,
        trials: Iterable[TrialMetrics],
        *,
        scoring_fn_complexity: int = 0,
    ) -> EvaluationResult:
        """Reaggregate existing trials under this evaluator's score weights."""

        return self._result_from_trials(list(trials), scoring_fn_complexity)

    def _run_trial(
        self,
        factory: Callable[..., PrefixKVPolicy],
        requests: tuple[WorkloadRequest, ...],
        *,
        split: str,
        workload: str,
        seed: int,
        capacity_blocks: int,
        scoring_fn_complexity: int,
    ) -> TrialMetrics:
        simulator = PrefixKVCacheSimulator(
            capacity_blocks=capacity_blocks,
            block_size_tokens=self.config.block_size_tokens,
            prefill_cost_per_token=self.config.prefill_cost_per_token,
            lookup_cost_per_block=self.config.lookup_cost_per_block,
            eviction_cost_per_block=self.config.eviction_cost_per_block,
            active_tokens_per_step=self.config.active_tokens_per_step,
            expose_future_reuse=self.expose_future_reuse,
            max_memory_bytes=self.config.max_memory_bytes,
        )
        tracing_already_started = tracemalloc.is_tracing()
        if tracing_already_started:
            tracemalloc.reset_peak()
        else:
            tracemalloc.start()
        try:
            try:
                policy = _build_policy(
                    factory,
                    capacity_blocks,
                    self.config.block_size_tokens,
                    seed,
                )
            except Exception as exc:
                return TrialMetrics(
                    split=split,
                    workload=workload,
                    seed=seed,
                    capacity_blocks=capacity_blocks,
                    scoring_fn_complexity=scoring_fn_complexity,
                    invalid=True,
                    invalid_reason=f"factory raised {type(exc).__name__}",
                )
            trial = simulator.run(
                policy,
                requests,
                split=split,
                workload=workload,
                seed=seed,
                scoring_fn_complexity=scoring_fn_complexity,
            )
            _, peak_memory_bytes = tracemalloc.get_traced_memory()
            if (
                self.config.max_memory_bytes
                and peak_memory_bytes > self.config.max_memory_bytes
            ):
                return TrialMetrics(
                    split=split,
                    workload=workload,
                    seed=seed,
                    capacity_blocks=capacity_blocks,
                    scoring_fn_complexity=scoring_fn_complexity,
                    invalid=True,
                    invalid_reason=(
                        f"candidate used {peak_memory_bytes} bytes "
                        f"(> {self.config.max_memory_bytes})"
                    ),
                )
            return trial
        finally:
            if not tracing_already_started:
                tracemalloc.stop()

    def _result_from_trials(
        self,
        trials: list[TrialMetrics],
        scoring_fn_complexity: int,
    ) -> EvaluationResult:
        invalid_fraction = (
            sum(1 for trial in trials if trial.invalid) / len(trials) if trials else 1.0
        )
        split_metrics = _aggregate_by((trial.split for trial in trials), trials)
        workload_metrics = _aggregate_by(
            (f"{trial.split}/{trial.workload}" for trial in trials), trials
        )
        capacity_metrics = _aggregate_by(
            (f"capacity_{trial.capacity_blocks}" for trial in trials), trials
        )
        score_breakdown = self._score_breakdown(
            trials,
            invalid_fraction,
            scoring_fn_complexity,
        )
        return EvaluationResult(
            combined_score=score_breakdown["combined_score"],
            success=invalid_fraction == 0.0,
            invalid_fraction=invalid_fraction,
            split_metrics=split_metrics,
            workload_metrics=workload_metrics,
            capacity_metrics=capacity_metrics,
            candidate_metadata={
                "capacity_blocks": self.config.capacity_blocks,
                "capacity_sweep_blocks": ",".join(
                    str(value) for value in self.config.effective_capacity_blocks()
                ),
                "block_size_tokens": self.config.block_size_tokens,
                "scoring_fn_complexity": scoring_fn_complexity,
                "churn_weight": self.config.churn_weight,
                "fairness_weight": self.config.fairness_weight,
                "complexity_weight": self.config.k_complex,
                "complexity_exponent": self.config.complexity_exponent,
                "min_workload_weight": self.config.min_workload_weight,
                "min_seed_weight": self.config.min_seed_weight,
                "request_tail_weight": self.config.request_tail_weight,
                "worst_window_weight": self.config.worst_window_weight,
                "priority_hit_weight": self.config.priority_hit_weight,
                "wasted_admission_weight": self.config.wasted_admission_weight,
                "wasted_admission_metric": "wasted_admission_token_rate",
                "admission_utility_weight": self.config.admission_utility_weight,
                "avoidable_eviction_weight": self.config.avoidable_eviction_weight,
                "latency_norm": self.config.latency_norm,
                "latency_norm_scope": (
                    "configured"
                    if self.config.latency_norm > 0.0
                    else "workload_capacity"
                ),
                "expose_future_reuse": self.expose_future_reuse,
            },
            score_breakdown=score_breakdown,
            trials=tuple(trials),
        )

    def _score_trials(
        self,
        trials: list[TrialMetrics],
        invalid_fraction: float,
        complexity: int,
    ) -> float:
        return self._score_breakdown(
            trials,
            invalid_fraction,
            complexity,
        )["combined_score"]

    def _score_breakdown(
        self,
        trials: list[TrialMetrics],
        invalid_fraction: float,
        complexity: int,
    ) -> dict[str, float]:
        """Returns the combined score and its top-level weighted components."""

        if invalid_fraction > 0.0:
            combined_score = (
                self.config.v_min
                - 1.0
                - self.config.invalid_surcharge * invalid_fraction
            )
            return {
                "combined_score": combined_score,
                "invalid_fraction": invalid_fraction,
                "invalid_surcharge_cost": (
                    self.config.invalid_surcharge * invalid_fraction
                ),
            }
        validation = [trial for trial in trials if trial.split == "validation"]
        if not validation:
            validation = (
                list(trials)
                if set(self.splits) == {"hidden"}
                else [trial for trial in trials if trial.split != "hidden"]
            )
        by_workload_capacity: dict[tuple[str, int], list[TrialMetrics]] = {}
        for trial in validation:
            by_workload_capacity.setdefault(
                (trial.workload, trial.capacity_blocks), []
            ).append(trial)
        workload_scores = []
        min_seed_weight = min(1.0, max(0.0, self.config.min_seed_weight))
        for workload_trials in by_workload_capacity.values():
            average_score = _workload_base_score(
                workload_trials,
                token_weight=self.config.w_avg_tok,
                block_weight=self.config.w_avg_blk,
                request_tail_weight=self.config.request_tail_weight,
                worst_window_weight=self.config.worst_window_weight,
                priority_hit_weight=self.config.priority_hit_weight,
                wasted_admission_weight=self.config.wasted_admission_weight,
                admission_utility_weight=self.config.admission_utility_weight,
                avoidable_eviction_weight=self.config.avoidable_eviction_weight,
                latency_weight=self.config.latency_weight,
                latency_cap=self.config.latency_cap,
                latency_norm=self.config.latency_norm,
            )
            seed_floor = min(
                _workload_base_score(
                    [trial],
                    token_weight=self.config.w_avg_tok,
                    block_weight=self.config.w_avg_blk,
                    request_tail_weight=self.config.request_tail_weight,
                    worst_window_weight=self.config.worst_window_weight,
                    priority_hit_weight=self.config.priority_hit_weight,
                    wasted_admission_weight=self.config.wasted_admission_weight,
                    admission_utility_weight=self.config.admission_utility_weight,
                    avoidable_eviction_weight=self.config.avoidable_eviction_weight,
                    latency_weight=self.config.latency_weight,
                    latency_cap=self.config.latency_cap,
                    latency_norm=self.config.latency_norm,
                )
                for trial in workload_trials
            )
            workload_scores.append(
                (1.0 - min_seed_weight) * average_score + min_seed_weight * seed_floor
            )
        mean_score = mean(workload_scores) if workload_scores else 0.0
        min_workload_score = min(workload_scores) if workload_scores else 0.0
        churn = (
            mean(trial.cache_churn_per_1k for trial in validation)
            if validation
            else 0.0
        )
        fairness = (
            mean(
                trial.tenant_fairness_penalty
                for trial in validation
                if trial.workload == "multi_tenant_skew"
            )
            if any(trial.workload == "multi_tenant_skew" for trial in validation)
            else 0.0
        )
        churn_cost = min(self.config.churn_cap, self.config.churn_weight * churn)
        fairness_cost = min(
            self.config.fairness_cap,
            self.config.fairness_weight * fairness,
        )
        complexity_cost = (
            self.config.k_complex * complexity**self.config.complexity_exponent
        )
        min_workload_contribution = self.config.min_workload_weight * min_workload_score
        combined_score = (
            mean_score
            + min_workload_contribution
            - churn_cost
            - fairness_cost
            - complexity_cost
        )
        return {
            "combined_score": combined_score,
            "mean_workload_score": mean_score,
            "min_workload_score": min_workload_score,
            "min_workload_contribution": min_workload_contribution,
            "churn_cost": churn_cost,
            "fairness_cost": fairness_cost,
            "complexity_cost": complexity_cost,
            "validation_trial_count": float(len(validation)),
        }


class _BasePolicy:
    def on_request_start(self, request: RequestInfo, now: int) -> None:
        return None

    def on_cache_hit(
        self, block: PrefixBlockInfo, request: RequestInfo, now: int
    ) -> None:
        return None

    def on_cache_miss(
        self, block: PrefixBlockInfo, request: RequestInfo, now: int
    ) -> None:
        return None


class _NoCachePolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return -1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return 0.0


class _LRUPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(now - block.last_accessed_at)


def _recency_tiebreak(block: PrefixBlockInfo, now: int) -> float:
    """Returns an LRU tie-break score that cannot cross an integer priority."""

    age = max(0, now - block.last_accessed_at)
    return float(age) / (float(age) + 1.0)


class _LFUPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(-block.hit_count) + _recency_tiebreak(block, now)


class _DepthPreferShallowPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(block.depth) + _recency_tiebreak(block, now)


class _RecomputeGreedyPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return block.estimated_recompute_cost

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return -block.estimated_recompute_cost


class _CostAwareLRUPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        age = max(0, now - block.last_accessed_at)
        return float(age) / max(1.0, block.estimated_recompute_cost)


class _PrefixFanoutPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(-block.descendant_count) + _recency_tiebreak(block, now)


class _PrefixAnchorPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        del now
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        age = max(0.0, float(now - block.last_accessed_at))
        descendant_protection = 3.0 * math.log1p(max(0, block.descendant_count))
        depth_protection = 1.5 / max(1.0, float(block.depth))
        return age - descendant_protection - depth_protection


class _TinyLFULRUPolicy(_BasePolicy):
    def __init__(self) -> None:
        self._frequency: dict[int, int] = {}

    def on_cache_hit(
        self, block: PrefixBlockInfo, request: RequestInfo, now: int
    ) -> None:
        del request, now
        self._frequency[block.prefix_hash] = (
            self._frequency.get(block.prefix_hash, 0) + 1
        )

    def on_cache_miss(
        self, block: PrefixBlockInfo, request: RequestInfo, now: int
    ) -> None:
        del request, now
        self._frequency[block.prefix_hash] = (
            self._frequency.get(block.prefix_hash, 0) + 1
        )

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        del now
        frequency = self._frequency.get(block.prefix_hash, 0)
        if block.depth <= 2:
            return 1.0
        return 1.0 if frequency >= 2 else -1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(now - block.last_accessed_at)


class _TenantFairLRUPolicy(_BasePolicy):
    def __init__(self) -> None:
        self._tenant_hit_tokens: dict[int, int] = {}
        self._tenant_seen_tokens: dict[int, int] = {}

    def on_cache_hit(
        self, block: PrefixBlockInfo, request: RequestInfo, now: int
    ) -> None:
        del request, now
        self._record_observation(block, hit=True)

    def on_cache_miss(
        self, block: PrefixBlockInfo, request: RequestInfo, now: int
    ) -> None:
        del request, now
        self._record_observation(block, hit=False)

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        tenant_hit_rate = self._tenant_hit_rate(block.tenant_id)
        return float(now - block.last_accessed_at) + 8.0 * tenant_hit_rate

    def _record_observation(self, block: PrefixBlockInfo, *, hit: bool) -> None:
        tenant_id = block.tenant_id
        self._tenant_seen_tokens[tenant_id] = (
            self._tenant_seen_tokens.get(tenant_id, 0) + block.token_count
        )
        if hit:
            self._tenant_hit_tokens[tenant_id] = (
                self._tenant_hit_tokens.get(tenant_id, 0) + block.token_count
            )

    def _tenant_hit_rate(self, tenant_id: int) -> float:
        hit_tokens = self._tenant_hit_tokens.get(tenant_id, 0)
        seen_tokens = self._tenant_seen_tokens.get(tenant_id, 0)
        return float(hit_tokens) / max(1.0, float(seen_tokens))


class _FutureReuseHeuristicPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        del now
        future_reuse = block.estimated_future_reuse
        next_distance = block.estimated_next_reuse_distance
        if future_reuse is None or next_distance is None:
            return 0.0
        if future_reuse <= 0.0 or math.isinf(next_distance):
            return 1_000_000.0
        return float(next_distance / (1.0 + future_reuse))


class _OracleFutureReusePolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        del now
        future_reuse = block.estimated_future_reuse
        if future_reuse is None:
            return 1.0
        return 1.0 if future_reuse > 0.0 else -1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        del now
        future_reuse = block.estimated_future_reuse
        next_distance = block.estimated_next_reuse_distance
        if future_reuse is None or next_distance is None:
            return 0.0
        if future_reuse <= 0.0 or math.isinf(next_distance):
            return 1_000_000.0
        return float(next_distance)


def baseline_no_cache(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _NoCachePolicy()


def baseline_lru_blocks(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _LRUPolicy()


def baseline_lfu_blocks(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _LFUPolicy()


def baseline_depth_prefer_shallow(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _DepthPreferShallowPolicy()


def baseline_recompute_cost_greedy(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _RecomputeGreedyPolicy()


def baseline_cost_aware_lru(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _CostAwareLRUPolicy()


def baseline_prefix_fanout(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _PrefixFanoutPolicy()


def baseline_prefix_anchor(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _PrefixAnchorPolicy()


def baseline_tinylfu_lru(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _TinyLFULRUPolicy()


def baseline_tenant_fair_lru(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _TenantFairLRUPolicy()


def baseline_future_reuse_heuristic(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _FutureReuseHeuristicPolicy()


def baseline_oracle_future_reuse(
    capacity_blocks: int, block_size_tokens: int, seed: int | None = None
) -> PrefixKVPolicy:
    return _OracleFutureReusePolicy()


BASELINES: dict[str, Callable[..., PrefixKVPolicy]] = {
    "no_cache": baseline_no_cache,
    "lru": baseline_lru_blocks,
    "lfu": baseline_lfu_blocks,
    "depth_prefer_shallow": baseline_depth_prefer_shallow,
    "recompute_greedy": baseline_recompute_cost_greedy,
    "cost_aware_lru": baseline_cost_aware_lru,
    "prefix_fanout": baseline_prefix_fanout,
    "prefix_anchor": baseline_prefix_anchor,
    "tinylfu_lru": baseline_tinylfu_lru,
    "tenant_fair_lru": baseline_tenant_fair_lru,
}

REPORTING_BASELINES: dict[str, Callable[..., PrefixKVPolicy]] = {
    **BASELINES,
    "future_reuse_heuristic": baseline_future_reuse_heuristic,
    "oracle_future_reuse": baseline_oracle_future_reuse,
}


def build_workload(
    family: str,
    *,
    request_count: int,
    block_size_tokens: int,
    seed: int,
) -> tuple[WorkloadRequest, ...]:
    rng = random.Random(seed)
    builder = {
        "shared_system_prompt": _shared_system_prompt,
        "rag_template_reuse": _rag_template_reuse,
        "agent_trace_branching": _agent_trace_branching,
        "multi_tenant_skew": _multi_tenant_skew,
        "phase_shift_prompts": _phase_shift_prompts,
        "long_context_mixed": _long_context_mixed,
        "session_continuation_growth": _session_continuation_growth,
        "hotset_cold_scan": _hotset_cold_scan,
        "cyclic_working_set_pressure": _cyclic_working_set_pressure,
        "cyclic_working_set_pressure_shifted": _cyclic_working_set_pressure_shifted,
        "concurrent_long_generation": _concurrent_long_generation,
        "stochastic_serving_mix": _stochastic_serving_mix,
        "stochastic_serving_mix_shifted": _stochastic_serving_mix_shifted,
        "rolling_template_versions": _rolling_template_versions,
        "rolling_template_versions_shifted": _rolling_template_versions_shifted,
        "heavy_tailed_prefix_lengths": _heavy_tailed_prefix_lengths,
        "heavy_tailed_prefix_lengths_shifted": _heavy_tailed_prefix_lengths_shifted,
        "priority_burst_recovery": _priority_burst_recovery,
        "priority_burst_recovery_shifted": _priority_burst_recovery_shifted,
        "priority_one_off_noise": _priority_one_off_noise,
        "priority_one_off_noise_shifted": _priority_one_off_noise_shifted,
        "tenant_phase_shift_cycles": _tenant_phase_shift_cycles,
        "tenant_phase_shift_cycles_shifted": _tenant_phase_shift_cycles_shifted,
        "adversarial_unique_prompts": _adversarial_unique_prompts,
        "cross_family_mixture": _cross_family_mixture,
        "tenant_session_reentry": _tenant_session_reentry,
    }.get(family)
    if builder is None:
        raise ValueError(f"unknown workload family {family!r}")
    return tuple(builder(request_count, block_size_tokens, rng))


def scoring_fn_complexity(source: str) -> int:
    """Count AST nodes in the candidate policy implementation."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 10_000

    ignored_top_level_functions = {"build_candidate", "candidate_factory", "run_demo"}
    total = 0
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            total += sum(1 for _ in ast.walk(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ignored_top_level_functions:
                total += _nested_implementation_complexity(node)
            else:
                total += sum(1 for _ in ast.walk(node))
    return total


def _nested_implementation_complexity(node: ast.AST) -> int:
    """Counts policy implementations nested inside an ignored factory wrapper."""

    total = 0
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            total += sum(1 for _ in ast.walk(child))
        else:
            total += _nested_implementation_complexity(child)
    return total


def _build_policy(
    factory: Callable[..., PrefixKVPolicy],
    capacity_blocks: int,
    block_size_tokens: int,
    seed: int,
) -> PrefixKVPolicy:
    argument_options = (
        (capacity_blocks, block_size_tokens, seed),
        (capacity_blocks, block_size_tokens),
        (),
    )
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(*argument_options[0])

    for args in argument_options:
        try:
            signature.bind(*args)
        except TypeError:
            continue
        return factory(*args)
    raise TypeError(
        "candidate factory must accept (capacity_blocks, block_size_tokens, seed), "
        "(capacity_blocks, block_size_tokens), or no arguments"
    )


def _aggregate_by(
    keys: Iterable[str],
    trials: list[TrialMetrics],
) -> dict[str, dict[str, float | int | bool | str]]:
    grouped: dict[str, list[TrialMetrics]] = {}
    for key, trial in zip(keys, trials):
        grouped.setdefault(key, []).append(trial)
    return {key: _aggregate_trials(value) for key, value in grouped.items()}


def _aggregate_trials(
    trials: list[TrialMetrics],
) -> dict[str, float | int | bool | str]:
    if not trials:
        return {}
    numeric_fields = [
        "block_hit_rate",
        "token_hit_rate",
        "priority_weighted_token_hit_rate",
        "high_priority_token_hit_rate",
        "low_priority_token_hit_rate",
        "priority_request_fraction",
        "request_token_hit_rate_p10",
        "request_token_hit_rate_p50",
        "high_priority_request_token_hit_rate_p10",
        "worst_quarter_token_hit_rate",
        "final_quarter_token_hit_rate",
        "quarter_token_hit_rate_stddev",
        "prefill_tokens_saved",
        "recompute_tokens",
        "recompute_cost",
        "lookup_block_count",
        "lookup_blocks_per_request",
        "eviction_count",
        "admission_count",
        "admission_score_count",
        "admission_rejection_count",
        "admission_rate",
        "useful_admission_count",
        "useful_admission_rate",
        "wasted_admission_count",
        "wasted_admission_rate",
        "admitted_token_count",
        "useful_admission_token_count",
        "useful_admission_token_rate",
        "wasted_admission_token_count",
        "wasted_admission_token_rate",
        "admission_saved_tokens",
        "admission_saved_tokens_per_admission",
        "admission_token_utility",
        "evicted_without_hit_count",
        "evicted_without_hit_rate",
        "policy_bypass_tokens",
        "policy_bypass_token_rate",
        "cache_churn_per_1k",
        "forced_bypass_count",
        "forced_bypass_tokens",
        "forced_bypass_token_rate",
        "short_reuse_after_eviction_missed_tokens",
        "short_reuse_after_eviction_missed_token_rate",
        "eviction_reuse_distance_p50",
        "eviction_reuse_distance_p95",
        "avoidable_eviction_count",
        "avoidable_eviction_rate",
        "avoidable_short_reuse_eviction_count",
        "avoidable_short_reuse_eviction_rate",
        "tenant_count",
        "tenant_fairness_penalty",
        "tenant_token_hit_rate_p10",
        "tenant_jain_fairness",
        "p50_latency_proxy",
        "p95_latency_proxy",
        "p99_latency_proxy",
        "high_priority_p95_latency_proxy",
        "high_priority_p99_latency_proxy",
        "p95_recompute_cost",
        "recovery_request_count",
        "recovery_token_hit_rate",
        "recovery_p95_latency_proxy",
        "recovery_phase_count",
        "worst_recovery_phase_token_hit_rate",
        "final_recovery_phase_token_hit_rate",
        "worst_recovery_phase_p95_latency_proxy",
        "memory_occupancy_mean",
        "arrival_span_steps",
        "max_prefill_cost",
        "scoring_fn_complexity",
    ]
    result: dict[str, float | int | bool | str] = {
        field: mean(float(getattr(trial, field)) for trial in trials)
        for field in numeric_fields
    }
    result["memory_occupancy_peak"] = max(
        trial.memory_occupancy_peak for trial in trials
    )
    result["active_request_count_peak"] = max(
        trial.active_request_count_peak for trial in trials
    )
    token_hit_rates = [trial.token_hit_rate for trial in trials]
    result["token_hit_rate_worst_trial"] = min(token_hit_rates)
    result["token_hit_rate_p10_across_trials"] = _percentile(token_hit_rates, 10)
    result["token_hit_rate_stddev_across_trials"] = (
        pstdev(token_hit_rates) if len(token_hit_rates) > 1 else 0.0
    )
    result["p95_latency_proxy_worst_trial"] = max(
        trial.p95_latency_proxy for trial in trials
    )
    result["cache_churn_per_1k_worst_trial"] = max(
        trial.cache_churn_per_1k for trial in trials
    )
    result["invalid_fraction"] = sum(1 for trial in trials if trial.invalid) / len(
        trials
    )
    result["invalid"] = any(trial.invalid for trial in trials)
    result["invalid_reason"] = "; ".join(
        sorted({trial.invalid_reason for trial in trials if trial.invalid_reason})
    )
    structural_keys = sorted(
        {key for trial in trials for key in trial.structural_metrics}
    )
    for key in structural_keys:
        result[key] = mean(
            float(trial.structural_metrics.get(key, 0.0)) for trial in trials
        )
    return result


def _workload_base_score(
    trials: list[TrialMetrics],
    *,
    token_weight: float,
    block_weight: float,
    request_tail_weight: float,
    worst_window_weight: float,
    priority_hit_weight: float,
    wasted_admission_weight: float,
    admission_utility_weight: float,
    avoidable_eviction_weight: float,
    latency_weight: float,
    latency_cap: float,
    latency_norm: float,
) -> float:
    token_score = token_weight * mean(trial.token_hit_rate for trial in trials)
    block_score = block_weight * mean(trial.block_hit_rate for trial in trials)
    request_tail_score = request_tail_weight * mean(
        trial.request_token_hit_rate_p10 for trial in trials
    )
    worst_window_score = worst_window_weight * mean(
        trial.worst_quarter_token_hit_rate for trial in trials
    )
    priority_trials = [
        trial for trial in trials if trial.priority_request_fraction > 0.0
    ]
    priority_score = (
        priority_hit_weight
        * mean(trial.high_priority_token_hit_rate for trial in priority_trials)
        if priority_trials
        else 0.0
    )
    wasted_admission_cost = wasted_admission_weight * mean(
        trial.wasted_admission_token_rate for trial in trials
    )
    admission_utility_score = admission_utility_weight * mean(
        math.log1p(trial.admission_token_utility) for trial in trials
    )
    avoidable_eviction_cost = avoidable_eviction_weight * mean(
        trial.avoidable_eviction_rate for trial in trials
    )
    latency = mean(trial.p95_latency_proxy for trial in trials)
    if latency_norm <= 0.0:
        latency_norm = max(
            (trial.max_prefill_cost for trial in trials),
            default=1.0,
        )
    latency_cost = min(
        latency_cap,
        latency_weight * latency / max(latency_norm, 1.0),
    )
    return (
        token_score
        + block_score
        + request_tail_score
        + worst_window_score
        + priority_score
        + admission_utility_score
        - latency_cost
        - wasted_admission_cost
        - avoidable_eviction_cost
    )


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if percentile == 50:
        return float(median(values))
    values = sorted(values)
    index = math.ceil((percentile / 100.0) * len(values)) - 1
    return float(values[max(0, min(index, len(values) - 1))])


def _window_token_hit_rates(
    request_hit_records: list[tuple[int, int]],
    *,
    window_count: int,
) -> list[float]:
    """Returns token-weighted hit rates for contiguous request windows."""

    if not request_hit_records or window_count <= 0:
        return []
    effective_window_count = min(window_count, len(request_hit_records))
    window_hits = [0] * effective_window_count
    window_tokens = [0] * effective_window_count
    for index, (hit_tokens, total_tokens) in enumerate(request_hit_records):
        window = min(
            effective_window_count - 1,
            index * effective_window_count // len(request_hit_records),
        )
        window_hits[window] += hit_tokens
        window_tokens[window] += total_tokens
    return [
        hits / tokens if tokens else 0.0
        for hits, tokens in zip(window_hits, window_tokens, strict=True)
    ]


def _jain_fairness(values: list[float]) -> float:
    """Returns Jain's fairness index, treating all-zero service as equal."""

    if not values:
        return 1.0
    squared_sum = sum(value * value for value in values)
    if squared_sum == 0.0:
        return 1.0
    return sum(values) ** 2 / (len(values) * squared_sum)


def _stable_hash(value: object) -> int:
    digest = hashlib.blake2b(repr(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _request_prefix_hashes(
    request: WorkloadRequest,
    block_size_tokens: int,
) -> list[int]:
    prefix_hashes: list[int] = []
    prefix_tokens: list[int] = []
    tokens = request.prompt_tokens or request.info.prompt_tokens
    for start in range(0, len(tokens), block_size_tokens):
        chunk = tokens[start : start + block_size_tokens]
        prefix_tokens.extend(chunk)
        prefix_hashes.append(
            _stable_hash((request.info.tenant_id, tuple(prefix_tokens)))
        )
    return prefix_hashes


def _depth_band(depth: int) -> str:
    for name, low, high in _DEPTH_BANDS:
        if depth >= low and (high is None or depth <= high):
            return name
    return _DEPTH_BANDS[0][0]


def _structural_metrics(
    *,
    depth_total_blocks: dict[str, int],
    depth_hit_blocks: dict[str, int],
    depth_total_tokens: dict[str, int],
    depth_hit_tokens: dict[str, int],
    prefix_role_hit_tokens: dict[str, int],
    total_hit_tokens: int,
    high_descendant_evictions: int,
    eviction_count: int,
    cold_deep_admission_opportunities: int,
    cold_deep_admissions: int,
    reuse_after_eviction_missed_blocks: int,
    reuse_after_eviction_missed_tokens: int,
    recompute_tokens: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for band, _, _ in _DEPTH_BANDS:
        total_blocks = depth_total_blocks.get(band, 0)
        total_tokens = depth_total_tokens.get(band, 0)
        hit_blocks = depth_hit_blocks.get(band, 0)
        hit_tokens = depth_hit_tokens.get(band, 0)
        metrics[f"{band}_block_hit_rate"] = (
            hit_blocks / total_blocks if total_blocks else 0.0
        )
        metrics[f"{band}_token_hit_rate"] = (
            hit_tokens / total_tokens if total_tokens else 0.0
        )
        metrics[f"{band}_recompute_tokens_saved"] = float(hit_tokens)

    metrics["high_descendant_eviction_count"] = float(high_descendant_evictions)
    metrics["high_descendant_eviction_rate"] = (
        high_descendant_evictions / eviction_count if eviction_count else 0.0
    )
    metrics["cold_deep_admission_opportunities"] = float(
        cold_deep_admission_opportunities
    )
    metrics["cold_deep_admission_count"] = float(cold_deep_admissions)
    metrics["cold_deep_admission_rate"] = (
        cold_deep_admissions / cold_deep_admission_opportunities
        if cold_deep_admission_opportunities
        else 0.0
    )
    metrics["reuse_after_eviction_missed_blocks"] = float(
        reuse_after_eviction_missed_blocks
    )
    metrics["reuse_after_eviction_missed_tokens"] = float(
        reuse_after_eviction_missed_tokens
    )
    metrics["reuse_after_eviction_missed_token_rate"] = (
        reuse_after_eviction_missed_tokens / recompute_tokens
        if recompute_tokens
        else 0.0
    )
    for role in _PREFIX_ROLES:
        hit_tokens = prefix_role_hit_tokens.get(role, 0)
        metrics[f"{role}_prefix_hit_tokens"] = float(hit_tokens)
        metrics[f"{role}_prefix_hit_contribution"] = (
            hit_tokens / total_hit_tokens if total_hit_tokens else 0.0
        )
    return metrics


def _prefix_role(tokens: tuple[int, ...]) -> str:
    roles = {
        _TOKEN_PREFIX_ROLES[token] for token in tokens if token in _TOKEN_PREFIX_ROLES
    }
    if len(roles) == 1:
        return roles.pop()
    return "unknown"


def _prefix_role_from_label(label: str) -> str:
    if any(
        marker in label
        for marker in (
            "tail",
            "query",
            "tool",
            "retry",
            "turn",
            "scan",
            "unique",
        )
    ):
        return "user"
    if any(
        marker in label
        for marker in (
            "shared-system",
            "rag/template",
            "agent/root",
            "/root/",
        )
    ):
        return "system"
    if any(
        marker in label
        for marker in (
            "shared-task",
            "rag/chunk",
            "agent/branch",
            "schema",
            "/branch/",
            "doc/",
        )
    ):
        return "developer"
    return "unknown"


def _block(
    label: str,
    block_size_tokens: int,
    token_count: int | None = None,
) -> tuple[int, ...]:
    count = block_size_tokens if token_count is None else max(1, token_count)
    base = _stable_hash(label) % 1_000_000
    tokens = tuple(base + index for index in range(count))
    role = _prefix_role_from_label(label)
    if role != "unknown":
        for token in tokens:
            _TOKEN_PREFIX_ROLES[token] = role
    return tokens


def _partial_tail(label: str, block_size_tokens: int) -> tuple[int, ...]:
    token_count = 1 + (_stable_hash(label) % max(block_size_tokens - 1, 1))
    return _block(label, block_size_tokens, token_count=token_count)


def _request(
    *,
    request_id: int,
    tenant_id: int,
    session_id: int,
    blocks: list[tuple[int, ...]],
    request_type: str,
    priority: int = 0,
    true_output_length: int = 96,
    predicted_output_length: int | None = None,
    arrival_step: int | None = None,
) -> WorkloadRequest:
    tokens = tuple(token for block in blocks for token in block)
    return WorkloadRequest(
        info=RequestInfo(
            request_id=request_id,
            tenant_id=tenant_id,
            session_id=session_id,
            prompt_length=len(tokens),
            priority=priority,
            request_type=request_type,
            prompt_tokens=(),
            predicted_output_length=predicted_output_length,
        ),
        true_output_length=true_output_length,
        prompt_tokens=tokens,
        arrival_step=arrival_step,
    )


def _reindex_request(
    request: WorkloadRequest,
    *,
    request_id: int,
    request_type: str,
    arrival_step: int | None = None,
) -> WorkloadRequest:
    """Copies a workload request with a new position and descriptive type."""
    info = request.info
    return WorkloadRequest(
        info=RequestInfo(
            request_id=request_id,
            tenant_id=info.tenant_id,
            session_id=info.session_id,
            prompt_length=info.prompt_length,
            priority=info.priority,
            request_type=request_type,
            prompt_tokens=info.prompt_tokens,
            predicted_output_length=info.predicted_output_length,
        ),
        true_output_length=request.true_output_length,
        prompt_tokens=request.prompt_tokens,
        arrival_step=request.arrival_step if arrival_step is None else arrival_step,
    )


def _shared_system_prompt(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    system = [
        _block("shared-system/a", block_size),
        _block("shared-system/b", block_size),
    ]
    tasks = [_block(f"shared-task/{idx}", block_size) for idx in range(5)]
    requests = []
    for request_id in range(count):
        task = tasks[request_id % len(tasks)]
        tail = _partial_tail(f"shared-tail/{request_id % 11}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % 8,
                blocks=[*system, task, tail],
                request_type="chat",
                true_output_length=64 + rng.randrange(96),
            )
        )
    return requests


def _rag_template_reuse(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    template = [
        _block("rag/template/a", block_size),
        _block("rag/template/b", block_size),
    ]
    chunks = [_block(f"rag/chunk/{idx}", block_size) for idx in range(8)]
    requests = []
    for request_id in range(count):
        chunk = chunks[(request_id // 2 + request_id) % len(chunks)]
        suffix = _partial_tail(f"rag/query/{request_id % 17}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % 13,
                blocks=[*template, chunk, suffix],
                request_type="rag",
                true_output_length=48 + rng.randrange(80),
            )
        )
    return requests


def _long_context_mixed(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    docs = [
        [_block(f"doc/{doc}/block/{idx}", block_size) for idx in range(6)]
        for doc in range(4)
    ]
    requests = []
    for request_id in range(count):
        doc = docs[(request_id // 3) % len(docs)]
        length = 3 + (request_id % 4)
        tail = _partial_tail(f"doc/tail/{request_id % 19}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % 9,
                blocks=[*doc[:length], tail],
                request_type="long_context",
                true_output_length=96 + rng.randrange(160),
            )
        )
    return requests


def _session_continuation_growth(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    system = [
        _block("session/shared-system/a", block_size),
        _block("session/shared-system/b", block_size),
    ]
    session_count = 4
    session_roots = {
        session_id: _block(f"session/{session_id}/root", block_size)
        for session_id in range(session_count)
    }
    histories: dict[int, list[tuple[int, ...]]] = {
        session_id: [] for session_id in range(session_count)
    }
    requests = []
    for request_id in range(count):
        session_id = request_id % session_count
        turn = _block(
            f"session/{session_id}/turn/{len(histories[session_id])}",
            block_size,
        )
        histories[session_id].append(turn)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=session_id,
                blocks=[*system, session_roots[session_id], *histories[session_id]],
                request_type="session_continuation",
                true_output_length=64 + rng.randrange(128),
            )
        )
    return requests


def _agent_trace_branching(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    root = [_block("agent/root/a", block_size), _block("agent/root/b", block_size)]
    schema = [_block(f"agent/schema/{index}", block_size) for index in range(3)]
    branches = [_block(f"agent/branch/{idx}", block_size) for idx in range(4)]
    tool_calls = [_block(f"agent/tool-call/{idx}", block_size) for idx in range(6)]
    tool_results = [
        _block(f"agent/tool-result/shared/{idx}", block_size) for idx in range(8)
    ]
    histories: dict[int, list[tuple[int, ...]]] = {
        branch_index: [] for branch_index in range(len(branches))
    }
    requests = []
    for request_id in range(count):
        branch_idx = (request_id // 2 + request_id) % len(branches)
        history = list(histories[branch_idx])
        request_type = "agent_loop"
        if request_id % 11 == 10 and len(history) >= 4:
            history = history[:-2]
            request_type = "agent_retry"

        loop_count = 1 + int(request_id % 3 == 0)
        for loop_index in range(loop_count):
            tool_index = (request_id + branch_idx * 3 + loop_index) % len(tool_calls)
            history.append(tool_calls[tool_index])
            if (request_id + loop_index) % 4 == 0:
                result_index = (tool_index + request_id // 4) % len(tool_results)
                history.append(tool_results[result_index])
            else:
                history.append(
                    _block(
                        f"agent/tool-result/unique/{branch_idx}/{request_id}/{loop_index}",
                        block_size,
                    )
                )
        histories[branch_idx] = history
        tail = _partial_tail(f"agent/tail/{branch_idx}/{request_id}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=branch_idx,
                blocks=[*root, *schema, branches[branch_idx], *history, tail],
                request_type=request_type,
                true_output_length=96 + rng.randrange(192),
            )
        )
    return requests


def _multi_tenant_skew(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    tenant_roots = {
        tenant: [_block(f"tenant/{tenant}/root/{idx}", block_size) for idx in range(2)]
        for tenant in range(3)
    }
    requests = []
    for request_id in range(count):
        tenant = (
            0 if request_id % 6 in {0, 1, 2, 3} else (1 if request_id % 6 == 4 else 2)
        )
        branch = _block(f"tenant/{tenant}/branch/{request_id % 5}", block_size)
        tail = _partial_tail(f"tenant/{tenant}/tail/{request_id % 13}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=tenant,
                session_id=tenant * 100 + request_id % 9,
                blocks=[*tenant_roots[tenant], branch, tail],
                request_type="tenant",
                true_output_length=64 + rng.randrange(128),
            )
        )
    return requests


def _phase_shift_prompts(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    phases = [
        [_block(f"phase/{phase}/root/{idx}", block_size) for idx in range(2)]
        for phase in range(2)
    ]
    requests = []
    for request_id in range(count):
        phase = 0 if request_id < count // 2 else 1
        branch = _block(f"phase/{phase}/branch/{request_id % 6}", block_size)
        tail = _partial_tail(f"phase/{phase}/tail/{request_id % 11}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % 10,
                blocks=[*phases[phase], branch, tail],
                request_type="phase_shift",
                true_output_length=64 + rng.randrange(128),
            )
        )
    return requests


def _hotset_cold_scan(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    hot_root = [
        _block("hotset/root/a", block_size),
        _block("hotset/root/b", block_size),
    ]
    hot_prompts = [
        [
            *hot_root,
            _block(f"hotset/branch/{index}", block_size),
            _partial_tail(f"hotset/tail/{index}", block_size),
        ]
        for index in range(4)
    ]
    warm_count = count // 3
    scan_end = 2 * warm_count
    requests = []
    for request_id in range(count):
        if warm_count <= request_id < scan_end:
            blocks = [
                _block(f"scan/{request_id}/block/{index}", block_size)
                for index in range(4)
            ]
            blocks[-1] = _partial_tail(f"scan/{request_id}/tail", block_size)
            request_type = "cold_scan"
        else:
            hot_index = request_id % len(hot_prompts)
            blocks = hot_prompts[hot_index]
            request_type = "hotset"
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % len(hot_prompts),
                blocks=blocks,
                request_type=request_type,
                true_output_length=32 + rng.randrange(64),
            )
        )
    return requests


def _cyclic_working_set_pressure(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _cyclic_working_set_pressure_workload(
        count,
        block_size,
        rng,
        shifted=False,
    )


def _cyclic_working_set_pressure_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _cyclic_working_set_pressure_workload(
        count,
        block_size,
        rng,
        shifted=True,
    )


def _cyclic_working_set_pressure_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    """Builds cyclic working sets slightly larger than common cache capacities."""

    root = [
        _block("cyclic/root/system", block_size),
        _block("cyclic/root/instructions", block_size),
    ]
    small_set_size = 12 if shifted else 9
    large_set_size = 22 if shifted else 17
    prompt_count = large_set_size
    prompts = [
        [
            *root,
            _block(f"cyclic/prompt/{index}/branch", block_size),
            _block(f"cyclic/prompt/{index}/context", block_size),
            _partial_tail(f"cyclic/prompt/{index}/tail", block_size),
        ]
        for index in range(prompt_count)
    ]
    requests = []
    for request_id in range(count):
        in_large_phase = request_id >= count // 2
        working_set_size = large_set_size if in_large_phase else small_set_size
        cycle_position = request_id if not shifted else request_id * 5
        prompt_index = cycle_position % working_set_size
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=prompt_index,
                blocks=prompts[prompt_index],
                request_type=(
                    "cyclic_working_set_large"
                    if in_large_phase
                    else "cyclic_working_set_small"
                ),
                true_output_length=48 + rng.randrange(96),
            )
        )
    return requests


def _concurrent_long_generation(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    root = [
        _block("concurrent/root/a", block_size),
        _block("concurrent/root/b", block_size),
    ]
    branches = [_block(f"concurrent/branch/{index}", block_size) for index in range(12)]
    requests = []
    for request_id in range(count):
        branch_index = request_id % len(branches)
        predicted_output_length = 512 + 64 * (request_id % 4)
        true_output_length = predicted_output_length + rng.randrange(-64, 65)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=branch_index,
                blocks=[
                    *root,
                    branches[branch_index],
                    _partial_tail(f"concurrent/tail/{request_id % 24}", block_size),
                ],
                request_type="long_generation",
                true_output_length=true_output_length,
                predicted_output_length=predicted_output_length,
                arrival_step=request_id // 2,
            )
        )
    return requests


def _stochastic_serving_mix(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _stochastic_serving_mix_workload(count, block_size, rng, shifted=False)


def _stochastic_serving_mix_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _stochastic_serving_mix_workload(count, block_size, rng, shifted=True)


def _stochastic_serving_mix_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    source_requests = {
        "chat": _shared_system_prompt(count, block_size, rng),
        "rag": _rag_template_reuse(count, block_size, rng),
        "agent": _agent_trace_branching(count, block_size, rng),
        "long": _concurrent_long_generation(count, block_size, rng),
        "oneoff": _adversarial_unique_prompts(count, block_size, rng),
    }
    source_indices = {source_name: 0 for source_name in source_requests}
    if shifted:
        regimes = (
            (
                ("chat", 0.20),
                ("rag", 0.15),
                ("agent", 0.35),
                ("long", 0.20),
                ("oneoff", 0.10),
            ),
            (
                ("chat", 0.10),
                ("rag", 0.10),
                ("agent", 0.25),
                ("long", 0.35),
                ("oneoff", 0.20),
            ),
            (
                ("chat", 0.10),
                ("rag", 0.10),
                ("agent", 0.15),
                ("long", 0.20),
                ("oneoff", 0.45),
            ),
        )
        burst_probability = 0.70
        max_burst_length = 7
        arrival_gaps = (0, 1, 2, 5)
        arrival_gap_weights = (0.55, 0.30, 0.10, 0.05)
    else:
        regimes = (
            (
                ("chat", 0.45),
                ("rag", 0.25),
                ("agent", 0.15),
                ("long", 0.10),
                ("oneoff", 0.05),
            ),
            (
                ("chat", 0.15),
                ("rag", 0.20),
                ("agent", 0.35),
                ("long", 0.20),
                ("oneoff", 0.10),
            ),
            (
                ("chat", 0.25),
                ("rag", 0.15),
                ("agent", 0.10),
                ("long", 0.15),
                ("oneoff", 0.35),
            ),
        )
        burst_probability = 0.55
        max_burst_length = 5
        arrival_gaps = (0, 1, 2, 5)
        arrival_gap_weights = (0.35, 0.45, 0.15, 0.05)

    requests = []
    active_source = ""
    remaining_burst = 0
    arrival_step = 0
    for request_id in range(count):
        if request_id:
            arrival_step += rng.choices(arrival_gaps, weights=arrival_gap_weights, k=1)[
                0
            ]
        regime_index = min(2, request_id * len(regimes) // max(1, count))
        choices, weights = zip(*regimes[regime_index], strict=True)
        if remaining_burst <= 0:
            active_source = rng.choices(choices, weights=weights, k=1)[0]
            if rng.random() < burst_probability:
                remaining_burst = rng.randrange(2, max_burst_length + 1) - 1
        else:
            remaining_burst -= 1

        source_index = source_indices[active_source]
        source_request = source_requests[active_source][source_index % count]
        source_indices[active_source] += 1
        requests.append(
            _reindex_request(
                source_request,
                request_id=request_id,
                request_type=f"mix_{active_source}_{source_request.info.request_type}",
                arrival_step=arrival_step,
            )
        )
    return requests


def _rolling_template_versions(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _rolling_template_versions_workload(count, block_size, rng, shifted=False)


def _rolling_template_versions_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _rolling_template_versions_workload(count, block_size, rng, shifted=True)


def _rolling_template_versions_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    shared_root = _block("rolling/root/shared", block_size)
    templates = {
        0: [
            shared_root,
            _block("rolling/version/0/instructions", block_size),
            _block("rolling/version/0/schema", block_size),
        ],
        1: [
            shared_root,
            _block("rolling/version/1/instructions", block_size),
            _block("rolling/version/1/schema", block_size),
        ],
        2: [
            _block("rolling/root/revised", block_size),
            _block("rolling/version/2/instructions", block_size),
            _block("rolling/version/2/schema", block_size),
        ],
    }
    tasks = [_block(f"rolling/task/{index}", block_size) for index in range(6)]
    requests = []
    for request_id in range(count):
        phase = min(3, request_id * 4 // max(1, count))
        if shifted:
            if phase == 0:
                version = 0
            elif phase == 1:
                version = int(request_id % 3 == 0)
            elif phase == 2:
                version = 1
            else:
                version = 2 if request_id % 3 else 1
        elif phase == 0:
            version = 0
        elif phase == 1:
            version = int(request_id % 4 == 0)
        elif phase == 2:
            version = int(request_id % 4 != 0)
        else:
            version = int(request_id % 5 == 0)

        task_index = (request_id + version * 2) % len(tasks)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=request_id % 12,
                blocks=[
                    *templates[version],
                    tasks[task_index],
                    _partial_tail(f"rolling/tail/{request_id}", block_size),
                ],
                request_type=f"rolling_template_v{version}",
                true_output_length=80 + rng.randrange(112),
            )
        )
    return requests


def _heavy_tailed_prefix_lengths(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _heavy_tailed_prefix_lengths_workload(count, block_size, rng, shifted=False)


def _heavy_tailed_prefix_lengths_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _heavy_tailed_prefix_lengths_workload(count, block_size, rng, shifted=True)


def _heavy_tailed_prefix_lengths_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    alpha = 1.25 if shifted else 1.55
    max_depth = 32 if shifted else 20
    doc_count = 6 if shifted else 4
    docs = [
        [
            _block(f"heavy/doc/{doc_index}/chunk/{depth}", block_size)
            for depth in range(max_depth)
        ]
        for doc_index in range(doc_count)
    ]
    doc_weights = list(range(doc_count, 0, -1))
    root = [
        _block("heavy/root/system", block_size),
        _block("heavy/root/instructions", block_size),
    ]
    requests = []
    for request_id in range(count):
        doc_index = rng.choices(range(doc_count), weights=doc_weights, k=1)[0]
        body_depth = min(
            max_depth,
            max(2, int(1 + rng.paretovariate(alpha) * (3 if shifted else 2))),
        )
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=request_id % 3,
                session_id=request_id % 16,
                blocks=[
                    *root,
                    *docs[doc_index][:body_depth],
                    _partial_tail(f"heavy/tail/{request_id}", block_size),
                ],
                request_type="heavy_tailed_prefix",
                true_output_length=64 + rng.randrange(224),
            )
        )
    return requests


def _priority_burst_recovery(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _priority_burst_recovery_workload(count, block_size, rng, shifted=False)


def _priority_burst_recovery_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _priority_burst_recovery_workload(count, block_size, rng, shifted=True)


def _priority_burst_recovery_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    high_priority = 5 if shifted else 3
    medium_priority = 2 if shifted else 1
    hot_prompt_count = 6 if shifted else 4
    scan_depth = 8 if shifted else 6
    high_during_burst_interval = 7 if shifted else 5
    root = [
        _block("priority/root/system", block_size),
        _block("priority/root/instructions", block_size),
    ]
    hot_prompts = [
        [
            *root,
            _block(f"priority/hot/{index}/branch", block_size),
            _block(f"priority/hot/{index}/context", block_size),
            _partial_tail(f"priority/hot/{index}/tail", block_size),
        ]
        for index in range(hot_prompt_count)
    ]
    medium_prompts = [
        [
            *root,
            _block(f"priority/medium/{index}/branch", block_size),
            _partial_tail(f"priority/medium/{index}/tail", block_size),
        ]
        for index in range(3)
    ]
    warm_end = max(1, count // 4)
    burst_end = max(warm_end + 1, 3 * count // 4)
    requests = []
    arrival_step = 0
    for request_id in range(count):
        if request_id < warm_end:
            hot_index = request_id % len(hot_prompts)
            blocks = hot_prompts[hot_index]
            priority = high_priority
            request_type = "priority_hot_warm"
        elif request_id < burst_end:
            burst_index = request_id - warm_end
            if burst_index % high_during_burst_interval == 0:
                hot_index = (request_id + burst_index // 2) % len(hot_prompts)
                blocks = hot_prompts[hot_index]
                priority = high_priority
                request_type = "priority_hot_during_burst"
            else:
                blocks = [
                    _block(
                        f"priority/background/{request_id}/block/{depth}",
                        block_size,
                    )
                    for depth in range(scan_depth)
                ]
                blocks[-1] = _partial_tail(
                    f"priority/background/{request_id}/tail", block_size
                )
                priority = 0
                request_type = "priority_background_scan"
        elif request_id % 5 == 0:
            medium_index = request_id % len(medium_prompts)
            blocks = medium_prompts[medium_index]
            priority = medium_priority
            request_type = "priority_medium_recovery"
        else:
            hot_index = request_id % len(hot_prompts)
            blocks = hot_prompts[hot_index]
            priority = high_priority
            request_type = "priority_hot_recovery"

        if request_id:
            arrival_step += (
                rng.choice((0, 0, 1))
                if warm_end <= request_id < burst_end
                else rng.choice((1, 1, 2))
            )
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=priority * 100 + request_id % max(1, hot_prompt_count),
                blocks=blocks,
                request_type=request_type,
                priority=priority,
                true_output_length=(
                    128 + rng.randrange(160) if priority > 0 else 24 + rng.randrange(48)
                ),
                arrival_step=arrival_step,
            )
        )
    return requests


def _priority_one_off_noise(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _priority_one_off_noise_workload(count, block_size, rng, shifted=False)


def _priority_one_off_noise_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _priority_one_off_noise_workload(count, block_size, rng, shifted=True)


def _priority_one_off_noise_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    """Mixes reusable normal traffic with high-priority one-off requests."""

    normal_root = [
        _block("priority-noise/normal/root/system", block_size),
        _block("priority-noise/normal/root/instructions", block_size),
    ]
    normal_prompt_count = 7 if shifted else 5
    normal_prompts = [
        [
            *normal_root,
            _block(f"priority-noise/normal/{index}/branch", block_size),
            _block(f"priority-noise/normal/{index}/context", block_size),
            _partial_tail(f"priority-noise/normal/{index}/tail", block_size),
        ]
        for index in range(normal_prompt_count)
    ]
    high_priority_root = _block("priority-noise/high/root/shared", block_size)
    sequence_length = 5
    high_priority_positions = {2, 3, 4} if shifted else {3, 4}
    high_priority = 6 if shifted else 4
    unique_depth = 8 if shifted else 6
    requests = []
    for request_id in range(count):
        if request_id % sequence_length in high_priority_positions:
            blocks = [
                high_priority_root,
                *[
                    _block(
                        f"priority-noise/high/{request_id}/unique/{depth}",
                        block_size,
                    )
                    for depth in range(unique_depth - 1)
                ],
            ]
            blocks[-1] = _partial_tail(
                f"priority-noise/high/{request_id}/tail",
                block_size,
            )
            priority = high_priority
            request_type = "priority_one_off_noise"
            session_id = 10_000 + request_id
        else:
            normal_index = (request_id // sequence_length + request_id) % len(
                normal_prompts
            )
            blocks = normal_prompts[normal_index]
            priority = 0
            request_type = "priority_normal_recurring"
            session_id = normal_index
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=session_id,
                blocks=blocks,
                request_type=request_type,
                priority=priority,
                true_output_length=(
                    160 + rng.randrange(192) if priority > 0 else 48 + rng.randrange(80)
                ),
            )
        )
    return requests


def _tenant_phase_shift_cycles(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _tenant_phase_shift_cycles_workload(count, block_size, rng, shifted=False)


def _tenant_phase_shift_cycles_shifted(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    return _tenant_phase_shift_cycles_workload(count, block_size, rng, shifted=True)


def _tenant_phase_shift_cycles_workload(
    count: int,
    block_size: int,
    rng: random.Random,
    *,
    shifted: bool,
) -> list[WorkloadRequest]:
    """Builds repeated tenant phase shifts with pollution and delayed recovery."""

    tenant_count = 5 if shifted else 4
    hot_prompt_count = 5 if shifted else 4
    pollution_depth = 10 if shifted else 7
    cycle_count = 8 if shifted else 6
    tenant_roots = {
        tenant: [
            _block(f"tenant-cycle/{tenant}/root/system", block_size),
            _block(f"tenant-cycle/{tenant}/root/instructions", block_size),
        ]
        for tenant in range(tenant_count)
    }
    hot_prompts = {
        tenant: [
            [
                *tenant_roots[tenant],
                _block(f"tenant-cycle/{tenant}/hot/{index}/branch", block_size),
                _block(f"tenant-cycle/{tenant}/hot/{index}/context", block_size),
                _partial_tail(f"tenant-cycle/{tenant}/hot/{index}/tail", block_size),
            ]
            for index in range(hot_prompt_count)
        ]
        for tenant in range(tenant_count)
    }
    cycle_length = max(12, math.ceil(count / cycle_count))
    warm_length = max(3, cycle_length // 4)
    pollution_end = max(warm_length + 3, 3 * cycle_length // 4)
    requests = []
    arrival_step = 0
    for request_id in range(count):
        cycle = min(cycle_count - 1, request_id // cycle_length)
        cycle_offset = request_id - cycle * cycle_length
        active_tenant = cycle % tenant_count
        if cycle_offset < warm_length:
            hot_index = (request_id + cycle) % hot_prompt_count
            tenant = active_tenant
            blocks = hot_prompts[tenant][hot_index]
            request_type = "tenant_cycle_warm"
            output_length = 96 + rng.randrange(160)
        elif cycle_offset < pollution_end:
            tenant = (active_tenant + 1 + cycle_offset) % tenant_count
            blocks = [
                *tenant_roots[tenant],
                *[
                    _block(
                        f"tenant-cycle/{cycle}/pollution/{request_id}/{depth}",
                        block_size,
                    )
                    for depth in range(pollution_depth - 2)
                ],
            ]
            blocks[-1] = _partial_tail(
                f"tenant-cycle/{cycle}/pollution/{request_id}/tail",
                block_size,
            )
            request_type = "tenant_cycle_pollution"
            output_length = 24 + rng.randrange(64)
        else:
            hot_index = (request_id + cycle) % hot_prompt_count
            tenant = active_tenant
            blocks = hot_prompts[tenant][hot_index]
            request_type = "tenant_cycle_recovery"
            output_length = 96 + rng.randrange(160)

        if request_id:
            arrival_step += (
                rng.choice((0, 0, 1))
                if request_type == "tenant_cycle_pollution"
                else rng.choice((1, 1, 2, 3))
            )
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=tenant,
                session_id=tenant * 100 + request_id % hot_prompt_count,
                blocks=blocks,
                request_type=request_type,
                true_output_length=output_length,
                arrival_step=arrival_step,
            )
        )
    return requests


def _adversarial_unique_prompts(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    requests = []
    for request_id in range(count):
        blocks = [
            _block(
                f"unique/{request_id}/block/{idx}/{rng.randrange(10_000)}", block_size
            )
            for idx in range(4)
        ]
        blocks[-1] = _partial_tail(
            f"unique/{request_id}/block/partial/{rng.randrange(10_000)}",
            block_size,
        )
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=request_id % 4,
                session_id=request_id,
                blocks=blocks,
                request_type="adversarial",
                true_output_length=32 + rng.randrange(64),
            )
        )
    return requests


def _tenant_session_reentry(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    tenant_roots = {
        tenant: [
            _block(f"reentry/tenant/{tenant}/root/{index}", block_size)
            for index in range(2)
        ]
        for tenant in range(3)
    }
    session_contexts = {
        (tenant, session): [
            _block(
                f"reentry/tenant/{tenant}/session/{session}/context/{index}",
                block_size,
            )
            for index in range(3)
        ]
        for tenant in range(3)
        for session in range(4)
    }
    tenant_pattern = (0, 1, 0, 2, 0, 1, 2, 0)
    visits = {key: 0 for key in session_contexts}
    requests = []
    for request_id in range(count):
        tenant = tenant_pattern[request_id % len(tenant_pattern)]
        session = (request_id // len(tenant_pattern) + 3 * tenant) % 4
        visits[(tenant, session)] += 1
        stable_context = session_contexts[(tenant, session)]
        stable_depth = 2 if visits[(tenant, session)] == 1 else 3
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=tenant,
                session_id=tenant * 100 + session,
                blocks=[
                    *tenant_roots[tenant],
                    *stable_context[:stable_depth],
                    _partial_tail(
                        f"reentry/tail/{tenant}/{session}/{request_id}",
                        block_size,
                    ),
                ],
                request_type="tenant_session_reentry",
                true_output_length=48 + rng.randrange(96),
            )
        )
    return requests


def _cross_family_mixture(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    shared = _shared_system_prompt(count // 3, block_size, rng)
    phase = _phase_shift_prompts(count // 3, block_size, rng)
    unique = _adversarial_unique_prompts(
        count - len(shared) - len(phase), block_size, rng
    )
    requests = []
    for request_id, request in enumerate([*shared, *phase, *unique]):
        info = request.info
        requests.append(
            WorkloadRequest(
                info=RequestInfo(
                    request_id=request_id,
                    tenant_id=info.tenant_id,
                    session_id=info.session_id,
                    prompt_length=info.prompt_length,
                    priority=info.priority,
                    request_type=f"hidden_{info.request_type}",
                    prompt_tokens=info.prompt_tokens,
                    predicted_output_length=info.predicted_output_length,
                ),
                true_output_length=request.true_output_length,
                prompt_tokens=request.prompt_tokens,
                arrival_step=request.arrival_step,
            )
        )
    return requests
