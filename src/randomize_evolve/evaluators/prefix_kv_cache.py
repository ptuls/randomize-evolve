"""Deterministic prefix KV-cache evaluator for scoring eviction heuristics."""

from __future__ import annotations

import ast
import hashlib
import math
import random
from dataclasses import dataclass, field
from statistics import mean, median
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
_PREFIX_ROLES = ("system", "developer", "user")
_TOKEN_PREFIX_ROLES: dict[int, str] = {}


@dataclass(frozen=True)
class WorkloadRequest:
    """Simulator-internal request, including true output length."""

    info: RequestInfo
    true_output_length: int
    prompt_tokens: tuple[int, ...] = ()


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
    )
    validation_families: tuple[str, ...] = (
        "agent_trace_branching",
        "phase_shift_prompts",
        "multi_tenant_skew",
    )
    hidden_families: tuple[str, ...] = (
        "adversarial_unique_prompts",
        "cross_family_mixture",
    )
    request_count: int = 96
    prefill_cost_per_token: float = 1.0
    lookup_cost_per_block: float = 0.035
    eviction_cost_per_block: float = 0.2
    active_tokens_per_step: int = 64
    w_avg_tok: float = 80.0
    w_avg_blk: float = 60.0
    min_workload_weight: float = 0.5
    latency_norm: float = 0.0
    latency_weight: float = 35.0
    latency_cap: float = 40.0
    churn_weight: float = 0.035
    churn_cap: float = 25.0
    fairness_weight: float = 80.0
    fairness_cap: float = 30.0
    k_complex: float = 0.025
    complex_cap: float = 15.0
    # Keep abs(v_min) above the sum of valid penalty caps so invalid candidates
    # remain strictly worse than every valid policy.
    v_min: float = -120.0
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
                configs.append(
                    WorkloadConfig(
                        family=family,
                        split=split,
                        request_count=self.request_count,
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
    prefill_tokens_saved: float = 0.0
    recompute_tokens: float = 0.0
    recompute_cost: float = 0.0
    eviction_count: int = 0
    admission_count: int = 0
    cache_churn_per_1k: float = 0.0
    forced_bypass_count: int = 0
    tenant_fairness_penalty: float = 0.0
    p50_latency_proxy: float = 0.0
    p95_latency_proxy: float = 0.0
    p99_latency_proxy: float = 0.0
    memory_occupancy_mean: float = 0.0
    memory_occupancy_peak: int = 0
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
            "prefill_tokens_saved": self.prefill_tokens_saved,
            "recompute_tokens": self.recompute_tokens,
            "recompute_cost": self.recompute_cost,
            "eviction_count": self.eviction_count,
            "admission_count": self.admission_count,
            "cache_churn_per_1k": self.cache_churn_per_1k,
            "forced_bypass_count": self.forced_bypass_count,
            "tenant_fairness_penalty": self.tenant_fairness_penalty,
            "p50_latency_proxy": self.p50_latency_proxy,
            "p95_latency_proxy": self.p95_latency_proxy,
            "p99_latency_proxy": self.p99_latency_proxy,
            "memory_occupancy_mean": self.memory_occupancy_mean,
            "memory_occupancy_peak": self.memory_occupancy_peak,
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
    resident_children: set[int] = field(default_factory=set)
    known_children: set[int] = field(default_factory=set)

    @property
    def block_id(self) -> int:
        return self.prefix_hash


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
        self._future_positions: dict[int, list[int]] = {}
        if not enabled:
            return

        for now, request in enumerate(requests):
            for prefix_hash in _request_prefix_hashes(request, block_size_tokens):
                self._remaining_counts[prefix_hash] = (
                    self._remaining_counts.get(prefix_hash, 0) + 1
                )
                self._future_positions.setdefault(prefix_hash, []).append(now)

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
            while positions and positions[0] <= now:
                positions.pop(0)

    def remaining_count(self, prefix_hash: int) -> float | None:
        if not self.enabled:
            return None
        return float(self._remaining_counts.get(prefix_hash, 0))

    def next_distance(self, prefix_hash: int, now: int) -> float | None:
        if not self.enabled:
            return None
        positions = self._future_positions.get(prefix_hash) or []
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
    ) -> None:
        self.capacity_blocks = capacity_blocks
        self.block_size_tokens = block_size_tokens
        self.prefill_cost_per_token = prefill_cost_per_token
        self.lookup_cost_per_block = lookup_cost_per_block
        self.eviction_cost_per_block = eviction_cost_per_block
        self.active_tokens_per_step = active_tokens_per_step
        self.expose_future_reuse = expose_future_reuse
        self.blocks: dict[int, _BlockState] = {}
        self._release_events: dict[int, list[int]] = {}
        self._resident_hashes: set[int] = set()
        self._leaf_hashes: set[int] = set()
        self._descendant_counts: dict[int, int] = {}
        self._evicted_hashes: set[int] = set()

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
        recompute_tokens = 0
        recompute_cost = 0.0
        admission_count = 0
        eviction_count = 0
        forced_bypass_count = 0
        latencies: list[float] = []
        occupancies: list[int] = []
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

        try:
            future_reuse = _FutureReuseTracker(
                requests,
                block_size_tokens=self.block_size_tokens,
                enabled=self.expose_future_reuse,
            )
            for now, request in enumerate(requests):
                self._release_expired(now)
                request_blocks = self._materialize_chain(request, now)
                future_reuse.advance(request_blocks, now)
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
                matched_lengths.append(matched_len)
                per_request_evictions = 0
                hit_blocks += matched_len
                tokens_hit = sum(
                    block.token_count for block in request_blocks[:matched_len]
                )
                hit_tokens += tokens_hit
                tenant_hits[request.info.tenant_id] = (
                    tenant_hits.get(request.info.tenant_id, 0) + tokens_hit
                )

                duration = max(
                    1,
                    math.ceil(request.true_output_length / self.active_tokens_per_step),
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
                    self._pin(block, now + duration)
                    self._call_hook(
                        policy.on_cache_hit,
                        self._info(block, now, future_reuse),
                        request.info,
                        now,
                    )

                admission_blocked = False
                for block in request_blocks[matched_len:]:
                    recompute_tokens += block.token_count
                    recompute_cost += self._estimated_recompute_cost(block)
                    if block.prefix_hash in self._evicted_hashes:
                        reuse_after_eviction_missed_blocks += 1
                        reuse_after_eviction_missed_tokens += block.token_count
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
                        continue
                    score = self._score(
                        policy.score_admission,
                        self._info(block, now, future_reuse),
                        now,
                    )
                    if score <= 0.0:
                        admission_blocked = True
                        continue
                    admitted, evictions, high_descendant_victims = self._admit_block(
                        policy,
                        block,
                        now,
                        duration,
                        future_reuse,
                    )
                    if admitted:
                        admission_count += 1
                        if is_cold_deep:
                            cold_deep_admissions += 1
                        per_request_evictions += evictions
                        eviction_count += evictions
                        high_descendant_evictions += high_descendant_victims
                    else:
                        forced_bypass_count += 1
                        admission_blocked = True

                uncached_cost = sum(
                    self._estimated_recompute_cost(block)
                    for block in request_blocks[matched_len:]
                )
                latency = (
                    uncached_cost
                    + matched_len * self.lookup_cost_per_block
                    + per_request_evictions * self.eviction_cost_per_block
                )
                latencies.append(latency)
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
        fairness_penalty = self._tenant_fairness_penalty(tenant_hits, tenant_tokens)
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
            prefill_tokens_saved=hit_tokens,
            recompute_tokens=recompute_tokens,
            recompute_cost=recompute_cost,
            eviction_count=eviction_count,
            admission_count=admission_count,
            cache_churn_per_1k=eviction_count * 1000.0 / request_count,
            forced_bypass_count=forced_bypass_count,
            tenant_fairness_penalty=fairness_penalty
            if workload == "multi_tenant_skew"
            else 0.0,
            p50_latency_proxy=_percentile(latencies, 50),
            p95_latency_proxy=_percentile(latencies, 95),
            p99_latency_proxy=_percentile(latencies, 99),
            memory_occupancy_mean=mean(occupancies) if occupancies else 0.0,
            memory_occupancy_peak=max(occupancies) if occupancies else 0,
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
    ) -> tuple[bool, int, int]:
        if block.resident:
            self._pin(block, now + duration)
            return True, 0, 0

        if block.parent_hash is not None:
            parent = self.blocks.get(block.parent_hash)
            if parent is None or not parent.resident:
                return False, 0, 0

        self._make_resident(block)
        release_at = now + duration
        self._pin(block, release_at)
        evictions = 0
        high_descendant_evictions = 0
        while self.resident_count > self.capacity_blocks:
            evictable = self._evictable_blocks()
            if not evictable:
                self._unpin(block)
                self._cancel_release(block, release_at)
                self._remove_resident(block)
                return False, evictions, high_descendant_evictions
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
            if (
                self._descendant_counts.get(victim.prefix_hash, 0)
                >= _HIGH_DESCENDANT_MIN_COUNT
            ):
                high_descendant_evictions += 1
            self._evicted_hashes.add(victim.prefix_hash)
            self._remove_resident(victim)
            evictions += 1
        return True, evictions, high_descendant_evictions

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
        return score

    def _call_hook(self, func: Callable, *args) -> None:
        try:
            func(*args)
        except Exception as exc:  # pragma: no cover - defensive
            raise InvalidCandidateError(
                f"{func.__name__} raised {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _tenant_fairness_penalty(
        tenant_hits: dict[int, int],
        tenant_tokens: dict[int, int],
    ) -> float:
        rates = [
            tenant_hits.get(tenant, 0) / tokens
            for tenant, tokens in tenant_tokens.items()
            if tokens > 0
        ]
        return max(rates) - min(rates) if rates else 0.0


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
                    simulator = PrefixKVCacheSimulator(
                        capacity_blocks=capacity_blocks,
                        block_size_tokens=self.config.block_size_tokens,
                        prefill_cost_per_token=self.config.prefill_cost_per_token,
                        lookup_cost_per_block=self.config.lookup_cost_per_block,
                        eviction_cost_per_block=self.config.eviction_cost_per_block,
                        active_tokens_per_step=self.config.active_tokens_per_step,
                        expose_future_reuse=self.expose_future_reuse,
                    )
                    try:
                        policy = _build_policy(
                            factory,
                            capacity_blocks,
                            self.config.block_size_tokens,
                            actual_seed,
                        )
                    except Exception as exc:
                        trials.append(
                            TrialMetrics(
                                split=workload.split,
                                workload=workload.family,
                                seed=actual_seed,
                                capacity_blocks=capacity_blocks,
                                scoring_fn_complexity=scoring_fn_complexity,
                                invalid=True,
                                invalid_reason=f"factory raised {type(exc).__name__}",
                            )
                        )
                        continue
                    trials.append(
                        simulator.run(
                            policy,
                            requests,
                            split=workload.split,
                            workload=workload.family,
                            seed=actual_seed,
                            scoring_fn_complexity=scoring_fn_complexity,
                        )
                    )

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
        combined = self._score_trials(trials, invalid_fraction, scoring_fn_complexity)
        return EvaluationResult(
            combined_score=combined,
            success=invalid_fraction == 0.0,
            invalid_fraction=invalid_fraction,
            split_metrics=split_metrics,
            workload_metrics=workload_metrics,
            capacity_metrics=capacity_metrics,
            candidate_metadata={
                "capacity_blocks": self.config.capacity_blocks,
                "capacity_sweep_blocks": ",".join(
                    str(value) for value in capacity_blocks_values
                ),
                "block_size_tokens": self.config.block_size_tokens,
                "scoring_fn_complexity": scoring_fn_complexity,
                "min_workload_weight": self.config.min_workload_weight,
                "expose_future_reuse": self.expose_future_reuse,
            },
            trials=tuple(trials),
        )

    def _score_trials(
        self,
        trials: list[TrialMetrics],
        invalid_fraction: float,
        complexity: int,
    ) -> float:
        if invalid_fraction > 0.0:
            return (
                self.config.v_min
                - 1.0
                - self.config.invalid_surcharge * invalid_fraction
            )
        validation = [trial for trial in trials if trial.split == "validation"]
        if not validation:
            validation = (
                list(trials)
                if set(self.splits) == {"hidden"}
                else [trial for trial in trials if trial.split != "hidden"]
            )
        latency_norm = self.config.latency_norm or max(
            (trial.max_prefill_cost for trial in validation),
            default=1.0,
        )
        by_workload_capacity: dict[tuple[str, int], list[TrialMetrics]] = {}
        for trial in validation:
            by_workload_capacity.setdefault(
                (trial.workload, trial.capacity_blocks), []
            ).append(trial)
        workload_scores = [
            _workload_base_score(
                workload_trials,
                token_weight=self.config.w_avg_tok,
                block_weight=self.config.w_avg_blk,
                latency_weight=self.config.latency_weight,
                latency_cap=self.config.latency_cap,
                latency_norm=latency_norm,
            )
            for workload_trials in by_workload_capacity.values()
        ]
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
        complexity_cost = min(
            self.config.complex_cap,
            self.config.k_complex * complexity,
        )
        return (
            mean_score
            + self.config.min_workload_weight * min_workload_score
            - churn_cost
            - fairness_cost
            - complexity_cost
        )


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


class _LFUPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(-block.hit_count)


class _DepthPreferShallowPolicy(_BasePolicy):
    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        return float(block.depth)


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
        return float(-block.descendant_count)


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
        self._current_tenant = 0

    def on_request_start(self, request: RequestInfo, now: int) -> None:
        self._current_tenant = request.tenant_id

    def score_admission(self, block: PrefixBlockInfo, now: int) -> float:
        return 1.0

    def score_eviction(self, block: PrefixBlockInfo, now: int) -> float:
        other_tenant_bias = 6.0 if block.tenant_id != self._current_tenant else 0.0
        return float(now - block.last_accessed_at) + other_tenant_bias


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
        return float(next_distance / (1.0 + future_reuse))


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
        "adversarial_unique_prompts": _adversarial_unique_prompts,
        "cross_family_mixture": _cross_family_mixture,
    }.get(family)
    if builder is None:
        raise ValueError(f"unknown workload family {family!r}")
    return tuple(builder(request_count, block_size_tokens, rng))


def scoring_fn_complexity(source: str) -> int:
    """Count AST nodes inside score_admission and score_eviction definitions."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 10_000
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "score_admission",
            "score_eviction",
        }:
            total += sum(1 for _ in ast.walk(node))
    return total


def _build_policy(
    factory: Callable[..., PrefixKVPolicy],
    capacity_blocks: int,
    block_size_tokens: int,
    seed: int,
) -> PrefixKVPolicy:
    try:
        return factory(capacity_blocks, block_size_tokens, seed)
    except TypeError:
        try:
            return factory(capacity_blocks, block_size_tokens)
        except TypeError:
            return factory()


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
        "prefill_tokens_saved",
        "recompute_tokens",
        "recompute_cost",
        "eviction_count",
        "admission_count",
        "cache_churn_per_1k",
        "forced_bypass_count",
        "tenant_fairness_penalty",
        "p50_latency_proxy",
        "p95_latency_proxy",
        "p99_latency_proxy",
        "memory_occupancy_mean",
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
    latency_weight: float,
    latency_cap: float,
    latency_norm: float,
) -> float:
    token_score = token_weight * mean(trial.token_hit_rate for trial in trials)
    block_score = block_weight * mean(trial.block_hit_rate for trial in trials)
    latency = mean(trial.p95_latency_proxy for trial in trials)
    latency_cost = min(
        latency_cap,
        latency_weight * latency / max(latency_norm, 1.0),
    )
    return token_score + block_score - latency_cost


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if percentile == 50:
        return float(median(values))
    values = sorted(values)
    index = math.ceil((percentile / 100.0) * len(values)) - 1
    return float(values[max(0, min(index, len(values) - 1))])


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


def _agent_trace_branching(
    count: int, block_size: int, rng: random.Random
) -> list[WorkloadRequest]:
    root = [_block("agent/root/a", block_size), _block("agent/root/b", block_size)]
    branches = [_block(f"agent/branch/{idx}", block_size) for idx in range(4)]
    tools = [_block(f"agent/tool/{idx}", block_size) for idx in range(10)]
    requests = []
    for request_id in range(count):
        branch_idx = (request_id // 4 + request_id) % len(branches)
        tool = tools[(request_id + branch_idx * 3) % len(tools)]
        retry = _partial_tail(f"agent/retry/{branch_idx}/{request_id % 4}", block_size)
        requests.append(
            _request(
                request_id=request_id,
                tenant_id=0,
                session_id=branch_idx,
                blocks=[*root, branches[branch_idx], tool, retry],
                request_type="agent",
                true_output_length=96 + rng.randrange(96),
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
            )
        )
    return requests
