"""Evaluator for evolving prompt canonicalization for prefix-cache efficiency."""

from __future__ import annotations

import inspect
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """A named chunk of prompt context."""

    name: str
    text: str
    required: bool


@dataclass(frozen=True, slots=True)
class AnnotatedContextBlock:
    """Internal context block carrying evaluator-only stability metadata."""

    name: str
    text: str
    stable: bool
    required: bool


@dataclass(frozen=True, slots=True)
class PromptTask:
    """Structured prompt-building task presented to candidate compactors."""

    family_id: str
    task_id: str
    system_message: str
    output_format: str
    instruction_blocks: tuple[str, ...]
    example_blocks: tuple[str, ...]
    context_blocks: tuple[ContextBlock, ...]
    user_request: str
    request_time_s: float

    @property
    def required_fragments(self) -> tuple[str, ...]:
        fragments: list[str] = [
            self.system_message,
            self.output_format,
            self.user_request,
        ]
        fragments.extend(_dedupe_strings(self.instruction_blocks))
        fragments.extend(block.text for block in self.context_blocks if block.required)
        return tuple(_dedupe_strings(fragments))


@dataclass(frozen=True, slots=True)
class EvaluatedTask:
    """Internal task bundle with hidden evaluator annotations."""

    task: PromptTask
    context_blocks: tuple[AnnotatedContextBlock, ...]


@dataclass(frozen=True, slots=True)
class CorpusHint:
    """Corpus-level statistics exposed to candidates for stability inference."""

    tasks: tuple[PromptTask, ...]
    family_task_counts: dict[str, int]
    instruction_counts: dict[tuple[str, str], int]
    example_counts: dict[tuple[str, str], int]
    context_counts: dict[tuple[str, str], int]
    context_name_counts: dict[tuple[str, str], int]
    family_instruction_order: dict[str, tuple[str, ...]]
    family_example_order: dict[str, tuple[str, ...]]
    family_context_blocks: dict[str, tuple[ContextBlock, ...]]
    family_grouped_context_texts: dict[tuple[str, str], tuple[str, ...]]

    def instruction_frequency(self, family_id: str, text: str) -> float:
        return self._frequency(self.instruction_counts, family_id, text)

    def example_frequency(self, family_id: str, text: str) -> float:
        return self._frequency(self.example_counts, family_id, text)

    def context_frequency(self, family_id: str, text: str) -> float:
        return self._frequency(self.context_counts, family_id, text)

    def context_name_frequency(self, family_id: str, name: str) -> float:
        return self._frequency(self.context_name_counts, family_id, name)

    def instruction_order(self, family_id: str) -> tuple[str, ...]:
        return self.family_instruction_order.get(family_id, ())

    def example_order(self, family_id: str) -> tuple[str, ...]:
        return self.family_example_order.get(family_id, ())

    def context_blocks_for_family(self, family_id: str) -> tuple[ContextBlock, ...]:
        return self.family_context_blocks.get(family_id, ())

    def grouped_context_texts(self, family_id: str, name: str) -> tuple[str, ...]:
        return self.family_grouped_context_texts.get((family_id, name), ())

    def _frequency(
        self, counts: dict[tuple[str, str], int], family_id: str, text: str
    ) -> float:
        family_count = max(1, self.family_task_counts.get(family_id, 1))
        return counts.get((family_id, text), 0) / family_count


class EvaluatorConfig(BaseModel):
    """Configuration knobs for prompt cache compaction evaluation."""

    model_config = ConfigDict(validate_assignment=True)

    seeds: Sequence[int] = Field(default_factory=lambda: (7, 19, 41))
    prompt_build_timeout_s: float = Field(default=0.25, gt=0.0)
    cache_ttl_s: float = Field(default=300.0, gt=0.0)
    cache_write_multiplier: float = Field(default=1.25, gt=0.0)
    cache_read_multiplier: float = Field(default=0.10, gt=0.0)
    required_coverage_penalty: float = Field(default=100.0, gt=0.0)
    cache_miss_weight: float = Field(default=100.0, ge=0.0)
    prefix_regret_weight: float = Field(default=250.0, ge=0.0)
    dynamic_regret_weight: float = Field(default=150.0, ge=0.0)
    token_weight: float = Field(default=0.1, ge=0.0)
    cached_cost_weight: float = Field(default=1.0, ge=0.0)
    latency_weight: float = Field(default=2.0, ge=0.0)
    instruction_inclusion_probability: float = Field(default=0.8, ge=0.0, le=1.0)
    example_inclusion_probability: float = Field(default=0.8, ge=0.0, le=1.0)
    stable_label_flip_probability: float = Field(default=0.05, ge=0.0, le=1.0)
    refreshable_stable_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    volatile_shared_inclusion_probability: float = Field(default=0.67, ge=0.0, le=1.0)


class PromptCompactor(Protocol):
    """Minimal protocol evolved candidates must satisfy."""

    def build_prompt(
        self, task: PromptTask, corpus_hint: Optional[CorpusHint] = None
    ) -> str: ...


@dataclass(slots=True)
class CacheReplayMetrics:
    """Aggregate metrics from replaying the prompt stream through a prefix cache."""

    avg_input_cost: float
    cache_hit_rate: float
    total_prompt_tokens: int
    cached_prompt_tokens: int


@dataclass(slots=True)
class TrialMetrics:
    seed: int
    required_coverage: float
    prefix_ratio: float
    oracle_prefix_ratio: float
    dynamic_offset_ratio: float
    oracle_dynamic_offset_ratio: float
    cache_hit_rate: float
    naive_cache_hit_rate: float
    oracle_cache_hit_rate: float
    avg_prompt_tokens: float
    avg_naive_prompt_tokens: float
    avg_cached_input_cost: float
    avg_naive_input_cost: float
    avg_oracle_cached_input_cost: float
    avg_oracle_prompt_tokens: float
    oracle_score: float
    avg_build_time_ms: float


@dataclass(slots=True)
class EvaluationResult:
    score: float
    success: bool
    trials: List[TrialMetrics]
    required_coverage: float
    prefix_ratio: float
    oracle_prefix_ratio: float
    dynamic_offset_ratio: float
    oracle_dynamic_offset_ratio: float
    cache_hit_rate: float
    naive_cache_hit_rate: float
    oracle_cache_hit_rate: float
    avg_prompt_tokens: float
    avg_naive_prompt_tokens: float
    avg_cached_input_cost: float
    avg_naive_input_cost: float
    avg_oracle_cached_input_cost: float
    avg_oracle_prompt_tokens: float
    oracle_score: float
    score_minus_oracle_score: float
    avg_build_time_ms: float
    error: Optional[str] = None


class Evaluator:
    """Scores prompt-compaction strategies using an offline synthetic corpus."""

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        self.config = config or EvaluatorConfig()

    def __call__(self, factory) -> EvaluationResult:
        trials: list[TrialMetrics] = []
        errors: list[str] = []

        for seed in self.config.seeds:
            try:
                trial = self._run_trial(factory, seed)
            except Exception as exc:  # noqa: BLE001 - evolutionary search is adversarial
                errors.append(f"seed {seed}: {exc!r}")
                logger.exception(
                    "Prompt compaction evaluation failed for seed {}", seed
                )
                continue
            trials.append(trial)

        if not trials:
            message = ", ".join(errors) if errors else "no successful trials"
            return EvaluationResult(
                score=self.config.required_coverage_penalty,
                success=False,
                trials=[],
                required_coverage=0.0,
                prefix_ratio=0.0,
                oracle_prefix_ratio=1.0,
                dynamic_offset_ratio=0.0,
                oracle_dynamic_offset_ratio=1.0,
                cache_hit_rate=0.0,
                naive_cache_hit_rate=0.0,
                oracle_cache_hit_rate=1.0,
                avg_prompt_tokens=0.0,
                avg_naive_prompt_tokens=0.0,
                avg_cached_input_cost=0.0,
                avg_naive_input_cost=0.0,
                avg_oracle_cached_input_cost=0.0,
                avg_oracle_prompt_tokens=0.0,
                oracle_score=0.0,
                score_minus_oracle_score=self.config.required_coverage_penalty,
                avg_build_time_ms=0.0,
                error=message,
            )

        required_coverage = statistics.fmean(t.required_coverage for t in trials)
        prefix_ratio = statistics.fmean(t.prefix_ratio for t in trials)
        oracle_prefix_ratio = statistics.fmean(t.oracle_prefix_ratio for t in trials)
        dynamic_offset_ratio = statistics.fmean(t.dynamic_offset_ratio for t in trials)
        oracle_dynamic_offset_ratio = statistics.fmean(
            t.oracle_dynamic_offset_ratio for t in trials
        )
        cache_hit_rate = statistics.fmean(t.cache_hit_rate for t in trials)
        naive_cache_hit_rate = statistics.fmean(t.naive_cache_hit_rate for t in trials)
        oracle_cache_hit_rate = statistics.fmean(
            t.oracle_cache_hit_rate for t in trials
        )
        avg_prompt_tokens = statistics.fmean(t.avg_prompt_tokens for t in trials)
        avg_naive_prompt_tokens = statistics.fmean(
            t.avg_naive_prompt_tokens for t in trials
        )
        avg_cached_input_cost = statistics.fmean(
            t.avg_cached_input_cost for t in trials
        )
        avg_naive_input_cost = statistics.fmean(t.avg_naive_input_cost for t in trials)
        avg_oracle_cached_input_cost = statistics.fmean(
            t.avg_oracle_cached_input_cost for t in trials
        )
        avg_oracle_prompt_tokens = statistics.fmean(
            t.avg_oracle_prompt_tokens for t in trials
        )
        oracle_score = statistics.fmean(t.oracle_score for t in trials)
        avg_build_time_ms = statistics.fmean(t.avg_build_time_ms for t in trials)

        score = self._score(
            required_coverage=required_coverage,
            prefix_ratio=prefix_ratio,
            oracle_prefix_ratio=oracle_prefix_ratio,
            dynamic_offset_ratio=dynamic_offset_ratio,
            oracle_dynamic_offset_ratio=oracle_dynamic_offset_ratio,
            cache_hit_rate=cache_hit_rate,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_cached_input_cost=avg_cached_input_cost,
            avg_build_time_ms=avg_build_time_ms,
        )
        score_minus_oracle_score = score - oracle_score

        message = ", ".join(errors) if errors else None
        if message:
            logger.warning(
                "Prompt compaction evaluator encountered partial failures: {}", message
            )

        return EvaluationResult(
            score=score,
            success=not errors,
            trials=trials,
            required_coverage=required_coverage,
            prefix_ratio=prefix_ratio,
            oracle_prefix_ratio=oracle_prefix_ratio,
            dynamic_offset_ratio=dynamic_offset_ratio,
            oracle_dynamic_offset_ratio=oracle_dynamic_offset_ratio,
            cache_hit_rate=cache_hit_rate,
            naive_cache_hit_rate=naive_cache_hit_rate,
            oracle_cache_hit_rate=oracle_cache_hit_rate,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_naive_prompt_tokens=avg_naive_prompt_tokens,
            avg_cached_input_cost=avg_cached_input_cost,
            avg_naive_input_cost=avg_naive_input_cost,
            avg_oracle_cached_input_cost=avg_oracle_cached_input_cost,
            avg_oracle_prompt_tokens=avg_oracle_prompt_tokens,
            oracle_score=oracle_score,
            score_minus_oracle_score=score_minus_oracle_score,
            avg_build_time_ms=avg_build_time_ms,
            error=message,
        )

    def _run_trial(self, factory, seed: int) -> TrialMetrics:
        evaluated_tasks = _build_evaluated_task_corpus(seed, self.config)
        tasks = [entry.task for entry in evaluated_tasks]
        corpus_hint = _build_corpus_hint(tasks)
        compactor = _build_compactor(
            factory,
            tasks=tasks,
            capacity=len(tasks),
            corpus_hint=corpus_hint,
        )
        prompts: list[str] = []
        build_times_ms: list[float] = []

        for task in tasks:
            started = time.perf_counter()
            prompt = _build_prompt(compactor, task, corpus_hint)
            elapsed_ms = (time.perf_counter() - started) * 1e3
            if elapsed_ms > self.config.prompt_build_timeout_s * 1e3:
                raise TimeoutError(
                    f"prompt build exceeded {self.config.prompt_build_timeout_s}s"
                )
            if not isinstance(prompt, str):
                raise TypeError("build_prompt() must return a string")
            prompts.append(prompt)
            build_times_ms.append(elapsed_ms)

        required_coverage = statistics.fmean(
            _fragment_coverage(task.required_fragments, prompt)
            for task, prompt in zip(tasks, prompts)
        )
        prefix_ratio = _mean_family_prefix_ratio(tasks, prompts)
        dynamic_offset_ratio = statistics.fmean(
            _dynamic_offset_ratio(task.user_request, prompt)
            for task, prompt in zip(tasks, prompts)
        )
        avg_prompt_tokens = statistics.fmean(
            len(_tokenize(prompt)) for prompt in prompts
        )

        naive_prompts = [render_naive_prompt(task) for task in tasks]
        avg_naive_prompt_tokens = statistics.fmean(
            len(_tokenize(prompt)) for prompt in naive_prompts
        )

        oracle_prompts = [render_oracle_prompt(entry) for entry in evaluated_tasks]
        oracle_prefix_ratio = _mean_family_prefix_ratio(tasks, oracle_prompts)
        oracle_dynamic_offset_ratio = statistics.fmean(
            _dynamic_offset_ratio(task.user_request, prompt)
            for task, prompt in zip(tasks, oracle_prompts)
        )
        replay_metrics = _replay_prefix_cache(
            tasks,
            prompts,
            ttl_s=self.config.cache_ttl_s,
            write_multiplier=self.config.cache_write_multiplier,
            read_multiplier=self.config.cache_read_multiplier,
        )
        naive_replay_metrics = _replay_prefix_cache(
            tasks,
            naive_prompts,
            ttl_s=self.config.cache_ttl_s,
            write_multiplier=self.config.cache_write_multiplier,
            read_multiplier=self.config.cache_read_multiplier,
        )
        oracle_replay_metrics = _replay_prefix_cache(
            tasks,
            oracle_prompts,
            ttl_s=self.config.cache_ttl_s,
            write_multiplier=self.config.cache_write_multiplier,
            read_multiplier=self.config.cache_read_multiplier,
        )
        avg_oracle_prompt_tokens = statistics.fmean(
            len(_tokenize(prompt)) for prompt in oracle_prompts
        )
        oracle_score = self._score(
            required_coverage=1.0,
            prefix_ratio=oracle_prefix_ratio,
            oracle_prefix_ratio=oracle_prefix_ratio,
            dynamic_offset_ratio=oracle_dynamic_offset_ratio,
            oracle_dynamic_offset_ratio=oracle_dynamic_offset_ratio,
            cache_hit_rate=oracle_replay_metrics.cache_hit_rate,
            avg_prompt_tokens=avg_oracle_prompt_tokens,
            avg_cached_input_cost=oracle_replay_metrics.avg_input_cost,
            avg_build_time_ms=0.0,
        )

        return TrialMetrics(
            seed=seed,
            required_coverage=required_coverage,
            prefix_ratio=prefix_ratio,
            oracle_prefix_ratio=oracle_prefix_ratio,
            dynamic_offset_ratio=dynamic_offset_ratio,
            oracle_dynamic_offset_ratio=oracle_dynamic_offset_ratio,
            cache_hit_rate=replay_metrics.cache_hit_rate,
            naive_cache_hit_rate=naive_replay_metrics.cache_hit_rate,
            oracle_cache_hit_rate=oracle_replay_metrics.cache_hit_rate,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_naive_prompt_tokens=avg_naive_prompt_tokens,
            avg_cached_input_cost=replay_metrics.avg_input_cost,
            avg_naive_input_cost=naive_replay_metrics.avg_input_cost,
            avg_oracle_cached_input_cost=oracle_replay_metrics.avg_input_cost,
            avg_oracle_prompt_tokens=avg_oracle_prompt_tokens,
            oracle_score=oracle_score,
            avg_build_time_ms=statistics.fmean(build_times_ms),
        )

    def _score(
        self,
        *,
        required_coverage: float,
        prefix_ratio: float,
        oracle_prefix_ratio: float,
        dynamic_offset_ratio: float,
        oracle_dynamic_offset_ratio: float,
        cache_hit_rate: float,
        avg_prompt_tokens: float,
        avg_cached_input_cost: float,
        avg_build_time_ms: float,
    ) -> float:
        coverage_regret = max(0.0, 1.0 - required_coverage)
        cache_miss_rate = max(0.0, 1.0 - cache_hit_rate)
        prefix_bonus = max(0.0, prefix_ratio - oracle_prefix_ratio)
        dynamic_bonus = max(0.0, dynamic_offset_ratio - oracle_dynamic_offset_ratio)
        return (
            coverage_regret * self.config.required_coverage_penalty
            + cache_miss_rate * self.config.cache_miss_weight
            + avg_cached_input_cost * self.config.cached_cost_weight
            + avg_prompt_tokens * self.config.token_weight
            + avg_build_time_ms * self.config.latency_weight
            - prefix_bonus * self.config.prefix_regret_weight
            - dynamic_bonus * self.config.dynamic_regret_weight
        )


def baseline_prompt_compactor() -> PromptCompactor:
    """Reference candidate used by tests and demos."""

    class BaselinePromptCompactor:
        def build_prompt(self, task: PromptTask) -> str:
            sections = [
                ("SYSTEM", task.system_message),
                ("OUTPUT FORMAT", task.output_format),
                ("INSTRUCTIONS", "\n".join(_dedupe_strings(task.instruction_blocks))),
            ]
            if task.example_blocks:
                sections.append(
                    ("EXAMPLES", "\n".join(_dedupe_strings(task.example_blocks)))
                )

            context_blocks = sorted(
                _dedupe_context(task.context_blocks),
                key=lambda block: (not block.required, block.name, block.text),
            )
            if context_blocks:
                sections.append(("CONTEXT", _render_context(context_blocks)))
            sections.append(("REQUEST", task.user_request))
            return _render_sections(sections)

    return BaselinePromptCompactor()


def render_naive_prompt(task: PromptTask) -> str:
    """A deliberately noisy baseline showing poor cache-prefix hygiene."""

    sections = [
        ("REQUEST", task.user_request),
        ("OUTPUT FORMAT", task.output_format),
        ("SYSTEM", task.system_message),
        ("INSTRUCTIONS", "\n".join(task.instruction_blocks)),
        ("EXAMPLES", "\n".join(task.example_blocks)),
        ("ALL CONTEXT", _render_context(task.context_blocks)),
    ]
    return _render_sections(sections)


def render_oracle_prompt(evaluated_task: EvaluatedTask) -> str:
    """A strong hand-written canonicalization target used for regret scoring."""

    task = evaluated_task.task
    instructions = sorted(_dedupe_strings(task.instruction_blocks))
    examples = sorted(_dedupe_strings(task.example_blocks))
    stable_blocks = sorted(
        _dedupe_annotated_context(
            block for block in evaluated_task.context_blocks if block.stable
        ),
        key=lambda block: (not block.required, block.name, block.text),
    )
    volatile_blocks = sorted(
        _dedupe_annotated_context(
            block for block in evaluated_task.context_blocks if not block.stable
        ),
        key=lambda block: (not block.required, block.name, block.text),
    )

    sections = [
        ("SYSTEM", task.system_message),
        ("OUTPUT FORMAT", task.output_format),
        ("INSTRUCTIONS", "\n".join(instructions)),
    ]
    if examples:
        sections.append(("EXAMPLES", "\n".join(examples)))
    if stable_blocks:
        sections.append(("STABLE CONTEXT", _render_context(stable_blocks)))
    if volatile_blocks:
        sections.append(("REQUEST CONTEXT", _render_context(volatile_blocks)))
    sections.append(("REQUEST", task.user_request))
    return _render_sections(sections)


def _build_corpus_hint(tasks: Sequence[PromptTask]) -> CorpusHint:
    family_task_counts: dict[str, int] = {}
    instruction_counts: dict[tuple[str, str], int] = {}
    example_counts: dict[tuple[str, str], int] = {}
    context_counts: dict[tuple[str, str], int] = {}
    context_name_counts: dict[tuple[str, str], int] = {}
    family_instruction_order: dict[str, list[str]] = {}
    family_example_order: dict[str, list[str]] = {}
    family_context_blocks: dict[str, list[ContextBlock]] = {}
    family_grouped_context_texts: dict[tuple[str, str], list[str]] = {}

    for task in tasks:
        family_task_counts[task.family_id] = (
            family_task_counts.get(task.family_id, 0) + 1
        )
        for text in dict.fromkeys(task.instruction_blocks):
            key = (task.family_id, text)
            instruction_counts[key] = instruction_counts.get(key, 0) + 1
            family_instruction_order.setdefault(task.family_id, [])
            if text not in family_instruction_order[task.family_id]:
                family_instruction_order[task.family_id].append(text)
        for text in dict.fromkeys(task.example_blocks):
            key = (task.family_id, text)
            example_counts[key] = example_counts.get(key, 0) + 1
            family_example_order.setdefault(task.family_id, [])
            if text not in family_example_order[task.family_id]:
                family_example_order[task.family_id].append(text)
        for block in _dedupe_context(task.context_blocks):
            key = (task.family_id, block.text)
            context_counts[key] = context_counts.get(key, 0) + 1
            name_key = (task.family_id, block.name)
            context_name_counts[name_key] = context_name_counts.get(name_key, 0) + 1
            family_context_blocks.setdefault(task.family_id, [])
            if block not in family_context_blocks[task.family_id]:
                family_context_blocks[task.family_id].append(block)
            family_grouped_context_texts.setdefault(name_key, [])
            if block.text not in family_grouped_context_texts[name_key]:
                family_grouped_context_texts[name_key].append(block.text)

    return CorpusHint(
        tasks=tuple(tasks),
        family_task_counts=family_task_counts,
        instruction_counts=instruction_counts,
        example_counts=example_counts,
        context_counts=context_counts,
        context_name_counts=context_name_counts,
        family_instruction_order={
            family_id: tuple(values)
            for family_id, values in family_instruction_order.items()
        },
        family_example_order={
            family_id: tuple(values)
            for family_id, values in family_example_order.items()
        },
        family_context_blocks={
            family_id: tuple(values)
            for family_id, values in family_context_blocks.items()
        },
        family_grouped_context_texts={
            key: tuple(values) for key, values in family_grouped_context_texts.items()
        },
    )


def _build_compactor(
    factory,
    *,
    tasks: Sequence[PromptTask],
    capacity: int,
    corpus_hint: CorpusHint,
):
    kwargs = {
        "key_bits": 0,
        "capacity": capacity,
        "tasks": tasks,
        "corpus_hint": corpus_hint,
    }
    return factory(**_supported_kwargs(factory, kwargs))


def _build_prompt(compactor, task: PromptTask, corpus_hint: CorpusHint) -> str:
    kwargs = {"task": task, "corpus_hint": corpus_hint}
    return compactor.build_prompt(**_supported_kwargs(compactor.build_prompt, kwargs))


def _supported_kwargs(callable_obj, kwargs: dict[str, object]) -> dict[str, object]:
    parameters = inspect.signature(callable_obj).parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in parameters}


def build_task_corpus(
    seed: int, config: Optional[EvaluatorConfig] = None
) -> tuple[PromptTask, ...]:
    """Build the public candidate-facing corpus without hidden stability labels."""

    return tuple(entry.task for entry in _build_evaluated_task_corpus(seed, config))


def _build_evaluated_task_corpus(
    seed: int, config: Optional[EvaluatorConfig] = None
) -> tuple[EvaluatedTask, ...]:
    """Build a deterministic corpus with hidden evaluator annotations."""

    config = config or EvaluatorConfig()
    rng = random.Random(seed)
    support_tasks = _build_support_tasks(rng, config)
    review_tasks = _build_code_review_tasks(rng, config)
    research_tasks = _build_research_tasks(rng, config)
    return tuple(support_tasks + review_tasks + research_tasks)


def _build_support_tasks(
    rng: random.Random, config: EvaluatorConfig
) -> list[PromptTask]:
    shared_instructions = [
        "Follow the refund policy exactly; do not promise unsupported credits.",
        "Escalate suspected security issues before discussing billing details.",
        "Quote only facts present in the context blocks.",
    ]
    examples = [
        "Example: For a password reset ticket, classify as account_access and recommend identity verification.",
        "Example: For a refund request outside the allowed window, explain the policy boundary and offer alternatives.",
    ]
    stable_context = [
        AnnotatedContextBlock(
            "refund_policy",
            "Refund Policy v3: annual plans are refundable within 30 days; monthly plans are non-refundable after renewal.",
            stable=True,
            required=True,
        ),
        AnnotatedContextBlock(
            "security_playbook",
            "Security Escalation: suspected account takeover requires immediate escalation before discussing invoice changes.",
            stable=True,
            required=True,
        ),
        AnnotatedContextBlock(
            "product_glossary",
            "Glossary: workspace owner can transfer seats; billing contacts can view invoices but cannot rotate secrets.",
            stable=True,
            required=False,
        ),
        AnnotatedContextBlock(
            "retrieval_digest",
            _long_reference_block(
                "Support Digest",
                (
                    "Refund routing depends on plan type, renewal timing, seat changes,"
                    " suspicious-access handling, ownership boundaries, and escalation"
                    " rules."
                ),
                18,
            ),
            stable=True,
            required=True,
        ),
    ]
    shared_volatile_context = [
        AnnotatedContextBlock(
            "refund_boundary_note",
            "Boundary Note: prorated credits are considered separately from renewal refund eligibility and should be explained with plan-specific timing.",
            stable=False,
            required=False,
        ),
        AnnotatedContextBlock(
            "security_triage_template",
            "Security Template: when suspicious access is mentioned, acknowledge the signal, route escalation first, and avoid changing billing state before verification.",
            stable=False,
            required=False,
        ),
    ]
    cases = [
        (
            "support-1",
            "A workspace owner asks for a refund after two accidental renewals and mentions they also saw a suspicious login from Berlin.",
            [
                AnnotatedContextBlock(
                    "account_state",
                    "Account State: annual business plan renewed 12 days ago; two invoices were issued after seat expansion.",
                    stable=False,
                    required=True,
                ),
                AnnotatedContextBlock(
                    "security_signal",
                    "Security Signal: login anomaly detected from Berlin on 2026-05-20 for admin@example.com.",
                    stable=False,
                    required=True,
                ),
            ],
            0.0,
        ),
        (
            "support-2",
            "A billing contact wants a refund for a monthly renewal from 45 days ago and asks whether they can also rotate API secrets.",
            [
                AnnotatedContextBlock(
                    "account_state",
                    "Account State: monthly team plan renewed 45 days ago; the requester is a billing contact, not the workspace owner.",
                    stable=False,
                    required=True,
                ),
                AnnotatedContextBlock(
                    "capability_note",
                    "Capability Note: billing contacts can view invoices but cannot rotate secrets or transfer ownership.",
                    stable=False,
                    required=True,
                ),
            ],
            120.0,
        ),
        (
            "support-3",
            "A workspace owner requests a prorated credit after removing 18 seats yesterday and wants the reply drafted in neutral language.",
            [
                AnnotatedContextBlock(
                    "account_state",
                    "Account State: annual enterprise plan; 18 seats removed yesterday after a reorg; no suspicious access events detected.",
                    stable=False,
                    required=True,
                ),
                AnnotatedContextBlock(
                    "style_note",
                    "Style Note: respond in neutral, concise language and avoid apologizing for policy constraints.",
                    stable=False,
                    required=True,
                ),
            ],
            260.0,
        ),
    ]
    return [
        _materialize_task(
            rng,
            family_id="support",
            task_id=task_id,
            system_message=(
                "You are a support triage assistant. Produce a response plan that is policy-safe and operationally precise."
            ),
            output_format=(
                "Return JSON with keys: issue_type, risk_level, next_action, customer_reply."
            ),
            instruction_blocks=shared_instructions,
            example_blocks=examples,
            stable_context=stable_context,
            shared_volatile_context=shared_volatile_context,
            volatile_context=volatile_context,
            user_request=user_request,
            request_time_s=request_time_s,
            config=config,
        )
        for task_id, user_request, volatile_context, request_time_s in cases
    ]


def _build_code_review_tasks(
    rng: random.Random, config: EvaluatorConfig
) -> list[PromptTask]:
    shared_instructions = [
        "Prioritize correctness, regressions, and missing tests.",
        "Do not praise the patch; be direct and specific.",
        "Reference concrete files and line numbers when available.",
    ]
    examples = [
        "Example: If a migration drops an index without a backfill plan, call out deployment risk before style concerns.",
        "Example: If a diff changes retry logic, discuss failure modes and missing coverage, not naming.",
    ]
    stable_context = [
        AnnotatedContextBlock(
            "review_rubric",
            "Review Rubric: P0 blocks shipping, P1 is urgent but shippable with mitigation, P2 is medium risk, P3 is low risk.",
            stable=True,
            required=True,
        ),
        AnnotatedContextBlock(
            "repo_context",
            "Repository Context: Python service with async workers; concurrency regressions and partial writes are high risk.",
            stable=True,
            required=True,
        ),
        AnnotatedContextBlock(
            "style_note",
            "Style Note: keep summaries brief and lead with findings.",
            stable=True,
            required=False,
        ),
        AnnotatedContextBlock(
            "retrieval_digest",
            _long_reference_block(
                "Review Digest",
                (
                    "The service uses async workers, write-behind queues, partial-failure"
                    " retries, and review policies that emphasize correctness over style."
                ),
                18,
            ),
            stable=True,
            required=True,
        ),
    ]
    shared_volatile_context = [
        AnnotatedContextBlock(
            "failure_mode_template",
            "Failure Mode Template: call out lost retries, dropped work, partial writes, and missing rollback paths before style issues.",
            stable=False,
            required=False,
        ),
        AnnotatedContextBlock(
            "test_gap_template",
            "Test Gap Template: check behavior under retries, process restarts, and duplicate delivery when control flow changes in async workers.",
            stable=False,
            required=False,
        ),
    ]
    cases = [
        (
            "review-1",
            "Review a patch that replaces an idempotent retry loop with a fire-and-forget background task in invoice settlement.",
            [
                AnnotatedContextBlock(
                    "diff_summary",
                    "Diff Summary: settlement retries moved from inline awaited calls to create_task() without error aggregation.",
                    stable=False,
                    required=True,
                ),
                AnnotatedContextBlock(
                    "files",
                    "Files: billing/settlement.py lines 88-153; tests/test_settlement.py lines 12-44.",
                    stable=False,
                    required=True,
                ),
            ],
            0.0,
        ),
        (
            "review-2",
            "Review a schema change that makes customer_id nullable in a hot write path and removes one validation branch.",
            [
                AnnotatedContextBlock(
                    "diff_summary",
                    "Diff Summary: writes now permit customer_id=None before enqueueing outbound events.",
                    stable=False,
                    required=True,
                ),
                AnnotatedContextBlock(
                    "files",
                    "Files: db/models.py lines 20-66; workers/outbox.py lines 140-191.",
                    stable=False,
                    required=True,
                ),
            ],
            180.0,
        ),
        (
            "review-3",
            "Review a patch that memoizes permission checks globally to reduce request latency in the admin API.",
            [
                AnnotatedContextBlock(
                    "diff_summary",
                    "Diff Summary: permission decisions are cached in a module-level dict keyed only by user_id.",
                    stable=False,
                    required=True,
                ),
                AnnotatedContextBlock(
                    "files",
                    "Files: authz/cache.py lines 1-87; api/admin.py lines 55-130.",
                    stable=False,
                    required=True,
                ),
            ],
            540.0,
        ),
    ]
    return [
        _materialize_task(
            rng,
            family_id="code-review",
            task_id=task_id,
            system_message=(
                "You are a senior code reviewer. Find behavior regressions and missing tests before commenting on anything cosmetic."
            ),
            output_format=(
                "Return markdown with sections: Findings, Risk, Suggested Tests."
            ),
            instruction_blocks=shared_instructions,
            example_blocks=examples,
            stable_context=stable_context,
            shared_volatile_context=shared_volatile_context,
            volatile_context=volatile_context,
            user_request=user_request,
            request_time_s=request_time_s,
            config=config,
        )
        for task_id, user_request, volatile_context, request_time_s in cases
    ]


def _build_research_tasks(
    rng: random.Random, config: EvaluatorConfig
) -> list[PromptTask]:
    shared_instructions = [
        "Separate direct evidence from inference.",
        "Do not overstate study conclusions.",
        "Cite source ids inline for each substantive claim.",
    ]
    examples = [
        "Example: If two sources disagree on causality, state the disagreement explicitly and lower confidence.",
        "Example: If only observational evidence is available, avoid therapeutic recommendations.",
    ]
    stable_context = [
        AnnotatedContextBlock(
            "synthesis_rubric",
            "Synthesis Rubric: summarize claim, supporting evidence, confidence, and remaining uncertainty for each answer.",
            stable=True,
            required=True,
        ),
        AnnotatedContextBlock(
            "terminology",
            "Terminology: reserve 'causal' for randomized or strongly instrumented evidence; otherwise say 'associated'.",
            stable=True,
            required=True,
        ),
        AnnotatedContextBlock(
            "format_note",
            "Format Note: keep each evidence bullet under 45 words when possible.",
            stable=True,
            required=False,
        ),
        AnnotatedContextBlock(
            "retrieval_digest",
            _long_reference_block(
                "Research Digest",
                (
                    "The synthesis policy separates evidence strength, causal language,"
                    " confidence labels, and unresolved uncertainty across related study"
                    " summaries."
                ),
                18,
            ),
            stable=True,
            required=True,
        ),
    ]
    shared_volatile_context = [
        AnnotatedContextBlock(
            "confidence_template",
            "Confidence Template: distinguish randomized evidence, observational evidence, and absence-of-evidence when summarizing intervention claims.",
            stable=False,
            required=False,
        ),
        AnnotatedContextBlock(
            "causality_template",
            "Causality Template: avoid causal phrasing unless intervention evidence directly supports the effect in the target population.",
            stable=False,
            required=False,
        ),
    ]
    cases = [
        (
            "research-1",
            "Summarize whether sleep extension improves insulin sensitivity in adults with chronic sleep restriction.",
            [
                AnnotatedContextBlock(
                    "source_a",
                    "Source A (S1): crossover trial of 42 adults found a modest insulin-sensitivity improvement after 14 nights of sleep extension.",
                    stable=False,
                    required=True,
                ),
                AnnotatedContextBlock(
                    "source_b",
                    "Source B (S2): observational cohort linked short sleep with insulin resistance but did not test intervention effects.",
                    stable=False,
                    required=True,
                ),
            ],
            0.0,
        ),
        (
            "research-2",
            "Summarize whether daily creatine improves short-term memory in older adults.",
            [
                AnnotatedContextBlock(
                    "source_a",
                    "Source A (S3): small randomized trial reported improved digit-span backward scores after six weeks of creatine.",
                    stable=False,
                    required=True,
                ),
                AnnotatedContextBlock(
                    "source_b",
                    "Source B (S4): meta-analysis found mixed effects and substantial study heterogeneity.",
                    stable=False,
                    required=True,
                ),
            ],
            240.0,
        ),
        (
            "research-3",
            "Summarize whether time-restricted eating reduces migraine frequency in adults.",
            [
                AnnotatedContextBlock(
                    "source_a",
                    "Source A (S5): single-arm pilot reported fewer monthly migraine days after an 8-hour eating window.",
                    stable=False,
                    required=True,
                ),
                AnnotatedContextBlock(
                    "source_b",
                    "Source B (S6): no randomized migraine-specific trials were identified in the provided materials.",
                    stable=False,
                    required=True,
                ),
            ],
            620.0,
        ),
    ]
    return [
        _materialize_task(
            rng,
            family_id="research",
            task_id=task_id,
            system_message=(
                "You are a research synthesis assistant. Produce concise evidence summaries without overstating certainty."
            ),
            output_format=(
                "Return bullets with labels: Claim, Evidence, Confidence, Open Questions."
            ),
            instruction_blocks=shared_instructions,
            example_blocks=examples,
            stable_context=stable_context,
            shared_volatile_context=shared_volatile_context,
            volatile_context=volatile_context,
            user_request=user_request,
            request_time_s=request_time_s,
            config=config,
        )
        for task_id, user_request, volatile_context, request_time_s in cases
    ]


def _materialize_task(
    rng: random.Random,
    *,
    family_id: str,
    task_id: str,
    system_message: str,
    output_format: str,
    instruction_blocks: Sequence[str],
    example_blocks: Sequence[str],
    stable_context: Sequence[AnnotatedContextBlock],
    shared_volatile_context: Sequence[AnnotatedContextBlock],
    volatile_context: Sequence[AnnotatedContextBlock],
    user_request: str,
    request_time_s: float,
    config: EvaluatorConfig,
) -> EvaluatedTask:
    instructions = _sample_shared_values(
        instruction_blocks,
        rng,
        inclusion_probability=config.instruction_inclusion_probability,
    )
    examples = _sample_shared_values(
        example_blocks,
        rng,
        inclusion_probability=config.example_inclusion_probability,
    )
    context_blocks = [
        _materialize_context_block(
            block,
            rng,
            family_id=family_id,
            task_id=task_id,
            config=config,
        )
        for block in list(stable_context)
        + _sample_shared_context_blocks(
            shared_volatile_context,
            rng,
            inclusion_probability=config.volatile_shared_inclusion_probability,
        )
        + list(volatile_context)
    ]

    rng.shuffle(instructions)
    rng.shuffle(examples)
    rng.shuffle(context_blocks)

    if instructions:
        instructions.append(instructions[0])
    if examples:
        examples.append(examples[-1])
    if context_blocks:
        context_blocks.insert(0, context_blocks[-1])

    task = PromptTask(
        family_id=family_id,
        task_id=task_id,
        system_message=system_message,
        output_format=output_format,
        instruction_blocks=tuple(instructions),
        example_blocks=tuple(examples),
        context_blocks=tuple(_to_public_context(block) for block in context_blocks),
        user_request=user_request,
        request_time_s=request_time_s,
    )
    return EvaluatedTask(task=task, context_blocks=tuple(context_blocks))


def _sample_shared_values(
    values: Sequence[str],
    rng: random.Random,
    *,
    inclusion_probability: float,
) -> list[str]:
    selected = [value for value in values if rng.random() < inclusion_probability]
    if not selected and values:
        selected.append(values[rng.randrange(len(values))])
    return selected


def _sample_shared_context_blocks(
    blocks: Sequence[AnnotatedContextBlock],
    rng: random.Random,
    *,
    inclusion_probability: float,
) -> list[AnnotatedContextBlock]:
    selected = [block for block in blocks if rng.random() < inclusion_probability]
    if not selected and blocks:
        selected.append(blocks[rng.randrange(len(blocks))])
    return selected


def _materialize_context_block(
    block: AnnotatedContextBlock,
    rng: random.Random,
    *,
    family_id: str,
    task_id: str,
    config: EvaluatorConfig,
) -> AnnotatedContextBlock:
    text = block.text
    stable = block.stable

    if (
        stable
        and block.name == "retrieval_digest"
        and rng.random() < config.refreshable_stable_probability
    ):
        text = _refresh_reference_block(text, family_id=family_id, task_id=task_id)

    if rng.random() < config.stable_label_flip_probability:
        stable = not stable

    return AnnotatedContextBlock(block.name, text, stable, block.required)


def _to_public_context(block: AnnotatedContextBlock) -> ContextBlock:
    return ContextBlock(block.name, block.text, block.required)


def _long_reference_block(title: str, sentence: str, repeat_count: int) -> str:
    lines = [f"{title} {index + 1}: {sentence}" for index in range(repeat_count)]
    return " ".join(lines)


def _refresh_reference_block(text: str, *, family_id: str, task_id: str) -> str:
    return f"{text} Snapshot:{family_id}/{task_id}"


def _replay_prefix_cache(
    tasks: Sequence[PromptTask],
    prompts: Sequence[str],
    *,
    ttl_s: float,
    write_multiplier: float,
    read_multiplier: float,
) -> CacheReplayMetrics:
    prompts_by_family: dict[str, list[tuple[PromptTask, list[str]]]] = {}
    for task, prompt in zip(tasks, prompts):
        prompts_by_family.setdefault(task.family_id, []).append(
            (task, _tokenize(prompt))
        )

    costs: list[float] = []
    total_prompt_tokens = 0
    cached_prompt_tokens = 0
    for family_entries in prompts_by_family.values():
        family_entries.sort(key=lambda item: item[0].request_time_s)
        for index, (task, tokens) in enumerate(family_entries):
            token_count = len(tokens)
            total_prompt_tokens += token_count

            previous_tokens = None
            previous_time_s = None
            if index > 0:
                previous_task, previous_tokens = family_entries[index - 1]
                previous_time_s = previous_task.request_time_s

            if (
                previous_tokens is not None
                and previous_time_s is not None
                and task.request_time_s - previous_time_s <= ttl_s
            ):
                cached_prefix_tokens = _common_prefix_length((previous_tokens, tokens))
                cached_prompt_tokens += cached_prefix_tokens
                cost = cached_prefix_tokens * read_multiplier + (
                    token_count - cached_prefix_tokens
                )
                costs.append(cost)
                continue

            next_tokens = None
            next_time_s = None
            if index + 1 < len(family_entries):
                next_task, next_tokens = family_entries[index + 1]
                next_time_s = next_task.request_time_s

            cacheable_write_tokens = 0
            if (
                next_tokens is not None
                and next_time_s is not None
                and next_time_s - task.request_time_s <= ttl_s
            ):
                cacheable_write_tokens = _common_prefix_length((tokens, next_tokens))

            cost = token_count + cacheable_write_tokens * (write_multiplier - 1.0)
            costs.append(cost)

    avg_input_cost = statistics.fmean(costs) if costs else 0.0
    cache_hit_rate = (
        cached_prompt_tokens / total_prompt_tokens if total_prompt_tokens else 0.0
    )
    return CacheReplayMetrics(
        avg_input_cost=avg_input_cost,
        cache_hit_rate=cache_hit_rate,
        total_prompt_tokens=total_prompt_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
    )


def _mean_cached_input_cost(
    tasks: Sequence[PromptTask],
    prompts: Sequence[str],
    *,
    ttl_s: float,
    write_multiplier: float,
    read_multiplier: float,
) -> float:
    return _replay_prefix_cache(
        tasks,
        prompts,
        ttl_s=ttl_s,
        write_multiplier=write_multiplier,
        read_multiplier=read_multiplier,
    ).avg_input_cost


def _mean_family_prefix_ratio(
    tasks: Sequence[PromptTask], prompts: Sequence[str]
) -> float:
    prompts_by_family: dict[str, list[list[str]]] = {}
    for task, prompt in zip(tasks, prompts):
        prompts_by_family.setdefault(task.family_id, []).append(_tokenize(prompt))

    ratios = []
    for family_prompts in prompts_by_family.values():
        if not family_prompts:
            continue
        common_prefix = _common_prefix_length(family_prompts)
        min_length = min(len(tokens) for tokens in family_prompts) or 1
        ratios.append(common_prefix / min_length)
    return statistics.fmean(ratios) if ratios else 0.0


def _fragment_coverage(fragments: Sequence[str], prompt: str) -> float:
    if not fragments:
        return 1.0
    normalized_prompt = _normalize_space(prompt)
    matches = sum(
        1 for fragment in fragments if _normalize_space(fragment) in normalized_prompt
    )
    return matches / len(fragments)


def _dynamic_offset_ratio(user_request: str, prompt: str) -> float:
    prompt_tokens = _tokenize(prompt)
    request_tokens = _tokenize(user_request)
    if not prompt_tokens:
        return 0.0
    index = _find_subsequence(prompt_tokens, request_tokens)
    if index < 0:
        return 0.0
    return index / len(prompt_tokens)


def _common_prefix_length(token_lists: Sequence[Sequence[str]]) -> int:
    if not token_lists:
        return 0
    prefix_length = min(len(tokens) for tokens in token_lists)
    for index in range(prefix_length):
        token = token_lists[0][index]
        if any(tokens[index] != token for tokens in token_lists[1:]):
            return index
    return prefix_length


def _find_subsequence(tokens: Sequence[str], subsequence: Sequence[str]) -> int:
    if not subsequence:
        return 0
    limit = len(tokens) - len(subsequence) + 1
    for start in range(max(0, limit)):
        if list(tokens[start : start + len(subsequence)]) == list(subsequence):
            return start
    return -1


def _render_sections(sections: Sequence[tuple[str, str]]) -> str:
    rendered = []
    for title, body in sections:
        if not body:
            continue
        rendered.append(f"[{title}]\n{body.strip()}")
    return "\n\n".join(rendered).strip()


def _render_context(blocks: Iterable[ContextBlock]) -> str:
    return "\n".join(f"- {block.name}: {block.text}" for block in blocks)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_context(values: Iterable[ContextBlock]) -> list[ContextBlock]:
    seen: set[tuple[str, str]] = set()
    result: list[ContextBlock] = []
    for value in values:
        key = (value.name, value.text)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _dedupe_annotated_context(
    values: Iterable[AnnotatedContextBlock],
) -> list[AnnotatedContextBlock]:
    seen: set[tuple[str, str]] = set()
    result: list[AnnotatedContextBlock] = []
    for value in values:
        key = (value.name, value.text)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
