"""Evaluator for evolving prompt canonicalization for prefix-cache efficiency."""

from __future__ import annotations

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
        fragments: list[str] = [self.system_message, self.output_format, self.user_request]
        fragments.extend(_dedupe_strings(self.instruction_blocks))
        fragments.extend(
            block.text for block in self.context_blocks if block.required
        )
        return tuple(_dedupe_strings(fragments))


class EvaluatorConfig(BaseModel):
    """Configuration knobs for prompt cache compaction evaluation."""

    model_config = ConfigDict(validate_assignment=True)

    seeds: Sequence[int] = Field(default_factory=lambda: (7, 19, 41))
    prompt_build_timeout_s: float = Field(default=0.25, gt=0.0)
    cache_ttl_s: float = Field(default=300.0, gt=0.0)
    cache_write_multiplier: float = Field(default=1.25, gt=0.0)
    cache_read_multiplier: float = Field(default=0.10, gt=0.0)
    required_coverage_penalty: float = Field(default=5000.0, gt=0.0)
    prefix_regret_weight: float = Field(default=250.0, ge=0.0)
    dynamic_regret_weight: float = Field(default=150.0, ge=0.0)
    token_weight: float = Field(default=0.1, ge=0.0)
    cached_cost_weight: float = Field(default=1.0, ge=0.0)
    latency_weight: float = Field(default=2.0, ge=0.0)


class PromptCompactor(Protocol):
    """Minimal protocol evolved candidates must satisfy."""

    def build_prompt(self, task: PromptTask) -> str: ...


@dataclass(slots=True)
class TrialMetrics:
    seed: int
    required_coverage: float
    prefix_ratio: float
    oracle_prefix_ratio: float
    dynamic_offset_ratio: float
    oracle_dynamic_offset_ratio: float
    avg_prompt_tokens: float
    avg_naive_prompt_tokens: float
    avg_cached_input_cost: float
    avg_naive_input_cost: float
    avg_oracle_cached_input_cost: float
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
    avg_prompt_tokens: float
    avg_naive_prompt_tokens: float
    avg_cached_input_cost: float
    avg_naive_input_cost: float
    avg_oracle_cached_input_cost: float
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
                logger.exception("Prompt compaction evaluation failed for seed {}", seed)
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
                avg_prompt_tokens=0.0,
                avg_naive_prompt_tokens=0.0,
                avg_cached_input_cost=0.0,
                avg_naive_input_cost=0.0,
                avg_oracle_cached_input_cost=0.0,
                avg_build_time_ms=0.0,
                error=message,
            )

        required_coverage = statistics.fmean(t.required_coverage for t in trials)
        prefix_ratio = statistics.fmean(t.prefix_ratio for t in trials)
        oracle_prefix_ratio = statistics.fmean(t.oracle_prefix_ratio for t in trials)
        dynamic_offset_ratio = statistics.fmean(
            t.dynamic_offset_ratio for t in trials
        )
        oracle_dynamic_offset_ratio = statistics.fmean(
            t.oracle_dynamic_offset_ratio for t in trials
        )
        avg_prompt_tokens = statistics.fmean(t.avg_prompt_tokens for t in trials)
        avg_naive_prompt_tokens = statistics.fmean(
            t.avg_naive_prompt_tokens for t in trials
        )
        avg_cached_input_cost = statistics.fmean(
            t.avg_cached_input_cost for t in trials
        )
        avg_naive_input_cost = statistics.fmean(
            t.avg_naive_input_cost for t in trials
        )
        avg_oracle_cached_input_cost = statistics.fmean(
            t.avg_oracle_cached_input_cost for t in trials
        )
        avg_build_time_ms = statistics.fmean(t.avg_build_time_ms for t in trials)

        score = self._score(
            required_coverage=required_coverage,
            prefix_ratio=prefix_ratio,
            oracle_prefix_ratio=oracle_prefix_ratio,
            dynamic_offset_ratio=dynamic_offset_ratio,
            oracle_dynamic_offset_ratio=oracle_dynamic_offset_ratio,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_cached_input_cost=avg_cached_input_cost,
            avg_oracle_cached_input_cost=avg_oracle_cached_input_cost,
            avg_build_time_ms=avg_build_time_ms,
        )

        message = ", ".join(errors) if errors else None
        if message:
            logger.warning("Prompt compaction evaluator encountered partial failures: {}", message)

        return EvaluationResult(
            score=score,
            success=not errors,
            trials=trials,
            required_coverage=required_coverage,
            prefix_ratio=prefix_ratio,
            oracle_prefix_ratio=oracle_prefix_ratio,
            dynamic_offset_ratio=dynamic_offset_ratio,
            oracle_dynamic_offset_ratio=oracle_dynamic_offset_ratio,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_naive_prompt_tokens=avg_naive_prompt_tokens,
            avg_cached_input_cost=avg_cached_input_cost,
            avg_naive_input_cost=avg_naive_input_cost,
            avg_oracle_cached_input_cost=avg_oracle_cached_input_cost,
            avg_build_time_ms=avg_build_time_ms,
            error=message,
        )

    def _run_trial(self, factory, seed: int) -> TrialMetrics:
        tasks = build_task_corpus(seed)
        compactor = factory(key_bits=0, capacity=len(tasks))
        prompts: list[str] = []
        build_times_ms: list[float] = []

        for task in tasks:
            started = time.perf_counter()
            prompt = compactor.build_prompt(task)
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
        avg_prompt_tokens = statistics.fmean(len(_tokenize(prompt)) for prompt in prompts)
        avg_naive_prompt_tokens = statistics.fmean(
            len(_tokenize(render_naive_prompt(task))) for task in tasks
        )

        oracle_prompts = [render_oracle_prompt(task) for task in tasks]
        oracle_prefix_ratio = _mean_family_prefix_ratio(tasks, oracle_prompts)
        oracle_dynamic_offset_ratio = statistics.fmean(
            _dynamic_offset_ratio(task.user_request, prompt)
            for task, prompt in zip(tasks, oracle_prompts)
        )
        avg_cached_input_cost = _mean_cached_input_cost(
            tasks,
            prompts,
            ttl_s=self.config.cache_ttl_s,
            write_multiplier=self.config.cache_write_multiplier,
            read_multiplier=self.config.cache_read_multiplier,
        )
        avg_naive_input_cost = statistics.fmean(
            len(_tokenize(prompt)) for prompt in prompts
        )
        avg_oracle_cached_input_cost = _mean_cached_input_cost(
            tasks,
            oracle_prompts,
            ttl_s=self.config.cache_ttl_s,
            write_multiplier=self.config.cache_write_multiplier,
            read_multiplier=self.config.cache_read_multiplier,
        )

        return TrialMetrics(
            seed=seed,
            required_coverage=required_coverage,
            prefix_ratio=prefix_ratio,
            oracle_prefix_ratio=oracle_prefix_ratio,
            dynamic_offset_ratio=dynamic_offset_ratio,
            oracle_dynamic_offset_ratio=oracle_dynamic_offset_ratio,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_naive_prompt_tokens=avg_naive_prompt_tokens,
            avg_cached_input_cost=avg_cached_input_cost,
            avg_naive_input_cost=avg_naive_input_cost,
            avg_oracle_cached_input_cost=avg_oracle_cached_input_cost,
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
        avg_prompt_tokens: float,
        avg_cached_input_cost: float,
        avg_oracle_cached_input_cost: float,
        avg_build_time_ms: float,
    ) -> float:
        coverage_regret = max(0.0, 1.0 - required_coverage)
        prefix_regret = max(0.0, oracle_prefix_ratio - prefix_ratio)
        dynamic_regret = max(0.0, oracle_dynamic_offset_ratio - dynamic_offset_ratio)
        cached_cost_regret = max(
            0.0, avg_cached_input_cost - avg_oracle_cached_input_cost
        )
        return (
            coverage_regret * self.config.required_coverage_penalty
            + prefix_regret * self.config.prefix_regret_weight
            + dynamic_regret * self.config.dynamic_regret_weight
            + cached_cost_regret * self.config.cached_cost_weight
            + avg_prompt_tokens * self.config.token_weight
            + avg_build_time_ms * self.config.latency_weight
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
                sections.append(("EXAMPLES", "\n".join(_dedupe_strings(task.example_blocks))))

            stable_blocks = [block for block in task.context_blocks if block.stable]
            volatile_blocks = [block for block in task.context_blocks if not block.stable]
            if stable_blocks:
                sections.append(("STABLE CONTEXT", _render_context(stable_blocks)))
            sections.append(("REQUEST", task.user_request))
            if volatile_blocks:
                sections.append(("REQUEST CONTEXT", _render_context(volatile_blocks)))
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


def render_oracle_prompt(task: PromptTask) -> str:
    """A strong hand-written canonicalization target used for regret scoring."""

    instructions = sorted(_dedupe_strings(task.instruction_blocks))
    examples = sorted(_dedupe_strings(task.example_blocks))
    stable_blocks = sorted(
        _dedupe_context(block for block in task.context_blocks if block.stable),
        key=lambda block: (not block.required, block.name, block.text),
    )
    volatile_blocks = sorted(
        _dedupe_context(block for block in task.context_blocks if not block.stable),
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


def build_task_corpus(seed: int) -> tuple[PromptTask, ...]:
    """Build a deterministic corpus with family-shared prefixes and noisy ordering."""

    rng = random.Random(seed)
    support_tasks = _build_support_tasks(rng)
    review_tasks = _build_code_review_tasks(rng)
    research_tasks = _build_research_tasks(rng)
    return tuple(support_tasks + review_tasks + research_tasks)


def _build_support_tasks(rng: random.Random) -> list[PromptTask]:
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
        ContextBlock(
            "refund_policy",
            "Refund Policy v3: annual plans are refundable within 30 days; monthly plans are non-refundable after renewal.",
            stable=True,
            required=True,
        ),
        ContextBlock(
            "security_playbook",
            "Security Escalation: suspected account takeover requires immediate escalation before discussing invoice changes.",
            stable=True,
            required=True,
        ),
        ContextBlock(
            "product_glossary",
            "Glossary: workspace owner can transfer seats; billing contacts can view invoices but cannot rotate secrets.",
            stable=True,
            required=False,
        ),
    ]
    cases = [
        (
            "support-1",
            "A workspace owner asks for a refund after two accidental renewals and mentions they also saw a suspicious login from Berlin.",
            [
                ContextBlock(
                    "account_state",
                    "Account State: annual business plan renewed 12 days ago; two invoices were issued after seat expansion.",
                    stable=False,
                    required=True,
                ),
                ContextBlock(
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
                ContextBlock(
                    "account_state",
                    "Account State: monthly team plan renewed 45 days ago; the requester is a billing contact, not the workspace owner.",
                    stable=False,
                    required=True,
                ),
                ContextBlock(
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
                ContextBlock(
                    "account_state",
                    "Account State: annual enterprise plan; 18 seats removed yesterday after a reorg; no suspicious access events detected.",
                    stable=False,
                    required=True,
                ),
                ContextBlock(
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
            volatile_context=volatile_context,
            user_request=user_request,
            request_time_s=request_time_s,
        )
        for task_id, user_request, volatile_context, request_time_s in cases
    ]


def _build_code_review_tasks(rng: random.Random) -> list[PromptTask]:
    shared_instructions = [
        "Prioritize correctness, regressions, and missing tests.",
        "Do not praise the patch; be direct and specific.",
        "Reference concrete files and line numbers when available.",
    ]
    examples = [
        "Example: If a migration drops an index without a backfill plan, call out deployment risk before style concerns.",
        "Example: If a diff changes retry logic, discuss failure modes and missing coverage, not naming."
    ]
    stable_context = [
        ContextBlock(
            "review_rubric",
            "Review Rubric: P0 blocks shipping, P1 is urgent but shippable with mitigation, P2 is medium risk, P3 is low risk.",
            stable=True,
            required=True,
        ),
        ContextBlock(
            "repo_context",
            "Repository Context: Python service with async workers; concurrency regressions and partial writes are high risk.",
            stable=True,
            required=True,
        ),
        ContextBlock(
            "style_note",
            "Style Note: keep summaries brief and lead with findings.",
            stable=True,
            required=False,
        ),
    ]
    cases = [
        (
            "review-1",
            "Review a patch that replaces an idempotent retry loop with a fire-and-forget background task in invoice settlement.",
            [
                ContextBlock(
                    "diff_summary",
                    "Diff Summary: settlement retries moved from inline awaited calls to create_task() without error aggregation.",
                    stable=False,
                    required=True,
                ),
                ContextBlock(
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
                ContextBlock(
                    "diff_summary",
                    "Diff Summary: writes now permit customer_id=None before enqueueing outbound events.",
                    stable=False,
                    required=True,
                ),
                ContextBlock(
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
                ContextBlock(
                    "diff_summary",
                    "Diff Summary: permission decisions are cached in a module-level dict keyed only by user_id.",
                    stable=False,
                    required=True,
                ),
                ContextBlock(
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
            volatile_context=volatile_context,
            user_request=user_request,
            request_time_s=request_time_s,
        )
        for task_id, user_request, volatile_context, request_time_s in cases
    ]


def _build_research_tasks(rng: random.Random) -> list[PromptTask]:
    shared_instructions = [
        "Separate direct evidence from inference.",
        "Do not overstate study conclusions.",
        "Cite source ids inline for each substantive claim.",
    ]
    examples = [
        "Example: If two sources disagree on causality, state the disagreement explicitly and lower confidence.",
        "Example: If only observational evidence is available, avoid therapeutic recommendations."
    ]
    stable_context = [
        ContextBlock(
            "synthesis_rubric",
            "Synthesis Rubric: summarize claim, supporting evidence, confidence, and remaining uncertainty for each answer.",
            stable=True,
            required=True,
        ),
        ContextBlock(
            "terminology",
            "Terminology: reserve 'causal' for randomized or strongly instrumented evidence; otherwise say 'associated'.",
            stable=True,
            required=True,
        ),
        ContextBlock(
            "format_note",
            "Format Note: keep each evidence bullet under 45 words when possible.",
            stable=True,
            required=False,
        ),
    ]
    cases = [
        (
            "research-1",
            "Summarize whether sleep extension improves insulin sensitivity in adults with chronic sleep restriction.",
            [
                ContextBlock(
                    "source_a",
                    "Source A (S1): crossover trial of 42 adults found a modest insulin-sensitivity improvement after 14 nights of sleep extension.",
                    stable=False,
                    required=True,
                ),
                ContextBlock(
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
                ContextBlock(
                    "source_a",
                    "Source A (S3): small randomized trial reported improved digit-span backward scores after six weeks of creatine.",
                    stable=False,
                    required=True,
                ),
                ContextBlock(
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
                ContextBlock(
                    "source_a",
                    "Source A (S5): single-arm pilot reported fewer monthly migraine days after an 8-hour eating window.",
                    stable=False,
                    required=True,
                ),
                ContextBlock(
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
            volatile_context=volatile_context,
            user_request=user_request,
            request_time_s=request_time_s,
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
    stable_context: Sequence[ContextBlock],
    volatile_context: Sequence[ContextBlock],
    user_request: str,
    request_time_s: float,
) -> PromptTask:
    instructions = list(instruction_blocks)
    examples = list(example_blocks)
    context_blocks = list(stable_context) + list(volatile_context)

    rng.shuffle(instructions)
    rng.shuffle(examples)
    rng.shuffle(context_blocks)

    if instructions:
        instructions.append(instructions[0])
    if examples:
        examples.append(examples[-1])
    if context_blocks:
        context_blocks.insert(0, context_blocks[-1])

    return PromptTask(
        family_id=family_id,
        task_id=task_id,
        system_message=system_message,
        output_format=output_format,
        instruction_blocks=tuple(instructions),
        example_blocks=tuple(examples),
        context_blocks=tuple(context_blocks),
        user_request=user_request,
        request_time_s=request_time_s,
    )


def _mean_cached_input_cost(
    tasks: Sequence[PromptTask],
    prompts: Sequence[str],
    *,
    ttl_s: float,
    write_multiplier: float,
    read_multiplier: float,
) -> float:
    prompts_by_family: dict[str, list[tuple[PromptTask, list[str]]]] = {}
    for task, prompt in zip(tasks, prompts):
        prompts_by_family.setdefault(task.family_id, []).append((task, _tokenize(prompt)))

    costs: list[float] = []
    for family_entries in prompts_by_family.values():
        family_entries.sort(key=lambda item: item[0].request_time_s)
        for index, (task, tokens) in enumerate(family_entries):
            token_count = len(tokens)
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
                cost = (
                    cached_prefix_tokens * read_multiplier
                    + (token_count - cached_prefix_tokens)
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

    return statistics.fmean(costs) if costs else 0.0


def _mean_family_prefix_ratio(tasks: Sequence[PromptTask], prompts: Sequence[str]) -> float:
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
        1
        for fragment in fragments
        if _normalize_space(fragment) in normalized_prompt
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
