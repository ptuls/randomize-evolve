"""Tests for prompt-cache compaction evaluation."""

from initial_program_prompt_cache import candidate_factory
from randomize_evolve.evaluators.prompt_cache_compaction import (
    Evaluator,
    EvaluatorConfig,
)


class EarlyRequestCompactor:
    def build_prompt(self, task) -> str:
        examples = "\n".join(task.example_blocks)
        context = "\n".join(
            f"- {block.name}: {block.text}" for block in task.context_blocks
        )
        parts = [
            ("REQUEST", task.user_request),
            ("SYSTEM", task.system_message),
            ("OUTPUT", task.output_format),
            ("INSTRUCTIONS", "\n".join(task.instruction_blocks)),
            ("EXAMPLES", examples),
            ("CONTEXT", context),
        ]
        return "\n\n".join(f"[{title}]\n{body}" for title, body in parts)


class CanonicalLateRequestCompactor:
    def build_prompt(self, task) -> str:
        instructions = "\n".join(sorted(dict.fromkeys(task.instruction_blocks)))
        stable = sorted(
            {(block.name, block.text) for block in task.context_blocks if block.stable}
        )
        volatile = sorted(
            {(block.name, block.text) for block in task.context_blocks if not block.stable}
        )
        parts = [
            ("SYSTEM", task.system_message),
            ("OUTPUT", task.output_format),
            ("INSTRUCTIONS", instructions),
        ]
        if stable:
            parts.append(
                (
                    "STABLE",
                    "\n".join(f"- {name}: {text}" for name, text in stable),
                )
            )
        if volatile:
            parts.append(
                (
                    "VOLATILE",
                    "\n".join(f"- {name}: {text}" for name, text in volatile),
                )
            )
        parts.append(("REQUEST", task.user_request))
        return "\n\n".join(f"[{title}]\n{body}" for title, body in parts)


def _make_factory(compactor):
    def factory(key_bits: int, capacity: int):
        del key_bits
        del capacity
        return compactor

    return factory


def test_late_canonical_prompts_score_better_than_early_request_prompts() -> None:
    evaluator = Evaluator(EvaluatorConfig(seeds=(7,)))

    early = evaluator(_make_factory(EarlyRequestCompactor()))
    late = evaluator(_make_factory(CanonicalLateRequestCompactor()))

    assert late.score < early.score
    assert late.prefix_ratio > early.prefix_ratio
    assert late.dynamic_offset_ratio > early.dynamic_offset_ratio
    assert late.avg_cached_input_cost < early.avg_cached_input_cost


def test_initial_prompt_cache_candidate_is_evaluable() -> None:
    evaluator = Evaluator(EvaluatorConfig(seeds=(7,)))

    result = evaluator(candidate_factory)

    assert result.success
    assert result.required_coverage == 1.0
    assert result.avg_prompt_tokens > 0.0
    assert result.avg_cached_input_cost < result.avg_naive_input_cost
