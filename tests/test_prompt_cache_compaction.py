"""Tests for prompt-cache compaction evaluation."""

from initial_program_prompt_cache import candidate_factory
from randomize_evolve.evaluators.prompt_cache_compaction import Evaluator
from randomize_evolve.evaluators.prompt_cache_compaction import EvaluatorConfig
from randomize_evolve.evaluators.prompt_cache_compaction import _build_corpus_hint
from randomize_evolve.evaluators.prompt_cache_compaction import build_task_corpus


class RequiredOnlyCompactor:
    def __init__(self, corpus_hint=None) -> None:
        del corpus_hint

    def build_prompt(self, task, corpus_hint=None) -> str:
        del corpus_hint
        instructions = "\n".join(sorted(dict.fromkeys(task.instruction_blocks)))
        examples = "\n".join(sorted(dict.fromkeys(task.example_blocks)))
        context_blocks = sorted(
            {(block.name, block.text, block.required) for block in task.context_blocks},
            key=lambda entry: (not entry[2], entry[0], entry[1]),
        )
        parts = [
            ("SYSTEM", task.system_message),
            ("OUTPUT", task.output_format),
            ("INSTRUCTIONS", instructions),
        ]
        if examples:
            parts.append(("EXAMPLES", examples))
        if context_blocks:
            parts.append(
                (
                    "CONTEXT",
                    "\n".join(
                        f"- {name}: {text}" for name, text, _required in context_blocks
                    ),
                )
            )
        parts.append(("REQUEST", task.user_request))
        return "\n\n".join(f"[{title}]\n{body}" for title, body in parts if body)


class FrequencyAwareCompactor:
    def __init__(self, corpus_hint=None) -> None:
        self._corpus_hint = corpus_hint

    def build_prompt(self, task, corpus_hint=None) -> str:
        corpus_hint = corpus_hint or self._corpus_hint
        self._corpus_hint = corpus_hint
        instructions = list(dict.fromkeys(task.instruction_blocks))
        examples = list(dict.fromkeys(task.example_blocks))
        blocks = self._dedupe_blocks(task.context_blocks)

        shared_instructions = sorted(
            [
                text
                for text in instructions
                if self._instruction_count(corpus_hint, task.family_id, text) > 1
            ],
            key=lambda text: (
                -self._instruction_count(corpus_hint, task.family_id, text),
                len(text),
                text,
            ),
        )
        variable_instructions = sorted(
            [
                text
                for text in instructions
                if self._instruction_count(corpus_hint, task.family_id, text) <= 1
            ],
            key=lambda text: (len(text), text),
        )
        shared_examples = sorted(
            [
                text
                for text in examples
                if self._example_count(corpus_hint, task.family_id, text) > 1
            ],
            key=lambda text: (
                -self._example_count(corpus_hint, task.family_id, text),
                len(text),
                text,
            ),
        )
        variable_examples = sorted(
            [
                text
                for text in examples
                if self._example_count(corpus_hint, task.family_id, text) <= 1
            ],
            key=lambda text: (len(text), text),
        )

        stableish = []
        volatileish = []
        for block in blocks:
            seen_count = self._context_count(corpus_hint, task.family_id, block.text)
            context_frequency = self._context_frequency(
                corpus_hint, task.family_id, block.text
            )
            is_large = len(block.text.split()) >= 35
            if context_frequency >= 0.6 or (block.required and not is_large):
                stableish.append(block)
            else:
                volatileish.append(block)

        stableish.sort(key=self._stable_block_key(task.family_id))
        volatileish.sort(key=self._volatile_block_key(task.family_id))

        parts = [task.system_message, task.output_format]
        if stableish:
            parts.extend(f"{block.name}: {block.text}" for block in stableish)
        if shared_instructions:
            parts.extend(shared_instructions)
        if shared_examples:
            parts.extend(shared_examples)
        if variable_instructions:
            parts.extend(variable_instructions)
        if variable_examples:
            parts.extend(variable_examples)
        if volatileish:
            parts.extend(f"{block.name}: {block.text}" for block in volatileish)
        parts.append(task.user_request)

        return "\n".join(parts)

    def _dedupe_blocks(self, blocks):
        seen = set()
        result = []
        for block in blocks:
            key = (block.name, block.text)
            if key in seen:
                continue
            seen.add(key)
            result.append(block)
        return result

    def _stable_block_key(self, family_id: str):
        return lambda block: (
            -self._context_count(self._corpus_hint, family_id, block.text),
            len(block.text.split()),
            not block.required,
            block.name,
            block.text,
        )

    def _volatile_block_key(self, family_id: str):
        return lambda block: (
            len(block.text.split()),
            -self._context_count(self._corpus_hint, family_id, block.text),
            block.name,
            block.text,
        )

    def _instruction_count(self, corpus_hint, family_id, text):
        if corpus_hint is None:
            return 0
        return corpus_hint.instruction_counts.get((family_id, text), 0)

    def _example_count(self, corpus_hint, family_id, text):
        if corpus_hint is None:
            return 0
        return corpus_hint.example_counts.get((family_id, text), 0)

    def _context_count(self, corpus_hint, family_id, text):
        if corpus_hint is None:
            return 0
        return corpus_hint.context_counts.get((family_id, text), 0)

    def _context_frequency(self, corpus_hint, family_id, text):
        if corpus_hint is None:
            return 0.0
        return corpus_hint.context_frequency(family_id, text)


def _make_factory(compactor_type):
    def factory(key_bits: int, capacity: int, tasks=None, corpus_hint=None):
        del key_bits
        del capacity
        del tasks
        return compactor_type(corpus_hint=corpus_hint)

    return factory


def test_frequency_aware_compactor_beats_required_only_on_noisy_corpus() -> None:
    evaluator = Evaluator(
        EvaluatorConfig(
            seeds=(7, 19, 41),
            instruction_inclusion_probability=0.8,
            example_inclusion_probability=0.8,
            stable_label_flip_probability=0.15,
            refreshable_stable_probability=0.6,
        )
    )

    required_only = evaluator(_make_factory(RequiredOnlyCompactor))
    frequency_aware = evaluator(_make_factory(FrequencyAwareCompactor))

    assert frequency_aware.score < required_only.score
    assert frequency_aware.avg_cached_input_cost < required_only.avg_cached_input_cost
    assert frequency_aware.avg_prompt_tokens < required_only.avg_prompt_tokens
    assert (
        frequency_aware.score_minus_oracle_score
        < required_only.score_minus_oracle_score
    )


def test_initial_prompt_cache_candidate_is_evaluable() -> None:
    evaluator = Evaluator(EvaluatorConfig(seeds=(7,)))

    result = evaluator(candidate_factory)

    assert result.success
    assert result.required_coverage == 1.0
    assert result.avg_prompt_tokens > 0.0
    assert result.avg_cached_input_cost < result.avg_naive_input_cost


def test_corpus_hint_exposes_name_and_order_helpers() -> None:
    tasks = build_task_corpus(7)
    corpus_hint = _build_corpus_hint(tasks)

    support_instructions = corpus_hint.instruction_order("support")
    support_examples = corpus_hint.example_order("support")
    support_blocks = corpus_hint.context_blocks_for_family("support")

    assert support_instructions
    assert support_examples
    assert support_blocks
    assert corpus_hint.context_name_frequency("support", "account_state") > 0.0
    assert corpus_hint.grouped_context_texts("support", "account_state")
