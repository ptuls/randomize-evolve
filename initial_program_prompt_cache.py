"""Baseline prompt compaction candidate for prefix-cache experiments."""


class BaselinePromptCompactor:
    """Simple prompt builder with some canonicalization, but not an optimal one."""

    def __init__(self, corpus_hint=None):
        self._corpus_hint = corpus_hint

    # EVOLVE-BLOCK-START
    def build_prompt(self, task, corpus_hint=None) -> str:
        corpus_hint = corpus_hint or self._corpus_hint
        sections = [
            ("SYSTEM", task.system_message),
            ("OUTPUT FORMAT", task.output_format),
            ("INSTRUCTIONS", "\n".join(self._dedupe(task.instruction_blocks))),
        ]

        examples = self._dedupe(task.example_blocks)
        if examples:
            sections.append(("EXAMPLES", "\n".join(examples)))

        context_blocks = self._dedupe_blocks(task.context_blocks)
        context_blocks.sort(
            key=lambda block: (
                -self._context_frequency(corpus_hint, task.family_id, block.text),
                not block.required,
                len(block.text.split()),
                block.name,
            )
        )

        if context_blocks:
            sections.append(("CONTEXT", self._render_blocks(context_blocks)))
        sections.append(("REQUEST", task.user_request))

        return self._render_sections(sections)

    # EVOLVE-BLOCK-END

    def _dedupe(self, values):
        seen = set()
        result = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

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

    def _render_blocks(self, blocks):
        return "\n".join(f"- {block.name}: {block.text}" for block in blocks)

    def _render_sections(self, sections):
        rendered = []
        for title, body in sections:
            if not body:
                continue
            rendered.append(f"[{title}]\n{body.strip()}")
        return "\n\n".join(rendered)

    def _context_frequency(self, corpus_hint, family_id, text):
        if corpus_hint is None:
            return 0.0
        return corpus_hint.context_frequency(family_id, text)


def candidate_factory(
    key_bits: int, capacity: int, tasks=None, corpus_hint=None
) -> BaselinePromptCompactor:
    """Factory function required by the evaluator."""
    del key_bits
    del capacity
    del tasks
    return BaselinePromptCompactor(corpus_hint=corpus_hint)


def run_demo() -> None:
    """Small demo showing the baseline compactor shape."""
    from randomize_evolve.evaluators.prompt_cache_compaction import build_task_corpus

    compactor = candidate_factory(0, 0)
    sample_task = build_task_corpus(7)[0]
    print(compactor.build_prompt(sample_task))


if __name__ == "__main__":
    run_demo()
