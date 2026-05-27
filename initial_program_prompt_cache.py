"""Baseline prompt compaction candidate for prefix-cache experiments."""


class BaselinePromptCompactor:
    """Simple prompt builder with some canonicalization, but not an optimal one."""

    def __init__(self, corpus_hint=None):
        self._corpus_hint = corpus_hint

    # EVOLVE-BLOCK-START
    def build_prompt(self, task, corpus_hint=None) -> str:
        corpus_hint = corpus_hint if corpus_hint is not None else self._corpus_hint
        n = self._normalize_text

        sections = [
            ("SYSTEM", n(task.system_message)),
            ("OUTPUT FORMAT", n(task.output_format)),
            (
                "INSTRUCTIONS",
                "\n".join(
                    self._corpus_ordered(
                        task.instruction_blocks,
                        task.family_id,
                        "instructions",
                        corpus_hint,
                    )
                ),
            ),
        ]

        examples = self._corpus_ordered(
            task.example_blocks, task.family_id, "examples", corpus_hint
        )
        if examples:
            sections.append(("EXAMPLES", "\n".join(examples)))

        blocks = self._dedupe_blocks(task.context_blocks)
        stable, volatile = self._split_by_stability(blocks, task.family_id, corpus_hint)
        if stable:
            sections.append(("STABLE CONTEXT", self._render_blocks(stable)))
        if volatile:
            sections.append(("REQUEST CONTEXT", self._render_blocks(volatile)))

        sections.append(("REQUEST", n(task.user_request)))
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

    def _normalize_text(self, value):
        return " ".join(str(value).split())

    def _corpus_ordered(self, values, family_id, kind, corpus_hint):
        normalized = self._dedupe([self._normalize_text(v) for v in values])
        if corpus_hint is None:
            return sorted(normalized)
        order = corpus_hint.canonical_order(family_id, kind)
        if not order:
            return sorted(normalized)
        present = set(normalized)
        ordered = [value for value in order if value in present]
        remainder = [value for value in normalized if value not in ordered]
        return ordered + sorted(remainder)

    def _split_by_stability(self, blocks, family_id, corpus_hint, threshold=0.5):
        def freq(block):
            if corpus_hint is None:
                return 0.0
            name_frequency = getattr(corpus_hint, "name_frequency", None)
            if name_frequency is not None:
                return name_frequency(family_id, block.name)
            return corpus_hint.context_frequency(family_id, block.text)

        stable = []
        volatile = []
        for block in blocks:
            (stable if freq(block) >= threshold else volatile).append(block)

        stable.sort(
            key=lambda block: (
                -freq(block),
                block.name,
                self._normalize_text(block.text),
            )
        )
        volatile.sort(key=lambda block: (block.name, self._normalize_text(block.text)))
        return stable, volatile

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
