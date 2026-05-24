"""Baseline prompt compaction candidate for prefix-cache experiments."""


class BaselinePromptCompactor:
    """Simple prompt builder with some canonicalization, but not an optimal one."""

    # EVOLVE-BLOCK-START
    def build_prompt(self, task) -> str:
        sections = [
            ("SYSTEM", task.system_message),
            ("OUTPUT FORMAT", task.output_format),
            ("INSTRUCTIONS", "\n".join(self._dedupe(task.instruction_blocks))),
        ]

        examples = self._dedupe(task.example_blocks)
        if examples:
            sections.append(("EXAMPLES", "\n".join(examples)))

        stable_blocks = []
        volatile_blocks = []
        for block in task.context_blocks:
            if block.stable:
                stable_blocks.append(block)
            else:
                volatile_blocks.append(block)

        stable_blocks = self._dedupe_blocks(stable_blocks)
        volatile_blocks = self._dedupe_blocks(volatile_blocks)

        if stable_blocks:
            sections.append(("STABLE CONTEXT", self._render_blocks(stable_blocks)))
        sections.append(("REQUEST", task.user_request))
        if volatile_blocks:
            sections.append(("REQUEST CONTEXT", self._render_blocks(volatile_blocks)))

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


def candidate_factory(key_bits: int, capacity: int) -> BaselinePromptCompactor:
    """Factory function required by the evaluator."""
    del key_bits
    del capacity
    return BaselinePromptCompactor()


def run_demo() -> None:
    """Small demo showing the baseline compactor shape."""
    from randomize_evolve.evaluators.prompt_cache_compaction import build_task_corpus

    compactor = candidate_factory(0, 0)
    sample_task = build_task_corpus(7)[0]
    print(compactor.build_prompt(sample_task))


if __name__ == "__main__":
    run_demo()
