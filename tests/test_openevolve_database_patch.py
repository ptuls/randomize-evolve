"""Regression tests for local OpenEvolve database patches."""

import asyncio

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase


class _DummyNoveltyLLM:
    async def generate_with_context(self, system_message, messages):
        del system_message
        del messages
        return "NOVEL"


def test_novelty_judge_works_inside_running_event_loop() -> None:
    database = ProgramDatabase(DatabaseConfig())
    database.novelty_llm = _DummyNoveltyLLM()

    candidate = Program(id="candidate", code="print('a')", language="python")
    similar = Program(id="similar", code="print('b')", language="python")

    async def _check() -> bool:
        return database._llm_judge_novelty(candidate, similar)

    assert asyncio.run(_check()) is True
