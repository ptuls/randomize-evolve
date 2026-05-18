"""Shared workflow abstractions."""

from pathlib import Path
from typing import Any, Protocol


class Runner(Protocol):
    """Runs an evolved program against a resolved configuration."""

    def run(self, program_path: Path, config: Any) -> Any: ...


class Reporter(Protocol):
    """Publishes workflow results for humans or automation."""

    def report(self, result: Any, iterations: int, config_label: str) -> None: ...
