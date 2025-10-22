import tempfile
import textwrap
from pathlib import Path
from typing import Optional


class ProgramSource:
    """Provides the seed program that OpenEvolve mutates."""

    def __init__(self, source: str) -> None:
        self._source = textwrap.dedent(source)

    def text(self) -> str:
        return self._source


class TemporaryProgramFile:
    """Persists a program source to a temporary file owned by the context."""

    def __init__(self, source: ProgramSource) -> None:
        self._source = source
        self._path: Optional[Path] = None

    def __enter__(self) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        handle.write(self._source.text())
        handle.flush()
        handle.close()
        self._path = Path(handle.name)
        return self._path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._path and self._path.exists():
            self._path.unlink()
