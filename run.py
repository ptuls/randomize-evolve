import asyncio
import os
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import yaml
from openevolve import OpenEvolve, evolve_function
from openevolve.config import Config, LLMModelConfig

from initial_program import candidate_factory


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


class APIKeyProvider:
    """Supplies the API key used by LLM model configurations."""

    def get(self) -> Optional[str]:
        return os.environ.get("OPENAI_API_KEY")


class CascadePolicy:
    """Ensures we always run in direct evaluation mode."""

    def apply(self, config: Config) -> None:
        evaluator_cfg = getattr(config, "evaluator", None)
        if evaluator_cfg and hasattr(evaluator_cfg, "cascade_evaluation"):
            setattr(evaluator_cfg, "cascade_evaluation", False)


class APIKeyInjector:
    """Patches the Config object with credentials from the environment."""

    def __init__(self, provider: APIKeyProvider) -> None:
        self._provider = provider

    def apply(self, config: Config, llm_section: dict[str, Any]) -> Config:
        api_key = self._provider.get()
        if not api_key:
            return config

        llm_cfg = getattr(config, "llm", None)
        if llm_cfg is None:
            return config

        models = getattr(llm_cfg, "models", None)
        if models:
            for model in models:
                if isinstance(model, dict):
                    model.setdefault("api_key", api_key)
                elif hasattr(model, "api_key") and not getattr(model, "api_key"):
                    model.api_key = api_key  # type: ignore[attr-defined]
            return config

        if llm_section:
            models = []
            primary = llm_section.get("primary_model")
            secondary = llm_section.get("secondary_model")
            primary_weight = llm_section.get("primary_model_weight")
            secondary_weight = llm_section.get("secondary_model_weight")
            if primary:
                models.append(LLMModelConfig(name=primary, weight=primary_weight, api_key=api_key))
            if secondary:
                models.append(
                    LLMModelConfig(name=secondary, weight=secondary_weight, api_key=api_key)
                )
            if hasattr(llm_cfg, "models"):
                llm_cfg.models = models  # type: ignore[attr-defined]
            else:
                setattr(llm_cfg, "models", models)
            return config

        for attr in ("primary_model", "secondary_model"):
            model = getattr(llm_cfg, attr, None)
            if not model:
                continue
            if isinstance(model, dict):
                model.setdefault("api_key", api_key)
            elif hasattr(model, "api_key") and not getattr(model, "api_key"):
                model.api_key = api_key  # type: ignore[attr-defined]

        return config


def _construct_blank_config() -> Config:
    constructors: list[Callable[[], Config]] = []
    if hasattr(Config, "model_validate"):
        constructors.append(lambda: Config.model_validate({}))  # type: ignore[attr-defined,return-value]
    constructors.append(lambda: Config())  # type: ignore[call-arg]

    for builder in constructors:
        try:
            return builder()
        except Exception:
            continue

    raise RuntimeError("Unable to construct OpenEvolve Config instance")


class ConfigLoader:
    """Loads OpenEvolve configuration files with post-processing."""

    def __init__(
        self,
        api_key_provider: Optional[APIKeyProvider] = None,
        cascade_policy: Optional[CascadePolicy] = None,
    ) -> None:
        self._api_key_injector = APIKeyInjector(api_key_provider or APIKeyProvider())
        self._cascade_policy = cascade_policy or CascadePolicy()

    def load(self, path: Path) -> Config:
        raw_data = self._read_yaml(path)
        config = (
            self._load_with_openevolve(path)
            or self._construct_from_dict(raw_data)
            or _construct_blank_config()
        )
        config = self._api_key_injector.apply(config, raw_data.get("llm", {}))
        self._cascade_policy.apply(config)
        return config

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data or {}

    def _load_with_openevolve(self, path: Path) -> Optional[Config]:
        try:
            from openevolve.config import load_config  # type: ignore
        except ImportError:
            load_config = None

        if load_config:
            try:
                return load_config(path)  # type: ignore[call-arg]
            except Exception:
                return None

        if hasattr(Config, "from_file"):
            try:
                return Config.from_file(path)  # type: ignore[call-arg]
            except Exception:
                return None

        return None

    def _construct_from_dict(self, data: dict[str, Any]) -> Optional[Config]:
        try:
            if hasattr(Config, "model_validate"):
                return Config.model_validate(data)  # type: ignore[attr-defined,return-value]
            return Config(**data)  # type: ignore[arg-type]
        except Exception:
            return None


class ConfigProvider(Protocol):
    """Abstracts the origin of OpenEvolve configuration objects."""

    def load(self, iterations: int) -> Config: ...

    def describe(self) -> str: ...


@dataclass(slots=True)
class YamlConfigProvider(ConfigProvider):
    """Loads configuration from YAML using the shared loader."""

    path: Path
    loader: ConfigLoader

    def load(self, iterations: int) -> Config:
        config = self.loader.load(self.path)
        config.max_iterations = iterations
        return config

    def describe(self) -> str:
        return str(self.path)


class MinimalConfigProvider(ConfigProvider):
    """Produces an in-memory configuration without external files."""

    def __init__(self, cascade_policy: Optional[CascadePolicy] = None) -> None:
        self._cascade_policy = cascade_policy or CascadePolicy()

    def load(self, iterations: int) -> Config:
        config = _construct_blank_config()
        config.max_iterations = iterations
        self._cascade_policy.apply(config)
        return config

    def describe(self) -> str:
        return "Inline minimal configuration"


class OpenEvolveRunner:
    """Coordinates asynchronous execution of OpenEvolve."""

    def __init__(
        self, open_evolve_factory: Callable[..., OpenEvolve], evaluator_path: Path
    ) -> None:
        self._factory = open_evolve_factory
        self._evaluator_path = evaluator_path

    async def _run_async(self, program_path: Path, config: Config) -> Any:
        orchestrator = self._factory(
            initial_program_path=str(program_path),
            evaluation_file=str(self._evaluator_path),
            config=config,
        )
        return await orchestrator.run()

    def run(self, program_path: Path, config: Config) -> Any:
        return asyncio.run(self._run_async(program_path, config))


class EvolutionReporter:
    """Formats evolution results for terminal output."""

    def report(self, result: Any, iterations: int, config_label: str) -> None:
        print("\n" + "=" * 60)
        print("EVOLUTION SUMMARY")
        print("=" * 60)
        print(f"Config: {config_label}")
        print(f"Iterations: {iterations}")
        print(f"Best score: {getattr(result, 'best_score', 'n/a')}")
        snippet = getattr(result, "best_code", "")
        print(f"\nBest program snippet:\n{snippet[:200]}...\n")


class EvolutionWorkflow:
    """High-level façade that ties together the supporting services."""

    def __init__(
        self,
        program_source: ProgramSource,
        config_provider: ConfigProvider,
        runner: OpenEvolveRunner,
        reporter: EvolutionReporter,
    ) -> None:
        self._program_source = program_source
        self._config_provider = config_provider
        self._runner = runner
        self._reporter = reporter

    def execute(self, iterations: int) -> Any:
        config = self._config_provider.load(iterations)
        with TemporaryProgramFile(self._program_source) as program_path:
            result = self._runner.run(program_path, config)
        self._reporter.report(result, iterations, self._config_provider.describe())
        return result


class FunctionEvolutionScenario:
    """Encapsulates the direct-function evolution example."""

    def __init__(self, factory: Callable[[int, int], Any]) -> None:
        self._factory = factory

    def run(self, iterations: int) -> None:
        def wrapper(bits_per_item: int):
            bloom = self._factory(key_bits=32, capacity=5000)
            bloom.bits_per_item = bits_per_item
            return bloom.bits_per_item

        def score_fn(bits: int) -> int:
            return abs(10 - bits)

        test_cases = [(value, score_fn(value)) for value in (8, 10, 12)]

        result = evolve_function(wrapper, test_cases=test_cases, iterations=iterations)
        print("=== Function evolution summary ===")
        print(f"iterations: {iterations}")
        print(f"best score: {getattr(result, 'best_score', 'n/a')}")
        print(f"best code:\n{getattr(result, 'best_code', '')}")


INITIAL_PROGRAM_SOURCE = ProgramSource(
    """
    import hashlib

    # EVOLVE-BLOCK-START
    class Candidate:
        def __init__(self, key_bits, capacity, bits_per_item=10):
            self.key_bits = key_bits
            self.capacity = capacity
            self.bits_per_item = bits_per_item
            self.fingerprint_bits = 16
            self.table = {}

        def add(self, item):
            h = hashlib.blake2b(item.to_bytes((self.key_bits + 7) // 8, 'little')).digest()
            key = int.from_bytes(h[:4], 'little') % self.capacity
            fingerprint = int.from_bytes(h[4:6], 'little')
            self.table[key] = fingerprint

        def query(self, item):
            h = hashlib.blake2b(item.to_bytes((self.key_bits + 7) // 8, 'little')).digest()
            key = int.from_bytes(h[:4], 'little') % self.capacity
            fingerprint = int.from_bytes(h[4:6], 'little')
            return self.table.get(key) == fingerprint
    # EVOLVE-BLOCK-END


    def candidate_factory(key_bits, capacity):
        return Candidate(key_bits, capacity)
    """
)

_CONFIG_LOADER = ConfigLoader()
_EVALUATOR_PATH = Path(__file__).parent / "evaluator.py"


def _build_workflow(config_provider: ConfigProvider) -> EvolutionWorkflow:
    runner = OpenEvolveRunner(OpenEvolve, _EVALUATOR_PATH)
    reporter = EvolutionReporter()
    return EvolutionWorkflow(
        program_source=INITIAL_PROGRAM_SOURCE,
        config_provider=config_provider,
        runner=runner,
        reporter=reporter,
    )


def demo_run_evolution_simple(iterations: int = 5) -> None:
    """Run evolution with an in-memory configuration."""
    workflow = _build_workflow(MinimalConfigProvider())
    workflow.execute(iterations)


def demo_run_evolution(
    iterations: int = 25,
    config_file: str = "configs/uniform_workload.yaml",
) -> None:
    """Run evolution with a YAML-backed configuration."""
    provider = YamlConfigProvider(Path(config_file), _CONFIG_LOADER)
    workflow = _build_workflow(provider)
    workflow.execute(iterations)


def demo_evolve_function(iterations: int = 10) -> None:
    """Run the direct-function evolution example."""
    FunctionEvolutionScenario(candidate_factory).run(iterations)


if __name__ == "__main__":
    demo_run_evolution(iterations=50, config_file="configs/minimal_hints_workload.yaml")
