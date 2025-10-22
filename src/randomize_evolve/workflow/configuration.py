import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import yaml
from openevolve.config import Config, LLMModelConfig


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
