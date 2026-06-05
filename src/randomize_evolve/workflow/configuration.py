import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import yaml


def _litellm_model_name(model: str | None) -> str | None:
    if not model:
        return None
    return model if "/" in model else f"openai/{model}"


@dataclass
class LeviRunConfig:
    """Resolved Levi configuration for one evolution run."""

    max_iterations: int
    problem_description: str
    function_signature: str
    model: str | None = None
    paradigm_model: str | None = None
    mutation_model: str | None = None
    budget_dollars: float | None = None
    budget_seconds: float | None = None
    output_dir: str | None = None
    pipeline: dict[str, Any] = field(default_factory=dict)
    behavior: dict[str, Any] = field(default_factory=dict)
    cvt: dict[str, Any] = field(default_factory=dict)
    init: dict[str, Any] = field(default_factory=dict)
    meta_advice: dict[str, Any] = field(default_factory=dict)
    punctuated_equilibrium: dict[str, Any] = field(default_factory=dict)
    cascade: dict[str, Any] = field(default_factory=dict)
    run_cost: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def evolve_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "budget_evals": self.max_iterations,
        }
        if self.model:
            kwargs["model"] = self.model
        else:
            kwargs["paradigm_model"] = self.paradigm_model
            kwargs["mutation_model"] = self.mutation_model
        if self.budget_dollars is not None:
            kwargs["budget_dollars"] = self.budget_dollars
        if self.budget_seconds is not None:
            kwargs["budget_seconds"] = self.budget_seconds
        if self.output_dir:
            kwargs["output_dir"] = self.output_dir
        if self.pipeline:
            kwargs["pipeline"] = self.pipeline
        if self.behavior:
            kwargs["behavior"] = self.behavior
        if self.cvt:
            kwargs["cvt"] = self.cvt
        if self.init:
            kwargs["init"] = self.init
        if self.meta_advice:
            kwargs["meta_advice"] = self.meta_advice
        if self.punctuated_equilibrium:
            kwargs["punctuated_equilibrium"] = self.punctuated_equilibrium
        if self.cascade:
            kwargs["cascade"] = self.cascade
        return kwargs


class ConfigLoader:
    """Loads repo YAML files into Levi run configuration objects."""

    def load(self, path: Path) -> LeviRunConfig:
        raw_data = self._read_yaml(path)
        return self.from_dict(raw_data)

    def from_dict(self, data: dict[str, Any]) -> LeviRunConfig:
        llm = data.get("llm", {}) or {}
        evaluator = data.get("evaluator", {}) or {}
        pipeline: dict[str, Any] = dict(data.get("pipeline", {}) or {})

        if llm.get("temperature") is not None:
            pipeline["temperature"] = llm["temperature"]
        if llm.get("max_tokens") is not None:
            pipeline["max_tokens"] = llm["max_tokens"]
        if evaluator.get("parallel_evaluations") is not None:
            pipeline["n_eval_processes"] = evaluator["parallel_evaluations"]
        if evaluator.get("timeout") is not None:
            pipeline["eval_timeout"] = evaluator["timeout"]
        cascade: dict[str, Any] = dict(data.get("cascade", {}) or {})
        if evaluator.get("cascade_evaluation") is not None:
            cascade["enabled"] = evaluator["cascade_evaluation"]

        model_override = os.environ.get("LEVI_MODEL")
        primary_model = _litellm_model_name(model_override or llm.get("primary_model"))
        secondary_model = _litellm_model_name(
            model_override or llm.get("secondary_model")
        )
        default_model = _litellm_model_name(os.environ.get("LEVI_MODEL", "gpt-4o-mini"))

        problem = data.get("problem", {}) or {}
        description = _compose_problem_description(data, problem)
        if not description:
            description = "Optimize the candidate_factory implementation for the configured evaluator."

        return LeviRunConfig(
            max_iterations=int(data.get("max_iterations") or 1),
            problem_description=description,
            function_signature=str(
                data.get("function_signature")
                or "def candidate_factory(*args, **kwargs):"
            ),
            paradigm_model=secondary_model or primary_model or default_model,
            mutation_model=primary_model or secondary_model or default_model,
            budget_dollars=data.get("budget_dollars"),
            budget_seconds=data.get("budget_seconds"),
            output_dir=data.get("output_dir"),
            pipeline=pipeline,
            behavior=data.get("behavior", {}) or {"score_keys": ["combined_score"]},
            cvt=data.get("cvt", {}) or {},
            init=data.get("init", {}) or {},
            meta_advice=data.get("meta_advice", {}) or {},
            punctuated_equilibrium=data.get("punctuated_equilibrium", {}) or {},
            cascade=cascade,
            run_cost=data.get("run_cost", {}) or {},
            raw=data,
        )

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data or {}


def _compose_problem_description(
    data: dict[str, Any],
    problem: dict[str, Any],
) -> str:
    """Build the full Levi prompt from YAML problem and prompt sections."""

    sections: list[str] = []
    base = str(problem.get("description") or data.get("description") or "").strip()
    if base:
        sections.append(base)

    prompt = data.get("prompt", {}) or {}
    system_message = str(prompt.get("system_message") or "").strip()
    if system_message:
        sections.append(system_message)

    objectives = prompt.get("objectives") or ()
    if objectives:
        objective_lines = "\n".join(f"- {objective}" for objective in objectives)
        sections.append(f"Objectives:\n{objective_lines}")

    search = data.get("search", {}) or {}
    notes = str(search.get("notes") or "").strip()
    if notes:
        sections.append(f"Search notes:\n{notes}")

    return "\n\n".join(sections)


class ConfigProvider(Protocol):
    """Abstracts the origin of Levi configuration objects."""

    def load(self, iterations: int) -> LeviRunConfig: ...

    def describe(self) -> str: ...


@dataclass(slots=True)
class YamlConfigProvider(ConfigProvider):
    """Loads configuration from YAML using the shared loader."""

    path: Path
    loader: ConfigLoader

    def load(self, iterations: int) -> LeviRunConfig:
        config = self.loader.load(self.path)
        config.max_iterations = iterations
        return config

    def describe(self) -> str:
        return str(self.path)


class MinimalConfigProvider(ConfigProvider):
    """Produces an in-memory Levi configuration without external files."""

    def __init__(
        self,
        problem_description: str = "Optimize the candidate_factory implementation.",
        function_signature: str = "def candidate_factory(*args, **kwargs):",
        model: Optional[str] = None,
    ) -> None:
        self._problem_description = problem_description
        self._function_signature = function_signature
        self._model = _litellm_model_name(
            model or os.environ.get("LEVI_MODEL", "gpt-4o-mini")
        )

    def load(self, iterations: int) -> LeviRunConfig:
        return LeviRunConfig(
            max_iterations=iterations,
            problem_description=self._problem_description,
            function_signature=self._function_signature,
            model=self._model,
            behavior={"score_keys": ["combined_score"]},
        )

    def describe(self) -> str:
        return "Inline minimal Levi configuration"
