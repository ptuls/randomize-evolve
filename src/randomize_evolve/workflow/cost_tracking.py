"""Utilities for tracking LLM token usage and estimated run cost."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

_TOKENS_PER_MILLION = 1_000_000
_RUN_COST_EVENTS_PATH_ENV = "RANDOMIZE_EVOLVE_RUN_COST_EVENTS_PATH"
_PROMPT_CACHE_KEY_PREFIX_ENV = "RANDOMIZE_EVOLVE_PROMPT_CACHE_KEY_PREFIX"
_PROMPT_CACHE_RETENTION_ENV = "RANDOMIZE_EVOLVE_PROMPT_CACHE_RETENTION"


@dataclass
class ModelPricing:
    """Per-model pricing expressed in USD per 1M tokens."""

    input_per_1m_tokens: Optional[float] = None
    output_per_1m_tokens: Optional[float] = None
    cached_input_per_1m_tokens: Optional[float] = None


@dataclass
class ModelUsage:
    """Aggregated token usage for a single model."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: Optional[float] = None


@dataclass
class RunCostSummary:
    """Run-level usage totals plus an optional dollar estimate."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: Optional[float] = None
    per_model: Dict[str, ModelUsage] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "per_model": {
                model: asdict(usage) for model, usage in sorted(self.per_model.items())
            },
        }


class RunCostTracker:
    """Thread-safe aggregation of LLM usage and optional model pricing."""

    def __init__(
        self, pricing_by_model: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        self._lock = threading.Lock()
        self._usage_by_model: Dict[str, ModelUsage] = {}
        self._pricing_by_model = {
            model: ModelPricing(**pricing)
            for model, pricing in (pricing_by_model or {}).items()
        }

    def record_usage(
        self,
        model: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: Optional[int] = None,
        cached_prompt_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        with self._lock:
            usage = self._usage_by_model.setdefault(model, ModelUsage())
            usage.requests += 1
            usage.prompt_tokens += int(prompt_tokens)
            usage.completion_tokens += int(completion_tokens)
            usage.cached_prompt_tokens += int(cached_prompt_tokens)
            usage.reasoning_tokens += int(reasoning_tokens)
            usage.total_tokens += int(
                total_tokens
                if total_tokens is not None
                else int(prompt_tokens) + int(completion_tokens)
            )

    def record_response(self, model: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            logger.debug(
                "OpenAI response for model %s did not include usage details", model
            )
            return

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
            or (prompt_tokens + completion_tokens)
        )

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        cached_prompt_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)

        self.record_usage(
            model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def build_summary(self) -> RunCostSummary:
        with self._lock:
            per_model = {
                model: ModelUsage(**asdict(usage))
                for model, usage in self._usage_by_model.items()
            }

        summary = RunCostSummary(per_model=per_model)
        missing_pricing = False

        for model, usage in per_model.items():
            summary.requests += usage.requests
            summary.prompt_tokens += usage.prompt_tokens
            summary.completion_tokens += usage.completion_tokens
            summary.total_tokens += usage.total_tokens
            summary.cached_prompt_tokens += usage.cached_prompt_tokens
            summary.reasoning_tokens += usage.reasoning_tokens

            model_cost = self._estimate_model_cost(model, usage)
            usage.estimated_cost_usd = model_cost
            if model_cost is None and usage.requests > 0:
                missing_pricing = True

        if missing_pricing:
            summary.estimated_cost_usd = None
        else:
            summary.estimated_cost_usd = sum(
                usage.estimated_cost_usd or 0.0 for usage in per_model.values()
            )

        return summary

    def _estimate_model_cost(self, model: str, usage: ModelUsage) -> Optional[float]:
        pricing = self._pricing_by_model.get(model)
        if pricing is None:
            return None
        if pricing.input_per_1m_tokens is None or pricing.output_per_1m_tokens is None:
            return None

        cached_prompt_tokens = min(usage.cached_prompt_tokens, usage.prompt_tokens)
        uncached_prompt_tokens = usage.prompt_tokens - cached_prompt_tokens

        input_cost = (
            uncached_prompt_tokens * pricing.input_per_1m_tokens / _TOKENS_PER_MILLION
        )
        if cached_prompt_tokens > 0:
            cached_rate = (
                pricing.cached_input_per_1m_tokens
                if pricing.cached_input_per_1m_tokens is not None
                else pricing.input_per_1m_tokens
            )
            input_cost += cached_prompt_tokens * cached_rate / _TOKENS_PER_MILLION

        output_cost = (
            usage.completion_tokens * pricing.output_per_1m_tokens / _TOKENS_PER_MILLION
        )
        return input_cost + output_cost


def tracked_openai_client_factory(model_cfg: Any) -> Any:
    """Create an OpenAI client that appends token usage events to a shared file."""

    from openevolve.llm.openai import OpenAILLM

    class TrackedOpenAILLM(OpenAILLM):
        async def _call_api(self, params: Dict[str, Any]) -> str:
            prompt_cache_key_prefix = os.environ.get(_PROMPT_CACHE_KEY_PREFIX_ENV)
            if prompt_cache_key_prefix and "prompt_cache_key" not in params:
                params["prompt_cache_key"] = f"{prompt_cache_key_prefix}:{self.model}"

            prompt_cache_retention = os.environ.get(_PROMPT_CACHE_RETENTION_ENV)
            if prompt_cache_retention and "prompt_cache_retention" not in params:
                params["prompt_cache_retention"] = prompt_cache_retention

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.client.chat.completions.create(**params)
            )

            events_path = os.environ.get(_RUN_COST_EVENTS_PATH_ENV)
            if events_path:
                _append_usage_event(Path(events_path), self.model, response)

            logger.debug("API parameters: %s", params)
            logger.debug("API response: %s", response.choices[0].message.content)
            return response.choices[0].message.content

    return TrackedOpenAILLM(model_cfg)


@contextmanager
def run_cost_tracking_environment(
    events_path: Path,
    *,
    prompt_cache_key_prefix: Optional[str] = None,
    prompt_cache_retention: Optional[str] = None,
) -> Iterator[None]:
    """Expose run cost and prompt-cache settings to all worker processes."""

    previous_values = {
        _RUN_COST_EVENTS_PATH_ENV: os.environ.get(_RUN_COST_EVENTS_PATH_ENV),
        _PROMPT_CACHE_KEY_PREFIX_ENV: os.environ.get(_PROMPT_CACHE_KEY_PREFIX_ENV),
        _PROMPT_CACHE_RETENTION_ENV: os.environ.get(_PROMPT_CACHE_RETENTION_ENV),
    }
    os.environ[_RUN_COST_EVENTS_PATH_ENV] = str(events_path)
    if prompt_cache_key_prefix:
        os.environ[_PROMPT_CACHE_KEY_PREFIX_ENV] = prompt_cache_key_prefix
    if prompt_cache_retention:
        os.environ[_PROMPT_CACHE_RETENTION_ENV] = prompt_cache_retention
    try:
        yield
    finally:
        for env_name, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous_value


def extract_run_cost_config(config: Any) -> Dict[str, Any]:
    """Read the repo's optional run cost settings attached to the Config object."""

    return getattr(config, "_run_cost_config", {}) or {}


def configure_tracked_model_clients(config: Any) -> None:
    """Configure OpenEvolve model configs to use the tracked OpenAI client factory."""

    llm_cfg = getattr(config, "llm", None)
    if llm_cfg is None:
        return

    for model in list(getattr(llm_cfg, "models", [])) + list(
        getattr(llm_cfg, "evaluator_models", [])
    ):
        if getattr(model, "init_client", None) is None:
            model.init_client = tracked_openai_client_factory


def build_summary_from_events_file(
    events_path: Path, pricing_by_model: Optional[Dict[str, Dict[str, Any]]] = None
) -> RunCostSummary:
    """Aggregate usage events from disk into a run-level summary."""

    tracker = RunCostTracker(pricing_by_model)
    if not events_path.exists():
        return tracker.build_summary()

    for raw_line in events_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        event = json.loads(raw_line)
        tracker.record_usage(
            event["model"],
            prompt_tokens=event["prompt_tokens"],
            completion_tokens=event["completion_tokens"],
            total_tokens=event["total_tokens"],
            cached_prompt_tokens=event.get("cached_prompt_tokens", 0),
            reasoning_tokens=event.get("reasoning_tokens", 0),
        )

    return tracker.build_summary()


def save_run_cost_summary(output_dir: Path, summary: RunCostSummary) -> Path:
    """Persist the run cost summary next to other OpenEvolve outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_cost_summary.json"
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return path


def _append_usage_event(events_path: Path, model: str, response: Any) -> None:
    event = _usage_event_from_response(model, response)
    if event is None:
        return

    with events_path.open("a", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            fcntl = None  # type: ignore[assignment]

        handle.write(json.dumps(event))
        handle.write("\n")
        handle.flush()

        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _usage_event_from_response(model: str, response: Any) -> Optional[Dict[str, Any]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.debug(
            "OpenAI response for model %s did not include usage details", model
        )
        return None

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        or (prompt_tokens + completion_tokens)
    )
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)

    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
        "reasoning_tokens": int(
            getattr(completion_details, "reasoning_tokens", 0) or 0
        ),
    }
