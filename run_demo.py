"""Example script demonstrating how to invoke OpenEvolve as a library.

It mirrors the inline ``run_evolution`` example but wires in the Bloom filter
evaluator, configuration, and baseline program defined in this repository.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import yaml

from openevolve import evolve_function, run_evolution, OpenEvolve
from openevolve.config import Config, LLMModelConfig

from evaluator import evaluate
from initial_program import candidate_factory

# Inline program string that will be written to disk by OpenEvolve while it
# applies evolutionary steps. The EVOLVE-BLOCK comment matches the baseline file
# to point out which section the search should mutate.
INITIAL_PROGRAM = textwrap.dedent(
    """\
    import hashlib

    BYTES_TO_BITS = 8

    class Candidate:
        def __init__(self, key_bits, capacity, bits_per_item=10):
            self.key_bits = key_bits
            self.capacity = capacity
            self.bits_per_item = bits_per_item
            self.bit_count = max(capacity * bits_per_item, 64)
            self.byte_count = (self.bit_count + 7) // 8
            self.storage = bytearray(self.byte_count)
            self.seeds = (
                b"inline-seed-00",
                b"inline-seed-01",
                b"inline-seed-02",
                b"inline-seed-03",
            )

        # EVOLVE-BLOCK-START
        def add(self, item):
            for index in self._indices(item):
                self._set_bit(index)

        def query(self, item):
            return all(self._get_bit(index) for index in self._indices(item))
        # EVOLVE-BLOCK-END

        def _indices(self, item):
            data = item.to_bytes(length=(self.key_bits + BYTES_TO_BITS - 1) // BYTES_TO_BITS, byteorder="little")
            for seed in self.seeds:
                digest = hashlib.blake2b(data, digest_size=8, person=seed).digest()
                yield int.from_bytes(digest, "little") % self.bit_count

        def _set_bit(self, index):
            byte_index = index // BYTES_TO_BITS
            self.storage[byte_index] |= 1 << (index % BYTES_TO_BITS)

        def _get_bit(self, index):
            byte_index = index // BYTES_TO_BITS
            return bool(self.storage[byte_index] & (1 << (index % BYTES_TO_BITS)))


    def candidate_factory(key_bits, capacity):
        return Candidate(key_bits, capacity)
    """
)


def load_bloom_config(config_path: Path) -> Config:
    """Load the YAML configuration and inject the OpenAI API key."""

    api_key = os.environ.get("OPENAI_API_KEY")

    try:
        from openevolve.config import load_config  # type: ignore
    except ImportError:
        load_config = None

    if load_config:
        cfg = load_config(config_path)  # type: ignore[call-arg]
        cfg = _inject_api_key(cfg, api_key)
        _disable_cascade_when_missing_stages(cfg)
        return cfg

    if hasattr(Config, "from_file"):
        try:
            cfg = Config.from_file(config_path)  # type: ignore[call-arg]
            cfg = _inject_api_key(cfg, api_key)
            _disable_cascade_when_missing_stages(cfg)
            return cfg
        except Exception:
            pass

    with config_path.open("r", encoding="utf-8") as handle:
        config_data: Any = yaml.safe_load(handle)

    cfg = _construct_config(config_data)
    cfg = _inject_api_key(cfg, api_key, raw_llm=config_data.get("llm", {}))
    _disable_cascade_when_missing_stages(cfg)
    return cfg


def demo_run_evolution_simple(iterations: int = 5) -> None:
    """Run OpenEvolve using the low-level API with file-based evaluator."""
    import asyncio
    import tempfile

    async def run_async():
        # Write initial program to a temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(INITIAL_PROGRAM)
            program_path = f.name

        try:
            # Use minimal config - disable multiprocessing features
            config = Config()
            config.max_iterations = iterations

            # Get evaluator file path
            evaluator_path = str(Path(__file__).parent / "evaluator.py")

            # Initialize OpenEvolve with file paths
            oe = OpenEvolve(
                initial_program_path=program_path,
                evaluation_file=evaluator_path,
                config=config,
            )

            # Run evolution
            result = await oe.run()

            print("=== Inline evolution summary ===")
            print(f"iterations: {iterations}")
            print(f"best score: {getattr(result, 'best_score', 'n/a')}")
            snippet = getattr(result, "best_code", "")[:200]
            print(f"best program snippet:\n{snippet}...\n")
            return result
        finally:
            import os
            if os.path.exists(program_path):
                os.unlink(program_path)

    # Run the async function
    asyncio.run(run_async())


def demo_run_evolution(iterations: int = 25) -> None:
    """Run OpenEvolve on the inline Bloom filter program with full config."""
    import asyncio
    import tempfile

    async def run_async():
        # Write initial program to a temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(INITIAL_PROGRAM)
            program_path = f.name

        try:
            # Load full config with LLM settings
            config = load_bloom_config(Path("configs/bloom_alternatives.yaml"))
            config.max_iterations = iterations

            # Get evaluator file path
            evaluator_path = str(Path(__file__).parent / "evaluator.py")

            # Initialize OpenEvolve with file paths
            oe = OpenEvolve(
                initial_program_path=program_path,
                evaluation_file=evaluator_path,
                config=config,
            )

            # Run evolution
            result = await oe.run()

            print("=== Inline evolution summary ===")
            print(f"iterations: {iterations}")
            print(f"best score: {getattr(result, 'best_score', 'n/a')}")
            snippet = getattr(result, "best_code", "")[:200]
            print(f"best program snippet:\n{snippet}...\n")
            return result
        finally:
            import os
            if os.path.exists(program_path):
                os.unlink(program_path)

    # Run the async function
    asyncio.run(run_async())


def demo_evolve_function(iterations: int = 10) -> None:
    """Show how to evolve functions directly using the Bloom candidate factory."""

    def wrapper(bits_per_item: int):
        bloom = candidate_factory(key_bits=32, capacity=5000)
        bloom.bits_per_item = bits_per_item
        return bloom.bits_per_item

    def score_fn(bits: int) -> int:
        # Lower is better: distance from a 10 bits-per-item baseline.
        return abs(10 - bits)

    test_cases = [
        (8, score_fn(8)),
        (10, score_fn(10)),
        (12, score_fn(12)),
    ]

    result = evolve_function(
        wrapper,
        test_cases=test_cases,
        iterations=iterations,
    )

    print("=== Function evolution summary ===")
    print(f"iterations: {iterations}")
    print(f"best score: {getattr(result, 'best_score', 'n/a')}")
    print(f"best code:\n{getattr(result, 'best_code', '')}")


def _construct_config(data: Any) -> Config:
    try:
        if hasattr(Config, "model_validate"):
            return Config.model_validate(data)  # type: ignore[return-value,attr-defined]
        return Config(**data)  # type: ignore[arg-type]
    except Exception:
        return Config()


def _disable_cascade_when_missing_stages(cfg: Config) -> None:
    evaluator_cfg = getattr(cfg, "evaluator", None)
    if evaluator_cfg is None:
        return
    if hasattr(evaluator_cfg, "cascade_evaluation"):
        setattr(evaluator_cfg, "cascade_evaluation", False)


def _inject_api_key(
    cfg: Config,
    api_key: str | None,
    *,
    raw_llm: dict[str, Any] | None = None,
) -> Config:
    if not api_key:
        return cfg

    llm = getattr(cfg, "llm", None)
    if llm is None:
        return cfg

    models = getattr(llm, "models", None)
    if not models and raw_llm:
        primary = raw_llm.get("primary_model")
        primary_weight = raw_llm.get("primary_model_weight")
        secondary = raw_llm.get("secondary_model")
        secondary_weight = raw_llm.get("secondary_model_weight")
        models = []
        if primary:
            models.append(LLMModelConfig(name=primary, weight=primary_weight, api_key=api_key))
        if secondary:
            models.append(LLMModelConfig(name=secondary, weight=secondary_weight, api_key=api_key))
        if hasattr(llm, "models"):
            llm.models = models  # type: ignore[attr-defined]
        else:
            setattr(llm, "models", models)
        return cfg

    if models:
        for model in models:
            if isinstance(model, dict):
                model.setdefault("api_key", api_key)
            elif hasattr(model, "api_key") and not getattr(model, "api_key"):
                model.api_key = api_key  # type: ignore[attr-defined]
        return cfg

    for attr in ("primary_model", "secondary_model"):
        model = getattr(llm, attr, None)
        if not model:
            continue
        if isinstance(model, dict):
            model.setdefault("api_key", api_key)
        elif hasattr(model, "api_key") and not getattr(model, "api_key"):
            model.api_key = api_key  # type: ignore[attr-defined]
    return cfg


if __name__ == "__main__":
    # Use the version with full config and LLM support
    demo_run_evolution(iterations=50)  # Start with 5 iterations for testing
