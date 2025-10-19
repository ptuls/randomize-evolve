# randomize-evolve

Utilities for running OpenEvolve experiments that search for alternatives to
traditional Bloom filters.

## Directory layout

- `evaluate.py`: OpenEvolve entry points that adapt the Bloom evaluator to the
  platform's evaluation API (includes cascade stages).
- `initial_program.py`: Baseline Bloom filter factory used as a starting point
  for evolutionary runs.
- `src/randomize_evolve/`: Python package housing evaluator implementations.
- `configs/`: Example OpenEvolve problem configurations that wire the evaluator
  into the search loop.

## Bloom alternative evaluator

The evaluator lives in
`src/randomize_evolve/evaluators/bloom_alternatives.py` and packages the steps
needed to score a candidate probabilistic set-membership structure:

1. Generate reproducible workloads across multiple random seeds.
2. Record throughput, false positives, and false negatives for each seed.
3. Convert the aggregated metrics into a scalar fitness score for OpenEvolve.

### Candidate contract

OpenEvolve should supply a factory callable to the evaluator. The callable must
accept `(key_bits, capacity)` and return an object that implements:

```python
def add(item: int) -> None
def query(item: int) -> bool
```

Any raised exception, timeout, or protocol violation is treated as a failed
trial and penalised accordingly.

### Baseline sanity check

Use `baseline_bloom_filter(bits_per_item)` to verify the evaluator before
launching a search:

```python
from randomize_evolve.evaluators import BloomAlternativeEvaluator, baseline_bloom_filter

evaluator = BloomAlternativeEvaluator()
result = evaluator(baseline_bloom_filter(bits_per_item=10))
print(result)
```

## OpenEvolve entry points

The root `evaluate.py` module exposes the functions OpenEvolve expects:

- `evaluate_stage1(path)` runs a lightweight cascade pass with a reduced
  workload.
- `evaluate_stage2(path)` and `evaluate(path)` run the full evaluator with the
  configuration mirrored from the YAML example.

`path` should point to a Python module that defines either
`candidate_factory(key_bits, capacity)` or `build_candidate(key_bits, capacity)`
and returns an object that implements `add()` and `query()`.

## Seed program

`initial_program.py` provides a deterministic Bloom filter implementation wired
through the `candidate_factory` entry point. It marks the section targeted for
evolution with an `EVOLVE-BLOCK` comment and ships with a simple `run_demo()`
smoke test:

```bash
uv run python initial_program.py
```

This script can serve as the initial population member when launching an
OpenEvolve run.

## OpenEvolve configuration

`configs/bloom_alternatives.yaml` demonstrates how to reference the evaluator
from an OpenEvolve problem definition. It includes LLM-assisted search settings,
database parameters, and evaluator coordination knobs. Adjust values to fit your
hardware budgets or organisational defaults.

## Development environment

Project metadata and dependencies live in `pyproject.toml` and are managed with
`uv`. Typical workflow:

```bash
uv sync --dev
uv run python -c "from randomize_evolve.evaluators import BloomAlternativeEvaluator, baseline_bloom_filter; print(BloomAlternativeEvaluator()(baseline_bloom_filter(10)))"
```

To execute the full evaluator against a local candidate module:

```bash
uv run python -c "from evaluate import evaluate; from pathlib import Path; print(evaluate(Path('initial_program.py')))"
```

To experiment with OpenEvolve's library API and the Bloom configuration, run the
inline demo script:

```bash
uv run python run_demo.py
```
