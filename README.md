# randomize-evolve

Utilities for running OpenEvolve experiments that search for alternatives to
traditional Bloom filters.

## Directory layout

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

## OpenEvolve configuration

`configs/bloom_alternatives.yaml` demonstrates how to reference the evaluator
from an OpenEvolve problem definition. Tweak the numeric knobs to match your
hardware and desired workload difficulty. The configuration model is built with
Pydantic, so declare `pydantic` in your environment if you vend the evaluator
as a standalone package.

## Development environment

Project metadata and dependencies live in `pyproject.toml` and are managed with
`uv`. Typical workflow:

```bash
uv sync --dev
uv run python -c "from randomize_evolve.evaluators import BloomAlternativeEvaluator, baseline_bloom_filter; print(BloomAlternativeEvaluator()(baseline_bloom_filter(10)))"
```
