# Randomized Data Structures Evolution

**Welcome to Randomized Data Structures Evolution**

Our goal is to use the power of evolutionary strategies with large language models (LLMs) to evolve randomized data structures for (currently) the set membership problem. 

This repository could be modified for other similar problems, e.g., heavy hitter detection, approximate counting etc.

## Directory layout

- `evaluate.py`: OpenEvolve entry points that adapt the evaluator to the
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
from randomize_evolve.evaluators import Evaluator, baseline_bloom_filter

evaluator = Evaluator()
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

`configs/` demonstrates how to reference the evaluator
from an OpenEvolve problem definition. It includes LLM-assisted search settings,
database parameters, and evaluator coordination knobs. Adjust values to fit your
hardware budgets or organisational defaults.

## Data Distribution

The evaluator supports multiple data distribution patterns to test how well evolved structures handle different workloads.

### Available Distributions

#### 1. **UNIFORM** (default)
- Items distributed uniformly across the entire keyspace
- **Use case**: General-purpose testing, simulates random access patterns
- **Example**: Cache keys, random IDs

```python
config = EvaluatorConfig(distribution=Distribution.UNIFORM)
```

#### 2. **CLUSTERED**
- Items grouped into spatial clusters with configurable radius
- **Use case**: Locality-aware structures, range queries
- **Example**: Time-series data, geographic coordinates, database keys with prefixes
- **Parameters**:
  - `num_clusters`: Number of cluster centers (default: 10)
  - `cluster_radius`: Maximum distance from cluster center (default: 1000)

```python
config = EvaluatorConfig(
    distribution=Distribution.CLUSTERED,
    num_clusters=10,
    cluster_radius=1000,
)
```

**Good structures for clustered data:**
- Range filters
- Hierarchical structures (trees, skip lists)
- Bucketing schemes
- Spatial partitioning

#### 3. **SEQUENTIAL**
- Contiguous range of IDs
- **Use case**: Auto-increment keys, sequential allocation
- **Example**: Database auto-increment IDs, file handles

```python
config = EvaluatorConfig(distribution=Distribution.SEQUENTIAL)
```

**Good structures for sequential data:**
- Simple range tracking
- Run-length encoding
- Bitmap with run compression

#### 4. **POWER_LAW**
- Zipf/power-law distribution - some items much more frequent than others
- **Use case**: Real-world skewed workloads with "heavy hitters"
- **Example**: Web URLs, word frequencies, social network connections
- **Parameters**:
  - `power_law_exponent`: Controls skew (default: 1.5, higher = more skewed)

```python
config = EvaluatorConfig(
    distribution=Distribution.POWER_LAW,
    power_law_exponent=1.5,  # 2.5 for more skew
)
```

**Good structures for power-law data:**
- Frequency-aware caching
- Tiered storage (hot/cold)
- Count-min sketches
- Hybrid exact + approximate storage

## Tips

1. **Start small**: Test with 5-10 iterations first to verify setup
2. **Compare distributions**: Run same number of iterations for each distribution to compare
3. **Check metrics**: Look for structures that exploit distribution patterns
4. **Iterate**: If results plateau, try adjusting:
   - Temperature (creativity)
   - Population size (diversity)
   - Exploitation ratio (exploration vs refinement)
5. **Prompt engineering**: Bloom filters are hard to beat, so some prompt engineering may be needed to escape this minimum.


## Development environment

Project metadata and dependencies live in `pyproject.toml` and are managed with
`uv`. Typical workflow:

```bash
uv sync --dev
uv run python -c "from randomize_evolve.evaluators import Evaluator, baseline_bloom_filter; print(Evaluator()(baseline_bloom_filter(10)))"
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

### Quick Test

```bash
python test_distributions.py
```

This runs the baseline implementation against all distributions and compares:
- False positive rates
- Memory usage
- Query latency
- Build time
