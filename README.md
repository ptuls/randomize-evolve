# Randomized Data Structures Evolution

**Welcome to Randomized Data Structures Evolution**

Our goal is to use the power of evolutionary strategies with large language models (LLMs) 
to evolve randomized data structures for (currently) the set membership problem.

In addition to Bloom-filter alternatives, the repository now ships with tooling for
streaming heavy-hitter detection based on approximate counting sketches.

## Why Levi

This repo uses [Levi](https://ttanv.github.io/levi) as the
outer LLM-assisted evolution loop. Levi takes a task description, a seed
program, a function signature, and a scoring callable, then iteratively proposes
candidate code and keeps the variants that improve the score.

We moved from OpenEvolve to Levi to keep the integration closer to the shape of
this project: the repo already owns the important domain logic in its
evaluators, and Levi lets us expose that logic directly as a Python `score_fn`.
The current adapter maps each task's `combined_score` into Levi's required
`{"score": ...}` result while preserving the existing Bloom, heavy-hitter, and
packet-switching evaluator contracts.

The tradeoff is that OpenEvolve-specific controls such as MAP-Elites feature
bins, island migration, archive sizing, novelty thresholds, and database tuning
are no longer active. Levi now owns the search behavior. The YAML files still
carry task descriptions, model settings, evaluator timeouts, and parallelism,
but performance should be benchmarked under equal evaluation budgets rather
than assumed to improve automatically.

## Directory layout

- `src/randomize_evolve/`: Python package housing evaluator logic and workflow
  helpers (`workflow/` contains small, composable orchestration utilities).
- `src/randomize_evolve/problems/`: Problem-specific Levi entry points,
  runners, initial programs, and seed portfolios.
- `src/randomize_evolve/problems/set_membership/`: Bloom-filter alternative
  runner, Levi evaluator entry point, initial program, and alternative seeds.
- `src/randomize_evolve/problems/heavy_hitters/`: Streaming heavy-hitter runner,
  Levi evaluator entry point, and Count-Min baseline program.
- `src/randomize_evolve/problems/packet_switching/`: Packet-switching runner,
  Levi evaluator entry point, baseline/evolution programs, and scheduler seeds.
- `configs/`: Example Levi problem configurations that wire the
  evaluators into the search loop for different workloads.
- `tests/`: Lightweight regression scripts for evaluator behavior and seeds.

## Bloom alternative evaluator

The evaluator lives in
`src/randomize_evolve/evaluators/bloom_alternatives.py` and packages the steps
needed to score a candidate probabilistic set-membership structure:

1. Generate reproducible workloads across multiple random seeds.
2. Record throughput, false positives, and false negatives for each seed.
3. Convert the aggregated metrics into a scalar fitness score for Levi.

### Candidate contract

Levi should supply a factory callable to the evaluator. The callable must
accept `(key_bits, capacity)` and return an object that implements:

```python
def add(item: int) -> None
def query(item: int) -> bool
```

Any raised exception, timeout, or protocol violation is treated as a failed
trial and penalized accordingly.

### Baseline sanity check

Use `baseline_bloom_filter(bits_per_item)` to verify the evaluator before
launching a search:

```python
from randomize_evolve.evaluators import Evaluator, baseline_bloom_filter

evaluator = Evaluator()
result = evaluator(baseline_bloom_filter(bits_per_item=10))
print(result)
```

## Levi entry points

Each problem keeps its Levi-facing evaluator beside its runner and seed
programs:

- `randomize_evolve.problems.set_membership.evaluator`
- `randomize_evolve.problems.heavy_hitters.evaluator`
- `randomize_evolve.problems.packet_switching.evaluator`

Each evaluator exposes `evaluate(path)`, `evaluate_factory(factory)`, and
`evaluate_source(source)`. Point `path` at a Python module that defines the
problem's `candidate_factory(...)` or `build_candidate(...)`.

## Seed program

`randomize_evolve.problems.set_membership.initial_program` provides a
deterministic Bloom filter implementation wired through the `candidate_factory`
entry point. It marks the section targeted for evolution with an `EVOLVE-BLOCK`
comment and ships with a simple `run_demo()` smoke test:

```bash
uv run python -m randomize_evolve.problems.set_membership.initial_program
```

This script can serve as the initial seed program when launching a
Levi run.

`randomize_evolve.problems.heavy_hitters.initial_program` mirrors this pattern
for heavy hitters by exposing a Count-Min sketch baseline that satisfies the
streaming interface. Run it directly to see the demo output:

```bash
uv run python -m randomize_evolve.problems.heavy_hitters.initial_program
```

## Heavy hitter evaluator

The streaming evaluator in `src/randomize_evolve/evaluators/heavy_hitters.py`
tracks approximate frequency estimators. Candidates must expose:

```python
def observe(item: int, weight: int = 1) -> None
def estimate(item: int) -> int
def top_k(k: int) -> List[Tuple[int, int]]
```

Trials generate skewed streams with configurable heavy-hitter fractions and
measure recall, precision, relative error, and zero-frequency mistakes. The
`baseline_count_min_sketch()` helper offers a sanity-check implementation:

```python
from randomize_evolve.evaluators.heavy_hitters import (
    Evaluator,
    EvaluatorConfig,
    baseline_count_min_sketch,
)

evaluator = Evaluator(EvaluatorConfig(stream_length=10000, top_k=8))
result = evaluator(baseline_count_min_sketch())
print(result)
```

## Levi configuration

`configs/` demonstrates how to reference the evaluators from a Levi
problem definition. It includes LLM-assisted search settings, evaluator
coordination knobs, and task-specific prompt context. Adjust values to fit your
hardware budgets or organizational defaults. Multiple workload-specific YAML
files (uniform, clustered, power-law, aggressive exploration, minimal hints,
packet switching, and heavy hitters) are available; pass them to the relevant
runner script to explore different regimes.

The active Levi adapter uses these settings:

- `max_iterations`, overridden by each runner's `--iterations` argument where
  available.
- `llm.primary_model` and `llm.secondary_model`, or `LEVI_MODEL` to override
  both for smoke tests.
- `llm.temperature` and `llm.max_tokens`.
- `evaluator.timeout` and `evaluator.parallel_evaluations`.
- `problem.description`, plus the seed program and evaluator-specific
  `combined_score`.

OpenEvolve-era `database` settings remain in some YAML files as historical
context, but they are not currently interpreted by Levi.

## Alternative seeds

The `randomize_evolve.problems.set_membership.alternative_seeds.available_seeds()`
helper exposes several pre-built program templates (Cuckoo-inspired,
quotient-based, XOR-based). Import the map and select the desired seed when you
want to start a run from a different candidate family:

```python
from randomize_evolve.problems.set_membership.alternative_seeds import available_seeds

seed = available_seeds()["cuckoo"]["program"]
# Persist the seed or inject it into your Levi run before launching.
```

## Workflow utilities

The problem runners coordinate evolution runs using the composable helpers under
`src/randomize_evolve/workflow/`:

Examples:

```bash
export OPENAI_API_KEY=...
LEVI_MODEL=gpt-4o-mini uv run python -m randomize_evolve.problems.set_membership.run --iterations 5 --config configs/uniform_workload.yaml
LEVI_MODEL=gpt-4o-mini uv run python -m randomize_evolve.problems.heavy_hitters.run --iterations 5 --config configs/heavy_hitters_workload.yaml
LEVI_MODEL=gpt-4o-mini uv run python -m randomize_evolve.problems.packet_switching.run --iterations 5 --config configs/packet_switching_workload.yaml
uv run python -m randomize_evolve.problems.packet_switching.run --compare-only
```

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
   - Evaluation budget (`--iterations`)
   - Seed program or workload config
   - Prompt context in the task YAML
5. **Prompt engineering**: Bloom filters are hard to beat, so some prompt engineering may be needed to escape this minimum.


## Development environment

Project metadata and dependencies live in `pyproject.toml` and are managed with
`uv`. Typical workflow:

```bash
uv sync --extra dev
uv run python -c "from randomize_evolve.evaluators import Evaluator, baseline_bloom_filter; print(Evaluator()(baseline_bloom_filter(10)))"
```

To execute the full evaluator against a local candidate module:

```bash
uv run python -c "from randomize_evolve.problems.set_membership.evaluator import evaluate; from pathlib import Path; print(evaluate(Path('src/randomize_evolve/problems/set_membership/initial_program.py')))"
```

To run a short Levi-backed Bloom evolution:

```bash
export OPENAI_API_KEY=...
LEVI_MODEL=gpt-4o-mini uv run python -m randomize_evolve.problems.set_membership.run --iterations 5 --config configs/uniform_workload.yaml
```

### Quick Test

```bash
uv run --extra dev pytest
```

This runs the baseline implementation against all distributions and compares:
- False positive rates
- Memory usage
- Query latency
- Build time
