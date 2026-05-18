# Randomized Data Structures Evolution

**Welcome to Randomized Data Structures Evolution**

Our goal is to use the power of evolutionary strategies with large language models (LLMs)
to evolve randomized data structures for (currently) the set membership problem.

In addition to Bloom-filter alternatives, the repository now ships with tooling for
streaming heavy-hitter detection based on approximate counting sketches and
packet-switch scheduling experiments.

## Directory layout

- `evaluator.py`: Direct evaluation entry point consumed by OpenEvolve for Bloom
  alternatives.
- `heavy_hitters_evaluator.py`: Evaluation entry point for approximate heavy
  hitter algorithms.
- `packet_switching_evaluator.py`: Evaluation entry point for packet-switching
  scheduler candidates.
- `initial_program.py`: Baseline Bloom filter factory used as a starting point
  for evolutionary runs.
- `initial_program_heavy_hitters.py`: Baseline Count-Min style heavy hitter
  implementation wired to the streaming evaluator.
- `initial_program_packet_switching.py`: Baseline round-robin style scheduler
  used as the seed for packet-switching evolution.
- `run_set_membership.py`: Convenience launcher for Bloom-filter evolution
  workflows.
- `run_heavy_hitters.py`: Convenience launcher for heavy-hitter evolution
  workflows.
- `run_packet_switching.py`: Convenience launcher for packet-switching search
  workflows.
- `alternative_seeds.py`: Optional seed programs that explore different design
  patterns (Cuckoo-style, quotient-based, XOR-based).
- `src/randomize_evolve/`: Python package housing evaluator logic and workflow
  helpers (`workflow/` contains small, composable orchestration utilities).
- `configs/`: Example OpenEvolve problem configurations that wire the
  evaluators into the search loop for different workloads.
- `tests/`: Lightweight regression scripts for evaluator behavior and seeds.

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

## OpenEvolve entry points

The root `evaluator.py` module exposes a single `evaluate(path)` function. Point
`path` at a Python module that defines `candidate_factory(key_bits, capacity)`
or `build_candidate(key_bits, capacity)` and returns objects implementing
`add()` and `query()`. The evaluator runs in direct mode; cascade evaluations
are disabled by default because `evaluator.py` implements only the full pass.

For streaming heavy-hitter experiments, use `heavy_hitters_evaluator.py` with
candidates that implement `observe(item, weight)`, `estimate(item)`, and
`top_k(k)`.

For packet-switching experiments, use `packet_switching_evaluator.py` with
candidates that implement `candidate_factory(ports)` and return a scheduler
object exposing `select_matches(requests, time_slot, queue_lengths)`.

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

`initial_program_heavy_hitters.py` mirrors this pattern for heavy hitters by
exposing a Count-Min sketch baseline that satisfies the streaming interface.
Run it directly to see the demo output:

```bash
uv run python initial_program_heavy_hitters.py
```

`initial_program_packet_switching.py` does the same for packet switching with a
deterministic queue-aware round-robin scheduler:

```bash
uv run python initial_program_packet_switching.py
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

## Packet switching evaluator

The packet-switching evaluator in
`src/randomize_evolve/evaluators/packet_switching.py` scores schedulers across
multiple traffic regimes including uniform, bursty, hotspot, and heavy-load
scenarios. Candidates must expose a factory that accepts `ports` and returns an
object implementing:

```python
def select_matches(
    requests: Dict[int, List[int]],
    time_slot: int,
    queue_lengths: Sequence[int],
) -> MutableMapping[int, int]
```

The evaluator combines throughput, input fairness, flow fairness, and drop rate
into a scalar score. To sanity-check the baseline scheduler without launching a
search:

```python
from randomize_evolve.evaluators.packet_switching import PacketSwitchingEvaluator
from randomize_evolve.packet_switching import RoundRobinScheduler

evaluator = PacketSwitchingEvaluator()
result = evaluator(lambda ports: RoundRobinScheduler(ports, ports))
print(result)
```

## OpenEvolve configuration

`configs/` demonstrates how to reference the evaluators from an OpenEvolve
problem definition. It includes LLM-assisted search settings, database
parameters, and evaluator coordination knobs. Adjust values to fit your
hardware budgets or organizational defaults. Multiple workload-specific YAML
files (uniform, clustered, power-law, aggressive exploration, minimal hints,
heavy-hitters, and packet-switching) are available; point the corresponding
runner script at any of them to explore different regimes.

## Alternative seeds

The `alternative_seeds.available_seeds()` helper exposes several pre-built
program templates (Cuckoo-inspired, quotient-based, XOR-based). Import the map
and select the desired seed to initialize the database with a more diverse
starting population:

```python
from alternative_seeds import available_seeds

seed = available_seeds()["cuckoo"]["program"]
# Persist the seed or inject it into your OpenEvolve database before launching.
```

## Workflow utilities

The `run_set_membership.py`, `run_heavy_hitters.py`, and
`run_packet_switching.py` scripts coordinate evolution runs using the
composable helpers under `src/randomize_evolve/workflow/`.

To launch the packet-switching search with the bundled workload:

```bash
uv run python run_packet_switching.py
```

The packet-switching runner defaults to
`configs/packet_switching_workload.yaml`. To change the number of iterations or
config file, call `demo_run_evolution()` directly:

```bash
uv run python - <<'PY'
from run_packet_switching import demo_run_evolution

demo_run_evolution(
    iterations=20,
    config_file="configs/packet_switching_workload.yaml",
)
PY
```

The set-membership runner exposes a few convenience functions:

- `demo_run_evolution_simple(iterations=5)` uses an in-memory configuration for
  quick smoke tests.
- `demo_run_evolution(iterations=25, config_file=...)` launches a full
  OpenEvolve session given any YAML config in `configs/`.
- `demo_evolve_function(iterations=10)` demonstrates direct function evolution
  with the baseline candidate factory.

You can `python - <<'PY'` to call these functions programmatically or invoke the
module as a script to run the `demo_run_evolution` scenario.

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
uv run python -c "from evaluator import evaluate; print(evaluate('initial_program.py'))"
```

To execute the packet-switching evaluator against the bundled baseline module:

```bash
uv run python -c "from packet_switching_evaluator import evaluate; print(evaluate('initial_program_packet_switching.py'))"
```

To experiment with OpenEvolve's library API, run one of the workflow launchers:

```bash
uv run python run_set_membership.py
uv run python run_heavy_hitters.py
uv run python run_packet_switching.py
```

### Quick Test

```bash
uv run pytest tests
```

This runs the baseline implementation against all distributions and compares:
- False positive rates
- Memory usage
- Query latency
- Build time
