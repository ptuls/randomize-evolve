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
- `randomize_evolve.problems.prefix_kv_cache.evaluator`

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

## Prefix KV-cache evaluator

The prefix KV-cache evaluator models an LLM serving prefill cache, not decode
token management. It lives in
`src/randomize_evolve/evaluators/prefix_kv_cache.py` and exposes the
problem package `randomize_evolve.problems.prefix_kv_cache`.

Candidates define:

```python
def build_candidate(capacity_blocks: int, block_size_tokens: int, seed: int | None = None):
    ...
```

The returned policy implements scoring methods only:

```python
def on_request_start(request, now): ...
def score_admission(block, now) -> float: ...
def score_eviction(block, now) -> float: ...
def on_cache_hit(block, request, now): ...
def on_cache_miss(block, request, now): ...
```

Admission is sign-based: a newly computed block is admitted iff
`score_admission(block, now) > 0.0`. Eviction is simulator-enforced: while the
cache is over capacity, the simulator scores only inactive resident leaves and
evicts the highest-scoring block. Candidates never name a victim, so they cannot
evict active, non-resident, or interior blocks.

Prefix residency is root-anchored. A request only hits the largest contiguous
resident path from the root; a deeper block whose ancestors were evicted does
not count as reusable. Eviction is leaf-only, so removing a cold subtree happens
by peeling inactive leaves over successive eviction steps. If all blocks are
pinned and the cache cannot make room, the simulator bypasses the new block
without marking the candidate invalid.

Workloads include partial final blocks so token hit rate and block hit rate are
distinct signals. The recompute-cost feature is prefix-depth sensitive: the
estimated cost of recomputing a block grows with the prefix length attended
through that block, not only with the block's own token count. Generated
workloads do not expose raw prompt content to candidates; `prompt_tokens` is
kept empty on candidate-visible `RequestInfo` to avoid content fingerprinting.

Workloads cover `shared_system_prompt`, `rag_template_reuse`,
`agent_trace_branching`, `multi_tenant_skew`, `phase_shift_prompts`,
`long_context_mixed`, `session_continuation_growth`, `hotset_cold_scan`,
`concurrent_long_generation`, and `adversarial_unique_prompts`. The RAG
workload only credits prefix-aligned template and chunk reuse, because arbitrary
repeated chunks at different prompt positions are not reachable by a prefix
cache.

### Prompt workload families

`shared_system_prompt` models repeated chat or assistant requests that start
with the same system instructions, then branch into a small set of recurring
task prefixes and request-specific tails. It checks whether a policy keeps
shallow, broadly reused roots resident instead of spending capacity on one-off
suffixes.

`rag_template_reuse` models retrieval-augmented prompts with a shared
instruction/template prefix, followed by chunks that recur in the same prompt
position and then a query-specific tail. This intentionally avoids crediting
arbitrary repeated chunks at different positions, because a prefix cache cannot
reuse those as hits.

`long_context_mixed` models longer document-style contexts. Requests revisit
the same document roots with varying prefix lengths and partial final blocks.
It stresses depth-sensitive recompute cost and exposes policies that evict
expensive deeper context too casually.

`session_continuation_growth` models several interleaved conversations whose
prefixes gain one full turn on each revisit. It tests whether a policy preserves
deep reusable histories while sessions pause and resume.

`agent_trace_branching` models agent workflows that share an initial trace,
then branch through recurring tool or retry paths. It tests fanout behavior:
good policies should preserve shared trunks and useful branch points without
letting cold leaves dominate the cache.

`phase_shift_prompts` models a workload whose popular prompt family changes
mid-run. It checks whether a policy adapts after a phase shift instead of
protecting old prefixes indefinitely.

`multi_tenant_skew` models several tenants with uneven request volume and
tenant-specific prefix roots. It is the only default validation family that
feeds the tenant fairness penalty, so it catches policies that improve global
hit rate by starving smaller tenants.

`hotset_cold_scan` warms a small recurring prompt set, streams mostly one-off
prompts through the cache, then returns to the original hot set. It tests scan
resistance and recovery instead of measuring only steady-state hit rate.

`concurrent_long_generation` issues prompts with shared roots, rotating
branches, and long output lengths. Its overlapping active prefixes create
temporary admission pressure, exercising pinning and forced-bypass behavior.

`adversarial_unique_prompts` models mostly unique prompts with little to no
reuse. It is hidden by default and is mainly a churn/bypass stress test:
admit-everything policies should waste work here, while conservative admission
should avoid filling the cache with dead prefixes.

`cross_family_mixture` is a hidden mixture of shared, phase-shifted, and unique
requests. It is used only for final reporting and should not influence Levi
selection.

The default split is family hold-out: train uses shared system prompts, RAG
template reuse, long-context mixes, and growing session continuations;
validation uses agent branching, phase shifts, multi-tenant skew, cold scans,
and concurrent long generations; hidden uses adversarial and cross-family
mixtures. Levi-facing `evaluate`, `evaluate_factory`, and `evaluate_source`
return train and validation metrics only. Hidden is quarantined behind the
separate `evaluate_hidden(factory)` path for final champion reporting.

Reported metrics include token and block hit rates, saved and recomputed prefill
tokens, deterministic p50/p95/p99 latency proxy, evictions, admissions, churn,
forced bypasses, occupancy, tenant fairness gap, invalid reason, and scoring
formula complexity. Baselines include no-cache, LRU, LFU, depth-preferring,
recompute-cost greedy, prefix-fanout, tenant-fair LRU, and a future-reuse
heuristic for reporting only. The reporting suite also includes a Belady-style
next-use oracle. Neither future-knowledge baseline is deployable. The
count-weighted future-reuse heuristic is not an offline optimum or upper bound;
the next-use oracle is a constrained benchmark for the simulator's leaf-only
eviction model.

Quick starts:

```bash
uv run python -m randomize_evolve.problems.prefix_kv_cache.initial_program
uv run python -m randomize_evolve.problems.prefix_kv_cache.runner --quick --baseline-report
uv run python -m randomize_evolve.problems.prefix_kv_cache.runner --quick --plot-report
uv run python -m randomize_evolve.problems.prefix_kv_cache.runner --quick --hidden-report \
  --candidate-program artifacts/prefix_kv_cache_runs/<run-id>
uv run python -m randomize_evolve.problems.prefix_kv_cache.runner --quick --iterations 3
```

Evolution runs save `best_program.py`, `metrics.json`, `artifacts.json`,
`metadata.json`, and `run_summary.json` under
`artifacts/prefix_kv_cache_runs/<timestamp>/`. The file
`artifacts/prefix_kv_cache_runs/latest_run.txt` points at the most recent saved
run. Use `--artifact-output <dir>` to change the destination or
`--no-save-artifacts` to disable saving. Pass an evolved run directory or
candidate `.py` file to `--hidden-report --candidate-program`; without that
argument, the hidden report evaluates the initial seed for comparison.

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
