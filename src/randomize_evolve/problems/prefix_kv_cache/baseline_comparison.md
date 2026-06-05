# Prefix KV-Cache Best Program Baseline Comparison

Candidate: `src/randomize_evolve/problems/prefix_kv_cache/compact_seed.py`

Command:

```bash
.venv/bin/python -m randomize_evolve.problems.prefix_kv_cache.runner --baseline-report --candidate-program src/randomize_evolve/problems/prefix_kv_cache/compact_seed.py
```

## Headline

The candidate clears the deployable credibility baselines in this capacity sweep. It trails `oracle_future_reuse`. It beats `future_reuse_heuristic`.

| Rank | Policy | Group | Combined score | Capacity 24 token hit | Capacity 48 token hit | Worst-quarter hit | Request p10 hit | Token-wtd admission waste | Admission token utility | Avoidable eviction | Priority-burst weighted hit | Priority-noise token hit | Churn per 1k |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `oracle_future_reuse` | reporting-only/future-knowledge | 54.554 | 0.639 | 0.704 | 0.514 | 0.306 | 0.139 | 10.917 | 0.000 | 0.780 | 0.602 | 578.1 |
| 2 | `candidate` | deployable | 44.681 | 0.631 | 0.668 | 0.495 | 0.315 | 0.442 | 8.823 | 0.129 | 0.758 | 0.568 | 578.7 |
| 3 | `future_reuse_heuristic` | reporting-only/future-knowledge | 29.501 | 0.614 | 0.690 | 0.495 | 0.300 | 0.677 | 2.298 | 0.010 | 0.741 | 0.573 | 1731.3 |
| 4 | `tinylfu_lru` | deployable | 27.138 | 0.580 | 0.638 | 0.425 | 0.284 | 0.452 | 6.346 | 0.276 | 0.698 | 0.575 | 839.0 |
| 5 | `depth_prefer_shallow` | deployable | 21.318 | 0.581 | 0.648 | 0.464 | 0.301 | 0.713 | 2.018 | 0.246 | 0.692 | 0.425 | 1992.2 |
| 6 | `prefix_fanout` | deployable | 20.807 | 0.579 | 0.646 | 0.461 | 0.301 | 0.721 | 2.003 | 0.242 | 0.684 | 0.386 | 2021.7 |
| 7 | `prefix_anchor` | deployable | 15.492 | 0.577 | 0.658 | 0.466 | 0.287 | 0.719 | 1.967 | 0.218 | 0.717 | 0.516 | 1944.5 |
| 8 | `cost_aware_lru` | deployable | 8.343 | 0.555 | 0.644 | 0.454 | 0.277 | 0.726 | 1.875 | 0.239 | 0.720 | 0.522 | 2005.6 |
| 9 | `lfu` | deployable | 7.883 | 0.582 | 0.666 | 0.478 | 0.293 | 0.757 | 2.023 | 0.196 | 0.741 | 0.556 | 1900.8 |
| 10 | `lru` | deployable | 6.933 | 0.555 | 0.646 | 0.452 | 0.277 | 0.735 | 1.878 | 0.236 | 0.721 | 0.559 | 1990.0 |
| 11 | `tenant_fair_lru` | deployable | 6.930 | 0.554 | 0.646 | 0.452 | 0.277 | 0.734 | 1.878 | 0.236 | 0.721 | 0.559 | 1991.4 |
| 12 | `recompute_greedy` | deployable | 4.024 | 0.554 | 0.640 | 0.442 | 0.273 | 0.726 | 1.951 | 0.267 | 0.690 | 0.321 | 2100.1 |
| 13 | `no_cache` | deployable | -50.824 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |

## Validation Workload Detail

| Policy | agent_trace_branching token hit | phase_shift_prompts token hit | multi_tenant_skew token hit | hotset_cold_scan token hit | cyclic_working_set_pressure token hit | concurrent_long_generation token hit | stochastic_serving_mix token hit | rolling_template_versions token hit | heavy_tailed_prefix_lengths token hit | priority_burst_recovery token hit | priority_one_off_noise token hit | tenant_phase_shift_cycles token hit | Validation block hit | Validation churn per 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `oracle_future_reuse` | 0.250 | 0.794 | 0.814 | 0.644 | 0.806 | 0.878 | 0.573 | 0.847 | 0.545 | 0.519 | 0.602 | 0.556 | 0.616 | 578.1 |
| `candidate` | 0.227 | 0.794 | 0.802 | 0.638 | 0.784 | 0.857 | 0.482 | 0.847 | 0.508 | 0.505 | 0.568 | 0.493 | 0.578 | 578.7 |
| `future_reuse_heuristic` | 0.245 | 0.794 | 0.812 | 0.644 | 0.799 | 0.872 | 0.516 | 0.847 | 0.459 | 0.494 | 0.573 | 0.510 | 0.593 | 1731.3 |
| `tinylfu_lru` | 0.240 | 0.759 | 0.745 | 0.602 | 0.590 | 0.821 | 0.477 | 0.814 | 0.441 | 0.462 | 0.575 | 0.516 | 0.548 | 839.0 |
| `depth_prefer_shallow` | 0.243 | 0.794 | 0.802 | 0.495 | 0.788 | 0.869 | 0.453 | 0.847 | 0.428 | 0.461 | 0.425 | 0.476 | 0.544 | 1992.2 |
| `prefix_fanout` | 0.244 | 0.794 | 0.802 | 0.499 | 0.788 | 0.869 | 0.480 | 0.847 | 0.421 | 0.453 | 0.386 | 0.437 | 0.538 | 2021.7 |
| `prefix_anchor` | 0.244 | 0.794 | 0.762 | 0.620 | 0.670 | 0.869 | 0.454 | 0.835 | 0.415 | 0.478 | 0.516 | 0.490 | 0.554 | 1944.5 |
| `cost_aware_lru` | 0.243 | 0.794 | 0.721 | 0.620 | 0.602 | 0.869 | 0.450 | 0.815 | 0.387 | 0.479 | 0.522 | 0.476 | 0.543 | 2005.6 |
| `lfu` | 0.244 | 0.794 | 0.735 | 0.644 | 0.661 | 0.869 | 0.467 | 0.837 | 0.418 | 0.494 | 0.556 | 0.487 | 0.562 | 1900.8 |
| `lru` | 0.243 | 0.794 | 0.715 | 0.620 | 0.590 | 0.869 | 0.438 | 0.814 | 0.394 | 0.481 | 0.559 | 0.484 | 0.546 | 1990.0 |
| `tenant_fair_lru` | 0.243 | 0.794 | 0.719 | 0.620 | 0.590 | 0.869 | 0.437 | 0.814 | 0.393 | 0.481 | 0.559 | 0.481 | 0.546 | 1991.4 |
| `recompute_greedy` | 0.244 | 0.764 | 0.731 | 0.574 | 0.783 | 0.872 | 0.465 | 0.817 | 0.373 | 0.458 | 0.321 | 0.404 | 0.526 | 2100.1 |
| `no_cache` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |

## Notes

- Candidate `scoring_fn_complexity` in this report is `473`; the combined score includes that penalty.
- Candidate score breakdown: mean workload `65.497`, minimum-workload contribution `2.120`, churn cost `8.681`, fairness cost `7.663`, and complexity cost `6.593`.
- `future_reuse_heuristic` and `oracle_future_reuse` use simulator-provided future knowledge and are not deployable. The former is count-weighted; the latter is a Belady-style next-use oracle constrained by the simulator's leaf-only eviction model.
- `tinylfu_lru` admits only shallow or repeated blocks, so it often trades lower hit rate for lower churn.
- `prefix_anchor` is a deployable structural anchor baseline; `prefix_fanout` is a simpler descendant-count protection baseline.
- Priority-burst weighted hit is reported from `priority_burst_recovery`; priority-noise token hit checks the opposite failure mode, where high priority does not imply reuse.
- Request p10, worst-quarter hit, token-weighted admission waste, admission token utility, and avoidable eviction are aggregated across the validation panel.
- This report uses `request_count=96`, seeds `(11, 23, 37)`, and capacity sweep `(24, 48)`.
