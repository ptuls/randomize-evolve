# Prefix KV-Cache Best Program Baseline Comparison

Candidate: `src/randomize_evolve/problems/prefix_kv_cache/compact_seed.py`

Command:

```bash
.venv/bin/python -m randomize_evolve.problems.prefix_kv_cache.runner --baseline-report --candidate-program src/randomize_evolve/problems/prefix_kv_cache/compact_seed.py --config configs/prefix_kv_cache.yaml
```

## Headline

The candidate clears the deployable credibility baselines in this capacity sweep. It trails `oracle_future_reuse`. It beats `future_reuse_heuristic`.

| Rank | Policy | Group | Combined score | Capacity 24 token hit | Capacity 48 token hit | Worst-quarter hit | Request p10 hit | Token-wtd admission waste | Admission token utility | Avoidable eviction | Priority-burst weighted hit | Priority-noise token hit | Churn per 1k |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `oracle_future_reuse` | reporting-only/future-knowledge | 87.777 | 0.638 | 0.702 | 0.527 | 0.318 | 0.095 | 12.454 | 0.000 | 0.780 | 0.602 | 207.5 |
| 2 | `candidate` | deployable | 57.788 | 0.628 | 0.664 | 0.499 | 0.318 | 0.491 | 7.143 | 0.110 | 0.760 | 0.564 | 685.5 |
| 3 | `tinylfu_lru` | deployable | 53.426 | 0.580 | 0.635 | 0.434 | 0.294 | 0.422 | 7.183 | 0.234 | 0.699 | 0.575 | 462.2 |
| 4 | `future_reuse_heuristic` | reporting-only/future-knowledge | 49.428 | 0.613 | 0.688 | 0.502 | 0.313 | 0.736 | 2.141 | 0.007 | 0.749 | 0.572 | 1565.1 |
| 5 | `depth_prefer_shallow` | deployable | 35.210 | 0.580 | 0.646 | 0.457 | 0.313 | 0.762 | 1.905 | 0.212 | 0.701 | 0.426 | 1846.5 |
| 6 | `prefix_anchor` | deployable | 33.602 | 0.577 | 0.657 | 0.474 | 0.293 | 0.744 | 1.961 | 0.164 | 0.719 | 0.516 | 1731.0 |
| 7 | `prefix_fanout` | deployable | 32.447 | 0.578 | 0.646 | 0.453 | 0.312 | 0.770 | 1.889 | 0.207 | 0.692 | 0.383 | 1874.4 |
| 8 | `cost_aware_lru` | deployable | 27.008 | 0.554 | 0.644 | 0.470 | 0.287 | 0.741 | 1.897 | 0.185 | 0.720 | 0.522 | 1766.3 |
| 9 | `lfu` | deployable | 26.655 | 0.580 | 0.665 | 0.489 | 0.300 | 0.786 | 1.987 | 0.139 | 0.747 | 0.549 | 1683.9 |
| 10 | `lru` | deployable | 25.337 | 0.554 | 0.645 | 0.470 | 0.289 | 0.752 | 1.911 | 0.181 | 0.721 | 0.559 | 1746.6 |
| 11 | `tenant_fair_lru` | deployable | 25.216 | 0.554 | 0.645 | 0.469 | 0.288 | 0.750 | 1.912 | 0.180 | 0.721 | 0.559 | 1748.4 |
| 12 | `recompute_greedy` | deployable | 12.724 | 0.556 | 0.641 | 0.433 | 0.280 | 0.774 | 1.776 | 0.242 | 0.690 | 0.326 | 1962.0 |
| 13 | `no_cache` | deployable | -50.445 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |

## Validation Workload Detail

| Policy | phase_shift_prompts token hit | multi_tenant_skew token hit | hotset_cold_scan token hit | concurrent_long_generation token hit | stochastic_serving_mix token hit | rolling_template_versions token hit | heavy_tailed_prefix_lengths token hit | priority_burst_recovery token hit | priority_one_off_noise token hit | tenant_phase_shift_cycles token hit | Validation block hit | Validation churn per 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `oracle_future_reuse` | 0.794 | 0.814 | 0.644 | 0.880 | 0.516 | 0.847 | 0.570 | 0.519 | 0.602 | 0.556 | 0.633 | 207.5 |
| `candidate` | 0.794 | 0.802 | 0.638 | 0.857 | 0.412 | 0.847 | 0.530 | 0.506 | 0.564 | 0.493 | 0.596 | 685.5 |
| `tinylfu_lru` | 0.759 | 0.745 | 0.602 | 0.821 | 0.407 | 0.814 | 0.479 | 0.463 | 0.575 | 0.516 | 0.575 | 462.2 |
| `future_reuse_heuristic` | 0.794 | 0.812 | 0.644 | 0.874 | 0.454 | 0.847 | 0.494 | 0.499 | 0.572 | 0.508 | 0.608 | 1565.1 |
| `depth_prefer_shallow` | 0.794 | 0.802 | 0.495 | 0.871 | 0.389 | 0.847 | 0.463 | 0.466 | 0.426 | 0.474 | 0.553 | 1846.5 |
| `prefix_anchor` | 0.794 | 0.762 | 0.620 | 0.871 | 0.401 | 0.835 | 0.455 | 0.479 | 0.516 | 0.487 | 0.577 | 1731.0 |
| `prefix_fanout` | 0.794 | 0.802 | 0.499 | 0.871 | 0.430 | 0.847 | 0.462 | 0.459 | 0.383 | 0.432 | 0.548 | 1874.4 |
| `cost_aware_lru` | 0.794 | 0.721 | 0.620 | 0.871 | 0.407 | 0.815 | 0.428 | 0.480 | 0.522 | 0.473 | 0.571 | 1766.3 |
| `lfu` | 0.794 | 0.735 | 0.644 | 0.871 | 0.411 | 0.837 | 0.465 | 0.498 | 0.549 | 0.479 | 0.586 | 1683.9 |
| `lru` | 0.794 | 0.715 | 0.620 | 0.871 | 0.386 | 0.814 | 0.436 | 0.481 | 0.559 | 0.483 | 0.575 | 1746.6 |
| `tenant_fair_lru` | 0.794 | 0.719 | 0.620 | 0.871 | 0.385 | 0.814 | 0.435 | 0.481 | 0.559 | 0.479 | 0.575 | 1748.4 |
| `recompute_greedy` | 0.762 | 0.732 | 0.574 | 0.874 | 0.422 | 0.818 | 0.425 | 0.458 | 0.326 | 0.411 | 0.534 | 1962.0 |
| `no_cache` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |

## Held-Out Structure-Generalization Probe

These recurrence-heavy families are evaluated and reported but excluded from the candidate-selection combined score.

| Policy | agent_trace_branching token hit | cyclic_working_set_pressure token hit | Probe block hit | Probe churn per 1k |
|---|---:|---:|---:|---:|
| `oracle_future_reuse` | 0.250 | 0.805 | 0.517 | 2283.9 |
| `candidate` | 0.227 | 0.782 | 0.468 | 170.1 |
| `tinylfu_lru` | 0.240 | 0.590 | 0.396 | 2670.1 |
| `future_reuse_heuristic` | 0.245 | 0.800 | 0.511 | 2546.9 |
| `depth_prefer_shallow` | 0.243 | 0.786 | 0.486 | 2705.7 |
| `prefix_anchor` | 0.244 | 0.670 | 0.434 | 2959.2 |
| `prefix_fanout` | 0.244 | 0.786 | 0.486 | 2695.3 |
| `cost_aware_lru` | 0.243 | 0.602 | 0.403 | 3123.3 |
| `lfu` | 0.244 | 0.661 | 0.436 | 2940.1 |
| `lru` | 0.243 | 0.590 | 0.398 | 3147.6 |
| `tenant_fair_lru` | 0.243 | 0.590 | 0.398 | 3147.6 |
| `recompute_greedy` | 0.244 | 0.781 | 0.496 | 2643.2 |
| `no_cache` | 0.000 | 0.000 | 0.000 | 0.0 |

## Notes

- Candidate `scoring_fn_complexity` in this report is `473`; the combined score includes that penalty.
- Candidate score breakdown: mean workload `68.642`, minimum-workload contribution `13.685`, churn cost `10.283`, fairness cost `7.663`, and complexity cost `6.593`.
- `future_reuse_heuristic` and `oracle_future_reuse` use simulator-provided future knowledge and are not deployable. The former is count-weighted; the latter is a Belady-style next-use oracle constrained by the simulator's leaf-only eviction model.
- `tinylfu_lru` admits only shallow or repeated blocks, so it often trades lower hit rate for lower churn.
- `prefix_anchor` is a deployable structural anchor baseline; `prefix_fanout` is a simpler descendant-count protection baseline.
- Priority-burst weighted hit is reported from `priority_burst_recovery`; priority-noise token hit checks the opposite failure mode, where high priority does not imply reuse.
- Request p10, worst-quarter hit, token-weighted admission waste, admission token utility, and avoidable eviction are aggregated across the validation panel.
- This report uses `request_count=96`, seeds `(11, 23, 37)`, and capacity sweep `(24, 48)`.
