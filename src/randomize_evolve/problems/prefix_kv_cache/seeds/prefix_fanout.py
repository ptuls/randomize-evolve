"""Prefix-fanout prefix KV-cache seed."""

from randomize_evolve.evaluators.prefix_kv_cache import baseline_prefix_fanout

candidate_factory = baseline_prefix_fanout
build_candidate = baseline_prefix_fanout

