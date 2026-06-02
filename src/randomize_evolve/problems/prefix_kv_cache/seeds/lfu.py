"""LFU prefix KV-cache seed."""

from randomize_evolve.evaluators.prefix_kv_cache import baseline_lfu_blocks

candidate_factory = baseline_lfu_blocks
build_candidate = baseline_lfu_blocks
