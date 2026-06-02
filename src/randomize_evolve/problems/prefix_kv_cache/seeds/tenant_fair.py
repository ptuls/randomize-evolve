"""Tenant-fair LRU prefix KV-cache seed."""

from randomize_evolve.evaluators.prefix_kv_cache import baseline_tenant_fair_lru

candidate_factory = baseline_tenant_fair_lru
build_candidate = baseline_tenant_fair_lru
