"""Baseline heavy hitter candidate factory for OpenEvolve runs."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from loguru import logger


class CountMinSketchHeavyHitters:
    """Simple Count-Min Sketch with a bounded exact cache for heavy keys."""

    def __init__(
        self,
        key_bits: int,
        capacity: int,
        *,
        depth: int = 4,
        width: int = 1024,
        heavy_store_factor: float = 4.0,
    ) -> None:
        if depth <= 0 or width <= 0:
            raise ValueError("depth and width must be positive")
        if heavy_store_factor <= 1.0:
            raise ValueError("heavy_store_factor must exceed 1.0")

        self._mask = (1 << key_bits) - 1
        self._depth = depth
        self._width = width
        self._tables = [[0] * width for _ in range(depth)]
        self._heavy_capacity = max(capacity, int(capacity * heavy_store_factor))
        self._heavy_counts: Dict[int, int] = {}

    # EVOLVE-BLOCK-START
    # The methods below define the heavy-hitter contract. OpenEvolve can mutate
    # these implementations to explore alternative sketches, hierarchical
    # counters, or hybrid exact/approximate data structures.

    def observe(self, item: int, weight: int = 1) -> None:
        if weight <= 0:
            return
        weight = int(weight)
        value = self._normalize_item(item)
        for row in range(self._depth):
            index = self._hash(value, row)
            self._tables[row][index] += weight

        self._heavy_counts[value] = self._heavy_counts.get(value, 0) + weight
        if len(self._heavy_counts) > self._heavy_capacity:
            lightest_key = min(self._heavy_counts, key=self._heavy_counts.get)
            self._heavy_counts.pop(lightest_key, None)

    def estimate(self, item: int) -> int:
        value = self._normalize_item(item)
        estimates = [self._tables[row][self._hash(value, row)] for row in range(self._depth)]
        return min(estimates)

    def top_k(self, k: int) -> List[Tuple[int, int]]:
        items = list(self._heavy_counts.items())
        items.sort(key=lambda kv: (kv[1], kv[0]), reverse=True)
        return items[:k]

    # EVOLVE-BLOCK-END

    def _hash(self, value: int, salt_idx: int) -> int:
        salt = salt_idx * 0x9E3779B97F4A7C15
        hashed = (value ^ salt) * 0x9E3779B185EBCA87
        return hashed & ((1 << 64) - 1) % self._width

    def _normalize_item(self, item: int) -> int:
        if self._mask:
            return item & self._mask
        return item


def candidate_factory(key_bits: int, capacity: int) -> CountMinSketchHeavyHitters:
    depth = max(3, int(round(math.log2(max(4, capacity)))))
    width = max(256, capacity * 8)
    return CountMinSketchHeavyHitters(
        key_bits=key_bits,
        capacity=capacity,
        depth=depth,
        width=width,
        heavy_store_factor=4.0,
    )


def run_demo() -> None:
    sketch = candidate_factory(18, 48)
    heavy_items = list(range(5))
    for epoch in range(20):
        for value in heavy_items:
            sketch.observe(value, 5 - (value % 3))
        sketch.observe(1000 + epoch, 1)

    logger.info("Top-5 heavy hitters: {}", sketch.top_k(5))
    for probe in [0, 1, 2, 1001]:
        logger.info("estimate({}) -> {}", probe, sketch.estimate(probe))


if __name__ == "__main__":
    run_demo()
