"""Baseline candidate factory for evolving Bloom filter alternatives.

The `candidate_factory` function below is the entry point OpenEvolve expects.
It returns an object implementing `add()` and `query()`; the implementation can
be evolved by the search algorithm. The scaffolding underneath provides a
plain Python Bloom filter that favours determinism and clarity over raw
performance.
"""

import hashlib
import math
from typing import Iterable, Tuple

from loguru import logger


class BaselineBloomFilter:
    """Reference Bloom filter using Blake2b-derived hashes."""

    def __init__(self, capacity: int, key_bits: int, *, bits_per_item: int = 10) -> None:
        self.capacity = max(1, capacity)
        self.key_bits = key_bits
        self.bits_per_item = max(4, bits_per_item)
        self.bit_count = max(self.capacity * self.bits_per_item, 64)
        self.byte_count = (self.bit_count + 7) // 8
        self.storage = bytearray(self.byte_count)
        self.hash_functions = self._default_hash_functions()

    # EVOLVE-BLOCK-START
    # The methods below are the primary targets for evolution. Swap in improved
    # hashing strategies, smarter bit layouts, or auxiliary data structures
    # that reduce false positives while staying within the evaluator budgets.

    def add(self, item: int) -> None:
        for index in self._indices(item):
            self._set_bit(index)

    def query(self, item: int) -> bool:
        return all(self._get_bit(index) for index in self._indices(item))

    # EVOLVE-BLOCK-END

    def _indices(self, item: int) -> Iterable[int]:
        for seed in self.hash_functions:
            digest = hashlib.blake2b(
                item.to_bytes(length=(self.key_bits + 7) // 8, byteorder="little"),
                digest_size=8,
                person=seed,
            ).digest()
            value = int.from_bytes(digest, byteorder="little")
            yield value % self.bit_count

    def _set_bit(self, index: int) -> None:
        byte_index = index // 8
        bit_mask = 1 << (index % 8)
        self.storage[byte_index] |= bit_mask

    def _get_bit(self, index: int) -> bool:
        byte_index = index // 8
        bit_mask = 1 << (index % 8)
        return bool(self.storage[byte_index] & bit_mask)

    def _default_hash_functions(self) -> Tuple[bytes, bytes, bytes, bytes]:
        # Four distinct personalization strings give independent-looking hashes.
        return (
            b"bloom-seed-00",
            b"bloom-seed-01",
            b"bloom-seed-02",
            b"bloom-seed-03",
        )


def candidate_factory(key_bits: int, capacity: int) -> BaselineBloomFilter:
    """Factory function required by the evaluator."""
    # Adjust bits per item based on expected false positive trade-off.
    bits_per_item = max(6, int(round(10.0 * math.log2(1 + capacity) / math.log2(2 + capacity))))
    return BaselineBloomFilter(capacity=capacity, key_bits=key_bits, bits_per_item=bits_per_item)


def run_demo() -> None:
    """Simple smoke test mirroring the evaluator contract."""
    bloom = candidate_factory(32, 5000)
    positives = list(range(1000))

    for value in positives:
        bloom.add(value)

    false_negatives = sum(1 for value in positives if not bloom.query(value))
    false_positives = sum(1 for value in range(1000, 2000) if bloom.query(value))

    logger.info(
        f"Inserted {len(positives)} items | FN={false_negatives} | FP={false_positives} "
        f"| bits={bloom.bit_count} | bytes={len(bloom.storage)}"
    )


if __name__ == "__main__":
    run_demo()
