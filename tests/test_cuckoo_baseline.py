"""Test if a simple cuckoo filter implementation actually scores better than Bloom."""

import hashlib
from randomize_evolve.evaluators.bloom_alternatives import Evaluator, EvaluatorConfig


class SimpleCuckooFilter:
    """Minimal cuckoo filter implementation for testing."""

    def __init__(self, key_bits: int, capacity: int):
        self.key_bits = key_bits
        self.capacity = capacity
        # 4 slots per bucket, ~capacity/4 buckets
        self.num_buckets = max(16, capacity // 4)
        self.slots_per_bucket = 4
        self.buckets = [[None] * self.slots_per_bucket for _ in range(self.num_buckets)]
        self.overflow = set()  # Fallback for displaced items

    def _fingerprint(self, item: int) -> int:
        """Generate 16-bit fingerprint."""
        h = hashlib.sha256(item.to_bytes((self.key_bits + 7) // 8, "little")).digest()
        return int.from_bytes(h[:2], "little") | 1  # Ensure non-zero

    def _hash1(self, item: int) -> int:
        """Primary hash function."""
        h = hashlib.sha256(item.to_bytes((self.key_bits + 7) // 8, "little")).digest()
        return int.from_bytes(h[2:6], "little") % self.num_buckets

    def _hash2(self, h1: int, fingerprint: int) -> int:
        """Alternate position using fingerprint."""
        return (h1 ^ (fingerprint * 0x5BD1E995)) % self.num_buckets

    def add(self, item: int) -> None:
        fp = self._fingerprint(item)
        h1 = self._hash1(item)
        h2 = self._hash2(h1, fp)

        # Try to insert in either bucket
        for bucket_idx in [h1, h2]:
            bucket = self.buckets[bucket_idx]
            for i in range(self.slots_per_bucket):
                if bucket[i] is None:
                    bucket[i] = fp
                    return

        # No space - use overflow set to guarantee zero false negatives
        self.overflow.add(item)

    def query(self, item: int) -> bool:
        if item in self.overflow:
            return True

        fp = self._fingerprint(item)
        h1 = self._hash1(item)
        h2 = self._hash2(h1, fp)

        for bucket_idx in [h1, h2]:
            if fp in self.buckets[bucket_idx]:
                return True

        return False


def candidate_factory(key_bits: int, capacity: int):
    return SimpleCuckooFilter(key_bits, capacity)


def main():
    config = EvaluatorConfig(
        key_bits=32,
        positives=5000,
        queries=10000,
        negative_fraction=0.5,
        seeds=(17, 23, 71, 89, 131),
        build_timeout_s=1.0,
        query_timeout_s=1.0,
        false_negative_penalty=1_000_000.0,
        false_positive_weight=180.0,
        memory_weight=0.05,
        latency_weight=8.0,
    )

    evaluator = Evaluator(config)
    result = evaluator(candidate_factory)

    print("\n" + "=" * 60)
    print("SIMPLE CUCKOO FILTER BASELINE TEST")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"False Positive Rate: {result.false_positive_rate:.4%}")
    print(f"False Negative Rate: {result.false_negative_rate:.4%}")
    print(f"Memory: {result.mean_peak_memory_bytes:,.0f} bytes")
    print(f"Bits per item: {result.bits_per_item:.1f}")
    print(f"Build time: {result.mean_build_time_ms:.2f} ms")
    print(f"Query time: {result.mean_query_time_ms:.2f} ms")
    print(f"Raw score: {result.score:.2f}")
    print("=" * 60)

    # Calculate penalty multiplier for comparison
    bytes_per_item = result.mean_peak_memory_bytes / config.positives
    print(f"\nMemory penalty multiplier: {1.0 + bytes_per_item:.2f}x")
    print(f"Target: Get this below 1.75x for competitive cuckoo performance")


if __name__ == "__main__":
    main()
