"""Test a memory-optimized cuckoo filter using bytearray for compact storage."""

import hashlib
from randomize_evolve.evaluators.bloom_alternatives import Evaluator, EvaluatorConfig


class CompactCuckooFilter:
    """Memory-optimized cuckoo filter using bytearray for packed storage."""

    def __init__(self, key_bits: int, capacity: int):
        self.key_bits = key_bits
        self.capacity = capacity

        # Use 2 bytes per slot for 16-bit fingerprints
        self.slots_per_bucket = 4
        self.num_buckets = max(16, capacity // 3)  # Slightly over-provision
        self.bytes_per_slot = 2

        # Packed storage: bytearray is 1 byte per entry (much better than list)
        total_bytes = self.num_buckets * self.slots_per_bucket * self.bytes_per_slot
        self.table = bytearray(total_bytes)

        # Small overflow for displaced items (guarantees zero FN)
        self.overflow = set()

    def _fingerprint(self, item: int) -> int:
        """Generate 16-bit fingerprint (non-zero)."""
        h = hashlib.sha256(item.to_bytes((self.key_bits + 7) // 8, "little")).digest()
        fp = int.from_bytes(h[:2], "little")
        return fp if fp != 0 else 1

    def _hash1(self, item: int) -> int:
        """Primary bucket index."""
        h = hashlib.sha256(item.to_bytes((self.key_bits + 7) // 8, "little")).digest()
        return int.from_bytes(h[2:6], "little") % self.num_buckets

    def _hash2(self, h1: int, fingerprint: int) -> int:
        """Alternate bucket using fingerprint."""
        return (h1 ^ (fingerprint * 0x5BD1E995)) % self.num_buckets

    def _get_slot(self, bucket_idx: int, slot_idx: int) -> int:
        """Read 16-bit fingerprint from packed storage."""
        offset = (bucket_idx * self.slots_per_bucket + slot_idx) * self.bytes_per_slot
        return int.from_bytes(self.table[offset : offset + 2], "little")

    def _set_slot(self, bucket_idx: int, slot_idx: int, fingerprint: int) -> None:
        """Write 16-bit fingerprint to packed storage."""
        offset = (bucket_idx * self.slots_per_bucket + slot_idx) * self.bytes_per_slot
        self.table[offset : offset + 2] = fingerprint.to_bytes(2, "little")

    def add(self, item: int) -> None:
        fp = self._fingerprint(item)
        h1 = self._hash1(item)
        h2 = self._hash2(h1, fp)

        # Try to insert in either bucket
        for bucket_idx in [h1, h2]:
            for slot_idx in range(self.slots_per_bucket):
                if self._get_slot(bucket_idx, slot_idx) == 0:  # Empty slot
                    self._set_slot(bucket_idx, slot_idx, fp)
                    return

        # No space - use overflow to guarantee zero false negatives
        self.overflow.add(item)

    def query(self, item: int) -> bool:
        if item in self.overflow:
            return True

        fp = self._fingerprint(item)
        h1 = self._hash1(item)
        h2 = self._hash2(h1, fp)

        # Check both buckets
        for bucket_idx in [h1, h2]:
            for slot_idx in range(self.slots_per_bucket):
                if self._get_slot(bucket_idx, slot_idx) == fp:
                    return True

        return False


def candidate_factory(key_bits: int, capacity: int):
    return CompactCuckooFilter(key_bits, capacity)


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
    print("COMPACT CUCKOO FILTER TEST (bytearray storage)")
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

    # Compare to theoretical limits
    bytes_per_item = result.mean_peak_memory_bytes / config.positives
    print(f"\nMemory penalty multiplier: {1.0 + bytes_per_item:.2f}x")
    print(f"Theoretical cuckoo: 0.5-0.75 bytes/item = 1.5x-1.75x multiplier")
    print(f"Achieved: {bytes_per_item:.2f} bytes/item")

    # Calculate overhead
    table_size = (config.positives // 3) * 4 * 2  # buckets * slots * 2 bytes
    print(f"\nTable size: {table_size:,} bytes ({table_size/config.positives:.2f} bytes/item)")
    print(f"Python overhead: {result.mean_peak_memory_bytes - table_size:,} bytes")


if __name__ == "__main__":
    main()
