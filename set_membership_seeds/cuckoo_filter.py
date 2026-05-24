"""Cuckoo-filter baseline for set-membership exploration."""

from array import array
from math import ceil

from loguru import logger


class BaselineCuckooFilter:
    """Approximate set using compact bucketed fingerprints."""

    def __init__(
        self,
        capacity: int,
        key_bits: int,
        *,
        bucket_size: int = 4,
        fingerprint_bits: int = 12,
        max_kicks: int = 64,
    ) -> None:
        self.capacity = max(1, capacity)
        self.key_bits = key_bits
        self.bucket_size = max(2, bucket_size)
        self.fingerprint_mask = (1 << max(4, fingerprint_bits)) - 1
        self.max_kicks = max(8, max_kicks)

        min_buckets = max(
            16,
            ceil(self.capacity / (self.bucket_size * 0.9)),
        )
        bucket_count = 1
        while bucket_count < min_buckets:
            bucket_count <<= 1

        self.bucket_count = bucket_count
        self.bucket_mask = bucket_count - 1
        self.fingerprints = array("H", [0]) * (bucket_count * self.bucket_size)
        self.seed0 = 0x9E3779B185EBCA87
        self.seed1 = 0xC2B2AE3D27D4EB4F

    # EVOLVE-BLOCK-START
    def add(self, item: int) -> None:
        fingerprint = self._fingerprint(item)
        primary = self._index(item)
        alternate = self._alternate_index(primary, fingerprint)

        if self._contains(primary, fingerprint) or self._contains(alternate, fingerprint):
            return
        if self._insert(primary, fingerprint) or self._insert(alternate, fingerprint):
            return

        current_bucket = primary if (fingerprint & 1) == 0 else alternate
        current_fingerprint = fingerprint
        for kick in range(self.max_kicks):
            slot = self._bucket_offset(current_bucket) + (
                (current_fingerprint + kick) % self.bucket_size
            )
            self.fingerprints[slot], current_fingerprint = (
                current_fingerprint,
                self.fingerprints[slot],
            )
            current_bucket = self._alternate_index(current_bucket, current_fingerprint)
            if self._insert(current_bucket, current_fingerprint):
                return

    def query(self, item: int) -> bool:
        fingerprint = self._fingerprint(item)
        primary = self._index(item)
        alternate = self._alternate_index(primary, fingerprint)
        return self._contains(primary, fingerprint) or self._contains(alternate, fingerprint)

    # EVOLVE-BLOCK-END

    def _index(self, item: int) -> int:
        return self._mix64((item ^ self.seed0) & ((1 << 64) - 1)) & self.bucket_mask

    def _alternate_index(self, bucket: int, fingerprint: int) -> int:
        mixed = self._mix64((fingerprint ^ self.seed1) & ((1 << 64) - 1))
        return (bucket ^ mixed) & self.bucket_mask

    def _fingerprint(self, item: int) -> int:
        fingerprint = self._mix64((item + self.seed1) & ((1 << 64) - 1))
        fingerprint &= self.fingerprint_mask
        return fingerprint or 1

    def _bucket_offset(self, bucket: int) -> int:
        return bucket * self.bucket_size

    def _contains(self, bucket: int, fingerprint: int) -> bool:
        offset = self._bucket_offset(bucket)
        for slot in range(offset, offset + self.bucket_size):
            if self.fingerprints[slot] == fingerprint:
                return True
        return False

    def _insert(self, bucket: int, fingerprint: int) -> bool:
        offset = self._bucket_offset(bucket)
        for slot in range(offset, offset + self.bucket_size):
            if self.fingerprints[slot] == 0:
                self.fingerprints[slot] = fingerprint
                return True
        return False

    def _mix64(self, value: int) -> int:
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return value ^ (value >> 31)


def candidate_factory(key_bits: int, capacity: int) -> BaselineCuckooFilter:
    """Factory function required by the evaluator."""
    return BaselineCuckooFilter(capacity=capacity, key_bits=key_bits)


def run_demo() -> None:
    """Simple smoke test mirroring the evaluator contract."""
    candidate = candidate_factory(32, 5000)
    positives = list(range(1000))

    for value in positives:
        candidate.add(value)

    false_negatives = sum(1 for value in positives if not candidate.query(value))
    false_positives = sum(1 for value in range(1000, 2000) if candidate.query(value))

    logger.info(
        "Inserted {} items | FN={} | FP={} | slots={}".format(
            len(positives),
            false_negatives,
            false_positives,
            len(candidate.fingerprints),
        )
    )


if __name__ == "__main__":
    run_demo()
