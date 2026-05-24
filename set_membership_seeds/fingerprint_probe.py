"""Compact fingerprint-table seed for set-membership exploration."""

from math import ceil

from loguru import logger


class FingerprintProbeFilter:
    """Approximate set with open-addressed fingerprint storage."""

    def __init__(
        self,
        capacity: int,
        key_bits: int,
        *,
        load_factor: float = 0.72,
        fingerprint_bits: int = 8,
    ) -> None:
        self.capacity = max(1, capacity)
        self.key_bits = key_bits
        self.load_factor = min(max(load_factor, 0.4), 0.9)
        self.fingerprint_mask = (1 << max(4, fingerprint_bits)) - 1

        min_slots = max(16, ceil(self.capacity / self.load_factor))
        slot_count = 1
        while slot_count < min_slots:
            slot_count <<= 1

        self.slot_mask = slot_count - 1
        self.slots = bytearray(slot_count)
        self.seed = 0x9E3779B185EBCA87

    # EVOLVE-BLOCK-START
    def add(self, item: int) -> None:
        fingerprint = self._fingerprint(item)
        index = self._index(item)
        for _ in range(len(self.slots)):
            value = self.slots[index]
            if value in (0, fingerprint):
                self.slots[index] = fingerprint
                return
            index = (index + 1) & self.slot_mask

    def query(self, item: int) -> bool:
        fingerprint = self._fingerprint(item)
        index = self._index(item)
        for _ in range(len(self.slots)):
            value = self.slots[index]
            if value == 0:
                return False
            if value == fingerprint:
                return True
            index = (index + 1) & self.slot_mask
        return False

    # EVOLVE-BLOCK-END

    def _index(self, item: int) -> int:
        return self._mix64((item ^ self.seed) & ((1 << 64) - 1)) & self.slot_mask

    def _fingerprint(self, item: int) -> int:
        fingerprint = self._mix64((item + self.seed) & ((1 << 64) - 1))
        fingerprint &= self.fingerprint_mask
        return fingerprint or 1

    def _mix64(self, value: int) -> int:
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return value ^ (value >> 31)


def candidate_factory(key_bits: int, capacity: int) -> FingerprintProbeFilter:
    """Factory function required by the evaluator."""
    return FingerprintProbeFilter(capacity=capacity, key_bits=key_bits)


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
            len(candidate.slots),
        )
    )


if __name__ == "__main__":
    run_demo()
