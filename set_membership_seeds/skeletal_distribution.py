"""Skeletal seed for distribution-specific set-membership searches.

This program intentionally avoids a complete Bloom-filter implementation. It
only gives OpenEvolve the evaluator contract plus a weak occupancy sketch, so
distribution-specific runs must invent most of the useful structure themselves.
"""


class SkeletalMembershipSketch:
    """Minimal one-probe membership sketch used as a weak starting point."""

    def __init__(self, capacity: int, key_bits: int) -> None:
        self.capacity = max(1, capacity)
        self.key_bits = key_bits
        self.bits_per_item = 3
        self.bit_count = max(64, self.capacity * self.bits_per_item)
        self.storage = bytearray((self.bit_count + 7) // 8)
        self.mask64 = (1 << 64) - 1

    # EVOLVE-BLOCK-START
    # The scaffold below is deliberately thin: it stores a single mixed bit per
    # item and exposes helpers that are easy to replace with bucketed, tiered,
    # range-aware, or frequency-aware structures for non-uniform workloads.

    def add(self, item: int) -> None:
        index = self._index(item)
        self._set_bit(index)

    def query(self, item: int) -> bool:
        index = self._index(item)
        return self._get_bit(index)

    def _index(self, item: int) -> int:
        return self._mix(item) % self.bit_count

    def _mix(self, value: int) -> int:
        value &= self.mask64
        value ^= value >> 33
        value = (value * 0xFF51AFD7ED558CCD) & self.mask64
        value ^= value >> 33
        value = (value * 0xC4CEB9FE1A85EC53) & self.mask64
        value ^= value >> 33
        return value

    # EVOLVE-BLOCK-END

    def _set_bit(self, index: int) -> None:
        self.storage[index >> 3] |= 1 << (index & 7)

    def _get_bit(self, index: int) -> bool:
        return bool(self.storage[index >> 3] & (1 << (index & 7)))


def candidate_factory(key_bits: int, capacity: int) -> SkeletalMembershipSketch:
    """Factory function required by the evaluator."""
    return SkeletalMembershipSketch(capacity=capacity, key_bits=key_bits)
