"""Seed policy for the cache eviction/admission search.

The framework owns the real resident set. This policy keeps only enough
metadata to decide whether to admit a miss and which key to evict.
"""

from collections import OrderedDict


class FrequencySketch:
    """Small aging Count-Min-style sketch for cache admission."""

    def __init__(self, width: int, depth: int = 4, sample_size: int = 10000) -> None:
        self.width = max(1, width)
        self.depth = max(1, depth)
        self.sample_size = max(1, sample_size)
        self.size = 0
        self.tables = [[0] * self.width for _ in range(self.depth)]

    def increment(self, key: int) -> None:
        for row in range(self.depth):
            index = self._hash(key, row)
            if self.tables[row][index] < 255:
                self.tables[row][index] += 1
        self.size += 1
        if self.size >= self.sample_size:
            self._age()

    def estimate(self, key: int) -> int:
        return min(self.tables[row][self._hash(key, row)] for row in range(self.depth))

    def metadata_bytes(self) -> int:
        return self.width * self.depth

    def _age(self) -> None:
        for row in self.tables:
            for index, value in enumerate(row):
                row[index] = value // 2
        self.size //= 2

    def _hash(self, key: int, row: int) -> int:
        hashed = (key ^ (row * 0x9E3779B97F4A7C15)) * 0xBF58476D1CE4E5B9
        return (hashed & ((1 << 64) - 1)) % self.width


class SketchSampledCachePolicy:
    """Frequency-aware admission with sampled eviction.

    Misses are admitted only when their sketch frequency can compete with a
    small deterministic sample of residents. The sketch ages over time, which
    lets the policy follow drift without keeping exact per-key history.
    """

    def __init__(self, key_bits: int, capacity: int) -> None:
        del key_bits
        self.capacity = max(1, capacity)
        self.resident = OrderedDict()
        self.sketch = FrequencySketch(
            width=max(64, self.capacity * 4),
            sample_size=max(1000, self.capacity * 20),
        )
        self.clock = 0
        self.pending = None
        self.operations = 0

    def on_access(self, key: int, hit: bool) -> None:
        self.clock += 1
        self.operations += 1
        self.sketch.increment(key)
        if hit and key in self.resident:
            self.resident.move_to_end(key)

    def should_admit(self, key: int) -> bool:
        if key in self.resident:
            return False
        if len(self.resident) < self.capacity:
            self.resident[key] = None
            return True

        victim = self._sampled_victim()
        self.operations += 4
        if self.sketch.estimate(key) < self.sketch.estimate(victim):
            return False

        self.pending = key
        self.resident[key] = None
        return True

    def pick_victim(self) -> int:
        victim = self._sampled_victim(exclude=self.pending)
        if victim is None:
            victim = self.pending
        if victim is None:
            raise RuntimeError("no resident keys to evict")
        self.resident.pop(victim, None)
        self.pending = None
        return victim

    def metadata_bytes(self) -> int:
        return self.sketch.metadata_bytes() + self.capacity * 24

    def operation_count(self) -> int:
        return self.operations

    def _sampled_victim(self, exclude=None):
        best_key = None
        best_score = None
        checked = 0
        for key in self.resident:
            if key == exclude:
                continue
            score = (self.sketch.estimate(key), checked)
            if best_score is None or score < best_score:
                best_key = key
                best_score = score
            checked += 1
            if checked >= 4:
                break
        return best_key


def candidate_factory(key_bits: int, capacity: int):
    return SketchSampledCachePolicy(key_bits, capacity)
