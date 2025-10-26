"""Traffic pattern definitions for packet switching simulations."""

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Protocol


class TrafficPatternType(Enum):
    """Enumerates the built-in traffic patterns."""

    UNIFORM = auto()
    BURSTY = auto()
    HOTSPOT = auto()
    HEAVY_LOAD = auto()


class TrafficPattern(Protocol):
    """Protocol implemented by traffic generators."""

    def sample(
        self,
        rng,
        time_slot: int,
        num_inputs: int,
        num_outputs: int,
    ) -> List[List[int]]:
        """Return the list of destination outputs for each input in a time slot."""


@dataclass
class TrafficPatternConfig:
    """Configuration shared by the pattern helpers."""

    pattern_type: TrafficPatternType
    offered_load: float = 0.5
    burst_rate: float = 4.0
    burst_length: int = 8
    burst_probability: float = 0.05
    hotspot_probability: float = 0.4
    hotspot_output: int | None = None
    heavy_load: float = 0.95
    light_load: float = 0.4
    heavy_duration: int = 50
    light_duration: int = 50


class _BasePattern:
    def __init__(self, cfg: TrafficPatternConfig):
        self.cfg = cfg

    @staticmethod
    def _poisson_sample(rng, lam: float) -> int:
        """Sample from a Poisson distribution using Knuth's algorithm."""

        if lam <= 0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= rng.random()
        return max(0, k - 1)


class UniformPattern(_BasePattern):
    """Generates independent uniform traffic for each input."""

    def sample(self, rng, time_slot: int, num_inputs: int, num_outputs: int) -> List[List[int]]:
        destinations: List[List[int]] = []
        lam = self.cfg.offered_load
        for _ in range(num_inputs):
            packets = self._poisson_sample(rng, lam)
            destinations.append([rng.randrange(num_outputs) for _ in range(packets)])
        return destinations


class BurstyPattern(_BasePattern):
    """Models inputs that occasionally generate bursts of traffic."""

    def __init__(self, cfg: TrafficPatternConfig):
        super().__init__(cfg)
        self._burst_counters: Dict[int, int] = {}

    def sample(self, rng, time_slot: int, num_inputs: int, num_outputs: int) -> List[List[int]]:
        lam = self.cfg.offered_load
        burst_rate = max(1, int(round(self.cfg.burst_rate)))
        result: List[List[int]] = []
        for i in range(num_inputs):
            remaining = self._burst_counters.get(i, 0)
            outputs: List[int] = []
            if remaining > 0:
                outputs = [rng.randrange(num_outputs) for _ in range(burst_rate)]
                self._burst_counters[i] = remaining - 1
            else:
                if rng.random() < self.cfg.burst_probability:
                    self._burst_counters[i] = max(0, self.cfg.burst_length - 1)
                    outputs = [rng.randrange(num_outputs) for _ in range(burst_rate)]
                else:
                    packets = self._poisson_sample(rng, lam)
                    outputs = [rng.randrange(num_outputs) for _ in range(packets)]
            result.append(outputs)
        return result


class HotspotPattern(_BasePattern):
    """Generates traffic with a preferred hotspot output."""

    def __init__(self, cfg: TrafficPatternConfig):
        super().__init__(cfg)
        self._hotspot_output = cfg.hotspot_output

    def sample(self, rng, time_slot: int, num_inputs: int, num_outputs: int) -> List[List[int]]:
        if self._hotspot_output is None:
            self._hotspot_output = rng.randrange(num_outputs)
        lam = self.cfg.offered_load
        hotspot = self._hotspot_output
        bias = min(max(self.cfg.hotspot_probability, 0.0), 1.0)
        destinations: List[List[int]] = []
        for _ in range(num_inputs):
            packets = self._poisson_sample(rng, lam)
            outputs: List[int] = []
            for _ in range(packets):
                if rng.random() < bias:
                    outputs.append(hotspot)
                else:
                    outputs.append(rng.randrange(num_outputs))
            destinations.append(outputs)
        return destinations


class HeavyLoadPattern(_BasePattern):
    """Alternates between heavy and light load phases."""

    def __init__(self, cfg: TrafficPatternConfig):
        super().__init__(cfg)
        self._phase_duration = cfg.heavy_duration
        self._light_duration = cfg.light_duration
        self._cycle_length = max(1, self._phase_duration + self._light_duration)

    def sample(self, rng, time_slot: int, num_inputs: int, num_outputs: int) -> List[List[int]]:
        phase_index = time_slot % self._cycle_length
        in_heavy_phase = phase_index < self._phase_duration
        lam = self.cfg.heavy_load if in_heavy_phase else self.cfg.light_load
        destinations: List[List[int]] = []
        for _ in range(num_inputs):
            packets = self._poisson_sample(rng, lam)
            destinations.append([rng.randrange(num_outputs) for _ in range(packets)])
        return destinations


_PATTERN_MAP: Dict[TrafficPatternType, type[_BasePattern]] = {
    TrafficPatternType.UNIFORM: UniformPattern,
    TrafficPatternType.BURSTY: BurstyPattern,
    TrafficPatternType.HOTSPOT: HotspotPattern,
    TrafficPatternType.HEAVY_LOAD: HeavyLoadPattern,
}


def build_pattern(cfg: TrafficPatternConfig) -> TrafficPattern:
    """Instantiate a traffic pattern implementation from configuration."""

    try:
        pattern_cls = _PATTERN_MAP[cfg.pattern_type]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported traffic pattern: {cfg.pattern_type}") from exc
    return pattern_cls(cfg)
