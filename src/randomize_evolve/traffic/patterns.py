"""Traffic pattern definitions for packet switching simulations."""

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Protocol, Sequence


class TrafficPatternType(Enum):
    """Enumerates the built-in traffic patterns."""

    UNIFORM = auto()
    BURSTY = auto()
    HOTSPOT = auto()
    HEAVY_LOAD = auto()
    ARRIVAL_MATRIX = auto()


class TrafficPattern(Protocol):
    """Protocol implemented by traffic generators."""

    def sample(
        self,
        rng,
        time_slot: int,
        num_inputs: int,
        num_outputs: int,
    ) -> List[List[int]]:
        """Samples packet destinations for a single time slot.

        Args:
            rng: Random number generator used to sample arrivals.
            time_slot: Current simulation slot.
            num_inputs: Number of switch input ports.
            num_outputs: Number of switch output ports.

        Returns:
            A list of packet destinations for each input port.
        """


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
    arrival_matrix: Sequence[Sequence[float]] | None = None


class _BasePattern:
    def __init__(self, cfg: TrafficPatternConfig):
        self.cfg = cfg

    def reset(self) -> None:
        """Reset any internal pattern state before a fresh simulation run."""

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

    def sample(
        self,
        rng,
        time_slot: int,
        num_inputs: int,
        num_outputs: int,
    ) -> List[List[int]]:
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

    def reset(self) -> None:
        """Clear burst state from prior simulation runs."""

        self._burst_counters = {}

    def sample(
        self,
        rng,
        time_slot: int,
        num_inputs: int,
        num_outputs: int,
    ) -> List[List[int]]:
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

    def reset(self) -> None:
        """Restore the configured hotspot selection state."""

        self._hotspot_output = self.cfg.hotspot_output

    def sample(
        self,
        rng,
        time_slot: int,
        num_inputs: int,
        num_outputs: int,
    ) -> List[List[int]]:
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
                    if num_outputs == 1:
                        outputs.append(hotspot)
                        continue
                    non_hotspot = rng.randrange(num_outputs - 1)
                    outputs.append(non_hotspot if non_hotspot < hotspot else non_hotspot + 1)
            destinations.append(outputs)
        return destinations


class HeavyLoadPattern(_BasePattern):
    """Alternates between heavy and light load phases."""

    def __init__(self, cfg: TrafficPatternConfig):
        super().__init__(cfg)
        self._phase_duration = cfg.heavy_duration
        self._light_duration = cfg.light_duration
        self._cycle_length = max(1, self._phase_duration + self._light_duration)

    def sample(
        self,
        rng,
        time_slot: int,
        num_inputs: int,
        num_outputs: int,
    ) -> List[List[int]]:
        phase_index = time_slot % self._cycle_length
        in_heavy_phase = phase_index < self._phase_duration
        lam = self.cfg.heavy_load if in_heavy_phase else self.cfg.light_load
        destinations: List[List[int]] = []
        for _ in range(num_inputs):
            packets = self._poisson_sample(rng, lam)
            destinations.append([rng.randrange(num_outputs) for _ in range(packets)])
        return destinations


class ArrivalMatrixPattern(_BasePattern):
    """Generates independent Poisson arrivals from a rate matrix."""

    def _rate_matrix(
        self,
        num_inputs: int,
        num_outputs: int,
    ) -> Sequence[Sequence[float]]:
        matrix = self.cfg.arrival_matrix
        if matrix is None:
            raise ValueError("arrival_matrix must be provided for ARRIVAL_MATRIX")
        if len(matrix) != num_inputs:
            raise ValueError("arrival_matrix must have one row per input")
        for row in matrix:
            if len(row) != num_outputs:
                raise ValueError("arrival_matrix must have one column per output")
            for rate in row:
                if rate < 0:
                    raise ValueError("arrival_matrix rates must be non-negative")
        return matrix

    def sample(
        self,
        rng,
        time_slot: int,
        num_inputs: int,
        num_outputs: int,
    ) -> List[List[int]]:
        del time_slot
        matrix = self._rate_matrix(num_inputs, num_outputs)
        destinations: List[List[int]] = []
        for input_idx, row in enumerate(matrix):
            del input_idx
            outputs: List[int] = []
            for output_idx, rate in enumerate(row):
                packet_count = self._poisson_sample(rng, rate)
                outputs.extend([output_idx] * packet_count)
            destinations.append(outputs)
        return destinations


_PATTERN_MAP: Dict[TrafficPatternType, type[_BasePattern]] = {
    TrafficPatternType.UNIFORM: UniformPattern,
    TrafficPatternType.BURSTY: BurstyPattern,
    TrafficPatternType.HOTSPOT: HotspotPattern,
    TrafficPatternType.HEAVY_LOAD: HeavyLoadPattern,
    TrafficPatternType.ARRIVAL_MATRIX: ArrivalMatrixPattern,
}


def build_pattern(cfg: TrafficPatternConfig) -> TrafficPattern:
    """Builds a traffic pattern from configuration.

    Args:
        cfg: Traffic pattern configuration.

    Returns:
        An initialized traffic pattern implementation.

    Raises:
        ValueError: The pattern type is unsupported.
    """

    try:
        pattern_cls = _PATTERN_MAP[cfg.pattern_type]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported traffic pattern: {cfg.pattern_type}") from exc
    return pattern_cls(cfg)
