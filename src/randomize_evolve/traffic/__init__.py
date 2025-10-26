"""Traffic generation utilities for packet switching simulations."""

from randomize_evolve.traffic.patterns import (
    TrafficPattern,
    TrafficPatternConfig,
    TrafficPatternType,
    UniformPattern,
    BurstyPattern,
    HotspotPattern,
    HeavyLoadPattern,
    build_pattern,
)
from randomize_evolve.traffic.simulator import SimulationResult, SwitchTrafficSimulator

__all__ = [
    "TrafficPattern",
    "TrafficPatternConfig",
    "TrafficPatternType",
    "UniformPattern",
    "BurstyPattern",
    "HotspotPattern",
    "HeavyLoadPattern",
    "build_pattern",
    "SimulationResult",
    "SwitchTrafficSimulator",
]
