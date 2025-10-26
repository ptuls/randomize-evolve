"""Core traffic simulator for packet switching evaluations."""

from collections import Counter, deque
from dataclasses import dataclass
from random import Random
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from randomize_evolve.packet_switching import SwitchScheduler
from randomize_evolve.traffic.patterns import TrafficPattern


@dataclass
class SimulationResult:
    """Aggregated metrics collected from a simulation run."""

    throughput: float
    fairness_inputs: float
    fairness_flows: float
    utilization: float
    drop_rate: float
    average_queue: float
    total_generated: int
    total_served: int
    total_dropped: int


class SwitchTrafficSimulator:
    """Discrete-time simulator for an input-queued packet switch."""

    def __init__(
        self,
        pattern: TrafficPattern,
        *,
        num_inputs: int,
        num_outputs: int,
        time_slots: int = 2000,
        warmup_slots: int = 200,
        queue_limit: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if num_inputs <= 0 or num_outputs <= 0:
            raise ValueError("The switch must have at least one input and output port.")
        if time_slots <= 0:
            raise ValueError("time_slots must be positive")
        if warmup_slots < 0:
            raise ValueError("warmup_slots must be non-negative")
        self.pattern = pattern
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.time_slots = time_slots
        self.warmup_slots = warmup_slots
        self.queue_limit = queue_limit
        self.seed = seed

    def run(self, scheduler: SwitchScheduler) -> SimulationResult:
        """Execute the simulation for the configured number of time slots."""

        rng = Random(self.seed)
        queues: List[Deque[Tuple[int, bool]]] = [deque() for _ in range(self.num_inputs)]
        total_generated = 0
        total_served = 0
        total_dropped = 0
        per_input_served = [0 for _ in range(self.num_inputs)]
        per_input_generated = [0 for _ in range(self.num_inputs)]
        flow_generated: Counter[tuple[int, int]] = Counter()
        flow_served: Counter[tuple[int, int]] = Counter()
        queue_length_sum = 0
        measured_slots = 0

        for slot in range(self.warmup_slots + self.time_slots):
            arrivals = self.pattern.sample(rng, slot, self.num_inputs, self.num_outputs)
            if len(arrivals) != self.num_inputs:
                raise ValueError("Traffic pattern produced invalid arrival vector")

            # Step 1: enqueue arrivals
            for input_idx, destinations in enumerate(arrivals):
                for output_idx in destinations:
                    if not 0 <= output_idx < self.num_outputs:
                        continue  # ignore malformed destinations
                    queue = queues[input_idx]
                    if self.queue_limit is not None and len(queue) >= self.queue_limit:
                        if slot >= self.warmup_slots:
                            total_dropped += 1
                        continue
                    active = slot >= self.warmup_slots
                    queue.append((output_idx, active))
                    if slot >= self.warmup_slots:
                        total_generated += 1
                        per_input_generated[input_idx] += 1
                        flow_generated[(input_idx, output_idx)] += 1

            # Step 2: compute requests for scheduling
            requests: Dict[int, List[int]] = {}
            for input_idx, queue in enumerate(queues):
                if not queue:
                    continue
                output_idx, _ = queue[0]
                requests.setdefault(output_idx, []).append(input_idx)

            queue_lengths_snapshot = [len(queue) for queue in queues]

            try:
                matches = scheduler.select_matches(requests, slot, queue_lengths_snapshot)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise RuntimeError("Scheduler failed to compute a matching") from exc

            used_outputs = set()
            for input_idx, output_idx in matches.items():
                if not 0 <= input_idx < self.num_inputs:
                    continue
                if not 0 <= output_idx < self.num_outputs:
                    continue
                if output_idx in used_outputs:
                    continue
                queue = queues[input_idx]
                if not queue:
                    continue
                head_output, is_active = queue[0]
                if head_output != output_idx:
                    continue
                queue.popleft()
                used_outputs.add(output_idx)
                if slot >= self.warmup_slots and is_active:
                    total_served += 1
                    per_input_served[input_idx] += 1
                    flow_served[(input_idx, output_idx)] += 1

            if slot >= self.warmup_slots:
                queue_length_sum += sum(queue_lengths_snapshot)
                measured_slots += 1

        throughput = total_served / total_generated if total_generated else 0.0
        utilization = total_served / (self.time_slots * self.num_outputs)
        drop_rate = total_dropped / total_generated if total_generated else 0.0
        average_queue = queue_length_sum / measured_slots if measured_slots else 0.0

        active_input_values = [per_input_served[i] for i, generated in enumerate(per_input_generated) if generated > 0]
        fairness_inputs = self._jain_index(active_input_values)
        flow_values = [flow_served[flow] for flow in flow_generated]
        fairness_flows = self._jain_index(flow_values)

        return SimulationResult(
            throughput=throughput,
            fairness_inputs=fairness_inputs,
            fairness_flows=fairness_flows,
            utilization=utilization,
            drop_rate=drop_rate,
            average_queue=average_queue,
            total_generated=total_generated,
            total_served=total_served,
            total_dropped=total_dropped,
        )

    @staticmethod
    def _jain_index(values: Iterable[int]) -> float:
        vector = [float(v) for v in values]
        if not vector:
            return 1.0
        numerator = sum(vector) ** 2
        denominator = len(vector) * sum(v * v for v in vector)
        if denominator == 0:
            return 1.0 if numerator > 0 else 0.0
        return numerator / denominator
