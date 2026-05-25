"""Baseline scheduler for packet-switching experiments.

The ``candidate_factory`` function is the OpenEvolve entry point for this task.
It returns a scheduler implementing ``select_matches`` over the virtual output
queue matrix ``Q_ij`` exposed by the simulator.
"""

from __future__ import annotations

from typing import Dict, List, MutableMapping, Optional, Sequence


class VOQRoundRobinScheduler:
    """Round-robin scheduler with VOQ-aware tie-breaking."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._output_priority = 0
        self._output_pointers = [0 for _ in range(self.num_outputs)]
        if self.num_inputs <= 0 or self.num_outputs <= 0:
            raise ValueError("Switch dimensions must be positive")

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
        voq_ages: Optional[Sequence[Sequence[int]]] = None,
    ) -> MutableMapping[int, int]:
        """Build a maximal matching using rotating output priorities."""

        del time_slot
        del voq_ages
        matches: Dict[int, int] = {}
        used_inputs = set()

        rotated_outputs = [
            (self._output_priority + offset) % self.num_outputs
            for offset in range(self.num_outputs)
        ]
        self._output_priority = (self._output_priority + 1) % self.num_outputs
        output_priority = {
            output_idx: priority for priority, output_idx in enumerate(rotated_outputs)
        }
        outputs_in_order = sorted(
            rotated_outputs,
            key=lambda output_idx: (
                len(requests.get(output_idx, ())) or self.num_inputs + 1,
                output_priority[output_idx],
            ),
        )

        for output_idx in outputs_in_order:
            candidates = requests.get(output_idx)
            if not candidates:
                continue
            pointer = self._output_pointers[output_idx]
            sorted_candidates = sorted(
                candidates,
                key=lambda input_idx: (
                    -self._voq_length(voq_lengths, input_idx, output_idx),
                    -self._queue_length(queue_lengths, input_idx),
                    (input_idx - pointer) % self.num_inputs,
                ),
            )
            for input_idx in sorted_candidates:
                if input_idx in used_inputs:
                    continue
                matches[input_idx] = output_idx
                used_inputs.add(input_idx)
                self._output_pointers[output_idx] = (input_idx + 1) % self.num_inputs
                break
        return matches

    @staticmethod
    def _queue_length(queue_lengths: Sequence[int], input_idx: int) -> int:
        if input_idx >= len(queue_lengths):
            return 0
        return queue_lengths[input_idx]

    @staticmethod
    def _voq_length(
        voq_lengths: Sequence[Sequence[int]],
        input_idx: int,
        output_idx: int,
    ) -> int:
        if input_idx >= len(voq_lengths):
            return 0
        row = voq_lengths[input_idx]
        if output_idx >= len(row):
            return 0
        return row[output_idx]


def candidate_factory(ports: int) -> VOQRoundRobinScheduler:
    """Build the baseline scheduler used for packet-switching evolution."""

    return VOQRoundRobinScheduler(ports, ports)


build_candidate = candidate_factory
