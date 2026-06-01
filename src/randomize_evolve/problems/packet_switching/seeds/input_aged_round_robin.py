"""Packet-switching seed that tracks per-input service age."""

from __future__ import annotations

from typing import Dict, List, MutableMapping, Sequence


class InputAgedRoundRobinScheduler:
    """Round-robin scheduler that boosts inputs left idle for many slots."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        if num_inputs <= 0 or num_outputs <= 0:
            raise ValueError("Switch dimensions must be positive")
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._output_priority = 0
        self._output_pointers = [0 for _ in range(self.num_outputs)]
        self._input_ages = [0 for _ in range(self.num_inputs)]

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
    ) -> MutableMapping[int, int]:
        """Build a maximal matching using per-input age as a tie-breaker."""

        del time_slot
        matches: Dict[int, int] = {}
        used_inputs = set()
        self._input_ages = [age + 1 for age in self._input_ages]

        rotated_outputs = [
            (self._output_priority + offset) % self.num_outputs
            for offset in range(self.num_outputs)
        ]
        self._output_priority = (self._output_priority + 1) % self.num_outputs

        outputs_in_order = sorted(
            rotated_outputs,
            key=lambda output_idx: (
                len(requests.get(output_idx, ())) == 0,
                -len(requests.get(output_idx, ())),
                -self._output_backlog(voq_lengths, output_idx),
                output_idx,
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
                    -self._input_age(input_idx),
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
                self._input_ages[input_idx] = 0
                break
        return matches

    def _input_age(self, input_idx: int) -> int:
        if input_idx >= len(self._input_ages):
            return 0
        return self._input_ages[input_idx]

    @staticmethod
    def _queue_length(queue_lengths: Sequence[int], input_idx: int) -> int:
        return queue_lengths[input_idx] if input_idx < len(queue_lengths) else 0

    @staticmethod
    def _voq_length(
        voq_lengths: Sequence[Sequence[int]],
        input_idx: int,
        output_idx: int,
    ) -> int:
        if input_idx >= len(voq_lengths):
            return 0
        row = voq_lengths[input_idx]
        return row[output_idx] if output_idx < len(row) else 0

    @staticmethod
    def _output_backlog(
        voq_lengths: Sequence[Sequence[int]],
        output_idx: int,
    ) -> int:
        return sum(row[output_idx] for row in voq_lengths if output_idx < len(row))


def candidate_factory(ports: int) -> InputAgedRoundRobinScheduler:
    """Build the packet-switching seed scheduler."""

    return InputAgedRoundRobinScheduler(ports, ports)


build_candidate = candidate_factory
