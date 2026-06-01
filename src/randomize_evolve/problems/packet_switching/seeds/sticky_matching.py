"""Packet-switching seed that preserves productive matches across slots."""

from __future__ import annotations

from typing import Dict, List, MutableMapping, Sequence


class StickyMatchingScheduler:
    """Reuse non-empty prior matches, then fill remaining ports greedily."""

    STICKY_BONUS = 2.0
    OUTPUT_PRESSURE_WEIGHT = 0.25

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        if num_inputs <= 0 or num_outputs <= 0:
            raise ValueError("Switch dimensions must be positive")
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._last_matches: Dict[int, int] = {}

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
    ) -> MutableMapping[int, int]:
        """Build a sticky maximal matching."""

        del time_slot, queue_lengths
        matches: Dict[int, int] = {}
        used_inputs = set()
        used_outputs = set()

        for input_idx, output_idx in list(self._last_matches.items()):
            if input_idx in used_inputs or output_idx in used_outputs:
                continue
            if input_idx not in requests.get(output_idx, ()):
                continue
            if self._voq_length(voq_lengths, input_idx, output_idx) <= 0:
                continue
            matches[input_idx] = output_idx
            used_inputs.add(input_idx)
            used_outputs.add(output_idx)

        candidate_edges: list[tuple[float, int, int]] = []
        for output_idx, input_indices in requests.items():
            if output_idx in used_outputs:
                continue
            output_pressure = self._output_backlog(voq_lengths, output_idx)
            for input_idx in input_indices:
                if input_idx in used_inputs:
                    continue
                voq = self._voq_length(voq_lengths, input_idx, output_idx)
                sticky_bonus = (
                    self.STICKY_BONUS if self._last_matches.get(input_idx) == output_idx else 0.0
                )
                score = voq + sticky_bonus + self.OUTPUT_PRESSURE_WEIGHT * output_pressure
                candidate_edges.append((-score, output_idx, input_idx))

        candidate_edges.sort()
        for _, output_idx, input_idx in candidate_edges:
            if input_idx in used_inputs or output_idx in used_outputs:
                continue
            matches[input_idx] = output_idx
            used_inputs.add(input_idx)
            used_outputs.add(output_idx)

        self._last_matches = dict(matches)
        return matches

    @staticmethod
    def _voq_length(
        voq_lengths: Sequence[Sequence[int]],
        input_idx: int,
        output_idx: int,
    ) -> int:
        if input_idx < 0 or input_idx >= len(voq_lengths):
            return 0
        row = voq_lengths[input_idx]
        if output_idx < 0 or output_idx >= len(row):
            return 0
        return row[output_idx]

    @staticmethod
    def _output_backlog(
        voq_lengths: Sequence[Sequence[int]],
        output_idx: int,
    ) -> int:
        return sum(row[output_idx] for row in voq_lengths if output_idx < len(row))


def candidate_factory(ports: int) -> StickyMatchingScheduler:
    """Build the sticky matching seed scheduler."""

    return StickyMatchingScheduler(ports, ports)


build_candidate = candidate_factory
