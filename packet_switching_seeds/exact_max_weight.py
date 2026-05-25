"""Exact max-weight matching baseline for packet-switching evaluation."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, MutableMapping, Sequence


class ExactMaxWeightScheduler:
    """Solve the slot matching exactly for small switches by search."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        if num_inputs <= 0 or num_outputs <= 0:
            raise ValueError("Switch dimensions must be positive")
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
    ) -> MutableMapping[int, int]:
        """Return an exact maximum-weight matching over visible VOQs."""

        del time_slot, queue_lengths
        output_requesters = tuple(
            tuple(
                sorted(
                    input_idx
                    for input_idx in requests.get(output_idx, ())
                    if 0 <= input_idx < self.num_inputs
                    and self._voq_length(voq_lengths, input_idx, output_idx) > 0
                )
            )
            for output_idx in range(self.num_outputs)
        )

        @lru_cache(maxsize=None)
        def solve(
            output_idx: int, used_inputs_mask: int
        ) -> tuple[int, tuple[tuple[int, int], ...]]:
            if output_idx >= self.num_outputs:
                return 0, ()

            best_weight, best_edges = solve(output_idx + 1, used_inputs_mask)

            for input_idx in output_requesters[output_idx]:
                input_mask = 1 << input_idx
                if used_inputs_mask & input_mask:
                    continue
                edge_weight = self._voq_length(voq_lengths, input_idx, output_idx)
                tail_weight, tail_edges = solve(
                    output_idx + 1,
                    used_inputs_mask | input_mask,
                )
                total_weight = edge_weight + tail_weight
                edge_tuple = ((input_idx, output_idx),) + tail_edges
                if total_weight > best_weight or (
                    total_weight == best_weight and edge_tuple < best_edges
                ):
                    best_weight = total_weight
                    best_edges = edge_tuple

            return best_weight, best_edges

        _, edges = solve(0, 0)
        return {input_idx: output_idx for input_idx, output_idx in edges}

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


def candidate_factory(ports: int) -> ExactMaxWeightScheduler:
    """Build the exact max-weight baseline."""

    return ExactMaxWeightScheduler(ports, ports)


build_candidate = candidate_factory
