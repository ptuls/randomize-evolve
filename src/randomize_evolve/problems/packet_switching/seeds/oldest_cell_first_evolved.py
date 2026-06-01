"""Evolved oldest-cell-first seed for packet-switching evolution."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, MutableMapping, Optional, Sequence


class EvolvedOldestCellFirstScheduler:
    """Age-dominant exact matching with stronger backlog tie-breaking."""

    AGE_MULTIPLIER = 1_000_000

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
        voq_ages: Optional[Sequence[Sequence[int]]] = None,
    ) -> MutableMapping[int, int]:
        """Return an age-priority matching with exact search fallback."""

        del time_slot, queue_lengths
        if voq_ages is None:
            raise ValueError("voq_ages are required for oldest-cell-first scheduling")

        output_requesters = tuple(
            tuple(
                input_idx
                for input_idx in requests.get(output_idx, ())
                if 0 <= input_idx < self.num_inputs
                and self._voq_length(voq_lengths, input_idx, output_idx) > 0
            )
            for output_idx in range(self.num_outputs)
        )

        used_inputs = 0
        greedy_edges = []
        for output_idx in range(self.num_outputs):
            best_input = -1
            best_weight = -1
            for input_idx in output_requesters[output_idx]:
                bit = 1 << input_idx
                if used_inputs & bit:
                    continue
                weight = self._edge_weight(voq_lengths, voq_ages, input_idx, output_idx)
                if weight > best_weight or (weight == best_weight and input_idx < best_input):
                    best_weight = weight
                    best_input = input_idx
            if best_input >= 0:
                used_inputs |= 1 << best_input
                greedy_edges.append((best_input, output_idx))

        if self.num_outputs > 12 or len(greedy_edges) <= 1:
            return {input_idx: output_idx for input_idx, output_idx in greedy_edges}

        @lru_cache(maxsize=None)
        def solve(
            output_idx: int,
            used_inputs_mask: int,
        ) -> tuple[int, tuple[tuple[int, int], ...]]:
            if output_idx >= self.num_outputs:
                return 0, ()

            skip_weight, skip_edges = solve(output_idx + 1, used_inputs_mask)
            best_weight, best_edges = skip_weight, skip_edges

            for input_idx in output_requesters[output_idx]:
                input_mask = 1 << input_idx
                if used_inputs_mask & input_mask:
                    continue
                edge_weight = self._edge_weight(
                    voq_lengths,
                    voq_ages,
                    input_idx,
                    output_idx,
                )
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

        _, exact_edges = solve(0, 0)
        chosen_edges = exact_edges if len(exact_edges) >= len(greedy_edges) else tuple(greedy_edges)
        return {input_idx: output_idx for input_idx, output_idx in chosen_edges}

    def _edge_weight(
        self,
        voq_lengths: Sequence[Sequence[int]],
        voq_ages: Sequence[Sequence[int]],
        input_idx: int,
        output_idx: int,
    ) -> int:
        age = self._voq_age(voq_ages, input_idx, output_idx)
        backlog = self._voq_length(voq_lengths, input_idx, output_idx)
        return self.AGE_MULTIPLIER * age + (backlog << 12) - (input_idx << 4) - output_idx

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
    def _voq_age(
        voq_ages: Sequence[Sequence[int]],
        input_idx: int,
        output_idx: int,
    ) -> int:
        if input_idx < 0 or input_idx >= len(voq_ages):
            return 0
        row = voq_ages[input_idx]
        if output_idx < 0 or output_idx >= len(row):
            return 0
        return row[output_idx]


def candidate_factory(ports: int) -> EvolvedOldestCellFirstScheduler:
    """Build the evolved oldest-cell-first seed."""

    return EvolvedOldestCellFirstScheduler(ports, ports)


build_candidate = candidate_factory
