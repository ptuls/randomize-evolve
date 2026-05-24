"""Abstract seed scheduler for packet-switching evolution.

This file is intentionally generic. It provides a valid scheduler contract and
simple deterministic behavior, but it is not meant to lock the search into a
specific paradigm such as round-robin, iSLIP, or pure max-weight matching.
Substantial rewrites are acceptable.
"""

from __future__ import annotations

from typing import Dict, List, MutableMapping, Sequence


class SchedulerScaffold:
    """Generic scaffold for packet-switch scheduling experiments."""

    VOQ_WEIGHT = 1.0
    INPUT_WEIGHT = 0.0
    OUTPUT_WEIGHT = 0.0
    CONTENTION_WEIGHT = 0.0

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
        """Return a valid matching with at most one edge per port.

        The current implementation is a simple score-ordered greedy matcher.
        It is intentionally lightweight so the mutator can replace it with a
        different paradigm if that improves the evaluator.
        """

        del time_slot
        candidate_edges: list[tuple[float, int, int]] = []
        for output_idx, input_indices in requests.items():
            output_backlog = self._output_backlog(voq_lengths, output_idx)
            contention = len(input_indices)
            for input_idx in input_indices:
                score = self._edge_score(
                    input_idx=input_idx,
                    output_idx=output_idx,
                    queue_lengths=queue_lengths,
                    voq_lengths=voq_lengths,
                    output_backlog=output_backlog,
                    contention=contention,
                )
                candidate_edges.append((-score, output_idx, input_idx))

        candidate_edges.sort()
        matches: Dict[int, int] = {}
        used_inputs = set()
        used_outputs = set()
        for _, output_idx, input_idx in candidate_edges:
            if input_idx in used_inputs or output_idx in used_outputs:
                continue
            matches[input_idx] = output_idx
            used_inputs.add(input_idx)
            used_outputs.add(output_idx)
        return matches

    def _edge_score(
        self,
        input_idx: int,
        output_idx: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
        output_backlog: int,
        contention: int,
    ) -> float:
        voq = self._voq_length(voq_lengths, input_idx, output_idx)
        input_backlog = self._queue_length(queue_lengths, input_idx)
        return (
            self.VOQ_WEIGHT * voq
            + self.INPUT_WEIGHT * input_backlog
            + self.OUTPUT_WEIGHT * output_backlog
            + self.CONTENTION_WEIGHT * contention
        )

    @staticmethod
    def _queue_length(queue_lengths: Sequence[int], input_idx: int) -> int:
        if input_idx < 0 or input_idx >= len(queue_lengths):
            return 0
        return queue_lengths[input_idx]

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


def candidate_factory(ports: int) -> SchedulerScaffold:
    """Build the abstract seed scheduler used for packet-switching evolution."""

    return SchedulerScaffold(ports, ports)


build_candidate = candidate_factory
