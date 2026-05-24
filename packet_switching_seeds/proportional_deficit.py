"""Packet-switching seed that tracks per-VOQ proportional service deficits."""

from __future__ import annotations

from typing import Dict, List, MutableMapping, Sequence


class ProportionalDeficitScheduler:
    """Approximate proportional-fair service with persistent VOQ deficits."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        if num_inputs <= 0 or num_outputs <= 0:
            raise ValueError("Switch dimensions must be positive")
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._output_priority = 0
        self._output_pointers = [0 for _ in range(self.num_outputs)]
        self._input_idle_age = [0 for _ in range(self.num_inputs)]
        self._service_deficits = [
            [0.0 for _ in range(self.num_outputs)] for _ in range(self.num_inputs)
        ]

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
    ) -> MutableMapping[int, int]:
        """Build a maximal matching from service deficits and queue pressure."""

        del time_slot
        self._advance_input_ages(queue_lengths)

        row_backlogs = [self._queue_length(queue_lengths, i) for i in range(self.num_inputs)]
        column_backlogs = [
            self._output_backlog(voq_lengths, output_idx) for output_idx in range(self.num_outputs)
        ]
        total_backlog = sum(row_backlogs)
        self._refresh_service_deficits(voq_lengths, row_backlogs, column_backlogs)

        used_inputs = set()
        matches: Dict[int, int] = {}
        for output_idx in self._ordered_outputs(requests, column_backlogs):
            input_indices = requests.get(output_idx)
            if not input_indices:
                continue
            ranked_inputs = sorted(
                input_indices,
                key=lambda input_idx: (
                    -self._edge_score(
                        input_idx=input_idx,
                        output_idx=output_idx,
                        row_backlogs=row_backlogs,
                        column_backlogs=column_backlogs,
                        voq_lengths=voq_lengths,
                    ),
                    (input_idx - self._output_pointers[output_idx]) % self.num_inputs,
                ),
            )
            for input_idx in ranked_inputs:
                if input_idx in used_inputs:
                    continue
                matches[input_idx] = output_idx
                used_inputs.add(input_idx)
                self._output_pointers[output_idx] = (input_idx + 1) % self.num_inputs
                self._input_idle_age[input_idx] = 0
                self._service_deficits[input_idx][output_idx] = max(
                    0.0,
                    self._service_deficits[input_idx][output_idx] - 1.0,
                )
                break

        return matches

    def _ordered_outputs(
        self,
        requests: Dict[int, List[int]],
        column_backlogs: Sequence[int],
    ) -> List[int]:
        output_priority = self._build_output_priority()
        return sorted(
            range(self.num_outputs),
            key=lambda output_idx: (
                len(requests.get(output_idx, ())) == 0,
                -column_backlogs[output_idx],
                -len(requests.get(output_idx, ())),
                output_priority[output_idx],
            ),
        )

    def _advance_input_ages(self, queue_lengths: Sequence[int]) -> None:
        for input_idx in range(self.num_inputs):
            backlog = self._queue_length(queue_lengths, input_idx)
            if backlog > 0:
                self._input_idle_age[input_idx] += 1
            else:
                self._input_idle_age[input_idx] = 0

    def _refresh_service_deficits(
        self,
        voq_lengths: Sequence[Sequence[int]],
        row_backlogs: Sequence[int],
        column_backlogs: Sequence[int],
    ) -> None:
        for input_idx in range(self.num_inputs):
            row_backlog = row_backlogs[input_idx]
            for output_idx in range(self.num_outputs):
                backlog = self._voq_length(voq_lengths, input_idx, output_idx)
                if backlog <= 0 or row_backlog <= 0 or column_backlogs[output_idx] <= 0:
                    self._service_deficits[input_idx][output_idx] = 0.0
                    continue
                target_share = min(
                    backlog / row_backlog,
                    backlog / column_backlogs[output_idx],
                )
                updated_deficit = self._service_deficits[input_idx][output_idx] + target_share
                self._service_deficits[input_idx][output_idx] = min(updated_deficit, 4.0)

    def _build_output_priority(self) -> Dict[int, int]:
        rotated_outputs = [
            (self._output_priority + offset) % self.num_outputs
            for offset in range(self.num_outputs)
        ]
        self._output_priority = (self._output_priority + 1) % self.num_outputs
        return {output_idx: priority for priority, output_idx in enumerate(rotated_outputs)}

    def _edge_score(
        self,
        input_idx: int,
        output_idx: int,
        row_backlogs: Sequence[int],
        column_backlogs: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
    ) -> float:
        voq_backlog = self._voq_length(voq_lengths, input_idx, output_idx)
        row_backlog = row_backlogs[input_idx]
        column_backlog = column_backlogs[output_idx]
        deficit = self._service_deficits[input_idx][output_idx]
        input_age = self._input_idle_age[input_idx]
        return (
            2.0 * deficit
            + 1.1 * voq_backlog
            + 0.08 * row_backlog
            + 0.12 * column_backlog
            + 0.05 * input_age
        )

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

    @staticmethod
    def _output_backlog(
        voq_lengths: Sequence[Sequence[int]],
        output_idx: int,
    ) -> int:
        return sum(row[output_idx] for row in voq_lengths if output_idx < len(row))


def candidate_factory(ports: int) -> ProportionalDeficitScheduler:
    """Build the packet-switching seed scheduler."""

    return ProportionalDeficitScheduler(ports, ports)


build_candidate = candidate_factory
