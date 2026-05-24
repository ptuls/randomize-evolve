"""Seed scheduler for packet-switching evolution.

This seed combines iterative request/grant/accept rounds, VOQ-weighted grants,
and per-VOQ aging so the search starts from a robust iSLIP-style policy.
"""

from __future__ import annotations

from typing import Dict, List, MutableMapping, Sequence


class VOQiSLIPScheduler:
    """iSLIP-style scheduler with VOQ-weighted grants and aging."""

    NUM_ROUNDS = 3
    AGE_WEIGHT = 0.25
    AGE_CAP = 16
    USE_LQF_TIEBREAK = True

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        if num_inputs <= 0 or num_outputs <= 0:
            raise ValueError("Switch dimensions must be positive")
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._grant_pointer = [0 for _ in range(num_outputs)]
        self._accept_pointer = [0 for _ in range(num_inputs)]
        self._age: List[List[int]] = [
            [0 for _ in range(num_outputs)] for _ in range(num_inputs)
        ]

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
    ) -> MutableMapping[int, int]:
        """Build a matching with weighted iSLIP grant and accept rounds."""

        del time_slot
        matches: Dict[int, int] = {}
        matched_inputs: set[int] = set()
        matched_outputs: set[int] = set()

        for round_idx in range(self.NUM_ROUNDS):
            grants: Dict[int, int] = {}

            for output_idx in range(self.num_outputs):
                if output_idx in matched_outputs:
                    continue
                requesters = [
                    input_idx
                    for input_idx in requests.get(output_idx, ())
                    if input_idx not in matched_inputs
                    and 0 <= input_idx < self.num_inputs
                ]
                if not requesters:
                    continue
                pointer = self._grant_pointer[output_idx]
                best_input = max(
                    requesters,
                    key=lambda input_idx: self._grant_score(
                        input_idx=input_idx,
                        output_idx=output_idx,
                        pointer=pointer,
                        voq_lengths=voq_lengths,
                        queue_lengths=queue_lengths,
                    ),
                )
                grants[output_idx] = best_input

            if not grants:
                break

            grants_by_input: Dict[int, List[int]] = {}
            for output_idx, input_idx in grants.items():
                grants_by_input.setdefault(input_idx, []).append(output_idx)

            new_matches = 0
            for input_idx, granted_outputs in grants_by_input.items():
                if input_idx in matched_inputs:
                    continue
                pointer = self._accept_pointer[input_idx]
                best_output = max(
                    granted_outputs,
                    key=lambda output_idx: self._accept_score(
                        input_idx=input_idx,
                        output_idx=output_idx,
                        pointer=pointer,
                        voq_lengths=voq_lengths,
                    ),
                )
                matches[input_idx] = best_output
                matched_inputs.add(input_idx)
                matched_outputs.add(best_output)
                new_matches += 1
                if round_idx == 0:
                    self._accept_pointer[input_idx] = (
                        best_output + 1
                    ) % self.num_outputs
                    self._grant_pointer[best_output] = (
                        input_idx + 1
                    ) % self.num_inputs

            if new_matches == 0:
                break

        self._update_ages(voq_lengths, matches)
        return matches

    def _grant_score(
        self,
        input_idx: int,
        output_idx: int,
        pointer: int,
        voq_lengths: Sequence[Sequence[int]],
        queue_lengths: Sequence[int],
    ) -> tuple[float, int, int]:
        voq = self._voq_length(voq_lengths, input_idx, output_idx)
        age = self._age[input_idx][output_idx]
        weight = voq + self.AGE_WEIGHT * age
        lqf = self._queue_length(queue_lengths, input_idx) if self.USE_LQF_TIEBREAK else 0
        rr_distance = -((input_idx - pointer) % self.num_inputs)
        return (weight, lqf, rr_distance)

    def _accept_score(
        self,
        input_idx: int,
        output_idx: int,
        pointer: int,
        voq_lengths: Sequence[Sequence[int]],
    ) -> tuple[float, int]:
        voq = self._voq_length(voq_lengths, input_idx, output_idx)
        age = self._age[input_idx][output_idx]
        weight = voq + self.AGE_WEIGHT * age
        rr_distance = -((output_idx - pointer) % self.num_outputs)
        return (weight, rr_distance)

    def _update_ages(
        self,
        voq_lengths: Sequence[Sequence[int]],
        matches: Dict[int, int],
    ) -> None:
        for input_idx in range(self.num_inputs):
            served_output = matches.get(input_idx)
            for output_idx in range(self.num_outputs):
                backlog = self._voq_length(voq_lengths, input_idx, output_idx)
                if backlog <= 0 or served_output == output_idx:
                    self._age[input_idx][output_idx] = 0
                    continue
                self._age[input_idx][output_idx] = min(
                    self._age[input_idx][output_idx] + 1,
                    self.AGE_CAP,
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


def candidate_factory(ports: int) -> VOQiSLIPScheduler:
    """Build the weighted iSLIP seed scheduler."""

    return VOQiSLIPScheduler(ports, ports)


build_candidate = candidate_factory
