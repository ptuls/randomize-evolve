"""Packet-switching seed that uses deterministic randomized iterations."""

from __future__ import annotations

from typing import Dict, List, MutableMapping, Sequence


class RandomizedIterativeScheduler:
    """PIM-style scheduler with deterministic pseudo-random tie-breaking."""

    NUM_ROUNDS = 3

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        if num_inputs <= 0 or num_outputs <= 0:
            raise ValueError("Switch dimensions must be positive")
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._state = 0x6D2B79F5

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
    ) -> MutableMapping[int, int]:
        """Build a matching with randomized request, grant, and accept rounds."""

        del queue_lengths, voq_lengths
        input_requests = self._invert_requests(requests)
        matches: Dict[int, int] = {}
        matched_inputs: set[int] = set()
        matched_outputs: set[int] = set()
        self._state = (self._state + time_slot + 1) & 0xFFFFFFFF

        for _ in range(self.NUM_ROUNDS):
            requested_inputs: Dict[int, List[int]] = {}
            for input_idx, output_indices in input_requests.items():
                if input_idx in matched_inputs or not output_indices:
                    continue
                available_outputs = [
                    output_idx
                    for output_idx in output_indices
                    if output_idx not in matched_outputs
                ]
                if not available_outputs:
                    continue
                output_idx = available_outputs[
                    self._draw_index(len(available_outputs))
                ]
                requested_inputs.setdefault(output_idx, []).append(input_idx)

            if not requested_inputs:
                break

            grants: Dict[int, int] = {}
            for output_idx, input_indices in requested_inputs.items():
                grants[output_idx] = input_indices[self._draw_index(len(input_indices))]

            grants_by_input: Dict[int, List[int]] = {}
            for output_idx, input_idx in grants.items():
                if output_idx in matched_outputs or input_idx in matched_inputs:
                    continue
                grants_by_input.setdefault(input_idx, []).append(output_idx)

            new_matches = 0
            for input_idx, output_indices in grants_by_input.items():
                output_idx = output_indices[self._draw_index(len(output_indices))]
                matches[input_idx] = output_idx
                matched_inputs.add(input_idx)
                matched_outputs.add(output_idx)
                new_matches += 1

            if new_matches == 0:
                break

        return matches

    def _draw_index(self, length: int) -> int:
        if length <= 1:
            return 0
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state % length

    def _invert_requests(
        self,
        requests: Dict[int, List[int]],
    ) -> Dict[int, List[int]]:
        input_requests: Dict[int, List[int]] = {
            input_idx: [] for input_idx in range(self.num_inputs)
        }
        for output_idx, input_indices in requests.items():
            for input_idx in input_indices:
                if 0 <= input_idx < self.num_inputs:
                    input_requests[input_idx].append(output_idx)
        return input_requests


def candidate_factory(ports: int) -> RandomizedIterativeScheduler:
    """Build the randomized iterative seed scheduler."""

    return RandomizedIterativeScheduler(ports, ports)


build_candidate = candidate_factory
