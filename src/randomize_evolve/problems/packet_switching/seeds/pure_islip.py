"""Reference iSLIP baseline for packet-switching evaluation."""

from __future__ import annotations

from typing import Dict, List, MutableMapping, Sequence


class ISLIPScheduler:
    """Classic iSLIP scheduler with iterative request, grant, and accept."""

    NUM_ROUNDS = 3

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        if num_inputs <= 0 or num_outputs <= 0:
            raise ValueError("Switch dimensions must be positive")
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._grant_pointer = [0 for _ in range(num_outputs)]
        self._accept_pointer = [0 for _ in range(num_inputs)]

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
    ) -> MutableMapping[int, int]:
        """Build a matching using unweighted iSLIP arbitration."""

        del time_slot, queue_lengths, voq_lengths
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
                    if 0 <= input_idx < self.num_inputs and input_idx not in matched_inputs
                ]
                if not requesters:
                    continue
                pointer = self._grant_pointer[output_idx]
                grants[output_idx] = min(
                    requesters,
                    key=lambda input_idx: (input_idx - pointer) % self.num_inputs,
                )

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
                accepted_output = min(
                    granted_outputs,
                    key=lambda output_idx: (output_idx - pointer) % self.num_outputs,
                )
                matches[input_idx] = accepted_output
                matched_inputs.add(input_idx)
                matched_outputs.add(accepted_output)
                new_matches += 1
                if round_idx == 0:
                    self._accept_pointer[input_idx] = (accepted_output + 1) % self.num_outputs
                    self._grant_pointer[accepted_output] = (input_idx + 1) % self.num_inputs

            if new_matches == 0:
                break

        return matches


def candidate_factory(ports: int) -> ISLIPScheduler:
    """Build the reference iSLIP baseline."""

    return ISLIPScheduler(ports, ports)


build_candidate = candidate_factory
