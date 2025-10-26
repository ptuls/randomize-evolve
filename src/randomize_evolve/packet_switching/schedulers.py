"""Scheduling strategies for the packet switching simulator."""

from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Protocol, Sequence


class SwitchScheduler(Protocol):
    """Protocol for algorithms that compute input-output matchings."""

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
    ) -> MutableMapping[int, int]:
        """Return a mapping of input index to output index for the current slot."""


@dataclass
class RoundRobinScheduler:
    """A simple round-robin scheduler with queue length awareness."""

    num_inputs: int
    num_outputs: int

    def __post_init__(self) -> None:
        if self.num_inputs <= 0 or self.num_outputs <= 0:
            raise ValueError("Switch dimensions must be positive")
        self._output_priority = 0
        self._output_pointers = [0 for _ in range(self.num_outputs)]

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
    ) -> MutableMapping[int, int]:
        matches: Dict[int, int] = {}
        used_inputs = set()
        outputs_in_order = [
            (self._output_priority + offset) % self.num_outputs
            for offset in range(self.num_outputs)
        ]
        self._output_priority = (self._output_priority + 1) % self.num_outputs

        for output_idx in outputs_in_order:
            candidates = requests.get(output_idx)
            if not candidates:
                continue
            pointer = self._output_pointers[output_idx]
            sorted_candidates = sorted(
                candidates,
                key=lambda idx: (
                    -queue_lengths[idx] if idx < len(queue_lengths) else 0,
                    (idx - pointer) % self.num_inputs,
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
