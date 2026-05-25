"""Scheduling strategies for the packet switching simulator."""

from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Optional, Protocol, Sequence


class SwitchScheduler(Protocol):
    """Protocol for algorithms that compute input-output matchings."""

    def select_matches(
        self,
        requests: Dict[int, List[int]],
        time_slot: int,
        queue_lengths: Sequence[int],
        voq_lengths: Sequence[Sequence[int]],
        voq_ages: Optional[Sequence[Sequence[int]]] = None,
    ) -> MutableMapping[int, int]:
        """Computes the matching for a single time slot.

        Args:
            requests: Mapping from output index to candidate input indices.
            time_slot: Current simulation slot.
            queue_lengths: Queue length snapshot for each input.
            voq_lengths: Virtual output queue lengths for each input-output pair.
            voq_ages: Age in slots of the oldest cell in each non-empty VOQ.

        Returns:
            A mapping from input index to output index.

        Notes:
            The simulator uses virtual output queues. ``requests`` therefore
            contains every input with a non-empty queue for a given output,
            while ``queue_lengths`` is the total backlog per input. Policies
            derived from the switched-networks literature should primarily use
            ``voq_lengths`` because they depend on the queue matrix ``Q_ij``.
            Age-based policies such as oldest-cell-first can also use
            ``voq_ages`` when they need head-of-line cell age information.
        """


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
        voq_lengths: Sequence[Sequence[int]],
        voq_ages: Optional[Sequence[Sequence[int]]] = None,
    ) -> MutableMapping[int, int]:
        del time_slot
        del voq_ages
        matches: Dict[int, int] = {}
        used_inputs = set()
        rotated_outputs = [
            (self._output_priority + offset) % self.num_outputs
            for offset in range(self.num_outputs)
        ]
        self._output_priority = (self._output_priority + 1) % self.num_outputs
        output_priority = {
            output_idx: priority for priority, output_idx in enumerate(rotated_outputs)
        }
        outputs_in_order = sorted(
            rotated_outputs,
            key=lambda output_idx: (
                len(requests.get(output_idx, ())) or self.num_inputs + 1,
                output_priority[output_idx],
            ),
        )

        for output_idx in outputs_in_order:
            candidates = requests.get(output_idx)
            if not candidates:
                continue
            pointer = self._output_pointers[output_idx]
            sorted_candidates = sorted(
                candidates,
                key=lambda idx: (
                    (
                        -voq_lengths[idx][output_idx]
                        if idx < len(voq_lengths) and output_idx < len(voq_lengths[idx])
                        else 0
                    ),
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
