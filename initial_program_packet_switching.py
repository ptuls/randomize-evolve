"""Baseline scheduler candidate for packet-switching evolution."""


# EVOLVE-BLOCK-START
class CandidateScheduler:
    def __init__(self, ports):
        self.ports = ports
        self.output_priority = 0
        self.output_pointers = [0] * ports

    def select_matches(self, requests, time_slot, queue_lengths):
        matches = {}
        used_inputs = set()
        outputs_in_order = [
            (self.output_priority + offset) % self.ports
            for offset in range(self.ports)
        ]
        self.output_priority = (self.output_priority + 1) % self.ports

        for output_idx in outputs_in_order:
            candidates = requests.get(output_idx)
            if not candidates:
                continue

            pointer = self.output_pointers[output_idx]
            ranked_inputs = sorted(
                candidates,
                key=lambda input_idx: (
                    -(queue_lengths[input_idx] if input_idx < len(queue_lengths) else 0),
                    (input_idx - pointer) % self.ports,
                    (input_idx + time_slot) % self.ports,
                ),
            )

            for input_idx in ranked_inputs:
                if input_idx in used_inputs:
                    continue
                matches[input_idx] = output_idx
                used_inputs.add(input_idx)
                self.output_pointers[output_idx] = (input_idx + 1) % self.ports
                break

        return matches


# EVOLVE-BLOCK-END


def candidate_factory(ports):
    return CandidateScheduler(ports)


def run_demo():
    scheduler = candidate_factory(4)
    requests = {0: [0, 1], 1: [1, 2], 2: [2, 3]}
    queue_lengths = [4, 2, 3, 1]
    print(scheduler.select_matches(requests, time_slot=0, queue_lengths=queue_lengths))


if __name__ == "__main__":
    run_demo()
