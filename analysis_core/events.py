"""Held-out events: placed shower clouds with their encoded primaries."""
from dataclasses import dataclass

import torch

from analysis_core.run import REPO  # noqa: F401  (puts the repo and plots/layouts on the path)
from true_utility import load_events


@dataclass
class Events:
    clouds: list            # one (B, P, 5) CPU tensor per species
    primary: torch.Tensor   # (B, 8) on the run device

    def __len__(self):
        return int(self.primary.shape[0])

    def part(self, i, k):
        """The i-th of k equal contiguous parts."""
        size = len(self) // k
        s = slice(i * size, (i + 1) * size)
        return Events([c[s] for c in self.clouds], self.primary[s])


def load(run, n_events, seed, multiple):
    """The first n_events held-out events, shuffled, trimmed to a multiple of `multiple`."""
    clouds, B, _, primary = load_events(n_events, run.device, run.mountain)
    order = torch.randperm(B, generator=torch.Generator().manual_seed(seed))
    order = order[: B - B % multiple]
    return Events([c[order] for c in clouds], primary[order.to(primary.device)])
