"""Utility of a layout on a fixed event set, split into equal chunks for paired errors."""
from dataclasses import dataclass

import numpy as np
import torch

from analysis_core.run import REPO  # noqa: F401  (puts the repo and plots/layouts on the path)
from modules.optimize import utility_of_xy
from true_utility import KernelDualLabels

LABELS = ("surrogate", "kernel")
PARTS = ("u_theta", "u_phi", "u_e")


class FixedLabels:
    """Stands in for the surrogate and returns precomputed detector labels."""

    def __init__(self, labels):
        self.labels = labels

    def __call__(self, primary, xy):
        if primary.shape[0] != self.labels.shape[0] or xy.shape[1] != self.labels.shape[1]:
            raise ValueError(f"labels {tuple(self.labels.shape)} do not match batch "
                             f"{primary.shape[0]} x {xy.shape[1]} detectors")
        return self.labels


@dataclass
class Score:
    chunks: np.ndarray   # (K,) U of each chunk
    parts: dict          # term name -> (K,)

    @property
    def U(self):
        return float(self.chunks.mean())

    @property
    def se(self):
        return float(self.chunks.std(ddof=1) / np.sqrt(len(self.chunks)))

    def summary(self):
        return {"U": self.U, "se": self.se, **{k: float(v.mean()) for k, v in self.parts.items()}}


def paired(a, b):
    """(mean, standard error) of a - b over matched chunks."""
    d = a.chunks - b.chunks
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


class Scorer:
    def __init__(self, run, events, labels, chunk, kernel_chunk=128):
        if labels not in LABELS:
            raise ValueError(f"labels must be one of {LABELS}, got {labels!r}")
        if len(events) % chunk:
            raise ValueError(f"{len(events)} events is not a multiple of chunk {chunk}")
        self.run, self.events, self.source, self.chunk = run, events, labels, chunk
        if labels == "kernel":
            self.kernel = KernelDualLabels(events.clouds, run.surface, run.device,
                                           chunk=kernel_chunk, expected_primary=events.primary)

    @torch.no_grad()
    def labels(self, e, n):
        """(B, n_det, 2) detector response: the surrogate MEAN, or the kernel."""
        xy = torch.stack([e, n], -1).to(self.run.device)
        prim = self.events.primary
        if self.source == "kernel":
            return self.kernel(prim, xy.expand(len(prim), -1, -1))
        return torch.cat([self.run.fnn(p, xy.expand(len(p), -1, -1))
                          for p in prim.split(self.chunk)])

    @torch.no_grad()
    def score(self, e, n):
        e, n = e.to(self.run.device), n.to(self.run.device)
        U, parts = [], {k: [] for k in PARTS}
        for p, lab in zip(self.events.primary.split(self.chunk), self.labels(e, n).split(self.chunk)):
            u, _, pr = utility_of_xy(e, n, p, FixedLabels(lab), self.run.recon, sample=False)
            U.append(u.item())
            for k in PARTS:
                parts[k].append(pr[k].item())
        return Score(np.array(U), {k: np.array(v) for k, v in parts.items()})
