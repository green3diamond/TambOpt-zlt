"""Counting noise: the kernel's per-species E (an expectation, in kernel units) -> random particle counts.

Per shower, detector and species s (constants.SPECIES_NAMES: electron, muon, photon), at the detector's distance r
from the shower axis:
    mu_s = E_s * particles_per_E(s, r)              expected particles in the 2 m^2 detector (background.py)
    mu_photon *= eps_gamma                          only photons that convert / Compton-scatter give a signal
    N_s ~ Poisson(mu_s)
    E_s' = N_s / (particles_per_E(s, r) [* eps_gamma])   back in kernel units, so the recon's input space is unchanged
    T_s' = T_s if N_s >= 1 else 0                   no particle -> no time; combine_species_outputs then drops the
                                                    species from the leading-edge min. T is not jittered further.
Hit = any counted particle (E' > 0). mode="expected" leaves the inputs unchanged (the reference path).
eps_gamma = 0.02 is an assumption (~1 cm plastic scintillator: thickness / X0 ~ 1/42 for conversion, Compton of
~40 MeV photons similar); replace it with the detector's real value when known.

Gradients (Stage 4), as in SWGOLO (swgolo148.cpp, PActive): never through the sampled count. straight_through=True
gives the counted E' in the forward pass and the gradient of its expectation (E[E'] = E, dE'/dE := 1), i.e. counts
held fixed; p_hit() = 1 - exp(-sum_s mu_s) is the smooth detection probability for the hit / gate path.
"""
import torch

from ..constants import SPECIES_NAMES
from .background import axis_distance, particles_per_E

KEY = {"electron": "e", "muon": "mu", "photon": "gamma"}
EPS_GAMMA = 0.02


def _scale(name, r, eps_gamma):
    """Detected particles per unit kernel E of species `name` at distance r (tensor or float)."""
    k = particles_per_E(KEY[name], r)
    return k * eps_gamma if name == "photon" else k


def expected_counts(preds, r, eps_gamma=EPS_GAMMA, species=SPECIES_NAMES):
    """preds: per-species (B, nd, 2) [log1p E, log1p(t * T_LOG_SCALE)] in `species` order; r: (B, nd) distance from
    the axis [m] or None (band values). Returns per-species (B, nd) expected detected particles mu_s."""
    return [torch.expm1(p[..., 0].clamp(min=0.0)) * _scale(nm, r, eps_gamma) for nm, p in zip(species, preds)]


def p_hit(preds, r, eps_gamma=EPS_GAMMA, species=SPECIES_NAMES):
    """Probability of >= 1 detected particle, 1 - exp(-sum_s mu_s), (B, nd); differentiable in E."""
    return 1.0 - torch.exp(-sum(expected_counts(preds, r, eps_gamma, species)))


def count_species(preds, r, gen, mode="poisson", eps_gamma=EPS_GAMMA, species=SPECIES_NAMES, straight_through=False):
    """Per-species labels -> the same labels with counted E and T zeroed where nothing arrived (module docstring).
    Returns (preds', n_detected (B, nd) total counted particles, or None for mode="expected")."""
    if mode == "expected":
        return list(preds), None
    if mode != "poisson":
        raise ValueError(f"mode must be 'poisson' or 'expected', got {mode!r}")
    out, n_tot = [], 0
    for nm, p in zip(species, preds):
        k = _scale(nm, r, eps_gamma)
        E = torch.expm1(p[..., 0].clamp(min=0.0))
        N = torch.poisson((E * k).detach().float(), generator=gen).to(E)
        Ec = N / k
        if straight_through:
            Ec = E + (Ec - E).detach()
        lit = N > 0
        out.append(torch.stack([torch.log1p(Ec), torch.where(lit, p[..., 1], torch.zeros_like(p[..., 1]))], -1))
        n_tot = n_tot + N
    return out, n_tot


class CountingTransform:
    """species_transform for KernelDualLabels / BatchKernel: applies count_species before the species combination,
    with r from the kernel's shower frames. `gen` is reset by the caller for reproducible draws; `n_detected` keeps
    the last call's mean detectors with >= 1 counted particle per shower."""

    def __init__(self, surface, gen, mode="poisson", eps_gamma=EPS_GAMMA, straight_through=False):
        self.surface, self.gen, self.mode = surface, gen, mode
        self.eps_gamma, self.st = eps_gamma, straight_through
        self.n_detected = []

    def __call__(self, preds, e_det, n_det, frames):
        r = axis_distance(e_det, n_det, self.surface, frames) if frames is not None else None
        out, n = count_species(preds, r, self.gen, self.mode, self.eps_gamma, straight_through=self.st)
        if n is not None:
            self.n_detected.append((n > 0).float().sum(1))
        return out
