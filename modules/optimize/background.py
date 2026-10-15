"""Cosmic-ray background in the detectors (SWGO / SWGOLO swgolo148.cpp, HAP-2024-004).

1,826 muons / m^2 / s on detectors of 2 m^2: per detector and shower lambda = 2 * 1.826e-6 * W[ns] in ONE readout
window W shared by all detectors, centred on the shower's earliest timed detector t_ref (observed, no truth). Per window:
    E' = E + N_bg,  N_bg ~ Poisson(lambda)  (one particle per background muon)
    T' = min(T, earliest background time),  background times ~ U(t_ref - W/2, t_ref + W/2)
(Until 2026-10-15 the window was centred on each detector's TRUE front arrival, t_ref + (r_i - r_ref) . n_true / c:
background times then carried the true direction, and a recon fine-tuned on them learned to read it.)
Used by the run 19 noise scoring (plots/analysis/run19/noise/score_noise.py) and the noise fine-tune
(scripts/03c_finetune_recon_noise.py). E in particles, t in us, positions in m.
"""
import torch

RATE_PER_M2_NS = 1.826e-6
AREA_M2 = 2.0


def window_centres(t, timed, t_fallback):
    """Readout-window centre per detector [us], (B, nd): the shower's earliest timed detector, the same for every
    detector; showers with no timed detector use t_fallback. t, timed: (B, nd)."""
    big = torch.finfo(t.dtype).max
    t_ref = torch.where(timed, t, torch.full_like(t, big)).min(1).values
    t_ref = torch.where(timed.any(1), t_ref, torch.full_like(t_ref, t_fallback))
    return t_ref[:, None].expand_as(t)


def add_background(E, t, timed, t_front, W, gen):
    """E, t, timed, t_front (window centres, window_centres()): (B, nd); W: window [ns], a float or a (B,) tensor
    (one window per shower).
    Returns (E', T', N_bg, has_bg)."""
    if torch.is_tensor(W):
        W = W.to(E)[:, None]
        lam = AREA_M2 * RATE_PER_M2_NS * W
        Nb = torch.poisson(lam.expand_as(E).contiguous(), generator=gen)
    else:
        lam = AREA_M2 * RATE_PER_M2_NS * W
        Nb = torch.poisson(torch.full_like(E, lam), generator=gen) if lam > 0 else torch.zeros_like(E)
    u = torch.rand(E.shape, generator=gen, device=E.device)
    frac = 1.0 - u.pow(1.0 / Nb.clamp(min=1.0))                                # min of N_bg uniforms ~ Beta(1, N_bg)
    t_bg = (t_front - W / 2000.0 + (W / 1000.0) * frac).clamp(min=1e-6)
    has_bg = Nb > 0
    Tn = torch.where(has_bg, torch.where(timed, torch.minimum(t, t_bg), t_bg), t)
    return E + Nb, Tn, Nb, has_bg
