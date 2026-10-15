"""Cosmic-ray background in the detectors (SWGO / SWGOLO swgolo148.cpp, HAP-2024-004).

1,826 muons / m^2 / s on detectors of 2 m^2: per detector and shower lambda = 2 * 1.826e-6 * W[ns] in a window W
centred on the true front arrival at the detector. Per window:
    E' = E + N_bg,  N_bg ~ Poisson(lambda)  (one particle per background muon)
    T' = min(T, earliest background time),  background times ~ U(t_front - W/2, t_front + W/2)
    t_front,i = t_ref + (r_i - r_ref) . n / c   (r = (E, N, Up); t_ref, r_ref = the earliest timed detector)
Used by the run 19 noise scoring (plots/analysis/run19/noise/score_noise.py) and the noise fine-tune
(scripts/03c_finetune_recon_noise.py). E in particles, t in us, positions in m.
"""
import torch

RATE_PER_M2_NS = 1.826e-6
AREA_M2 = 2.0
C_M_PER_US = 299.792458


def front_times(t, timed, P, n, t_fallback):
    """True front arrival per detector [us]. t, timed: (B, nd); P: (nd, 3) or (B, nd, 3) ENU m; n: (B, 3) unit
    propagation direction. Reference = the earliest timed detector; showers with none use the layout centroid at
    t_fallback."""
    B = t.shape[0]
    P = P.expand(B, -1, -1) if P.dim() == 2 else P
    big = torch.finfo(t.dtype).max
    k_ref = torch.where(timed, t, torch.full_like(t, big)).argmin(1)
    any_t = timed.any(1)
    r_ref = torch.where(any_t[:, None], P[torch.arange(B, device=t.device), k_ref], P.mean(1))
    t_ref = torch.where(any_t, t.gather(1, k_ref[:, None]).squeeze(1), torch.full_like(t[:, 0], t_fallback))
    return t_ref[:, None] + ((P - r_ref[:, None]) * n[:, None]).sum(-1) / C_M_PER_US


def add_background(E, t, timed, t_front, W, gen):
    """E, t, timed, t_front: (B, nd); W: window [ns], a float or a (B,) tensor (one window per shower).
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
