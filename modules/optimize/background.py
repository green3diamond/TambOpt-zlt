"""Cosmic-ray background in the detectors (SWGO / SWGOLO swgolo148.cpp, HAP-2024-004).

1,826 muons / m^2 / s on detectors of 2 m^2: per detector and shower lambda = 2 * 1.826e-6 * W[ns] in ONE readout
window W shared by all detectors, centred on the shower's earliest timed detector t_ref (observed, no truth). Per window:
    E' = E + N_bg,  N_bg ~ Poisson(lambda)  (one particle per background muon)
    T' = min(T, earliest background time),  background times ~ U(t_ref - W/2, t_ref + W/2)
(Until 2026-10-15 the window was centred on each detector's TRUE front arrival, t_ref + (r_i - r_ref) . n_true / c:
background times then carried the true direction, and a recon fine-tuned on them learned to read it.)
Local coincidence (knn_pairs, knn_gate): a detector is read out only if >= 2 of {itself, its 2 nearest neighbours}
are hit (E > 0.001); otherwise E = T = 0.
Used by the run 19 noise scoring (plots/analysis/run19/noise/score_noise.py), the noise fine-tune
(scripts/03c_finetune_recon_noise.py) and the noise-aware Stage 4 (scripts/04_optimize_lbfgs_kernel.py).
E in particles, t in us, positions in m.
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


def knn_pairs(x, y, k=2):
    """(nd, k) indices of each detector's k nearest other detectors in the East-North plane (no gradient)."""
    with torch.no_grad():
        p = torch.stack([x, y], -1)
        d = torch.cdist(p, p).fill_diagonal_(float("inf"))
        return d.topk(k, largest=False).indices


def knn_gate(E, nn, hit_min=1e-3, soft_tau=0.5, gate_tau=0.25):
    """kNN 2-of-3 gate: detector i is read out iff >= 2 of {i, nn[i, 0], nn[i, 1]} have E > hit_min.
    E: (B, nd) particles; nn: (nd, 2) from knn_pairs. Returns (hard, st): hard is the exact 0/1 gate; st has the
    same forward value but the gradient of a soft gate (straight-through), for training:
        s = sigmoid((ln E - ln hit_min) / soft_tau),  count = s_i + s_nn1 + s_nn2,  g = sigmoid((count - 1.5) / gate_tau)."""
    h = (E > hit_min).to(E.dtype)
    hard = ((h + h[:, nn[:, 0]] + h[:, nn[:, 1]]) >= 2).to(E.dtype)
    s = torch.sigmoid((torch.log(E.clamp(min=1e-12)) - torch.log(torch.tensor(hit_min, dtype=E.dtype))) / soft_tau)
    g = torch.sigmoid((s + s[:, nn[:, 0]] + s[:, nn[:, 1]] - 1.5) / gate_tau)
    return hard, hard + g - g.detach()
