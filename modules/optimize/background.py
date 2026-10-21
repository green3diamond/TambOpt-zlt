"""Cosmic-ray background in the detectors (SWGO / SWGOLO swgolo148.cpp, HAP-2024-004).

1,826 muons / m^2 / s on detectors of 2 m^2: per detector and shower lambda = 2 * 1.826e-6 * W[ns] in ONE readout
window W shared by all detectors, centred on the shower's earliest timed detector t_ref (observed, no truth). Per window:
    E' = E + N_bg * e_per_bg_muon(r),  N_bg ~ Poisson(lambda),  r = the detector's distance from the shower axis
    T' = min(T, earliest background time),  background times ~ U(t_ref - W/2, t_ref + W/2)
(Until 2026-10-15 the window was centred on each detector's TRUE front arrival, t_ref + (r_i - r_ref) . n_true / c:
background times then carried the true direction, and a recon fine-tuned on them learned to read it.)
Local coincidence (knn_pairs, knn_gate): a detector is read out only if >= 2 of {itself, its 2 nearest neighbours}
are hit (E > 0.001); otherwise E = T = 0.
Used by the run 19 noise scoring (plots/analysis/run19/noise/score_noise.py), the noise fine-tune
(scripts/03c_finetune_recon_noise.py) and the noise-aware Stage 4 (scripts/04_optimize_lbfgs_kernel.py).
t in us, positions in m.

Units (2026-10-21). The kernel's E is NOT a particle count: it is summed kinetic energy [GeV] of CORSIKA 8 cells,
weighted by a 10 m Gaussian with peak 1 and total 2 pi sigma^2 ~ 628 m^2 (modules/showers/kernel.py docstring), i.e.
GeV collected over an effective ~628 m^2. On top, the ring kernel places each super-particle on a ring of radius |t1|
(horizontal transverse offset only, place_clouds_shower) instead of its true distance r = sqrt(t1^2 + t2^2), which
lowers the energy density away from the core: ring / true = ratio_s(r), ~1.0 at the core, 0.6 at 300 m, 0.45 at 1 km.
Per species s, a kernel E at a detector a distance r from the axis therefore means
    particles_s = E * A / (2 pi sigma^2 * KE_s(r)) / ratio_s(r)                         (particles_per_E)
with KE_s(r) the mean kinetic energy per particle at true distance r. Both are VECTORS over r per species, measured on
16 CORSIKA 8 pi- showers (scripts/diagnostics/c8_ring_radius_check.py -> c8_scaling.json next to this file; 28 log
bins 1 m - 10 km, interpolated in log r, clamped at the measured ends). r=None uses the 100 m - 3 km band values.
A background muon is the inverse: one real muon in the 2 m^2 detector reads as E = 1 / particles_per_E('mu', r) in
the same (ring-kernel) units as the signal: ~650 for the band value, ~1,500 at 100 m, ~210 at 1 km
(a MIP view: the background muon counts as one particle, read like a shower muon at that r). axis_distance() gives r per detector and
shower from the kernel's shower frames. Shower muons stand in for cosmic-ray muons. Until 2026-10-21 a background
muon added 1 to E (count units mixed with kernel units); TAMBOPT_BG_UNITS=count restores that.
"""
import json
import math
import os

import torch

from ..constants import SIGMA_SPATIAL

RATE_PER_M2_NS = 1.826e-6
AREA_M2 = 2.0
KERNEL_AREA_M2 = 2.0 * math.pi * SIGMA_SPATIAL ** 2                            # total kernel weight per point, ~628 m^2
BG_UNITS = os.environ.get("TAMBOPT_BG_UNITS", "kernel")                        # "kernel" (default) or "count" (pre 2026-10-21)
SPECIES = ("e", "mu", "gamma")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "c8_scaling.json")) as _f:
    _TAB = json.load(_f)
_R = [r for r in _TAB["r_m"]]


def _table(species, key):
    """(log r, values) of one per-species vector, NaN bins (< 200 particles) dropped."""
    v = _TAB["species"][species][key]
    keep = [i for i, x in enumerate(v) if x == x]
    return [math.log(_R[i]) for i in keep], [v[i] for i in keep]


def _interp(species, key, r, log_values):
    lr, v = _table(species, key)
    lr = torch.as_tensor(lr, dtype=r.dtype, device=r.device)
    v = torch.as_tensor(v, dtype=r.dtype, device=r.device)
    if log_values:
        v = torch.log(v)
    x = torch.log(r.clamp(min=float(lr[0].exp()), max=float(lr[-1].exp())))
    i = torch.searchsorted(lr, x.contiguous(), right=True).clamp(1, len(lr) - 1)
    w = (x - lr[i - 1]) / (lr[i] - lr[i - 1])
    y = v[i - 1] + w * (v[i] - v[i - 1])
    return torch.exp(y) if log_values else y


def ke_per_particle(species, r=None):
    """Mean kinetic energy [GeV] per particle of `species` ('e', 'mu', 'gamma') at true distance r [m] from the axis
    (any tensor shape; log-log interpolation of c8_scaling.json). r=None: 100 m - 3 km band value (e 0.081, mu 3.23,
    gamma 0.043 GeV)."""
    if r is None:
        return _TAB["band_100m_3km"][species]["ke_gev"]
    return _interp(species, "ke_gev", r, log_values=True)


def ring_ratio(species, r=None, ring="t1"):
    """Ring-kernel energy density / true energy density for `species` at distance r: ring="t1" for the current kernel
    (rings at |t1|), "t1x2" for rings at sqrt(2) |t1|. r=None: band value (e 0.59, mu 0.64, gamma 0.60 for t1)."""
    key = f"ring_ratio_{ring}"
    if r is None:
        return _TAB["band_100m_3km"][species][key]
    return _interp(species, key, r, log_values=False)


def particles_per_E(species, r=None, area_m2=AREA_M2, ring="t1"):
    """Expected particles of `species` in a detector of area_m2 per unit of that species' kernel E, at distance r from
    the axis: A / (2 pi sigma^2 KE_s(r)) / ratio_s(r). The per-species scaling for the per-species surrogates."""
    return area_m2 / (KERNEL_AREA_M2 * ke_per_particle(species, r)) / ring_ratio(species, r, ring)


def e_per_bg_muon(r=None):
    """Kernel E of one background muon in a 2 m^2 detector at distance r from the axis: 1 / particles_per_E('mu', r)
    (~650 for the band value; 1 with TAMBOPT_BG_UNITS=count)."""
    if BG_UNITS != "kernel":
        return 1.0
    return 1.0 / particles_per_E("mu", r)


E_PER_BG_MUON = e_per_bg_muon()                                                 # band value, for logs


def axis_distance(e_det, n_det, surface, frames):
    """Distance [m] of every detector from every shower's axis, (B, nd), from the kernel's shower frames
    (KernelDualLabels.frames, (B, 12)) and the surface; the kernel's own ring coordinate (detectors_in_frames)."""
    from ..data.dataset_builder import detectors_in_frames
    with torch.no_grad():
        return detectors_in_frames(e_det, n_det, surface, frames, ring=True)[0].float()


def window_centres(t, timed, t_fallback):
    """Readout-window centre per detector [us], (B, nd): the shower's earliest timed detector, the same for every
    detector; showers with no timed detector use t_fallback. t, timed: (B, nd)."""
    big = torch.finfo(t.dtype).max
    t_ref = torch.where(timed, t, torch.full_like(t, big)).min(1).values
    t_ref = torch.where(timed.any(1), t_ref, torch.full_like(t_ref, t_fallback))
    return t_ref[:, None].expand_as(t)


def add_background(E, t, timed, t_front, W, gen, r_axis=None):
    """E, t, timed, t_front (window centres, window_centres()): (B, nd); W: window [ns], a float or a (B,) tensor
    (one window per shower); r_axis: (B, nd) distance from the shower axis [m] (axis_distance()) for the muon
    scaling KE_mu(r), or None for the band value.
    Returns (E', T', N_bg, has_bg); N_bg in muons, E' in kernel units (e_per_bg_muon(r_axis) per muon, module docstring)."""
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
    e_mu = e_per_bg_muon(r_axis)
    return E + Nb * (e_mu.to(E) if torch.is_tensor(e_mu) else e_mu), Tn, Nb, has_bg


def knn_pairs(x, y, k=2):
    """(nd, k) indices of each detector's k nearest other detectors in the East-North plane (no gradient)."""
    with torch.no_grad():
        p = torch.stack([x, y], -1)
        d = torch.cdist(p, p).fill_diagonal_(float("inf"))
        return d.topk(k, largest=False).indices


def knn_gate(E, nn, hit_min=1e-3, soft_tau=0.5, gate_tau=0.25):
    """kNN 2-of-3 gate: detector i is read out iff >= 2 of {i, nn[i, 0], nn[i, 1]} have E > hit_min.
    E: (B, nd) kernel units (GeV over ~628 m^2, not particles); nn: (nd, 2) from knn_pairs. Returns (hard, st): hard is the exact 0/1 gate; st has the
    same forward value but the gradient of a soft gate (straight-through), for training:
        s = sigmoid((ln E - ln hit_min) / soft_tau),  count = s_i + s_nn1 + s_nn2,  g = sigmoid((count - 1.5) / gate_tau)."""
    h = (E > hit_min).to(E.dtype)
    hard = ((h + h[:, nn[:, 0]] + h[:, nn[:, 1]]) >= 2).to(E.dtype)
    s = torch.sigmoid((torch.log(E.clamp(min=1e-12)) - torch.log(torch.tensor(hit_min, dtype=E.dtype))) / soft_tau)
    g = torch.sigmoid((s + s[:, nn[:, 0]] + s[:, nn[:, 1]] - 1.5) / gate_tau)
    return hard, hard + g - g.detach()
