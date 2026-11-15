"""Detector arrival times with shower-front structure (TAMBOPT_TIME_MODEL=mean or lead).

mean: T = t_plane + <lag> + N(0, sqrt(var / mu + jitter^2)), <lag> and var the mean and variance of the lags of the
points the kernel reads at the detector (its ~8-nearest-neighbour group, kernel weight only, not energy): the group's
mean arrival, with the error of a mean of mu particles. Species mixed by mu_s.

edge: T = t_plane + F^-1(q) + jitter, F the group's own lag CDF (kernel weight only, on a log grid of lags, species
mixed by mu_s), q = 1 - (1 - u)^(1 / mu): the leading edge of mu particles drawn from the group, no shape assumed;
F^-1 interpolated in log lag between grid points (differentiable in the weights, mu and t_plane).

lead (below): leading edge of mu log-normal lags.

The kernel's default T (modules/showers/kernel.py) is the earliest detected point carried to the detector's depth:
the bright core point clears the threshold everywhere, so every detector reads the same point and T is an exact
plane front at c, with no curvature, thickness or count dependence. Here, per detector:

  * the points the kernel reads there give the deposit-weighted mean m and spread s of log lag, the lag being a
    point's time behind its plane's front (GetCounts_planeaware(moments=...)): curvature and thickness of the
    actual shower at that distance from the axis;
  * species are mixed by their expected detected particles mu_s (moment matching in log lag);
  * the detector reports the leading edge of n ~ mu particles with log-normal lags: the first of mu draws,
    reparameterised as lag = exp(m + s z), z = Phi^-1(1 - (1 - u)^(1 / mu)), u ~ U(0, 1) (mu clamped >= 1);
  * plus Gaussian electronics jitter (TAMBOPT_TIME_JITTER_NS, default 2 ns) and an extra spread in log lag for the
    10 m pixel averaging of the C8 times (TAMBOPT_TIME_PIX, default 0; scripts/paper2/fit_delays.py).

T = t_plane + lag + jitter at detectors the default T calls timed (some point detected), 0 elsewhere, so the hit
pattern is unchanged. Differentiable in the detector position through t_plane (depth carried at c), m, s (kernel
weights) and mu. Draws come from one generator per device, seeded by TAMBOPT_TIME_SEED (default 0); reseed() resets it.
"""
import math
import os

import torch

MODEL = os.environ.get("TAMBOPT_TIME_MODEL", "min")          # "min" = the kernel's default T; "mean", "edge" or "lead"
JITTER_NS = float(os.environ.get("TAMBOPT_TIME_JITTER_NS", "2.0"))
PIX = float(os.environ.get("TAMBOPT_TIME_PIX", "0.0"))         # extra spread in log lag (pixel averaging)
SEED = int(os.environ.get("TAMBOPT_TIME_SEED", "0"))
_GENS = {}
LAST = {}                                                     # last call's internals (gradient checks)


def enabled():
    return MODEL in ("mean", "lead", "edge")


_SEED = [SEED]                                               # seed for generators created from now on


def _gen(device):
    k = str(device)
    if k not in _GENS:
        _GENS[k] = torch.Generator(device=device).manual_seed(_SEED[0])
    return _GENS[k]


def reseed(seed=SEED):
    """reset every generator, including ones created later, to `seed`"""
    _SEED[0] = seed
    for g in _GENS.values():
        g.manual_seed(seed)


def mix(moments, mus):
    """per-species moments (dicts with m, s) and expected particles mu_s -> combined (m, s, mu), log-lag moment
    matching weighted by mu_s"""
    mu = sum(mus)
    wt = [q / mu.clamp(min=1e-9) for q in mus]
    m = sum(w * d["m"] for w, d in zip(wt, moments))
    v = sum(w * (d["s"] ** 2 + d["m"] ** 2) for w, d in zip(wt, moments)) - m ** 2
    return m, torch.sqrt(v.clamp(min=0.0) + PIX ** 2 + 1e-12), mu


def leading_edge(t_plane, m, s, mu, timed):
    """(B, nd) arrival times [s]: t_plane + first-of-mu log-normal lag + jitter at timed detectors, 0 elsewhere"""
    g = _gen(m.device)
    u = torch.rand(m.shape, generator=g, device=m.device, dtype=m.dtype).clamp(1e-6, 1 - 1e-6)
    n = mu.clamp(min=1.0)
    q = (1.0 - torch.exp(torch.log1p(-u) / n)).clamp(1e-7, 1 - 1e-7)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
    lag_ns = torch.exp(m + s * z)
    jit = JITTER_NS * torch.randn(m.shape, generator=g, device=m.device, dtype=m.dtype)
    t = t_plane + (lag_ns + jit) * 1e-9
    return torch.where(timed, t.clamp(min=1e-12), torch.zeros_like(t))


def mix_mean(moments, mus):
    """per-species group means / variances and mu_s -> combined (mean, var, mu) of the lag [ns], weighted by mu_s"""
    mu = sum(mus)
    wt = [q / mu.clamp(min=1e-9) for q in mus]
    mean = sum(w * d["mean"] for w, d in zip(wt, moments))
    var = sum(w * (d["var"] + d["mean"] ** 2) for w, d in zip(wt, moments)) - mean ** 2
    return mean, var.clamp(min=0.0), mu


def group_mean(t_plane, mean, var, mu, timed):
    """(B, nd) arrival times [s]: t_plane + group mean lag + N(0, sqrt(var / mu + jitter^2)) at timed detectors"""
    g = _gen(mean.device)
    z = torch.randn(mean.shape, generator=g, device=mean.device, dtype=mean.dtype)
    sd = torch.sqrt(var / mu.clamp(min=1.0) + JITTER_NS ** 2)
    t = t_plane + (mean + sd * z) * 1e-9
    return torch.where(timed, t.clamp(min=1e-12), torch.zeros_like(t))


def group_edge(moments, mus, timed):
    """(B, nd) arrival times [s]: leading edge of mu particles from the mu_s-mixed group lag CDF + jitter"""
    mu = sum(mus)
    wt = [q / mu.clamp(min=1e-9) for q in mus]
    F = sum(w.unsqueeze(1) * d["cdf"] for w, d in zip(wt, moments))                  # (B, K + 1, nd)
    F = F / F[:, -1:].clamp(min=1e-12)
    le = torch.log(moments[0]["edges"])                                              # (K + 1,)
    g = _gen(mu.device)
    u = torch.rand(mu.shape, generator=g, device=mu.device, dtype=mu.dtype).clamp(1e-6, 1 - 1e-6)
    q = (1.0 - torch.exp(torch.log1p(-u) / mu.clamp(min=1.0))).clamp(1e-6, 1 - 1e-6)
    Ft = F.transpose(1, 2).contiguous()                                               # (B, nd, K + 1)
    k = torch.searchsorted(Ft.detach(), q.detach().unsqueeze(-1)).squeeze(-1).clamp(1, Ft.shape[-1] - 1)
    F1 = Ft.gather(-1, k.unsqueeze(-1)).squeeze(-1); F0 = Ft.gather(-1, (k - 1).unsqueeze(-1)).squeeze(-1)
    frac = ((q - F0) / (F1 - F0).clamp(min=1e-9)).clamp(0.0, 1.0)
    LAST.update(k=k.detach(), timed=timed.detach())
    lag_ns = torch.exp(le[k - 1] + frac * (le[k] - le[k - 1]))
    jit = JITTER_NS * torch.randn(mu.shape, generator=g, device=mu.device, dtype=mu.dtype)
    t = moments[0]["t_plane"] + (lag_ns + jit) * 1e-9
    return torch.where(timed, t.clamp(min=1e-12), torch.zeros_like(t))


def arrival(moments, mus, timed):
    """the selected model's (B, nd) arrival times [s] from the per-species kernel moments"""
    if MODEL == "edge":
        return group_edge(moments, mus, timed)
    if MODEL == "mean":
        mean, var, mu = mix_mean(moments, mus)
        return group_mean(moments[0]["t_plane"], mean, var, mu, timed)
    m, s, mu = mix(moments, mus)
    return leading_edge(moments[0]["t_plane"], m, s, mu, timed)
