"""Differentiable stand-in for analytic_reco.run_reco_v1 (TamboDirReco v1), for gradient-based layout optimisation.

v1 fits the arrival times with an iterative minimiser, which gives the direction but no gradient with respect to the
detector positions. Here the same two chi^2 fits are solved with a fixed number of Gauss-Newton steps, plain tensor
algebra that autograd differentiates through:

  1. seed: v1's closed-form weighted linear fit c t = s . x + c t0, turned into a unit direction on the physical side
     of the detector plane (analytic_reco._lsq_seed);
  2. plane front: residual c (t - t0) - x . d, weights 1 / (km^2 n) (v1 / Julia: sigma = 1 km sqrt(n));
  3. curved front: residual (t - t0) - x . d / c - a r^2, r the distance from the axis through the core, the core
     fixed at the count-weighted hit centre, weights n (sigma_t = 1 us / sqrt(n)), a in [0, 5] us/km^2.

Hits are soft for the gradient: w = sigmoid(kappa (mu - 1)) for timed detectors (v1: mu >= 1 and a time), with the
count clamped to >= 1 inside the weights so that faint detectors cannot dominate the 1 / n plane weights. A shower
counts as reconstructible with a soft hit sum >= 2.5 (v1: >= 3 hits); the others get no direction (the caller scores
them as failures). Positions, times and counts are those of score_all.py's RECO=v1 path: x (B, nd, 3) East, North,
Up [m]; t [us]; mu expected particles (2 m^2).
"""
import torch
import torch.nn.functional as F

from .analytic_reco import C_LIGHT, KM, A_UNIT, angles

DTYPE = torch.float64           # normal equations (4 x 4 per shower)


def _dirs(th, ph):
    """unit direction and its derivatives in theta and phi"""
    st, ct, sp, cp = th.sin(), th.cos(), ph.sin(), ph.cos()
    d = torch.stack([st * cp, st * sp, ct], -1)
    dth = torch.stack([ct * cp, ct * sp, -st], -1)
    dph = torch.stack([-st * sp, st * cp, torch.zeros_like(st)], -1)
    return d, dth, dph


def _gn_update(J, r, w, lam):
    """damped Gauss-Newton step -(J^T W J + lam diag)^-1 J^T W r, per shower"""
    Jw = J * w[..., None]
    H = torch.einsum("bni,bnj->bij", Jw, J).to(DTYPE)
    g = torch.einsum("bni,bn->bi", Jw, r).to(DTYPE)
    D = H.diagonal(dim1=-2, dim2=-1)
    eye = torch.eye(H.shape[-1], device=H.device, dtype=DTYPE)
    A = H + lam * torch.diag_embed(D) + 1e-12 * (D.sum(-1)[:, None, None] + 1.0) * eye
    return -torch.linalg.solve(A, g[..., None])[..., 0].to(J.dtype)


def seed(x, t, w_plane, n_plane):
    """v1's closed-form start (analytic_reco._lsq_seed), differentiable: (theta, phi, t0)"""
    A = torch.cat([x, torch.full_like(t[..., None], C_LIGHT)], -1).to(DTYPE)
    w = w_plane.to(DTYPE)
    AtA = torch.einsum("bni,bn,bnj->bij", A, w, A)
    ridge = 1e-9 * AtA.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12)
    AtA = AtA + (ridge[:, None, None] + 1e-12) * torch.eye(4, device=x.device, dtype=DTYPE)
    Atb = torch.einsum("bni,bn,bn->bi", A, w, C_LIGHT * t.to(DTYPE))
    sol = torch.linalg.solve(AtA, Atb[..., None])[..., 0].to(x.dtype)
    s, t0 = sol[:, :3], sol[:, 3]
    sn = (s * n_plane).sum(-1)
    disc = sn ** 2 - ((s * s).sum(-1) - 1)
    pos = disc > 0
    root = torch.sqrt(torch.where(pos, disc, torch.ones_like(disc)))          # no sqrt(0) in the backward pass
    d = torch.where(pos[:, None], s - (sn + root)[:, None] * n_plane, s)
    th, ph = angles(F.normalize(d, dim=-1))
    return th, ph, t0


def fit(x, t, mu, n_plane, kappa=10.0, n_plane_steps=5, n_time_steps=5, lam=1e-3, max_dang=0.3, params=False):
    """direction (B, 3) and a validity mask (B,) for the showers of one layout; x (B, nd, 3) [m], t (B, nd) [us]
    (0 = no time), mu (B, nd) expected particles"""
    x = x - x.mean(1, keepdim=True)                                   # layout centre (analytic_reco._centre)
    timed = (t > 0).to(x.dtype)
    w = torch.sigmoid(kappa * (mu - 1.0)) * timed                    # soft hit
    n = mu.clamp(min=1.0)
    w_plane = w / (KM ** 2 * n)
    w_time = w * n
    valid = (w.sum(-1) >= 2.5).detach()
    th, ph, t0 = seed(x, t, w_plane, n_plane)
    for _ in range(n_plane_steps):                                    # plane front
        d, dth, dph = _dirs(th, ph)
        r = C_LIGHT * (t - t0[:, None]) - (x * d[:, None]).sum(-1)
        J = torch.stack([-(x * dth[:, None]).sum(-1), -(x * dph[:, None]).sum(-1),
                         torch.full_like(r, -C_LIGHT)], -1)
        step = _gn_update(J, r, w_plane, lam)
        th = th + step[:, 0].clamp(-max_dang, max_dang)
        ph = ph + step[:, 1].clamp(-max_dang, max_dang)
        t0 = t0 + step[:, 2]
    core = (x * (w * mu)[..., None]).sum(1) / (w * mu).sum(1, keepdim=True).clamp(min=1e-9)
    v = x - core[:, None]
    a = torch.ones_like(th)                                            # us / km^2, v1's start
    for _ in range(n_time_steps):                                     # curved front
        d, dth, dph = _dirs(th, ph)
        vd = (v * d[:, None]).sum(-1)
        r2 = (v * v).sum(-1) - vd ** 2
        rho = (t - t0[:, None]) - (x * d[:, None]).sum(-1) / C_LIGHT - a[:, None] * A_UNIT * r2
        jth = -(x * dth[:, None]).sum(-1) / C_LIGHT + 2 * a[:, None] * A_UNIT * vd * (v * dth[:, None]).sum(-1)
        jph = -(x * dph[:, None]).sum(-1) / C_LIGHT + 2 * a[:, None] * A_UNIT * vd * (v * dph[:, None]).sum(-1)
        J = torch.stack([jth, jph, -torch.ones_like(rho), -A_UNIT * r2], -1)
        step = _gn_update(J, rho, w_time, lam)
        th = th + step[:, 0].clamp(-max_dang, max_dang)
        ph = ph + step[:, 1].clamp(-max_dang, max_dang)
        t0 = t0 + step[:, 2]
        a = (a + step[:, 3]).clamp(0.0, 5.0)
    d, _, _ = _dirs(th, ph)
    if params:              # for comparisons with analytic_reco (centred coordinates, a in us/m^2)
        return d, valid, dict(th=th, ph=ph, t0=t0, a=a * A_UNIT, core=core)
    return d, valid


def fit_safe(x, t, mu, n_plane, **kw):
    """fit() on the reconstructible showers only (the others never enter the graph, so a degenerate shower cannot put
    a NaN into the gradient); showers whose fit is not finite are dropped the same way. Returns (d, valid)."""
    B = x.shape[0]
    d_all = torch.zeros(B, 3, device=x.device, dtype=x.dtype)
    ok = ((torch.sigmoid(kw.get("kappa", 10.0) * (mu - 1.0)) * (t > 0)).sum(-1) >= 2.5).detach()
    for _ in range(2):
        idx = ok.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return d_all, ok
        d, _ = fit(x[idx], t[idx], mu[idx], n_plane, **kw)
        fin = torch.isfinite(d).all(-1).detach()
        if bool(fin.all()):
            d_all = d_all.index_put((idx,), d)
            return d_all, ok
        ok = ok.clone(); ok[idx[~fin]] = False
    return d_all, torch.zeros_like(ok)


class AnalyticRecon:
    """Drop-in for the DeepSets recon in objective.utility_of_xy: call bind(kernel, primaries) per sub-batch, then
    recon(features) -> (B, 4) in the primary encoding (direction, log-energy). The energy column is the true one (the
    analytic fit has none; the trigger utility does not use it). Showers without a fit get the reversed true direction
    (180 deg, no gradient)."""

    def __init__(self, surface, n_plane, **fit_kw):
        self.surface, self.n_plane, self.fit_kw = surface, n_plane, fit_kw
        self.kern = self.prim = None

    def bind(self, kern, prim):
        self.kern, self.prim = kern, prim

    def __call__(self, feats):
        e, nth = feats[0, :, 0], feats[0, :, 1]                     # one layout for the whole sub-batch
        B = feats.shape[0]
        x = torch.stack([e, nth, self.surface(nth, e)], -1)[None].expand(B, -1, -1)
        t_us = torch.expm1(feats[..., 3].clamp(min=0.0))
        d, ok = fit_safe(x, t_us, self.kern.mu, self.n_plane, **self.fit_kw)
        dtrue = F.normalize(self.prim[:, :3], dim=-1).detach()
        d = torch.where(ok[:, None], d, -dtrue)
        return torch.cat([d, self.prim[:, 3:4]], -1)
