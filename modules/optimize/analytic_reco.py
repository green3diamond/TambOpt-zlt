"""Analytical direction reconstruction: a batched PyTorch port of TamboDirReco.jl (src/reco.jl, src/utils.jl).

Same fits, likelihoods and pipelines, applied to many showers at once on the GPU:

  plane_chisq     plane-front chi^2 over hit times               fits theta, phi, t0
  timing_llh      curved front: delay = a r^2, core fixed         fits theta, phi, t0, a
  ldf_llh         Poisson LDF over hit and unhit modules          fits core_x, core_y, log_A, beta (direction fixed)
  joint_llh       LDF + timing                                    fits all eight
  run_reco_v1     plane x2 -> timing (core = count-weighted mean of the hits)
  run_reco_v2     plane x2 -> timing -> [LDF -> timing] x 4
  run_reco_v3     plane x2 -> timing -> LDF -> timing -> joint

Conventions as in the Julia code: dir(theta, phi) = (sin th cos ph, sin th sin ph, cos th) is the PROPAGATION
direction (arrival time grows along it, c (t - t0) = dir . x); a direction is physical when it goes into the detector
plane (dir . n_plane <= 0, n_plane the outward plane normal), otherwise the per-hit cost is multiplied by 100;
sigma = 1 km sqrt(n) (plane), 1 us / sqrt(n) (timing); LDF Q(r) = (r/100 m)^(-beta - 0.35 log10(r/100 m)), y_pred =
exp(log_A) Q, Poisson log-likelihood. Units: metres, microseconds; a in us/km^2.

Differences from the Julia code, all in the minimiser, not in the model:
  * Minuit/MIGRAD -> torch L-BFGS (strong Wolfe), one batch of showers at a time; each shower's cost only depends on
    its own parameters, so the summed cost separates.
  * Minuit parameter limits -> the same sin transform Minuit uses internally (p = lo + (hi - lo)(sin q + 1)/2), on
    a, log_A, beta, core_x, core_y; theta and phi are left free (they are periodic; Minuit's limits only keep them
    in range).
  * The first plane fit is seeded with the unconstrained linear least-squares plane (optional, seed="lsq"), not
    theta = 90 deg, phi = 200 deg, t0 = 10 us: those defaults are for the Julia simulation's frame, not ours.
  * Hits come from the kernel (per detector: position, leading-edge time, expected particle count n), not from
    CORSIKA module hits; a detector is "hit" when the caller's mask says so (e.g. n >= 1).
  * "valid" = finite parameters and a small final gradient, instead of mini.is_valid.
"""
import math
import torch

C_LIGHT = 299.792458          # m / us
KM = 1000.0
A_UNIT = 1.0 / KM ** 2        # 1 us/km^2 in us/m^2


def dir_vec(th, ph):
    """get_dir_vec: unit vector from polar th and azimuth ph [rad]."""
    return torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], -1)


def angles(v):
    """get_angles: (theta, phi) [rad] of a vector, phi in [0, 2 pi)."""
    th = torch.atan2(torch.hypot(v[..., 0], v[..., 1]), v[..., 2])
    return th, torch.remainder(torch.atan2(v[..., 1], v[..., 0]), 2 * math.pi)


def shower_axes(d):
    """Two unit vectors orthogonal to d (B, 3): the transverse plane of the shower frame. The Julia code rotates with
    get_R(flip_dir(theta, phi)); only distances in that plane are used, so any orthonormal pair gives the same r."""
    ref = torch.where((d[..., 2:3].abs() < 0.9), d.new_tensor([0.0, 0.0, 1.0]), d.new_tensor([1.0, 0.0, 0.0]))
    e1 = torch.nn.functional.normalize(torch.cross(ref.expand_as(d), d, dim=-1), dim=-1)
    return e1, torch.cross(d, e1, dim=-1)


def _bounded(q, lo, hi):
    return lo + (hi - lo) * (torch.sin(q) + 1) / 2


def _unbounded(p, lo, hi):
    return torch.asin((2 * (p - lo) / (hi - lo) - 1).clamp(-1 + 1e-6, 1 - 1e-6))


def _penalty(d, n_plane):
    """A = 1 for directions into the detector plane, 100 otherwise (is_valid_air_shower_dir)."""
    return torch.where((d * n_plane).sum(-1) <= 0, 1.0, 100.0)


# ── per-shower costs (B,), as -2 log L / chi^2 summed over each shower's hits ─────────────────────────────────
def plane_cost(th, ph, t0, x, t, n, hit, n_plane):
    d = dir_vec(th, ph)
    resid = C_LIGHT * (t - t0[:, None]) - (x * d[:, None]).sum(-1)
    sig2 = (KM ** 2) * n.clamp(min=1e-9)
    return _penalty(d, n_plane) * ((resid ** 2 / sig2) * hit).sum(-1)


def timing_cost(th, ph, t0, a, core, x, t, n, hit, n_plane):
    """a in us/m^2; core (B, 3) any point on the axis."""
    d = dir_vec(th, ph)
    e1, e2 = shower_axes(d)
    v = x - core[:, None]
    r2 = (v * e1[:, None]).sum(-1) ** 2 + (v * e2[:, None]).sum(-1) ** 2
    delay = (t - t0[:, None]) - (x * d[:, None]).sum(-1) / C_LIGHT
    chi = (delay - a[:, None] * r2) ** 2 * n                 # sigma_t = 1 us / sqrt(n)
    return _penalty(d, n_plane) * (chi * hit).sum(-1)


def q_ldf(r, beta, kappa=0.35, r_ref=100.0):
    """(r/r_ref)^(-beta - kappa log10(r/r_ref)), evaluated in log space and clamped so it cannot overflow."""
    u = (r / r_ref).clamp(min=1e-3)
    lg = (-beta[:, None] - kappa * torch.log10(u)) * torch.log(u)
    return torch.exp(lg.clamp(-60.0, 60.0))


def ldf_cost(th, ph, cx, cy, log_A, beta, x, n, det):
    """-2 Poisson log L over every detector in `det` (hit and unhit; unhit y = n = 0); (cx, cy) the core in the
    shower's transverse plane [m]."""
    d = dir_vec(th, ph)
    e1, e2 = shower_axes(d)
    r = torch.hypot((x * e1[:, None]).sum(-1) - cx[:, None], (x * e2[:, None]).sum(-1) - cy[:, None])
    y_pred = log_A.exp()[:, None] * q_ldf(r, beta)
    llh = n * torch.log(y_pred.clamp(min=1e-30)) - y_pred
    return (-2 * llh * det).sum(-1)


def _core_3d(th, ph, cx, cy):
    d = dir_vec(th, ph)
    e1, e2 = shower_axes(d)
    return cx[:, None] * e1 + cy[:, None] * e2


def _core_xy(th, ph, core):
    d = dir_vec(th, ph)
    e1, e2 = shower_axes(d)
    return (core * e1).sum(-1), (core * e2).sum(-1)


# ── minimiser ─────────────────────────────────────────────────────────────────────────────────────────────────
def _minimise(params, cost_fn, iters):
    """L-BFGS on the summed per-shower cost; returns the per-shower cost and gradient norm at the end."""
    params = [p.detach().clone().requires_grad_(True) for p in params]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12, history_size=20)

    def closure():
        opt.zero_grad()
        c = cost_fn(*params).sum()
        c.backward()
        return c
    opt.step(closure)
    with torch.enable_grad():
        c = cost_fn(*params)
        g = torch.autograd.grad(c.sum(), params, allow_unused=True)
    gn = torch.sqrt(sum((gi ** 2) if gi.dim() == 1 else (gi ** 2).sum(-1) for gi in g if gi is not None))
    return [p.detach() for p in params], c.detach(), gn


def _lsq_seed(x, t, n, hit, n_plane):
    """Weighted linear fit c t = s . x + c t0 (s free) as the start. On a (near-)flat detector plane the times fix
    only the in-plane part of the direction, and an in-plane start is a saddle of the constrained fit; so the start
    is s + k n_plane with the k < 0 (into the plane, the physical side) that makes it a unit vector, or s/|s| when
    |s| >= 1."""
    w = hit / (KM ** 2 * n.clamp(min=1e-9))
    A = torch.cat([x, torch.full_like(t[..., None], C_LIGHT)], -1)            # (B, nd, 4)
    A, w = A.double(), w.double()
    AtA = torch.einsum("bni,bn,bnj->bij", A, w, A)
    ridge = 1e-9 * AtA.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-12)       # keeps < 4-hit showers solvable
    AtA = AtA + (ridge[:, None, None] + 1e-12) * torch.eye(4, device=x.device, dtype=A.dtype)
    Atb = torch.einsum("bni,bn,bn->bi", A, w, C_LIGHT * t.double())
    sol = torch.linalg.lstsq(AtA, Atb[..., None]).solution[..., 0].to(x.dtype)
    s = sol[:, :3]
    sn = (s * n_plane).sum(-1)
    disc = sn ** 2 - ((s * s).sum(-1) - 1)
    k = -sn - torch.sqrt(disc.clamp(min=0))
    d = torch.where((disc > 0)[:, None], s + k[:, None] * n_plane, s)
    th, ph = angles(torch.nn.functional.normalize(d, dim=-1))
    return th, ph, sol[:, 3]


def fit_plane(x, t, n, hit, n_plane, th=None, ph=None, t0=None, iters=200):
    if th is None:
        th, ph, t0_s = _lsq_seed(x, t, n, hit, n_plane)
        t0 = t0_s if t0 is None else t0
    (th, ph, t0), c, g = _minimise([th, ph, t0], lambda a, b, c_: plane_cost(a, b, c_, x, t, n, hit, n_plane), iters)
    return dict(th=th, ph=ph, t0=t0), c, g


def fit_timing(x, t, n, hit, n_plane, th, ph, t0, core, a=None, iters=200):
    a = torch.full_like(th, 1.0) if a is None else a / A_UNIT          # us/km^2, limits (0, 5)
    qa = _unbounded(a, 0.0, 5.0)

    def cost(th_, ph_, t0_, qa_):
        return timing_cost(th_, ph_, t0_, _bounded(qa_, 0.0, 5.0) * A_UNIT, core, x, t, n, hit, n_plane)
    (th, ph, t0, qa), c, g = _minimise([th, ph, t0, qa], cost, iters)
    return dict(th=th, ph=ph, t0=t0, a=_bounded(qa, 0.0, 5.0) * A_UNIT, core=core), c, g


LIM = dict(cx=(-3 * KM, 3 * KM), cy=(-3 * KM, 3 * KM), log_A=(-5.0, 15.0), beta=(0.0, 5.0))


def fit_ldf(x, n, det, th, ph, core, log_A=None, beta=None, iters=200):
    cx, cy = _core_xy(th, ph, core)
    log_A = torch.full_like(th, 1.0) if log_A is None else log_A
    beta = torch.full_like(th, 2.5) if beta is None else beta
    q = [_unbounded(v, *LIM[k]) for k, v in (("cx", cx), ("cy", cy), ("log_A", log_A), ("beta", beta))]

    def cost(qx, qy, qA, qb):
        return ldf_cost(th, ph, _bounded(qx, *LIM["cx"]), _bounded(qy, *LIM["cy"]), _bounded(qA, *LIM["log_A"]),
                        _bounded(qb, *LIM["beta"]), x, n, det)
    q, c, g = _minimise(q, cost, iters)
    cx, cy, log_A, beta = (_bounded(v, *LIM[k]) for v, k in zip(q, ("cx", "cy", "log_A", "beta")))
    return dict(cx=cx, cy=cy, log_A=log_A, beta=beta, core=_core_3d(th, ph, cx, cy)), c, g


def fit_joint(x, t, n, hit, det, n_plane, th, ph, t0, a, core, log_A, beta, iters=300):
    cx, cy = _core_xy(th, ph, core)
    q = [th, ph, t0, _unbounded(a / A_UNIT, 0.0, 5.0)] + [_unbounded(v, *LIM[k]) for k, v in
                                                            (("cx", cx), ("cy", cy), ("log_A", log_A), ("beta", beta))]

    def unpack(th_, ph_, t0_, qa, qx, qy, qA, qb):
        return (th_, ph_, t0_, _bounded(qa, 0.0, 5.0) * A_UNIT, _bounded(qx, *LIM["cx"]), _bounded(qy, *LIM["cy"]),
                _bounded(qA, *LIM["log_A"]), _bounded(qb, *LIM["beta"]))

    def cost(*qq):
        th_, ph_, t0_, a_, cx_, cy_, lA, b = unpack(*qq)
        return (ldf_cost(th_, ph_, cx_, cy_, lA, b, x, n, det)
                + timing_cost(th_, ph_, t0_, a_, _core_3d(th_, ph_, cx_, cy_), x, t, n, hit, n_plane))
    q, c, g = _minimise(q, cost, iters)
    th, ph, t0, a, cx, cy, log_A, beta = unpack(*q)
    return dict(th=th, ph=ph, t0=t0, a=a, cx=cx, cy=cy, log_A=log_A, beta=beta, core=_core_3d(th, ph, cx, cy)), c, g


def get_core(x, n, hit):
    """Count-weighted mean position of the hits."""
    w = n * hit
    return (x * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-9)


def _centre(x):
    """Coordinates relative to the layout centre (TamboDirReco's tambo_center): the +-3 km core limits are about the
    array, not the site origin. A shift only moves t0, so the direction fits are unchanged."""
    return x - x.mean(1, keepdim=True)


def _keep(new, old):
    """Per shower: the new fit where its direction is finite, else the previous stage's (a failed LDF / joint fit
    falls back to the timing fit instead of failing the shower)."""
    ok = torch.isfinite(dir_vec(new["th"], new["ph"])).all(-1)
    out = dict(new)
    for k, v in old.items():
        if k in new and torch.is_tensor(v) and v.shape[:1] == ok.shape:
            m = ok.view(-1, *([1] * (v.dim() - 1)))
            out[k] = torch.where(m, new[k], v)
    return out


def _finish(res, g, hit, min_hits, gtol):
    d = dir_vec(res["th"], res["ph"])
    ok = (hit.sum(-1) >= min_hits) & torch.isfinite(d).all(-1)
    return dict(res, dir=d, valid=ok, converged=g < gtol)


@torch.no_grad()
def run_reco_v1(x, t, n, hit, n_plane, min_hits=3, gtol=1e-2):
    """x (B, nd, 3) detector positions [m]; t (B, nd) leading-edge hit times [us]; n (B, nd) particle counts;
    hit (B, nd) 1/0 mask; n_plane (3,) outward unit normal of the detector plane. Returns a dict with dir (B, 3),
    th, ph, t0, a, core, valid."""
    x = _centre(x)
    hit = hit.float()
    with torch.enable_grad():
        p, _, _ = fit_plane(x, t, n, hit, n_plane)
        p, _, _ = fit_plane(x, t, n, hit, n_plane, t0=p["t0"])
        core = get_core(x, n, hit)
        p, _, g = fit_timing(x, t, n, hit, n_plane, p["th"], p["ph"], p["t0"], core)
    return _finish(p, g, hit, min_hits, gtol)


@torch.no_grad()
def run_reco_v2(x, t, n, hit, n_plane, det=None, min_hits=3, gtol=1e-2):
    """As v1, then alternating LDF (hit + unhit modules in `det`, default all) and timing fits."""
    x = _centre(x)
    hit = hit.float(); det = torch.ones_like(hit) if det is None else det.float()
    with torch.enable_grad():
        p, _, _ = fit_plane(x, t, n, hit, n_plane)
        p, _, _ = fit_plane(x, t, n, hit, n_plane, t0=p["t0"])
        p, _, _ = fit_timing(x, t, n, hit, n_plane, p["th"], p["ph"], p["t0"], get_core(x, n, hit))
        l, _, _ = fit_ldf(x, n, det, p["th"], p["ph"], get_core(x, n, hit))
        p2, _, g = fit_timing(x, t, n, hit, n_plane, p["th"], p["ph"], p["t0"], l["core"], a=p["a"])
        p = _keep(p2, p)
        for _ in range(3):
            l, _, _ = fit_ldf(x, n, det, p["th"], p["ph"], p["core"], l["log_A"], l["beta"])
            p2, _, g = fit_timing(x, t, n, hit, n_plane, p["th"], p["ph"], p["t0"], l["core"], a=p["a"])
            p = _keep(p2, p)
    return _finish(dict(l, **p), g, hit, min_hits, gtol)


@torch.no_grad()
def run_reco_v3(x, t, n, hit, n_plane, det=None, min_hits=3, gtol=1e-2):
    """Plane x2 -> timing -> LDF -> timing -> joint LDF + timing fit over all eight parameters."""
    x = _centre(x)
    hit = hit.float(); det = torch.ones_like(hit) if det is None else det.float()
    with torch.enable_grad():
        p, _, _ = fit_plane(x, t, n, hit, n_plane)
        p, _, _ = fit_plane(x, t, n, hit, n_plane, t0=p["t0"])
        p, _, _ = fit_timing(x, t, n, hit, n_plane, p["th"], p["ph"], p["t0"], get_core(x, n, hit))
        l, _, _ = fit_ldf(x, n, det, p["th"], p["ph"], get_core(x, n, hit))
        p2, _, _ = fit_timing(x, t, n, hit, n_plane, p["th"], p["ph"], p["t0"], l["core"], a=p["a"])
        p = _keep(p2, p)
        j, _, g = fit_joint(x, t, n, hit, det, n_plane, p["th"], p["ph"], p["t0"], p["a"], p["core"], l["log_A"],
                            l["beta"])
        j = _keep(j, p)
    return _finish(j, g, hit, min_hits, gtol)


RUNS = {"v1": run_reco_v1, "v2": run_reco_v2, "v3": run_reco_v3}


def plane_normal(points):
    """Outward (Up > 0) unit normal of the least-squares plane through (N, 3) points: the detector-plane direction."""
    p = points - points.mean(0)
    n = torch.linalg.svd(p, full_matrices=False).Vh[-1]
    return n if n[2] >= 0 else -n
