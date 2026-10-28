"""Gradients of expected observables through the per-plane energy sampler: d E[obs] / d(log10 E, cos zenith) by
autodiff with common random numbers, against local regression slopes on CORSIKA test showers.

The sampler is re-expressed in torch: fixed normal draws -> copula -> margins (ppf) -> profile curve at the plane
depths -> per-plane correction and AR(1) noise. Draws whose curve ends inside the planes are left out, as the sampler
redraws them. cos zenith enters through the laws and through the plane depths (chain rule; dX/d cos zenith by a
central difference of the depth table). Gate: autodiff should equal a narrow central finite difference of the same
draws; the pathwise gradient leaves out planes switching on as the start depth moves, so a wide finite difference
(which includes them on average) is compared with CORSIKA as well, and the gate is recorded, not enforced.

Rule (study_plan.md 1g): PASS per class if the sampler's slope is within 2 se of CORSIKA's at >= 80% of the
evaluation points (species x observable x derivative x point).
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch

from symbolic import logs
from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere as atm, fit_profiles, longitudinal, planes, sampler

FOLDER = "gradients"
N_MC = 20000
N_BATCH = 10
GRID_L = (5.25, 5.5, 5.75)
GRID_U_Q = (0.25, 0.5, 0.75)             # quantiles of cos zenith among the class's test showers
WINDOW = (0.25, 0.1)                     # half-widths in log10 E and cos zenith of the CORSIKA regression
H_FD = 1e-3                              # depth table step in cos zenith
H_GATE = 1e-4                            # finite difference that should equal autodiff
H_WIDE = 0.02                            # finite difference wide enough to include planes switching on at the start
OBS = ("log10 total", "mean plane")


def depths(u):
    return atm.plane_depths(np.degrees(np.arccos(np.atleast_1d(u))), N_PLANES)[0]


def observables(E):
    """Per shower and species: log10(1 + energy on planes 2-24) and the energy-weighted mean plane (1-based)."""
    Ek = E[..., 1:]
    k = torch.arange(2, E.shape[-1] + 1, dtype=E.dtype)
    tot = Ek.sum(-1)
    return {"log10 total": torch.log10(1 + tot), "mean plane": (Ek * k).sum(-1) / tot.clamp(min=1e-12)}


class TorchSampler:
    """One class's fitted sampler with fixed random numbers, as a differentiable function of (log10 E, cos zenith,
    plane depths)."""

    def __init__(self, model, n, seed):
        self.m = model
        g = np.random.default_rng(seed)
        L = np.linalg.cholesky(model["C"])
        self.z = torch.as_tensor(g.normal(size=(n, len(model["C"]))) @ L.T)
        self.noise = {s: torch.as_tensor(sampler.ar1(n, N_PLANES, model["planes"][s]["rho"], g))
                      for s in SPECIES_NAMES}

    def latent(self, l, u):
        d = self.m["design"]
        a, b = (l - d.m[0]) / d.s[0], (u - d.m[1]) / d.s[1]
        D = torch.stack([torch.ones_like(a), a, b, a * b]).reshape(1, 4).expand(len(self.z), 4)
        return torch.stack([mg.loc + mg.scale * mg._g_inv(self.z[:, j], mg._params(D, mg.W))
                            for j, mg in enumerate(self.m["margins"])], 1)

    def energies(self, l, u, X):
        """Energy per plane per species (n, planes) and the rows whose curve ends inside the planes."""
        lat = self.latent(l, u)
        out, ends = {}, torch.zeros(len(lat), dtype=torch.bool)
        for i, s in enumerate(SPECIES_NAMES):
            b = lat[:, sampler.BLOCK * i: sampler.BLOCK * (i + 1)]
            xmax, L, R, ln_n = b[:, 1], torch.exp(b[:, 2]), b[:, 3], (b[:, 0] + l) * np.log(10)
            ends |= (R < 0) & (xmax - L / R.where(R < 0, -torch.ones_like(R)) < X[-1])
            y = ln_n[:, None] + longitudinal.shape_torch(X[None, :], xmax[:, None], L[:, None], R[:, None])
            p = self.m["planes"][s]
            y = y + torch.as_tensor(p["bias"]) + torch.as_tensor(p["sd"]) * self.noise[s]
            live = torch.isfinite(y) & (X[None, :] >= b[:, -1:])
            out[s] = torch.where(live, torch.exp(torch.where(live, y, torch.zeros_like(y))), torch.zeros_like(y))
        return out, ends

    def expected(self, l, u, X):
        """Batch means of every observable over the kept draws: {(species, obs): (N_BATCH,)}."""
        E, ends = self.energies(l, u, X)
        keep = (~ends).to(l.dtype)
        batch = torch.arange(len(keep)) % N_BATCH
        res = {}
        for s in SPECIES_NAMES:
            for k, v in observables(E[s]).items():
                res[(s, k)] = torch.stack([(v * keep)[batch == j].sum() / keep[batch == j].sum() for j in range(N_BATCH)])
        return res


def model_slopes(ts, l0, u0):
    """Autodiff slopes (mean and se over batches) and the finite-difference gate, per (species, obs)."""
    X0 = depths(u0)
    dXdu = (depths(u0 + H_FD) - depths(u0 - H_FD)) / (2 * H_FD)
    l, u = torch.tensor(l0, requires_grad=True), torch.tensor(u0, requires_grad=True)
    X = torch.tensor(X0, requires_grad=True)
    ex = ts.expected(l, u, X)
    out = {}
    for key, v in ex.items():
        gl, gu, gX = [], [], []
        for j in range(N_BATCH):
            a, b, c = torch.autograd.grad(v[j], (l, u, X), retain_graph=True)
            gl.append(float(a)), gu.append(float(b) + float(c @ torch.as_tensor(dXdu)))
        out[key] = {"log10 E": (np.mean(gl), np.std(gl) / np.sqrt(N_BATCH)),
                    "cos zenith": (np.mean(gu), np.std(gu) / np.sqrt(N_BATCH))}
    with torch.no_grad():
        f = lambda dl, du: {k: v.numpy() for k, v in ts.expected(torch.tensor(l0 + dl), torch.tensor(u0 + du),
                                                                   torch.tensor(depths(u0 + du))).items()}

        def fd(h):
            pl, ml, pu, mu = f(h, 0), f(-h, 0), f(0, h), f(0, -h)
            g = lambda p, m: (np.mean((p - m) / (2 * h)), np.std((p - m) / (2 * h)) / np.sqrt(N_BATCH))
            return {k: {"log10 E": g(pl[k], ml[k]), "cos zenith": g(pu[k], mu[k])} for k in out}
        return out, fd(H_GATE), fd(H_WIDE)


def corsika_slopes(E, l, u, l0, u0):
    """Local quadratic regression of each observable around (l0, u0): slope and se in log10 E and cos zenith."""
    w = (np.abs(l - l0) < WINDOW[0]) & (np.abs(u - u0) < WINDOW[1])
    dl, du = l[w] - l0, u[w] - u0
    A = np.column_stack([np.ones_like(dl), dl, du, dl * dl, du * du, dl * du])
    out = {}
    for s in SPECIES_NAMES:
        for k, v in observables(torch.as_tensor(E[s][w])).items():
            y = v.numpy()
            ok = np.isfinite(y)
            beta, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
            r = y[ok] - A[ok] @ beta
            cov = np.linalg.pinv(A[ok].T @ A[ok]) * (r @ r) / max(1, ok.sum() - A.shape[1])
            out[(s, k)] = {"log10 E": (float(beta[1]), float(np.sqrt(cov[1, 1]))),
                           "cos zenith": (float(beta[2]), float(np.sqrt(cov[2, 2])))}
    return out, int(w.sum())


def stage_run(out, args, log):
    torch.set_default_dtype(torch.float64)
    tr = sampler.load(os.path.join(args.out, fit_profiles.TRAIN))
    zt = np.load(os.path.join(args.out, fit_profiles.TEST))
    res = {"rule": __doc__.split("Rule")[1].strip(), "classes": {}}
    for c, cn in planes.CLASS.items():
        model = sampler.fit(tr, c)
        ts = TorchSampler(model, args.n_mc, args.seed)
        m = (zt["meta.labels"] == c) & (zt["meta.log10_E"] < sampler.TOP)
        l, u = zt["meta.log10_E"][m], zt["meta.cos_zenith"][m]
        E = {s: zt[f"{s}.E"][m].astype(np.float64) for s in SPECIES_NAMES}
        pts = []
        for l0 in GRID_L:
            for u0 in np.quantile(u, GRID_U_Q):
                ms, gate, wide = model_slopes(ts, float(l0), float(u0))
                cs, n_win = corsika_slopes(E, l, u, l0, u0)
                for key in ms:
                    for d in ("log10 E", "cos zenith"):
                        (gm, sm), (gw, sw), (gc, sc) = ms[key][d], wide[key][d], cs[key][d]
                        pts.append({"log10 E": l0, "cos zenith": float(u0), "n_corsika": n_win, "species": key[0],
                                    "observable": key[1], "derivative": d, "sampler": [gm, sm],
                                    "sampler_fd_narrow": list(gate[key][d]), "sampler_fd_wide": [gw, sw],
                                    "CORSIKA": [gc, sc], "z": float((gm - gc) / np.hypot(sm, sc)),
                                    "z_wide": float((gw - gc) / np.hypot(sw, sc))})
        frac = np.mean([abs(p["z"]) <= 2 for p in pts])
        frac_wide = np.mean([abs(p["z_wide"]) <= 2 for p in pts])
        gate = [abs(p["sampler"][0] - p["sampler_fd_narrow"][0]) / max(abs(p["sampler_fd_narrow"][0]), 1e-6) for p in pts]
        res["classes"][cn] = {"points": pts, "pass_fraction": float(frac), "pass_fraction_fd_wide": float(frac_wide),
                              "gate_rel_err": {"median": float(np.median(gate)), "max": float(np.max(gate))},
                              "verdict": "PASS" if frac >= 0.8 else "FAIL"}
        log.info("%s: autodiff %d/%d within 2 se of CORSIKA (%.2f) -> %s; wide finite difference %.2f; autodiff vs "
                 "narrow finite difference rel err median %.2e max %.2e", cn, round(frac * len(pts)), len(pts), frac,
                 res["classes"][cn]["verdict"], frac_wide, np.median(gate), np.max(gate))
        for p in pts:
            if abs(p["z"]) > 2:
                log.info("  miss %s %s d/d %s at (%.2f, %.2f): sampler %.4g +- %.2g, CORSIKA %.4g +- %.2g (n %d)",
                         p["species"], p["observable"], p["derivative"], p["log10 E"], p["cos zenith"],
                         *p["sampler"], *p["CORSIKA"], p["n_corsika"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-mc", type=int, default=N_MC)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, "run"), args)
    logs.save(out, stage_run(out, args, log), args, log)


if __name__ == "__main__":
    main()
