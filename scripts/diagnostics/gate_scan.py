"""Would a higher trigger threshold or a softer gate help the layout optimization?

The utility weights every event by the reconstructability gate

    lit_d = sigmoid(TAU_LAYOUT * (counts_d - LAYOUT_THRESHOLD))        per detector (trigger)
    r     = sigmoid(TAU_RECONSTRUCT * (sum_d lit_d - RECONSTRUCT_THRESHOLD))   per event

This scans both thresholds and both slopes, and judges each setting by a yardstick
that does NOT depend on the gate: the frozen recon's median angle and log-energy
error on held-out events, with the KERNEL (ground-truth) detector response.

  Part A (static): U under each setting, through the SURROGATE (what the optimizer
    reads), for a fixed set of layouts. Reports whether U ranks the layouts like
    the gate-free recon quality does (Spearman), how much it separates them, and
    how consistent its gradient is across independent primary batches.
  Part B (dynamic): a short Adam run from the grid under each setting, then the
    yardstick on the result. The default setting runs twice (two seeds) to show
    the run-to-run noise.

Runs against whatever run constants.py points at (use the run's own checkout or
worktree). --recon-dir overrides the recon (run 16 has none; it borrows run 15's).

    python scripts/diagnostics/gate_scan.py [--steps 1500] [--n-kernel 1536]
"""
import argparse
import json
import math
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import modules  # noqa: E402,F401
import modules.constants as C  # noqa: E402
from modules.geometry import (  # noqa: E402
    load_tr_mountain, SurfaceUpMap, sample_initial_layout_ne, project_to_mountain_ne,
)
from modules.layouts import LearnableXY  # noqa: E402
from modules.layouts.strategies import _STRATEGY_FNS  # noqa: E402
from modules.optimize import load_models  # noqa: E402
import modules.optimize.objective as OBJ  # noqa: E402
from modules.optimize.utility import reconstructability, U_E, U_angle  # noqa: E402
from plots.layouts.true_utility import load_events, KernelDualLabels  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B0 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/"
EXTRA_LAYOUTS = {
    "r12_rim_upr640": B0 + "12_10pct_mean_analytic/test_v6_run_04_optimize_lbfgs_ensemble_upr640_grid/layout_best.pt",
    "r14_spine_upr200": B0 + "14_sigma10_10pct/test_v6_run_04_optimize_lbfgs_ensemble_upr200_grid/layout_best.pt",
}


def settings(rt0):
    """(name, layout_threshold, tau_layout, reconstruct_threshold, tau_reconstruct)."""
    base = (1.0, 5.0, rt0, 0.2)
    s = [("default", *base)]
    for lt in (3.0, 10.0, 30.0):
        s.append((f"trigger>{lt:g}", lt, 5.0, rt0, 0.2))
    for tl in (1.0, 0.3):
        s.append((f"trigger_tau={tl:g}", 1.0, tl, rt0, 0.2))
    for rt in (5.0, 10.0, 20.0, 30.0):
        if rt != rt0:
            s.append((f"event>{rt:g}", 1.0, 5.0, rt, 0.2))
    for tr in (0.1, 0.05):
        s.append((f"event_tau={tr:g}", 1.0, 5.0, rt0, tr))
    s.append(("event>20,tau=0.1", 1.0, 5.0, 20.0, 0.1))
    s.append(("trigger>3,event>20", 3.0, 5.0, 20.0, 0.2))
    return s


# ── kernel with cached interpolation widths (identical output, one neighbour search) ──
class CachedKernel(KernelDualLabels):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._bw = {}
        self._k = getattr(C, "KERNEL_INTERP_K", None)

    def _labels(self, clouds, e_det, n_det):
        if not self._k:
            return super()._labels(clouds, e_det, n_det)
        from modules.data.dataset_builder import compute_labels_batch
        from modules.showers.kernel import adaptive_bandwidths
        key = (clouds.data_ptr(), tuple(clouds.shape))
        if key not in self._bw:
            with torch.no_grad():
                self._bw[key] = adaptive_bandwidths(clouds, self._k, C.SIGMA_SPATIAL)
        E, T = compute_labels_batch(clouds, e_det, n_det, self.surface,
                                    differentiable=self.differentiable, bandwidth=self._bw[key])
        return torch.stack([torch.log1p(E), torch.log1p(T * C.T_LOG_SCALE)], dim=-1)


def gate(counts, s):
    _, lt, tl, rt, tr = s
    return reconstructability(counts, layout_threshold=lt, tau_layout=tl,
                              reconstruct_threshold=rt, tau_reconstruct=tr)


def utility(parts, r):
    E_pred, th_p, ph_p, E_true, th_t, ph_t = parts
    return (OBJ.W_THETA * U_angle(th_p, th_t, r, cap=OBJ.CAP_THETA)
            + OBJ.W_PHI * U_angle(ph_p, ph_t, r, cap=OBJ.CAP_PHI, period=2 * math.pi)
            + OBJ.W_E * U_E(E_pred, E_true, r, cap=OBJ.CAP_E)) / OBJ.W_DIV


def predict(x, y, prim, fwd, recon):
    E_det, E_pred, th_p, ph_p = OBJ._predict_primary(x, y, prim, fwd, recon, sample=False)
    E_true, th_t, ph_t = OBJ.primary_to_physical_labels(prim)
    return torch.expm1(E_det), (E_pred, th_p, ph_p, E_true, th_t, ph_t)


def yardstick(x, y, prim_k, kern, recon, sets):
    """Gate-free recon quality on the kernel, plus kernel U under each setting."""
    with torch.no_grad():
        counts, parts = predict(x, y, prim_k, kern, recon)
    E_pred, th_p, ph_p, E_true, th_t, ph_t = parts
    dp = torch.stack([torch.sin(th_p) * torch.cos(ph_p), torch.sin(th_p) * torch.sin(ph_p), torch.cos(th_p)], 1)
    dt = torch.stack([torch.sin(th_t) * torch.cos(ph_t), torch.sin(th_t) * torch.sin(ph_t), torch.cos(th_t)], 1)
    ang = torch.rad2deg(torch.arccos((dp * dt).sum(1).clamp(-1, 1)))
    de = (torch.log10(E_pred) - torch.log10(E_true)).abs()
    n_lit = (counts > 1.0).sum(1).float()
    out = dict(ang_med=float(ang.median()), dE_med=float(de.median()),
               U_nogate=float(utility(parts, torch.ones_like(ang))), n_lit_med=float(n_lit.median()))
    out["U_kernel"] = {s[0]: float(utility(parts, gate(counts, s))) for s in sets}
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kernel", type=int, default=1536)
    ap.add_argument("--n-surr", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--grad-batches", type=int, default=8)
    ap.add_argument("--grad-batch", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--step-batch", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5.0)
    ap.add_argument("--recon-dir", default=None)
    ap.add_argument("--out", default=os.path.join(C.RUN_LOCATION, "diagnostics"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sets = settings(float(OBJ.RECONSTRUCT_THRESHOLD))
    print(f"RUN_LOCATION={C.RUN_LOCATION}  sigma={C.SIGMA_SPATIAL}  "
          f"interp_k={getattr(C, 'KERNEL_INTERP_K', None)}  default gate: trigger>{OBJ.LAYOUT_THRESHOLD:g} "
          f"tau {OBJ.TAU_LAYOUT:g}, event>{OBJ.RECONSTRUCT_THRESHOLD:g} tau {OBJ.TAU_RECONSTRUCT:g}", flush=True)

    mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY,
                                east_entry=C.EAST_ENTRY, layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    fnn, recon = load_models(DEVICE, recon_dir=args.recon_dir)
    for m in (fnn, recon):                       # only layout gradients are needed
        for p in (m.parameters() if hasattr(m, "parameters") else []):
            p.requires_grad_(False)

    # ── layouts ──
    L = {}
    for scheme in ("grid", "center"):
        torch.manual_seed(0)
        e, n = sample_initial_layout_ne(mountain, n_units=C.N_DETECTORS, scheme=scheme)
        e, n = project_to_mountain_ne(mountain, torch.as_tensor(e).float(), torch.as_tensor(n).float())
        L[scheme] = (e.float(), n.float())
    for name, fn in _STRATEGY_FNS.items():
        for seed in (1, 2):
            e, n = fn(mountain, rng=np.random.default_rng(seed))
            L[f"{name.replace('layout_', '')}_{seed}"] = (torch.as_tensor(e).float(), torch.as_tensor(n).float())
    for scheme in ("grid", "center"):
        p = os.path.join(C.RUN_LOCATION, f"test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_{scheme}", "layout_best.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=False)
            L[f"optimized_{scheme}"] = (torch.as_tensor(d["x"]).float(), torch.as_tensor(d["y"]).float())
    for name, p in EXTRA_LAYOUTS.items():
        if os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=False)
            L[name] = (torch.as_tensor(d["x"]).float(), torch.as_tensor(d["y"]).float())
    L = {k: (e.to(DEVICE), n.to(DEVICE)) for k, (e, n) in L.items()}
    print(f"[layouts] {len(L)}: {', '.join(L)}", flush=True)

    clouds, Bk, _, prim_k = load_events(args.n_kernel, DEVICE, mountain)
    kern = CachedKernel(clouds, surface, DEVICE, chunk=args.chunk, expected_primary=prim_k)
    prim_all = torch.load(os.path.join(C.TRAINING_DATASET_FOLDER, "primary.pt")).float()
    g = torch.Generator().manual_seed(0)
    prim_s = prim_all[torch.randint(0, prim_all.shape[0], (args.n_surr,), generator=g)].to(DEVICE)

    # ── Part A ──
    t0 = time.time()
    res = dict(settings=[list(s) for s in sets], layouts={}, partA={}, partB={})
    for name, (e, n) in L.items():
        ys = yardstick(e, n, prim_k, kern, recon, sets)
        with torch.no_grad():
            counts, parts = predict(e, n, prim_s, fnn, recon)
        ys["U_surr"] = {s[0]: float(utility(parts, gate(counts, s))) for s in sets}
        ys["mean_r_surr"] = {s[0]: float(gate(counts, s).mean()) for s in sets}
        res["layouts"][name] = ys
        print(f"  [{name:24s}] angle {ys['ang_med']:6.2f} deg  |dlog10E| {ys['dE_med']:.3f}  "
              f"lit {ys['n_lit_med']:5.1f}  U_surr(default) {ys['U_surr']['default']:.2f}", flush=True)
    names = list(L)
    realistic = [k for k in names if k != "center"]
    ang = np.array([res["layouts"][k]["ang_med"] for k in names])
    dE = np.array([res["layouts"][k]["dE_med"] for k in names])
    for s in sets:
        us = np.array([res["layouts"][k]["U_surr"][s[0]] for k in names])
        uk = np.array([res["layouts"][k]["U_kernel"][s[0]] for k in names])
        ur = np.array([res["layouts"][k]["U_surr"][s[0]] for k in realistic])
        res["partA"][s[0]] = dict(rank_vs_angle=spearman(us, -ang), rank_vs_energy=spearman(us, -dE),
                                  surr_vs_kernel=spearman(us, uk),
                                  spread=float((ur.max() - ur.min()) / np.median(ur)))
    print(f"[partA] static metrics in {time.time() - t0:.0f}s", flush=True)

    # gradient consistency at grid and at the run's optimized layout
    for where in ("grid", "optimized_grid"):
        if where not in L:
            continue
        for s in sets:
            gs = []
            for k in range(args.grad_batches):
                gk = torch.Generator().manual_seed(100 + k)
                pb = prim_all[torch.randint(0, prim_all.shape[0], (args.grad_batch,), generator=gk)].to(DEVICE)
                xy = LearnableXY(L[where][0].clone(), L[where][1].clone(), device=str(DEVICE))
                x, y = xy()
                counts, parts = predict(x, y, pb, fnn, recon)
                U = utility(parts, gate(counts, s))
                gx, gy = torch.autograd.grad(U, (xy.x, xy.y))
                gs.append(torch.cat([gx, gy]))
            G = torch.stack(gs)
            res["partA"][s[0]][f"grad_consistency_{where}"] = float(G.mean(0).norm() / G.norm(dim=1).mean())
            res["partA"][s[0]][f"grad_norm_{where}"] = float(G.norm(dim=1).mean())
    print(f"[partA] done in {time.time() - t0:.0f}s", flush=True)

    # ── Part B: short optimization per setting ──
    runs = [(s, 0) for s in sets] + [(sets[0], 1)]
    for s, seed in runs:
        tag = s[0] + ("" if seed == 0 else f"_seed{seed}")
        torch.manual_seed(1000 + seed)
        xy = LearnableXY(L["grid"][0].clone(), L["grid"][1].clone(), device=str(DEVICE))
        opt = torch.optim.Adam(xy.parameters(), lr=args.lr)
        t1 = time.time()
        for step in range(args.steps):
            gk = torch.Generator().manual_seed(10_000 * (seed + 1) + step)
            pb = prim_all[torch.randint(0, prim_all.shape[0], (args.step_batch,), generator=gk)].to(DEVICE)
            x, y = xy()
            counts, parts = predict(x, y, pb, fnn, recon)
            loss = -utility(parts, gate(counts, s))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                e2, n2 = project_to_mountain_ne(mountain, xy.x.detach().cpu(), xy.y.detach().cpu())
                xy.x.data.copy_(e2.to(DEVICE)); xy.y.data.copy_(n2.to(DEVICE))
        e, n = xy.x.detach(), xy.y.detach()
        ys = yardstick(e, n, prim_k, kern, recon, sets)
        ys["moved_m_median"] = float(torch.hypot(e - L["grid"][0], n - L["grid"][1]).median())
        res["partB"][tag] = ys
        torch.save(dict(x=e.cpu(), y=n.cpu()), os.path.join(args.out, f"gate_scan_layout_{tag.replace('>', 'gt').replace(',', '_')}.pt"))
        print(f"  [B {tag:22s}] angle {ys['ang_med']:6.2f}  |dlog10E| {ys['dE_med']:.3f}  "
              f"U_kernel(default) {ys['U_kernel']['default']:.2f}  U_nogate {ys['U_nogate']:.2f}  "
              f"moved {ys['moved_m_median']:.0f} m  ({time.time() - t1:.0f}s)", flush=True)

    grid = res["layouts"]["grid"]
    print(f"\n[summary] grid start: angle {grid['ang_med']:.2f} deg, |dlog10E| {grid['dE_med']:.3f}, "
          f"U_kernel(default) {grid['U_kernel']['default']:.2f}")
    print(f"[summary] {'setting':22s} {'rank~angle':>10s} {'rank~E':>7s} {'surr~kern':>9s} {'spread':>7s} "
          f"{'gradcons':>8s} | {'B angle':>8s} {'B dlogE':>8s} {'B U_def':>8s}")
    for s in sets:
        a, b = res["partA"][s[0]], res["partB"][s[0]]
        print(f"[summary] {s[0]:22s} {a['rank_vs_angle']:10.2f} {a['rank_vs_energy']:7.2f} "
              f"{a['surr_vs_kernel']:9.2f} {a['spread']:7.3f} {a.get('grad_consistency_grid', float('nan')):8.2f} | "
              f"{b['ang_med']:8.2f} {b['dE_med']:8.3f} {b['U_kernel']['default']:8.2f}")
    b = res["partB"]["default_seed1"]
    print(f"[summary] {'default (seed 1)':22s} {'':>10s} {'':>7s} {'':>9s} {'':>7s} {'':>8s} | "
          f"{b['ang_med']:8.2f} {b['dE_med']:8.3f} {b['U_kernel']['default']:8.2f}")
    with open(os.path.join(args.out, "gate_scan.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[write] {args.out}/gate_scan.json")


if __name__ == "__main__":
    main()
