"""Is detector density what keeps the optimized layout uniform?

Short Adam optimizations (current utility, through the surrogate) with n = 20, 25,
50 and 100 detectors, from the grid and from the center blob, judged on held-out
events with the KERNEL. If sparser arrays develop structure (clustering, a core,
a rim) or gain more over their grid, density is the limit.

The surrogate and the recon are DeepSets models with MEAN pooling over detectors
and fixed 100-slot normalisation, so an n-detector layout is fed as 100 slots that
repeat the n detectors 100/n times: mean pooling over the repeated set equals the
n-detector set exactly. The reconstructability gate counts lit detectors on the n
real detectors only. The models were trained on 100-detector layouts, so sparse
arrays are a (mild) extrapolation for them.

    python scripts/diagnostics/detector_count_scan.py [--steps 3000]
"""
import argparse
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("gate_scan", os.path.join(_HERE, "gate_scan.py"))
GS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GS)

import modules.constants as C  # noqa: E402
import modules.optimize.objective as OBJ  # noqa: E402
from modules.geometry import (  # noqa: E402
    load_tr_mountain, SurfaceUpMap, sample_initial_layout_ne, project_to_mountain_ne,
)
from modules.layouts import LearnableXY  # noqa: E402
from modules.optimize import load_models  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

DEVICE = GS.DEVICE
SLOTS = C.N_DETECTORS
COUNTS = (20, 25, 50, 100)


def sph(th, ph):
    return torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], -1)


def ang_err(parts):
    E_pred, th_p, ph_p, E_true, th_t, ph_t = parts
    u, v = sph(th_p, ph_p), sph(th_t, ph_t)
    a = torch.rad2deg(2 * torch.atan2((u - v).norm(dim=-1), (u + v).norm(dim=-1)))
    return a, (torch.log10(E_pred) - torch.log10(E_true)).abs()


def two_tier(a, e, c, lam):
    """U_two_tier (branch run16-emean, modules/optimize/utility.py): count well-reconstructed
    showers first, bounded partial credit for the rest. c = surrogate-calibrated thresholds."""
    g = torch.sigmoid((c["theta_good"] - a) / c["s_theta"]) * torch.sigmoid((c["e_good"] - e) / c["s_e"])
    q = 1.0 / (1.0 + (a / c["theta_ref"]) ** 2 + (e / c["e_ref"]) ** 2)
    return torch.mean(g + lam * (1.0 - g) * q)


def full(x, n):
    return x.repeat(SLOTS // n)


def structure(e, n):
    xy = torch.stack([e, n], -1).cpu().numpy()
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    r = np.hypot(*(xy - xy.mean(0)).T)
    return dict(nn_mean=float(nn.mean()), nn_cv=float(nn.std() / nn.mean()),
                r_p50=float(np.median(r)), r_p90=float(np.percentile(r, 90)))


def evaluate(e, n_, nd, prim_k, kern, recon, gate, tg=5.0, eg=0.2):
    with torch.no_grad():
        counts, parts = GS.predict(full(e, nd), full(n_, nd), prim_k, kern, recon)
        a, de = ang_err(parts)
        c = counts[:, :nd]
        good = (a < 5) & (de < 0.2)
        out = dict(U_kernel=float(GS.utility(parts, GS.gate(c, gate))), ang_med=float(a.median()),
                   dE_med=float(de.median()), frac_good_5=float(good.float().mean()),
                   frac_good_t=float(((a < tg) & (de < eg)).float().mean()), good_threshold=[tg, eg],
                   lit_med=float((c > 1.0).sum(1).float().median()),
                   frac_lt5_lit=float(((c > 1.0).sum(1) < 5).float().mean()))
    out.update(structure(e, n_))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kernel", type=int, default=1536)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--step-batch", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5.0)
    ap.add_argument("--counts", type=int, nargs="+", default=list(COUNTS))
    ap.add_argument("--starts", nargs="+", default=["grid", "center"])
    ap.add_argument("--utilities", nargs="+", default=["current"], choices=["current", "two_tier"])
    ap.add_argument("--lam", type=float, default=0.6, help="two-tier lambda")
    ap.add_argument("--theta-good", type=float, default=5.0,
                    help="two-tier 'well reconstructed' angle threshold on the kernel [deg]")
    ap.add_argument("--e-good", type=float, default=0.2,
                    help="two-tier 'well reconstructed' |dlog10 E| threshold on the kernel")
    ap.add_argument("--out", default=os.path.join(C.RUN_LOCATION, "diagnostics"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"RUN_LOCATION={C.RUN_LOCATION}", flush=True)

    mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY,
                                east_entry=C.EAST_ENTRY, layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    fnn, recon = load_models(DEVICE)
    for m in (fnn, recon):
        for p in (m.parameters() if hasattr(m, "parameters") else []):
            p.requires_grad_(False)
    gate = GS.settings(float(OBJ.RECONSTRUCT_THRESHOLD))[0]
    clouds, _, _, prim_k = load_events(args.n_kernel, DEVICE, mountain)
    kern = GS.CachedKernel(clouds, surface, DEVICE, chunk=args.chunk, expected_primary=prim_k)
    prim_all = torch.load(os.path.join(C.TRAINING_DATASET_FOLDER, "primary.pt")).float()

    res, layouts = {}, {}
    for nd in args.counts:
        # two-tier thresholds for this count: the surrogate threshold that passes the same
        # share of showers as (5 deg, 0.2) does on the kernel, on this count's grid layout
        torch.manual_seed(0)
        eg, ng = sample_initial_layout_ne(mountain, n_units=nd, scheme="grid")
        eg, ng = project_to_mountain_ne(mountain, torch.as_tensor(eg).float(), torch.as_tensor(ng).float())
        eg, ng = eg.float().to(DEVICE), ng.float().to(DEVICE)
        with torch.no_grad():
            a_k, e_k = ang_err(GS.predict(full(eg, nd), full(ng, nd), prim_k, kern, recon)[1])
            a_s, e_s = ang_err(GS.predict(full(eg, nd), full(ng, nd), prim_k, fnn, recon)[1])
        tg, eg = args.theta_good, args.e_good
        ka = float(torch.quantile(a_s, float((a_k < tg).float().mean()))) / tg
        ke = float(torch.quantile(e_s, float((e_k < eg).float().mean()))) / eg
        # widths scale with the thresholds (1 deg and 0.05 at the default 5 deg / 0.2)
        cal = dict(theta_good=tg * ka, e_good=eg * ke, s_theta=0.2 * tg * ka, s_e=0.25 * eg * ke,
                   theta_ref=20.0 * ka, e_ref=0.5 * ke)
        print(f"[calib n={nd}] kernel ({tg:g} deg, {eg:g}) -> surrogate ({cal['theta_good']:.2f} deg, {cal['e_good']:.3f})",
              flush=True)
        for scheme, util in [(a, b) for a in args.starts for b in args.utilities]:
            torch.manual_seed(0)
            e0, n0 = sample_initial_layout_ne(mountain, n_units=nd, scheme=scheme)
            e0, n0 = project_to_mountain_ne(mountain, torch.as_tensor(e0).float(), torch.as_tensor(n0).float())
            e0, n0 = e0.float().to(DEVICE), n0.float().to(DEVICE)
            start = evaluate(e0, n0, nd, prim_k, kern, recon, gate, tg, eg)
            xy = LearnableXY(e0.clone(), n0.clone(), device=str(DEVICE))
            opt = torch.optim.Adam(xy.parameters(), lr=args.lr)
            t0 = time.time()
            for step in range(args.steps):
                gk = torch.Generator().manual_seed(10_000 + step)
                pb = prim_all[torch.randint(0, prim_all.shape[0], (args.step_batch,), generator=gk)].to(DEVICE)
                x, y = xy()
                counts, parts = GS.predict(full(x, nd), full(y, nd), pb, fnn, recon)
                if util == "current":
                    U = GS.utility(parts, GS.gate(counts[:, :nd], gate))
                else:
                    a_b, e_b = ang_err(parts)
                    U = two_tier(a_b, e_b, cal, args.lam)
                opt.zero_grad(set_to_none=True)
                (-U).backward()
                opt.step()
                with torch.no_grad():
                    e2, n2 = project_to_mountain_ne(mountain, xy.x.detach().cpu(), xy.y.detach().cpu())
                    xy.x.data.copy_(e2.to(DEVICE)); xy.y.data.copy_(n2.to(DEVICE))
            e, n = xy.x.detach(), xy.y.detach()
            end = evaluate(e, n, nd, prim_k, kern, recon, gate, tg, eg)
            key = f"n{nd}_{scheme}_{util}"
            res[key] = dict(n=nd, start_scheme=scheme, utility=util, lam=args.lam if util != "current" else None,
                            start=start, optimized=end)
            layouts[key] = (e0.cpu().numpy(), n0.cpu().numpy(), e.cpu().numpy(), n.cpu().numpy())
            torch.save(dict(x=e.cpu(), y=n.cpu()), os.path.join(args.out, f"det_count_layout_{key}.pt"))
            gain = (end["U_kernel"] - start["U_kernel"]) / max(abs(start["U_kernel"]), 1e-9)
            print(f"  [{key:12s}] U_kernel {start['U_kernel']:6.2f} -> {end['U_kernel']:6.2f} ({gain:+.1%})  "
                  f"angle {start['ang_med']:5.2f} -> {end['ang_med']:5.2f}  good {start['frac_good_5']:.3f} -> "
                  f"{end['frac_good_5']:.3f} (good@{tg:g}deg {start['frac_good_t']:.3f} -> {end['frac_good_t']:.3f})  lit {end['lit_med']:.0f}  <5 lit {end['frac_lt5_lit']:.2f}  "
                  f"NN CV {start['nn_cv']:.2f} -> {end['nn_cv']:.2f}  r50 {start['r_p50']:.0f} -> {end['r_p50']:.0f} m  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print("\n[summary] n    start   utility   U_start  U_opt   gain    angle   good5   lit  NN-CV(start->opt)  r50(start->opt)")
    for k, r in res.items():
        s, o = r["start"], r["optimized"]
        print(f"[summary] {r['n']:3d}  {r['start_scheme']:6s}  {r['utility']:8s}  {s['U_kernel']:6.2f}  {o['U_kernel']:6.2f}  "
              f"{(o['U_kernel'] - s['U_kernel']) / max(abs(s['U_kernel']), 1e-9):+6.1%}  {o['ang_med']:5.2f}  "
              f"{o['frac_good_5']:.3f}  {o['lit_med']:3.0f}  {s['nn_cv']:.2f}->{o['nn_cv']:.2f}  "
              f"{s['r_p50']:.0f}->{o['r_p50']:.0f}")
    with open(os.path.join(args.out, "detector_count_scan.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(layouts, res, os.path.join(args.out, "detector_count_layouts.png"))


def plot(layouts, res, path):
    keys = list(layouts)
    fig, ax = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 6), sharex=True, sharey=True, squeeze=False)
    for a, k in zip(ax.ravel(), keys):
        e0, n0, e, n = layouts[k]
        r = res[k]
        a.scatter(e0, n0, s=10, color="#c9c7bf", lw=0, label="start")
        a.scatter(e, n, s=18, color="#2a78d6", lw=0, label="optimized")
        s, o = r["start"], r["optimized"]
        a.set_title(f"{r['n']} detectors, {r['utility']}\nU {s['U_kernel']:.1f} → {o['U_kernel']:.1f} · "
                    f"angle {o['ang_med']:.1f}° · NN CV {o['nn_cv']:.2f}", fontsize=10)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    ax[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Detector-count scan (current utility, surrogate) — {os.path.basename(C.RUN_LOCATION)}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
