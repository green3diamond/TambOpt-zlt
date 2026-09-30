"""Does the two-tier utility (modules/optimize/utility.py U_two_tier) give better
layouts than the current one?

Goal it encodes: as many showers as possible reconstructed well (angle < theta_good
AND |dlog10 E| < e_good), then as much accuracy as possible for the rest.

Short Adam optimizations through the surrogate (what Stage 4 reads), current
utility vs two-tier variants, from the grid (and one pair from the center blob).
Every result is judged on held-out events with the KERNEL response:
  - fraction of showers reconstructed well at (5 deg, 0.2) and (3 deg, 0.15)
  - median errors overall and of the showers that are not good
  - the current utility, for reference
  - layout structure: nearest-neighbour distance CV, radial spread

Calibration: through the surrogate the recon looks ~4x too accurate, so the
two-tier thresholds are set by quantile matching on the grid layout: a surrogate
threshold passes the same fraction of showers as the kernel threshold does.

    python scripts/diagnostics/two_tier_test.py --recon-dir <recon> [--steps 3000]
"""
import argparse
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

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
from modules.optimize.utility import U_two_tier, angle_between_deg  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

DEVICE = GS.DEVICE
INK, GREY = "#0b0b0b", "#52514e"


def sph(th, ph):
    return torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], -1)


def errors(parts):
    E_pred, th_p, ph_p, E_true, th_t, ph_t = parts
    a = angle_between_deg(sph(th_p, ph_p), sph(th_t, ph_t))
    e = (torch.log10(E_pred) - torch.log10(E_true)).abs()
    return a, e


def structure(e, n):
    xy = torch.stack([e, n], -1).cpu().numpy()
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    r = np.hypot(*(xy - xy.mean(0)).T)
    return dict(nn_mean=float(nn.mean()), nn_cv=float(nn.std() / nn.mean()),
                r_p50=float(np.median(r)), r_p90=float(np.percentile(r, 90)))


def yardstick(e, n, prim_k, kern, recon, default_gate):
    with torch.no_grad():
        counts, parts = GS.predict(e, n, prim_k, kern, recon)
        a, de = errors(parts)
        good5, good3 = (a < 5) & (de < 0.2), (a < 3) & (de < 0.15)
        out = dict(frac_good_5=float(good5.float().mean()), frac_good_3=float(good3.float().mean()),
                   ang_med=float(a.median()), dE_med=float(de.median()),
                   ang_med_rest=float(a[~good5].median()), dE_med_rest=float(de[~good5].median()),
                   U_two_tier_kernel=float(U_two_tier(a, de)[0]),
                   U_current_kernel=float(GS.utility(parts, GS.gate(counts, default_gate))))
    out.update(structure(e, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kernel", type=int, default=1536)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--step-batch", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5.0)
    ap.add_argument("--recon-dir", default=None)
    ap.add_argument("--out", default=os.path.join(C.RUN_LOCATION, "diagnostics"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"RUN_LOCATION={C.RUN_LOCATION}  hurdle={getattr(C, 'SURROGATE_HURDLE', None)}  "
          f"e_mode={getattr(C, 'KERNEL_E_MODE', 'sum')}", flush=True)

    mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY,
                                east_entry=C.EAST_ENTRY, layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    fnn, recon = load_models(DEVICE, recon_dir=args.recon_dir)
    for m in (fnn, recon):
        for p in (m.parameters() if hasattr(m, "parameters") else []):
            p.requires_grad_(False)
    default_gate = GS.settings(float(OBJ.RECONSTRUCT_THRESHOLD))[0]

    inits = {}
    for scheme in ("grid", "center"):
        torch.manual_seed(0)
        e, n = sample_initial_layout_ne(mountain, n_units=C.N_DETECTORS, scheme=scheme)
        e, n = project_to_mountain_ne(mountain, torch.as_tensor(e).float(), torch.as_tensor(n).float())
        inits[scheme] = (e.float().to(DEVICE), n.float().to(DEVICE))

    clouds, Bk, _, prim_k = load_events(args.n_kernel, DEVICE, mountain)
    kern = GS.CachedKernel(clouds, surface, DEVICE, chunk=args.chunk, expected_primary=prim_k)
    prim_all = torch.load(os.path.join(C.TRAINING_DATASET_FOLDER, "primary.pt")).float()

    # ── calibration: surrogate thresholds that pass the same fraction as the kernel ones ──
    e0, n0 = inits["grid"]
    with torch.no_grad():
        a_k, e_k = errors(GS.predict(e0, n0, prim_k, kern, recon)[1])
        a_s, e_s = errors(GS.predict(e0, n0, prim_k, fnn, recon)[1])

    def calib(theta_good, e_good):
        fa = float((a_k < theta_good).float().mean())
        fe = float((e_k < e_good).float().mean())
        ka = float(torch.quantile(a_s, fa)) / theta_good
        ke = float(torch.quantile(e_s, fe)) / e_good
        return dict(theta_good=theta_good * ka, e_good=e_good * ke, s_theta=1.0 * ka, s_e=0.05 * ke,
                    theta_ref=20.0 * ka, e_ref=0.5 * ke, kernel_frac_angle=fa, kernel_frac_energy=fe)

    cal = {5: calib(5.0, 0.2), 3: calib(3.0, 0.15)}
    print(f"[calib] grid, kernel vs surrogate: median angle {a_k.median():.2f} vs {a_s.median():.2f} deg, "
          f"|dlog10E| {e_k.median():.3f} vs {e_s.median():.3f}", flush=True)
    for k, c in cal.items():
        print(f"[calib] kernel ({k} deg, {0.2 if k == 5 else 0.15}) -> surrogate "
              f"({c['theta_good']:.2f} deg, {c['e_good']:.3f})  [kernel passes angle {c['kernel_frac_angle']:.2f}, "
              f"energy {c['kernel_frac_energy']:.2f}]", flush=True)

    variants = [
        dict(name="current", kind="current", init="grid", seed=0),
        dict(name="current_seed1", kind="current", init="grid", seed=1),
        dict(name="two_tier_l0.3", kind="tt", lam=0.3, th=5, init="grid", seed=0),
        dict(name="two_tier_l0.3_seed1", kind="tt", lam=0.3, th=5, init="grid", seed=1),
        dict(name="two_tier_l0.1", kind="tt", lam=0.1, th=5, init="grid", seed=0),
        dict(name="two_tier_l0", kind="tt", lam=0.0, th=5, init="grid", seed=0),
        dict(name="two_tier_l0.3_3deg", kind="tt", lam=0.3, th=3, init="grid", seed=0),
        dict(name="current_center", kind="current", init="center", seed=0),
        dict(name="two_tier_l0.3_center", kind="tt", lam=0.3, th=5, init="center", seed=0),
    ]
    res = dict(calibration=cal, start={s: yardstick(*inits[s], prim_k, kern, recon, default_gate) for s in inits},
               variants={})
    for s, v in res["start"].items():
        print(f"[start {s}] good(5deg) {v['frac_good_5']:.3f}  good(3deg) {v['frac_good_3']:.3f}  "
              f"angle {v['ang_med']:.2f}  NN CV {v['nn_cv']:.2f}", flush=True)

    layouts = {}
    for v in variants:
        torch.manual_seed(1000 + v["seed"])
        xy = LearnableXY(inits[v["init"]][0].clone(), inits[v["init"]][1].clone(), device=str(DEVICE))
        opt = torch.optim.Adam(xy.parameters(), lr=args.lr)
        t0 = time.time()
        for step in range(args.steps):
            gk = torch.Generator().manual_seed(10_000 * (v["seed"] + 1) + step)
            pb = prim_all[torch.randint(0, prim_all.shape[0], (args.step_batch,), generator=gk)].to(DEVICE)
            x, y = xy()
            counts, parts = GS.predict(x, y, pb, fnn, recon)
            if v["kind"] == "current":
                U = GS.utility(parts, GS.gate(counts, default_gate))
                g_mean = float("nan")
            else:
                c = cal[v["th"]]
                a, de = errors(parts)
                U, g = U_two_tier(a, de, c["theta_good"], c["e_good"], c["s_theta"], c["s_e"],
                                  c["theta_ref"], c["e_ref"], v["lam"])
                g_mean = float(g.mean())
            opt.zero_grad(set_to_none=True)
            (-U).backward()
            opt.step()
            with torch.no_grad():
                e2, n2 = project_to_mountain_ne(mountain, xy.x.detach().cpu(), xy.y.detach().cpu())
                xy.x.data.copy_(e2.to(DEVICE)); xy.y.data.copy_(n2.to(DEVICE))
            if step % 1000 == 0 or step == args.steps - 1:
                print(f"    [{v['name']} {step:5d}] U_surr {float(U):.4f}  mean g {g_mean:.3f}", flush=True)
        e, n = xy.x.detach(), xy.y.detach()
        layouts[v["name"]] = (e.cpu().numpy(), n.cpu().numpy())
        ys = yardstick(e, n, prim_k, kern, recon, default_gate)
        res["variants"][v["name"]] = dict(v, **ys)
        torch.save(dict(x=e.cpu(), y=n.cpu()), os.path.join(args.out, f"two_tier_layout_{v['name']}.pt"))
        print(f"  [{v['name']:22s}] good(5deg) {ys['frac_good_5']:.3f}  good(3deg) {ys['frac_good_3']:.3f}  "
              f"angle {ys['ang_med']:.2f}  rest angle {ys['ang_med_rest']:.2f}  |dlog10E| {ys['dE_med']:.3f}  "
              f"U_current {ys['U_current_kernel']:.2f}  NN CV {ys['nn_cv']:.2f}  r90 {ys['r_p90']:.0f} m  "
              f"({time.time() - t0:.0f}s)", flush=True)

    print("\n[summary] variant                 good5   good3   angle  rest-ang  dlogE  U_cur   NN-CV")
    for k, s in res["start"].items():
        print(f"[summary] start {k:17s} {s['frac_good_5']:.3f}   {s['frac_good_3']:.3f}   {s['ang_med']:5.2f}  "
              f"{s['ang_med_rest']:6.2f}   {s['dE_med']:.3f}  {s['U_current_kernel']:5.2f}  {s['nn_cv']:.2f}")
    for k, s in res["variants"].items():
        print(f"[summary] {k:23s} {s['frac_good_5']:.3f}   {s['frac_good_3']:.3f}   {s['ang_med']:5.2f}  "
              f"{s['ang_med_rest']:6.2f}   {s['dE_med']:.3f}  {s['U_current_kernel']:5.2f}  {s['nn_cv']:.2f}")
    with open(os.path.join(args.out, "two_tier_test.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot_layouts(layouts, res, mountain, os.path.join(args.out, "two_tier_layouts.png"))


def plot_layouts(layouts, res, mountain, path):
    names = list(layouts)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(5.2 * cols, 6.2 * rows))
    mesh = np.asarray(mountain.centroids_ENU)
    for a, name in zip(ax.ravel(), names):
        e, n = layouts[name]
        s = res["variants"][name]
        a.scatter(mesh[:, 0], mesh[:, 1], s=4, color="#d9d7d0", lw=0)
        a.scatter(e, n, s=22, color="#2a78d6", edgecolor="white", lw=0.5)
        a.set_title(f"{name}\ngood(5°) {s['frac_good_5']:.1%} · angle {s['ang_med']:.2f}° · NN CV {s['nn_cv']:.2f}",
                    fontsize=10, color=INK)
        a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    for a in ax.ravel()[len(names):]:
        a.axis("off")
    fig.suptitle(f"Layouts: current vs two-tier utility — {os.path.basename(C.RUN_LOCATION)}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
