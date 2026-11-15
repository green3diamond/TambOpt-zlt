"""Run 20: what repeated triggers of the SAME shower look like over the array. The kernel response (shower frame, ring,
k = 8 widths, interpolated leading-edge T) of a few showers on the 100-detector grid, then several Poisson count draws
of each (modules/optimize/counting.count_species, the validation's drawn counts, no background):
  column 0   expected detected particles mu per detector (2 m^2, eps_gamma 0.02), log colour; P(trigger) = P(>= 3 hit)
  columns 1+ one draw each: hit detectors (>= 1 counted particle) coloured by arrival time relative to the draw's
             earliest hit [ns]; hit with T = 0 (no point above the kernel's per-point threshold) black x; not hit grey.
Showers: from the Stage 4 scoring set (the last pool events), P(trigger) between 0.2 and 0.95, spread in log10 E.
The dashed line is the shower axis projected on the map, the star where it reaches the surface.
Writes <run 20>/diagnostics/trigger_draws/trigger_draws.png (+ .json of the per-draw numbers).

    source slurm/run20_guard.sh; python plots/analysis/run20/plot_trigger_draws.py [--n-showers 4 --n-draws 6]
"""
import argparse
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("s4", os.path.join(REPO, "scripts", "04_optimize_lbfgs_kernel.py"))
S4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S4)
import modules.constants as C  # noqa: E402
from modules.data.dataset_builder import compute_labels_batch  # noqa: E402
from modules.geometry.mountain import _read_detector_region, _ecef_to_enu  # noqa: E402
from modules.optimize.background import axis_distance  # noqa: E402
from modules.optimize.counting import count_species, expected_counts, trigger_prob  # noqa: E402
from modules.surrogates.dual import combine_species_outputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-showers", type=int, default=4)
ap.add_argument("--n-draws", type=int, default=6)
ap.add_argument("--n-pool", type=int, default=1024, help="scoring-set events searched for showers")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
dev = S4.DEVICE
OUT = os.path.join(C.RUN_LOCATION, "diagnostics", "trigger_draws")
os.makedirs(OUT, exist_ok=True)

mountain = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                               layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)
g = torch.load(os.path.join(C.RUN_LOCATION, "diagnostics", "ker5", "grid_n100.pt"), map_location="cpu")
e_det, n_det = g["x"].float().to(dev), g["y"].float().to(dev)
pool = S4.MainPool(mountain)
a = pool.n - args.n_pool
clouds, bws, prim, frames = pool.load(a, pool.n)
log_e = (prim[:, 3] * (S4.LOG_E_MAX - S4.LOG_E_MIN) + S4.LOG_E_MIN).numpy()


@torch.no_grad()
def response(idx):
    """Per-species log-encoded kernel outputs and mu (B, nd) for pool rows idx."""
    fr = frames[idx].to(dev)
    outs = []
    for c, w in zip(clouds, bws):
        E, T = compute_labels_batch(c[idx].to(dev), e_det, n_det, surface, bandwidth=w[idx].to(dev)
                                    if w is not None else None, frames=fr)
        outs.append(torch.stack([torch.log1p(E), torch.log1p(T * C.T_LOG_SCALE)], -1))
    r = axis_distance(e_det, n_det, surface, fr)
    return outs, r, sum(expected_counts(outs, r))


# P(trigger) for the whole search set, in chunks; pick showers spread in energy with an uncertain trigger.
ptrig = []
for c0 in range(0, prim.shape[0], 64):
    _, _, mu = response(torch.arange(c0, min(c0 + 64, prim.shape[0])))
    ptrig.append(trigger_prob(mu).float().cpu())
ptrig = torch.cat(ptrig).numpy()
cand = np.flatnonzero((ptrig > 0.2) & (ptrig < 0.95))
if cand.size < args.n_showers:
    raise SystemExit(f"only {cand.size} showers with 0.2 < P(trigger) < 0.95 in {prim.shape[0]}")
cand = cand[np.argsort(log_e[cand])]
pick = cand[np.linspace(0, cand.size - 1, args.n_showers).round().astype(int)]
print(f"[pick] {cand.size} candidates of {prim.shape[0]}; rows {pick.tolist()}")

outs, r, mu = response(torch.as_tensor(pick))
gen = torch.Generator(device=dev)
gen.manual_seed(args.seed)
draws = []                                           # per draw: (hit (B, nd), T [ns] (B, nd))
for _ in range(args.n_draws):
    cnt, n_tot = count_species(outs, r, gen)
    comb = combine_species_outputs(*cnt)
    T = torch.expm1(comb[..., 1]) / C.T_LOG_SCALE * 1e9
    draws.append(((n_tot > 0).cpu().numpy(), T.cpu().numpy()))
mu = mu.cpu().numpy()
pt = ptrig[pick]

verts, faces, det_idx, loc = _read_detector_region(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY)
tri = np.stack([_ecef_to_enu(verts[:, faces[k, det_idx]], loc[0], loc[1]).T for k in range(3)], 1)
xlim = (tri[..., 0].min() - 80, tri[..., 0].max() + 80)
ylim = (tri[..., 1].min() - 80, tri[..., 1].max() + 80)
E0, N0 = e_det.cpu().numpy(), n_det.cpu().numpy()


def axis_track(row):
    """Axis points (E, N) inside the map and where the axis meets the surface (None if not in range)."""
    f = frames[pick[row]].double().numpy()
    p, d = f[0:3], f[3:6]
    s = np.linspace(0, 30000, 30001)
    pts = p[None] + s[:, None] * d[None]
    up = surface(torch.as_tensor(pts[:, 1], dtype=torch.float32, device=dev),
                 torch.as_tensor(pts[:, 0], dtype=torch.float32, device=dev)).cpu().numpy()
    below = np.flatnonzero(pts[:, 2] <= up)
    hit = pts[below[0], :2] if below.size else None
    return pts[:, :2], hit


nr, nc = args.n_showers, args.n_draws + 1
fig, axs = plt.subplots(nr, nc, figsize=(2.9 * nc, 3.1 * nr), constrained_layout=True, squeeze=False)
summary = []
tmax = max(float(np.nanpercentile([np.ptp(T[i][h[i]]) if h[i].any() else 0 for h, T in draws
                                   for i in range(nr)], 90)), 1.0)
for i in range(nr):
    track, foot = axis_track(i)
    row = dict(pool_row=int(a + pick[i]), log10E=float(log_e[pick[i]]), p_trigger=float(pt[i]),
               mu_sum=float(mu[i].sum()), draws=[])
    for j in range(nc):
        ax = axs[i, j]
        ax.add_collection(PolyCollection(tri[:, :, :2], facecolors="#f4f4f4", edgecolors="#dddddd", linewidths=0.3))
        ax.plot(track[:, 0], track[:, 1], ls="--", lw=0.8, color="#555555")
        if foot is not None:
            ax.scatter(*foot, marker="*", s=90, color="#d62728", zorder=5)
        if j == 0:
            m = np.clip(mu[i], 1e-3, None)
            sc = ax.scatter(E0, N0, c=m, s=16, cmap="viridis", norm=LogNorm(1e-3, max(m.max(), 1.0)), lw=0)
            fig.colorbar(sc, ax=ax, shrink=0.8, label="μ (expected particles)")
            ax.set_title(f"log10 E = {log_e[pick[i]]:.2f}, P(trig) = {pt[i]:.2f}\n"
                         f"expected: Σμ = {mu[i].sum():.1f}, P(hit)>0.5: {(1 - np.exp(-mu[i]) > 0.5).sum()}",
                         fontsize=8)
        else:
            hit, T = draws[j - 1][0][i], draws[j - 1][1][i]
            timed = hit & (T > 0)
            t_rel = T - (T[timed].min() if timed.any() else 0.0)
            ax.scatter(E0[~hit], N0[~hit], s=10, facecolors="none", edgecolors="#b0b0b0", lw=0.5)
            sc = ax.scatter(E0[timed], N0[timed], c=t_rel[timed], s=26, cmap="plasma", vmin=0, vmax=tmax,
                            edgecolors="k", lw=0.4, zorder=4)
            ax.scatter(E0[hit & ~timed], N0[hit & ~timed], marker="x", s=22, color="k", lw=0.9, zorder=4)
            nh = int(hit.sum())
            ax.set_title(f"draw {j}: {nh} hit ({int((hit & ~timed).sum())} no T) — "
                         f"{'TRIGGER' if nh >= 3 else 'no trigger'}", fontsize=8,
                         color="k" if nh >= 3 else "#c0392b")
            row["draws"].append(dict(n_hit=nh, n_hit_no_t=int((hit & ~timed).sum()), trigger=nh >= 3,
                                     hit=np.flatnonzero(hit).tolist(), t_ns=t_rel[timed].round(1).tolist()))
            if j == nc - 1:
                fig.colorbar(sc, ax=ax, shrink=0.8, label="T − earliest hit [ns]")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    summary.append(row)
fig.suptitle("Run 20: repeated Poisson draws of the same showers on the 100-detector grid (kernel, no background). "
             "Grey = not hit, colour = hit (time), x = hit with no time; dashed = axis, star = axis at the surface",
             fontsize=10)
out = os.path.join(OUT, "trigger_draws")
fig.savefig(out + ".png", dpi=150, facecolor="white")
json.dump(summary, open(out + ".json", "w"), indent=1)
for row in summary:
    print(f"  row {row['pool_row']}: log10E {row['log10E']:.2f}  P(trig) {row['p_trigger']:.2f}  "
          f"hits per draw {[d['n_hit'] for d in row['draws']]}")
print("[write]", out + ".png / .json")
