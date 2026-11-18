"""Paper 1, Sec. results: three held-out flow-matching showers over the Malata slope with the uniform-grid detectors,
one row (low, mid and high energy). Kernel, placement and expected counts as score_all.py (noise-free); a detector
has a signal when its expected count is >= 1 and it has an arrival time (as the trigger and the analytical recon).

Shower points are drawn at their ENU position p + (l + 1) 500 m d + t1 e1 (decay vertex p, axis d, horizontal
transverse offset t1 = x e1_E + y e1_N of the native cloud; the near-vertical offset t2 was not recorded, so the
points lie in the horizontal-transverse plane of the shower). Run under slurm/run20_guard.sh.

    python plot_shower_patterns.py --out <overleaf clone> [--n 600]
    -> <out>/figures/results/detector_patterns_representative.pdf
"""
import argparse, importlib.util, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=600)                 # held-out showers searched
args = ap.parse_args()
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["score_all.py", os.devnull + ".none", str(args.n)]   # no layouts: loads showers, mesh, surface only
spec = importlib.util.spec_from_file_location("score_all", f"{R}/scripts/paper2/score_all.py")
SA = importlib.util.module_from_spec(spec); spec.loader.exec_module(SA)
S4, C, dev, surface, mountain = SA.S4, SA.C, SA.dev, SA.surface, SA.mountain
import modules.optimize.objective as O
from modules.data.dataset_builder import shower_frames

L = torch.load(f"{C.RUN_LOCATION}/diagnostics/paper1/ugrid.pt", map_location="cpu", weights_only=False)
x = torch.as_tensor(L["x"]).float().to(dev); y = torch.as_tensor(L["y"]).float().to(dev)
xy = torch.stack([x, y], -1)
mu_all, hit_all, e_all = [], [], []
with torch.no_grad():
    for lo in range(0, SA.N, SA.CH):
        hi = min(lo + SA.CH, SA.N)
        kern = S4.BatchKernel([c[lo:hi].to(dev) for c in SA.clouds], [None] * len(SA.clouds), surface,
                              differentiable=False, frames=SA.frames[lo:hi].to(dev))
        out = kern(SA.prim[lo:hi].to(dev), xy[None].expand(hi - lo, -1, -1))
        mu_all.append(kern.mu.cpu()); hit_all.append(((kern.mu >= 1) & (out[..., 1] > 0)).cpu())
        e_all.append(torch.expm1(out[..., 0].clamp(min=0)).cpu())
mu, hit = torch.cat(mu_all).numpy(), torch.cat(hit_all).numpy()
Ekern = torch.cat(e_all).numpy()
E, th, ph = (v.cpu().numpy() for v in O.primary_to_physical_labels(SA.prim))
nh = hit.sum(1)

# one shower per energy tercile (of the showers with >= 3 hits): the one with the median number of hits
ok = np.flatnonzero(nh >= 3)
edges = np.quantile(np.log10(E[ok]), [0, 1 / 3, 2 / 3, 1])
pick = []
for a, b in zip(edges[:-1], edges[1:]):
    m = ok[(np.log10(E[ok]) >= a) & (np.log10(E[ok]) <= b)]
    pick.append(int(m[np.argsort(nh[m])[len(m) // 2]]))
print("showers:", [(i, f"{E[i]:.2e}", int(nh[i])) for i in pick], flush=True)

# raw (unplaced) clouds of the picked showers -> ENU points
import showerdata
meta = SA.pool.meta
fr = shower_frames(torch.as_tensor(meta.positions[pick]).float(), torch.as_tensor(meta.dirs[pick]).float()).numpy()
pts = {}
for s in range(len(C.SPECIES_NAMES)):
    for k, i in enumerate(pick):
        c = showerdata.load(SA.pool.path, start=s * meta.per_sp + i, stop=s * meta.per_sp + i + 1).points[0]
        c = c[c[:, 3] > 0]
        t1 = c[:, 0] * fr[k, 6] + c[:, 1] * fr[k, 7]
        pts.setdefault(k, []).append(np.column_stack([np.abs(t1), (c[:, 2] + 1) * C.C8_PLANE_DX, c[:, 3]]))

# as the kernel sees them: each point is a ring of radius |t1| around the axis, in the plane perpendicular to it at
# its depth; the N_RING brightest points of each shower (all species), each drawn at N_PHI angles
N_RING, N_PHI = 250, 120
phi = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
for k in list(pts):
    q = np.concatenate(pts[k]); q = q[np.argsort(-q[:, 2])[:N_RING]]
    p, d, e1, e2 = fr[k, 0:3], fr[k, 3:6], fr[k, 6:9], fr[k, 9:12]
    off = q[:, 0, None, None] * (np.cos(phi)[None, :, None] * e1 + np.sin(phi)[None, :, None] * e2)
    pts[k] = [(p + q[:, 1, None, None] * d + off).reshape(-1, 3)]

# drawing: the earlier figure's panel (05_paper_figures.make_detector_panel before e8b7487), with its helpers
# opt_plotting.mountain_enu / draw_detectors_enu_3d and geometry_plots.tidy_3d_axes; detectors coloured by the
# kernel counts as there. One row of three showers.
def _lib(name):
    sp = importlib.util.spec_from_file_location(name, f"{R}/plots/lib/{name}.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
sys.path.insert(0, f"{R}/plots/lib"); sys.path.insert(0, f"{R}/plots")
optplt, gp = _lib("opt_plotting"), _lib("geometry_plots")
mtn_enu = optplt.mountain_enu(mountain)
surf_cpu = surface.to("cpu")
ctr = mtn_enu.mean(axis=0); view_r = 1500.0
up_lo, up_hi = mtn_enu[:, 2].min() - 300.0, mtn_enu[:, 2].max() + 300.0
in_view = lambda P: ((np.abs(P[:, 0] - ctr[0]) <= view_r) & (np.abs(P[:, 1] - ctr[1]) <= view_r)
                     & (P[:, 2] >= up_lo) & (P[:, 2] <= up_hi))
e_det, n_det = x.cpu().numpy(), y.cpu().numpy()
tags = ("Elow", "Emid", "Ehigh")

fig, axes = plt.subplots(1, 3, figsize=(7.0 * 3, 6.2), squeeze=False, subplot_kw=dict(projection="3d"),
                         gridspec_kw={"wspace": 0.35})
legend_ax = None
for k, (ax, i) in enumerate(zip(axes.flat, pick)):
    ax.scatter(mtn_enu[:, 0], mtn_enu[:, 1], mtn_enu[:, 2], s=3, c="#DDD6C9", alpha=0.6, depthshade=False,
               label="mountain")
    P = np.concatenate(pts[k]); P = P[in_view(P)]
    if len(P):
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1.0, c="tab:green", alpha=0.35, linewidths=0, depthshade=False,
                   label="shower cloud (rings)")
    pdec = np.asarray(meta.positions[i], dtype=float)
    if in_view(pdec[None])[0]:
        ax.scatter(*pdec, c="red", s=110, marker="*", depthshade=False, label="decay vertex")
    sc, n_live, n_over = optplt.draw_detectors_enu_3d(ax, Ekern[i], e_det, n_det, surf_cpu, log1p_space=False)
    ax.set_xlim(ctr[0] - view_r, ctr[0] + view_r); ax.set_ylim(ctr[1] - view_r, ctr[1] + view_r)
    ax.set_zlim(up_lo, up_hi)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_zlabel("Up [m]")
    gp.tidy_3d_axes(ax)
    ax.view_init(elev=25, azim=-125)
    if sc is not None:
        plt.colorbar(sc, ax=ax, label="counts", shrink=0.55, pad=0.04)
    ax.set_title(f"#{i} ({tags[k]})\nE={E[i]:.1e}, \u03b8={np.degrees(th[i]):.0f}\u00b0, "
                 f"\u03c6={np.degrees(ph[i]) % 360:.0f}\u00b0, {n_live}/100", fontsize=9)
    if legend_ax is None and ax.get_legend_handles_labels()[1]:
        legend_ax = ax
if legend_ax is not None:
    handles, labels = legend_ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=4, mfc="none", mec="tab:green") if l.startswith("shower") else h
               for h, l in zip(handles, labels)]
    fig.legend(handles, labels, loc="lower center", fontsize=7, ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(pad=1.2)
os.makedirs(f"{args.out}/figures/results", exist_ok=True)
fig.savefig(f"{args.out}/figures/results/detector_patterns_representative.pdf", dpi=200, bbox_inches="tight")
print("wrote detector_patterns_representative.pdf", flush=True)
