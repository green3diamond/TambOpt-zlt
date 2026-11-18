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
mu_all, hit_all = [], []
with torch.no_grad():
    for lo in range(0, SA.N, SA.CH):
        hi = min(lo + SA.CH, SA.N)
        kern = S4.BatchKernel([c[lo:hi].to(dev) for c in SA.clouds], [None] * len(SA.clouds), surface,
                              differentiable=False, frames=SA.frames[lo:hi].to(dev))
        out = kern(SA.prim[lo:hi].to(dev), xy[None].expand(hi - lo, -1, -1))
        mu_all.append(kern.mu.cpu()); hit_all.append(((kern.mu >= 1) & (out[..., 1] > 0)).cpu())
mu, hit = torch.cat(mu_all).numpy(), torch.cat(hit_all).numpy()
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
        p, d, e1 = fr[k, 0:3], fr[k, 3:6], fr[k, 6:9]
        t1 = c[:, 0] * e1[0] + c[:, 1] * e1[1]
        P = p + ((c[:, 2] + 1) * C.C8_PLANE_DX)[:, None] * d + t1[:, None] * e1
        pts.setdefault(k, []).append(P)

# drawing as the earlier figure (explore_trigger_counts.ipynb `panel`): mountain vertices, shower cloud, decay
# vertex, detectors coloured by counts; 1,500 m view around the mesh centre, Up = mesh range +- 300 m
tris = np.load("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings/sites/malata.npz")["tris"]
mtn = np.unique(tris.reshape(-1, 3), axis=0)
ctr, view_r = mtn.mean(0), 1500.0
up_lo, up_hi = mtn[:, 2].min() - 300.0, mtn[:, 2].max() + 300.0
in_view = lambda P: ((np.abs(P[:, 0] - ctr[0]) <= view_r) & (np.abs(P[:, 1] - ctr[1]) <= view_r)
                     & (P[:, 2] >= up_lo) & (P[:, 2] <= up_hi))
gx, gy = x.cpu().numpy(), y.cpu().numpy(); gz = surface(y, x).cpu().numpy()
tags = ("Elow", "Emid", "Ehigh")

fig, axes = plt.subplots(1, 3, figsize=(21, 6.2), subplot_kw=dict(projection="3d"), gridspec_kw={"wspace": 0.35})
for k, (a, i) in enumerate(zip(axes, pick)):
    a.scatter(mtn[:, 0], mtn[:, 1], mtn[:, 2], s=3, c="#DDD6C9", alpha=0.6, depthshade=False, label="mountain")
    P = np.concatenate(pts[k]); P = P[in_view(P)][::2]
    if len(P):
        a.scatter(P[:, 0], P[:, 1], P[:, 2], s=8, c="tab:green", alpha=0.85, linewidths=0, depthshade=False,
                  label="shower cloud")
    pdec = np.asarray(meta.positions[i], dtype=float)
    if in_view(pdec[None])[0]:
        a.scatter(*pdec, c="red", s=110, marker="*", depthshade=False, label="decay vertex")
    h = hit[i]
    a.scatter(gx[~h], gy[~h], gz[~h], s=40, marker="^", c="0.45", depthshade=False, label="no signal")
    sc = a.scatter(gx[h], gy[h], gz[h], s=40, marker="^", c=mu[i][h], cmap="plasma",
                   norm=LogNorm(1.0, max(float(mu[i][h].max()), 1.01)), depthshade=False, label="signal")
    plt.colorbar(sc, ax=a, label="counts", shrink=0.55, pad=0.04)
    a.set_xlim(ctr[0] - view_r, ctr[0] + view_r); a.set_ylim(ctr[1] - view_r, ctr[1] + view_r); a.set_zlim(up_lo, up_hi)
    a.set_xlabel("East [m]"); a.set_ylabel("North [m]"); a.set_zlabel("Up [m]")
    a.set_title(f"#{i} ({tags[k]})\nE={E[i]:.1e}, \u03b8={np.degrees(th[i]):.0f}\u00b0, "
                f"\u03c6={np.degrees(ph[i]) % 360:.0f}\u00b0, {int(nh[i])}/100", fontsize=11)
hd, lb = {}, {}
for a in axes:
    for hh, ll in zip(*a.get_legend_handles_labels()):
        hd.setdefault(ll, hh)
fig.legend(list(hd.values()), list(hd.keys()), loc="lower center", ncol=len(hd), fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(pad=1.2)
os.makedirs(f"{args.out}/figures/results", exist_ok=True)
fig.savefig(f"{args.out}/figures/results/detector_patterns_representative.pdf", dpi=200, bbox_inches="tight")
print("wrote detector_patterns_representative.pdf", flush=True)
