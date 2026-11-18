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

tris = np.load("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings/sites/malata.npz")["tris"]
gx, gy = x.cpu().numpy(), y.cpu().numpy(); gz = surface(y, x).cpu().numpy()
lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
pad = 500.0
vmax = max(float(mu[i][hit[i]].max()) for i in pick)
norm = LogNorm(1.0, vmax)

plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(15, 4.6), constrained_layout=True)
for k, i in enumerate(pick):
    a = fig.add_subplot(1, 3, k + 1, projection="3d", computed_zorder=False)
    up = tris[:, :, 2].mean(1)
    a.add_collection3d(Poly3DCollection(tris, facecolors=matplotlib.cm.terrain(matplotlib.colors.Normalize(
        up.min() - 300, up.max())(up)), edgecolors="0.4", linewidths=0.15, alpha=0.55, zorder=1))
    P = np.concatenate(pts[k])
    inb = np.all((P >= lo - pad) & (P <= hi + pad), 1)
    P = P[inb]
    if len(P) > 8000:
        P = P[np.random.default_rng(0).choice(len(P), 8000, replace=False)]
    a.scatter(P[:, 0], P[:, 1], P[:, 2], s=1.5, c="tab:green", alpha=0.6, lw=0, depthshade=False, zorder=2,
              label="shower points")
    h = hit[i]
    a.scatter(gx[~h], gy[~h], gz[~h] + 5, s=10, marker="^", c="0.45", depthshade=False, zorder=3, label="no signal")
    sc = a.scatter(gx[h], gy[h], gz[h] + 5, s=18, marker="^", c=mu[i][h], cmap="plasma", norm=norm,
                   depthshade=False, zorder=4, label="signal")
    a.set_xlim(lo[0] - pad, hi[0] + pad); a.set_ylim(lo[1] - pad, hi[1] + pad); a.set_zlim(lo[2] - pad, hi[2] + pad)
    a.set_box_aspect((hi[0] - lo[0] + 2 * pad, hi[1] - lo[1] + 2 * pad, hi[2] - lo[2] + 2 * pad))
    a.set_xlabel("East [m]"); a.set_ylabel("North [m]"); a.set_zlabel("Up [m]")
    a.set_title(f"$E$ = {E[i]:.1e} GeV, $\\theta$ = {np.degrees(th[i]):.0f}$^\\circ$, "
                f"$\\phi$ = {np.degrees(ph[i]) % 360:.0f}$^\\circ$\n{int(nh[i])}/100 detectors with a signal")
    a.view_init(elev=28, azim=-60)
    a.tick_params(labelsize=7)
axs = list(fig.axes)
hd, lb = axs[0].get_legend_handles_labels()
fig.colorbar(sc, ax=axs, location="bottom", shrink=0.25, aspect=30, pad=0.01, label=r"expected particles $\mu$")
fig.legend(hd, lb, loc="outside lower left", ncol=3, frameon=False, markerscale=2)
os.makedirs(f"{args.out}/figures/results", exist_ok=True)
fig.savefig(f"{args.out}/figures/results/detector_patterns_representative.pdf", dpi=200)
print("wrote detector_patterns_representative.pdf", flush=True)
