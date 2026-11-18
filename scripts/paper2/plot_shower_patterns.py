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

# slope as a shaded mesh, shower cloud (points within the view), shower axis from the decay vertex, detectors
# coloured by expected counts (shared log scale); view = mesh bounding box + PAD, true aspect, seen from the South
tris = np.load("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/tmp_rings/sites/malata.npz")["tris"]
PAD = 400.0
lo, hi = tris.reshape(-1, 3).min(0) - PAD, tris.reshape(-1, 3).max(0) + PAD
in_view = lambda P: np.all((P >= lo) & (P <= hi), 1)
gx, gy = x.cpu().numpy(), y.cpu().numpy(); gz = surface(y, x).cpu().numpy()
norm = LogNorm(1.0, max(float(mu[i][hit[i]].max()) for i in pick))
up = tris[:, :, 2].mean(1)
shade = matplotlib.cm.Greys(0.12 + 0.3 * (up - up.min()) / np.ptp(up))
from matplotlib.ticker import MaxNLocator

fig = plt.figure(figsize=(15, 5.6))
axes = [fig.add_subplot(1, 3, k + 1, projection="3d", computed_zorder=False) for k in range(3)]
for k, (a, i) in enumerate(zip(axes, pick)):
    a.add_collection3d(Poly3DCollection(tris, facecolors=shade, edgecolors=(0, 0, 0, 0.12), linewidths=0.3,
                                        alpha=0.9, zorder=1))
    P = np.concatenate(pts[k]); P = P[in_view(P)]
    if len(P) > 6000:
        P = P[np.random.default_rng(0).choice(len(P), 6000, replace=False)]
    a.scatter(P[:, 0], P[:, 1], P[:, 2], s=3, c="tab:green", alpha=0.5, lw=0, depthshade=False, zorder=2,
              label="shower cloud")
    p0, d0 = fr[k, 0:3].astype(float), fr[k, 3:6].astype(float)
    tt = np.linspace(0, 15000, 3000); A = p0 + tt[:, None] * d0; A = A[in_view(A)]
    if len(A):
        a.plot(A[:, 0], A[:, 1], A[:, 2], c="tab:green", lw=1.2, ls="--", zorder=3, label="shower axis")
    if in_view(p0[None])[0]:
        a.scatter(*p0, c="red", s=90, marker="*", depthshade=False, zorder=6, label="decay vertex")
    h = hit[i]
    a.scatter(gx[~h], gy[~h], gz[~h], s=22, marker="^", c="white", edgecolors="0.25", linewidths=0.5,
              depthshade=False, zorder=4, label="no signal")
    sc = a.scatter(gx[h], gy[h], gz[h], s=34, marker="^", c=mu[i][h], cmap="plasma", norm=norm, edgecolors="k",
                   linewidths=0.4, depthshade=False, zorder=5, label="signal")
    a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1]); a.set_zlim(lo[2], hi[2])
    a.set_box_aspect(tuple(hi - lo), zoom=0.95)
    a.view_init(elev=22, azim=-70)
    for ax_ in (a.xaxis, a.yaxis, a.zaxis):
        ax_.set_major_locator(MaxNLocator(4)); ax_.pane.set_alpha(0.15)
    a.tick_params(labelsize=7, pad=1)
    a.set_xlabel("East [m]", fontsize=8, labelpad=2); a.set_ylabel("North [m]", fontsize=8, labelpad=2)
    a.set_zlabel("Up [m]", fontsize=8, labelpad=2)
    a.set_title(f"$E$ = {E[i]:.1e} GeV, $\\theta$ = {np.degrees(th[i]):.0f}$^\\circ$, "
                f"$\\phi$ = {np.degrees(ph[i]) % 360:.0f}$^\\circ$\n{int(nh[i])}/100 detectors with a signal",
                fontsize=10, pad=0)
fig.subplots_adjust(left=0.0, right=0.9, bottom=0.08, top=0.9, wspace=0.08)
cax = fig.add_axes([0.92, 0.25, 0.012, 0.55])
fig.colorbar(sc, cax=cax, label=r"expected particles $\mu$")
hd = {}
for a in axes:
    for hh, ll in zip(*a.get_legend_handles_labels()):
        hd.setdefault(ll, hh)
fig.legend(list(hd.values()), list(hd.keys()), loc="lower center", ncol=len(hd), fontsize=9, frameon=False,
           bbox_to_anchor=(0.45, 0.02), markerscale=1.5)
os.makedirs(f"{args.out}/figures/results", exist_ok=True)
fig.savefig(f"{args.out}/figures/results/detector_patterns_representative.pdf", dpi=200)
print("wrote detector_patterns_representative.pdf", flush=True)
