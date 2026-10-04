"""3-D view of run 17 primaries: decay points, their directions (decay -> target segment) and the
45-degree plane, two viewing angles. Writes <run17>/test_v6_run_01_northeast/decays_3d.png."""
import math, os
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with h5py.File(os.path.join(R, "inputs", "plane_taus.h5"), "r") as f:
    pos, tgt, en = f["position"][...].T, f["target"][...].T, f["energy"][...]
rng = np.random.default_rng(0)
sub = rng.choice(len(en), 300, replace=False)
loge = np.log10(en[sub])
norm = colors.Normalize(5, 7); cmap = cm.viridis

d_e = 3000 * math.cos(math.radians(45))
E, N = np.meshgrid(np.linspace(-d_e, 0, 2), np.linspace(-1500, 1500, 2))
fig = plt.figure(figsize=(16, 7.5))
for k, (elev, azim) in enumerate(((22, -60), (8, -95))):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.plot_surface(E, N, -E, color="#b9b7af", alpha=0.35, edgecolor="#6b6a64", lw=0.6)
    for (e0, e1) in ((500, 3500),):                       # decay cube outline
        for n0 in (-1500, 1500):
            for u0 in (0, 3000):
                ax.plot([e0, e1], [n0, n0], [u0, u0], color="#999", lw=0.6, ls="--")
        for n0 in (-1500, 1500):
            for e in (e0, e1):
                ax.plot([e, e], [n0, n0], [0, 3000], color="#999", lw=0.6, ls="--")
        for u0 in (0, 3000):
            for e in (e0, e1):
                ax.plot([e, e], [-1500, 1500], [u0, u0], color="#999", lw=0.6, ls="--")
    for i, j in enumerate(sub):
        c = cmap(norm(loge[i]))
        ax.plot([pos[j, 0], tgt[j, 0]], [pos[j, 1], tgt[j, 1]], [pos[j, 2], tgt[j, 2]], color=c, lw=0.6, alpha=0.55)
    ax.scatter(*pos[sub].T, c=loge, cmap=cmap, norm=norm, s=10, depthshade=False, label="decay")
    ax.scatter(*tgt[sub].T, color="#eb6834", s=5, depthshade=False, label="target on plane")
    ax.set(xlabel="East [m]", ylabel="North [m]", zlabel="Up [m]",
           xlim=(-2200, 3600), ylim=(-1600, 1600), zlim=(0, 3100))
    ax.set_box_aspect((5800, 3200, 3100))
    ax.view_init(elev=elev, azim=azim)
    if k == 1:
        ax.set_yticks([-1500, 0, 1500])
    if k == 0:
        ax.legend(loc="upper left", fontsize=9)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=fig.axes, shrink=0.6, pad=0.02, label="log10(E / GeV)")
fig.suptitle(f"Run 17 primaries: 300 of {len(en):,} decays (dots, coloured by energy), "
             "line = direction to the plane target, dashed = decay cube")
out = os.path.join(R, "test_v6_run_01_northeast", "decays_3d.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("[write]", out)
