"""How well do the kernel's rings reproduce the true lateral energy density? CORSIKA 8 per-particle check.

The ring kernel (place_clouds_shower, transverse="ring") puts each super-particle on a ring of radius |t1| (horizontal
transverse offset only); the true distance from the axis is r = sqrt(t1^2 + t2^2). For annular bins a ring of radius u
deposits all its energy in the annulus holding u, so the rings' energy density per bin is sum(E over u in bin) / area.
Compared per bin, all species and per species, pooled over planes and the 16 pi- showers:
    eps_true(r)   : binned by r
    eps_ring(u)   : binned by u = |t1|          (current kernel)
    eps_ring2(u)  : binned by u = sqrt(2) |t1|  (proposed)
plus the local slope k(r) = -dln eps_true / dln r and the power-law prediction B((k-1)/2, 1/2)/pi (* 2^(k/2-1)),
and per species the mean kinetic energy per particle vs the TRUE r. Writes c8_scaling.json, the table
modules/optimize/background.py uses (copy it to modules/optimize/c8_scaling.json):
    particles of species s in a detector of area A per unit kernel E = A / (2 pi sigma^2 KE_s(r)) / ratio_s(r)
Frame per shower from the data: n = normal of the most populated plane (SVD), e1 = horizontal across the heading,
e2 = n x e1, offsets from each plane's energy-weighted centre (planes with < 1000 particles skipped).

    python scripts/diagnostics/c8_ring_radius_check.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy.special import beta  # noqa: E402

SRC = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training2/pdg_-211"
OUT = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/c8_calibration"
R_E = np.logspace(0, 4, 29)                                                # 1 m - 10 km
RC = np.sqrt(R_E[:-1] * R_E[1:]); AREA = np.pi * (R_E[1:] ** 2 - R_E[:-1] ** 2)
SP = {"all": None, "e": 11, "mu": 13, "gamma": 22}
MODES = ("r", "t1", "t1x2")
files = sorted(glob.glob(f"{SRC}/energy_*/*/*_hits.parquet"))
E = {(s, m): np.zeros((len(files), len(RC))) for s in SP for m in MODES}
N = {s: np.zeros(len(RC)) for s in SP if s != "all"}                       # particle counts per true-r bin
zen = []
for i, f in enumerate(files):
    t = pq.read_table(f, columns=["x", "y", "z", "pdg", "kinetic_energy", "plane_index"])
    P = np.stack([t.column(c).to_numpy().astype(np.float64) for c in ("x", "y", "z")], 1)
    pdg, k, pl = np.abs(t.column("pdg").to_numpy()), t.column("kinetic_energy").to_numpy().astype(np.float64), t.column("plane_index").to_numpy()
    big = np.bincount(pl).argmax(); mb = pl == big
    d = P[mb] - P[mb].mean(0)
    n = np.linalg.svd(d[np.random.default_rng(0).choice(len(d), min(len(d), 50000), replace=False)], full_matrices=False)[2][-1]
    cg = np.array([P[pl == q].mean(0) for q in np.unique(pl) if (pl == q).sum() >= 1000])
    if (cg[-1] - cg[0]) @ n < 0:
        n = -n                                                              # point along the travel
    e1 = np.array([-n[1], n[0], 0.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    zen.append(float(np.degrees(np.arccos(n[2]))))
    t1, t2, use = np.zeros(len(k)), np.zeros(len(k)), np.zeros(len(k), bool)
    for q in np.unique(pl):
        m = pl == q
        if m.sum() < 1000:
            continue
        dd = P[m] - np.average(P[m], axis=0, weights=np.maximum(k[m], 1e-12))
        t1[m], t2[m], use[m] = dd @ e1, dd @ e2, True
    rad = {"r": np.hypot(t1, t2), "t1": np.abs(t1), "t1x2": np.sqrt(2) * np.abs(t1)}
    for s, pid in SP.items():
        sel = use if pid is None else use & (pdg == pid)
        for mo, u in rad.items():
            E[(s, mo)][i] = np.histogram(u[sel], bins=R_E, weights=k[sel])[0] / AREA
        if pid is not None:
            N[s] += np.histogram(rad["r"][sel], bins=R_E)[0]
    print(os.path.basename(f), f"zenith (from the planes) {zen[-1]:.1f}", flush=True)

pool = {key: v.sum(0) for key, v in E.items()}
ratio = {(s, mo): pool[(s, mo)] / np.where(pool[(s, "r")] > 0, pool[(s, "r")], np.nan) for s in SP for mo in ("t1", "t1x2")}
band = {}
for s in SP:
    for mo in ("t1", "t1x2"):
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = E[(s, mo)] / E[(s, "r")]
        band[(s, mo)] = np.nanpercentile(np.where(np.isfinite(rr), rr, np.nan), [16, 84], axis=0)
lt = np.log(pool[("all", "r")])
k_loc = -np.gradient(lt, np.log(RC))
with np.errstate(invalid="ignore"):
    pred1 = np.where(k_loc > 1.05, beta((k_loc - 1) / 2, 0.5) / np.pi, np.nan)
pred2 = pred1 * 2 ** (k_loc / 2 - 1)

res = dict(r_centres_m=RC.tolist(), zenith_from_planes=zen, local_slope_all=k_loc.tolist(),
           ratio={f"{s}_{mo}": ratio[(s, mo)].tolist() for s in SP for mo in ("t1", "t1x2")},
           predicted_powerlaw={"t1": pred1.tolist(), "t1x2": pred2.tolist()})
json.dump(res, open(os.path.join(OUT, "c8_ring_radius_check.json"), "w"), indent=1)

# table for modules/optimize/background.py: per species KE(r) [GeV] (true r) and ring/true ratios; bins with < 200
# particles are dropped (interpolation carries the neighbours). Band values: pooled over 100 m - 3 km.
bsel = (R_E[:-1] >= 100) & (R_E[1:] <= 3000)
tab = dict(source=f"{len(files)} CORSIKA 8 pi- showers, {SRC}", script="scripts/diagnostics/c8_ring_radius_check.py",
           r_m=RC.tolist(), species={}, band_100m_3km={})
for s in N:
    ke_s = pool[(s, "r")] * AREA / np.maximum(N[s], 1)                      # sum KE / count per true-r bin
    ok = N[s] >= 200
    tab["species"][s] = dict(n=N[s].tolist(), ke_gev=np.where(ok, ke_s, np.nan).tolist(),
                             ring_ratio_t1=np.where(ok, ratio[(s, "t1")], np.nan).tolist(),
                             ring_ratio_t1x2=np.where(ok, ratio[(s, "t1x2")], np.nan).tolist())
    tab["band_100m_3km"][s] = dict(ke_gev=float((pool[(s, "r")] * AREA)[bsel].sum() / N[s][bsel].sum()),
                                   ring_ratio_t1=float((pool[(s, "t1")] * AREA)[bsel].sum() / (pool[(s, "r")] * AREA)[bsel].sum()),
                                   ring_ratio_t1x2=float((pool[(s, "t1x2")] * AREA)[bsel].sum() / (pool[(s, "r")] * AREA)[bsel].sum()))
json.dump(tab, open(os.path.join(OUT, "c8_scaling.json"), "w"), indent=1)
print("band 100 m - 3 km:", json.dumps(tab["band_100m_3km"]))
for s in N:
    print(s, "KE(r):", " ".join(f"{RC[j]:.0f}:{tab['species'][s]['ke_gev'][j]:.3g}" for j in range(0, len(RC), 3)))
print("\n   r [m]   k(r)   |t1|: all    e    mu   gam  (pred)   sqrt2|t1|: all    e    mu   gam  (pred)")
for j in range(len(RC)):
    if 3 <= RC[j] <= 5000:
        print(f"{RC[j]:8.0f} {k_loc[j]:6.2f}   " + " ".join(f"{ratio[(s, 't1')][j]:5.2f}" for s in SP) + f"  ({pred1[j]:.2f})   "
              + " ".join(f"{ratio[(s, 't1x2')][j]:5.2f}" for s in SP) + f"  ({pred2[j]:.2f})")

fig, ax = plt.subplots(1, 3, figsize=(23, 6.6))
for mo, ls, lab in (("r", "-", "true r"), ("t1", "--", "rings at |t1| (current)"), ("t1x2", ":", "rings at √2·|t1|")):
    ax[0].loglog(RC, pool[("all", mo)] / len(files), "k" + ls, lw=2, label=lab)
ax[0].set(xlabel="distance from the axis [m]", ylabel="kinetic energy density [GeV/m², per shower, all planes]",
          title="Lateral energy density, all species")
col = {"all": "k", "e": "#2a78d6", "mu": "#c0392b", "gamma": "#e8a33a"}
for mo, ls in (("t1", "--"), ("t1x2", "-")):
    for s in SP:
        ax[1].semilogx(RC, ratio[(s, mo)], ls, color=col[s], lw=2.2 if s == "all" else 1.4,
                       label=f"{s}, {'|t1|' if mo == 't1' else '√2·|t1|'}")
    lo, hi = band[("all", mo)]
    ax[1].fill_between(RC, lo, hi, color="k", alpha=0.08)
ax[1].axhline(1, color="#888", lw=1); ax[1].set(ylim=(0, 2.5), xlabel="distance from the axis [m]", ylabel="ring density / true density",
                                                title="Ratio (lines: pooled; grey band: 16–84% over showers, all species)")
ax[2].semilogx(RC, ratio[("all", "t1")], "k--", lw=2, label="measured, |t1|")
ax[2].semilogx(RC, pred1, "--", color="#7b8794", lw=1.5, label="power-law prediction from k(r), |t1|")
ax[2].semilogx(RC, ratio[("all", "t1x2")], "k-", lw=2, label="measured, √2·|t1|")
ax[2].semilogx(RC, pred2, "-", color="#7b8794", lw=1.5, label="power-law prediction, √2·|t1|")
a2 = ax[2].twinx(); a2.semilogx(RC, k_loc, color="#1baf7a", lw=1.5); a2.set_ylabel("local slope k(r) of the true density", color="#1baf7a")
ax[2].axhline(1, color="#888", lw=1); ax[2].set(ylim=(0, 2.5), xlabel="distance from the axis [m]", ylabel="ring / true (all species)",
                                                title="Measured vs the power-law formula; green: k(r)")
for a in ax:
    a.grid(alpha=0.25, lw=0.5); a.legend(fontsize=9, frameon=False)
fig.suptitle(f"Ring radius |t1| vs √2·|t1| vs true r · {len(files)} CORSIKA 8 π⁻ showers, real particles, planes with ≥ 1000 particles", fontsize=14)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_ring_radius_check.png"), dpi=120, facecolor="white")
print("[write]", os.path.join(OUT, "c8_ring_radius_check.png"))
