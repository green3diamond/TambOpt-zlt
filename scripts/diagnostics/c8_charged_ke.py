"""Kinetic energy per charged particle and charged particles per GeV, from CORSIKA 8 per-particle output.

Input: the hit-level parquet files of Hamza's CORSIKA 8 runs (one row per e+-, mu+-, gamma crossing an observation
plane; columns x, y, z, pdg, time, kinetic_energy [GeV], plane_index, z_depth; no thinning; cuts 10 MeV e/gamma,
0.5 GeV mu). Only pi- primaries (0.11-1.2e6 GeV, 16 showers) have such files left.

Why: the kernel's E is SUMMED KINETIC ENERGY [GeV] over a 10 m Gaussian (peak weight 1, total weight 2 pi sigma^2 ~
628 m^2 per super-particle, see modules/showers/kernel.py), not a particle count. A detector (scintillator) counts
charged particles (MIPs). With n_ch / eps = charged particles per GeV of kinetic energy (all species) at a distance r
from the axis, the expected number of charged particles in a detector of area A is
    mu = E * (n_ch / eps) * A / (2 pi sigma^2)
This script measures n_ch / eps and the mean kinetic energy per particle, per species and distance from the axis.
Ratios of sums over an annulus = ratios of densities, so no cells are needed.

    python scripts/diagnostics/c8_charged_ke.py [--out DIR]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

SRC = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training2/pdg_-211"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/c8_calibration")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

R_E = np.array([0, 10, 30, 100, 300, 1000, 3000, 1e4, 1e9])
SPECIES = {"e": (11,), "mu": (13,), "gamma": (22,)}
files = sorted(glob.glob(f"{SRC}/energy_*/*/*_hits.parquet"))
print(len(files), "showers", flush=True)
nb = len(R_E) - 1
cnt = {s: np.zeros(nb) for s in SPECIES}                 # particles per r bin, summed over showers and planes
ke = {s: np.zeros(nb) for s in SPECIES}                  # kinetic energy [GeV]
per_shower, ke_samples = [], {s: [] for s in SPECIES}
rng = np.random.default_rng(0)
for f in files:
    t = pq.read_table(f, columns=["x", "y", "z", "pdg", "kinetic_energy", "plane_index"])
    x, y, z = (t.column(c).to_numpy().astype(np.float64) for c in ("x", "y", "z"))
    pdg, k = np.abs(t.column("pdg").to_numpy()), t.column("kinetic_energy").to_numpy().astype(np.float64)
    pl = t.column("plane_index").to_numpy()
    # distance from the axis: 3-D distance from each plane's energy-weighted centre. The planes are perpendicular to
    # the axis, so this is the full transverse distance (x, y AND z: for these near-horizontal showers the planes are
    # ~vertical and z is the second transverse direction, as large as the horizontal one). Also report the origin offset.
    r = np.empty_like(x); off = []
    for p in np.unique(pl):
        m = pl == p
        w = np.maximum(k[m], 0)
        cx, cy, cz = (np.average(v[m], weights=w) for v in (x, y, z))
        off.append(np.hypot(cx, cy))
        r[m] = np.sqrt((x[m] - cx) ** 2 + (y[m] - cy) ** 2 + (z[m] - cz) ** 2)
    rb = np.clip(np.digitize(r, R_E) - 1, 0, nb - 1)
    row = dict(file=os.path.basename(f), n=int(len(k)), centre_offset_m_median=float(np.median(off)))
    for s, ids in SPECIES.items():
        m = np.isin(pdg, ids)
        cnt[s] += np.bincount(rb[m], minlength=nb); ke[s] += np.bincount(rb[m], weights=k[m], minlength=nb)
        row[f"n_{s}"] = int(m.sum()); row[f"mean_ke_{s}"] = float(k[m].mean()) if m.any() else None
        if m.any():
            ke_samples[s].append(rng.choice(k[m], size=min(20000, int(m.sum())), replace=False))
    per_shower.append(row)
    print(f"{row['file']}: {row['n']:,} particles, axis offset {row['centre_offset_m_median']:.1f} m, mean KE e "
          f"{row['mean_ke_e']:.3f} mu {row['mean_ke_mu']:.3f} gamma {row['mean_ke_gamma']:.3f} GeV", flush=True)

tot_ke = sum(ke.values())
n_ch = cnt["e"] + cnt["mu"]
res = dict(source=SRC, n_showers=len(files), r_edges_m=R_E.tolist(), per_shower=per_shower, bins=[])
for i in range(nb):
    b = dict(r=f"{R_E[i]:g}-{R_E[i + 1]:g} m")
    for s in SPECIES:
        b[f"n_{s}"] = float(cnt[s][i]); b[f"mean_ke_{s}_GeV"] = float(ke[s][i] / cnt[s][i]) if cnt[s][i] else None
    b["mean_ke_charged_GeV"] = float((ke["e"][i] + ke["mu"][i]) / n_ch[i]) if n_ch[i] else None
    b["charged_per_GeV_all"] = float(n_ch[i] / tot_ke[i]) if tot_ke[i] else None      # n_ch / eps (eps: e + mu + gamma)
    b["share_energy"] = {s: float(ke[s][i] / tot_ke[i]) if tot_ke[i] else None for s in SPECIES}
    res["bins"].append(b)
allc = dict(mean_ke_charged_GeV=float((ke["e"].sum() + ke["mu"].sum()) / n_ch.sum()),
            charged_per_GeV_all=float(n_ch.sum() / tot_ke.sum()),
            **{f"mean_ke_{s}_GeV": float(ke[s].sum() / cnt[s].sum()) for s in SPECIES})
res["all_r"] = allc
# The band most detectors sit in (100 m - 3 km from the axis): per-species mean kinetic energy, i.e. the per-species
# scaling between the kernel's E (GeV) and a particle count. Read by modules/optimize/background.py (KE_PER_PARTICLE_GEV).
band = (R_E[:-1] >= 100) & (R_E[1:] <= 3000)
res["band_100m_3km"] = dict(**{f"mean_ke_{s}_GeV": float(ke[s][band].sum() / cnt[s][band].sum()) for s in SPECIES},
                            charged_per_GeV_all=float(n_ch[band].sum() / tot_ke[band].sum()))
json.dump(res, open(os.path.join(args.out, "c8_charged_ke.json"), "w"), indent=1)
print("\nall r:", json.dumps(allc))
print("100 m - 3 km:", json.dumps(res["band_100m_3km"]))
for b in res["bins"]:
    print(f"{b['r']:>14s}  mean KE e {b['mean_ke_e_GeV'] or 0:.3f}  mu {b['mean_ke_mu_GeV'] or 0:.3f}  gamma "
          f"{b['mean_ke_gamma_GeV'] or 0:.3f}  charged {b['mean_ke_charged_GeV'] or 0:.3f} GeV  |  charged per GeV(all) "
          f"{b['charged_per_GeV_all'] or 0:.3f}  |  n_e {b['n_e']:.3g} n_mu {b['n_mu']:.3g} n_gamma {b['n_gamma']:.3g}")

fig, ax = plt.subplots(1, 3, figsize=(20, 5.6))
rc = np.sqrt(np.maximum(R_E[:-1], 1) * np.minimum(R_E[1:], 3e4))
col = {"e": "#2a78d6", "mu": "#c0392b", "gamma": "#e8a33a"}
for s in SPECIES:
    v = np.where(cnt[s] > 0, ke[s] / np.maximum(cnt[s], 1), np.nan)
    ax[0].plot(rc, v, "o-", color=col[s], lw=2, label=s)
    ax[2].hist(np.log10(np.concatenate(ke_samples[s])), bins=60, histtype="step", lw=2, color=col[s], density=True, label=s)
ax[0].plot(rc, np.where(n_ch > 0, (ke["e"] + ke["mu"]) / np.maximum(n_ch, 1), np.nan), "k--", lw=2, label="charged (e + mu)")
ax[1].plot(rc, np.where(tot_ke > 0, n_ch / np.maximum(tot_ke, 1e-30), np.nan), "ko-", lw=2)
for a in ax[:2]:
    a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=0.25, lw=0.5); a.set_xlabel("distance from the shower axis [m]")
ax[0].set(ylabel="mean kinetic energy per particle [GeV]", title="Mean kinetic energy per particle")
ax[1].set(ylabel="charged particles per GeV of kinetic energy (e + mu + gamma)", title="Charged particles per GeV (n_ch / eps)")
ax[2].set(xlabel="log10 kinetic energy [GeV]", ylabel="density", title="Kinetic energy spectrum (all r)"); ax[2].grid(alpha=0.25, lw=0.5)
for a in ax:
    if a.get_legend_handles_labels()[0]:
        a.legend(frameon=False, fontsize=10)
fig.suptitle(f"CORSIKA 8, {len(files)} pi- showers (0.11-1.2e6 GeV), all observation planes · cuts 10 MeV e/gamma, 0.5 GeV mu")
fig.tight_layout(); fig.savefig(os.path.join(args.out, "c8_charged_ke.png"), dpi=120, facecolor="white")
print("[write]", os.path.join(args.out, "c8_charged_ke.json"), os.path.join(args.out, "c8_charged_ke.png"))
