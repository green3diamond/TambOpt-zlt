"""Are the deposited kinetic energy and the particle count in a detector proportional? CORSIKA 8 per-particle check.

Same 16 pi- showers and shower frame as c8_ring_radius_check.py. Each plane is cut into square cells of 2 m^2 (the
detector area) in the shower-plane coordinates (t1, t2); per cell with >= 1 particle: summed kinetic energy KE and
count N, per species (e, mu, gamma) and for all species. Per true-r bin (cell centre):
    median and 16-84 % of KE / N over cells     (proportional at fixed r if the band is narrow)
    Pearson correlation of log KE and log N over cells with N >= 2
    mean N per cell
If KE / N changes with r, energy and count are proportional only at fixed distance; the kernel's particles_per_E(s, r)
carries that factor. Writes c8_energy_count_check.json and .png into OUT.

    python scripts/diagnostics/c8_energy_count_check.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

SRC = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training2/pdg_-211"
OUT = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/c8_calibration"
CELL = np.sqrt(2.0)                                                         # 2 m^2 cells
R_E = np.logspace(0, 4, 17)                                                 # 1 m - 10 km
RC = np.sqrt(R_E[:-1] * R_E[1:])
SP = {"all": None, "e": 11, "mu": 13, "gamma": 22}
files = sorted(glob.glob(f"{SRC}/energy_*/*/*_hits.parquet"))
print("schema:", pq.read_schema(files[0]), flush=True)
ratio = {s: [[] for _ in RC] for s in SP}                                   # KE / N per cell, per r bin
lke = {s: [[] for _ in RC] for s in SP}
ln = {s: [[] for _ in RC] for s in SP}
for f in files:
    t = pq.read_table(f, columns=["x", "y", "z", "pdg", "kinetic_energy", "plane_index"])
    P = np.stack([t.column(c).to_numpy().astype(np.float64) for c in ("x", "y", "z")], 1)
    pdg, k, pl = np.abs(t.column("pdg").to_numpy()), t.column("kinetic_energy").to_numpy().astype(np.float64), t.column("plane_index").to_numpy()
    big = np.bincount(pl).argmax(); mb = pl == big
    d = P[mb] - P[mb].mean(0)
    n = np.linalg.svd(d[np.random.default_rng(0).choice(len(d), min(len(d), 50000), replace=False)], full_matrices=False)[2][-1]
    e1 = np.array([-n[1], n[0], 0.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    for q in np.unique(pl):
        m = pl == q
        if m.sum() < 1000:
            continue
        dd = P[m] - np.average(P[m], axis=0, weights=np.maximum(k[m], 1e-12))
        i1, i2 = np.floor(dd @ e1 / CELL).astype(np.int64), np.floor(dd @ e2 / CELL).astype(np.int64)
        key = (i1 - i1.min()) * (i2.max() - i2.min() + 1) + (i2 - i2.min())
        for s, pid in SP.items():
            sel = np.ones(m.sum(), bool) if pid is None else pdg[m] == pid
            if sel.sum() == 0:
                continue
            u, inv = np.unique(key[sel], return_inverse=True)
            N = np.bincount(inv).astype(np.float64); KE = np.bincount(inv, weights=k[m][sel])
            c1 = np.bincount(inv, weights=i1[sel]) / N; c2 = np.bincount(inv, weights=i2[sel]) / N
            r = np.hypot((c1 + 0.5) * CELL, (c2 + 0.5) * CELL)
            b = np.digitize(r, R_E) - 1
            for j in range(len(RC)):
                w = b == j
                if w.any():
                    ratio[s][j].append(KE[w] / N[w])
                    w2 = w & (N >= 2) & (KE > 0)
                    lke[s][j].append(np.log(KE[w2])); ln[s][j].append(np.log(N[w2]))
    print(os.path.basename(f), flush=True)

res = dict(source=f"{len(files)} CORSIKA 8 pi- showers, {SRC}", cell_m2=CELL ** 2, r_m=RC.tolist(), species={})
for s in SP:
    med, lo, hi, cor, ncell = [], [], [], [], []
    for j in range(len(RC)):
        x = np.concatenate(ratio[s][j]) if ratio[s][j] else np.array([])
        a = np.concatenate(lke[s][j]) if lke[s][j] else np.array([]); bb = np.concatenate(ln[s][j]) if ln[s][j] else np.array([])
        ok = len(x) >= 50
        p = np.percentile(x, [16, 50, 84]) if ok else [np.nan] * 3
        lo.append(p[0]); med.append(p[1]); hi.append(p[2]); ncell.append(int(len(x)))
        cor.append(float(np.corrcoef(a, bb)[0, 1]) if len(a) >= 50 and bb.std() > 0 else np.nan)
    res["species"][s] = dict(ke_per_n_median=med, ke_per_n_p16=lo, ke_per_n_p84=hi, corr_logke_logn=cor, n_cells=ncell)
json.dump(res, open(os.path.join(OUT, "c8_energy_count_check.json"), "w"), indent=1)

print(f"\n{'r [m]':>7} " + " ".join(f"{s:>28}" for s in SP))
print(f"{'':>7} " + " ".join(f"{'KE/N med [16-84] GeV, corr':>28}" for s in SP))
for j in range(len(RC)):
    cells = []
    for s in SP:
        d = res["species"][s]
        cells.append(f"{d['ke_per_n_median'][j]:7.3g} [{d['ke_per_n_p16'][j]:.2g}-{d['ke_per_n_p84'][j]:.2g}] {d['corr_logke_logn'][j]:5.2f}")
    print(f"{RC[j]:7.0f} " + " ".join(f"{c:>28}" for c in cells))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
col = {"all": "k", "e": "#2a78d6", "mu": "#c0392b", "gamma": "#e8a33a"}
for s in SP:
    d = res["species"][s]
    ax[0].loglog(RC, d["ke_per_n_median"], color=col[s], lw=2, label=s)
    ax[0].fill_between(RC, d["ke_per_n_p16"], d["ke_per_n_p84"], color=col[s], alpha=0.12)
    ax[1].semilogx(RC, d["corr_logke_logn"], color=col[s], lw=2, label=s)
ax[0].set(xlabel="distance from the axis [m]", ylabel="KE / N per 2 m² cell [GeV]", title="Energy per particle (median, 16–84 % over cells)")
ax[1].set(xlabel="distance from the axis [m]", ylabel="corr(log KE, log N), cells with N ≥ 2", ylim=(0, 1), title="Proportionality over cells")
for a in ax:
    a.grid(alpha=0.25, lw=0.5); a.legend(frameon=False)
fig.suptitle(f"Kinetic energy vs particle count in 2 m² cells · {len(files)} CORSIKA 8 π⁻ showers")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_energy_count_check.png"), dpi=120, facecolor="white")
print("[write]", os.path.join(OUT, "c8_energy_count_check.png"))
