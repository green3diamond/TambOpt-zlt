"""Does the relation between distance from the axis, kinetic energy and particle count change with the observation
plane (depth along the shower)? c8_scaling.json pools all planes; this splits by plane.

Input: all_showers.parquet (pi+ showers, 1e5 - 5e7 GeV; every particle has KE >= 1 GeV, so e and gamma describe only
that tail; muons are nearly complete). Frame and r as c8_ke_per_particle_check.py. Per plane index and species, pooled
over showers: KE per particle(r), particle density per shower(r) [1/m^2] and energy density per shower(r) [GeV/m^2],
thinning weights applied. Writes c8_ke_per_plane.json and .png into OUT.

    python scripts/diagnostics/c8_ke_per_plane.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

SRC = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo/all_showers.parquet"
OUT = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/c8_calibration"
R_E = np.logspace(0, 4, 21); RC = np.sqrt(R_E[:-1] * R_E[1:]); AREA = np.pi * (R_E[1:] ** 2 - R_E[:-1] ** 2)
SP = {"e": 11, "mu": 13, "gamma": 22}
COLS = ["event_id", "plane", "pdg", "kinetic_energy", "weight", "X_transformed", "Y_transformed", "Z_transformed"]

pf = pq.ParquetFile(SRC)
c = {k: [] for k in COLS}
for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g, columns=COLS)
    for k in COLS:
        a = t.column(k).to_numpy()
        c[k].append(a.astype(np.float64) if k.endswith("_transformed") else a.astype(np.float32))
c = {k: np.concatenate(v) for k, v in c.items()}
order = np.argsort(c["event_id"], kind="stable"); c = {k: v[order] for k, v in c.items()}
ev = c["event_id"]; st = np.flatnonzero(np.r_[True, ev[1:] != ev[:-1]]); en = np.r_[st[1:], len(ev)]
planes = np.unique(c["plane"]).astype(int)
print(f"{len(st)} showers, planes {planes.tolist()}", flush=True)

KE = {(q, s): np.zeros(len(RC)) for q in planes for s in SP}
N = {(q, s): np.zeros(len(RC)) for q in planes for s in SP}
NR = {(q, s): np.zeros(len(RC)) for q in planes for s in SP}
nsh = {int(q): 0 for q in planes}
for a, b in zip(st, en):
    if b - a < 5000:
        continue
    P = np.stack([c["X_transformed"][a:b], c["Y_transformed"][a:b], c["Z_transformed"][a:b]], 1)
    pl = c["plane"][a:b].astype(np.int64); pdg = np.abs(c["pdg"][a:b]).astype(np.int64)
    k = c["kinetic_energy"][a:b].astype(np.float64); ww = c["weight"][a:b].astype(np.float64)
    big = np.bincount(pl - pl.min()).argmax() + pl.min(); mb = pl == big
    d = P[mb] - P[mb].mean(0)
    n = np.linalg.svd(d[np.random.default_rng(0).choice(len(d), min(len(d), 50000), replace=False)], full_matrices=False)[2][-1]
    e1 = np.cross([0, 0, 1.0], n) if abs(n[2]) < 0.99 else np.array([1.0, 0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    for q in np.unique(pl):
        m = pl == q
        if m.sum() < 1000:
            continue
        nsh[int(q)] += 1
        dd = P[m] - np.average(P[m], axis=0, weights=np.maximum(k[m] * ww[m], 1e-12))
        r = np.hypot(dd @ e1, dd @ e2)
        for s, pid in SP.items():
            sel = pdg[m] == pid
            KE[(q, s)] += np.histogram(r[sel], bins=R_E, weights=k[m][sel] * ww[m][sel])[0]
            N[(q, s)] += np.histogram(r[sel], bins=R_E, weights=ww[m][sel])[0]
            NR[(q, s)] += np.histogram(r[sel], bins=R_E)[0]

res = dict(source=SRC, r_m=RC.tolist(), showers_per_plane=nsh, planes={})
for q in planes:
    res["planes"][int(q)] = {}
    for s in SP:
        ok = NR[(q, s)] >= 100; ns = max(nsh[int(q)], 1)
        res["planes"][int(q)][s] = dict(
            ke_per_particle=np.where(ok, KE[(q, s)] / np.maximum(N[(q, s)], 1e-30), np.nan).tolist(),
            particle_density=np.where(ok, N[(q, s)] / AREA / ns, np.nan).tolist(),
            energy_density=np.where(ok, KE[(q, s)] / AREA / ns, np.nan).tolist())
json.dump(res, open(os.path.join(OUT, "c8_ke_per_plane.json"), "w"), indent=1)
print("showers per plane:", nsh)
for s in SP:
    print(f"\n{s}: KE per particle [GeV] by plane (columns) vs r (rows)")
    use = [q for q in planes if nsh[int(q)] >= 10]
    print(f"{'r [m]':>7} " + " ".join(f"{'p' + str(int(q)):>7}" for q in use))
    for j in range(0, len(RC), 2):
        print(f"{RC[j]:7.0f} " + " ".join(f"{res['planes'][int(q)][s]['ke_per_particle'][j]:7.3g}" for q in use))

use = [q for q in planes if nsh[int(q)] >= 10]
cmap = plt.get_cmap("plasma"); norm = plt.Normalize(min(use), max(use))
fig, ax = plt.subplots(3, 3, figsize=(19, 14), sharex=True)
for j, s in enumerate(SP):
    for q in use:
        d = res["planes"][int(q)][s]; col = cmap(norm(q))
        ax[0, j].loglog(RC, d["ke_per_particle"], color=col, lw=1.3)
        ax[1, j].loglog(RC, d["particle_density"], color=col, lw=1.3)
        ax[2, j].loglog(RC, d["energy_density"], color=col, lw=1.3)
    ax[0, j].set_title(s + ("" if s == "mu" else "  (only particles > 1 GeV)"))
    ax[0, j].set_ylabel("KE per particle [GeV]"); ax[1, j].set_ylabel("particles / m² per shower")
    ax[2, j].set_ylabel("KE / m² per shower [GeV]"); ax[2, j].set_xlabel("distance from the axis [m]")
    for a in ax[:, j]:
        a.grid(alpha=0.25, lw=0.5); a.set_xlim(1, 1e4)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="plane index", shrink=0.5)
fig.suptitle(f"Per observation plane · {len(st)} CORSIKA 8 π⁺ showers (all_showers.parquet), planes with ≥ 10 showers")
fig.savefig(os.path.join(OUT, "c8_ke_per_plane.png"), dpi=110, facecolor="white")
print("[write]", os.path.join(OUT, "c8_ke_per_plane.png"))
