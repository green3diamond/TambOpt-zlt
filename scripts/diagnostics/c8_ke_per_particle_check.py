"""Check the per-species kinetic energy per particle KE_s(r) of modules/optimize/c8_scaling.json (16 CORSIKA 8 pi-
showers; it sets the particle counts of the counting model and the SWGO trigger) on a larger corpus over a wide range
of primary energies: all_showers.parquet (99 pi+ showers, 1e5 - 5e7 GeV, one row per particle with thinning weight).

Same method as c8_ring_radius_check.py: per shower, n = normal of the most populated plane (SVD of its positions),
e1 = horizontal across the heading, e2 = n x e1; per plane (>= 1000 particles) offsets from the energy-weighted
centre, r = sqrt(t1^2 + t2^2); KE_s(r) = sum KE / count in log-r bins (bins with < 200 particles dropped).
Counts and energies are weighted with the thinning weight (also printed unweighted). Per primary-energy decade and
pooled, compared with the table. Writes c8_ke_per_particle_check.json and .png into OUT.

    python scripts/diagnostics/c8_ke_per_particle_check.py
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
TAB = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../modules/optimize/c8_scaling.json")))
R_E = np.logspace(0, 4, 29); RC = np.sqrt(R_E[:-1] * R_E[1:])
SP = {"e": 11, "mu": 13, "gamma": 22}
GRP = {"1e5-1e6": (1e5, 1e6), "1e6-1e7": (1e6, 1e7), "1e7-1e8": (1e7, 1e8), "all": (0, np.inf)}
COLS = ["event_id", "plane", "pdg", "kinetic_energy", "weight", "X_transformed", "Y_transformed", "Z_transformed",
        "primary_kinetic_energy"]

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
w = c["weight"]
print(f"{len(ev)} particles, {len(st)} showers; weight min {w.min():.3g} max {w.max():.3g} mean {w.mean():.3g}, "
      f"fraction != 1: {(w != 1).mean():.3g}", flush=True)

acc = {(gk, s, u): (np.zeros(len(RC)), np.zeros(len(RC))) for gk in GRP for s in SP for u in ("w", "u")}
n_sh = {gk: 0 for gk in GRP}
per, per_E = {}, []                                       # per-shower KE(r) per species; primary energies
for a, b in zip(st, en):
    if b - a < 5000:
        continue
    E0 = float(c["primary_kinetic_energy"][a])
    P = np.stack([c["X_transformed"][a:b], c["Y_transformed"][a:b], c["Z_transformed"][a:b]], 1)
    pl = c["plane"][a:b].astype(np.int64); pdg = np.abs(c["pdg"][a:b]).astype(np.int64)
    k = c["kinetic_energy"][a:b].astype(np.float64); ww = c["weight"][a:b].astype(np.float64)
    big = np.bincount(pl - pl.min()).argmax() + pl.min(); mb = pl == big
    d = P[mb] - P[mb].mean(0)
    n = np.linalg.svd(d[np.random.default_rng(0).choice(len(d), min(len(d), 50000), replace=False)], full_matrices=False)[2][-1]
    e1 = np.cross([0, 0, 1.0], n) if abs(n[2]) < 0.99 else np.array([1.0, 0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    r = np.full(b - a, np.nan)
    for q in np.unique(pl):
        m = pl == q
        if m.sum() < 1000:
            continue
        dd = P[m] - np.average(P[m], axis=0, weights=np.maximum(k[m] * ww[m], 1e-12))
        r[m] = np.hypot(dd @ e1, dd @ e2)
    ok = np.isfinite(r)
    for s, pid in SP.items():                             # per-shower curve (weighted), bins with >= 50 particles
        sel = ok & (pdg == pid)
        ke_s = np.histogram(r[sel], bins=R_E, weights=k[sel] * ww[sel])[0]
        n_s = np.histogram(r[sel], bins=R_E, weights=ww[sel])[0]; n_r = np.histogram(r[sel], bins=R_E)[0]
        per.setdefault(s, []).append(np.where(n_r >= 50, ke_s / np.maximum(n_s, 1e-30), np.nan))
    per_E.append(E0)
    for gk, (lo, hi) in GRP.items():
        if not lo <= E0 < hi:
            continue
        n_sh[gk] += 1
        for s, pid in SP.items():
            sel = ok & (pdg == pid)
            for u, wt in (("w", ww), ("u", np.ones_like(ww))):
                acc[(gk, s, u)][0][:] += np.histogram(r[sel], bins=R_E, weights=k[sel] * wt[sel])[0]
                acc[(gk, s, u)][1][:] += np.histogram(r[sel], bins=R_E, weights=wt[sel])[0]
    print(f"event {int(ev[a])}: E {E0:.3g} GeV, {b - a} particles", flush=True)

nraw = {}
for gk in GRP:
    for s in SP:
        nraw[(gk, s)] = acc[(gk, s, "u")][1]
res = dict(source=f"{len(st)} pi+ showers, {SRC}", r_m=RC.tolist(), n_showers=n_sh, ke_gev={}, table_ke_gev={})
for s in SP:
    res["table_ke_gev"][s] = TAB["species"][s]["ke_gev"]
    for gk in GRP:
        for u in ("w", "u"):
            ke, nn = acc[(gk, s, u)]
            v = np.where(nraw[(gk, s)] >= 200, ke / np.maximum(nn, 1e-30), np.nan)
            res["ke_gev"][f"{s}_{gk}_{u}"] = v.tolist()
json.dump(res, open(os.path.join(OUT, "c8_ke_per_particle_check.json"), "w"), indent=1)

print("\nshowers per group:", n_sh)
for s in SP:
    tab = np.array([np.nan if x is None else x for x in TAB["species"][s]["ke_gev"]], float)
    print(f"\n{s}: KE per particle [GeV] (weighted); ratio to the 16-shower table in ()")
    print(f"{'r [m]':>7} {'table':>8} " + " ".join(f"{gk:>16}" for gk in GRP) + f" {'all, unweighted':>16}")
    for j in range(0, len(RC)):
        if not 3 <= RC[j] <= 5000:
            continue
        row = [f"{res['ke_gev'][f'{s}_{gk}_w'][j]:7.3g} ({res['ke_gev'][f'{s}_{gk}_w'][j] / tab[j]:4.2f})" for gk in GRP]
        uu = res['ke_gev'][f'{s}_all_u'][j]
        print(f"{RC[j]:7.0f} {tab[j]:8.3g} " + " ".join(f"{x:>16}" for x in row) + f" {uu:7.3g} ({uu / tab[j]:4.2f})")
    bsel = (R_E[:-1] >= 100) & (R_E[1:] <= 3000)
    for gk in GRP:
        ke, nn = acc[(gk, s, "w")]
        print(f"  band 100 m - 3 km, {gk}: {ke[bsel].sum() / max(nn[bsel].sum(), 1e-30):.3g} GeV "
              f"(table {TAB['band_100m_3km'][s]['ke_gev']:.3g})")

fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
cols = {"1e5-1e6": "#2a78d6", "1e6-1e7": "#1baf7a", "1e7-1e8": "#c0392b", "all": "k"}
for a, s in zip(ax, SP):
    tab = np.array([np.nan if x is None else x for x in TAB["species"][s]["ke_gev"]], float)
    a.loglog(RC, tab, color="#888", lw=3, alpha=0.6, label="table (16 π⁻ showers)")
    for gk in GRP:
        a.loglog(RC, res["ke_gev"][f"{s}_{gk}_w"], color=cols[gk], lw=2 if gk == "all" else 1.4,
                 label=f"{gk} GeV ({n_sh[gk]} showers)" if gk != "all" else f"all ({n_sh[gk]})")
    a.set(xlabel="distance from the axis [m]", ylabel="kinetic energy per particle [GeV]", title=s, xlim=(1, 1e4))
    a.grid(alpha=0.25, lw=0.5); a.legend(fontsize=9, frameon=False)
fig.suptitle(f"KE per particle vs r: table vs {len(st)} CORSIKA 8 π⁺ showers (all_showers.parquet), by primary energy")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_ke_per_particle_check.png"), dpi=120, facecolor="white")
print("[write]", os.path.join(OUT, "c8_ke_per_particle_check.png"))

# shower-to-shower differences: each shower's KE(r), coloured by log10 primary energy; median and 16-84 % over showers;
# bottom row: each shower divided by the median of all showers
per_E = np.array(per_E); lE = np.log10(per_E)
cmap = plt.get_cmap("viridis"); norm = plt.Normalize(lE.min(), lE.max())
fig, ax = plt.subplots(2, 3, figsize=(19, 9.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
spread = {}
for j, s in enumerate(SP):
    A = np.array(per[s]); med = np.nanmedian(A, 0); lo, hi = np.nanpercentile(A, [16, 84], axis=0)
    tab = np.array([np.nan if x is None else x for x in TAB["species"][s]["ke_gev"]], float)
    for row, e in zip(A, lE):
        ax[0, j].loglog(RC, row, color=cmap(norm(e)), lw=0.7, alpha=0.6)
        ax[1, j].semilogx(RC, row / med, color=cmap(norm(e)), lw=0.7, alpha=0.6)
    ax[0, j].loglog(RC, med, "k", lw=2.2, label="median over showers")
    ax[0, j].fill_between(RC, lo, hi, color="k", alpha=0.12, label="16–84 % over showers")
    ax[0, j].loglog(RC, tab, color="#888", lw=3, alpha=0.6, label="table (16 π⁻, cuts 10 MeV e/γ, 0.5 GeV μ)")
    ax[1, j].axhline(1, color="k", lw=1)
    ax[0, j].set(title=f"{s}" + ("" if s == "mu" else "  (only particles > 1 GeV in this file)"), ylabel="KE per particle [GeV]")
    ax[1, j].set(xlabel="distance from the axis [m]", ylabel="shower / median", ylim=(0, 3), xlim=(1, 1e4))
    for a in ax[:, j]:
        a.grid(alpha=0.25, lw=0.5)
    ax[0, j].legend(fontsize=8, frameon=False)
    with np.errstate(invalid="ignore"):
        spread[s] = ((hi - lo) / 2 / med).tolist()
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=ax, label="log10 primary energy [GeV]", shrink=0.6)
fig.suptitle(f"Shower-to-shower differences of KE per particle · {len(per_E)} CORSIKA 8 π⁺ showers (all_showers.parquet, cut 1 GeV)")
fig.savefig(os.path.join(OUT, "c8_ke_per_shower.png"), dpi=120, facecolor="white")
print("[write]", os.path.join(OUT, "c8_ke_per_shower.png"))
print("\nrelative shower-to-shower spread ((p84 - p16) / 2 / median):")
for jr in range(0, len(RC), 2):
    if 3 <= RC[jr] <= 5000:
        print(f"{RC[jr]:7.0f} m  " + "  ".join(f"{s}: {spread[s][jr]:.2f}" for s in SP))
