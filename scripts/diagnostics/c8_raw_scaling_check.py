"""Check the counting-model table (modules/optimize/c8_scaling.json: kinetic energy per particle KE_s(r) per species,
16 CORSIKA 8 pi- showers, all planes pooled, unweighted) against 100 raw CORSIKA 8 showers with the same cuts.

Input: Hamza's tambo_simulations_test_raw_v2 on netscratch (submit_jobs.py, --emcut 0.01 --hadcut 0.5 --mucut 0.5,
injection 2,500 m, parquet kept): 50 e- primaries (1e5 - 1e8 GeV) and 50 pi+ primaries (1e5 - 1.8e7 GeV), 24
observation planes perpendicular to the axis (particles, particles1 ... particles23), one row per particle with
thinning weight. Per shower and plane: r = distance in the plane from the energy-weighted centre (as
c8_ring_radius_check.py); KE per particle = sum(w KE) / sum(w) in log-r bins. Plane depth = distance along the axis
from the plane centres in each plane's config.yaml.

Figures (OUT):
  c8_raw_by_energy.png : rows species, columns primary type; median over showers (all planes pooled per shower) and
                         16-84 % band, per primary-energy decade; grey: the table
  c8_raw_by_plane.png  : same, but per depth group of planes (all energies)
  c8_raw_counts.png    : particles per shower and plane vs depth (median, 16-84 %), per species and primary type
Also c8_raw_scaling_check.json (per-shower curves).

    python scripts/diagnostics/c8_raw_scaling_check.py [n_workers]
"""
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

SRC = "/n/netscratch/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_test_raw_v2"
OUT = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/c8_calibration"
TAB = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../modules/optimize/c8_scaling.json")))
R_E = np.logspace(0, 4, 21); RC = np.sqrt(R_E[:-1] * R_E[1:])
SP = {"e": 11, "mu": 13, "gamma": 22}
PRIM = {"pdg_211": "π⁺", "pdg_11": "e⁻"}
MIN_N = 20                                                   # particles (rows) per bin for a per-shower value


def plane_geom(cfg):
    txt = open(cfg).read()
    c = [float(v) for v in re.findall(r"center:\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)", txt)[0]]
    n = [float(v) for v in re.findall(r"normal:\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)", txt)[0]]
    return np.array(c), np.array(n) / np.linalg.norm(n)


def one(sdir):
    args = open(f"{sdir}/config.yaml").read()
    E0 = float(re.search(r"--energy (\S+)", args).group(1))
    planes = sorted(glob.glob(f"{sdir}/particles*/particles.parquet"))
    geo = {os.path.basename(os.path.dirname(p)): plane_geom(os.path.join(os.path.dirname(p), "config.yaml")) for p in planes}
    n = next(iter(geo.values()))[1]
    s = {k: float(v[0] @ n) for k, v in geo.items()}
    s0 = min(s.values())
    out = dict(E0=E0, prim=os.path.basename(os.path.dirname(os.path.dirname(sdir))), planes={})
    pooled = {sp: [np.zeros(len(RC)), np.zeros(len(RC)), np.zeros(len(RC))] for sp in SP}   # sum wKE, sum w, rows
    for p in planes:
        name = os.path.basename(os.path.dirname(p))
        t = pq.read_table(p, columns=["pdg", "kinetic_energy", "x", "y", "weight"])
        pdg = np.abs(t.column("pdg").to_numpy()); k = t.column("kinetic_energy").to_numpy().astype(np.float64)
        x = t.column("x").to_numpy().astype(np.float64); y = t.column("y").to_numpy().astype(np.float64)
        w = t.column("weight").to_numpy().astype(np.float64)
        if len(k) < 100:
            continue
        x0, y0 = np.average(x, weights=k * w + 1e-30), np.average(y, weights=k * w + 1e-30)
        r = np.hypot(x - x0, y - y0)
        d = dict(depth=s[name] - s0, n={}, ke={})
        for sp, pid in SP.items():
            m = pdg == pid
            ke = np.histogram(r[m], bins=R_E, weights=k[m] * w[m])[0]
            nw = np.histogram(r[m], bins=R_E, weights=w[m])[0]
            nr = np.histogram(r[m], bins=R_E)[0]
            for a, b in zip(pooled[sp], (ke, nw, nr)):
                a += b
            d["ke"][sp] = np.where(nr >= MIN_N, ke / np.maximum(nw, 1e-30), np.nan).tolist()
            d["n"][sp] = float(w[m].sum())
        out["planes"][name] = d
    out["ke_pooled"] = {sp: np.where(v[2] >= MIN_N, v[0] / np.maximum(v[1], 1e-30), np.nan).tolist() for sp, v in pooled.items()}
    out["ke_pooled_unweighted_rows"] = {sp: v[2].tolist() for sp, v in pooled.items()}
    print(f"{out['prim']} E={E0:.3g} planes={len(out['planes'])}", flush=True)
    return out


def band(rows):
    A = np.array(rows, float)
    with np.errstate(all="ignore"):
        return np.nanmedian(A, 0), np.nanpercentile(A, 16, 0), np.nanpercentile(A, 84, 0)


if __name__ == "__main__":
    dirs = sorted(d.rstrip("/") for p in PRIM for d in glob.glob(f"{SRC}/{p}/*/*/"))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        res = [r for r in pool.map(one, dirs) if r["planes"]]
    json.dump(res, open(os.path.join(OUT, "c8_raw_scaling_check.json"), "w"))
    tab = {sp: np.array([np.nan if v is None else v for v in TAB["species"][sp]["ke_gev"]], float) for sp in SP}
    rt = np.array(TAB["r_m"])
    depths = sorted({round(d["depth"]) for r in res for d in r["planes"].values()})
    print("plane depths along the axis [m]:", depths)
    print("showers:", {p: sum(r["prim"] == p for r in res) for p in PRIM})

    # 1) by primary energy
    DEC = [(1e5, 1e6), (1e6, 1e7), (1e7, 1e8 + 1)]
    cols = ["#2a78d6", "#1baf7a", "#c0392b"]
    fig, ax = plt.subplots(3, 2, figsize=(14, 14), sharex=True)
    for j, (pk, pl) in enumerate(PRIM.items()):
        for i, sp in enumerate(SP):
            a = ax[i, j]
            a.loglog(rt, tab[sp], color="#777", lw=3, alpha=0.6, label="table: 16 π⁻ showers, planes pooled")
            for (lo, hi), c in zip(DEC, cols):
                rows = [r["ke_pooled"][sp] for r in res if r["prim"] == pk and lo <= r["E0"] < hi]
                if not rows:
                    continue
                med, l16, h84 = band(rows)
                a.loglog(RC, med, color=c, lw=2, label=f"{lo:.0e}–{min(hi, 1e8):.0e} GeV ({len(rows)} showers)")
                a.fill_between(RC, l16, h84, color=c, alpha=0.15)
            a.set(title=f"{sp}, primary {pl}", ylabel="kinetic energy per particle [GeV]")
            a.grid(alpha=0.25, lw=0.5); a.legend(fontsize=8, frameon=False)
    for a in ax[-1]:
        a.set_xlabel("distance from the shower centre in the plane [m]"); a.set_xlim(1, 1e4)
    fig.suptitle("KE per particle vs r by primary energy · 100 raw CORSIKA 8 showers (cuts 10 MeV e/γ, 0.5 GeV μ)\n"
                 "lines: median over showers (planes pooled per shower); bands: 16–84 % over showers")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_raw_by_energy.png"), dpi=110, facecolor="white")

    # 2) by plane depth
    groups = np.array_split(np.array(depths), 6)
    cmap = plt.get_cmap("plasma")
    fig, ax = plt.subplots(3, 2, figsize=(14, 14), sharex=True)
    for j, (pk, pl) in enumerate(PRIM.items()):
        for i, sp in enumerate(SP):
            a = ax[i, j]
            a.loglog(rt, tab[sp], color="#777", lw=3, alpha=0.6, label="table: 16 π⁻ showers, planes pooled")
            for g, grp in enumerate(groups):
                rows = [d["ke"][sp] for r in res if r["prim"] == pk for d in r["planes"].values()
                        if round(d["depth"]) in set(grp.tolist())]
                if not rows:
                    continue
                med, l16, h84 = band(rows)
                c = cmap(g / max(len(groups) - 1, 1) * 0.9)
                a.loglog(RC, med, color=c, lw=2, label=f"depth {grp.min() / 1e3:.1f}–{grp.max() / 1e3:.1f} km")
                a.fill_between(RC, l16, h84, color=c, alpha=0.12)
            a.set(title=f"{sp}, primary {pl}", ylabel="kinetic energy per particle [GeV]")
            a.grid(alpha=0.25, lw=0.5); a.legend(fontsize=8, frameon=False)
    for a in ax[-1]:
        a.set_xlabel("distance from the shower centre in the plane [m]"); a.set_xlim(1, 1e4)
    fig.suptitle("KE per particle vs r by plane depth (distance along the axis from the first plane) · all energies\n"
                 "lines: median over showers × planes in the group; bands: 16–84 %")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_raw_by_plane.png"), dpi=110, facecolor="white")

    # 3) particle numbers per plane vs depth, by primary energy decade
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5), sharex=True)
    for i, sp in enumerate(SP):
        a = ax[i]
        for pk, ls in zip(PRIM, ("-", "--")):
            for (lo, hi), c in zip(DEC, cols):
                sel = [r for r in res if r["prim"] == pk and lo <= r["E0"] < hi]
                if not sel:
                    continue
                M = np.array([[r["planes"].get(nm, {}).get("n", {}).get(sp, np.nan) for nm in
                               sorted(r["planes"], key=lambda q: r["planes"][q]["depth"])] for r in sel
                              if len(r["planes"]) == len(depths)], float)
                if len(M) == 0:
                    continue
                med, l16, h84 = band(M)
                a.semilogy(np.array(depths) / 1e3, med, ls, color=c, lw=1.8, label=f"{PRIM[pk]} {lo:.0e}–{min(hi, 1e8):.0e}")
                a.fill_between(np.array(depths) / 1e3, l16, h84, color=c, alpha=0.1)
        a.set(title=sp, xlabel="plane depth along the axis [km]", ylabel="particles per plane (weighted)")
        a.grid(alpha=0.25, lw=0.5); a.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Particle numbers per plane vs depth, by primary type (solid π⁺, dashed e⁻) and energy; 16–84 % over showers")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_raw_counts.png"), dpi=110, facecolor="white")

    # numbers: band values 100 m - 3 km, ratio to the table, per primary type and decade
    bsel = (R_E[:-1] >= 100) & (R_E[1:] <= 3000)
    for pk in PRIM:
        for (lo, hi) in DEC:
            rows = [r for r in res if r["prim"] == pk and lo <= r["E0"] < hi]
            if rows:
                med = {sp: np.nanmedian([np.nanmedian(np.array(r["ke_pooled"][sp], float)[bsel]) for r in rows]) for sp in SP}
                print(f"{PRIM[pk]} {lo:.0e}-{hi:.0e}: {len(rows)} showers; band KE/particle " +
                      " ".join(f"{sp} {med[sp]:.3g} (table {TAB['band_100m_3km'][sp]['ke_gev']:.3g})" for sp in SP))
    print("[write] c8_raw_by_energy.png c8_raw_by_plane.png c8_raw_counts.png in", OUT)
