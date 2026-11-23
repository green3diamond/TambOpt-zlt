"""Axis reference of the counting-model table (audit, suspected): c8_scaling.json measures r from each plane's
energy-weighted centre, while the kernel measures r from the nominal axis (decay vertex + tau direction). This
measures the offset between the two on the 100 raw CORSIKA 8 showers of c8_raw_scaling_check.py and its effect on
the expected counts.

Per shower and plane (plane-local x, y; origin = the plane centre on the nominal axis, checked: the plane centres
lie on a line along the normal):
  offset   = |energy-weighted centre| (all particles, weights w KE) and per species (weights w)
  r50      = energy-weighted median distance from the centre
  mu ratio = per species, energy-weighted mean of KE_tab(r_centre) / KE_tab(r_nominal) in bins of r_nominal: the factor
             by which mu = E / KE_tab(r) changes when r is measured from the nominal axis (as the kernel does)
             instead of from the centre (as the table was built). 1 = no effect.
Also the angle between the shower axis and the CORSIKA field (c8_air_shower.cpp: field along the frame's +x).

Figures (OUT): c8_axis_offset.png (offset and r50 vs depth; mu ratio vs r). JSON: c8_axis_offset_check.json.

    python scripts/diagnostics/c8_axis_offset_check.py [n_workers]
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
B_DIR = np.array([22.7266e-6, -2.5322e-9, -4.2859e-9]); B_DIR /= np.linalg.norm(B_DIR)
RT = np.array(TAB["r_m"], float)


def ke_tab(sp, r):
    v = np.array([np.nan if x is None else x for x in TAB["species"][sp]["ke_gev"]], float)
    ok = np.isfinite(v)
    return np.exp(np.interp(np.log(np.maximum(r, RT[ok][0])), np.log(RT[ok]), np.log(v[ok])))


def vec(txt, key):
    return np.array([float(v) for v in re.findall(key + r":\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)", txt)[0]])


def one(sdir):
    E0 = float(re.search(r"--energy (\S+)", open(f"{sdir}/config.yaml").read()).group(1))
    planes = sorted(glob.glob(f"{sdir}/particles*/particles.parquet"))
    geo = {}
    for p in planes:
        txt = open(os.path.join(os.path.dirname(p), "config.yaml")).read()
        geo[os.path.basename(os.path.dirname(p))] = (vec(txt, "center"), vec(txt, "normal"))
    n = next(iter(geo.values()))[1]; n = n / np.linalg.norm(n)
    C = np.array([g[0] for g in geo.values()])
    s = (C - C.mean(0)) @ n
    perp = np.linalg.norm((C - C.mean(0)) - np.outer(s, n), axis=1).max()     # centres off the line along n [m]
    s0 = s.min()
    out = dict(E0=E0, prim=os.path.basename(os.path.dirname(os.path.dirname(sdir))), centre_line_dev_m=float(perp),
               sin_vB=float(np.linalg.norm(np.cross(n, B_DIR))), planes={})
    for (name, (c, _)), si in zip(geo.items(), s):
        p = os.path.join(sdir, name, "particles.parquet")
        t = pq.read_table(p, columns=["pdg", "kinetic_energy", "x", "y", "weight"])
        pdg = np.abs(t.column("pdg").to_numpy()); k = t.column("kinetic_energy").to_numpy().astype(np.float64)
        x = t.column("x").to_numpy().astype(np.float64); y = t.column("y").to_numpy().astype(np.float64)
        w = t.column("weight").to_numpy().astype(np.float64)
        if len(k) < 100:
            continue
        we = k * w + 1e-30
        x0, y0 = np.average(x, weights=we), np.average(y, weights=we)
        rc = np.hypot(x - x0, y - y0); rn = np.hypot(x, y)
        o = np.argsort(rc); cw = np.cumsum(we[o]); r50 = float(rc[o][np.searchsorted(cw, 0.5 * cw[-1])])
        d = dict(depth=float(si - s0), offset=float(np.hypot(x0, y0)), r50=r50, offset_sp={}, ratio={})
        for sp, pid in SP.items():
            m = pdg == pid
            if m.sum() < 20:
                continue
            d["offset_sp"][sp] = float(np.hypot(np.average(x[m], weights=w[m]), np.average(y[m], weights=w[m])))
            q = ke_tab(sp, rc[m]) / ke_tab(sp, rn[m])
            num = np.histogram(rn[m], bins=R_E, weights=we[m] * q)[0]
            den = np.histogram(rn[m], bins=R_E, weights=we[m])[0]
            cnt = np.histogram(rn[m], bins=R_E)[0]
            d["ratio"][sp] = np.where(cnt >= 20, num / np.maximum(den, 1e-300), np.nan).tolist()
        out["planes"][name] = d
    print(f"{out['prim']} E={E0:.3g} sin(v,B)={out['sin_vB']:.2f} centre-line dev={perp:.2g} m "
          f"max offset={max(d['offset'] for d in out['planes'].values()):.1f} m", flush=True)
    return out


def band(rows):
    A = np.array(rows, float)
    with np.errstate(all="ignore"):
        return np.nanmedian(A, 0), np.nanpercentile(A, 16, 0), np.nanpercentile(A, 84, 0)


if __name__ == "__main__":
    dirs = sorted(d.rstrip("/") for p in PRIM for d in glob.glob(f"{SRC}/{p}/*/*/"))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        res = [r for r in pool.map(one, dirs) if r["planes"]]
    json.dump(res, open(os.path.join(OUT, "c8_axis_offset_check.json"), "w"))
    print("max plane-centre deviation from a line along the normal [m]:", max(r["centre_line_dev_m"] for r in res))

    depths = sorted({round(d["depth"]) for r in res for d in r["planes"].values()})
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for j, (pk, pl) in enumerate(PRIM.items()):
        rr = [r for r in res if r["prim"] == pk]
        a = ax[0, j]
        for r in rr:
            P = sorted(r["planes"].values(), key=lambda d: d["depth"])
            a.semilogy([d["depth"] / 1e3 for d in P], [max(d["offset"], 1e-2) for d in P],
                       color=plt.cm.viridis(r["sin_vB"]), lw=0.8, alpha=0.7)
        med = [np.median([d["r50"] for r in rr for d in r["planes"].values() if round(d["depth"]) == z]) for z in depths]
        a.semilogy(np.array(depths) / 1e3, med, "k--", lw=2, label="median r50 (energy-weighted)")
        a.set(title=f"{pl}: centre offset from the nominal axis (colour: sin(v, B))", xlabel="depth along the axis [km]",
              ylabel="offset [m]"); a.legend(); a.grid(alpha=0.3, which="both")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1)); fig.colorbar(sm, ax=ax[0, 1], label="sin(v, B)")
    a = ax[0, 2]
    for pk, pl in PRIM.items():
        rr = [r for r in res if r["prim"] == pk]
        a.semilogy([r["sin_vB"] for r in rr], [np.median([d["offset"] / max(d["r50"], 1e-3) for d in r["planes"].values()])
                                                for r in rr], "o", label=pl, alpha=0.7)
    a.set(title="median over planes of offset / r50", xlabel="sin(v, B)", ylabel="offset / r50"); a.legend(); a.grid(alpha=0.3, which="both")
    for i, sp in enumerate(SP):
        a = ax[1, i]
        for pk, pl in PRIM.items():
            rows = [d["ratio"][sp] for r in res if r["prim"] == pk for d in r["planes"].values() if sp in d["ratio"]]
            m, lo, hi = band(rows)
            a.semilogx(RC, m, lw=2, label=f"{pl} median"); a.fill_between(RC, lo, hi, alpha=0.2)
        a.axhline(1, color="k", lw=0.8)
        a.set(title=f"{sp}: mu(r from nominal axis) / mu(r from centre)", xlabel="distance from the nominal axis [m]",
              ylabel="ratio (energy-weighted, per shower and plane; 16-84 %)", xlim=(1, 1e4))
        a.legend(); a.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_axis_offset.png"), dpi=110)

    for pk, pl in PRIM.items():
        rr = [r for r in res if r["prim"] == pk]
        off = np.array([d["offset"] for r in rr for d in r["planes"].values()])
        rat = np.array([d["offset"] / max(d["r50"], 1e-3) for r in rr for d in r["planes"].values()])
        print(f"{pl}: {len(rr)} showers; offset median {np.median(off):.1f} m, 90 % {np.percentile(off, 90):.1f} m, "
              f"max {off.max():.0f} m; offset/r50 median {np.median(rat):.3f}, 90 % {np.percentile(rat, 90):.3f}")
        for sp in SP:
            rows = [d["ratio"][sp] for r in rr for d in r["planes"].values() if sp in d["ratio"]]
            m = band(rows)[0]
            pick = {f"{RC[b]:.0f} m": round(float(m[b]), 3) for b in (4, 8, 12, 16) if np.isfinite(m[b])}
            print(f"   {sp}: median mu ratio at {pick}")
    print("[write] c8_axis_offset.png c8_axis_offset_check.json in", OUT)
