"""Per shower, species and plane of repaired CORSIKA: energy, cell count and median arrival delay in rings of radius
about the shower axis, plus an NKG fit to each profile. The generator's in-plane laws are fitted to these.

Stages: hist (CPU array, one slice of the training showers per task), merge, laws (NKG fit quality, diagnostic).
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py
import numpy as np
from scipy.optimize import minimize

from analysis_core import logs
from data_lane import sources
from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere, lateral, m1, m2, unwrap

FOLDER = "annuli"
EDGES = np.array([0.0, 10, 20, 30, 50, 70, 100, 140, 200, 280, 400, 560, 800, 1100, 1600, 2300, 3200, 4500, 6400,
                  9000, np.inf])     # m, in-plane radius about the axis
N_BIN = len(EDGES) - 1
C_LIGHT = 299792458.0
MIN_CELLS = 5
FIT = ("ln_r50", "s", "ok")


def slice_rows(out, limit):
    """The training showers of the profile fits (m1); a smoke run takes `limit` of them evenly spread."""
    rows = np.load(os.path.join(out, m1.FOLDER, "fit", "fits_train.npz"))["meta.rows"]
    return rows[np.linspace(0, len(rows) - 1, limit).astype(int)] if limit else rows


def read_showers(s, rows):
    """Metadata and repaired cells (unwrap) of species s at the training rows."""
    with h5py.File(sources.corsika_path(s, "train"), "r") as f:
        meta = {"rows": rows, "energies": f["energies"][rows].reshape(-1).astype(np.float64),
                "directions": f["directions"][rows].astype(np.float64), "labels": f["pdg"][rows].astype(np.int64)}
        pts = [np.asarray(f["showers"][r], np.float64).reshape(-1, 5) for r in rows]
    pts = [unwrap.unwrap_shower(p[p[:, 3] > 0], d, N_PLANES)[0] for p, d in zip(pts, meta["directions"])]
    return meta, pts


def nkg_shares(edges, ln_r50, s, r_ref):
    r_s = r_ref * np.exp(ln_r50) / lateral.x_median(s)
    q = lateral.q(edges, r_s, s)
    return (q[:-1] - q[1:]) / (1 - q[-1])


def fit_profile(w, r_hi, r_ref):
    """NKG fit (ln(r50 / r_ref), s, converged) to a ring profile w, seen out to radius r_hi."""
    edges = np.minimum(EDGES, r_hi)
    n = int(np.searchsorted(EDGES, r_hi, side="left"))      # annuli that reach into [0, r_hi]
    edges, w = edges[:n + 1], w[:n] / w[:n].sum()

    def nll(u):
        return -np.sum(w * np.log(np.maximum(nkg_shares(edges, u[0], u[1], r_ref), 1e-300)))

    best = min((minimize(nll, np.array([np.log(0.1), s0]), method="L-BFGS-B",
                         bounds=[lateral.LN_R_BOUNDS, lateral.S_BOUNDS]) for s0 in lateral.S_STARTS),
               key=lambda r: r.fun)
    return best.x[0], best.x[1], float(best.success)


def plane_profiles(pts, k, d, h, perp, c):
    """Energy, count and median delay per annulus, largest stored in-plane radius, of plane k."""
    p = pts[np.rint(pts[:, 2]).astype(int) == k]
    r = np.hypot(*lateral.to_plane(p[:, :2] - (k + 1) * atmosphere.PLANE_SPACING * d[:2], h, perp, c).T)
    j = np.digitize(r, EDGES) - 1
    e_ann = np.bincount(j, p[:, 3], N_BIN)
    n_ann = np.bincount(j, minlength=N_BIN).astype(float)
    delay = np.full(N_BIN, np.nan)
    tk = p[:, 4] - (k + 1) * atmosphere.PLANE_SPACING / C_LIGHT
    for b in np.unique(j):
        delay[b] = np.median(tk[j == b])
    return e_ann, n_ann, delay, (r.max() if len(r) else 0.0)


def shower_task(task):
    pts, d = task
    pts, d = pts[pts[:, 3] > 0], d / np.linalg.norm(d)
    h, perp, c = lateral.frame(d)
    out = {"e_ann": np.zeros((N_PLANES, N_BIN)), "n_ann": np.zeros((N_PLANES, N_BIN)),
           "delay": np.full((N_PLANES, N_BIN), np.nan), "r_hi": np.zeros(N_PLANES), "r_ref": np.zeros(N_PLANES),
           "t_cap": np.full(N_PLANES, float(pts[:, 3].min()) if len(pts) else 0.0),
           **{f"{w}.{f}": np.full(N_PLANES, np.nan) for w in ("E", "N") for f in FIT}}
    for k in range(N_PLANES):
        if not np.any(np.rint(pts[:, 2]).astype(int) == k):
            continue
        e_ann, n_ann, delay, r_hi = plane_profiles(pts, k, d, h, perp, c)
        r_ref = float(lateral.r_m_ref(k, d[2]))
        out["e_ann"][k], out["n_ann"][k], out["delay"][k], out["r_hi"][k], out["r_ref"][k] = e_ann, n_ann, delay, \
            r_hi, r_ref
        if n_ann.sum() >= MIN_CELLS:
            for w, prof in (("E", e_ann), ("N", n_ann)):
                for f, v in zip(FIT, fit_profile(prof, max(r_hi, EDGES[1]), r_ref)):
                    out[f"{w}.{f}"][k] = v
    return out


def stage_hist(out, args, log):
    rows = slice_rows(args.out, args.limit)
    rows = rows[m2.part_rows(len(rows), args.n_parts, args.part)]
    path = m2.part_path(out, args.part, args.n_parts)
    if os.path.exists(path):
        log.info("part %d already on disk, skipped", args.part)
        return {"skipped": True}
    arrays = {}
    with Pool(args.workers) as pool:
        for s in SPECIES_NAMES:
            t0 = time.time()
            meta, pts = read_showers(s, rows)
            res = pool.map(shower_task, [(p, meta["directions"][i]) for i, p in enumerate(pts)], chunksize=4)
            arrays.update({f"{s}.{k}": np.stack([r[k] for r in res]) for k in res[0]})
            arrays.update({f"meta.{k}": v for k, v in meta.items()})
            log.info("%-8s %d showers, %.1f s", s, len(rows), time.time() - t0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    log.info("wrote %s", path)
    return {"rows": [int(rows[0]), int(rows[-1])], "n_rows": len(rows)}


def stage_merge(out, args, log):
    rows = slice_rows(args.out, args.limit)
    parts = [dict(np.load(m2.part_path(out, i, args.n_parts))) for i in range(args.n_parts)]
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if not np.array_equal(arrays["meta.rows"], rows):
        raise ValueError("parts do not cover the rows in order")
    np.savez(os.path.join(out, "annuli.npz"), **arrays)
    res = {}
    for s in SPECIES_NAMES:
        fitted = np.isfinite(arrays[f"{s}.E.s"])
        res[s] = {"planes fitted": int(fitted.sum()),
                  **{f"{w} {f} 16/50/84": np.nanpercentile(arrays[f"{s}.{w}.{f}"][fitted], [16, 50, 84]).round(3).tolist()
                     for w in ("E", "N") for f in ("ln_r50", "s")},
                  **{f"{w} on a bound": float(np.mean(
                      (np.abs(arrays[f"{s}.{w}.s"][fitted] - lateral.S_BOUNDS[0]) < 1e-3)
                      | (np.abs(arrays[f"{s}.{w}.s"][fitted] - lateral.S_BOUNDS[1]) < 1e-3)
                      | (np.abs(arrays[f"{s}.{w}.ln_r50"][fitted] - lateral.LN_R_BOUNDS[0]) < 1e-3)
                      | (np.abs(arrays[f"{s}.{w}.ln_r50"][fitted] - lateral.LN_R_BOUNDS[1]) < 1e-3)))
                     for w in ("E", "N")}}
        log.info("%s %s", s, res[s])
    return res


def features(age, log10_e, cos_z, ln_ek, r_ref):
    """Inputs of every in-plane law: age, age^2, log10 E, |cos z|, ln |cos z|, plane energy, Moliere radius."""
    c = np.maximum(np.abs(cos_z), 1e-3)
    return np.column_stack([np.ones_like(age), age, age ** 2, log10_e - 5.5, c, np.log(c), ln_ek - 10.0,
                            np.log(r_ref / 100.0)])


def law_table(out, args):
    """Per species, one row per usable plane: its ring profiles and law inputs."""
    z = dict(np.load(os.path.join(out, "annuli.npz")))
    m1f = np.load(os.path.join(args.out, m1.FOLDER, "fit", "fits_train.npz"))
    if not np.array_equal(m1f["meta.rows"], z["meta.rows"]):
        raise ValueError("annuli rows differ from M1's training rows")
    tab = {}
    for s in SPECIES_NAMES:
        X, xmax = m1f["X"], m1f[f"{s}.{m2.NAME}.xmax"][:, None]
        e_k = z[f"{s}.e_ann"].sum(2)
        ok = (np.isfinite(z[f"{s}.E.s"]) & np.isfinite(z[f"{s}.N.s"]) & (z[f"{s}.E.ok"] > 0) & (z[f"{s}.N.ok"] > 0)
              & np.isfinite(xmax) & (e_k > 0))
        i, k = np.nonzero(ok)
        age = (3 * X / (X + 2 * xmax))[i, k]
        y = np.column_stack([z[f"{s}.E.ln_r50"][i, k], z[f"{s}.E.s"][i, k], z[f"{s}.N.ln_r50"][i, k],
                             z[f"{s}.N.s"][i, k], np.log(z[f"{s}.n_ann"].sum(2)[i, k]),
                             np.log(np.maximum(z[f"{s}.r_hi"][i, k], EDGES[1]))])
        tab[s] = {"i": i, "k": k, "y": y, "label": m1f["meta.labels"][i],
                  "x": features(age, m1f["meta.log10_E"][i], m1f["meta.cos_zenith"][i], np.log(e_k[i, k]), z[f"{s}.r_ref"][i, k]),
                  "cos_z": np.abs(m1f["meta.cos_zenith"][i]), "age": age,
                  "e_ann": z[f"{s}.e_ann"][i, k], "n_ann": z[f"{s}.n_ann"][i, k], "r_hi": z[f"{s}.r_hi"][i, k],
                  "r_ref": z[f"{s}.r_ref"][i, k], "delay": z[f"{s}.delay"][i, k]}
    return tab


def profile_check(t, groups):
    """Per group, annulus by annulus: mean observed share over the planes against the mean fitted NKG share."""
    res = {}
    for name, m in groups.items():
        rows = np.nonzero(m)[0]
        if len(rows) < 50:
            continue
        obs = {"E": np.zeros(N_BIN), "N": np.zeros(N_BIN)}
        fit = {"E": np.zeros(N_BIN), "N": np.zeros(N_BIN)}
        for j in rows[:: max(1, len(rows) // 3000)]:
            n = int(np.searchsorted(EDGES, max(t["r_hi"][j], EDGES[1])))
            edges = np.minimum(EDGES, max(t["r_hi"][j], EDGES[1]))[:n + 1]
            for w, col, prof in (("E", 0, t["e_ann"][j]), ("N", 2, t["n_ann"][j])):
                obs[w][:n] += prof[:n] / prof[:n].sum()
                fit[w][:n] += nkg_shares(edges, t["y"][j, col], t["y"][j, col + 1], t["r_ref"][j])
        res[name] = {w: [round(float(a / b), 3) if b > 1e-3 * fit[w].sum() else None for a, b in zip(obs[w], fit[w])]
                     for w in ("E", "N")}
    return res


LAW_NAMES = ("E.ln_r50", "E.s", "N.ln_r50", "N.s", "ln_n_cells", "ln_r_hi")
COS_BANDS = (0.0, 0.1, 0.3, 1.01)


def stage_laws(out, args, log):
    """NKG fit quality and linear laws for the NKG numbers (held-out R2)."""
    tab = law_table(out, args)
    res, laws = {"annulus edges [m]": EDGES[:-1].tolist()}, {}
    for s, t in tab.items():
        res[s] = {"profile check, observed / fitted share per annulus": profile_check(
            t, {f"{m1.CLASS[c]} |cos z| {lo}-{hi}": (t["label"] == c) & (t["cos_z"] >= lo) & (t["cos_z"] < hi)
                for c in m1.CLASS for lo, hi in zip(COS_BANDS[:-1], COS_BANDS[1:])})}
        for c, cn in m1.CLASS.items():
            m = t["label"] == c
            test = m & (t["i"] % 5 == 0)
            train = m & ~test
            coef, *_ = np.linalg.lstsq(t["x"][train], t["y"][train], rcond=None)
            r_tr, r_te = t["y"][train] - t["x"][train] @ coef, t["y"][test] - t["x"][test] @ coef
            laws[f"{s}.{cn}.coef"], laws[f"{s}.{cn}.cov"] = coef, np.cov(r_tr.T)
            r2 = 1 - r_te.var(0) / t["y"][test].var(0)
            ages = np.digitize(t["age"][m], [0.6, 0.8, 1.0, 1.2, 1.4])
            res[s][cn] = {"planes train / test": [int(train.sum()), int(test.sum())],
                          "held-out R2": dict(zip(LAW_NAMES, r2.round(3).tolist())),
                          "residual sd": dict(zip(LAW_NAMES, r_te.std(0).round(3).tolist())),
                          "E.s median by age bin (<0.6, 0.6-0.8, ..., >1.4)":
                              [round(float(np.median(t["y"][m][ages == b, 1])), 3) if np.any(ages == b) else None
                               for b in range(6)]}
            log.info("%s %s R2 %s", s, cn, res[s][cn]["held-out R2"])
    np.savez(os.path.join(out, "laws.npz"), names=np.array(LAW_NAMES), **laws)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["hist", "merge", "laws"])
    ap.add_argument("--n-parts", type=int, required=True)
    ap.add_argument("--part", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)))
    ap.add_argument("--limit", type=int, default=0, help="LIMIT rows spread over M1's training slice")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    folder = FOLDER + (f"_n{args.limit}" if args.limit else "")
    name = args.stage if args.stage != "hist" else f"hist_{args.part:04d}"
    log_dir, log = logs.start(os.path.join(folder, "logs", name), args)
    stage = {"hist": stage_hist, "merge": stage_merge, "laws": stage_laws}[args.stage]
    logs.save(log_dir, stage(os.path.join(args.out, folder), args, log), args, log)


if __name__ == "__main__":
    main()
