"""Per shower, species and plane of repaired CORSIKA: energy, cell count and median arrival delay in rings of radius
about the shower axis, on the training slice. The generator's in-plane laws are fitted to these (table()).

Stages: hist (CPU array, one part of the slice per task), merge -> OUT/annuli/annuli.npz.
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

from analysis_core import logs
from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere, fit_profiles, lateral, sources, unwrap

FOLDER = "annuli"
EDGES = np.array([0.0, 10, 20, 30, 50, 70, 100, 140, 200, 280, 400, 560, 800, 1100, 1600, 2300, 3200, 4500, 6400,
                  9000, np.inf])     # m, in-plane radius about the axis
N_BIN = len(EDGES) - 1
C_LIGHT = 299792458.0
MIN_CELLS = 5                        # cells a plane needs to enter the laws


def slice_rows(root, limit):
    """The training slice of the profile fits; a smoke run takes `limit` of them evenly spread."""
    rows = np.load(os.path.join(root, fit_profiles.SLICE))["meta.rows"]
    return rows[np.linspace(0, len(rows) - 1, limit).astype(int)] if limit else rows


def read_showers(s, rows):
    """Metadata and repaired cells of species s at the training rows."""
    with h5py.File(sources.corsika_path(s, "train"), "r") as f:
        meta = {"rows": rows, "energies": f["energies"][rows].reshape(-1).astype(np.float64),
                "directions": f["directions"][rows].astype(np.float64), "labels": f["pdg"][rows].astype(np.int64)}
        pts = [np.asarray(f["showers"][r], np.float64).reshape(-1, 5) for r in rows]
    pts = [unwrap.unwrap_shower(p[p[:, 3] > 0], d, N_PLANES)[0] for p, d in zip(pts, meta["directions"])]
    return meta, pts


def plane_profiles(pts, k, d, h, perp, c):
    """Energy, cell count and median delay per ring of plane k."""
    p = pts[np.rint(pts[:, 2]).astype(int) == k]
    r = np.hypot(*lateral.to_plane(p[:, :2] - (k + 1) * atmosphere.PLANE_SPACING * d[:2], h, perp, c).T)
    j = np.digitize(r, EDGES) - 1
    e_ann = np.bincount(j, p[:, 3], N_BIN)
    n_ann = np.bincount(j, minlength=N_BIN).astype(float)
    delay = np.full(N_BIN, np.nan)
    tk = p[:, 4] - (k + 1) * atmosphere.PLANE_SPACING / C_LIGHT
    for b in np.unique(j):
        delay[b] = np.median(tk[j == b])
    return e_ann, n_ann, delay


def shower_task(task):
    pts, d = task
    pts, d = pts[pts[:, 3] > 0], d / np.linalg.norm(d)
    h, perp, c = lateral.frame(d)
    out = {"e_ann": np.zeros((N_PLANES, N_BIN)), "n_ann": np.zeros((N_PLANES, N_BIN)),
           "delay": np.full((N_PLANES, N_BIN), np.nan), "r_ref": np.zeros(N_PLANES)}
    for k in range(N_PLANES):
        if np.any(np.rint(pts[:, 2]).astype(int) == k):
            out["e_ann"][k], out["n_ann"][k], out["delay"][k] = plane_profiles(pts, k, d, h, perp, c)
            out["r_ref"][k] = float(lateral.r_m_ref(k, d[2]))
    return out


def stage_hist(args, log):
    rows = slice_rows(args.out, args.limit)
    rows = rows[fit_profiles.part_rows(len(rows), args.n_parts, args.part)]
    path = fit_profiles.part_path(os.path.join(args.out, FOLDER), args.part, args.n_parts)
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
    fit_profiles.save(path, arrays, log)
    return {"rows": [int(rows[0]), int(rows[-1])], "n_rows": len(rows)}


def stage_merge(args, log):
    folder = os.path.join(args.out, FOLDER)
    parts = [dict(np.load(fit_profiles.part_path(folder, i, args.n_parts))) for i in range(args.n_parts)]
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if not np.array_equal(arrays["meta.rows"], slice_rows(args.out, args.limit)):
        raise ValueError("parts do not cover the slice rows in order")
    fit_profiles.save(os.path.join(folder, "annuli.npz"), arrays, log)
    return {s: {"planes with >= MIN_CELLS cells": int((arrays[f"{s}.n_ann"].sum(2) >= MIN_CELLS).sum())}
            for s in SPECIES_NAMES}


def features(age, log10_e, cos_z, ln_ek, r_ref):
    """Inputs of every in-plane law: age, age^2, log10 E, |cos z|, ln |cos z|, plane energy, Moliere radius."""
    c = np.maximum(np.abs(cos_z), 1e-3)
    return np.column_stack([np.ones_like(age), age, age ** 2, log10_e - 5.5, c, np.log(c), ln_ek - 10.0,
                            np.log(r_ref / 100.0)])


def table(root):
    """Per species, one row per plane with >= MIN_CELLS cells and a fitted profile: its rings and conditions."""
    z = np.load(os.path.join(root, FOLDER, "annuli.npz"))
    fs = np.load(os.path.join(root, fit_profiles.SLICE))
    if not np.array_equal(fs["meta.rows"], z["meta.rows"]):
        raise ValueError("annuli rows differ from the profile-fit slice")
    tab = {}
    for s in SPECIES_NAMES:
        ok = ((z[f"{s}.n_ann"].sum(2) >= MIN_CELLS) & (z[f"{s}.e_ann"].sum(2) > 0)
              & np.isfinite(fs[f"{s}.xmax"])[:, None])
        i, k = np.nonzero(ok)
        tab[s] = {"i": i, "k": k, "label": fs["meta.labels"][i], "cos_z": np.abs(fs["meta.cos_zenith"][i]),
                  **{w: z[f"{s}.{w}"][i, k] for w in ("e_ann", "n_ann", "r_ref", "delay")}}
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["hist", "merge"])
    ap.add_argument("--n-parts", type=int, required=True)
    ap.add_argument("--part", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)))
    ap.add_argument("--limit", type=int, default=0, help="smoke run: LIMIT showers spread over the slice")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    name = args.stage if args.stage != "hist" else f"hist_{args.part:04d}"
    log_dir, log = logs.start(os.path.join(FOLDER, "logs", name), args)
    stage = {"hist": stage_hist, "merge": stage_merge}[args.stage]
    logs.save(log_dir, stage(args, log), args, log)


if __name__ == "__main__":
    main()
