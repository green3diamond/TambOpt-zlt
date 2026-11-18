"""Fit the model from CORSIKA, one stage per command; each law stage writes <law>.npz and <law>.json into --model.

  tables    shower tables of the simulation folders listed in --list (task i of N): plane_stats summaries, or with
            --raw the raw particles (adds the particle-energy histograms the spread is fitted on)
  profiles  profile curves (longitudinal.py) fitted to the total energy per plane of the listed tables, log10 E <= --top
  part1     the sampler of each class (sampler.py) from a profiles file
  rows      plane rows (rows.py) of the listed tables
  rings, width, times, energy, spread   the in-plane laws from a rows file (width needs rings.npz in --model)
"""
import argparse
import os
import sys
from multiprocessing import Pool, get_context

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere, data, energy, longitudinal, model, rings, rows, sampler, spread, times, width

CHUNK = 64


def listed(args):
    """The table files named in --list (one name per line) under --tables, sorted."""
    return [os.path.join(args.tables, n) for n in sorted(set(open(args.list).read().split()))]


def groups():
    return [(s, cn) for s in SPECIES_NAMES for cn in data.CLASS.values()]


def stage_tables(args):
    sims = open(args.list).read().split()
    data.write_tables(sims[args.task::args.tasks], args.out, args.raw)


def read_table(path):
    z = np.load(path)
    d = np.asarray(z["direction"], np.float64)
    return {s: np.asarray(z[f"{s}.E_total"], np.float64) for s in SPECIES_NAMES}, float(z["log10E"]), d / np.linalg.norm(d), \
        data.LABEL[str(z["class"])]


def stage_profiles(args):
    if os.path.exists(args.out):
        raise ValueError(f"exists, refusing: {args.out}")
    files = listed(args)
    with Pool(args.workers) as pool:
        t = pool.map(read_table, files, chunksize=64)
        names = [os.path.basename(f) for f, r in zip(files, t) if r[1] <= args.top]
        t = [r for r in t if r[1] <= args.top]
        dirs = np.array([r[2] for r in t])
        d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        zenith = np.degrees(np.arccos(np.clip(d[:, 2], -1, 1)))
        X = atmosphere.plane_depths(zenith, N_PLANES)
        out = {"meta.names": np.array(names), "meta.log10_E": np.array([r[1] for r in t]), "meta.cos_zenith": dirs[:, 2],
               "meta.labels": np.array([r[3] for r in t], np.int64), "X": X}
        for s in SPECIES_NAMES:
            E = np.array([r[0][s] for r in t])
            Y = longitudinal.log_energy(E)
            parts = pool.map(longitudinal.fit_block, [(X[i:i + CHUNK], Y[i:i + CHUNK]) for i in range(0, len(Y), CHUNK)])
            out.update({f"{s}.E": E, **{f"{s}.{k}": np.concatenate([p[k] for p in parts]) for k in ("ln_n", "theta")}})
            print(f"{s}: {len(Y)} showers, fitted {np.isfinite(out[f'{s}.ln_n']).mean():.4f}", flush=True)
    np.savez(args.out, **out)


def stage_part1(args):
    d = sampler.load(args.fits)
    arrays = {}
    for c, cn in data.CLASS.items():
        arrays.update(sampler.to_arrays(sampler.fit(d, c), cn))
        print(f"{cn}: fitted on {int(arrays[f'{cn}.n_train'])} clean showers", flush=True)
    model.save(args.model, "part1", arrays, {"fits": args.fits, "showers": np.load(args.fits)["meta.names"].tolist()})


def stage_rows(args):
    if os.path.exists(args.out):
        raise ValueError(f"exists, refusing: {args.out}")
    np.savez(args.out, **rows.build(listed(args), args.workers))


def by_class(r, cn):
    m = r["cls"] == cn
    return {w: v[m] for w, v in r.items()}


_JOB = None


def _group(g):
    data_, one = _JOB
    s, cn = g
    res = one(by_class(data_[s], cn), s, cn)
    print(f"{s} {cn} done", flush=True)
    return {f"{s}.{cn}{k}": v for k, v in res.items()} if isinstance(res, dict) else {f"{s}.{cn}": res}


def fit_laws(args, name, one):
    """one(class rows, species, class) -> array or {suffix: array} for every species and class, the groups in --procs
    forked processes, into <model>/<name>.npz."""
    global _JOB
    data_, showers = rows.load(args.rows)
    _JOB = (data_, one)
    with get_context("fork").Pool(min(args.procs, len(groups()))) as pool:
        parts = pool.map(_group, groups(), chunksize=1)
    _JOB = None
    model.save(args.model, name, {k: v for p in parts for k, v in p.items()}, {"rows": args.rows, "showers": showers})


def stage_rings(args):
    fit_laws(args, "rings", lambda r, s, cn: rings.fit(r["x"], r["ring_energy"] / r["E_total"][:, None]))


def stage_width(args):
    B = np.load(os.path.join(args.model, "rings.npz"))
    fit_laws(args, "width", lambda r, s, cn: width.fit(r, B[f"{s}.{cn}"]))


def stage_times(args):
    fit_laws(args, "times", lambda r, s, cn: {f".{w}": times.fit(r, w) for w in times.TIMES})


def stage_spread(args):
    fit_laws(args, "spread", lambda r, s, cn: spread.fit(r, s))


def energy_group(job):
    """One species and class, in its own process with `threads` torch threads."""
    path, s, cn, threads, seeds, epochs = job
    import torch
    torch.set_num_threads(threads)
    z = np.load(path)
    r = by_class({w: z[f"{s}.{w}"] for w in ("cls", "x", "k", "i", "ring_energy", "ring_count")}, cn)
    err, theta = energy.fit(r, seeds, epochs)
    print(f"energy {s} {cn}: planes {len(r['k'])}, validation error {err:.4f}", flush=True)
    return f"{s}.{cn}", theta


def stage_energy(args):
    procs = min(args.procs, len(groups()))
    jobs = [(args.rows, s, cn, max(1, args.workers // procs), args.seeds, args.epochs) for s, cn in groups()]
    with get_context("spawn").Pool(procs) as pool:
        arrays = dict(pool.map(energy_group, jobs, chunksize=1))
    model.save(args.model, "energy", arrays, {"rows": args.rows, "showers": np.load(args.rows)["showers"].tolist(),
                                              "seeds": args.seeds, "epochs": args.epochs})


def main():
    stages = {"tables": stage_tables, "profiles": stage_profiles, "part1": stage_part1, "rows": stage_rows,
              "rings": stage_rings, "width": stage_width, "times": stage_times, "energy": stage_energy, "spread": stage_spread}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=stages)
    ap.add_argument("--model", help="model folder (law stages)")
    ap.add_argument("--tables", help="tables folder (profiles, rows)")
    ap.add_argument("--list", help="simulation folders (tables) or table names (profiles, rows), one per line")
    ap.add_argument("--raw", action="store_true", help="tables: from the raw particles")
    ap.add_argument("--top", type=float, default=7.0, help="profiles: keep log10 E <= top")
    ap.add_argument("--fits", help="part1: profiles file")
    ap.add_argument("--rows", help="law stages: rows file")
    ap.add_argument("--out", help="tables: folder; profiles, rows: file")
    ap.add_argument("--seeds", type=int, default=energy.SEEDS)
    ap.add_argument("--epochs", type=int, default=energy.EPOCHS)
    ap.add_argument("--procs", type=int, default=6, help="law stages: groups fitted in parallel processes")
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    stages[args.stage](args)


if __name__ == "__main__":
    main()
