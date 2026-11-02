"""Symbolic regression on the arrival times: log10(1 + t / ns) of a ring's earliest time (lead) and mean delay, relative to
light on the axis, one formula per species, class and time in place of the ring-by-ring lead and delay laws.

Stages:
  table  (tambo env) per group, one row per plane: inputs, the two times and the particle count per ring, the fold it is
         judged in (-1: training only), and the fold's ring-by-ring laws on it, the reference.
  fit    (symbolic env, PySR) per group, time and fold, a formula search on the fold's training showers, scored on its
         judged showers against the reference. Size chosen per group and time: the smallest within TOLERANCE of the best
         out-of-fold RMSE weighted by particle count.
"""
import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import time
from multiprocessing.pool import ThreadPool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import rings_sr

FOLDER = rings_sr.FOLDER
TABLE = "times_table.npz"
TIMES = ("lead", "delay")
N_RING = 23
EDGE_IN = np.r_[0.5, rings_sr.EDGES]                  # m, ring inner edges, the first floored
EDGE_OUT = np.r_[rings_sr.EDGES, 12600.0]             # m, ring outer edges, the open last ring closed
LN_R = 0.5 * (np.log(EDGE_IN) + np.log(EDGE_OUT))     # ln of each ring's centre
GATED = rings_sr.GATED
TOLERANCE = 0.05
N_ITER = 300
N_ROWS = 4000


def expand(v):
    """Rows (plane, ring) of the inputs: ln r first."""
    return np.column_stack([np.tile(LN_R, len(v)), np.repeat(v, N_RING, axis=0)])


def weighted_rmse(pred, y, n):
    """RMSE over rings with particles, weighted by particle count."""
    m = (n > 0) & np.isfinite(y)
    return float(np.sqrt(np.average((pred[m] - y[m]) ** 2, weights=n[m])))


def choose(rows):
    best = min(q["rmse"] for q in rows.values())
    return min(s for s, q in rows.items() if q["rmse"] <= best * (1 + TOLERANCE))


def merge_folds(found, fold, y, n):
    """Per size, each fold's largest formula not above that size, on its judged planes. found: {f: {size: (pred, text)}}."""
    judged = fold >= 0
    rows = {}
    for size in sorted(set(found[0]) | set(found[1])):
        pick = [max((s for s in found[f] if s <= size), default=None) for f in (0, 1)]
        if None in pick:
            continue
        pred = np.empty(y.shape)
        for f, s in zip((0, 1), pick):
            pred[fold == f] = found[f][s][0]
        rows[size] = {"rmse": weighted_rmse(pred[judged], y[judged], n[judged]),
                      "formulas": [found[f][s][1] for f, s in zip((0, 1), pick)]}
    return rows


def group_data(args, group, what):
    t = np.load(os.path.join(args.out, TABLE))
    key = group.replace(" ", ".")
    return {"v": t[f"{key}.v"], "n": t[f"{key}.n"], "fold": t[f"{key}.fold"], "y": t[f"{key}.{what}"], "ref": t[f"{key}.ref_{what}"]}


def found_path(run, group, what, f):
    return os.path.join(run, "found", f"{group.replace(' ', '_')}_{what}_fold{f}.pkl")


def stage_search(args):
    """One formula search, saved as {size: (prediction on the fold's judged planes, text)}; exits without Julia's teardown."""
    run, group, what, f = os.path.join(args.out, args.name), args.group, args.what, args.fold
    d = group_data(args, group, what)
    train, judged = d["fold"] != f, d["fold"] == f
    X, y, n = expand(d["v"][train]), d["y"][train].ravel(), d["n"][train].ravel()
    keep = np.flatnonzero((n > 0) & np.isfinite(y))
    keep = np.random.default_rng(f).choice(keep, min(N_ROWS, len(keep)), replace=False)
    model = rings_sr.regressor(run, f"{group.replace(' ', '_')}_{what}_fold{f}", f, args.iterations)
    model.fit(X[keep], y[keep], variable_names=rings_sr.VARIABLES)
    test = expand(d["v"][judged])
    found = {}
    for i, row in model.equations_.iterrows():
        p = np.asarray(model.predict(test, index=i), dtype=float)
        if np.isfinite(p).all():
            found[int(row["complexity"])] = (p.reshape(-1, N_RING).astype(np.float32), str(row["equation"]))
    path = found_path(run, group, what, f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as h:
        pickle.dump(found, h)
    os.replace(path + ".tmp", path)
    sys.stdout.flush()
    os._exit(0)


def run_search(job):
    """A search in its own process; the time limit means a stuck one fails the run instead of hanging it."""
    args, group, what, f = job
    t0 = time.time()
    subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "search", "--out", args.out, "--name", args.name, "--group", group,
                    "--what", what, "--fold", str(f), "--iterations", str(args.iterations)],
                   check=True, timeout=rings_sr.TIMEOUT + rings_sr.MARGIN)
    return group, what, f, time.time() - t0


def stage_table(args, log):
    from symbolic import draw, particles
    files = sorted(glob.glob(os.path.join(args.out, "tables", "*.npz")))
    fold_of = {}
    for f in (0, 1):
        with open(os.path.join(args.out, f"skip_fold{f}.txt")) as h:
            fold_of.update({n: f for n in h.read().split()})
    tabs = [dict(np.load(f)) for f in files]
    shower_fold = np.array([fold_of.get(os.path.basename(f), -1) for f in files])
    laws = {f: np.load(os.path.join(args.out, f"fold{f}", "particle_laws.npz")) for f in (0, 1)}
    out = {}
    for s, r in particles.plane_rows(tabs).items():
        for cn in rings_sr.CLASSES:
            m = r["cls"] == cn
            x, k, fold = r["x"][m], r["k"][m], shower_fold[r["i"][m]]
            key = f"{s}.{cn}"
            out.update({f"{key}.v": rings_sr.plane_inputs(x, k), f"{key}.n": r["ring_count"][m], f"{key}.fold": fold})
            for what in TIMES:
                ref = np.full((len(k), N_RING), np.nan)
                for f in (0, 1):
                    j = fold == f
                    ref[j] = x[j] @ draw.fill_rings(laws[f][f"{key}.{what}"]).T
                out.update({f"{key}.{what}": r[what][m], f"{key}.ref_{what}": ref})
            log.info("%-8s %-8s planes %4d judged %4d", s, cn, m.sum(), (fold >= 0).sum())
    np.savez(os.path.join(args.out, TABLE), **out)


def stage_fit(args, log):
    groups = args.groups.split(",") if args.groups else [f"{s} {c}" for s in rings_sr.SPECIES for c in rings_sr.CLASSES]
    jobs = [(args, g, w, f) for g in groups for w in TIMES for f in (0, 1)]
    with ThreadPool(args.workers) as pool:
        for g, w, f, sec in pool.imap_unordered(run_search, jobs):
            log.info("%-18s %-5s fold %d searched in %.0f s", g, w, f, sec)
    res, ratio = {}, {w: [] for w in TIMES}
    for g in groups:
        res[g] = {}
        for w in TIMES:
            d = group_data(args, g, w)
            found = {}
            for f in (0, 1):
                with open(found_path(args.run, g, w, f), "rb") as h:
                    found[f] = pickle.load(h)
            j = d["fold"] >= 0
            sizes = merge_folds(found, d["fold"], d["y"], d["n"])
            size = choose(sizes)
            ref = weighted_rmse(d["ref"][j], d["y"][j], d["n"][j])
            res[g][w] = {"reference rmse": ref, "chosen size": size, "sizes": {str(s): q for s, q in sizes.items()}}
            log.info("%-18s %-5s reference RMSE %.4f | size %d RMSE %.4f | %s", g, w, ref, size, sizes[size]["rmse"],
                     " || ".join(sizes[size]["formulas"]))
            if g in GATED:
                ratio[w].append(sizes[size]["rmse"] / ref)
    res["ratio to the reference"] = {w: float(np.mean(v)) if v else None for w, v in ratio.items()}
    log.info("ratio to the reference, mean over e/gamma: %s", json.dumps(res["ratio to the reference"]))
    return res


def main():
    from symbolic import logs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("table", "fit", "search"))
    ap.add_argument("--out", default=FOLDER)
    ap.add_argument("--name", default="sr_times")
    ap.add_argument("--groups", help="fit stage: comma-separated groups, default all")
    ap.add_argument("--group", help="search stage")
    ap.add_argument("--what", choices=TIMES, help="search stage")
    ap.add_argument("--fold", type=int, help="search stage")
    ap.add_argument("--iterations", type=int, default=N_ITER)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    if args.stage == "table":
        stage_table(args, logs.start("sr_times_table", args)[1])
    elif args.stage == "search":
        stage_search(args)
    else:
        args.run, log = logs.start(args.name, args)
        logs.save(args.run, stage_fit(args, log), args, log)


if __name__ == "__main__":
    main()
