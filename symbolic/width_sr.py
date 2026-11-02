"""Symbolic regression on the width of a plane's lateral profile: log10 of the stretch of the rings law that best matches
the plane on the BAND ring edges, one formula per species and class.

Stages:
  table  (tambo env) one row per plane of the laws' training rows: inputs, target, shower, plane, fold, steep.
  fit    (symbolic env, PySR) per group, a formula search on each fold of showers, each scored on the other fold, against
         linear ridge laws on the same folds. Size chosen per group: the smallest within TOLERANCE of the best
         out-of-fold RMSE. Score: mean over the GATED groups of the RMS of the steep zone gaps (ZONES), a gap being the
         mean over a zone's planes of (mean prediction - median target).
Inputs to the formulas: VARIABLES; age, log10 E - 5.5, |cos z|, ln plane energy - 10 and ln Moliere radius are annuli.features.
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

FOLDER = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic_trim/particles"
LAWS = os.path.join("fit_v2", "particle_laws.npz")
TABLE = "width_table.npz"
VARIABLES = ["age", "ln_k", "ln_X", "l", "c", "ln_Ek", "ln_r"]
FEATURE_COLUMNS = (1, 3, 4, 6, 7)                     # annuli.features columns: age, l, c, ln_Ek, ln_r
SPECIES = ("electron", "muon", "photon")
CLASSES = ("EM", "hadronic")
GATED = ("electron EM", "electron hadronic", "photon EM", "photon hadronic")
ZONES = {"planes 1-8": (0, 8), "planes 11-18": (10, 18), "planes 21-24": (20, 24)}
STEEP = 0.1
RIDGE = 1.0
TOLERANCE = 0.01
MAX_SIZE = 20
N_ITER = 300
TIMEOUT = 3000                                        # s, one search
MARGIN = 600                                          # s beyond TIMEOUT before a search process counts as stuck


def zone_gaps(pred, y, k):
    """Mean over each zone's planes of (mean prediction - median target)."""
    return [float(np.mean([pred[k == p].mean() - np.median(y[k == p]) for p in range(*z) if (k == p).any()]))
            for z in ZONES.values()]


def summary(pred, y, k, steep):
    g = zone_gaps(pred[steep], y[steep], k[steep])
    return {"rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "gaps": g, "rms gap": float(np.sqrt(np.mean(np.square(g))))}


def ridge(x, y):
    pen = RIDGE * np.eye(x.shape[1])
    pen[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + pen, x.T @ y)


def out_of_fold(x, y, fold):
    """Linear ridge fitted on each fold, predicted on the other."""
    pred = np.empty(len(y))
    for f in (0, 1):
        pred[fold != f] = x[fold != f] @ ridge(x[fold == f], y[fold == f])
    return pred


def choose(rows):
    """Smallest size within TOLERANCE of the best RMSE; rows: {size: summary}."""
    best = min(q["rmse"] for q in rows.values())
    return min(s for s, q in rows.items() if q["rmse"] <= best * (1 + TOLERANCE))


def merge_folds(found, fold, y, k, steep):
    """Per size, each fold's largest formula not above that size, on the other fold. found: {f: {size: (pred, text)}}."""
    rows = {}
    for size in sorted(set(found[0]) | set(found[1])):
        pick = [max((s for s in found[f] if s <= size), default=None) for f in (0, 1)]
        if None in pick:
            continue
        pred = np.empty(len(y))
        for f, s in zip((0, 1), pick):
            pred[fold != f] = found[f][s][0]
        rows[size] = {**summary(pred, y, k, steep), "formulas": [found[f][s][1] for f, s in zip((0, 1), pick)]}
    return rows


def group_data(args, group):
    t = np.load(os.path.join(args.out, TABLE))
    return {w: t[f"{group.replace(' ', '.')}.{w}"] for w in ("x", "v", "y", "i", "k", "fold", "steep")}


def found_path(run, group, f):
    return os.path.join(run, "found", f"{group.replace(' ', '_')}_fold{f}.pkl")


def stage_search(args):
    """One formula search on one fold, saved as {size: (prediction on the other fold, text)}. Julia can hang when asked
    to shut down, so the process exits without its teardown once the result is on disk."""
    run, group, f = os.path.join(args.out, args.name), args.group, args.fold
    d = group_data(args, group)
    X, y, fold = d["v"], d["y"], d["fold"]
    from pysr import PySRRegressor
    model = PySRRegressor(niterations=args.iterations, binary_operators=["+", "-", "*", "/"],
                          unary_operators=["exp", "log", "square", "sqrt", "tanh"], maxsize=MAX_SIZE,
                          nested_constraints={"exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0},
                                              "tanh": {"tanh": 0}},
                          output_directory=os.path.join(run, "pysr"), run_id=f"{group.replace(' ', '_')}_fold{f}",
                          random_state=f,
                          deterministic=True, parallelism="serial", progress=False, verbosity=0,
                          timeout_in_seconds=TIMEOUT, model_selection="best")
    model.fit(X[fold == f], y[fold == f], variable_names=VARIABLES)
    found = {}
    for i, row in model.equations_.iterrows():
        p = np.asarray(model.predict(X[fold != f], index=i), dtype=float)
        if np.isfinite(p).all():
            found[int(row["complexity"])] = (p, str(row["equation"]))
    path = found_path(run, group, f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as h:
        pickle.dump(found, h)
    os.replace(path + ".tmp", path)
    sys.stdout.flush()
    os._exit(0)


def run_search(job):
    """A search in its own process; the time limit means a stuck one fails the run instead of hanging it."""
    args, group, f = job
    t0 = time.time()
    subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "search", "--out", args.out, "--name", args.name,
                    "--group", group, "--fold", str(f), "--iterations", str(args.iterations)],
                   check=True, timeout=TIMEOUT + MARGIN)
    return group, f, time.time() - t0


def stage_table(args, log):
    from scipy.special import softmax
    from symbolic import atmosphere, particles
    tabs = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(args.out, "tables", "*.npz")))]
    log.info("%d shower tables", len(tabs))
    log10e = np.array([float(t["log10E"]) for t in tabs])
    cz = np.array([float(t["direction"][2]) for t in tabs])
    X = atmosphere.plane_depths(np.degrees(np.arccos(cz)), particles.N_PLANE)
    laws, out = np.load(os.path.join(args.out, LAWS)), {}
    for s, r in particles.plane_rows(tabs).items():
        test = particles.split_showers(r["i"], r["cls"], log10e)
        for cn in CLASSES:
            m = r["cls"] == cn
            x, i, k = r["x"][m], r["i"][m], r["k"][m]
            true = r["ring_energy"][m] / r["E_total"][m, None]
            base = softmax(x @ laws[f"{s}.{cn}.rings"], axis=1)
            y = particles.plane_shapes(true, base, tilt=False, edges=particles.BAND)[0]
            v = np.column_stack([x[:, 1], np.log(k + 1.0), np.log(X[i, k] / 100.0), x[:, 3], x[:, 4], x[:, 6], x[:, 7]])
            if not (np.isfinite(v).all() and np.isfinite(y).all()):
                raise ValueError(f"{s} {cn}: inputs or target not finite")
            out.update({f"{s}.{cn}.{w}": q for w, q in (("x", x), ("v", v), ("y", y), ("i", i), ("k", k),
                                                       ("fold", test[m].astype(int)),
                                                       ("steep", np.abs(cz[i]) >= STEEP))})
            log.info("%-8s %-8s planes %4d showers %2d", s, cn, m.sum(), len(np.unique(i)))
    np.savez(os.path.join(args.out, TABLE), **out)


def stage_fit(args, log):
    groups = [f"{s} {c}" for s in SPECIES for c in CLASSES]
    if args.groups:
        groups = args.groups.split(",")
    data = {g: group_data(args, g) for g in groups}
    with ThreadPool(args.workers) as pool:
        for g, f, sec in pool.imap_unordered(run_search, [(args, g, f) for g in groups for f in (0, 1)]):
            log.info("%-18s fold %d searched in %.0f s", g, f, sec)
    found = {g: {} for g in groups}
    for g in groups:
        for f in (0, 1):
            with open(found_path(args.run, g, f), "rb") as h:
                found[g][f] = pickle.load(h)
    res, score = {}, {"present law form": [], "present + ln k": [], "present + ln k, ln X": [], "chosen formula": []}
    for g in groups:
        d = data[g]
        y, k, fold, st = d["y"], d["k"], d["fold"], d["steep"]
        lin = {"present law form": d["x"], "present + ln k": np.column_stack([d["x"], d["v"][:, 1]]),
               "present + ln k, ln X": np.column_stack([d["x"], d["v"][:, 1], d["v"][:, 2]])}
        rows = {n: summary(out_of_fold(x, y, fold), y, k, st) for n, x in lin.items()}
        sizes = merge_folds(found[g], fold, y, k, st)
        size = choose(sizes)
        rows["chosen formula"] = {**sizes[size], "size": size}
        res[g] = {"rows": rows, "sizes": {str(s): q for s, q in sizes.items()}}
        for n, q in rows.items():
            log.info("%-18s %-22s RMSE %.3f | gaps %+.2f %+.2f %+.2f | RMS %.3f", g, n, q["rmse"], *q["gaps"], q["rms gap"])
            if g in GATED:
                score[n].append(q["rms gap"])
        log.info("%-18s size %d: %s", g, size, " || ".join(rows["chosen formula"]["formulas"]))
    res["score"] = {n: float(np.mean(v)) if v else None for n, v in score.items()}
    log.info("score %s", json.dumps(res["score"]))
    return res


def main():
    from symbolic import logs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("table", "fit", "search"))
    ap.add_argument("--out", default=FOLDER)
    ap.add_argument("--name", default="sr_width")
    ap.add_argument("--groups", help="fit stage: comma-separated groups, default all")
    ap.add_argument("--group", help="search stage")
    ap.add_argument("--fold", type=int, help="search stage")
    ap.add_argument("--iterations", type=int, default=N_ITER)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    if args.stage == "table":
        stage_table(args, logs.start("sr_width_table", args)[1])
    elif args.stage == "search":
        stage_search(args)
    else:
        args.run, log = logs.start(args.name, args)
        logs.save(args.run, stage_fit(args, log), args, log)


if __name__ == "__main__":
    main()
