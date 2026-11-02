"""Symbolic regression on the width of a plane's lateral profile: log10 of the stretch of the rings law that best matches
the plane on the BAND ring edges, one formula per species and class.

Stages:
  table  (tambo env) one row per plane of the laws' training rows: inputs, target, shower, plane, fold, steep.
  fit    (symbolic env, PySR) per group, a formula search on each fold of showers, each scored on the other fold, against
         linear ridge laws on the same folds. Size chosen per group: the smallest within TOLERANCE of the best
         out-of-fold RMSE. Score: mean over the GATED groups of the RMS of the steep zone gaps (ZONES), a gap being the
         mean over a zone's planes of (mean prediction - median target).
  common (tambo env) one form for all groups, a tanh along the shower plus linear terms (FORMS), constants per group,
         scored across the same folds.
  law    (tambo env) the chosen form on every shower with C inside the planes, its spread along the shower and memory.
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
FORMS = {"tanh(ln k)": (1, []), "tanh(ln X)": (2, []), "tanh(ln k) + l, age": (1, [3, 0]),
         "tanh(ln k) + all": (1, [0, 3, 4, 5, 6])}     # column of VARIABLES inside the tanh, columns added linearly
TANH_B = (0.5, 1.0, 2.0, 4.0)
TANH_C = (0.2, 0.4, 0.6, 0.8)                         # quantiles of the input inside the tanh
COMMON_TOLERANCE = 0.02
COMMON_SCORE = 0.075
LAW = "width_law.npz"
N_PLANE = 24
SPREAD_RATIO = 1.5                                    # max / min spread across ZONES above which it varies with the plane


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
    t = np.load(os.path.join(args.out, args.table))
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


def tanh_basis(p, L, B, C):
    return np.column_stack([np.tanh(B * (p - C)), np.ones(len(p)), L])


def fit_tanh(p, L, y, c_range=(-np.inf, np.inf)):
    """A tanh(B (p - C)) + D + L e by least squares: A, D, e solved exactly for each (B, C), which are searched on a grid
    and then refined, C kept inside c_range -> [B, C, A, D, e...]."""
    from scipy.optimize import minimize

    def basis(theta):
        return tanh_basis(p, L, theta[0], np.clip(theta[1], *c_range))

    def loss(theta):
        b = basis(theta)
        return ((y - b @ np.linalg.lstsq(b, y, rcond=None)[0]) ** 2).sum()

    start = min(((B, C) for B in TANH_B for C in np.quantile(p, TANH_C)), key=loss)
    best = minimize(loss, start, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 2000}).x
    return np.r_[best[0], np.clip(best[1], *c_range), np.linalg.lstsq(basis(best), y, rcond=None)[0]]


def fit_spread(p, e):
    """ln sd = a + b p of residuals e by maximum likelihood -> (a, b)."""
    from scipy.optimize import minimize

    def nll(t):
        s = t[0] + t[1] * p
        return (s + 0.5 * e ** 2 * np.exp(-2 * s)).sum()

    return minimize(nll, [np.log(e.std()), 0.0], method="BFGS").x


def lag_one(v, unit, k):
    """Correlation of v on adjacent planes of one shower."""
    by = dict(zip(zip(unit.tolist(), k.tolist()), v))
    return float(np.corrcoef(np.array([(a, by[(u, p + 1)]) for (u, p), a in by.items() if (u, p + 1) in by]).T)[0, 1])


def law_mean(w, k, l, age):
    """Mean log10 width of planes k (0-based), w = [B, C, A, D, E, F, a, b, rho]."""
    return tanh_predict(w[:6], np.log(k + 1.0), np.column_stack([l, age]))


def law_sd(w, k):
    return np.exp(w[6] + w[7] * np.log(k + 1.0))


def stage_law(args, log):
    groups = [f"{s} {c}" for s in SPECIES for c in CLASSES]
    data = {g: group_data(args, g) for g in groups}
    theta, resid, ratio, zones = {}, {}, {}, {}
    for g, d in data.items():
        theta[g] = fit_tanh(d["v"][:, 1], d["v"][:, [3, 0]], d["y"], (0.0, np.log(N_PLANE)))
        resid[g] = d["y"] - tanh_predict(theta[g], d["v"][:, 1], d["v"][:, [3, 0]])
        zones[g] = [float(resid[g][(d["k"] >= lo) & (d["k"] < hi)].std()) for lo, hi in ZONES.values()]
        ratio[g] = max(zones[g]) / min(zones[g])
    varying = any(ratio[g] > SPREAD_RATIO for g in GATED)
    log.info("spread by zone %s | varying along the shower: %s", json.dumps({g: np.round(z, 3).tolist() for g, z in zones.items()}), varying)
    laws, res = {}, {"varying": bool(varying), "zones": zones, "groups": {}}
    for g, d in data.items():
        p, e = d["v"][:, 1], resid[g]
        a, b = fit_spread(p, e) if varying else (np.log(e.std()), 0.0)
        rho = lag_one(e * np.exp(-(a + b * p)), d["i"], d["k"])
        laws[g.replace(" ", ".")] = np.r_[theta[g], a, b, rho]
        res["groups"][g] = dict(zip(("B", "C", "A", "D", "E", "F", "a", "b", "rho"), laws[g.replace(" ", ".")].tolist()))
        log.info("%-18s %s | sd planes 1 / 12 / 24: %s", g, " ".join(f"{n} {v:.3f}" for n, v in res["groups"][g].items()),
                 np.round(np.exp(a + b * np.log([1, 12, 24])), 3).tolist())
    np.savez(os.path.join(args.out, args.name, LAW), **laws)
    return res


def tanh_predict(theta, p, L):
    return tanh_basis(p, L, theta[0], theta[1]) @ theta[2:]


def choose_form(rows, sizes):
    """The fewest numbers within COMMON_TOLERANCE of the best mean RMSE. rows: {form: mean RMSE}, sizes: {form: numbers}."""
    best = min(rows.values())
    return min((f for f, v in rows.items() if v <= best * (1 + COMMON_TOLERANCE)), key=lambda f: sizes[f])


def stage_common(args, log):
    groups = [f"{s} {c}" for s in SPECIES for c in CLASSES]
    data = {g: group_data(args, g) for g in groups}
    res, mean_rmse, gated_rmse, score = {}, {}, {}, {}
    for name, (pos, linear) in FORMS.items():
        rows = {}
        for g in groups:
            d = data[g]
            p, L, y, fold = d["v"][:, pos], d["v"][:, linear], d["y"], d["fold"]
            pred = np.empty(len(y))
            for f in (0, 1):
                a, b = fold == f, fold != f
                pred[b] = tanh_predict(fit_tanh(p[a], L[a], y[a]), p[b], L[b])
            theta = fit_tanh(p, L, y)
            rows[g] = {**summary(pred, y, d["k"], d["steep"]), "constants": theta.tolist()}
            log.info("%-18s %-22s RMSE %.3f | gaps %+.2f %+.2f %+.2f | RMS %.3f | B %.2f C %.2f A %.2f D %.2f", g, name,
                     rows[g]["rmse"], *rows[g]["gaps"], rows[g]["rms gap"], *theta[:4])
        res[name] = rows
        mean_rmse[name] = float(np.mean([q["rmse"] for q in rows.values()]))
        gated_rmse[name] = float(np.mean([rows[g]["rmse"] for g in GATED]))
        score[name] = float(np.mean([rows[g]["rms gap"] for g in GATED]))
    linear = {g: summary(out_of_fold(np.column_stack([data[g]["x"], data[g]["v"][:, 1]]), data[g]["y"], data[g]["fold"]),
                         data[g]["y"], data[g]["k"], data[g]["steep"]) for g in groups}
    reference = {"rmse e/gamma": float(np.mean([linear[g]["rmse"] for g in GATED])),
                 "score": float(np.mean([linear[g]["rms gap"] for g in GATED]))}
    chosen = choose_form(mean_rmse, {f: 4 + len(FORMS[f][1]) for f in FORMS})
    stands = score[chosen] <= COMMON_SCORE and gated_rmse[chosen] <= reference["rmse e/gamma"]
    out = {"forms": res, "mean rmse": mean_rmse, "rmse e/gamma": gated_rmse, "score": score, "linear comparison": reference,
           "chosen": chosen, "stands": bool(stands)}
    for f in FORMS:
        log.info("%-22s mean RMSE %.4f | e/gamma RMSE %.4f | score %.3f", f, mean_rmse[f], gated_rmse[f], score[f])
    log.info("linear comparison %s | chosen %s | stands %s", json.dumps(reference), chosen, stands)
    with open(os.path.join(args.out, args.name, "common.json"), "w") as h:
        json.dump(out, h, indent=1)
    return out


def stage_table(args, log):
    from scipy.special import softmax
    from symbolic import atmosphere, particles
    skip = set(open(args.skip).read().split()) if args.skip else set()
    files = [f for f in sorted(glob.glob(os.path.join(args.out, "tables", "*.npz"))) if os.path.basename(f) not in skip]
    fit = os.path.join(args.out, os.path.dirname(args.laws), "results.json")
    with open(fit) as h:
        if sorted(json.load(h)["showers"]) != [os.path.basename(f) for f in files]:
            raise ValueError(f"the showers differ from those of the rings law in {fit}")
    tabs = [dict(np.load(f)) for f in files]
    log.info("%d shower tables", len(tabs))
    log10e = np.array([float(t["log10E"]) for t in tabs])
    cz = np.array([float(t["direction"][2]) for t in tabs])
    X = atmosphere.plane_depths(np.degrees(np.arccos(cz)), particles.N_PLANE)
    laws, out = np.load(os.path.join(args.out, args.laws)), {}
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
    np.savez(os.path.join(args.out, args.table), **out)


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
    ap.add_argument("stage", choices=("table", "fit", "search", "common", "law"))
    ap.add_argument("--out", default=FOLDER)
    ap.add_argument("--name", default="sr_width")
    ap.add_argument("--groups", help="fit stage: comma-separated groups, default all")
    ap.add_argument("--table", default=TABLE, help="table file under --out")
    ap.add_argument("--laws", default=LAWS, help="table stage: rings law, relative to --out")
    ap.add_argument("--skip", help="table stage: file of table names to leave out")
    ap.add_argument("--group", help="search stage")
    ap.add_argument("--fold", type=int, help="search stage")
    ap.add_argument("--iterations", type=int, default=N_ITER)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    if args.stage == "table":
        stage_table(args, logs.start("sr_width_table", args)[1])
    elif args.stage == "search":
        stage_search(args)
    elif args.stage == "common":
        stage_common(args, logs.start(args.name + "_common", args)[1])
    elif args.stage == "law":
        log = logs.start(args.name + "_law", args)[1]
        with open(os.path.join(args.out, args.name, "law.json"), "w") as h:
            json.dump(stage_law(args, log), h, indent=1)
    else:
        args.run, log = logs.start(args.name, args)
        logs.save(args.run, stage_fit(args, log), args, log)


if __name__ == "__main__":
    main()
