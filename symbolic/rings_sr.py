"""Symbolic regression on a plane's cumulative energy profile: logit of the share of its energy inside radius r, one formula
per species and class in place of the rings and width laws.

Stages:
  table  (tambo env) per group, one row per plane of the judge's shower tables: inputs, the true shares and logit F at the
         ring edges, the fold it is judged in (-1: training only), and the adopted model's mean profile from that fold's
         laws, the reference.
  fit    (symbolic env, PySR) per group and fold, a formula search on the fold's training showers, scored on its judged
         showers against the reference. Size chosen per group: the smallest within TOLERANCE of the best out-of-fold RMSE.
  common (tambo env) one form for all groups, linear in terms of ln r and the inputs (FORMS), fractional logit fit to F at
         the edges, scored across the same folds.
  law    (tambo env) the chosen form on the judged showers and its plane-to-plane stretch: [beta..., a, b, rho].
Inputs to the formulas: VARIABLES; age, log10 E - 5.5, |cos z|, ln plane energy - 10 and ln Moliere radius are
annuli.features.
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

FOLDER = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic_trim/particles/judge"
TABLE = "rings_table.npz"
VARIABLES = ["ln_r", "age", "ln_k", "l", "c", "ln_Ek", "ln_rm"]
FEATURE_COLUMNS = (1, 3, 4, 6, 7)                     # annuli.features columns: age, l, c, ln_Ek, ln_rm
EDGES = np.array([1, 2, 5, 10, 20, 30, 50, 70, 100, 140, 200, 280, 400, 560, 800, 1100, 1600, 2300, 3200, 4500, 6400,
                  9000], float)                       # the interior ring edges of particles.EDGES
BAND = (EDGES >= 10.0) & (EDGES <= 400.0)
LIMIT = np.log((1 - 1e-6) / 1e-6) - 1e-9              # |logit F| at particles.CUM_CLIP: F was exactly 0 or 1
SPECIES = ("electron", "muon", "photon")
CLASSES = ("EM", "hadronic")
GATED = ("electron EM", "electron hadronic", "photon EM", "photon hadronic")
TOLERANCE = 0.05
MAX_SIZE = 30
N_ITER = 300
N_ROWS = 4000
BATCH = 500
TIMEOUT = 3000                                        # s, one search
MARGIN = 600                                          # s beyond TIMEOUT before a search process counts as stuck
LAW = "profile_law.npz"
N_PLANE = 24
FORMS = ("a", "b", "c", "d", "e")
FORM_NUMBERS = {"a": 6, "b": 7, "c": 8, "d": 8, "e": 8}


def to_shares(y):
    """Ring shares of logit F at the edges, made non-decreasing in r first."""
    f = 1 / (1 + np.exp(-np.maximum.accumulate(y, axis=1)))
    return np.diff(np.column_stack([np.zeros(len(f)), f, np.ones(len(f))]), axis=1)


def measures(y, true_y, ok, shares):
    """Median KL(true || predicted) per plane, RMSE of logit F on all measurable edges and on BAND, non-monotonic planes."""
    p = np.maximum(to_shares(y), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(shares > 0, shares * np.log(shares / p), 0.0).sum(1)
    e = (y - true_y) ** 2
    return {"kl": float(np.median(kl)), "rmse": float(np.sqrt(e[ok].mean())),
            "rmse band": float(np.sqrt(e[ok & BAND[None, :]].mean())),
            "not monotonic": float((np.diff(y, axis=1) < 0).any(1).mean())}


def plane_inputs(x, k):
    """VARIABLES after ln r, from annuli.features rows of planes k (0-based)."""
    return np.column_stack([x[:, 1], np.log(k + 1.0), x[:, 3], x[:, 4], x[:, 6], x[:, 7]])


def design(form, X):
    """Terms of form on rows X = [ln r, age, ln k, l, c, ln Ek, ln rm]."""
    lr, age, lk, l, _, le, _ = X.T
    terms = [np.ones(len(X)), lr, lr ** 2, lk, l, age]
    if form != "a":
        terms.append(le)
    terms += {"a": [], "b": [], "c": [lr * lk], "d": [np.tanh(lr)], "e": [lr ** 3]}[form]
    return np.column_stack(terms)


def fit_fractional(D, f, n_iter=50, ridge=1e-6):
    """Coefficients of a logit link fitted to shares f by Newton steps on the binomial likelihood."""
    b = np.zeros(D.shape[1])
    for _ in range(n_iter):
        eta = np.clip(D @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = np.maximum(p * (1 - p), 1e-10)
        step = np.linalg.solve((D * w[:, None]).T @ D + ridge * np.eye(D.shape[1]), D.T @ (f - p) - ridge * b)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b


def form_profile(form, beta, v):
    """logit F at the edges, planes x edges."""
    return (design(form, expand(v)) @ beta).reshape(-1, len(EDGES))


def expand(v, y=None):
    """Rows (plane, edge) of the inputs: ln r first; and of the target, if given."""
    X = np.column_stack([np.tile(np.log(EDGES), len(v)), np.repeat(v, len(EDGES), axis=0)])
    return X if y is None else (X, y.ravel())


def choose(rows):
    """Smallest size within TOLERANCE of the best RMSE; rows: {size: measures}."""
    best = min(q["rmse"] for q in rows.values())
    return min(s for s, q in rows.items() if q["rmse"] <= best * (1 + TOLERANCE))


def merge_folds(found, fold, y, ok, shares):
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
        rows[size] = {**measures(pred[judged], y[judged], ok[judged], shares[judged]),
                      "formulas": [found[f][s][1] for f, s in zip((0, 1), pick)]}
    return rows


def group_data(args, group):
    t = np.load(os.path.join(args.out, TABLE))
    return {w: t[f"{group.replace(' ', '.')}.{w}"] for w in ("v", "y", "shares", "ref", "fold", "k", "i")}


def found_path(run, group, f):
    return os.path.join(run, "found", f"{group.replace(' ', '_')}_fold{f}.pkl")


def regressor(run, run_id, seed, iterations, **extra):
    from pysr import PySRRegressor
    return PySRRegressor(niterations=iterations, binary_operators=["+", "-", "*", "/"],
                         unary_operators=["exp", "log", "square", "sqrt", "tanh"], maxsize=MAX_SIZE,
                         nested_constraints={"exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0},
                                             "tanh": {"tanh": 0}},
                         batching=True, batch_size=BATCH, output_directory=os.path.join(run, "pysr"), run_id=run_id,
                         random_state=seed, deterministic=True, parallelism="serial", progress=False, verbosity=0,
                         timeout_in_seconds=TIMEOUT, model_selection="best", **extra)


def stage_search(args):
    """One formula search for fold f, saved as {size: (prediction on its judged planes, text)}. Julia can hang when asked to
    shut down, so the process exits without its teardown once the result is on disk."""
    run, group, f = os.path.join(args.out, args.name), args.group, args.fold
    d = group_data(args, group)
    train, judged = d["fold"] != f, d["fold"] == f
    X, y = expand(d["v"][train], d["y"][train])
    keep = np.flatnonzero(np.abs(y) < LIMIT)
    keep = np.random.default_rng(f).choice(keep, min(N_ROWS, len(keep)), replace=False)
    model = regressor(run, f"{group.replace(' ', '_')}_fold{f}", f, args.iterations)
    model.fit(X[keep], y[keep], variable_names=VARIABLES)
    test = expand(d["v"][judged])
    found = {}
    for i, row in model.equations_.iterrows():
        p = np.asarray(model.predict(test, index=i), dtype=float)
        if np.isfinite(p).all():
            found[int(row["complexity"])] = (p.reshape(-1, len(EDGES)).astype(np.float32), str(row["equation"]))
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
    from symbolic import particles, width_sr
    files = sorted(glob.glob(os.path.join(args.out, "tables", "*.npz")))
    names = [os.path.basename(f) for f in files]
    fold_of = {}
    for f in (0, 1):
        with open(os.path.join(args.out, f"skip_fold{f}.txt")) as h:
            fold_of.update({n: f for n in h.read().split()})
    tabs = [dict(np.load(f)) for f in files]
    shower_fold = np.array([fold_of.get(n, -1) for n in names])
    log.info("%d shower tables, judged per fold %s", len(tabs), np.bincount(shower_fold[shower_fold >= 0]).tolist())
    laws = {f: np.load(os.path.join(args.out, f"fold{f}", "particle_laws.npz")) for f in (0, 1)}
    width = {f: np.load(os.path.join(args.out, f"fold{f}", width_sr.LAW)) for f in (0, 1)}
    out = {}
    for s, r in particles.plane_rows(tabs).items():
        for cn in CLASSES:
            m = r["cls"] == cn
            x, k = r["x"][m], r["k"][m]
            shares = r["ring_energy"][m] / r["E_total"][m, None]
            fold = shower_fold[r["i"][m]]
            v = plane_inputs(x, k)
            ref = np.full((len(k), len(EDGES)), np.nan)
            for f in (0, 1):
                j = np.flatnonzero(fold == f)
                base = softmax(x[j] @ laws[f][f"{s}.{cn}.rings"], axis=1)
                ll = width_sr.law_mean(width[f][f"{s}.{cn}"], k[j], x[j, 3], x[j, 1])
                ref[j] = particles.cum_logodds(np.array([particles.stretch(q, w) for q, w in zip(base, ll)]))
            if not np.isfinite(v).all():
                raise ValueError(f"{s} {cn}: inputs not finite")
            out.update({f"{s}.{cn}.{w}": q for w, q in (("v", v), ("y", particles.cum_logodds(shares)), ("shares", shares),
                                                       ("ref", ref), ("fold", fold), ("k", k), ("i", r["i"][m]))})
            log.info("%-8s %-8s planes %4d judged %4d", s, cn, m.sum(), (fold >= 0).sum())
    np.savez(os.path.join(args.out, TABLE), **out)


def stage_common(args, log):
    groups = [f"{s} {c}" for s in SPECIES for c in CLASSES]
    data = {g: group_data(args, g) for g in groups}
    res, kl = {}, {}
    for form in FORMS:
        res[form], ratios = {}, []
        for g, d in data.items():
            y, ok, shares, fold = d["y"], np.abs(d["y"]) < LIMIT, d["shares"], d["fold"]
            pred = np.empty(y.shape)
            for f in (0, 1):
                X, l = expand(d["v"][fold != f], y[fold != f])
                beta = fit_fractional(design(form, X), 1 / (1 + np.exp(-l)))
                pred[fold == f] = form_profile(form, beta, d["v"][fold == f])
            j = fold >= 0
            q = measures(pred[j], y[j], ok[j], shares[j])
            ref = measures(d["ref"][j], y[j], ok[j], shares[j])
            res[form][g] = {**q, "reference kl": ref["kl"], "reference rmse band": ref["rmse band"]}
            log.info("%-2s %-18s KL %.4f (reference %.4f) RMSE %.3f band %.3f (reference %.3f) not monotonic %.3f", form, g,
                     q["kl"], ref["kl"], q["rmse"], q["rmse band"], ref["rmse band"], q["not monotonic"])
            if g in GATED:
                ratios.append(q["kl"] / ref["kl"])
        kl[form] = float(np.mean(ratios))
    best = min(kl.values())
    chosen = min((f for f in FORMS if kl[f] <= best * (1 + TOLERANCE)), key=lambda f: (FORM_NUMBERS[f], f))
    log.info("KL ratio to the reference, mean over e/gamma %s | chosen %s", json.dumps(kl), chosen)
    with open(os.path.join(args.out, args.name, "common.json"), "w") as h:
        json.dump({"forms": res, "kl ratio": kl, "chosen": chosen}, h, indent=1)


def stage_law(args, log):
    """The chosen form on the judged showers, then each plane's stretch against it on the BAND edges."""
    from symbolic import particles, width_sr
    with open(os.path.join(args.out, args.name, "common.json")) as h:
        form = json.load(h)["chosen"]
    laws = {"form": form}
    for g in (f"{s} {c}" for s in SPECIES for c in CLASSES):
        d = group_data(args, g)
        j = d["fold"] >= 0
        X, l = expand(d["v"][j], d["y"][j])
        beta = fit_fractional(design(form, X), 1 / (1 + np.exp(-l)))
        base = to_shares(form_profile(form, beta, d["v"][j]))
        lam = particles.plane_shapes(d["shares"][j], base, tilt=False, edges=particles.BAND)[0]
        lk = np.log(d["k"][j] + 1.0)
        e = lam - lam.mean()
        a, b = width_sr.fit_spread(lk, e)
        rho = width_sr.lag_one(e * np.exp(-(a + b * lk)), d["i"][j], d["k"][j])
        laws[g.replace(" ", ".")] = np.r_[beta, a, b, rho]
        log.info("%-18s form %s beta %s | mean stretch %+.3f sd planes 1 / 12 / 24 %s rho %.3f", g, form, np.round(beta, 3).tolist(),
                 lam.mean(), np.round(np.exp(a + b * np.log([1, 12, 24])), 3).tolist(), rho)
    np.savez(os.path.join(args.out, args.name, LAW), **laws)


def stage_fit(args, log):
    groups = args.groups.split(",") if args.groups else [f"{s} {c}" for s in SPECIES for c in CLASSES]
    with ThreadPool(args.workers) as pool:
        for g, f, sec in pool.imap_unordered(run_search, [(args, g, f) for g in groups for f in (0, 1)]):
            log.info("%-18s fold %d searched in %.0f s", g, f, sec)
    res, ratio = {}, {"kl": [], "rmse band": []}
    for g in groups:
        d = group_data(args, g)
        found = {}
        for f in (0, 1):
            with open(found_path(args.run, g, f), "rb") as h:
                found[f] = pickle.load(h)
        ok = np.abs(d["y"]) < LIMIT
        judged = d["fold"] >= 0
        sizes = merge_folds(found, d["fold"], d["y"], ok, d["shares"])
        size = choose(sizes)
        ref = measures(d["ref"][judged], d["y"][judged], ok[judged], d["shares"][judged])
        res[g] = {"reference": ref, "chosen size": size, "sizes": {str(s): q for s, q in sizes.items()}}
        c = sizes[size]
        log.info("%-18s reference KL %.4f RMSE %.3f band %.3f | size %d KL %.4f RMSE %.3f band %.3f not monotonic %.3f",
                 g, ref["kl"], ref["rmse"], ref["rmse band"], size, c["kl"], c["rmse"], c["rmse band"], c["not monotonic"])
        log.info("%-18s %s", g, " || ".join(c["formulas"]))
        if g in GATED:
            for w in ratio:
                ratio[w].append(c[w] / ref[w])
    res["ratio to the reference"] = {w: float(np.mean(v)) if v else None for w, v in ratio.items()}
    log.info("ratio to the reference, mean over e/gamma: %s", json.dumps(res["ratio to the reference"]))
    return res


def main():
    from symbolic import logs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("table", "fit", "search", "common", "law"))
    ap.add_argument("--out", default=FOLDER)
    ap.add_argument("--name", default="sr_rings")
    ap.add_argument("--groups", help="fit stage: comma-separated groups, default all")
    ap.add_argument("--group", help="search stage")
    ap.add_argument("--fold", type=int, help="search stage")
    ap.add_argument("--iterations", type=int, default=N_ITER)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    if args.stage == "table":
        stage_table(args, logs.start("sr_rings_table", args)[1])
    elif args.stage == "search":
        stage_search(args)
    elif args.stage in ("common", "law"):
        os.makedirs(os.path.join(args.out, args.name), exist_ok=True)
        {"common": stage_common, "law": stage_law}[args.stage](args, logs.start(f"{args.name}_{args.stage}", args)[1])
    else:
        args.run, log = logs.start(args.name, args)
        logs.save(args.run, stage_fit(args, log), args, log)


if __name__ == "__main__":
    main()
