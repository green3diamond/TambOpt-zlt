"""Symbolic regression on the arrival times: log10(1 + t / ns) of a ring's earliest time (lead) and mean delay, relative to
light on the axis, one formula per species, class and time in place of the ring-by-ring lead and delay laws.

Stages:
  table  (tambo env) per group, one row per plane: inputs, the two times and the particle count per ring, the fold it is
         judged in (-1: training only), and the fold's ring-by-ring laws on it, the reference.
  fit    (symbolic env, PySR) per group, time and fold, a formula search on the fold's training showers, scored on its
         judged showers against the reference. Size chosen per group and time: the smallest within TOLERANCE of the best
         out-of-fold RMSE weighted by particle count.
  common_search  (symbolic env, PySR) per time and fold, a search for one shape shared by the groups, constants per group.
  common (tambo env) the common forms against each other and the reference, the search's shape among them.
  law    (tambo env) a common form's constants per group, on the showers of a rings law or on all of them.
"""
import argparse
import glob
import json
import os
import pickle
import re
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
COMMON_ITER = 1000
COMMON_TOLERANCE = 0.02
MEAN_BAR = 1.25                                       # common form / reference, mean over GATED, each time
GROUP_BAR = 1.5                                       # the same, each group
GEOMETRIC = np.log(1 / (2 * 500.0 * 0.299792458))     # ln ns / m^2, t = r^2 / (2 k x 500 m x c)
SHARED = {"A": 6, "B": 6, "C": 6}                     # numbers per group of the searched shape, 6 groups
TEMPLATE = ("A[g] * f(ln_r - C[g], age, ln_k, l, c, ln_Ek, ln_rm) + "
            "B[g] * h(ln_r - C[g], age, ln_k, l, c, ln_Ek, ln_rm)")
LAW = "time_law.npz"


def form_a(th, X):
    return np.log10(1 + np.exp(np.minimum(th[0] + 2 * X[:, 0] - X[:, 2], 50.0)))


def form_b(th, X):
    return np.log10(1 + np.exp(np.minimum(th[0] + th[1] * X[:, 0] - th[2] * X[:, 2], 50.0)))


def form_c(th, X):
    return (th[0] * np.tanh(th[1] * X[:, 0]) ** 2 + th[2]) * (X[:, 0] - th[3] * X[:, 2])


def form_d(th, X):
    return (th[0] * np.tanh(th[1] * X[:, 0]) ** 2 + th[2]) * (X[:, 0] - th[3] * X[:, 2] + th[4] * X[:, 3] + th[5] * X[:, 1])


# form: (y of constants and expanded rows, starting constants); X columns are rings_sr.VARIABLES
GATE = [(0.5, b, 0.0, q) for b in (0.2, 0.5) for q in (0.5, 1.0)]
FORMS = {"a": (form_a, [(GEOMETRIC,)]),
         "b": (form_b, [(GEOMETRIC, p, q) for p in (1.0, 2.0) for q in (0.5, 1.0)]),
         "c": (form_c, GATE),
         "d": (form_d, [g + (0.0, 0.0) for g in GATE])}


def fit_form(form, X, y):
    """Least-squares constants of a form, unweighted as the search fitted, the best of its starts."""
    from scipy.optimize import least_squares
    f, starts = FORMS[form]
    fits = [least_squares(lambda th: f(th, X) - y, s, max_nfev=200) for s in starts]
    return min(fits, key=lambda r: r.cost).x


def usable(d):
    """Expanded inputs and target of a group's rows with particles, and the mask that made them."""
    m = (d["n"] > 0) & np.isfinite(d["y"])
    return expand(d["v"])[m.ravel()], d["y"][m], m


def choose_common(rows, numbers):
    """The fewest numbers within COMMON_TOLERANCE of the best. rows: {form: mean RMSE}, numbers: {form: numbers per group}."""
    best = min(rows.values())
    return min((f for f, v in rows.items() if v <= best * (1 + COMMON_TOLERANCE)), key=lambda f: numbers[f])


def stands(ratios):
    """ratios: {time: [form / reference per GATED group]}: both bars met, for every time."""
    return all(np.mean(v) <= MEAN_BAR and max(v) <= GROUP_BAR for v in ratios.values())


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


def all_groups():
    return [f"{s} {c}" for s in rings_sr.SPECIES for c in rings_sr.CLASSES]


def common_path(run, what, f):
    return os.path.join(run, "found", f"{what}_fold{f}.pkl")


def stage_common_search(args):
    """One search for the shared shape, saved as {size: ({group: prediction on its judged planes}, text)}."""
    from pysr import TemplateExpressionSpec
    run, what, f = os.path.join(args.out, args.name), args.what, args.fold
    groups = all_groups()
    data = {g: group_data(args, g, what) for g in groups}
    rng = np.random.default_rng(f)
    Xs, ys = [], []
    for a, g in enumerate(groups):
        d = data[g]
        train = d["fold"] != f
        X, y, n = expand(d["v"][train]), d["y"][train].ravel(), d["n"][train].ravel()
        keep = np.flatnonzero((n > 0) & np.isfinite(y))
        keep = rng.choice(keep, min(args.rows, len(keep)), replace=False)
        Xs.append(np.column_stack([X[keep], np.full(len(keep), a + 1.0)]))
        ys.append(y[keep])
    names = rings_sr.VARIABLES + ["g"]
    spec = TemplateExpressionSpec(combine=TEMPLATE, expressions=["f", "h"], variable_names=names, parameters=SHARED)
    model = rings_sr.regressor(run, f"{what}_fold{f}", f, args.iterations, expression_spec=spec)
    model.fit(np.concatenate(Xs), np.concatenate(ys), variable_names=names)
    test = {g: expand(data[g]["v"][data[g]["fold"] == f]) for g in groups}
    found = {}
    for i, row in model.equations_.iterrows():
        preds = {g: np.asarray(model.predict(np.column_stack([test[g], np.full(len(test[g]), a + 1.0)]), index=i), dtype=float)
                 for a, g in enumerate(groups)}
        if all(np.isfinite(p).all() for p in preds.values()):
            found[int(row["complexity"])] = ({g: p.reshape(-1, N_RING).astype(np.float32) for g, p in preds.items()}, str(row["equation"]))
    path = common_path(run, what, f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as h:
        pickle.dump(found, h)
    os.replace(path + ".tmp", path)
    sys.stdout.flush()
    os._exit(0)


def run_common_search(job):
    args, what, f = job
    t0 = time.time()
    subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "common_one", "--out", args.out, "--name", args.name, "--what", what,
                    "--fold", str(f), "--iterations", str(args.iterations), "--rows", str(args.rows)],
                   check=True, timeout=rings_sr.TIMEOUT + rings_sr.MARGIN)
    return what, f, time.time() - t0


def stage_common_searches(args, log):
    with ThreadPool(args.workers) as pool:
        for what, f, sec in pool.imap_unordered(run_common_search, [(args, w, f) for w in TIMES for f in (0, 1)]):
            log.info("%-5s fold %d searched in %.0f s", what, f, sec)
    return {"searches": len(TIMES) * 2}


def score_rows(pred, d):
    """Scores on the judged planes: RMSE weighted by count, overall and by radius, median |error|, worst 1% of planes' share."""
    j = d["fold"] >= 0
    y, n, p = d["y"][j], d["n"][j], pred[j]
    m = (n > 0) & np.isfinite(y)
    e2 = np.where(m, (p - y) ** 2 * n, 0.0)
    plane = np.sort(e2.sum(1))[::-1]
    bands = {"under 10 m": EDGE_OUT <= 10.0, "10-400 m": (EDGE_IN >= 10.0) & (EDGE_OUT <= 400.0), "beyond 400 m": EDGE_IN >= 400.0}
    return {"rmse": weighted_rmse(p, y, n), "median": float(np.median(np.abs(p - y)[m])),
            "top share": float(plane[: max(1, len(plane) // 100)].sum() / plane.sum()),
            **{b: weighted_rmse(p[:, r], y[:, r], n[:, r]) for b, r in bands.items()}}


def shared_constants(text):
    """Constants of a searched shape's f and h, from its printed form 'f = ...; h = ...; A = [...]; ...'; #n are its inputs."""
    return len(re.findall(r"(?<![#\w.])-?\d+\.?\d*(?:[eE][-+]?\d+)?", text.split("; A =")[0]))


def searched_form(args, what, data):
    """The search's chosen size: {group: prediction}, numbers per group, the formulas. Each fold's largest formula not above the size."""
    run = os.path.join(args.out, args.name)
    found = {}
    for f in (0, 1):
        with open(common_path(run, what, f), "rb") as h:
            found[f] = pickle.load(h)
    groups = all_groups()
    rows = {}
    for size in sorted(set(found[0]) | set(found[1])):
        pick = [max((s for s in found[f] if s <= size), default=None) for f in (0, 1)]
        if None in pick:
            continue
        pred = {g: np.full(data[g]["y"].shape, np.nan) for g in groups}
        for f, s in zip((0, 1), pick):
            for g in groups:
                pred[g][data[g]["fold"] == f] = found[f][s][0][g]
        j = {g: data[g]["fold"] >= 0 for g in groups}
        rows[size] = {"rmse": float(np.mean([weighted_rmse(pred[g][j[g]], data[g]["y"][j[g]], data[g]["n"][j[g]]) for g in groups])),
                      "pred": pred, "formulas": [found[f][s][1] for f, s in zip((0, 1), pick)]}
    best = min(q["rmse"] for q in rows.values())
    size = min(s for s, q in rows.items() if q["rmse"] <= best * (1 + TOLERANCE))
    shared = max(shared_constants(t) for t in rows[size]["formulas"])
    return rows[size]["pred"], len(SHARED) + shared / len(groups), rows[size]["formulas"], {str(s): q["rmse"] for s, q in rows.items()}


def stage_common(args, log):
    groups = all_groups()
    numbers = {"a": 1, "b": 3, "c": 4, "d": 6}
    scores, res = {w: {} for w in TIMES}, {"formulas": {}, "search sizes": {}, "geometric": {}, "reference": {}, "constants": {}}
    for w in TIMES:
        data = {g: group_data(args, g, w) for g in groups}
        preds = {}
        for name in FORMS:
            preds[name] = {g: np.full(data[g]["y"].shape, np.nan) for g in groups}
            res["constants"].setdefault(name, {})[w] = {}
            for g in groups:
                d = data[g]
                for f in (0, 1):
                    train, judged = d["fold"] != f, d["fold"] == f
                    X, y, _ = usable({"n": d["n"][train], "y": d["y"][train], "v": d["v"][train]})
                    theta = fit_form(name, X, y)
                    preds[name][g][judged] = FORMS[name][0](theta, expand(d["v"][judged])).reshape(-1, N_RING)
                    res["constants"][name][w].setdefault(g, []).append(theta.tolist())
        if args.searched:
            preds["e"], n_e, res["formulas"][w], res["search sizes"][w] = searched_form(args, w, data)
            numbers[f"e {w}"] = n_e
        for name, per in preds.items():
            scores[w][name] = {g: score_rows(per[g], data[g]) for g in groups}
        res["reference"][w] = {g: score_rows(data[g]["ref"], data[g]) for g in groups}
        res["geometric"][w] = {g: score_rows(form_a((GEOMETRIC,), expand(data[g]["v"])).reshape(-1, N_RING), data[g]) for g in groups}
    forms = list(scores[TIMES[0]])
    if "e" in forms:
        numbers["e"] = max(numbers.pop(f"e {w}") for w in TIMES)
    mean = {f: float(np.mean([scores[w][f][g]["rmse"] for w in TIMES for g in groups])) for f in forms}
    ratios = {f: {w: [scores[w][f][g]["rmse"] / res["reference"][w][g]["rmse"] for g in GATED] for w in TIMES} for f in forms}
    chosen = choose_common(mean, numbers)
    out = {**res, "scores": scores, "mean rmse": mean, "numbers": numbers, "ratios": ratios,
           "chosen": chosen, "stands": bool(stands(ratios[chosen]))}
    for w in TIMES:
        for g in groups:
            r = res["reference"][w][g]
            log.info("%-18s %-5s ref %.4f geometric %.4f | %s", g, w, r["rmse"], res["geometric"][w][g]["rmse"],
                     " ".join(f"{f} {scores[w][f][g]['rmse']:.4f}" for f in forms))
    for f in forms:
        log.info("%s numbers %.1f mean RMSE %.4f | lead %s mean %.2f | delay %s mean %.2f | stands %s", f, numbers[f], mean[f],
                 np.round(ratios[f]["lead"], 2).tolist(), np.mean(ratios[f]["lead"]),
                 np.round(ratios[f]["delay"], 2).tolist(), np.mean(ratios[f]["delay"]), stands(ratios[f]))
    log.info("chosen %s stands %s", chosen, out["stands"])
    return out


def stage_law(args, log):
    """Constants of --form per group and time, on the showers of --skip's complement, for draw.py --time-law."""
    skip = set()
    if args.skip:
        with open(args.skip) as h:
            skip = set(h.read().split())
    from symbolic import particles
    files = [f for f in sorted(glob.glob(os.path.join(args.out, "tables", "*.npz"))) if os.path.basename(f) not in skip]
    tabs = [dict(np.load(f)) for f in files]
    laws = {"form": np.array(args.form)}
    for s, r in particles.plane_rows(tabs).items():
        for cn in rings_sr.CLASSES:
            m = r["cls"] == cn
            for w in TIMES:
                X, y, _ = usable({"n": r["ring_count"][m], "y": r[w][m], "v": rings_sr.plane_inputs(r["x"][m], r["k"][m])})
                laws[f"{s}.{cn}.{w}"] = fit_form(args.form, X, y)
                log.info("%-8s %-8s %-5s %s", s, cn, w, np.round(laws[f"{s}.{cn}.{w}"], 3).tolist())
    os.makedirs(os.path.join(args.out, args.name), exist_ok=True)
    np.savez(os.path.join(args.out, args.name, LAW), **laws)
    return {"form": args.form, "showers": [os.path.basename(f) for f in files]}


def ring_times(theta, form, x, k):
    """log10(1 + t / ns) per ring of one plane from a common form's constants; x: its annuli.features row, k: its 0-based number."""
    v = rings_sr.plane_inputs(x[None], np.array([k]))
    return FORMS[form][0](theta, expand(v))


def main():
    from symbolic import logs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("table", "fit", "search", "common_search", "common_one", "common", "law"))
    ap.add_argument("--out", default=FOLDER)
    ap.add_argument("--name", default="sr_times")
    ap.add_argument("--groups", help="fit stage: comma-separated groups, default all")
    ap.add_argument("--group", help="search stage")
    ap.add_argument("--what", choices=TIMES, help="search stage")
    ap.add_argument("--fold", type=int, help="search stage")
    ap.add_argument("--iterations", type=int, default=N_ITER)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--rows", type=int, default=N_ROWS, help="common_search stage: rows per group")
    ap.add_argument("--searched", action="store_true", help="common stage: include the shared shape's search")
    ap.add_argument("--form", choices=tuple(FORMS), help="law stage")
    ap.add_argument("--skip", help="law stage: file listing the showers left out")
    args = ap.parse_args()
    if args.stage == "table":
        stage_table(args, logs.start("sr_times_table", args)[1])
    elif args.stage == "search":
        stage_search(args)
    elif args.stage == "common_one":
        stage_common_search(args)
    elif args.stage in ("common_search", "common", "law"):
        run, log = logs.start(args.name, args)
        logs.save(run, {"common_search": stage_common_searches, "common": stage_common, "law": stage_law}[args.stage](args, log), args, log)
    else:
        args.run, log = logs.start(args.name, args)
        logs.save(args.run, stage_fit(args, log), args, log)


if __name__ == "__main__":
    main()
