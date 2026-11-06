"""Symbolic regression on the mean particle energy per ring: ln mean energy, one formula per species and class in place
of the ring-by-ring law.

Stages:
  table  (tambo env) per group, one row per plane: inputs, target and particle count per ring, the fold it is judged in
         (-1: training only), and the fold's ring-by-ring law on it, the reference.
  fit    (symbolic env, PySR) per group and fold, a formula search on the fold's training showers, scored on its judged
         showers against the reference.
  common (tambo env) the common forms against each other and the reference.
  law    (tambo env) a common form's constants per group, on the showers left by --skip, for draw.py --energy-law.
"""
import argparse
import glob
import os
import pickle
import subprocess
import sys
import time
from multiprocessing.pool import ThreadPool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import rings_sr, times_sr

FOLDER = rings_sr.FOLDER
TABLE = "energy_table.npz"
GATED = rings_sr.GATED
N_ITER = 300
N_ROWS = 4000
LAW = "energy_law.npz"


def design(form, X):
    """Columns of a linear form; X columns are rings_sr.VARIABLES: ln r, age, ln k, l, c, ln plane energy, ln Moliere radius."""
    r, age, k, l, c, ek, rm = X.T
    cols = [np.ones(len(X)), r, k]
    if form in ("b", "c", "d"):
        cols.append(r ** 2)
    if form in ("c", "d"):
        cols += [ek, age]
    if form == "d":
        cols += [l, c, rm]
    return np.column_stack(cols)


FORMS = {"a": 3, "b": 4, "c": 6, "d": 9}                   # form: numbers per group


def group_data(args, group):
    t = np.load(os.path.join(args.out, TABLE))
    key = group.replace(" ", ".")
    return {w: t[f"{key}.{w}"] for w in ("v", "n", "fold", "y", "ref", "shower")}


def fit_form(form, v, y, n):
    """Least-squares constants on the rows with particles, unweighted as the search fits."""
    m = (n > 0) & np.isfinite(y)
    return np.linalg.lstsq(design(form, times_sr.expand(v)[m.ravel()]), y[m], rcond=None)[0]


def tanh_sum(theta, X):
    """c + sum_u a_u tanh(w_u . x + b_u) on raw inputs; theta = [w (U rows of len(x)), b (U), a (U), c] (tiny_net.py)."""
    u = (len(theta) - 1) // (X.shape[1] + 2)
    w, b, a, c = theta[:u * X.shape[1]].reshape(u, -1), theta[-1 - 2 * u:-1 - u], theta[-1 - u:-1], theta[-1]
    return np.tanh(X @ w.T + b) @ a + c


def predict(theta, form, v):
    X = times_sr.expand(v)
    y = tanh_sum(theta, X) if form == "tanh" else design(form, X) @ theta
    return y.reshape(-1, times_sr.N_RING)


def ring_means(theta, form, x, k):
    """ln mean energy per ring of one plane; x: its annuli.features row, k: its 0-based number."""
    return predict(theta, form, rings_sr.plane_inputs(x[None], np.array([k])))[0]


def found_path(run, group, f):
    return os.path.join(run, "found", f"{group.replace(' ', '_')}_fold{f}.pkl")


def stage_search(args):
    """One formula search, saved as {size: (prediction on the fold's judged planes, text)}; exits without Julia's teardown."""
    run, group, f = os.path.join(args.out, args.name), args.group, args.fold
    d = group_data(args, group)
    train, judged = d["fold"] != f, d["fold"] == f
    X, y, n = times_sr.expand(d["v"][train]), d["y"][train].ravel(), d["n"][train].ravel()
    keep = np.flatnonzero((n > 0) & np.isfinite(y))
    keep = np.random.default_rng(f).choice(keep, min(N_ROWS, len(keep)), replace=False)
    model = rings_sr.regressor(run, f"{group.replace(' ', '_')}_fold{f}", f, args.iterations)
    model.fit(X[keep], y[keep], variable_names=rings_sr.VARIABLES)
    test = times_sr.expand(d["v"][judged])
    found = {}
    for i, row in model.equations_.iterrows():
        p = np.asarray(model.predict(test, index=i), dtype=float)
        if np.isfinite(p).all():
            found[int(row["complexity"])] = (p.reshape(-1, times_sr.N_RING).astype(np.float32), str(row["equation"]))
    path = found_path(run, group, f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as h:
        pickle.dump(found, h)
    os.replace(path + ".tmp", path)
    sys.stdout.flush()
    os._exit(0)


def run_search(job):
    args, group, f = job
    t0 = time.time()
    subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "search", "--out", args.out, "--name", args.name, "--group", group,
                    "--fold", str(f), "--iterations", str(args.iterations)], check=True, timeout=rings_sr.TIMEOUT + rings_sr.MARGIN)
    return group, f, time.time() - t0


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
        y = particles.targets(r)["mean_energy"][1]
        for cn in rings_sr.CLASSES:
            m = r["cls"] == cn
            x, k, fold = r["x"][m], r["k"][m], shower_fold[r["i"][m]]
            key = f"{s}.{cn}"
            ref = np.full((len(k), times_sr.N_RING), np.nan)
            for f in (0, 1):
                j = fold == f
                ref[j] = x[j] @ draw.fill_rings(laws[f][f"{key}.mean_energy"]).T
            out.update({f"{key}.v": rings_sr.plane_inputs(x, k), f"{key}.n": r["ring_count"][m], f"{key}.fold": fold,
                        f"{key}.y": y[m], f"{key}.ref": ref, f"{key}.shower": r["i"][m]})
            log.info("%-8s %-8s planes %4d judged %4d", s, cn, m.sum(), (fold >= 0).sum())
    np.savez(os.path.join(args.out, TABLE), **out)
    return {"groups": len(out) // 6}


def stage_fit(args, log):
    groups = times_sr.all_groups()
    with ThreadPool(args.workers) as pool:
        for g, f, sec in pool.imap_unordered(run_search, [(args, g, f) for g in groups for f in (0, 1)]):
            log.info("%-18s fold %d searched in %.0f s", g, f, sec)
    res, ratio = {}, []
    for g in groups:
        d = group_data(args, g)
        found = {}
        for f in (0, 1):
            with open(found_path(os.path.join(args.out, args.name), g, f), "rb") as h:
                found[f] = pickle.load(h)
        j = d["fold"] >= 0
        sizes = times_sr.merge_folds(found, d["fold"], d["y"], d["n"])
        size = times_sr.choose(sizes)
        ref = times_sr.weighted_rmse(d["ref"][j], d["y"][j], d["n"][j])
        res[g] = {"reference rmse": ref, "chosen size": size, "sizes": {str(s): q for s, q in sizes.items()}}
        log.info("%-18s reference RMSE %.4f | size %d RMSE %.4f | %s", g, ref, size, sizes[size]["rmse"], " || ".join(sizes[size]["formulas"]))
        if g in GATED:
            ratio.append(sizes[size]["rmse"] / ref)
    res["ratio to the reference"] = float(np.mean(ratio))
    log.info("ratio to the reference, mean over e/gamma: %.3f", res["ratio to the reference"])
    return res


def stage_common(args, log):
    groups = times_sr.all_groups()
    data = {g: group_data(args, g) for g in groups}
    scores, constants = {}, {}
    for name in FORMS:
        scores[name], constants[name] = {}, {}
        for g, d in data.items():
            pred = np.full(d["y"].shape, np.nan)
            constants[name][g] = []
            for f in (0, 1):
                train, judged = d["fold"] != f, d["fold"] == f
                theta = fit_form(name, d["v"][train], d["y"][train], d["n"][train])
                pred[judged] = predict(theta, name, d["v"][judged])
                constants[name][g].append(theta.tolist())
            scores[name][g] = times_sr.score_rows(pred, d)
    reference = {g: times_sr.score_rows(d["ref"], d) for g, d in data.items()}
    mean = {f: float(np.mean([scores[f][g]["rmse"] for g in groups])) for f in FORMS}
    ratios = {f: [scores[f][g]["rmse"] / reference[g]["rmse"] for g in GATED] for f in FORMS}
    chosen = times_sr.choose_common(mean, FORMS)
    out = {"scores": scores, "reference": reference, "constants": constants, "mean rmse": mean, "ratios": ratios, "chosen": chosen,
           "stands": bool(times_sr.stands({"ln mean energy": ratios[chosen]}))}
    for g in groups:
        log.info("%-18s ref %.4f | %s", g, reference[g]["rmse"], " ".join(f"{f} {scores[f][g]['rmse']:.4f}" for f in FORMS))
    for f in FORMS:
        log.info("%s numbers %d mean RMSE %.4f | e/gamma ratio %s mean %.2f", f, FORMS[f], mean[f], np.round(ratios[f], 2).tolist(), np.mean(ratios[f]))
    log.info("chosen %s stands %s", chosen, out["stands"])
    return out


def stage_law(args, log):
    from symbolic import particles
    skip = set()
    if args.skip:
        with open(args.skip) as h:
            skip = set(h.read().split())
    files = [f for f in sorted(glob.glob(os.path.join(args.out, "tables", "*.npz"))) if os.path.basename(f) not in skip]
    laws = {"form": np.array(args.form)}
    for s, r in particles.plane_rows([dict(np.load(f)) for f in files]).items():
        y = particles.targets(r)["mean_energy"][1]
        for cn in rings_sr.CLASSES:
            m = r["cls"] == cn
            laws[f"{s}.{cn}"] = fit_form(args.form, rings_sr.plane_inputs(r["x"][m], r["k"][m]), y[m], r["ring_count"][m])
            log.info("%-8s %-8s %s", s, cn, np.round(laws[f"{s}.{cn}"], 3).tolist())
    np.savez(os.path.join(args.out, args.name, LAW), **laws)
    return {"form": args.form, "showers": [os.path.basename(f) for f in files]}


def main():
    from symbolic import logs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("table", "fit", "search", "common", "law"))
    ap.add_argument("--out", default=FOLDER)
    ap.add_argument("--name", default="sr_energy")
    ap.add_argument("--group", help="search stage")
    ap.add_argument("--fold", type=int, help="search stage")
    ap.add_argument("--iterations", type=int, default=N_ITER)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--form", choices=tuple(FORMS), help="law stage")
    ap.add_argument("--skip", help="law stage: file listing the showers left out")
    args = ap.parse_args()
    if args.stage == "search":
        stage_search(args)
        return
    run, log = logs.start(args.name, args)
    logs.save(run, {"table": stage_table, "fit": stage_fit, "common": stage_common, "law": stage_law}[args.stage](args, log), args, log)


if __name__ == "__main__":
    main()
