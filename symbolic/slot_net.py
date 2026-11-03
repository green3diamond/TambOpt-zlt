"""A small network in one failing slot of the model, checked on held-out showers before distilling it: one network for
all groups, the formula inputs plus one-hot group, ln mean particle energy per ring.

Stages:
  fit    (tambo env) per fold and seed a network on the fold's training showers, scored on its judged showers against
         the ring-by-ring law. The networks are saved for querying.
  query  (tambo env) per fold and group, training rows with ln r spread across each ring and the networks' answer there,
         and the same rows at the ring centres with the CORSIKA values, the control.
  distil_search  (symbolic env, PySR) per source, fold and seed, a search for one shape shared by the groups.
  distil (tambo env) both sources' searches scored on the judged showers against the ring-by-ring law.
"""
import argparse
import os
import pickle
import subprocess
import sys
import time
from multiprocessing.pool import ThreadPool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import energy_sr, rings_sr, times_sr

WIDTH = 64
DEPTH = 3
LR = 1e-3
DECAY = 1e-4
BATCH = 4096
PATIENCE = 30
MAX_EPOCHS = 1000
SEEDS = 3
VAL_SHARE = 0.2
MEAN_BAR = 1.0
GROUP_BAR = 1.25
SHARED = {"A": 6, "B": 6, "C": 6, "D": 6}                  # numbers per group of the searched shape
TEMPLATE = ("A[g] * f(ln_r - C[g], age, ln_k, l, c, ln_Ek, ln_rm) + "
            "B[g] * h(ln_r - C[g], age, ln_k, l, c, ln_Ek, ln_rm) + D[g]")
QUERY = "query.npz"
SEARCH_SECONDS = 1800
SEARCH_ITER = 100000
SEARCH_SEEDS = 3
ROWS = 4000
SOURCES = ("network", "corsika")


def split_showers(showers, rng, share=VAL_SHARE):
    """Mask of the rows whose shower is in a random share of the showers, so no shower sits on both sides."""
    ids = np.unique(showers)
    return np.isin(showers, rng.choice(ids, max(1, int(round(share * len(ids)))), replace=False))


def passes(ratios):
    return bool(np.mean(ratios) <= MEAN_BAR and max(ratios) <= GROUP_BAR)


class Standard:
    """Inputs and target scaled by the training rows' mean and sd; one-hot columns left alone."""

    def __init__(self, X, y, n_plain):
        self.mu, self.sd = X[:, :n_plain].mean(0), X[:, :n_plain].std(0) + 1e-9
        self.y_mu, self.y_sd = y.mean(), y.std() + 1e-9

    def x(self, X):
        out = X.copy()
        k = len(self.mu)
        out[:, :k] = (X[:, :k] - self.mu) / self.sd
        return out


def train(X, y, val, seed, max_epochs=MAX_EPOCHS, log=None):
    """A network on the rows not in val, early stopping on val -> predict(X)."""
    import torch
    torch.manual_seed(seed)
    n_plain = X.shape[1] - np.isin(X, (0.0, 1.0)).all(0).sum()
    sc = Standard(X[~val], y[~val], n_plain)
    tx = torch.as_tensor(sc.x(X), dtype=torch.float32)
    ty = torch.as_tensor((y - sc.y_mu) / sc.y_sd, dtype=torch.float32)
    layers, size = [], X.shape[1]
    for _ in range(DEPTH):
        layers += [torch.nn.Linear(size, WIDTH), torch.nn.Tanh()]
        size = WIDTH
    net = torch.nn.Sequential(*layers, torch.nn.Linear(size, 1))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=DECAY)
    idx = np.flatnonzero(~val)
    rng = np.random.default_rng(seed)
    best, best_state, wait, epoch = np.inf, None, 0, 0
    for epoch in range(max_epochs):
        for b in np.array_split(rng.permutation(idx), max(1, len(idx) // BATCH)):
            opt.zero_grad()
            loss = ((net(tx[b]).squeeze(1) - ty[b]) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(((net(tx[val]).squeeze(1) - ty[val]) ** 2).mean())
        if v < best - 1e-5:
            best, best_state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    if log:
        log.info("seed %d stopped at epoch %d, validation %.4f of the target's variance", seed, epoch, best)

    def predict(Q):
        with torch.no_grad():
            return net(torch.as_tensor(sc.x(Q), dtype=torch.float32)).squeeze(1).numpy().astype(np.float64) * sc.y_sd + sc.y_mu
    predict.net, predict.scale = net, sc
    return predict


def one_hot(a, n):
    out = np.zeros((1, n))
    out[0, a] = 1.0
    return out


def pooled(data, groups, pick):
    """Rows with particles of the planes pick(d) keeps, all groups: inputs with one-hot group, target, shower key."""
    Xs, ys, ss = [], [], []
    for a, g in enumerate(groups):
        d = data[g]
        keep = pick(d)
        m = ((d["n"][keep] > 0) & np.isfinite(d["y"][keep])).ravel()
        X = times_sr.expand(d["v"][keep])[m]
        Xs.append(np.column_stack([X, np.repeat(one_hot(a, len(groups)), len(X), axis=0)]))
        ys.append(d["y"][keep].ravel()[m])
        ss.append(np.repeat(d["shower"][keep], times_sr.N_RING)[m])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)


def load_nets(run, f, seeds, n_in):
    """The fold's saved networks -> predict(Q), their mean."""
    import torch
    nets = []
    for seed in range(seeds):
        st = torch.load(os.path.join(run, f"net_fold{f}_seed{seed}.pt"), weights_only=False)
        layers, size = [], n_in
        for _ in range(DEPTH):
            layers += [torch.nn.Linear(size, WIDTH), torch.nn.Tanh()]
            size = WIDTH
        net = torch.nn.Sequential(*layers, torch.nn.Linear(size, 1))
        net.load_state_dict(st["state"])
        nets.append((net, st))

    def predict(Q):
        out = []
        for net, st in nets:
            x = Q.copy()
            k = len(st["mu"])
            x[:, :k] = (Q[:, :k] - st["mu"]) / st["sd"]
            with torch.no_grad():
                out.append(net(torch.as_tensor(x, dtype=torch.float32)).squeeze(1).numpy().astype(np.float64) * st["y_sd"] + st["y_mu"])
        return np.mean(out, axis=0)
    return predict


def stage_query(args, log):
    groups = times_sr.all_groups()
    data = {g: energy_sr.group_data(args, g) for g in groups}
    run = os.path.join(args.out, args.net)
    lo, hi = np.log(times_sr.EDGE_IN), np.log(times_sr.EDGE_OUT)
    out = {}
    for f in (0, 1):
        predict = load_nets(run, f, SEEDS, 7 + len(groups))
        for a, g in enumerate(groups):
            d = data[g]
            train = d["fold"] != f
            X, y, n = times_sr.expand(d["v"][train]), d["y"][train].ravel(), d["n"][train].ravel()
            ok = np.flatnonzero((n > 0) & np.isfinite(y))
            rng = np.random.default_rng([f, a])
            keep = rng.choice(ok, min(args.rows, len(ok)), replace=False)
            ring = keep % times_sr.N_RING
            spread = X[keep].copy()
            spread[:, 0] = rng.uniform(lo[ring], hi[ring])
            Q = np.column_stack([spread, np.repeat(one_hot(a, len(groups)), len(keep), axis=0)])
            out[f"network.{f}.{a}.X"], out[f"network.{f}.{a}.y"] = spread, predict(Q)
            out[f"corsika.{f}.{a}.X"], out[f"corsika.{f}.{a}.y"] = X[keep], y[keep]
            log.info("fold %d %-18s rows %d, network vs CORSIKA rms on these rows at ring centres %.3f", f, g, len(keep),
                     float(np.sqrt(np.mean((predict(np.column_stack([X[keep], Q[:, 7:]])) - y[keep]) ** 2))))
    np.savez(os.path.join(run, QUERY), **out)
    return {"rows": args.rows}


def found_path(run, source, f, seed):
    return os.path.join(run, "found", f"{source}_fold{f}_seed{seed}.pkl")


def stage_distil_one(args):
    """One search, saved as {size: ({group: prediction on its judged planes}, text, search loss)}."""
    from pysr import TemplateExpressionSpec
    run, f = os.path.join(args.out, args.name), args.fold
    groups = times_sr.all_groups()
    q = np.load(os.path.join(args.out, args.net, QUERY))
    X = np.concatenate([np.column_stack([q[f"{args.source}.{f}.{a}.X"], np.full(len(q[f"{args.source}.{f}.{a}.X"]), a + 1.0)]) for a in range(len(groups))])
    y = np.concatenate([q[f"{args.source}.{f}.{a}.y"] for a in range(len(groups))])
    names = rings_sr.VARIABLES + ["g"]
    spec = TemplateExpressionSpec(combine=TEMPLATE, expressions=["f", "h"], variable_names=names, parameters=SHARED)
    model = rings_sr.regressor(run, f"{args.source}_fold{f}_seed{args.seed}", 10 * f + args.seed, args.iterations, timeout=args.seconds, expression_spec=spec)
    model.fit(X, y, variable_names=names)
    found = {}
    for i, row in model.equations_.iterrows():
        preds = {}
        for a, g in enumerate(groups):
            d = energy_sr.group_data(args, g)
            test = times_sr.expand(d["v"][d["fold"] == f])
            preds[g] = np.asarray(model.predict(np.column_stack([test, np.full(len(test), a + 1.0)]), index=i), dtype=float)
        if all(np.isfinite(p).all() for p in preds.values()):
            found[int(row["complexity"])] = ({g: p.reshape(-1, times_sr.N_RING).astype(np.float32) for g, p in preds.items()}, str(row["equation"]), float(row["loss"]))
    path = found_path(run, args.source, f, args.seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as h:
        pickle.dump(found, h)
    os.replace(path + ".tmp", path)
    sys.stdout.flush()
    os._exit(0)


def run_search(job):
    args, source, f, seed = job
    t0 = time.time()
    subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "distil_one", "--out", args.out, "--name", args.name, "--net", args.net, "--source", source,
                    "--fold", str(f), "--seed", str(seed), "--iterations", str(args.iterations), "--seconds", str(args.seconds)],
                   check=True, timeout=args.seconds + rings_sr.MARGIN)
    return source, f, seed, time.time() - t0


def stage_distil_search(args, log):
    jobs = [(args, src, f, s) for src in SOURCES for f in (0, 1) for s in range(args.search_seeds)]
    with ThreadPool(args.workers) as pool:
        for src, f, s, sec in pool.imap_unordered(run_search, jobs):
            log.info("%s fold %d seed %d searched in %.0f s", src, f, s, sec)
    return {"searches": len(jobs)}


def best_by_size(found_seeds):
    """Per size, the entry with the lowest search loss over the seeds. found_seeds: [{size: (preds, text, loss)}]."""
    out = {}
    for found in found_seeds:
        for size, entry in found.items():
            if size not in out or entry[2] < out[size][2]:
                out[size] = entry
    return out


def stage_distil(args, log):
    groups = times_sr.all_groups()
    data = {g: energy_sr.group_data(args, g) for g in groups}
    run = os.path.join(args.out, args.name)
    reference = {g: times_sr.score_rows(data[g]["ref"], data[g]) for g in groups}
    res = {"reference": reference}
    for src in SOURCES:
        found = {}
        for f in (0, 1):
            seeds = []
            for s in range(args.search_seeds):
                with open(found_path(run, src, f, s), "rb") as h:
                    seeds.append(pickle.load(h))
            found[f] = best_by_size(seeds)
        rows = {}
        for size in sorted(set(found[0]) | set(found[1])):
            pick = [max((s for s in found[f] if s <= size), default=None) for f in (0, 1)]
            if None in pick:
                continue
            pred = {g: np.full(data[g]["y"].shape, np.nan) for g in groups}
            for f, s in zip((0, 1), pick):
                for g in groups:
                    pred[g][data[g]["fold"] == f] = found[f][s][0][g]
            scores = {g: times_sr.score_rows(pred[g], data[g]) for g in groups}
            rows[size] = {"mean": float(np.mean([scores[g]["rmse"] for g in groups])), "scores": scores,
                          "formulas": [found[f][s][1] for f, s in zip((0, 1), pick)]}
        best = min(q["mean"] for q in rows.values())
        size = min(s for s, q in rows.items() if q["mean"] <= best * (1 + times_sr.TOLERANCE))
        q = rows[size]
        ratios = [q["scores"][g]["rmse"] / reference[g]["rmse"] for g in energy_sr.GATED]
        numbers = len(SHARED) + max(times_sr.shared_constants(t) for t in q["formulas"]) / len(groups)
        res[src] = {"size": size, "mean rmse": q["mean"], "ratios": ratios, "mean ratio": float(np.mean(ratios)), "numbers": numbers,
                    "stands": bool(times_sr.stands({"ln mean energy": ratios})), "scores": q["scores"], "formulas": q["formulas"],
                    "sizes": {str(s): v["mean"] for s, v in rows.items()}}
        log.info("%s: size %d, numbers %.1f, mean RMSE %.4f | e/gamma ratio %s mean %.3f | stands %s", src, size, numbers, q["mean"], np.round(ratios, 2).tolist(), np.mean(ratios), res[src]["stands"])
        for g in groups:
            log.info("   %-18s law %.4f formula %.4f", g, reference[g]["rmse"], q["scores"][g]["rmse"])
        log.info("   %s", " || ".join(t[:420] for t in q["formulas"]))
    return res


def stage_fit(args, log):
    import torch
    torch.set_num_threads(args.workers)
    groups = times_sr.all_groups()
    data = {g: energy_sr.group_data(args, g) for g in groups}
    preds = {g: np.full(data[g]["y"].shape, np.nan) for g in groups}
    run = os.path.join(args.out, args.name)
    for f in (0, 1):
        X, y, showers = pooled(data, groups, lambda d: d["fold"] != f)
        log.info("fold %d: %d rows, %d showers", f, len(y), len(np.unique(showers)))
        nets = []
        for seed in range(args.seeds):
            val = split_showers(showers, np.random.default_rng(100 * f + seed))
            nets.append(train(X, y, val, seed, args.max_epochs, log))
            torch.save({"state": nets[-1].net.state_dict(), "mu": nets[-1].scale.mu, "sd": nets[-1].scale.sd, "y_mu": nets[-1].scale.y_mu,
                        "y_sd": nets[-1].scale.y_sd}, os.path.join(run, f"net_fold{f}_seed{seed}.pt"))
        for a, g in enumerate(groups):
            j = data[g]["fold"] == f
            Q = times_sr.expand(data[g]["v"][j])
            Q = np.column_stack([Q, np.repeat(one_hot(a, len(groups)), len(Q), axis=0)])
            preds[g][j] = np.mean([n(Q) for n in nets], axis=0).reshape(-1, times_sr.N_RING)
    scores = {g: times_sr.score_rows(preds[g], data[g]) for g in groups}
    reference = {g: times_sr.score_rows(data[g]["ref"], data[g]) for g in groups}
    ratios = [scores[g]["rmse"] / reference[g]["rmse"] for g in energy_sr.GATED]
    for g in groups:
        log.info("%-18s law %.4f network %.4f | median %.3f -> %.3f | worst 1%% share %.2f -> %.2f | under 10 m %.3f -> %.3f, 10-400 m %.3f -> %.3f, beyond 400 m %.3f -> %.3f",
                 g, reference[g]["rmse"], scores[g]["rmse"], reference[g]["median"], scores[g]["median"], reference[g]["top share"], scores[g]["top share"],
                 reference[g]["under 10 m"], scores[g]["under 10 m"], reference[g]["10-400 m"], scores[g]["10-400 m"], reference[g]["beyond 400 m"], scores[g]["beyond 400 m"])
    log.info("e/gamma ratio %s mean %.3f | passes %s", np.round(ratios, 2).tolist(), np.mean(ratios), passes(ratios))
    return {"scores": scores, "reference": reference, "ratios": ratios, "passes": passes(ratios)}


def main():
    from symbolic import logs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("fit", "query", "distil_search", "distil_one", "distil"))
    ap.add_argument("--out", default=rings_sr.FOLDER)
    ap.add_argument("--name", default="slot_energy")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--net", default="slot_energy", help="query and distil stages: folder of the networks")
    ap.add_argument("--rows", type=int, default=ROWS, help="query stage: rows per group")
    ap.add_argument("--source", choices=SOURCES)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--iterations", type=int, default=SEARCH_ITER)
    ap.add_argument("--seconds", type=int, default=SEARCH_SECONDS)
    ap.add_argument("--search-seeds", type=int, default=SEARCH_SEEDS)
    args = ap.parse_args()
    if args.stage == "distil_one":
        stage_distil_one(args)
        return
    run, log = logs.start(args.name, args)
    logs.save(run, {"fit": stage_fit, "query": stage_query, "distil_search": stage_distil_search, "distil": stage_distil}[args.stage](args, log), args, log)


if __name__ == "__main__":
    main()
