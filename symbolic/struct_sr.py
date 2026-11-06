"""Distil the structured network of struct_net.py piece by piece (SymTorch's method): ln mean energy = c + f(ln r, ln k)
+ h(age, l, c, ln Ek, ln rm), one formula search per piece, group and fold on the network's own answers.

Stages:
  query  (tambo env) per group and fold, each piece of the fold's network on its training support: f on the (ln r, ln k)
         grid of the training rings, h on the training planes, both centred (offsets into the constant); and on the judged
         planes, the network's prediction and the inputs. -> struct_sr/query.npz
  fit    (symbolic env) one search per group, fold and piece in its own process, then the score: PySR's best formula of each
         piece substituted, on the judged showers against the ring-by-ring law (as before) and against the network.
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

QUERY = "query.npz"
NAME = "struct_sr"
NET = "struct_energy"
H_COLS = [0, 2, 3, 4, 5]                                  # columns of rings_sr.plane_inputs that h sees
VARS = {"f": ["ln_r", "ln_k", "ln_r_minus_ln_k"], "h": ["age", "l", "c", "ln_Ek", "ln_rm"]}
MAXSIZE = {"f": 25, "h": 30}
PARSIMONY = 0.01
TIMEOUT = 1200
N_ITER = 7000


def f_inputs(ln_r, ln_k):
    return np.column_stack([ln_r, ln_k, ln_r - ln_k])


def judged_inputs(v):
    """f and h inputs of every (judged plane, ring) row and of every judged plane."""
    return f_inputs(np.tile(times_sr.LN_R, len(v)), np.repeat(v[:, 1], times_sr.N_RING)), v[:, H_COLS]


def stage_query(args, log):
    import torch
    from symbolic import struct_net
    nets = {f: torch.load(os.path.join(args.out, NET, f"nets_fold{f}.pt"), weights_only=False) for f in (0, 1)}
    out = {}
    for g in times_sr.all_groups():
        d = energy_sr.group_data(args, g)
        key = g.replace(" ", ".")
        for f in (0, 1):
            net = nets[f][g]
            train, judged = d["fold"] != f, d["fold"] == f
            ln_k = np.unique(d["v"][train][:, 1])
            grid = np.column_stack([np.tile(times_sr.LN_R, len(ln_k)), np.repeat(ln_k, times_sr.N_RING)])
            X_f = np.zeros((len(grid), 7))
            X_f[:, 0], X_f[:, 2] = grid[:, 0], grid[:, 1]
            planes = d["v"][train]
            X_h = times_sr.expand(planes)[:: times_sr.N_RING]
            f_val, h_val = struct_net.predict(net, X_f, "f"), struct_net.predict(net, X_h, "h")
            X_j = times_sr.expand(d["v"][judged])
            full = struct_net.predict(net, X_j)
            const = full - struct_net.predict(net, X_j, "f") - struct_net.predict(net, X_j, "h")
            if np.ptp(const) > 1e-4:
                raise ValueError(f"{g} fold {f}: the network is not c + f + h")
            base = float(const[0]) + f_val.mean() + h_val.mean()
            out.update({f"{key}.{f}.f_X": f_inputs(grid[:, 0], grid[:, 1]), f"{key}.{f}.f_y": f_val - f_val.mean(),
                        f"{key}.{f}.h_X": planes[:, H_COLS], f"{key}.{f}.h_y": h_val - h_val.mean(), f"{key}.{f}.base": base,
                        f"{key}.{f}.network": full.reshape(-1, times_sr.N_RING)})
            log.info("%-18s fold %d | f grid %d points, sd %.3f | h planes %d, sd %.3f", g, f, len(grid), f_val.std(), len(planes), h_val.std())
    os.makedirs(os.path.join(args.out, args.name), exist_ok=True)
    np.savez(os.path.join(args.out, args.name, QUERY), **out)
    return {"groups": len(times_sr.all_groups())}


def found_path(run, group, f, piece):
    return os.path.join(run, "found", f"{group.replace(' ', '_')}_fold{f}_{piece}.pkl")


def stage_search(args):
    """One search: {complexity: (prediction on the judged rows or planes, equation, loss)} and PySR's best complexity.
    Exits without Julia's teardown once the result is on disk."""
    from pysr import PySRRegressor
    run, g, f, piece = os.path.join(args.out, args.name), args.group, args.fold, args.piece
    q = np.load(os.path.join(run, QUERY))
    key = f"{g.replace(' ', '.')}.{f}"
    d = energy_sr.group_data(args, g)
    jf, jh = judged_inputs(d["v"][d["fold"] == f])
    model = PySRRegressor(niterations=args.iterations, binary_operators=["+", "-", "*", "/"], unary_operators=["exp", "log", "tanh", "square"],
                          maxsize=MAXSIZE[piece], parsimony=PARSIMONY,
                          nested_constraints={"exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0}, "tanh": {"tanh": 0}},
                          output_directory=os.path.join(run, "pysr"), run_id=f"{g.replace(' ', '_')}_fold{f}_{piece}", random_state=f,
                          deterministic=True, parallelism="serial", progress=False, verbosity=0, timeout_in_seconds=TIMEOUT,
                          model_selection="best")
    model.fit(q[f"{key}.{piece}_X"], q[f"{key}.{piece}_y"], variable_names=VARS[piece])
    test = jf if piece == "f" else jh
    found = {}
    for i, row in model.equations_.iterrows():
        p = np.asarray(model.predict(test, index=i), dtype=float)
        if np.isfinite(p).all():
            found[int(row["complexity"])] = (p.astype(np.float32), str(row["equation"]), float(row["loss"]))
    best = int(model.get_best()["complexity"])
    path = found_path(run, g, f, piece)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as h:
        pickle.dump({"found": found, "best": best}, h)
    os.replace(path + ".tmp", path)
    sys.stdout.flush()
    os._exit(0)


def run_search(job):
    args, g, f, piece = job
    t0 = time.time()
    subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "search", "--out", args.out, "--name", args.name, "--group", g,
                    "--fold", str(f), "--piece", piece, "--iterations", str(args.iterations)], check=True, timeout=TIMEOUT + rings_sr.MARGIN)
    return g, f, piece, time.time() - t0


def compose(q, key, f_found, h_found, f_size, h_size):
    """ln mean energy on the judged rows from one f and one h formula."""
    f_pred = f_found[f_size][0].reshape(-1, times_sr.N_RING)
    return q[f"{key}.base"] + f_pred + h_found[h_size][0][:, None]


def stage_fit(args, log):
    groups = times_sr.all_groups()
    jobs = [(args, g, f, piece) for g in groups for f in (0, 1) for piece in ("f", "h")]
    with ThreadPool(args.workers) as pool:
        for g, f, piece, sec in pool.imap_unordered(run_search, jobs):
            log.info("%-18s fold %d %s searched in %.0f s", g, f, piece, sec)
    run = os.path.join(args.out, args.name)
    q = np.load(os.path.join(run, QUERY))
    res, ratios, net_ratios = {}, [], []
    for g in groups:
        d = energy_sr.group_data(args, g)
        pred, net = np.full(d["y"].shape, np.nan), np.full(d["y"].shape, np.nan)
        res[g] = {"formulas": {}}
        for f in (0, 1):
            key = f"{g.replace(' ', '.')}.{f}"
            with open(found_path(run, g, f, "f"), "rb") as h:
                F = pickle.load(h)
            with open(found_path(run, g, f, "h"), "rb") as h:
                H = pickle.load(h)
            judged = d["fold"] == f
            pred[judged] = compose(q, key, F["found"], H["found"], F["best"], H["best"])
            net[judged] = q[f"{key}.network"]
            res[g]["formulas"][f] = {"f": F["found"][F["best"]][1], "f size": F["best"], "h": H["found"][H["best"]][1], "h size": H["best"]}
        j = d["fold"] >= 0
        law = times_sr.weighted_rmse(d["ref"][j], d["y"][j], d["n"][j])
        res[g].update({"law": law, "network": times_sr.weighted_rmse(net[j], d["y"][j], d["n"][j]),
                       "formula": times_sr.weighted_rmse(pred[j], d["y"][j], d["n"][j]),
                       "formula vs network": times_sr.weighted_rmse(pred[j], net[j], d["n"][j])})
        r = res[g]
        log.info("%-18s law %.4f network %.4f formula %.4f | formula vs network %.4f | ratio %.2f | %s", g, r["law"], r["network"], r["formula"],
                 r["formula vs network"], r["formula"] / r["law"], " || ".join(f"f{f} {v['f']} ; h{f} {v['h']}" for f, v in r["formulas"].items()))
        if g in energy_sr.GATED:
            ratios.append(r["formula"] / r["law"])
            net_ratios.append(r["formula vs network"] / r["network"])
    res["e/gamma ratio"], res["copy over network error"] = ratios, net_ratios
    log.info("e/gamma ratio to the law %s mean %.3f | formula-vs-network over network-vs-CORSIKA %s",
             np.round(ratios, 2).tolist(), np.mean(ratios), np.round(net_ratios, 2).tolist())
    return res


def main():
    from symbolic import logs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("query", "fit", "search"))
    ap.add_argument("--out", default=rings_sr.FOLDER)
    ap.add_argument("--name", default="struct_sr")
    ap.add_argument("--group")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--piece", choices=("f", "h"))
    ap.add_argument("--iterations", type=int, default=N_ITER)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    if args.stage == "search":
        stage_search(args)
        return
    run, log = logs.start(args.name, args)
    logs.save(run, {"query": stage_query, "fit": stage_fit}[args.stage](args, log), args, log)


if __name__ == "__main__":
    main()
