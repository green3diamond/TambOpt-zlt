"""Distil the f(ln r, ln k) piece of the structured network (struct_net.py) with SymTorch (torch-symbolic), the method's own
library: the network piece wrapped as the block, ln r, ln k and ln r - ln k as variable transforms, one search per group
and fold with a larger size limit and a longer search than struct_sr.py. Scored with struct_sr.py's h formulas.

Stages (symbolic env):
  search  one distillation in its own process -> found/<group>_fold<f>_f.pkl: per complexity the prediction on the
          judged rows, the equation and the loss, PySR's best complexity, and the network's f on those rows.
  fit     every search in parallel, then the score: the f copy error (count-weighted, against the network's f) next to the
          network's error against CORSIKA, and c + f + h on the judged showers against the ring-by-ring law.
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

from symbolic import energy_sr, rings_sr, struct_sr, times_sr

MAXSIZE = 40
N_ITER = 7000
TIMEOUT = 7200
MARGIN = 1800
SR_PARAMS = {"binary_operators": ["+", "-", "*", "/"], "unary_operators": ["exp", "log", "tanh", "square"],
             "nested_constraints": {"exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0}, "tanh": {"tanh": 0}},
             "maxsize": MAXSIZE, "parsimony": struct_sr.PARSIMONY, "niterations": N_ITER, "timeout_in_seconds": TIMEOUT,
             "deterministic": True, "parallelism": "serial", "progress": False, "verbosity": 0, "model_selection": "best"}


def f_block(saved, offset):
    """The network's f piece as a torch module on (ln r, ln k), centred as in struct_sr's query."""
    import torch
    from symbolic import struct_net

    class FPiece(torch.nn.Module):
        def forward(self, x):
            X = np.zeros((len(x), 7))
            X[:, 0], X[:, 2] = x[:, 0].numpy(), x[:, 1].numpy()
            return torch.as_tensor(struct_net.predict(saved, X, "f") - offset, dtype=torch.float32)[:, None]
    return FPiece()


def found_path(run, group, f):
    return os.path.join(run, "found", f"{group.replace(' ', '_')}_fold{f}_f.pkl")


def stage_search(args):
    import torch
    from symtorch import SymbolicModel
    run, g, f = os.path.join(args.out, args.name), args.group, args.fold
    key = f"{g.replace(' ', '.')}.{f}"
    q = np.load(os.path.join(args.out, struct_sr.NAME, struct_sr.QUERY))
    saved = torch.load(os.path.join(args.out, struct_sr.NET, f"nets_fold{f}.pt"), weights_only=False)[g]
    grid = torch.as_tensor(q[f"{key}.f_X"][:, :2], dtype=torch.float32)
    block = f_block(saved, 0.0)
    offset = float(block(grid).mean())
    model = SymbolicModel(f_block(saved, offset), block_name=f"{g.replace(' ', '_')}_fold{f}_f")
    model.distill(grid, save_path=os.path.join(run, "symtorch"),
                  variable_transforms=[lambda x: x[:, 0], lambda x: x[:, 1], lambda x: x[:, 0] - x[:, 1]],
                  sr_params={**SR_PARAMS, "random_state": f, "niterations": args.iterations},
                  fit_params={"variable_names": struct_sr.VARS["f"]})
    d = energy_sr.group_data(args, g)
    judged = torch.as_tensor(struct_sr.judged_inputs(d["v"][d["fold"] == f])[0][:, :2], dtype=torch.float32)
    reg = model.pysr_regressor[0]
    found = {}
    for _, row in reg.equations_.iterrows():
        c = int(row["complexity"])
        p = np.asarray(model.get_symbolic_function(dim=0, complexity=c)(judged), dtype=float).ravel()
        if np.isfinite(p).all():
            found[c] = (p.astype(np.float32), str(row["equation"]), float(row["loss"]))
    out = {"found": found, "best": int(reg.get_best()["complexity"]), "network": model.symtorch_block(judged).numpy().ravel()}
    path = found_path(run, g, f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as h:
        pickle.dump(out, h)
    os.replace(path + ".tmp", path)
    sys.stdout.flush()
    os._exit(0)


def run_search(job):
    args, g, f = job
    t0 = time.time()
    subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "search", "--out", args.out, "--name", args.name, "--group", g,
                    "--fold", str(f), "--iterations", str(args.iterations)], check=True, timeout=TIMEOUT + MARGIN)
    return g, f, time.time() - t0


def stage_fit(args, log):
    groups = times_sr.all_groups()
    with ThreadPool(args.workers) as pool:
        for g, f, sec in pool.imap_unordered(run_search, [(args, g, f) for g in groups for f in (0, 1)]):
            log.info("%-18s fold %d searched in %.0f s", g, f, sec)
    run, old = os.path.join(args.out, args.name), os.path.join(args.out, struct_sr.NAME)
    q = np.load(os.path.join(old, struct_sr.QUERY))
    res, copied, ratios = {}, [], []
    for g in groups:
        d = energy_sr.group_data(args, g)
        pred, net, f_hat, f_net = (np.full(d["y"].shape, np.nan) for _ in range(4))
        res[g] = {"formulas": {}}
        for f in (0, 1):
            key = f"{g.replace(' ', '.')}.{f}"
            with open(found_path(run, g, f), "rb") as h:
                F = pickle.load(h)
            with open(struct_sr.found_path(old, g, f, "h"), "rb") as h:
                H = pickle.load(h)
            judged = d["fold"] == f
            pred[judged] = struct_sr.compose(q, key, F["found"], H["found"], F["best"], H["best"])
            net[judged] = q[f"{key}.network"]
            f_hat[judged] = F["found"][F["best"]][0].reshape(-1, times_sr.N_RING)
            f_net[judged] = F["network"].reshape(-1, times_sr.N_RING)
            res[g]["formulas"][f] = {"f": F["found"][F["best"]][1], "f size": F["best"]}
        j = d["fold"] >= 0
        rmse = lambda a, b: times_sr.weighted_rmse(a[j], b[j], d["n"][j])
        r = res[g]
        r.update({"law": rmse(d["ref"], d["y"]), "network": rmse(net, d["y"]), "formula": rmse(pred, d["y"]),
                  "f copy": rmse(f_hat, f_net), "formula vs network": rmse(pred, net)})
        log.info("%-18s law %.4f network %.4f formula %.4f | f copy %.4f (%.2f of the network's error) | ratio %.2f | %s", g, r["law"],
                 r["network"], r["formula"], r["f copy"], r["f copy"] / r["network"], r["formula"] / r["law"],
                 " || ".join(f"f{f} [{v['f size']}] {v['f']}" for f, v in r["formulas"].items()))
        if g in energy_sr.GATED:
            copied.append(r["f copy"] / r["network"])
            ratios.append(r["formula"] / r["law"])
    passes = sum(c <= 0.5 for c in copied) >= 3
    log.info("e/gamma f copy over network error %s | copied well %d of 4 | passes %s | ratio to the law %s mean %.3f",
             np.round(copied, 2).tolist(), sum(c <= 0.5 for c in copied), passes, np.round(ratios, 2).tolist(), np.mean(ratios))
    res.update({"f copy over network error": copied, "passes": passes, "e/gamma ratio": ratios})
    return res


def main():
    from symbolic import logs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("fit", "search"))
    ap.add_argument("--out", default=rings_sr.FOLDER)
    ap.add_argument("--name", default="struct_symtorch")
    ap.add_argument("--group")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--iterations", type=int, default=N_ITER)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    if args.stage == "search":
        stage_search(args)
        return
    run, log = logs.start(args.name, args)
    logs.save(run, stage_fit(args, log), args, log)


if __name__ == "__main__":
    main()
