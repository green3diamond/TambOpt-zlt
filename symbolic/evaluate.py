"""Phase-1 comparison harness: the symbolic sampler (linear laws) against the same sampler with MLP laws, over
training ranges and sizes, scored against CORSIKA next to the neural chain.

Stages:
  draw   --task i: fit one (model, training set, seed) per class, 8 draws on the range set (the chain's 1,000
         primaries) and the band set (every test shower in [CUT, TOP)) -> OUT/evaluate/draws/<name>.npz
  table  every metric of study_plan.md (1b, 1c, 1d) from the saved draws -> OUT/evaluate/table/results.json
"""
import argparse
import glob
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from symbolic import logs
from modules.constants import SPECIES_NAMES
from symbolic import chain, copula as cp, fit_profiles, longitudinal, planes, sampler, score

FOLDER = "evaluate"
DRAWS = os.path.join(FOLDER, "draws")
CUT = 5.9
N_DRAWS = 8
SEEDS = (0, 1, 2)
SIZES = {"1k": 1000, "10k": 10000}
SETS = {"range": chain.DRAWS, "band": chain.DRAWS_BAND}
DRAW_SEED = 1000           # every model gets the same random numbers per class, set and draw (common random numbers)


def tasks():
    """(model, training set, seed): one fit each. The symbolic fit is deterministic, so it gets one seed on the
    full and cut sets; subsets are drawn per seed for both models."""
    out = [("sym", "full", 0), ("sym", "cut", 0)]
    out += [("sym", n, s) for n in SIZES for s in SEEDS]
    out += [("mlp", t, s) for t in ("full", "cut", *SIZES) for s in SEEDS]
    return out


def name(model, train, seed):
    return f"{model}_{train}_s{seed}"


def eval_set(out, key):
    """Rows, conditions, depths, classes, CORSIKA energies and chain draws of one evaluation set."""
    rows, nn = score.load_draws(out, SETS[key])
    zt = np.load(os.path.join(out, fit_profiles.TEST))
    if not np.array_equal(zt["meta.rows"], np.arange(len(zt["meta.rows"]))):
        raise ValueError("needs the profile fits of the whole test file")
    return {"rows": rows, "l": zt["meta.log10_E"][rows], "u": zt["meta.cos_zenith"][rows],
            "lab": zt["meta.labels"][rows], "X": zt["X"][rows],
            "cor": {s: zt[f"{s}.E"][rows].astype(np.float64) for s in SPECIES_NAMES}, "chain": nn}


def stage_draw(out, args, log):
    model, train, seed = tasks()[args.task]
    tr = sampler.load(os.path.join(args.out, fit_profiles.TRAIN))
    keep, top = None, sampler.TOP
    if train == "cut":
        top = CUT
    elif train in SIZES:
        keep = np.zeros(len(tr["meta"]["log10_E"]), bool)
        keep[np.random.default_rng(seed).choice(len(keep), SIZES[train], replace=False)] = True
    margin = (lambda: cp.Margin("jsu")) if model == "sym" else (lambda: cp.MLPMargin(seed=seed))
    sets = {k: eval_set(args.out, k) for k in SETS}
    res, arrays = {"task": [model, train, seed], "classes": {}}, {}
    for k, e in sets.items():
        for s in SPECIES_NAMES:
            arrays[f"{k}.{s}.E"] = np.zeros((len(e["rows"]), N_DRAWS, e["X"].shape[1]), np.float32)
        arrays[f"{k}.rows"] = e["rows"]
    for c, cn in planes.CLASS.items():
        t0 = time.time()
        m = sampler.fit(tr, c, top=top, keep=keep, margin=margin)
        fit_s = time.time() - t0
        used = tr["meta"]["log10_E"][sampler.class_rows(tr, c, top, keep)["rows"]]
        if used.max() >= top:
            raise ValueError(f"{cn}: training showers at or above {top}")
        steps = [g.steps for g in m["margins"]] if model == "mlp" else None
        log.info("%s %s: %d training showers, fit %.0f s%s", name(model, train, seed), cn, m["n_train"], fit_s,
                 f", MLP steps {min(steps)}-{max(steps)}" if steps else "")
        for ki, (k, e) in enumerate(sets.items()):
            sel = e["lab"] == c
            for j in range(N_DRAWS):
                rng = np.random.default_rng([DRAW_SEED, c, ki, j])
                d, _ = sampler.sample(m, e["l"][sel], e["u"][sel], e["X"][sel], rng)
                for s in SPECIES_NAMES:
                    arrays[f"{k}.{s}.E"][sel, j] = d[s]
        res["classes"][cn] = {"n_train": m["n_train"], "fit_seconds": fit_s, "mlp_steps": steps,
                              "redrawn_frac": m["n_redrawn"] / m["n_drawn"]}
    path = os.path.join(args.out, DRAWS, name(model, train, seed) + ".npz")
    chain.save(path, log, **arrays)
    return res


# ---------------------------------------------------------------- metrics

def summaries(E):
    """Total (log10 GeV), onset, peak and end plane of each shower; E (..., planes)."""
    lit = E >= longitudinal.E_EMPTY
    k = E.shape[-1]
    onset = np.where(lit.any(-1), np.argmax(lit, -1), k)
    end = np.where(lit.any(-1), k - 1 - np.argmax(lit[..., ::-1], -1), -1)
    return {"total": np.log10(1 + E.sum(-1)), "onset": onset, "peak": np.argmax(E, -1), "end": end}


def corr_matrix(Y, planes_ok):
    return np.nan_to_num(np.corrcoef(Y[:, planes_ok].T))


def classifier_auc(A, B, cond_a, cond_b, seed=0):
    """Two-sample classifier AUC (0.5 = indistinguishable), conditions as inputs, 2-fold; row i of A and B fall in
    the same fold."""
    n = min(len(A), len(B))
    fold = np.random.default_rng(seed).permutation(n) % 2
    X = np.vstack([np.column_stack([A[:n], cond_a[:n]]), np.column_stack([B[:n], cond_b[:n]])])
    y = np.r_[np.zeros(n), np.ones(n)]
    f = np.r_[fold, fold]
    aucs = []
    for k in (0, 1):
        tr, te = f != k, f == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        xt, xe = torch.as_tensor((X[tr] - mu) / sd), torch.as_tensor((X[te] - mu) / sd)
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                                  torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        yt = torch.as_tensor(y[tr])
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(xt)[:, 0], yt).backward()
            opt.step()
        with torch.no_grad():
            p = net(xe)[:, 0].numpy()
        r = stats.rankdata(p)
        pos = y[te] == 1
        aucs.append((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))
    return float(np.mean(aucs))


def obs_metrics(Ym, Yc, cond_m, cond_c):
    """Per-plane mean and sd, correlation matrix and classifier AUC: Ym (n, M, planes) model, Yc (n, planes)
    reference, both already log10(1 + E); the classifier sees draw 0 of each primary."""
    flat = Ym.reshape(-1, Yc.shape[-1])
    ok = (Yc.std(0) > 0) & (flat.std(0) > 0)
    return {"mean_abs_diff": float(np.mean(np.abs(flat.mean(0) - Yc.mean(0)))),
            "sd_abs_diff": float(np.mean(np.abs(flat.std(0) - Yc.std(0)))),
            "corr_frobenius": float(np.linalg.norm(corr_matrix(flat, ok) - corr_matrix(Yc, ok))),
            "auc": classifier_auc(Yc, Ym[:, 0], cond_c, cond_m)}


def species_metrics(Em, Ec, cond_m, cond_c):
    """obs_metrics plus Wasserstein-1 of the shower summaries: Em (n, M, planes), Ec (n, planes) energies."""
    out = obs_metrics(score.obs(Em), score.obs(Ec), cond_m, cond_c)
    sm, sc = summaries(Em.reshape(-1, Ec.shape[-1])), summaries(Ec)
    out.update({f"w1_{k}": float(stats.wasserstein_distance(sm[k], sc[k])) for k in sm})
    return out


def all_metrics(Em, Ec, cond_m, cond_c):
    """Per species and joint (the three species' observed planes side by side)."""
    out = {s: species_metrics(Em[s], Ec[s], cond_m, cond_c) for s in SPECIES_NAMES}
    out["joint"] = obs_metrics(np.concatenate([score.obs(Em[s]) for s in SPECIES_NAMES], -1),
                               np.concatenate([score.obs(Ec[s]) for s in SPECIES_NAMES], -1), cond_m, cond_c)
    return out


def split_half(Ec, cond, seed=0):
    """The same metrics between two random halves of CORSIKA, for scale."""
    idx = np.random.default_rng(seed).permutation(len(cond))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2: 2 * (len(idx) // 2)]
    return all_metrics({s: Ec[s][b][:, None] for s in SPECIES_NAMES}, {s: Ec[s][a] for s in SPECIES_NAMES},
                       cond[b], cond[a])


def worse(m, ref):
    """Metric names where m is further from CORSIKA than ref (AUC by its distance from 0.5)."""
    d = lambda k, x: abs(x - 0.5) if k == "auc" else x
    return [f"{s}.{k}" for s, v in m.items() for k, x in v.items() if d(k, x) > d(k, ref[s][k])]


def load_group(out, prefix, key):
    """Draws of every seed of one (model, training set): {seed: {species: (n, M, planes)}}."""
    paths = sorted(glob.glob(os.path.join(out, DRAWS, prefix + "_s*.npz")))
    g = {}
    for p in paths:
        z = np.load(p)
        g[int(p.rsplit("_s", 1)[1][:-4])] = (z[f"{key}.rows"], {s: z[f"{key}.{s}.E"].astype(np.float64)
                                                                   for s in SPECIES_NAMES})
    return g


def es_all(dr, cor):
    """Energy score per primary, per species and joint."""
    y = {s: score.obs(cor[s]) for s in SPECIES_NAMES}
    sc = {s: score.energy_score(score.obs(dr[s]), y[s]) for s in SPECIES_NAMES}
    sc["joint"] = score.energy_score(np.concatenate([score.obs(dr[s]) for s in SPECIES_NAMES], 2),
                                     np.concatenate([y[s] for s in SPECIES_NAMES], 1))
    return sc


def stage_table(out, args, log):
    groups = [f"{m}_{t}" for m in ("sym", "mlp") for t in ("full", "cut", *SIZES)]
    res = {"sets": {}}
    curve = {}
    for key in SETS:
        e = eval_set(args.out, key)
        cond = np.column_stack([e["l"], e["u"]])
        R = res["sets"][key] = {"n": len(e["rows"]), "classes": {}}
        es, draws0 = {}, {}
        for gname in groups:
            g = load_group(args.out, gname, key)
            if not g:
                raise ValueError(f"no draws for {gname}")
            for seed, (rows, dr) in g.items():
                if not np.array_equal(rows, e["rows"]):
                    raise ValueError(f"{gname} seed {seed}: draws on other primaries")
            es[gname] = {seed: es_all(dr, e["cor"]) for seed, (_, dr) in g.items()}
            draws0[gname] = g[min(g)][1]
        es["chain"] = {0: es_all(e["chain"], e["cor"])}
        mean_es = {k: {s: np.mean([v[s] for v in per.values()], 0) for s in per[min(per)]} for k, per in es.items()}
        for c, cn in planes.CLASS.items():
            sel = e["lab"] == c
            C = R["classes"][cn] = {"n": int(sel.sum())}
            pr = lambda a, b: {s: score.paired(mean_es[a][s][sel], mean_es[b][s][sel]) for s in mean_es[a]}
            C["energy_score"] = {k: {s: [float(v[s][sel].mean()), float(v[s][sel].std() / np.sqrt(sel.sum()))]
                                     for s in v} for k, v in mean_es.items()}
            C["seed_spread_joint"] = {k: [float(v[s]["joint"][sel].mean()) for s in v] for k, v in es.items()}
            C["paired"] = {f"{a} minus {b}": pr(a, b) for a, b in
                           [("sym_full", "mlp_full"), ("sym_cut", "mlp_cut"), ("sym_1k", "mlp_1k"),
                            ("sym_10k", "mlp_10k"), ("sym_full", "chain"), ("mlp_full", "chain"),
                            ("sym_cut", "chain"), ("mlp_cut", "chain"), ("sym_cut", "sym_full"),
                            ("mlp_cut", "mlp_full")]}
            sources = {"chain": e["chain"], "sym_full": draws0["sym_full"], "mlp_full": draws0["mlp_full"],
                       "sym_cut": draws0["sym_cut"], "mlp_cut": draws0["mlp_cut"]}
            cor = {s: e["cor"][s][sel] for s in SPECIES_NAMES}
            C["metrics"] = {k: all_metrics({s: dr[s][sel] for s in SPECIES_NAMES}, cor, cond[sel], cond[sel])
                            for k, dr in sources.items()}
            C["metrics"]["corsika_split_half"] = split_half(cor, cond[sel])
            C["sym_full_worse_than_chain"] = worse(C["metrics"]["sym_full"], C["metrics"]["chain"])
            log.info("%s %s (n %d): sym-mlp full %s | cut %s | 1k %s | 10k %s | sym-chain %s", key, cn, sel.sum(),
                     *[np.round(C["paired"][p]["joint"], 4).tolist() for p in
                       ("sym_full minus mlp_full", "sym_cut minus mlp_cut", "sym_1k minus mlp_1k",
                        "sym_10k minus mlp_10k", "sym_full minus chain")])
            log.info("%s %s: sym_full worse than chain on %s", key, cn, C["sym_full_worse_than_chain"])
            if key == "range":
                curve[cn] = {m: [C["energy_score"][f"{m}_{t}"]["joint"] for t in (*SIZES, "full")]
                             for m in ("sym", "mlp")}
    with np.load(os.path.join(args.out, fit_profiles.TRAIN)) as z:
        n_all = int((z["meta.log10_E"] <= sampler.LOG10_E_MAX).sum())
    n_train = {cn: [SIZES["1k"], SIZES["10k"], n_all] for cn in curve}
    fig, axes = plt.subplots(1, len(curve), figsize=(7.6, 3.4))
    for ax, (cn, v) in zip(np.atleast_1d(axes), curve.items()):
        for m, lab, col in (("sym", "linear laws", "steelblue"), ("mlp", "MLP laws", "#2ca02c")):
            mu, se = np.array(v[m]).T
            ax.errorbar(n_train[cn], mu, 2 * se, color=col, marker="o", capsize=3, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("training showers (both classes)")
        ax.set_title(cn)
    np.atleast_1d(axes)[0].set_ylabel("energy score, joint")
    np.atleast_1d(axes)[0].legend()
    logs.save_fig(fig, out, "data_efficiency", log)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["draw", "table"])
    ap.add_argument("--task", type=int)
    args = ap.parse_args()
    if args.stage == "draw":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
        out, log = logs.start(os.path.join(FOLDER, "logs", f"draw_{args.task:02d}"), args)
        logs.save(out, stage_draw(out, args, log), args, log)
    else:
        out, log = logs.start(os.path.join(FOLDER, "table"), args)
        logs.save(out, stage_table(out, args, log), args, log)


if __name__ == "__main__":
    main()
