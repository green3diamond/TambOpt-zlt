"""Utility as detectors are removed: greedy most-important first, greedy least-important first, random."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import modules.optimize.objective as objective
from analysis_core import events, logs, plots
from analysis_core.run import LAYOUTS, Run, add_run_args
from analysis_core.scoring import Scorer

BASE_THRESHOLD = objective.RECONSTRUCT_THRESHOLD


class Evaluator:
    """Scores a subset of a layout's detectors, optionally rescaling the reconstruction threshold."""

    def __init__(self, e, n, rescale):
        self.e, self.n, self.rescale = e, n, rescale

    def __call__(self, scorer, keep):
        if self.rescale:
            objective.RECONSTRUCT_THRESHOLD = BASE_THRESHOLD * len(keep) / len(self.e)
        idx = torch.as_tensor(keep)
        return scorer.score(self.e[idx], self.n[idx])


def greedy_order(evaluate, scorer, floor, most_important, log):
    """Removal order picked on `scorer`'s events; also returns the picked U at each step."""
    keep, order, picked = list(range(len(evaluate.e))), [], []
    while len(keep) > floor:
        U = [evaluate(scorer, [j for j in keep if j != c]).U for c in keep]
        k = int(np.argmin(U)) if most_important else int(np.argmax(U))
        order.append(keep.pop(k))
        picked.append(U[k])
        log.info("%s first: removed %3d, %3d left, U(selection) %.4f",
                 "most" if most_important else "least", order[-1], len(keep), U[k])
    return order, picked


def curve(evaluate, scorer, order, floor, every=1):
    """U and standard error after each removal in `order` (every `every` steps), from the full layout."""
    keep = list(range(len(evaluate.e)))
    n, U, se = [], [], []
    for step in range(len(keep) - floor + 1):
        if step % every == 0:
            s = evaluate(scorer, keep)
            n.append(len(keep))
            U.append(s.U)
            se.append(s.se)
        if step < len(order):
            keep.remove(order[step])
    return {"n": n, "U": U, "se": se}


def figure(res, floor):
    """Absolute U on the left, U relative to the full layout on the right; lines surrogate, squares kernel."""
    fig, axes = plots.plt.subplots(1, 2, figsize=(13, 4.5))
    curves = (("most_important_first", "C3"), ("least_important_first", "C2"), ("random", "0.3"))
    for ax, relative in zip(axes, (False, True)):
        for name, color in curves:
            r = res[name]
            c = r["report"] if name != "random" else {"n": r["n"], "U": r["mean"], "se": r["sd"]}
            k = r["kernel"]
            s, sk = (c["U"][0], k["U"][0]) if relative else (1.0, 1.0)
            U, se = np.array(c["U"]) / s, np.array(c["se"]) / s
            ax.plot(c["n"], U, color=color, label=name.replace("_", " "))
            ax.fill_between(c["n"], U - se, U + se, color=color, alpha=0.25)
            ax.errorbar(k["n"], np.array(k["U"]) / sk, yerr=np.array(k["se"]) / sk, fmt="s", ms=4, color=color, mfc="none")
        ax.set_xlim(max(c["n"]), floor)
        ax.set_xlabel("detectors remaining")
    axes[0].set_ylabel("U on the reporting events")
    axes[1].set_ylabel("U relative to the full layout")
    axes[0].set_title("lines: surrogate, squares: kernel", fontsize=9)
    axes[1].set_title("bands: SE (random: sd over orders)", fontsize=9)
    axes[0].legend()
    return fig


def selection_figure(res):
    fig, ax = plots.plt.subplots(figsize=(7, 4))
    for name, color in (("most_important_first", "C3"), ("least_important_first", "C2")):
        c = res[name]
        ax.plot(c["report"]["n"][1:], c["selection_U"], "--", color=color, label=f"{name} selection events")
        ax.plot(c["report"]["n"][1:], c["report"]["U"][1:], color=color, label=f"{name} reporting events")
    ax.invert_xaxis()
    ax.set_xlabel("detectors remaining")
    ax.set_ylabel("U")
    ax.legend(fontsize=8)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_run_args(ap)
    ap.add_argument("--layout", default="opt_grid", choices=LAYOUTS)
    ap.add_argument("--floor", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--kernel-every", type=int, default=10)
    ap.add_argument("--kernel-random-orders", type=int, default=3)
    ap.add_argument("--rescale-threshold", action="store_true")
    args = ap.parse_args()

    name = f"removal_curves_{args.layout}" + ("_rescaled" if args.rescale_threshold else "")
    out, log = logs.start(name, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run(args.run_dir, device)
    ev = events.load(run, args.n_events, args.seed, 2 * args.chunk)
    select_ev, report_ev = ev.part(0, 2), ev.part(1, 2)
    select = Scorer(run, select_ev, "surrogate", args.chunk)
    report = Scorer(run, report_ev, "surrogate", args.chunk)
    kernel = Scorer(run, report_ev, "kernel", args.chunk, args.kernel_chunk)
    log.info("layout %s  selection events %d  reporting events %d", args.layout, len(select_ev), len(report_ev))

    evaluate = Evaluator(*run.layout(args.layout), args.rescale_threshold)
    res = {}
    for key, most in (("most_important_first", True), ("least_important_first", False)):
        order, picked = greedy_order(evaluate, select, args.floor, most, log)
        res[key] = {"order": order, "selection_U": picked,
                    "report": curve(evaluate, report, order, args.floor),
                    "kernel": curve(evaluate, kernel, order, args.floor, args.kernel_every)}

    g = torch.Generator().manual_seed(args.seed)
    orders = [torch.randperm(len(evaluate.e), generator=g).tolist() for _ in range(args.n_random)]
    rand = [curve(evaluate, report, o, args.floor) for o in orders]
    k_rand = [curve(evaluate, kernel, o, args.floor, args.kernel_every) for o in orders[: args.kernel_random_orders]]
    U = np.array([c["U"] for c in rand])
    kU = np.array([c["U"] for c in k_rand])
    res["random"] = {"orders": orders, "n": rand[0]["n"], "U": U.tolist(),
                     "mean": U.mean(0).tolist(), "sd": U.std(0, ddof=1).tolist(),
                     "kernel": {"n": k_rand[0]["n"], "U": kU.mean(0).tolist(),
                                "se": (kU.std(0, ddof=1) / np.sqrt(len(kU))).tolist()}}
    log.info("random mean U at %d detectors: %.4f (sd %.4f over %d orders)",
             args.floor, res["random"]["mean"][-1], res["random"]["sd"][-1], len(orders))
    for key in ("most_important_first", "least_important_first"):
        c = res[key]["report"]
        log.info("%s: U at %d detectors %.4f +- %.4f (full %.4f)", key, args.floor, c["U"][-1], c["se"][-1], c["U"][0])

    plots.save(figure(res, args.floor), out, "removal_curves", log)
    plots.save(selection_figure(res), out, "selection_vs_reporting", log)
    logs.save(out, {"layout": args.layout, "floor": args.floor, "curves": res}, args, log)


if __name__ == "__main__":
    main()
