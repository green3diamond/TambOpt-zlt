"""Which detectors light up, kernel against surrogate, and detection efficiency against energy and decay distance."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from analysis_core import events, logs, plots, stats
from analysis_core.run import LAYOUTS, Run, add_run_args
from analysis_core.scoring import LABELS, Scorer
from modules.constants import LOG_E_MAX, LOG_E_MIN
from modules.optimize import LAYOUT_THRESHOLD

MIN_LIT = (1, 5, 10, 20)


@torch.no_grad()
def lit(scorer, e, n):
    """(B, n_det) detectors whose counts pass the layout threshold."""
    return (torch.expm1(scorer.labels(e, n)[..., 0]).clamp(min=0) > LAYOUT_THRESHOLD).cpu().numpy()


def confusion(lit_kernel, lit_surrogate):
    both = int((lit_kernel & lit_surrogate).sum())
    kernel_only = int((lit_kernel & ~lit_surrogate).sum())
    surrogate_only = int((~lit_kernel & lit_surrogate).sum())
    return {"both": both, "kernel_only": kernel_only, "surrogate_only": surrogate_only,
            "neither": int((~lit_kernel & ~lit_surrogate).sum()),
            "precision": both / max(both + surrogate_only, 1), "recall": both / max(both + kernel_only, 1)}


def efficiency(n_lit, x, edges):
    """Fraction of events with at least m lit detectors per bin, with Wilson intervals."""
    idx = stats.bin_index(x, edges)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    out = {"centres": ((edges[:-1] + edges[1:]) / 2).tolist(), "counts": counts.tolist()}
    for m in MIN_LIT:
        k = np.bincount(idx, weights=(n_lit >= m), minlength=len(edges) - 1)
        lo, hi = stats.wilson(k, counts)
        out[str(m)] = {"frac": (k / np.maximum(counts, 1)).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    return out


def lit_figure(res, n_det):
    names = list(res)
    fig, axes = plots.plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), sharey=True)
    bins = np.arange(n_det + 2) - 0.5
    for ax, name in zip(np.atleast_1d(axes), names):
        for src, color in zip(LABELS, (plots.SURROGATE, plots.KERNEL)):
            ax.hist(res[name]["n_lit"][src], bins=bins, histtype="step", color=color, label=src)
        ax.set_title(name)
        ax.set_xlabel("lit detectors per event")
    np.atleast_1d(axes)[0].set_ylabel("events")
    np.atleast_1d(axes)[0].legend()
    return fig


def efficiency_figure(res, m):
    fig, axes = plots.plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, label in ((axes[0], "vs_energy", "true log10(E / GeV)"),
                           (axes[1], "vs_distance", "decay vertex distance from array centre [m]")):
        for i, name in enumerate(res):
            for src, ls in zip(LABELS, ("--", "-")):
                b = res[name][src][key]
                f = b[str(m)]
                ax.plot(b["centres"], f["frac"], ls + "o", ms=3, color=f"C{i}", label=f"{name} {src}")
                ax.fill_between(b["centres"], f["lo"], f["hi"], color=f"C{i}", alpha=0.15)
        ax.set_xlabel(label)
    axes[0].set_ylabel(f"fraction with at least {m} lit detectors")
    axes[0].legend(fontsize=7)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_run_args(ap)
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS), choices=LAYOUTS)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--plot-min-lit", type=int, default=10, choices=MIN_LIT)
    args = ap.parse_args()

    out, log = logs.start("detection_stats", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run(args.run_dir, device)
    ev = events.load(run, args.n_events, args.seed, args.chunk)
    prim = ev.primary.cpu().numpy()
    log_e = prim[:, 3] * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    distance = np.hypot(prim[:, 5], prim[:, 6])
    e_edges = np.linspace(LOG_E_MIN, LOG_E_MAX, args.bins + 1)
    d_edges = np.quantile(distance, np.linspace(0, 1, args.bins + 1))
    scorers = {src: Scorer(run, ev, src, args.chunk, args.kernel_chunk) for src in LABELS}
    log.info("events %d  layout threshold %g counts", len(ev), LAYOUT_THRESHOLD)

    res = {}
    for name in args.layouts:
        e, n = run.layout(name)
        lits = {src: lit(sc, e, n) for src, sc in scorers.items()}
        n_lit = {src: v.sum(1) for src, v in lits.items()}
        excess = n_lit["surrogate"] - n_lit["kernel"]
        res[name] = {"confusion": confusion(lits["kernel"], lits["surrogate"]),
                     "n_lit": {src: v.tolist() for src, v in n_lit.items()},
                     "excess_median_p10_p90": np.percentile(excess, [50, 10, 90]).tolist()}
        for src in LABELS:
            res[name][src] = {"vs_energy": efficiency(n_lit[src], log_e, e_edges),
                              "vs_distance": efficiency(n_lit[src], distance, d_edges),
                              "overall": {str(m): float((n_lit[src] >= m).mean()) for m in MIN_LIT}}
        c = res[name]["confusion"]
        log.info("%-10s lit per event kernel %.1f surrogate %.1f  precision %.3f recall %.3f  excess p50 %+.0f",
                 name, n_lit["kernel"].mean(), n_lit["surrogate"].mean(), c["precision"], c["recall"],
                 res[name]["excess_median_p10_p90"][0])
        log.info("%-10s kernel fraction with at least %s lit: %s", name, MIN_LIT,
                 " ".join(f"{v:.3f}" for v in res[name]["kernel"]["overall"].values()))

    plots.save(lit_figure(res, len(run.layout(args.layouts[0])[0])), out, "lit_per_event", log)
    plots.save(efficiency_figure(res, args.plot_min_lit), out, "efficiency", log)
    logs.save(out, {"n_events": len(ev), "threshold": LAYOUT_THRESHOLD, "distance_edges": d_edges.tolist(),
                    "results": res}, args, log)


if __name__ == "__main__":
    main()
