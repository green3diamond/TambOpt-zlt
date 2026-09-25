"""Check the shared scorer against the run's own true-utility evaluation, and time it."""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from analysis_core import events, logs, plots
from analysis_core.run import ENSEMBLE_DIR, LAYOUTS, SCHEMES, Run, add_run_args
from analysis_core.scoring import LABELS, FixedLabels, Scorer, paired
from modules.optimize import utility_of_xy


def reference(run_dir, scheme):
    """Event count, baseline true-U and optimized true-U from true_utility_eval.txt."""
    text = open(os.path.join(run_dir, ENSEMBLE_DIR.format(scheme), "true_utility_eval.txt")).read()
    rows = dict((k, float(v)) for k, _, v in
                re.findall(r"^\s+(baseline grid|optimized)\s+(\S+)\s+(\S+)\s*$", text, re.M))
    n_events = int(re.search(r"^events\s*:\s*(\d+)", text, re.M).group(1))
    return n_events, rows["baseline grid"], rows["optimized"]


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return result, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_run_args(ap)
    ap.add_argument("--tol", type=float, default=1e-3, help="relative tolerance for reproduction")
    args = ap.parse_args()

    out, log = logs.start("scoring_check", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run(args.run_dir, device)
    ev = events.load(run, args.n_events, args.seed, args.chunk)
    log.info("events %d  chunk %d  device %s", len(ev), args.chunk, device)

    layouts = {name: run.layout(name) for name in LAYOUTS}
    scorers = {src: Scorer(run, ev, src, args.chunk, args.kernel_chunk) for src in LABELS}
    scores, seconds, checks = {src: {} for src in LABELS}, {}, {}
    for src, sc in scorers.items():
        for name, (e, n) in layouts.items():
            s, seconds[f"{src}/{name}"] = timed(lambda: sc.score(e, n), device)
            scores[src][name] = s
            log.info("%-9s %-10s U %9.4f +- %.4f  (%.1f s)", src, name, s.U, s.se, seconds[f"{src}/{name}"])

    for scheme in SCHEMES:
        n_ref, base, opt = reference(args.run_dir, scheme)
        for name, want in ((scheme, base), (f"opt_{scheme}", opt)):
            got = scores["kernel"][name].U
            ok = len(ev) == n_ref and abs(got - want) <= args.tol * max(1.0, abs(want))
            checks[f"reproduces_true_u_{name}"] = ok
            log.info("reproduce %-10s got %.4f want %.4f  (%d vs %d events)  %s",
                     name, got, want, len(ev), n_ref, "ok" if ok else "FAIL")

    e, n = (t.to(device) for t in layouts["opt_center"])
    for src, sc in scorers.items():
        with torch.no_grad():
            full = utility_of_xy(e, n, ev.primary, FixedLabels(sc.labels(e, n)), run.recon,
                                 sample=False)[0].item()
        checks[f"chunks_equal_full_batch_{src}"] = abs(full - scores[src]["opt_center"].U) <= 1e-5 * abs(full)

    again = scorers["surrogate"].score(e, n)
    checks["surrogate_deterministic"] = bool(np.array_equal(again.chunks, scores["surrogate"]["opt_center"].chunks))
    perm = torch.randperm(len(e), generator=torch.Generator().manual_seed(1)).to(device)
    shuffled = scorers["kernel"].score(e[perm], n[perm]).U
    checks["detector_order_invariant"] = abs(shuffled - scores["kernel"]["opt_center"].U) <= 1e-4 * abs(shuffled)

    diffs = {}
    for src in LABELS:
        for scheme in SCHEMES:
            d, se = paired(scores[src][f"opt_{scheme}"], scores[src][scheme])
            diffs[f"{src}/opt_{scheme}-{scheme}"] = {"diff": d, "se": se}
            log.info("%-9s opt_%s - %s  %+9.4f +- %.4f", src, scheme, scheme, d, se)
    for name, ok in checks.items():
        log.info("check %-36s %s", name, "ok" if ok else "FAIL")

    plots.save(layout_figure(scores), out, "utility_by_layout", log)
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    logs.save(out, {"checks": checks, "paired": diffs, "seconds": seconds, "peak_gpu_gib": peak,
                    "scores": {src: {k: v.summary() for k, v in s.items()} for src, s in scores.items()}},
              args, log)
    sys.exit(0 if all(checks.values()) else 1)


def layout_figure(scores):
    """U with its standard error for every layout, surrogate beside kernel."""
    fig, ax = plots.plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(LAYOUTS))
    for i, (src, color) in enumerate(((LABELS[0], plots.SURROGATE), (LABELS[1], plots.KERNEL))):
        s = [scores[src][name] for name in LAYOUTS]
        ax.errorbar(x + (i - 0.5) * 0.2, [v.U for v in s], yerr=[v.se for v in s],
                    fmt="o", color=color, label=src)
    ax.set_xticks(x, LAYOUTS)
    ax.set_ylabel("U")
    ax.legend()
    return fig


if __name__ == "__main__":
    main()
