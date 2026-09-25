"""Reconstruction resolution in physical units, kernel against surrogate inputs, for each layout."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from analysis_core import events, logs, plots, stats
from analysis_core.run import LAYOUTS, SCHEMES, Run, add_run_args
from analysis_core.scoring import LABELS, Scorer
from modules.constants import LOG_E_MAX, LOG_E_MIN

SPAN = LOG_E_MAX - LOG_E_MIN


def angular_deg(pred, true):
    p = pred / np.linalg.norm(pred, axis=1, keepdims=True).clip(1e-12)
    t = true / np.linalg.norm(true, axis=1, keepdims=True).clip(1e-12)
    return np.degrees(np.arccos(np.clip((p * t).sum(1), -1.0, 1.0)))


def errors(pred, primary):
    """Angular error in degrees and absolute log10 energy error in dex."""
    return angular_deg(pred[:, :3], primary[:, :3]), np.abs(pred[:, 3] - primary[:, 3]) * SPAN


@torch.no_grad()
def reconstruct(run, scorer, e, n, chunk):
    labels = scorer.labels(e, n)
    xy = torch.stack([e, n], -1).to(run.device)
    return torch.cat([run.recon(torch.cat([xy.expand(len(lab), -1, -1), lab], -1))
                      for lab in labels.split(chunk)]).cpu().numpy()


def summarize(x, n_boot):
    q50, q68, q95 = np.percentile(x, [50, 68, 95])
    return {"median": q50, "q68": q68, "q95": q95, "median_ci": stats.bootstrap_ci(x, n_boot=n_boot)}


def by_energy(x, log_e, edges, n_boot):
    idx = stats.bin_index(log_e, edges)
    rows = [summarize(x[idx == b], n_boot) if (idx == b).sum() > 1 else None for b in range(len(edges) - 1)]
    return {"centres": ((edges[:-1] + edges[1:]) / 2).tolist(), "counts": np.bincount(idx, minlength=len(edges) - 1).tolist(),
            "median": [r and r["median"] for r in rows], "median_ci": [r and r["median_ci"] for r in rows]}


def cdf_figure(errs, prior):
    fig, axes = plots.plt.subplots(1, 2, figsize=(11, 4))
    for ax, k, label in ((axes[0], 0, "angular error [deg]"), (axes[1], 1, "|Δlog10 E| [dex]")):
        for i, name in enumerate(errs):
            for src, ls in zip(LABELS, ("--", "-")):
                x = np.sort(errs[name][src][k])
                ax.plot(x, np.linspace(0, 1, len(x)), ls, color=f"C{i}", label=f"{name} {src}")
        x = np.sort(prior[k])
        ax.plot(x, np.linspace(0, 1, len(x)), ":", color="0.4", label="predict the mean")
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("fraction of events")
    axes[0].legend(fontsize=7)
    return fig


def energy_figure(res):
    fig, axes = plots.plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in ((axes[0], "angle_by_energy", "median angular error [deg]"),
                           (axes[1], "energy_by_energy", "median |Δlog10 E| [dex]")):
        for i, name in enumerate(n for n in res if n != "prior" and n != "paired"):
            for src, ls in zip(LABELS, ("--", "-")):
                b = res[name][src][key]
                ok = [m is not None for m in b["median"]]
                x = np.array(b["centres"])[ok]
                y = np.array([m for m in b["median"] if m is not None])
                ci = np.array([c for c in b["median_ci"] if c is not None])
                ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="o" + ls, color=f"C{i}", ms=3,
                            label=f"{name} {src}")
        ax.set_xlabel("true log10(E / GeV)")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=7)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_run_args(ap)
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS), choices=LAYOUTS)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--energy-bins", type=int, default=6)
    ap.add_argument("--save-errors", action="store_true",
                    help="also write the per-event errors to errors.npz")
    args = ap.parse_args()

    out, log = logs.start("recon_resolution", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Run(args.run_dir, device)
    ev = events.load(run, args.n_events, args.seed, args.chunk)
    prim = ev.primary.cpu().numpy()
    log_e = prim[:, 3] * SPAN + LOG_E_MIN
    edges = np.linspace(LOG_E_MIN, LOG_E_MAX, args.energy_bins + 1)
    scorers = {src: Scorer(run, ev, src, args.chunk, args.kernel_chunk) for src in LABELS}

    d = prim[:, :3].mean(0)
    prior = errors(np.tile(np.r_[d / np.linalg.norm(d), prim[:, 3].mean()], (len(prim), 1)), prim)
    res = {"prior": {"angle_deg": summarize(prior[0], args.n_boot), "energy_dex": summarize(prior[1], args.n_boot)}}
    log.info("events %d  prior median angle %.2f deg  energy %.3f dex", len(ev), *(np.median(p) for p in prior))

    errs = {}
    for name in args.layouts:
        e, n = run.layout(name)
        errs[name], res[name] = {}, {}
        for src, sc in scorers.items():
            a, dex = errors(reconstruct(run, sc, e, n, args.chunk), prim)
            errs[name][src] = (a, dex)
            res[name][src] = {"angle_deg": summarize(a, args.n_boot), "energy_dex": summarize(dex, args.n_boot),
                              "angle_by_energy": by_energy(a, log_e, edges, args.n_boot),
                              "energy_by_energy": by_energy(dex, log_e, edges, args.n_boot)}
            log.info("%-10s %-9s angle median %.2f (68%% %.2f, 95%% %.2f) deg  energy median %.3f dex",
                     name, src, np.median(a), np.percentile(a, 68), np.percentile(a, 95), np.median(dex))

    res["paired"] = {}
    for scheme in SCHEMES:
        if scheme in errs and f"opt_{scheme}" in errs:
            for k, unit in ((0, "angle_deg"), (1, "energy_dex")):
                opt, base = errs[f"opt_{scheme}"]["kernel"][k], errs[scheme]["kernel"][k]
                diff = float(np.median(opt) - np.median(base))
                ci = stats.paired_bootstrap_ci(opt, base, n_boot=args.n_boot)
                res["paired"][f"kernel/opt_{scheme}-{scheme}/{unit}"] = {"median_diff": diff, "ci": ci}
                log.info("kernel opt_%s - %s median %s %+.3f  [%+.3f, %+.3f]", scheme, scheme, unit, diff, *ci)

    if args.save_errors:
        path = os.path.join(out, "errors.npz")
        np.savez_compressed(path, prior_angle_deg=prior[0], prior_energy_dex=prior[1],
                            **{f"{name}.{src}.{unit}": errs[name][src][k]
                               for name in errs for src in errs[name]
                               for k, unit in ((0, "angle_deg"), (1, "energy_dex"))})
        log.info("wrote %s (%d bytes)", path, os.path.getsize(path))

    plots.save(cdf_figure(errs, prior), out, "error_cdf", log)
    plots.save(energy_figure(res), out, "error_vs_energy", log)
    logs.save(out, {"n_events": len(ev), "results": res}, args, log)


if __name__ == "__main__":
    main()
