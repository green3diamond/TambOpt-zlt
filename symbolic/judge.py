"""A fair judge of the particle route: correct particles as the target instead of h5_files_v3, whose cell split the
chain learned. Target: the raw showers through the per-simulation preprocessing with the old caps (particles.py truth).
Two sources at those same primaries: the symbolic draw (draw.py, laws fitted without the judged showers) and the neural
chain (chain.py given).

Stages:
  split  the in-range showers (log10 E < chain.TOP) into two folds: within class and zenith band, alternately in log10 E
         order. Fold f's laws are fitted on the other fold plus every shower above the range; fold f is judged.
         -> particles/judge/{primaries.npz, primaries_fold{f}.npz, skip_fold{f}.txt}
  run    per species, class and zenith band, pooled and per-plane W1 to the target of suite.observables, each source
         pooled over its DRAWS draws per primary.
  clouds target and symbolic clouds in the chain's order (primary by primary, DRAWS entries each), for detector.py.
         -> particles/judge/clouds_{target,symbolic}.npz

Rule (fixed before the run): a row PASSES when the symbolic W1 is at most the chain's (suite.py's rule); it is CLEAR when
the 95% interval of the difference, resampling primaries (the same resample for both sources), excludes 0. Rows JUDGED
are judged, the rest reported only. Floor per source: W1 of one draw per primary against the source's other draws, the
score a perfect model would get against a target of this size.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from symbolic import logs
from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere, chain, planes, sources, suite

FOLDER = os.path.join("particles", "judge")
DRAW = os.path.join("particles", "draw", "judge_fold{}", "clouds.npz")
CHAIN = os.path.join(chain.FOLDER, "given", "judge.npz")
LAWS = os.path.join(FOLDER, "fold{}", "particle_laws.npz")
LABEL = {cn: c for c, cn in planes.CLASS.items()}
JUDGED = ("abs_t1", "cell_energy", "cells", "centre")
REPORTED = ("lead_time", "mean_time", "plane_energy")
SOURCES = ("symbolic", "neural chain")
DRAWS = 8
BOOT = 200
E_TOL = 1e-4


def fold_of(cls, l, u):
    """Fold (0 or 1) of each shower: alternating in log10 E order within class and zenith band."""
    fold = np.zeros(len(l), int)
    for c in np.unique(cls):
        for sel in suite.BANDS.values():
            m = np.flatnonzero((cls == c) & sel(u))
            fold[m[np.argsort(l[m], kind="stable")][::2]] = 1
    return fold


def stage_split(out, args, log):
    files = sorted(glob.glob(os.path.join(args.out, FOLDER, "tables", "*.npz")))
    t = [np.load(f) for f in files]
    names = np.array([os.path.basename(f) for f in files])
    cls = np.array([str(z["class"]) for z in t])
    l = np.array([float(z["log10E"]) for z in t])
    dirs = np.array([z["direction"] for z in t])
    if len(np.unique(names)) != len(names) or not np.isclose(np.linalg.norm(dirs, axis=1), 1).all():
        raise ValueError("duplicate tables or directions that are not unit vectors")
    keep = l < chain.TOP
    u = dirs[:, 2]
    fold = fold_of(cls[keep], l[keep], u[keep])
    prim = {"names": names[keep], "cls": cls[keep], "labels": np.array([LABEL[c] for c in cls[keep]]), "l": l[keep],
            "u": u[keep], "dirs": dirs[keep], "fold": fold,
            "X": atmosphere.plane_depths(np.degrees(np.arccos(u[keep])), N_PLANES)}
    np.savez(os.path.join(args.out, FOLDER, "primaries.npz"), rows=np.arange(keep.sum()), **prim)
    res = {"tables": len(names), "above the range": int((~keep).sum())}
    for f in (0, 1):
        m = fold == f
        np.savez(os.path.join(args.out, FOLDER, f"primaries_fold{f}.npz"), rows=np.flatnonzero(m),
                 **{k: v[m] for k, v in prim.items()})
        with open(os.path.join(args.out, FOLDER, f"skip_fold{f}.txt"), "w") as fh:
            fh.write("\n".join(prim["names"][m]) + "\n")
        res[f"fold {f}"] = {f"{c} {b}": int((m & (prim["cls"] == c) & sel(prim["u"])).sum()) for c in LABEL
                            for b, sel in suite.BANDS.items()}
        log.info("fold %d judged: %s", f, res[f"fold {f}"])
    return res


def positive(cells):
    return cells[cells[:, 3] > 0].astype(np.float64)


def read_truth(root, prim, rows):
    """Per species the target's cells at the primaries `rows`, each checked against its shower's table."""
    out = {s: [] for s in SPECIES_NAMES}
    for r in rows:
        z, t = (np.load(os.path.join(root, FOLDER, d, str(prim["names"][r]))) for d in ("truth", "tables"))
        for s in SPECIES_NAMES:
            c = positive(z[f"{s}.cells"])
            e = np.bincount(np.rint(c[:, 2]).astype(int), c[:, 3], N_PLANES)
            if not np.allclose(e, t[f"{s}.E_stored"], rtol=E_TOL, atol=1e-6):
                raise ValueError(f"{prim['names'][r]} {s}: truth cells differ from the table's stored energies")
            out[s].append(c)
    return out


def read_clouds(path):
    """rows and, per species, the cells of each padded cloud."""
    z = np.load(path)
    return z["rows"], {s: [positive(p) for p in z[f"{s}.points"]] for s in SPECIES_NAMES}


def read_symbolic(root, folds):
    rows, out = [], {s: [] for s in SPECIES_NAMES}
    for f in folds:
        r, c = read_clouds(os.path.join(root, DRAW.format(f)))
        rows.append(r)
        for s in SPECIES_NAMES:
            out[s] += c[s]
    return np.concatenate(rows), out


def read_chain(root, judged):
    rows, c = read_clouds(os.path.join(root, CHAIN))
    m = np.isin(rows, judged)
    return rows[m], {s: [p for p, k in zip(c[s], m) if k] for s in SPECIES_NAMES}


def draw_index(rows):
    """Each entry's position among the entries of the same primary."""
    seen, out = {}, np.zeros(len(rows), int)
    for j, r in enumerate(rows):
        out[j] = seen.get(r, 0)
        seen[r] = out[j] + 1
    return out


def observables(clouds, dirs, rows):
    """name -> (values, weights, primary, draw, plane) of one species' clouds; rows = the primary of each cloud."""
    cells, pl = suite.tables(clouds, dirs[rows])
    out = suite.observables(cells, pl)
    out["plane_energy"] = (np.log10(pl["E"]), np.ones(len(pl["E"])), pl["i"], pl["k"], pl["group"])
    draw = draw_index(rows)
    return {name: (v, w, rows[i], draw[i], k) for name, (v, w, i, k, _) in out.items()}


def take(o, m):
    return tuple(x[m] for x in o)


def dist(a, b, mult=None):
    """W1 between two observables, each primary weighted by mult; NaN if either side is empty."""
    wa, wb = (x[1] if mult is None else x[1] * mult[x[2]] for x in (a, b))
    if wa.sum() <= 0 or wb.sum() <= 0:
        return np.nan
    return float(stats.wasserstein_distance(a[0], b[0], wa, wb))


def floor(o):
    """Mean W1 of one draw per primary against the same source's other draws."""
    return float(np.nanmean([dist(take(o, o[3] == d), take(o, o[3] != d)) for d in np.unique(o[3])]))


def per_plane(a, b):
    """Mean over planes of the W1 on each plane, over planes with data on both sides: a pooled W1 ignores plane order."""
    return float(np.nanmean([dist(take(a, a[4] == k), take(b, b[4] == k)) for k in range(N_PLANES)]))


def median(o):
    i = np.argsort(o[0])
    c = np.cumsum(o[1][i])
    return float(o[0][i][np.searchsorted(c, 0.5 * c[-1])])


def score(target, symbolic, neural, primaries, rng, n_boot=BOOT):
    """One observable, one group: W1, floors, medians, and the resampled symbolic minus chain difference."""
    size = int(primaries.max()) + 1
    d = []
    for _ in range(n_boot):
        mult = np.zeros(size)
        np.add.at(mult, rng.choice(primaries, len(primaries)), 1.0)
        d.append(dist(target, symbolic, mult) - dist(target, neural, mult))
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return {"symbolic": dist(target, symbolic), "neural chain": dist(target, neural), "difference 95%": [float(lo), float(hi)],
            "symbolic per plane": per_plane(target, symbolic), "neural chain per plane": per_plane(target, neural),
            "floor symbolic": floor(symbolic), "floor neural chain": floor(neural), "median target": median(target),
            "median symbolic": median(symbolic), "median neural chain": median(neural)}


def verdict(r, name):
    if name not in JUDGED:
        return "reported"
    lo, hi = r["difference 95%"]
    return ("PASS" if r["symbolic"] <= r["neural chain"] else "FAIL") + (" clear" if lo > 0 or hi < 0 else "")


_DATA = {}


def one_group(key):
    s, cn, band, seed, n_boot = key
    prim = _DATA["prim"]
    primaries = np.flatnonzero((prim["cls"] == cn) & suite.BANDS[band](prim["u"]) & _DATA["judged"])
    keep = lambda o: take(o, np.isin(o[2], primaries))
    rng = np.random.default_rng(seed)
    res = {"primaries": len(primaries)}
    for name in (*JUDGED, *REPORTED):
        o = [keep(_DATA[src][s][name]) for src in ("target", *SOURCES)]
        res[name] = score(*o, primaries, rng, n_boot)
        res[name]["verdict"] = verdict(res[name], name)
    return s, cn, band, res


def fig_rows(res):
    """W1 to the target per row and group, each source's floor as a small dash."""
    names = (*JUDGED, *REPORTED[:2])
    keys = [(s, cn, b) for cn in LABEL for s in SPECIES_NAMES for b in suite.BANDS]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4), sharex=True)
    x = np.arange(len(keys))
    for ax, name in zip(axes.flat, names):
        for o, src in enumerate(SOURCES):
            pos = x + 0.2 * (o - 0.5)
            ax.scatter(pos, [res[s][cn][b][name][src] for s, cn, b in keys], color=suite.COLORS[src], s=14, label=src)
            ax.scatter(pos, [res[s][cn][b][name][f"floor {src}"] for s, cn, b in keys], color=suite.COLORS[src], s=20,
                       marker="_")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s[:2]} {cn[:3]} {b[:4]}" for s, cn, b in keys], rotation=90, fontsize=6)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_ylabel("W1 to the target", fontsize=8)
    axes[1, 0].set_ylabel("W1 to the target", fontsize=8)
    fig.tight_layout()
    return fig


def judged_laws_apart(root, prim, folds):
    """Fold f's laws must not have seen fold f."""
    for f in folds:
        with open(os.path.join(root, os.path.dirname(LAWS.format(f)), "results.json")) as fh:
            seen = set(json.load(fh)["showers"])
        if seen & set(prim["names"][prim["fold"] == f]):
            raise ValueError(f"fold {f}: the laws were fitted on judged showers")


def stage_run(out, args, log):
    z = np.load(os.path.join(args.out, FOLDER, "primaries.npz"))
    prim = {k: z[k] for k in z.files}
    judged_laws_apart(args.out, prim, args.folds)
    judged = np.flatnonzero(np.isin(prim["fold"], args.folds))
    log.info("%d judged primaries, folds %s", len(judged), args.folds)
    sym_rows, sym = read_symbolic(args.out, args.folds)
    neu_rows, neu = read_chain(args.out, judged)
    for src, r in (("symbolic", sym_rows), ("neural chain", neu_rows)):
        if not np.array_equal(np.unique(r, return_counts=True)[0], judged) or (np.bincount(r)[judged] != args.draws).any():
            raise ValueError(f"{src}: expected {args.draws} draws at every judged primary")
    target = read_truth(args.out, prim, judged)
    log.info("sources read, truth checked against the tables")
    _DATA.update(prim=prim, judged=np.isin(np.arange(len(prim["fold"])), judged))
    for src, c, r in (("target", target, judged), ("symbolic", sym, sym_rows), ("neural chain", neu, neu_rows)):
        _DATA[src] = {s: observables(c[s], prim["dirs"], r) for s in SPECIES_NAMES}
    del target, sym, neu
    keys = [(s, cn, b, args.seed + j, args.boot) for j, (s, cn, b) in enumerate(
        (s, cn, b) for s in SPECIES_NAMES for cn in LABEL for b in suite.BANDS)]
    res = {s: {cn: {} for cn in LABEL} for s in SPECIES_NAMES}
    with Pool(args.workers) as pool:
        for s, cn, band, r in pool.imap_unordered(one_group, keys):
            res[s][cn][band] = r
            log.info("%s %s %s (%d primaries) | %s", s, cn, band, r["primaries"], " | ".join(
                f"{n} {r[n]['symbolic']:.3g} vs {r[n]['neural chain']:.3g} (floor {r[n]['floor symbolic']:.2g} / "
                f"{r[n]['floor neural chain']:.2g}) {r[n]['verdict']}" for n in (*JUDGED, *REPORTED)))
    logs.save_fig(fig_rows(res), out, "w1_rows", log)
    return {"rule": __doc__.split("Rule (fixed before the run): ")[1].strip(), "folds": args.folds,
            "judged": int(len(judged)), "results": res}


def stage_clouds(out, args, log):
    z = np.load(os.path.join(args.out, FOLDER, "primaries.npz"))
    prim = {k: z[k] for k in z.files}
    rows = np.repeat(prim["rows"], args.draws)
    if not np.array_equal(np.load(os.path.join(args.out, CHAIN))["rows"], rows):
        raise ValueError("the chain's clouds are not in the expected order")
    sym_rows, sym = read_symbolic(args.out, (0, 1))
    order = np.argsort(sym_rows, kind="stable")
    if not np.array_equal(sym_rows[order], rows):
        raise ValueError("the symbolic draws do not cover every primary DRAWS times")
    target = read_truth(args.out, prim, prim["rows"])
    for name, per in (("target", {s: [target[s][r] for r in rows] for s in SPECIES_NAMES}),
                      ("symbolic", {s: [sym[s][j] for j in order] for s in SPECIES_NAMES})):
        pad = {f"{s}.points": np.stack([np.pad(c, ((0, sources.CAP[s] - len(c)), (0, 0))) for c in per[s]]).astype(np.float32)
               for s in SPECIES_NAMES}
        np.savez(os.path.join(args.out, FOLDER, f"clouds_{name}.npz"), rows=rows, **pad)
        log.info("clouds_%s: %d entries", name, len(rows))
    return {"entries": int(len(rows))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("split", "run", "clouds"))
    ap.add_argument("--out", required=True, help="the symbolic_trim root")
    ap.add_argument("--name", default="run", help="output folder under particles/judge/")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1], help="run stage: the folds judged")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, args.name if args.stage == "run" else args.stage), args)
    logs.save(out, {"split": stage_split, "run": stage_run, "clouds": stage_clouds}[args.stage](out, args, log), args, log)


if __name__ == "__main__":
    main()
