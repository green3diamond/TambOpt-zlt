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
  primaries  a held-out list instead of folds (the 150k runs): the listed summary tables -> <folder>/primaries.npz,
         one fold (0), with each shower's simulation directory. Then run with --truth h5: the target is the shower's
         own stored cells in its h5 files (the v2 / 150k per-simulation preprocessing, caps sources.CAP_NEW; render
         reproduces them, render closure 10-31), the laws' fit list (--fit) must not contain a judged shower, and the
         symbolic draw uses draw.py --caps new. --chain none scores the symbolic draw alone (no neural chain trained
         on these caps yet): its W1, floors and medians, verdict "reported". --compare <draw folder> puts a second
         symbolic draw (other laws, same primaries and draws) where the chain would be: scored and judged the same way,
         under the name "comparison"; --compare-fit refuses its laws if they saw a judged shower.

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
CHAIN = os.path.join(chain.FOLDER, "given", "judge.npz")
LAWS = os.path.join(FOLDER, "fold{}", "particle_laws.npz")
LABEL = {cn: c for c, cn in planes.CLASS.items()}
JUDGED = ("abs_t1", "cell_energy", "cells", "centre")
REPORTED = ("lead_time", "mean_time", "plane_energy")
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
    files = sorted(glob.glob(os.path.join(args.out, args.folder, "tables", "*.npz")))
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
    np.savez(os.path.join(args.out, args.folder, "primaries.npz"), rows=np.arange(keep.sum()), **prim)
    res = {"tables": len(names), "above the range": int((~keep).sum())}
    for f in (0, 1):
        m = fold == f
        np.savez(os.path.join(args.out, args.folder, f"primaries_fold{f}.npz"), rows=np.flatnonzero(m),
                 **{k: v[m] for k, v in prim.items()})
        with open(os.path.join(args.out, args.folder, f"skip_fold{f}.txt"), "w") as fh:
            fh.write("\n".join(prim["names"][m]) + "\n")
        res[f"fold {f}"] = {f"{c} {b}": int((m & (prim["cls"] == c) & sel(prim["u"])).sum()) for c in LABEL
                            for b, sel in suite.BANDS.items()}
        log.info("fold %d judged: %s", f, res[f"fold {f}"])
    return res


def stage_primaries(out, args, log):
    """The held-out showers named in --list (summary table names under --tables), one fold, in log10 E range."""
    names = sorted(set(open(args.list).read().split()))
    t = [np.load(os.path.join(args.tables, n)) for n in names]
    names = np.array(names)
    cls = np.array([str(z["class"]) for z in t])
    l = np.array([float(z["log10E"]) for z in t])
    dirs = np.array([z["direction"] for z in t])
    if not np.isclose(np.linalg.norm(dirs, axis=1), 1).all():
        raise ValueError("directions that are not unit vectors")
    keep = l < args.top
    u = dirs[keep, 2]
    prim = {"names": names[keep], "sims": np.array([str(z["sim"]) for z in t])[keep], "cls": cls[keep],
            "labels": np.array([LABEL[c] for c in cls[keep]]), "l": l[keep], "u": u, "dirs": dirs[keep],
            "fold": np.zeros(keep.sum(), int), "X": atmosphere.plane_depths(np.degrees(np.arccos(u)), N_PLANES)}
    os.makedirs(os.path.join(args.out, args.folder), exist_ok=True)
    np.savez(os.path.join(args.out, args.folder, "primaries.npz"), rows=np.arange(keep.sum()), **prim)
    res = {"listed": len(names), "above the range": int((~keep).sum()),
           "judged": {f"{c} {b}": int(((prim["cls"] == c) & sel(u)).sum()) for c in LABEL for b, sel in suite.BANDS.items()}}
    log.info("%s", res)
    return res


def read_truth_h5(prim, rows):
    """Per species the stored cells of each judged shower from its own h5 file, checked against the file: as many
    cells as num_points, at or below the cap, per-plane energy at or below plane_stats' energy before the cut, the
    direction as in the primaries."""
    import h5py
    out = {s: [] for s in SPECIES_NAMES}
    for r in rows:
        for s in SPECIES_NAMES:
            path = glob.glob(os.path.join(str(prim["sims"][r]), f"*_{sources.CORSIKA_NAME[s]}.h5"))
            if len(path) != 1:
                raise ValueError(f"{prim['sims'][r]}: {len(path)} {s} h5 files")
            with h5py.File(path[0], "r") as f:
                n, cap = int(f["num_points"][0]), int(f.attrs["nmax"])
                c = np.asarray(f["showers"][0], np.float64).reshape(n, -1)
                total = f["plane_stats"]["energy"][:].astype(np.float64)
                d = f["directions"][0].astype(np.float64)
            if cap != sources.CAP_NEW[s] or n > cap:
                raise ValueError(f"{prim['names'][r]} {s}: cap {cap} (expected {sources.CAP_NEW[s]}), {n} cells")
            if not np.allclose(d / np.linalg.norm(d), prim["dirs"][r], atol=1e-6):
                raise ValueError(f"{prim['names'][r]} {s}: direction differs from the primaries")
            c = positive(c)
            e = np.bincount(np.rint(c[:, 2]).astype(int), c[:, 3], N_PLANES)
            if (e > total * (1 + E_TOL) + 1e-6).any():
                raise ValueError(f"{prim['names'][r]} {s}: stored energy above the energy before the cut")
            out[s].append(c)
    return out


def positive(cells):
    return cells[cells[:, 3] > 0].astype(np.float64)


def read_truth(root, prim, rows, folder=FOLDER):
    """Per species the target's cells at the primaries `rows`, each checked against its shower's table."""
    out = {s: [] for s in SPECIES_NAMES}
    for r in rows:
        z, t = (np.load(os.path.join(root, folder, d, str(prim["names"][r]))) for d in ("truth", "tables"))
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


def read_symbolic(root, folds, source="judge_fold{}"):
    rows, out = [], {s: [] for s in SPECIES_NAMES}
    for f in folds:
        r, c = read_clouds(os.path.join(root, "particles", "draw", source.format(f), "clouds.npz"))
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


def primary_hists(o, primaries, lo, dx, nb):
    """Weighted histogram (nb bins of width dx from lo) of one observable per primary, rows in the order of primaries."""
    i = np.clip(np.searchsorted(primaries, o[2]), 0, len(primaries) - 1)
    ok = (primaries[i] == o[2]) & np.isfinite(o[0])
    b = np.clip(((o[0][ok] - lo) / dx).astype(np.int64), 0, nb - 1)
    return np.bincount(i[ok] * nb + b, o[1][ok], len(primaries) * nb).reshape(len(primaries), nb)


def hist_w1(ha, hb, dx):
    """W1 between two histograms on the same bins (mass at the bin centres); NaN if either is empty."""
    sa, sb = ha.sum(), hb.sum()
    if sa <= 0 or sb <= 0:
        return np.nan
    return float(np.abs(np.cumsum(ha)[:-1] / sa - np.cumsum(hb)[:-1] / sb).sum() * dx)


def score(target, symbolic, neural, primaries, rng, n_boot=BOOT, other="neural chain", boot_bins=0):
    """One observable, one group: W1, floors, medians, and the resampled symbolic minus other difference; neural
    None: the symbolic side alone. other names the second source (the chain, or a comparison draw). boot_bins > 0:
    the resampled differences from per-primary histograms of that many bins (the same resamples; every other number
    exact), instead of re-sorting every cell at every resample."""
    if neural is None:
        return {"symbolic": dist(target, symbolic), "symbolic per plane": per_plane(target, symbolic),
                "floor symbolic": floor(symbolic), "median target": median(target), "median symbolic": median(symbolic)}
    size = int(primaries.max()) + 1
    if boot_bins:
        ps = np.sort(primaries)
        vals = np.concatenate([o[0][np.isfinite(o[0])] for o in (target, symbolic, neural)])
        lo, hi = float(vals.min()), float(vals.max())
        dx = max(hi - lo, 1e-12) / boot_bins
        H = [primary_hists(o, ps, lo, dx, boot_bins) for o in (target, symbolic, neural)]
    d = []
    for _ in range(n_boot):
        mult = np.zeros(size)
        np.add.at(mult, rng.choice(primaries, len(primaries)), 1.0)
        if boot_bins:
            ht, hs, hn = (mult[ps] @ h for h in H)
            d.append(hist_w1(ht, hs, dx) - hist_w1(ht, hn, dx))
            continue
        d.append(dist(target, symbolic, mult) - dist(target, neural, mult))
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return {"symbolic": dist(target, symbolic), other: dist(target, neural), "difference 95%": [float(lo), float(hi)],
            "symbolic per plane": per_plane(target, symbolic), f"{other} per plane": per_plane(target, neural),
            "floor symbolic": floor(symbolic), f"floor {other}": floor(neural), "median target": median(target),
            "median symbolic": median(symbolic), f"median {other}": median(neural)}


def verdict(r, name, other="neural chain"):
    if name not in JUDGED or other not in r:
        return "reported"
    lo, hi = r["difference 95%"]
    return ("PASS" if r["symbolic"] <= r[other] else "FAIL") + (" clear" if lo > 0 or hi < 0 else "")


_DATA = {}


def one_piece(key):
    """One observable of one group (species, class, band), with its own resampling seed: the pieces run in parallel."""
    s, cn, band, name, seed, n_boot, *rest = key
    boot_bins = rest[0] if rest else 0
    prim = _DATA["prim"]
    primaries = np.flatnonzero((prim["cls"] == cn) & suite.BANDS[band](prim["u"]) & _DATA["judged"])
    keep = lambda o: take(o, np.isin(o[2], primaries))
    other = _DATA["other"]
    o = [keep(_DATA[src][s][name]) if src in _DATA else None for src in ("target", "symbolic", other)]
    r = score(*o, primaries, np.random.default_rng(seed), n_boot, other, boot_bins)
    r["verdict"] = verdict(r, name, other)
    return s, cn, band, name, len(primaries), r


def fig_rows(res, other="neural chain"):
    """W1 to the target per row and group, each source's floor as a small dash."""
    names = (*JUDGED, *REPORTED[:2])
    keys = [(s, cn, b) for cn in LABEL for s in SPECIES_NAMES for b in suite.BANDS]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.4), sharex=True)
    x = np.arange(len(keys))
    for ax, name in zip(axes.flat, names):
        for o, src in enumerate(("symbolic", other)):
            if src not in res[keys[0][0]][keys[0][1]][keys[0][2]][name]:
                continue
            pos = x + 0.2 * (o - 0.5)
            color = suite.COLORS.get(src, "tab:green")
            ax.scatter(pos, [res[s][cn][b][name][src] for s, cn, b in keys], color=color, s=14, label=src)
            ax.scatter(pos, [res[s][cn][b][name][f"floor {src}"] for s, cn, b in keys], color=color, s=20,
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


def fit_apart(fit, prim):
    """Laws fitted on the showers of a fit's results.json must not include a judged shower."""
    with open(fit) as fh:
        seen = set(json.load(fh)["showers"])
    if seen & set(prim["names"]):
        raise ValueError(f"the laws in {fit} were fitted on judged showers")


def stage_run(out, args, log):
    if args.compare and (args.chain != "none" or args.truth != "h5" or not args.compare_fit):
        raise ValueError("--compare needs --chain none, --truth h5 and --compare-fit")
    z = np.load(os.path.join(args.out, args.folder, "primaries.npz"))
    prim = {k: z[k] for k in z.files}
    if args.truth == "h5":
        if not args.fit:
            raise ValueError("--truth h5 needs --fit, the results.json of the laws' fit")
        for f in (*args.fit, *(args.compare_fit or ())):
            fit_apart(f, prim)
    else:
        judged_laws_apart(args.out, prim, args.folds)
    other = "comparison" if args.compare else "neural chain"
    judged = np.flatnonzero(np.isin(prim["fold"], args.folds))
    log.info("%d judged primaries, folds %s, truth %s, chain %s, compare %s", len(judged), args.folds, args.truth,
             args.chain, args.compare)
    sym_rows, sym = read_symbolic(args.out, args.folds, args.source)
    if args.compare:
        neu_rows, neu = read_symbolic(args.out, args.folds, args.compare)
    else:
        neu_rows, neu = read_chain(args.out, judged) if args.chain != "none" else (None, None)
    for src, r in (("symbolic", sym_rows), (other, neu_rows)):
        if r is None:
            continue
        if not np.array_equal(np.unique(r, return_counts=True)[0], judged) or (np.bincount(r)[judged] != args.draws).any():
            raise ValueError(f"{src}: expected {args.draws} draws at every judged primary")
    target = read_truth_h5(prim, judged) if args.truth == "h5" else read_truth(args.out, prim, judged, args.folder)
    log.info("sources read, truth checked")
    _DATA.clear()
    _DATA.update(prim=prim, judged=np.isin(np.arange(len(prim["fold"])), judged), other=other)
    for src, c, r in (("target", target, judged), ("symbolic", sym, sym_rows), (other, neu, neu_rows)):
        if c is not None:
            _DATA[src] = {s: observables(c[s], prim["dirs"], r) for s in SPECIES_NAMES}
    del target, sym, neu
    keys = [(s, cn, b, n, args.seed + j, args.boot, args.boot_bins) for j, (s, cn, b, n) in enumerate(
        (s, cn, b, n) for s in SPECIES_NAMES for cn in LABEL for b in suite.BANDS for n in (*JUDGED, *REPORTED))]
    res = {s: {cn: {b: {} for b in suite.BANDS} for cn in LABEL} for s in SPECIES_NAMES}
    with Pool(args.workers) as pool:
        for s, cn, band, n, size, r in pool.imap_unordered(one_piece, keys):
            res[s][cn][band].update({"primaries": size, n: r})
            log.info("%s %s %s (%d primaries) | %s %.3g vs %.3g (floor %.2g / %.2g) %s", s, cn, band, size, n,
                     r["symbolic"], r.get(other, np.nan), r["floor symbolic"], r.get(f"floor {other}", np.nan),
                     r["verdict"])
    logs.save_fig(fig_rows(res, other), out, "w1_rows", log)
    return {"rule": __doc__.split("Rule (fixed before the run): ")[1].strip(), "folds": args.folds, "truth": args.truth,
            "chain": args.chain, "comparison": args.compare, "judged": int(len(judged)), "results": res}


def stage_clouds(out, args, log):
    if args.truth == "h5":
        raise NotImplementedError("clouds for the detector row with the new caps: not built yet")
    z = np.load(os.path.join(args.out, args.folder, "primaries.npz"))
    prim = {k: z[k] for k in z.files}
    rows = np.repeat(prim["rows"], args.draws)
    if not np.array_equal(np.load(os.path.join(args.out, CHAIN))["rows"], rows):
        raise ValueError("the chain's clouds are not in the expected order")
    sym_rows, sym = read_symbolic(args.out, (0, 1), args.source)
    order = np.argsort(sym_rows, kind="stable")
    if not np.array_equal(sym_rows[order], rows):
        raise ValueError("the symbolic draws do not cover every primary DRAWS times")
    target = read_truth(args.out, prim, prim["rows"], args.folder)
    for name, per in (("target", {s: [target[s][r] for r in rows] for s in SPECIES_NAMES}),
                      (f"symbolic{args.suffix}", {s: [sym[s][j] for j in order] for s in SPECIES_NAMES})):
        pad = {f"{s}.points": np.stack([np.pad(c, ((0, sources.CAP[s] - len(c)), (0, 0))) for c in per[s]]).astype(np.float32)
               for s in SPECIES_NAMES}
        np.savez(os.path.join(args.out, args.folder, f"clouds_{name}.npz"), rows=rows, **pad)
        log.info("clouds_%s: %d entries", name, len(rows))
    return {"entries": int(len(rows))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("split", "primaries", "run", "clouds"))
    ap.add_argument("--out", required=True, help="the symbolic_trim root")
    ap.add_argument("--name", default="run", help="output folder under particles/judge/")
    ap.add_argument("--source", default="judge_fold{}", help="run and clouds stages: draw folder under particles/draw, {} the fold")
    ap.add_argument("--suffix", default="", help="clouds stage: added to the symbolic clouds' file name")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1], help="run stage: the folds judged")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--boot-bins", type=int, default=0, help="run stage: resampled differences from per-primary "
                    "histograms of this many bins (0: exact, every cell re-sorted at every resample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--folder", default=FOLDER, help="judge folder under --out (primaries, results)")
    ap.add_argument("--truth", choices=("tables", "h5"), default="tables",
                    help="run stage: target cells re-rendered with the old caps (tables) or the h5 files' own stored cells")
    ap.add_argument("--chain", choices=("given", "none"), default="given", help="run stage: neural chain, or none")
    ap.add_argument("--fit", nargs="+", help="run stage, --truth h5: the results.json of every fit behind the laws")
    ap.add_argument("--compare", help="run stage, --chain none: a second symbolic draw folder under particles/draw, {} the "
                    "fold, judged where the chain would be")
    ap.add_argument("--compare-fit", nargs="+", help="run stage, with --compare: the results.json of every fit behind its laws")
    ap.add_argument("--tables", help="primaries stage: summary tables folder")
    ap.add_argument("--list", help="primaries stage: file of held-out table names")
    ap.add_argument("--top", type=float, default=chain.TOP, help="primaries stage: judged below this log10 E")
    args = ap.parse_args()
    out, log = logs.start(os.path.join(args.folder, args.name if args.stage == "run" else args.stage + args.suffix), args)
    logs.save(out, {"split": stage_split, "primaries": stage_primaries, "run": stage_run,
                    "clouds": stage_clouds}[args.stage](out, args, log), args, log)


if __name__ == "__main__":
    main()
