"""Judge drawn showers against CORSIKA at held-out primaries: the target is each shower's own stored cells (data.py).

Stages:
  primaries  the held-out tables named in --list with log10 E < --top -> <out>/primaries.npz (with each simulation folder)
  run        per species, class and zenith band, W1 to the target of each observable, pooled over the draws of each
             primary, at the primaries the model's clouds cover, for the model's draw and optionally a second source (--other, any clouds file in the same
             format), with the floor of each (one draw per primary against its other draws).

Rule: a row PASSES when the model's W1 is at most the other source's; it is CLEAR when the 95% interval of the
difference, resampling primaries (the same resample for both sources), excludes 0. Rows in JUDGED are judged, the rest
reported. The resampled differences come from per-primary histograms on BINS bins; every other number is exact.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
from scipy import stats

from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere, data, lateral, model, rows

JUDGED = ("abs_t1", "cell_energy", "cells", "centre")
REPORTED = ("lead_time", "mean_time", "plane_energy")
GRAZE = 0.1                   # |cos z| below this: grazing (the in-plane frame is unresolved there)
BANDS = {"steep": lambda u: np.abs(u) >= GRAZE, "grazing": lambda u: np.abs(u) < GRAZE}
BOOT = 200
BINS = 65536


def stage_primaries(args):
    names = np.array(sorted(set(open(args.list).read().split())))
    t = [np.load(os.path.join(args.tables, n)) for n in names]
    l = np.array([float(z["log10E"]) for z in t])
    dirs = np.array([z["direction"] for z in t])
    if not np.isclose(np.linalg.norm(dirs, axis=1), 1).all():
        raise ValueError("directions that are not unit vectors")
    keep = l < args.top
    u = dirs[keep, 2]
    cls = np.array([str(z["class"]) for z in t])[keep]
    prim = {"names": names[keep], "sims": np.array([str(z["sim"]) for z in t])[keep], "cls": cls,
            "labels": np.array([data.LABEL[c] for c in cls]), "l": l[keep], "u": u, "dirs": dirs[keep],
            "X": atmosphere.plane_depths(np.degrees(np.arccos(u)), N_PLANES)}
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "primaries.npz"), rows=np.arange(keep.sum()), **prim)
    return {"listed": len(names), "above the range": int((~keep).sum()),
            "judged": {f"{c} {b}": int(((cls == c) & sel(u)).sum()) for c in data.LABEL for b, sel in BANDS.items()}}


def read_clouds(path):
    """rows and, per species, the cells (energy > 0) of each padded cloud."""
    z = np.load(path)
    return z["rows"], {s: [p[p[:, 3] > 0].astype(np.float64) for p in z[f"{s}.points"]] for s in SPECIES_NAMES}


def draw_index(r):
    """Each entry's position among the entries of the same primary."""
    seen, out = {}, np.zeros(len(r), int)
    for j, q in enumerate(r):
        out[j] = seen.get(q, 0)
        seen[q] = out[j] + 1
    return out


def observables(clouds, dirs, prim_rows):
    """name -> (values, weights, primary, draw, plane) of one species' clouds; prim_rows: the primary of each cloud."""
    cells, planes = [], []
    for i, (c, d) in enumerate(zip(clouds, dirs[prim_rows])):
        if not len(c):
            continue
        d = d / np.linalg.norm(d)
        _, perp, _ = lateral.frame(d)
        k = np.rint(c[:, 2]).astype(int)
        t1 = (c[:, :2] - (k + 1)[:, None] * atmosphere.PLANE_SPACING * d[None, :2]) @ perp
        e = c[:, 3]
        trel = (c[:, 4] - (k + 1) * atmosphere.PLANE_SPACING / rows.C_LIGHT) * 1e9
        E_k, N_k = np.bincount(k, e, N_PLANES), np.bincount(k, minlength=N_PLANES)
        live = np.nonzero(N_k)[0]
        lead = np.full(N_PLANES, np.inf)
        np.minimum.at(lead, k, trel)
        cells.append(np.column_stack([np.full(len(k), i), k, np.abs(t1), e]))
        planes.append(np.column_stack([np.full(len(live), i), live, N_k[live], E_k[live],
                                       np.abs(np.bincount(k, e * t1, N_PLANES)[live] / E_k[live]), lead[live],
                                       np.bincount(k, e * trel, N_PLANES)[live] / E_k[live]]))
    c, p = np.concatenate(cells), np.concatenate(planes)
    ci, ck, pi, pk = c[:, 0].astype(int), c[:, 1].astype(int), p[:, 0].astype(int), p[:, 1].astype(int)
    ones = np.ones(len(p))
    out = {"abs_t1": (np.log10(1 + c[:, 2]), c[:, 3], ci, ck), "cell_energy": (np.log10(c[:, 3]), np.ones(len(c)), ci, ck),
           "cells": (np.log10(p[:, 2]), ones, pi, pk), "centre": (np.log10(1 + p[:, 4]), ones, pi, pk),
           "lead_time": (p[:, 5], ones, pi, pk), "mean_time": (p[:, 6], ones, pi, pk),
           "plane_energy": (np.log10(p[:, 3]), ones, pi, pk)}
    draw = draw_index(prim_rows)
    return {name: (v, w, prim_rows[i], draw[i], k) for name, (v, w, i, k) in out.items()}


def take(o, m):
    return tuple(x[m] for x in o)


def dist(a, b):
    """W1 between two observables; NaN if either side is empty."""
    if a[1].sum() <= 0 or b[1].sum() <= 0:
        return np.nan
    return float(stats.wasserstein_distance(a[0], b[0], a[1], b[1]))


def floor(o):
    """Mean W1 of one draw per primary against the same source's other draws."""
    return float(np.nanmean([dist(take(o, o[3] == d), take(o, o[3] != d)) for d in np.unique(o[3])]))


def per_plane(a, b):
    """Mean over planes of the W1 on each plane (a pooled W1 ignores plane order)."""
    return float(np.nanmean([dist(take(a, a[4] == k), take(b, b[4] == k)) for k in range(N_PLANES)]))


def median(o):
    i = np.argsort(o[0])
    c = np.cumsum(o[1][i])
    return float(o[0][i][np.searchsorted(c, 0.5 * c[-1])])


def primary_hists(o, primaries, edges):
    """Weighted histogram of one observable per primary, rows in the order of primaries."""
    nb = len(edges) - 1
    i = np.clip(np.searchsorted(primaries, o[2]), 0, len(primaries) - 1)
    ok = (primaries[i] == o[2]) & np.isfinite(o[0])
    b = np.clip(np.searchsorted(edges, o[0][ok], side="right") - 1, 0, nb - 1)
    return np.bincount(i[ok] * nb + b, o[1][ok], len(primaries) * nb).reshape(len(primaries), nb)


def bin_edges(vals, n):
    """The distinct values if there are at most 4 n of them (exact), else the union of n quantile and n uniform edges."""
    u = np.unique(vals)
    if len(u) > 4 * n:
        u = np.unique(np.r_[np.quantile(vals, np.linspace(0.0, 1.0, n + 1)), np.linspace(u[0], u[-1], n + 1)])
    return np.r_[u, u[-1] + 1.0]


def hist_w1(ha, hb, step):
    """W1 between two histograms on the same bins (mass at each bin's left edge); NaN if either is empty."""
    sa, sb = ha.sum(), hb.sum()
    if sa <= 0 or sb <= 0:
        return np.nan
    return float((np.abs(np.cumsum(ha)[:-1] / sa - np.cumsum(hb)[:-1] / sb) * step).sum())


def score(target, mine, other, primaries, rng, other_name, n_boot=BOOT, bins=BINS):
    """One observable, one group: W1, floors and medians of each source, and the 95% interval of the resampled
    difference model minus other (other None: the model alone)."""
    res = {"symbolic": dist(target, mine), "symbolic per plane": per_plane(target, mine), "floor symbolic": floor(mine),
           "median target": median(target), "median symbolic": median(mine)}
    if other is None:
        return res
    ps = np.sort(primaries)
    edges = bin_edges(np.concatenate([o[0][np.isfinite(o[0])] for o in (target, mine, other)]), bins)
    step = np.diff(edges[:-1])
    H = [primary_hists(o, ps, edges) for o in (target, mine, other)]
    d = []
    for _ in range(n_boot):
        mult = np.zeros(int(primaries.max()) + 1)
        np.add.at(mult, rng.choice(primaries, len(primaries)), 1.0)
        ht, hm, ho = (mult[ps] @ h for h in H)
        d.append(hist_w1(ht, hm, step) - hist_w1(ht, ho, step))
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return {**res, other_name: dist(target, other), "difference 95%": [float(lo), float(hi)],
            f"{other_name} per plane": per_plane(target, other), f"floor {other_name}": floor(other),
            f"median {other_name}": median(other)}


def verdict(r, name, other_name):
    if name not in JUDGED or other_name not in r:
        return "reported"
    lo, hi = r["difference 95%"]
    return ("PASS" if r["symbolic"] <= r[other_name] else "FAIL") + (" clear" if lo > 0 or hi < 0 else "")


_DATA = {}


def one_piece(key):
    """One observable of one group, with its own resampling seed (the pieces run in parallel)."""
    s, cn, band, name, seed = key
    prim, other_name = _DATA["prim"], _DATA["other name"]
    primaries = np.flatnonzero((prim["cls"] == cn) & BANDS[band](prim["u"]) & _DATA["judged"])
    o = [take(_DATA[src][s][name], np.isin(_DATA[src][s][name][2], primaries)) if src in _DATA else None
         for src in ("target", "model", "other")]
    r = score(*o, primaries, np.random.default_rng(seed), other_name)
    r["verdict"] = verdict(r, name, other_name)
    return s, cn, band, name, len(primaries), r


def stage_run(args):
    z = np.load(args.primaries)
    prim = {k: z[k] for k in z.files}
    for folder in [args.model] + (args.other_model or []):
        if model.showers(folder) & set(prim["names"]):
            raise ValueError(f"{folder} was fitted on judged showers")
    sources = {"model": read_clouds(args.clouds)}
    if args.other:
        sources["other"] = read_clouds(args.other)
    judged = np.unique(sources["model"][0])
    for src, (r, _) in sources.items():
        if not np.array_equal(np.unique(r), judged) or (np.bincount(r)[judged] != args.draws).any():
            raise ValueError(f"{src}: expected {args.draws} draws at every primary the model's clouds cover")
    truth = [data.stored_cells(str(prim["sims"][i]), prim["dirs"][i]) for i in judged]
    _DATA.clear()
    _DATA.update({"prim": prim, "judged": np.isin(np.arange(len(prim["names"])), judged), "other name": args.other_name,
                  "target": {s: observables([t[s] for t in truth], prim["dirs"], judged) for s in SPECIES_NAMES}})
    for src, (r, c) in sources.items():
        _DATA[src] = {s: observables(c[s], prim["dirs"], r) for s in SPECIES_NAMES}
    del truth, sources
    keys = [(s, cn, b, n, args.seed + j) for j, (s, cn, b, n) in enumerate(
        (s, cn, b, n) for s in SPECIES_NAMES for cn in data.LABEL for b in BANDS for n in (*JUDGED, *REPORTED))]
    res = {s: {cn: {b: {} for b in BANDS} for cn in data.LABEL} for s in SPECIES_NAMES}
    with Pool(args.workers) as pool:
        for s, cn, band, n, size, r in pool.imap_unordered(one_piece, keys):
            res[s][cn][band].update({"primaries": size, n: r})
    return {"rule": __doc__.split("Rule: ")[1].strip(), "judged": len(judged), "results": res}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("primaries", "run"))
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--tables", help="primaries: summary tables folder")
    ap.add_argument("--list", help="primaries: held-out table names")
    ap.add_argument("--top", type=float, default=7.0, help="primaries: judged below this log10 E")
    ap.add_argument("--primaries", help="run: primaries file")
    ap.add_argument("--model", help="run: the model folder the clouds were drawn from")
    ap.add_argument("--clouds", help="run: the model's clouds")
    ap.add_argument("--other", help="run: a second source's clouds at the same entries")
    ap.add_argument("--other-name", default="comparison", help="run: its name in the results")
    ap.add_argument("--other-model", nargs="+", help="run: model folders behind --other, checked as --model")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    res = {"primaries": stage_primaries, "run": stage_run}[args.stage](args)
    os.makedirs(args.out, exist_ok=True)
    name = "primaries.json" if args.stage == "primaries" else "results.json"
    with open(os.path.join(args.out, name), "w") as h:
        json.dump({"args": vars(args), "git": data.git_head(), **res}, h, indent=1)


if __name__ == "__main__":
    main()
