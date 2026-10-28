"""Fixed CORSIKA-level test suite for full shower clouds (cells within the planes), run-independent.

On the scored test primaries (chain.primaries), each source (symbolic generator, neural chain) is compared with
repaired CORSIKA per species and class by Wasserstein-1 on:
  abs_t1       log10(1 + |t1| / m), energy-weighted cells; t1 = across-track offset from the axis (exact in the data)
  cell_energy  log10 cell energy [GeV], cells
  cells        log10 cells per plane
  centre       log10(1 + |energy-weighted t1| / m) per plane
  lead_time    earliest cell time per plane minus the axis light time [ns]
  mean_time    energy-weighted mean cell time per plane minus the axis light time [ns]
pooled over planes, by plane group about each shower's own peak plane (early, peak, late) and per plane index; plus a
classifier two-sample AUC on per-plane summary vectors. Scale: the same numbers between two random halves of the
CORSIKA primaries, divided by sqrt(2) to match two full samples (mean of SPLITS splits).
"""
import argparse
import os
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from symbolic import logs
from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import annuli, atmosphere, chain, lateral, planes, sources, unwrap

FOLDER = "suite"
SOURCES = {"symbolic": os.path.join("generator", "clouds.npz"), "neural chain": chain.CLOUDS}
COLORS = {"CORSIKA": "black", "symbolic": "steelblue", "neural chain": "#eb6834"}
CELL_OBS = ("abs_t1", "cell_energy")
PLANE_OBS = ("cells", "centre", "lead_time", "mean_time")
GROUPS = ("early", "peak", "late")
PEAK_HALF = 2                 # planes within this of a shower's peak plane form its "peak" group
SPLITS = 5
CLF_EPOCHS = 300
NS = 1e9


def corsika_clouds(rows, dirs, workers):
    """Repaired CORSIKA cells (energy > 0) per species at the test rows."""
    out = {}
    for s in SPECIES_NAMES:
        with h5py.File(sources.corsika_path(s, "test"), "r") as f:
            raw = [np.asarray(f["showers"][r], np.float64).reshape(-1, 5) for r in rows]
        with Pool(workers) as pool:
            out[s] = pool.starmap(_unwrap, [(c[c[:, 3] > 0], d) for c, d in zip(raw, dirs)], chunksize=8)
    return out


def _unwrap(cells, d):
    return unwrap.unwrap_shower(cells, d, N_PLANES)[0]


def npz_clouds(path, rows):
    """Cells (energy > 0) per species of a padded cloud file made on the same primaries."""
    z = np.load(path)
    if not np.array_equal(z["rows"], rows):
        raise ValueError(f"{path} was made on other primaries")
    out = {}
    for s in SPECIES_NAMES:
        pts = z[f"{s}.points"]
        out[s] = [p[p[:, 3] > 0].astype(np.float64) for p in pts]
    return out


def tables(clouds, dirs):
    """Per-cell and per-plane tables of one species' clouds.

    cells: shower i, plane k, |t1| [m], energy, peak group; planes: i, k, cell count, energy, |centre| [m], lead and
    mean time [ns], peak group, and the energy-weighted rms |t1| (classifier input)."""
    cells, pl = [], []
    for i, (c, d) in enumerate(zip(clouds, dirs)):
        if not len(c):
            continue
        d = d / np.linalg.norm(d)
        h, perp, _ = lateral.frame(d)
        k = np.rint(c[:, 2]).astype(int)
        axis = (k + 1)[:, None] * atmosphere.PLANE_SPACING * d[None, :2]
        t1 = (c[:, :2] - axis) @ perp
        e = c[:, 3]
        trel = (c[:, 4] - (k + 1) * atmosphere.PLANE_SPACING / annuli.C_LIGHT) * NS
        E_k = np.bincount(k, e, N_PLANES)
        N_k = np.bincount(k, minlength=N_PLANES)
        live = np.nonzero(N_k)[0]
        centre = np.bincount(k, e * t1, N_PLANES)[live] / E_k[live]
        rms = np.sqrt(np.bincount(k, e * t1 ** 2, N_PLANES)[live] / E_k[live])
        tmean = np.bincount(k, e * trel, N_PLANES)[live] / E_k[live]
        lead = np.full(N_PLANES, np.inf)
        np.minimum.at(lead, k, trel)
        group = np.sign(np.arange(N_PLANES) - np.argmax(E_k)) * (np.abs(np.arange(N_PLANES) - np.argmax(E_k)) > PEAK_HALF)
        cells.append(np.column_stack([np.full(len(k), i), k, np.abs(t1), e, group[k]]))
        pl.append(np.column_stack([np.full(len(live), i), live, N_k[live], E_k[live], np.abs(centre), lead[live],
                                   tmean, group[live], rms]))
    cells, pl = np.concatenate(cells), np.concatenate(pl)
    return ({"i": cells[:, 0].astype(int), "k": cells[:, 1].astype(int), "abs_t1": cells[:, 2], "e": cells[:, 3],
             "group": cells[:, 4].astype(int)},
            {"i": pl[:, 0].astype(int), "k": pl[:, 1].astype(int), "N": pl[:, 2], "E": pl[:, 3], "centre": pl[:, 4],
             "lead": pl[:, 5], "tmean": pl[:, 6], "group": pl[:, 7].astype(int), "rms": pl[:, 8]})


def observables(cells, pl):
    """name -> (values, weights, shower, plane, group)."""
    cw = (cells["i"], cells["k"], cells["group"])
    pw = (pl["i"], pl["k"], pl["group"])
    ones = np.ones(len(pl["i"]))
    return {"abs_t1": (np.log10(1 + cells["abs_t1"]), cells["e"], *cw),
            "cell_energy": (np.log10(cells["e"]), np.ones(len(cells["e"])), *cw),
            "cells": (np.log10(pl["N"]), ones, *pw),
            "centre": (np.log10(1 + pl["centre"]), ones, *pw),
            "lead_time": (pl["lead"], ones, *pw),
            "mean_time": (pl["tmean"], ones, *pw)}


def w1(a, b, ma=None, mb=None):
    """Weighted Wasserstein-1 between the selected entries of two observables (None if either side is empty)."""
    ma = np.ones(len(a[0]), bool) if ma is None else ma
    mb = np.ones(len(b[0]), bool) if mb is None else mb
    if not ma.any() or not mb.any() or a[1][ma].sum() <= 0 or b[1][mb].sum() <= 0:
        return None
    return float(stats.wasserstein_distance(a[0][ma], b[0][mb], a[1][ma], b[1][mb]))


def w1_report(a, b, sa=None, sb=None):
    """Pooled, by peak group and per plane index; sa / sb select showers (boolean over shower index)."""
    ma = np.ones(len(a[0]), bool) if sa is None else sa[a[2]]
    mb = np.ones(len(b[0]), bool) if sb is None else sb[b[2]]
    out = {"pooled": w1(a, b, ma, mb)}
    for g, name in zip((-1, 0, 1), GROUPS):
        out[name] = w1(a, b, ma & (a[4] == g), mb & (b[4] == g))
    out["per_plane"] = [w1(a, b, ma & (a[3] == k), mb & (b[3] == k)) for k in range(N_PLANES)]
    return out


def plane_features(pl, l, u):
    """Per-plane summary vectors (classifier input) and the shower each row belongs to."""
    x = np.column_stack([pl["k"], np.log10(pl["E"]), np.log10(pl["N"]), np.log10(1 + pl["rms"]),
                         np.log10(1 + pl["centre"]), pl["lead"] / 100, pl["tmean"] / 100, l[pl["i"]], u[pl["i"]]])
    return x, pl["i"]


def auc(score, y):
    r = stats.rankdata(score)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def classifier_auc(xa, ia, xb, ib, seed=0):
    """Two-sample AUC of a small MLP, two folds split by shower (both sources share primaries)."""
    rng = np.random.default_rng(seed)
    showers = np.union1d(ia, ib)
    fold = dict(zip(showers, rng.permutation(len(showers)) % 2))
    fa, fb = np.array([fold[i] for i in ia]), np.array([fold[i] for i in ib])
    x, y, f = np.vstack([xa, xb]), np.r_[np.zeros(len(xa)), np.ones(len(xb))], np.r_[fa, fb]
    ok = np.all(np.isfinite(x), 1)
    x, y, f = x[ok], y[ok], f[ok]
    out = []
    for k in (0, 1):
        tr, te = f != k, f == k
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        xt = torch.tensor((x - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        torch.manual_seed(seed + k)
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                                  torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        w = torch.tensor(np.where(y[tr] == 1, 0.5 / y[tr].mean(), 0.5 / (1 - y[tr].mean())), dtype=torch.float32)
        for _ in range(CLF_EPOCHS):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[tr])[:, 0], yt[tr], weight=w)
            loss.backward()
            opt.step()
        with torch.no_grad():
            out.append(auc(net(xt[te])[:, 0].numpy(), y[te]))
    return float(np.mean(out))


def compare(ref, other):
    """Every metric of `other` against `ref` (tables of one species and class), with the split-half scale of `ref`."""
    oa, ob = observables(*ref["tables"]), observables(*other["tables"])
    res = {name: {"vs CORSIKA": w1_report(oa[name], ob[name])} for name in oa}
    xa, ia = ref["features"]
    xb, ib = other["features"]
    res["classifier AUC"] = {"vs CORSIKA": classifier_auc(xa, ia, xb, ib)}
    return res


def split_half(ref, n_showers, members, rng):
    """Split-half scale of every metric among the CORSIKA primaries `members` (one species and class)."""
    oa = observables(*ref["tables"])
    acc = {name: [] for name in oa}
    aucs = []
    xa, ia = ref["features"]
    for j in range(SPLITS):
        perm = rng.permutation(members)
        sa, sb = np.zeros(n_showers, bool), np.zeros(n_showers, bool)
        sa[perm[: len(perm) // 2]], sb[perm[len(perm) // 2:]] = True, True
        for name, o in oa.items():
            acc[name].append(w1_report(o, o, sa, sb))
        if j == 0:
            aucs.append(classifier_auc(xa[sa[ia]], ia[sa[ia]], xa[sb[ia]], ia[sb[ia]]))
    scale = lambda vals: None if any(v is None for v in vals) else float(np.mean(vals) / np.sqrt(2))
    out = {}
    for name, reps in acc.items():
        out[name] = {g: scale([r[g] for r in reps]) for g in ("pooled", *GROUPS)}
        out[name]["per_plane"] = [scale([r["per_plane"][k] for r in reps]) for k in range(N_PLANES)]
    out["classifier AUC"] = {"pooled": float(np.mean(aucs))}
    return out


def select(cells, pl, keep):
    """Rows of the tables whose shower is in `keep` (boolean over shower index)."""
    return ({k: v[keep[cells["i"]]] for k, v in cells.items()}, {k: v[keep[pl["i"]]] for k, v in pl.items()})


def fig_t1(data, labels):
    """Energy-weighted |t1| distribution on each shower's peak planes, per species and class."""
    fig, axes = plt.subplots(len(planes.CLASS), len(SPECIES_NAMES), figsize=(4.2 * len(SPECIES_NAMES), 3.6 * len(planes.CLASS)))
    bins = np.linspace(0, 4, 41)
    for r, (c, cn) in enumerate(planes.CLASS.items()):
        for j, s in enumerate(SPECIES_NAMES):
            ax = axes[r, j]
            for src, d in data.items():
                cells, _ = d[s]
                m = (labels[cells["i"]] == c) & (cells["group"] == 0)
                ax.hist(np.log10(1 + cells["abs_t1"][m]), bins, weights=cells["e"][m] / cells["e"][m].sum(),
                        histtype="step", color=COLORS[src], lw=1.4, label=src)
            ax.set_yscale("log")
            ax.set_title(f"{s}, {cn}, peak planes", fontsize=9)
            ax.set_xlabel("log10(1 + |across-track offset| / m)", fontsize=8)
            if j == 0:
                ax.set_ylabel("energy fraction", fontsize=8)
            if r == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_ratio(res):
    """Pooled W1 over the split-half scale, per observable, species and class."""
    names = (*CELL_OBS, *PLANE_OBS)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4), sharey=True)
    keys = [(s, cn) for cn in planes.CLASS.values() for s in SPECIES_NAMES]
    x = np.arange(len(keys))
    for ax, name in zip(axes, names):
        for o, src in enumerate(SOURCES):
            v = [res[s][cn][src][name]["vs CORSIKA"]["pooled"] / res[s][cn]["split-half scale"][name]["pooled"]
                 for s, cn in keys]
            ax.scatter(x + 0.15 * (o - 0.5), v, color=COLORS[src], s=18, label=src)
        ax.axhline(1, color="gray", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s[:2]} {cn[:3]}" for s, cn in keys], rotation=60, fontsize=7)
        ax.set_title(name, fontsize=9)
    axes[0].set_ylabel("W1 / split-half scale", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def stage_run(out, args, log):
    rows, E, D, lab = chain.primaries()
    dirs = D.astype(np.float64)
    l, u = np.log10(E[:, 0].astype(np.float64)), dirs[:, 2] / np.linalg.norm(dirs, axis=1)
    n = len(rows)
    clouds = {"CORSIKA": corsika_clouds(rows, dirs, args.workers)}
    log.info("CORSIKA read and repaired: %d primaries", n)
    for src, rel in SOURCES.items():
        clouds[src] = npz_clouds(os.path.join(args.out, rel), rows)
        log.info("%s read from %s", src, rel)
    data = {src: {s: tables(c[s], dirs) for s in SPECIES_NAMES} for src, c in clouds.items()}
    del clouds
    rng = np.random.default_rng(args.seed)
    res = {s: {} for s in SPECIES_NAMES}
    for s in SPECIES_NAMES:
        for c, cn in planes.CLASS.items():
            keep = lab == c
            sub = {src: {"tables": select(*d[s], keep)} for src, d in data.items()}
            for v in sub.values():
                v["features"] = plane_features(v["tables"][1], l, u)
            r = {"split-half scale": split_half(sub["CORSIKA"], n, np.nonzero(keep)[0], rng)}
            for src in SOURCES:
                r[src] = compare(sub["CORSIKA"], sub[src])
            res[s][cn] = r
            log.info("%s %s pooled W1 vs CORSIKA (split-half scale): %s | AUC symbolic %.3f chain %.3f halves %.3f", s, cn,
                     {name: [round(r[src][name]["vs CORSIKA"]["pooled"], 4) for src in SOURCES]
                      + [round(r["split-half scale"][name]["pooled"], 4)] for name in (*CELL_OBS, *PLANE_OBS)},
                     r["symbolic"]["classifier AUC"]["vs CORSIKA"], r["neural chain"]["classifier AUC"]["vs CORSIKA"],
                     r["split-half scale"]["classifier AUC"]["pooled"])
    logs.save_fig(fig_t1(data, lab), out, "abs_t1_peak", log)
    logs.save_fig(fig_ratio(res), out, "w1_over_scale", log)
    return {"n_primaries": n, "sources": {"CORSIKA": "h5 test files, repaired (unwrap)", **SOURCES},
            "observables": __doc__.split("by Wasserstein-1 on:")[1].split("pooled over")[0].strip(), "results": res}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run folder holding generator/clouds.npz and chain/clouds_draw00.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--symbolic", default=SOURCES["symbolic"], help="symbolic clouds, relative to --out")
    ap.add_argument("--name", default="run", help="output folder under suite/")
    args = ap.parse_args()
    SOURCES["symbolic"] = args.symbolic
    out, log = logs.start(os.path.join(FOLDER, args.name), args)
    logs.save(out, stage_run(out, args, log), args, log)


if __name__ == "__main__":
    main()
