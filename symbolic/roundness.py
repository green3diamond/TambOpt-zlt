"""Is the energy in each plane round about the shower axis? (study_plan.md 2.2; data only, no fitting.)

On planes of repaired CORSIKA test showers with |cos z| >= COS_MIN, each cell's offset from the axis point splits
into the across-track distance a (exact up to the cell) and the in-plane along-track distance b = g / |cos z| (ground
offset g, so the cell's spread is stretched by 1 / |cos z|). Each side gets the cell noise of the other side, so a
round density gives the same distribution on both: A = a + (u.h) / |cos z|, B = b + u'.perp, u, u' uniform in a
cell. Compared by energy-weighted Wasserstein-1 on log10(1 + |distance| / m), every shower-plane weighted equally,
within windows about the axis (the radius cut and the cap act far out and not symmetrically), against CORSIKA
split-half noise.

Rule (fixed before the run): PASS for e/gamma if W1(A, B) <= 2 x split-half noise in the 300 m window.

--permutations N: a null made from the data themselves. Under roundness A and B are exchangeable, so swapping them in
randomly chosen shower-planes leaves W1 statistically unchanged; N such swaps give the W1 a round shower would show at
this sample size and sparsity. Rule (fixed before that run): round if the observed W1 in the 300 m window is within
the 95th percentile of the swapped ones (p > 0.05). --window W runs the same test in a W m window (100 m stays inside
the region the radius cut does not touch); --species runs one species, and results are saved after every group.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from symbolic import logs
from modules.constants import SPECIES_NAMES
from symbolic import atmosphere, fit_profiles, lateral, planes, render, suite

FOLDER = "roundness"
COS_MIN = 0.3
WINDOWS = (100.0, 300.0, 1000.0)
RULE_WINDOW = 300.0
SPLITS = 5


def offsets(cells, d, rng):
    """Symmetrized across (A) and along (B) distances and per-plane normalized energy weights of one shower."""
    d = d / np.linalg.norm(d)
    h, perp, c = lateral.frame(d)
    k = np.rint(cells[:, 2]).astype(int)
    rel = cells[:, :2] - (k + 1)[:, None] * atmosphere.PLANE_SPACING * d[None, :2]
    t = lateral.to_plane(rel, h, perp, c)
    u1, u2 = (rng.uniform(-render.CELL / 2, render.CELL / 2, (len(k), 2)) for _ in range(2))
    A = t[:, 0] + (u1 @ h) / c
    B = t[:, 1] + u2 @ perp
    w = cells[:, 3] / np.bincount(k, cells[:, 3], k.max() + 1)[k]
    return A, B, w, k


def w1(x, y, wx, wy, window):
    mx, my = np.abs(x) <= window, np.abs(y) <= window
    return float(stats.wasserstein_distance(np.log10(1 + np.abs(x[mx])), np.log10(1 + np.abs(y[my])), wx[mx], wy[my]))


def tables(clouds, dirs, rng, species=SPECIES_NAMES):
    """Per species: shower index, A, B, weight of every cell."""
    out = {}
    for s in species:
        parts = [(i, *offsets(c[c[:, 3] > 0], dirs[i], rng)) for i, c in enumerate(clouds[s]) if np.any(c[:, 3] > 0)]
        out[s] = {"i": np.concatenate([np.full(len(p[1]), p[0]) for p in parts]),
                  **{k: np.concatenate([p[j] for p in parts]) for j, k in ((1, "A"), (2, "B"), (3, "w"), (4, "k"))}}
    return out


def split_half(t, members, rng):
    """Mean W1 between the B distributions of two random halves of the showers, / sqrt(2), per window."""
    vals = {w: [] for w in WINDOWS}
    for _ in range(SPLITS):
        half = rng.permutation(members)[: len(members) // 2]
        m = np.isin(t["i"], half)
        for w in WINDOWS:
            vals[w].append(w1(t["B"][m], t["B"][~m], t["w"][m], t["w"][~m], w))
    return {w: float(np.mean(v) / np.sqrt(2)) for w, v in vals.items()}


def permutation_p(t, window, n_perm, rng):
    """Observed W1(A, B) and the share of plane-wise A/B swaps that give at least as large a W1."""
    obs = w1(t["A"], t["B"], t["w"], t["w"], window)
    unit = np.unique(t["i"] * 64 + t["k"], return_inverse=True)[1].ravel()
    null = []
    for _ in range(n_perm):
        swap = rng.random(unit.max() + 1)[unit] < 0.5
        A, B = np.where(swap, t["B"], t["A"]), np.where(swap, t["A"], t["B"])
        null.append(w1(A, B, t["w"], t["w"], window))
    null = np.array(null)
    return obs, float((null >= obs).mean()), float(np.percentile(null, 95))


def fig_profiles(tab, lab):
    bins = np.logspace(0, 3.3, 34)
    fig, axes = plt.subplots(2, len(tab), figsize=(7.6, 4.4), sharex=True, sharey=True, squeeze=False)
    for r, (c, cn) in enumerate(planes.CLASS.items()):
        for j, s in enumerate(tab):
            ax, t = axes[r, j], tab[s]
            m = lab[t["i"]] == c
            for key, col, name in (("A", "black", "across track"), ("B", "#2a78d6", "along track")):
                ax.hist(1 + np.abs(t[key][m]), bins, weights=t["w"][m] / t["w"][m].sum(), histtype="step", lw=1.4,
                        color=col, label=name)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{s}s, {cn}", fontsize=10)
            if r == 1:
                ax.set_xlabel("1 + distance [m]")
    axes[0, 0].set_ylabel("energy fraction")
    axes[1, 0].set_ylabel("energy fraction")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def stage_run(out, args, log):
    zt = np.load(os.path.join(args.out, fit_profiles.TEST))
    cz = np.abs(zt["meta.cos_zenith"])
    rows = np.nonzero(cz >= COS_MIN)[0]
    dirs = zt["meta.directions"][rows].astype(np.float64)
    lab = zt["meta.labels"][rows]
    log.info("%d test showers with |cos z| >= %.2f (EM %d, hadronic %d)", len(rows), COS_MIN, (lab == 0).sum(),
             (lab == 1).sum())
    clouds = suite.corsika_clouds(rows, dirs, args.workers)
    rng = np.random.default_rng(args.seed)
    species = [args.species] if args.species else SPECIES_NAMES
    tab = tables(clouds, dirs, rng, species)
    res = {"n_showers": len(rows), "cos_min": COS_MIN, "permutation window": args.window,
           "rule": __doc__.split("Rule")[1].strip(), "results": {}}
    for s in species:
        for c, cn in planes.CLASS.items():
            t = tab[s]
            m = lab[t["i"]] == c
            sub = {k: v[m] for k, v in t.items()}
            members = np.unique(sub["i"])
            scale = split_half(sub, members, rng)
            d = {w: w1(sub["A"], sub["B"], sub["w"], sub["w"], w) for w in WINDOWS}
            r = {"n_showers": len(members), "W1 across vs along": d, "split-half noise": scale,
                 "ratio": {w: d[w] / scale[w] for w in WINDOWS}}
            if args.permutations:
                obs, p, q95 = permutation_p(sub, args.window, args.permutations, rng)
                r["permutation"] = {"W1": obs, "p": p, "null 95th percentile": q95,
                                    "verdict": "ROUND" if p > 0.05 else "NOT ROUND"}
                log.info("%-8s %-8s permutation test at %d m: W1 %.4f, null 95%% %.4f, p %.3f -> %s", s, cn,
                         args.window, obs, q95, p, r["permutation"]["verdict"])
            if s != "muon":
                r["verdict"] = "PASS" if d[RULE_WINDOW] <= 2 * scale[RULE_WINDOW] else "FAIL"
            res["results"][f"{s} {cn}"] = r
            log.info("%-8s %-8s n %4d | W1 / noise per window %s %s", s, cn, len(members),
                     {int(w): f"{d[w]:.4f}/{scale[w]:.4f}" for w in WINDOWS}, r.get("verdict", ""))
            logs.save(out, res, args, log)
    logs.save_fig(fig_profiles(tab, lab), out, "across_vs_along", log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--permutations", type=int, default=0, help="plane-wise A/B swaps for the permutation null")
    ap.add_argument("--window", type=float, default=RULE_WINDOW, help="window [m] of the permutation test")
    ap.add_argument("--species", choices=SPECIES_NAMES, help="one species only")
    ap.add_argument("--name", default="run", help="output folder under roundness/")
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, args.name), args)
    stage_run(out, args, log)


if __name__ == "__main__":
    main()
