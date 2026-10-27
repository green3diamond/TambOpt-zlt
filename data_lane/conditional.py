"""Step 3: AllShowers against CORSIKA at the same primaries inside the training range, and a generation-only energy scan beyond it.

Stages: generate_match (GPU: one event per CORSIKA test primary plus repeats at fixed primaries), generate_scan (GPU: energy
grid at fixed directions), compare (CPU: CORSIKA summaries, binned comparison, calibration ranks, energy trends).
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py
import numpy as np
import torch
from scipy.stats import kstest

from analysis_core import logs, plots
from data_lane import showers, sources
from data_lane.census import angles
from modules.constants import SPECIES_NAMES

E_EDGES = np.arange(5.0, 6.251, 0.25)     # CORSIKA training is effectively cut at log10 E 6.25 (census)
HIGH = 6.25
Z_EDGES = np.arange(60.0, 100.1, 10.0)
CLASS = {0: "EM", 1: "hadronic"}
N_REPEAT = 16
N_FIXED_IN = 54                            # in-range fixed primaries, stratified by energy bin and class
SCAN_LOG_E = np.round(np.arange(5.0, 7.001, 0.1), 2)
SCAN_ZENITH = (65.0, 80.0, 95.0)
SCAN_AZIMUTH = 100.0
METRICS = ("log10_energy", "onset_layer", "peak_layer", "end_layer", "mean_layer", "peak_width_m", "n_points",
           "peak_time_rms")
N_BOOT = 200
FOLDER = os.path.join("data_lane", "conditional")


# ---------------------------------------------------------------- primaries

def test_primaries(limit=0):
    """CORSIKA test primaries (species files are row-matched; checked in the gate). With `limit`, the first `limit` rows
    plus every row above HIGH (smoke runs). `rows` holds the test-file row of each entry."""
    with h5py.File(sources.corsika_path(SPECIES_NAMES[0], "test"), "r") as f:
        m = {"ids": f["shower_ids"][...], "energies": f["energies"][...].reshape(-1, 1).astype(np.float32),
             "directions": f["directions"][...].astype(np.float32), "labels": f["pdg"][...].astype(np.int64)}
        em = np.isin(np.abs(f["actual_pdg"][...]), (11, 111))
    if not np.array_equal(m["labels"] == 0, em):
        raise ValueError("test-file class label 0 is not exactly the e/pi0 primaries")
    rows = np.arange(len(m["ids"]))
    if limit:
        high = np.nonzero(np.log10(m["energies"][:, 0]) > HIGH)[0]
        rows = np.union1d(rows[:limit], high)
    m = {k: v[rows] for k, v in m.items()}
    m["rows"] = rows
    m["log10_E"] = np.log10(m["energies"][:, 0].astype(np.float64))
    m["zenith"], m["azimuth"] = angles(m["directions"].astype(np.float64))
    return m


def fixed_primaries(meta, seed=0):
    """Test rows that get N_REPEAT generator draws: N_FIXED_IN stratified over (energy bin, class), plus every row above HIGH."""
    rng = np.random.default_rng(seed)
    e_bin = np.digitize(meta["log10_E"], E_EDGES) - 1
    strata = [(b, c) for b in range(len(E_EDGES) - 1) for c in CLASS]
    picks = []
    for k, (b, c) in enumerate(strata):
        n = N_FIXED_IN // len(strata) + (k < N_FIXED_IN % len(strata))
        pool = np.nonzero((e_bin == b) & (meta["labels"] == c))[0]
        picks.append(rng.choice(pool, size=min(n, len(pool)), replace=False))
    high = np.nonzero(meta["log10_E"] > HIGH)[0]
    return np.sort(np.concatenate(picks)), high


def scan_primaries(repeats=N_REPEAT):
    """Energy grid at fixed directions and both classes, `repeats` draws per point."""
    rows = [(e, z, c) for z in SCAN_ZENITH for c in CLASS for e in SCAN_LOG_E for _ in range(repeats)]
    log_e, zen, lab = (np.array(v) for v in zip(*rows))
    t, p = np.radians(zen), np.radians(SCAN_AZIMUTH)
    d = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], 1).astype(np.float32)
    return {"log10_E": log_e, "zenith": zen, "labels": lab.astype(np.int64),
            "energies": (10 ** log_e).reshape(-1, 1).astype(np.float32), "directions": d}


# ---------------------------------------------------------------- generation stages

def generate(energies, directions, labels, seed, log):
    s0 = sources.stage0("fork")
    clouds = sources.generate(s0, torch.as_tensor(energies), torch.as_tensor(directions), torch.as_tensor(labels), seed=seed)
    log.info("generated %d events: %s", len(energies), {s: tuple(v.shape) for s, v in clouds.items()})
    return clouds


def save_summaries(path, prim, desc, log, **extra):
    np.savez(path, **{k: v for k, v in prim.items()}, **extra,
             **{f"{s}.{k}": v for s in desc for k, v in desc[s].items()})
    log.info("wrote %s (%d bytes)", path, os.path.getsize(path))


def stage_generate_match(out, figs, args, log):
    meta = test_primaries(args.limit)
    fixed_in, high = fixed_primaries(meta, args.seed)
    fixed = np.concatenate([fixed_in, high])
    rep = np.repeat(fixed, N_REPEAT)
    take = lambda k: np.concatenate([meta[k], meta[k][rep]])
    clouds = generate(take("energies"), take("directions"), take("labels"), args.seed + 1, log)
    n = len(meta["ids"])
    desc = {s: showers.describe(clouds[s]) for s in SPECIES_NAMES}
    save_summaries(os.path.join(out, "generated_match.npz"), {k: take(k) for k in meta}, desc, log,
                   repeat_of=np.concatenate([np.full(n, -1), rep]), fixed_in=fixed_in, fixed_high=high)
    for s in SPECIES_NAMES:
        path = os.path.join(out, f"generated_match_{s}.pt")
        torch.save(clouds[s][:n].clone(), path)
        log.info("wrote %s (%d bytes)", path, os.path.getsize(path))
    return {"n_matched": n, "n_fixed_in": len(fixed_in), "n_fixed_high": len(high), "n_repeat": N_REPEAT}


def stage_generate_scan(out, figs, args, log):
    prim = scan_primaries(2 if args.limit else N_REPEAT)
    clouds = generate(prim["energies"], prim["directions"], prim["labels"], args.seed + 2, log)
    desc = {s: showers.describe(clouds[s]) for s in SPECIES_NAMES}
    save_summaries(os.path.join(out, "generated_scan.npz"), prim, desc, log)
    return {"n_scan": len(prim["labels"])}


# ---------------------------------------------------------------- comparison helpers

def boot_ci(f, *arrays, rng, n_boot=N_BOOT):
    """Point value of f and its 16th to 84th bootstrap interval (rows resampled jointly)."""
    n = len(arrays[0])
    if n < 5:
        return [float("nan")] * 3
    vals = [f(*(a[i] for a in arrays)) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return [float(f(*arrays))] + np.nanpercentile(vals, [16, 84]).tolist()


def width68(a):
    q = np.nanpercentile(a, [16, 84])
    return float(q[1] - q[0])


def compare_bins(cor, gen, mask_sets, rng):
    """Per bin and metric: paired median difference (generated - CORSIKA) and the ratio of 16-84 widths, with intervals."""
    out = {}
    for name, m in mask_sets.items():
        out[name] = {"n": int(m.sum())}
        for k in METRICS:
            c, g = cor[k][m], gen[k][m]
            ok = np.isfinite(c) & np.isfinite(g)
            c, g = c[ok], g[ok]
            ratio = lambda a, b: width68(a) / width68(b) if width68(b) > 0 else np.nan
            out[name][k] = {"median_corsika": float(np.median(c)) if len(c) else float("nan"),
                            "width68_corsika": width68(c) if len(c) else float("nan"),
                            "width68_generated": width68(g) if len(g) else float("nan"),
                            "paired_median_diff": boot_ci(lambda a, b: np.median(a - b), g, c, rng=rng),
                            "width_ratio": boot_ci(ratio, g, c, rng=rng)}
    return out


def masks(meta):
    e_bin = np.digitize(meta["log10_E"], E_EDGES) - 1
    z_bin = np.digitize(meta["zenith"], Z_EDGES) - 1
    sets = {}
    for b in range(len(E_EDGES) - 1):
        for c, cn in CLASS.items():
            base = (e_bin == b) & (meta["labels"] == c)
            sets[f"E{E_EDGES[b]:.2f}|{cn}|all_zenith"] = base
            for z in range(len(Z_EDGES) - 1):
                sets[f"E{E_EDGES[b]:.2f}|{cn}|Z{Z_EDGES[z]:.0f}"] = base & (z_bin == z)
    return sets


def response(cor, gen, meta, rng):
    """Slope of log10 total deposit against log10 primary energy inside the training range, per class."""
    out = {}
    fit = lambda x, y: np.polyfit(x, y, 1)[0]
    for c, cn in CLASS.items():
        m = (meta["labels"] == c) & (meta["log10_E"] <= HIGH)
        out[cn] = {}
        for src, d in (("corsika", cor), ("generated", gen)):
            ok = m & np.isfinite(d["log10_energy"])
            out[cn][src] = boot_ci(fit, meta["log10_E"][ok], d["log10_energy"][ok], rng=rng)
    return out


def ranks(cor, gen_rep, rep_of, rows):
    """Rank of the CORSIKA value among the N_REPEAT generated draws at the same primary (0 to N_REPEAT; ties count half)."""
    out = {}
    for k in METRICS:
        r = []
        for i in rows:
            g = gen_rep[k][rep_of == i]
            c = cor[k][i]
            r.append(float(np.sum(g < c) + 0.5 * np.sum(g == c)) if np.isfinite(c) and np.isfinite(g).all() else np.nan)
        out[k] = np.array(r)
    return out


def rank_summary(r):
    r = r[np.isfinite(r)]
    pit = (r + 0.5) / (N_REPEAT + 1)
    return {"n": len(r), "outside_repeat_range": float(np.mean((r <= 0) | (r >= N_REPEAT))),
            "expected_outside": 2 / (N_REPEAT + 1), "mean_pit": float(pit.mean()) if len(r) else float("nan"),
            "ks_p_uniform": float(kstest(pit, "uniform").pvalue) if len(r) >= 5 else float("nan")}


def scan_trends(scan, cor, meta):
    """Per species, class and scan zenith: generated medians over the energy grid, and the CORSIKA linear trend below HIGH
    (test showers within 5 deg of the scan zenith) extrapolated to the grid."""
    out = {}
    for s in SPECIES_NAMES:
        out[s] = {}
        for c, cn in CLASS.items():
            for z in SCAN_ZENITH:
                key = f"{cn}|Z{z:.0f}"
                m = (scan["labels"] == c) & (scan["zenith"] == z)
                near = (meta["labels"] == c) & (np.abs(meta["zenith"] - z) <= 5) & (meta["log10_E"] <= HIGH)
                row = {"n_corsika_fit": int(near.sum())}
                for k in METRICS + ("blob",):
                    v = scan[f"{s}.{k}"]
                    med = [float(np.nanmedian(v[m & (scan["log10_E"] == e)])) for e in SCAN_LOG_E]
                    row[k] = {"generated_median": med}
                    x, y = meta["log10_E"][near], cor[s][k][near]
                    ok = np.isfinite(y)
                    if k != "blob" and ok.sum() >= 20:
                        coef = np.polyfit(x[ok], y[ok], 1)
                        row[k]["corsika_fit"] = coef.tolist()
                        g_in = SCAN_LOG_E <= HIGH
                        gcoef = np.polyfit(SCAN_LOG_E[g_in], np.array(med)[g_in], 1)
                        row[k]["generated_fit_in_range"] = gcoef.tolist()
                        top = SCAN_LOG_E >= 6.5
                        row[k]["generated_minus_own_trend_6.5_to_7"] = float(np.mean(np.array(med)[top] - np.polyval(gcoef, SCAN_LOG_E[top])))
                        row[k]["generated_minus_corsika_trend_6.5_to_7"] = float(np.mean(np.array(med)[top] - np.polyval(coef, SCAN_LOG_E[top])))
                cap = sources.CAP[s]
                row["cap_rate"] = [float(np.mean(scan[f"{s}.n_points"][m & (scan["log10_E"] == e)] >= cap)) for e in SCAN_LOG_E]
                out[s][key] = row
    return out


# ---------------------------------------------------------------- figures

def fig_matched(res, key, ylabel, ref):
    fig, axes = plots.plt.subplots(len(SPECIES_NAMES), 4, figsize=(17, 10), sharex=True)
    centres = E_EDGES[:-1] + 0.125
    for r, s in enumerate(SPECIES_NAMES):
        for col, k in enumerate(("log10_energy", "mean_layer", "peak_width_m", "n_points")):
            ax = axes[r, col]
            for c, cn in CLASS.items():
                v = np.array([res[s][f"E{e:.2f}|{cn}|all_zenith"][k][key] for e in E_EDGES[:-1]])
                x = centres + (c - 0.5) * 0.03
                ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o-", ms=4, capsize=2,
                            color=f"C{c}", label=cn)
            ax.axhline(ref, color="0.5", lw=0.8)
            ax.set_title(f"{s}: {showers.LABEL[k]}", fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel)
            if r == len(SPECIES_NAMES) - 1:
                ax.set_xlabel("log10 E [GeV]")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_scan(trends, cor, meta, zenith=80.0):
    ks = ("log10_energy", "mean_layer", "peak_width_m", "n_points")
    fig, axes = plots.plt.subplots(len(SPECIES_NAMES), len(ks), figsize=(17, 10), sharex=True)
    centres = E_EDGES[:-1] + 0.125
    for r, s in enumerate(SPECIES_NAMES):
        for col, k in enumerate(ks):
            ax = axes[r, col]
            for c, cn in CLASS.items():
                row = trends[s][f"{cn}|Z{zenith:.0f}"]
                ax.plot(SCAN_LOG_E, row[k]["generated_median"], "-", color=f"C{c}", label=f"generated, {cn}")
                near = (meta["labels"] == c) & (np.abs(meta["zenith"] - zenith) <= 5)
                e_bin = np.digitize(meta["log10_E"], E_EDGES) - 1
                med = [np.nanmedian(cor[s][k][near & (e_bin == b)]) for b in range(len(E_EDGES) - 1)]
                ax.plot(centres, med, "o", color=f"C{c}", ms=5, mfc="none", label=f"CORSIKA, {cn}")
                if "corsika_fit" in row[k]:
                    ax.plot(SCAN_LOG_E, np.polyval(row[k]["corsika_fit"], SCAN_LOG_E), ":", color=f"C{c}", lw=1)
            ax.axvspan(HIGH, SCAN_LOG_E[-1], color="0.92", zorder=0)
            ax.set_title(f"{s}: {showers.LABEL[k]}, zenith {zenith:.0f}", fontsize=9)
            if r == len(SPECIES_NAMES) - 1:
                ax.set_xlabel("log10 E [GeV]")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_examples(examples, cor, gen, rep_of):
    fig, axes = plots.plt.subplots(len(examples), len(SPECIES_NAMES), figsize=(15, 3.6 * len(examples)), squeeze=False)
    layers = np.arange(cor[SPECIES_NAMES[0]]["profile"].shape[1])
    for r, (name, i) in enumerate(examples.items()):
        for col, s in enumerate(SPECIES_NAMES):
            ax = axes[r, col]
            for j in np.nonzero(rep_of == i)[0]:
                ax.plot(layers, gen[s]["profile"][j], color="0.75", lw=0.8)
            ax.plot(layers, gen[s]["profile"][i], color="C1", lw=1.5, label="generated (matched draw)")
            ax.plot(layers, cor[s]["profile"][i], color="k", lw=1.5, label="CORSIKA")
            ax.set_yscale("log")
            ax.set_title(f"{name}: {s} energy per layer", fontsize=9)
            ax.set_xlabel("layer")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- compare stage

def corsika_summaries(rows, log, chunk=512):
    parts = {s: [] for s in SPECIES_NAMES}
    n = len(rows)
    for lo in range(0, n, chunk):
        clouds, _ = sources.corsika("test", rows[lo:lo + chunk])
        for s in SPECIES_NAMES:
            parts[s].append(showers.describe(clouds[s]))
    log.info("described %d CORSIKA test showers", n)
    return {s: {k: np.concatenate([p[k] for p in parts[s]]) for k in parts[s][0]} for s in SPECIES_NAMES}


def stage_compare(out, figs, args, log):
    rng = np.random.default_rng(args.seed)
    g = np.load(os.path.join(out, "generated_match.npz"))
    scan = np.load(os.path.join(out, "generated_scan.npz"))
    meta = test_primaries(args.limit)
    n = len(meta["ids"])
    for k in ("ids", "energies", "directions", "labels", "rows"):
        if not np.array_equal(g[k][:n], meta[k]):
            raise ValueError(f"generated_match {k} differ from the CORSIKA test primaries")
    rep_of = g["repeat_of"]
    gen = {s: {k: g[f"{s}.{k}"] for k in showers.SCALARS + ("profile", "width")} for s in SPECIES_NAMES}
    gen_m = {s: {k: v[:n] for k, v in gen[s].items()} for s in SPECIES_NAMES}
    gen_r = {s: {k: v[n:] for k, v in gen[s].items()} for s in SPECIES_NAMES}
    cor = corsika_summaries(meta["rows"], log)
    np.savez(os.path.join(out, "corsika_test.npz"), **{f"{s}.{k}": v for s in cor for k, v in cor[s].items()})

    sets = masks(meta)
    res = {"binned": {s: compare_bins(cor[s], gen_m[s], sets, rng) for s in SPECIES_NAMES},
           "energy_response_slope": {s: response(cor[s], gen_m[s], meta, rng) for s in SPECIES_NAMES},
           "cap_rate": {s: {src: float(np.mean(d["n_points"][meta["log10_E"] <= HIGH] >= sources.CAP[s]))
                            for src, d in (("corsika", cor[s]), ("generated", gen_m[s]))} for s in SPECIES_NAMES},
           "blob_rate": {s: {src: float(np.nanmean(d["blob"])) for src, d in (("corsika", cor[s]), ("generated", gen_m[s]))}
                         for s in SPECIES_NAMES}}
    fixed_in, high = g["fixed_in"], g["fixed_high"]
    res["calibration"] = {}
    for s in SPECIES_NAMES:
        r_in = ranks(cor[s], gen_r[s], rep_of[n:], fixed_in)
        r_hi = ranks(cor[s], gen_r[s], rep_of[n:], high)
        res["calibration"][s] = {k: {"in_range": rank_summary(r_in[k]), "above_6.25_ranks": r_hi[k].tolist()} for k in METRICS}
        intrinsic = {k: float(np.nanmedian([width68(gen_r[s][k][rep_of[n:] == i]) for i in fixed_in])) for k in METRICS}
        res["calibration"][s]["generator_width68_at_fixed_primary_median"] = intrinsic
    res["high_energy_test_rows"] = [{"row": int(i), "log10_E": float(meta["log10_E"][i]), "zenith": float(meta["zenith"][i]),
                                     "class": CLASS[int(meta["labels"][i])]} for i in high]
    trends = scan_trends({k: scan[k] for k in scan.files}, cor, meta)
    res["scan"] = trends

    for s in SPECIES_NAMES:
        log.info("%s energy response slope (in range): %s; cap rate %s; blob rate %s", s, res["energy_response_slope"][s],
                 res["cap_rate"][s], res["blob_rate"][s])
        for k in METRICS:
            log.info("  %-14s calibration in range %s | above 6.25 ranks %s", k, res["calibration"][s][k]["in_range"],
                     res["calibration"][s][k]["above_6.25_ranks"])
        for b in E_EDGES[:-1]:
            for cn in CLASS.values():
                row = res["binned"][s][f"E{b:.2f}|{cn}|all_zenith"]
                log.info("  E %.2f %-8s n %4d  " + "  ".join(f"{k} d %.3g r %.3g" for k in METRICS), b, cn, row["n"],
                         *[v for k in METRICS for v in (row[k]["paired_median_diff"][0], row[k]["width_ratio"][0])])
        for key, row in trends[s].items():
            log.info("  scan %s: " + "  ".join(f"{k} own %.3g cors %.3g" for k in ("log10_energy", "mean_layer", "peak_width_m")),
                     key, *[row[k].get(f"generated_minus_{w}_trend_6.5_to_7", np.nan) for k in ("log10_energy", "mean_layer", "peak_width_m")
                            for w in ("own", "corsika")])

    order = np.argsort(meta["log10_E"][fixed_in])
    examples = {"low energy (fixed)": int(fixed_in[order[0]]), "6.0 to 6.25 (fixed)": int(fixed_in[order[-1]]),
                "highest-energy test shower": int(high[np.argmax(meta["log10_E"][high])])}
    for name, i in examples.items():
        log.info("example %s: test row %d id %d, log10 E %.3f, zenith %.1f, azimuth %.1f, class %s", name, i, meta["ids"][i],
                 meta["log10_E"][i], meta["zenith"][i], meta["azimuth"][i], CLASS[int(meta["labels"][i])])
        for s in SPECIES_NAMES:
            reps = {k: np.round(np.nanpercentile(gen_r[s][k][rep_of[n:] == i], [16, 50, 84]), 3).tolist() for k in METRICS}
            log.info("  %-8s CORSIKA %s", s, {k: round(float(cor[s][k][i]), 3) for k in METRICS})
            log.info("  %-8s matched %s", s, {k: round(float(gen_m[s][k][i]), 3) for k in METRICS})
            log.info("  %-8s repeats q16/50/84 %s", s, reps)
    res["examples"] = examples

    plots.save(fig_matched(res["binned"], "paired_median_diff", "generated - CORSIKA (paired median)", 0.0), figs, "matched_median_diff", log)
    plots.save(fig_matched(res["binned"], "width_ratio", "16-84 width, generated / CORSIKA", 1.0), figs, "matched_width_ratio", log)
    plots.save(fig_scan(trends, cor, meta), figs, "energy_scan", log)
    plots.save(fig_examples(examples, cor, gen, rep_of), figs, "examples", log)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["generate_match", "generate_scan", "compare"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke run: first LIMIT test rows plus those above 6.25; 2 scan repeats")
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER + ("_smoke" if args.limit else ""), args.stage), args)
    shared = os.path.dirname(out)                  # data shared between stages; logs, results and figures per stage
    stage = {"generate_match": stage_generate_match, "generate_scan": stage_generate_scan, "compare": stage_compare}[args.stage]
    logs.save(out, stage(shared, out, args, log), args, log)


if __name__ == "__main__":
    main()
