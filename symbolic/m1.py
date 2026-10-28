"""Profile fits of the energy per plane, every candidate shape, on a training slice, the CORSIKA test file and the
neural generator's showers. Picks the shape and writes fits_train.npz / fits_test.npz, which later steps read.

Stages: fit (CPU), pcfm (PointCountFM cell totals against the cap; diagnostic).
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis_core import logs, plots
from data_lane import conditional, sources
from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere, families as fam, planes, structure

FOLDER = "m1"
N_TRAIN = 10000
CHUNK = 64
CLASS = {0: "EM", 1: "hadronic"}
E_EDGES = conditional.E_EDGES
GEN_DIR = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/analysis_run12/data_lane/conditional"
N_BOOT = 200
N_PCFM_SEEDS = 3
DATASETS = ("train", "test", "generated")


# ---------------------------------------------------------------- data

def _n_rows(split):
    with h5py.File(sources.corsika_path(SPECIES_NAMES[0], split), "r") as f:
        return len(f["shower_ids"])


def rows_for(split, limit, seed):
    """Training: a random slice of N_TRAIN rows. Test: every row (a random `limit` rows in a smoke run)."""
    n = _n_rows(split)
    if split == "test" and not limit:
        return np.arange(n)
    rng = np.random.default_rng(seed + (split == "test"))
    return np.sort(rng.choice(n, limit or N_TRAIN, replace=False))


def generated(rows, test_meta, log):
    """Per-plane energies of the neural generator's showers at the given test rows."""
    g = np.load(os.path.join(GEN_DIR, "generated_match.npz"))
    n_matched = int(np.sum(g["repeat_of"] == -1))
    if n_matched != _n_rows("test") or not np.array_equal(g["rows"][:n_matched], np.arange(n_matched)):
        raise ValueError("generated_match.npz does not hold one draw per test row in order")
    if not (np.array_equal(g["ids"][rows], test_meta["ids"]) and np.array_equal(g["labels"][rows], test_meta["labels"])
            and np.allclose(g["energies"][rows, 0], test_meta["energies"], rtol=1e-6)
            and np.allclose(g["directions"][rows], test_meta["directions"], atol=1e-6)):
        raise ValueError("generated primaries differ from the CORSIKA test primaries")
    out = {}
    for s in SPECIES_NAMES:
        clouds = torch.load(os.path.join(GEN_DIR, f"generated_match_{s}.pt"), mmap=True)
        if len(clouds) != n_matched or clouds.shape[1] != sources.CAP[s]:
            raise ValueError(f"{s}: generated clouds have shape {tuple(clouds.shape)}")
        parts = [planes.from_clouds(clouds[torch.as_tensor(rows[lo:lo + 512])]) for lo in range(0, len(rows), 512)]
        out[s] = {"E": np.concatenate([p[0] for p in parts]), "N": np.concatenate([p[1] for p in parts])}
        log.info("generated %s: %d showers, points per shower median %.0f", s, len(rows), np.median(out[s]["N"].sum(1)))
    return out


def gate_profiles(data, log):
    """Gate: per-plane energies must equal the ones stored by data_lane/conditional.py for the same rows."""
    rows = data["test"]["meta"]["rows"]
    ref = {"test": np.load(os.path.join(GEN_DIR, "corsika_test.npz")), "generated": np.load(os.path.join(GEN_DIR, "generated_match.npz"))}
    for name, r in ref.items():
        for s in SPECIES_NAMES:
            if not np.allclose(data[name]["species"][s]["E"], r[f"{s}.profile"][rows], rtol=1e-7, atol=0):
                raise RuntimeError(f"gate: {name} {s} per-plane energies differ from Step 3's stored profiles")
    log.info("gate: per-plane energies equal Step 3's stored profiles (test and generated, %d rows)", len(rows))


def gate_cap_rates(s, cor, gen, log10_e, log):
    """Gate: cap rates must equal the ones stored by data_lane/conditional.py (full test set only)."""
    with open(os.path.join(GEN_DIR, "compare", "results.json")) as f:
        ref = json.load(f)["cap_rate"][s]
    m = log10_e <= conditional.HIGH
    got = {"corsika": float(np.mean(cor[m] >= sources.CAP[s])), "generated": float(np.mean(gen[m] >= sources.CAP[s]))}
    if any(abs(got[k] - ref[k]) > 1e-12 for k in got):
        raise RuntimeError(f"gate: {s} cap rates {got} differ from Step 3 {ref}")
    log.info("gate: %s cap rates reproduce Step 3: %s", s, got)


def load(limit, seed, log):
    data = {}
    for split in ("train", "test"):
        rows = rows_for(split, limit, seed)
        d, meta = planes.from_corsika(split, rows)
        data[split] = {"species": d, "meta": meta}
        log.info("%s: %d rows, EM %d hadronic %d, log10 E %.2f-%.2f", split, len(rows), np.sum(meta["labels"] == 0),
                 np.sum(meta["labels"] == 1), meta["log10_E"].min(), meta["log10_E"].max())
    data["generated"] = {"species": generated(data["test"]["meta"]["rows"], data["test"]["meta"], log),
                         "meta": data["test"]["meta"]}
    gate_profiles(data, log)
    for name, d in data.items():
        d["X"] = atmosphere.plane_depths(d["meta"]["zenith"], N_PLANES)
        for s in SPECIES_NAMES:
            E = d["species"][s]["E"]
            d["species"][s]["Y"] = np.where(E > 0, np.log(np.where(E > 0, E, 1.0)), np.nan)
        log.info("%s: plane depths %.0f-%.0f g/cm2 (first plane), %.0f-%.0f (last)", name, d["X"][:, 0].min(),
                 d["X"][:, 0].max(), d["X"][:, -1].min(), d["X"][:, -1].max())
    return data


# ---------------------------------------------------------------- fits

def fit(pool, name, X, Y, extra=None):
    blocks = [(name, X[i:i + CHUNK], Y[i:i + CHUNK], None if extra is None else extra[i:i + CHUNK])
              for i in range(0, len(Y), CHUNK)]
    parts = pool.map(fam.fit_block, blocks)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def inside(f, X):
    return np.isfinite(f["xmax"]) & (f["xmax"] >= X[:, 0]) & (f["xmax"] <= X[:, -1])


def fit_everything(data, pool, log):
    """Every family on every dataset."""
    fits = {d: {s: {} for s in SPECIES_NAMES} for d in DATASETS}
    for d in DATASETS:
        for s in SPECIES_NAMES:
            for name in fam.FOR_SPECIES[s]:
                if name != "gh3":
                    fits[d][s][name] = fit(pool, name, data[d]["X"], data[d]["species"][s]["Y"])
    lam = {}
    tr = data["train"]
    for s in SPECIES_NAMES:
        f = fits["train"][s]["gh4"]
        lam[s] = {}
        for c in CLASS:
            m = inside(f, tr["X"]) & (tr["meta"]["labels"] == c)
            lam[s][c] = float(np.median(f["theta"][m, 2]))
            log.info("%s %s: gh3 lambda = %.1f g/cm2 (median of %d gh4 fits peaking inside)", s, CLASS[c], lam[s][c], m.sum())
    for d in DATASETS:
        for s in SPECIES_NAMES:
            extra = np.array([lam[s][c] for c in data[d]["meta"]["labels"]])
            fits[d][s]["gh3"] = fit(pool, "gh3", data[d]["X"], data[d]["species"][s]["Y"], extra)
    for d in DATASETS:
        for s in SPECIES_NAMES:
            for name, f in fits[d][s].items():
                ok = np.isfinite(f["ln_n"])
                log.info("fit %-9s %-8s %-8s fitted %5d/%d  peak inside %.3f  rms resid %.3f", d, s, name, ok.sum(), len(ok),
                         inside(f, data[d]["X"])[ok].mean(), np.sqrt(np.nanmedian(f["mse"])))
    return fits, lam


# ---------------------------------------------------------------- analysis

def public(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def structure_all(data, fits, rng, log):
    out, mats = {}, {}
    for d in DATASETS:
        meta, X = data[d]["meta"], data[d]["X"]
        for s in SPECIES_NAMES:
            Y = data[d]["species"][s]["Y"]
            for c, cn in CLASS.items():
                m = meta["labels"] == c
                for name, f in fits[d][s].items():
                    Yhat = fam.evaluate(name, {k: v[m] for k, v in f.items()}, X[m])
                    r = structure.summarize(Y[m], Yhat, meta["log10_E"][m], meta["cos_zenith"][m],
                                            fam.FAMILIES[name].n_par, rng)
                    out[f"{d}|{s}|{cn}|{name}"] = public(r)
                    mats[f"{d}|{s}|{cn}|{name}"] = (r["_C"], r["_Cm"], r["planes"])
                    log.info("struct %-9s %-8s %-8s %-8s n %5d  R2cond median %.3f  planes>=%.2f %.2f  cov %.3f [%.3f, %.3f]"
                             "  rank-%d ref %.3f  floor %d", d, s, cn, name, r["n"], r["r2_cond_median"], structure.R2_GO,
                             r["frac_planes_r2_go"], *r["cov_explained"], fam.FAMILIES[name].n_par,
                             r["cov_explained_rank_k_reference"], r["n_model_floor"])
    return out, mats


def select(struct, log):
    """Best family per species and class on the training slice."""
    sel = {}
    for s in SPECIES_NAMES:
        for cn in CLASS.values():
            rows = {n: struct[f"train|{s}|{cn}|{n}"] for n in fam.FOR_SPECIES[s]}
            passing = [n for n, r in rows.items() if r["go_r2"]]
            best = (max(passing, key=lambda n: rows[n]["cov_explained"][0]) if passing
                    else max(rows, key=lambda n: rows[n]["r2_cond_median"]))
            r = rows[best]
            sel[f"{s}|{cn}"] = {"family": best, "go_r2": r["go_r2"], "go_cov": r["go_cov"], "go": r["go_r2"] and r["go_cov"]}
            log.info("SELECT %-8s %-8s -> %-8s  go_r2 %s  go_cov %s", s, cn, best, r["go_r2"], r["go_cov"])
    return sel


def boot(f, *arrays, rng, n=N_BOOT):
    k = len(arrays[0])
    vals = [f(*(a[i] for a in arrays)) for i in (rng.integers(0, k, k) for _ in range(n))]
    return [float(f(*arrays))] + np.nanpercentile(vals, [16, 84]).tolist()


def physics(data, fits, sel, rng):
    """Physics checks of the fits: elongation rate, Xmax spread, Nmax energy slope."""
    tr, out = data["train"], {}
    meta, X = tr["meta"], tr["X"]
    l, u = meta["log10_E"] - 5.5, meta["cos_zenith"]
    D = np.stack([np.ones_like(l), l, u], 1)
    coef = lambda D_, y: np.linalg.lstsq(D_, y, rcond=None)[0]
    e_bin = np.digitize(meta["log10_E"], E_EDGES) - 1
    for s in SPECIES_NAMES:
        for c, cn in CLASS.items():
            for name in sorted({"gh4", sel[f"{s}|{cn}"]["family"]}):
                f = fits["train"][s][name]
                ok = np.isfinite(f["ln_n"]) & (meta["labels"] == c)
                ins = ok & inside(f, X)
                r = {"frac_inside": float(ins.sum() / max(ok.sum(), 1))}
                for tag, m in (("inside", ins), ("all", ok)):
                    if m.sum() < 30:
                        continue
                    r[f"elongation_per_decade_{tag}"] = boot(lambda d, y: coef(d, y)[1], D[m], f["xmax"][m], rng=rng)
                    r[f"xmax_per_unit_cos_zenith_{tag}"] = boot(lambda d, y: coef(d, y)[2], D[m], f["xmax"][m], rng=rng)
                    r[f"log10_nmax_energy_slope_{tag}"] = boot(lambda d, y: coef(d, y)[1], D[m], f["ln_nmax"][m] / np.log(10), rng=rng)
                r["by_energy_bin"] = {}
                for b in range(len(E_EDGES) - 1):
                    mb, ib = ok & (e_bin == b), ins & (e_bin == b)
                    r["by_energy_bin"][f"{E_EDGES[b]:.2f}"] = {
                        "n": int(mb.sum()), "frac_inside": float(ib.sum() / max(mb.sum(), 1)),
                        "xmax_median_inside": float(np.median(f["xmax"][ib])) if ib.sum() >= 10 else None,
                        "xmax_sd_inside": float(np.std(f["xmax"][ib])) if ib.sum() >= 10 else None}
                out[f"{s}|{cn}|{name}"] = r
    return out


def generator_vs_corsika(data, fits, sel, rng):
    """Neural generator against CORSIKA at the same test primaries."""
    meta, X = data["test"]["meta"], data["test"]["X"]
    e_bin = np.digitize(meta["log10_E"], E_EDGES) - 1
    out = {}
    for s in SPECIES_NAMES:
        for c, cn in CLASS.items():
            m = meta["labels"] == c
            Ya, Yb = data["test"]["species"][s]["Y"][m], data["generated"]["species"][s]["Y"][m]
            r = structure.compare_covariance(Ya, Yb, meta["log10_E"][m], meta["cos_zenith"][m], rng)
            name = sel[f"{s}|{cn}"]["family"]
            fa, fb = fits["test"][s][name], fits["generated"][s][name]
            r["family"] = name
            r["frac_inside"] = {"corsika": float(inside(fa, X)[m].mean()), "generated": float(inside(fb, X)[m].mean())}
            r["by_energy_bin"] = {}
            for b in range(len(E_EDGES) - 1):
                mb = m & (e_bin == b) & np.isfinite(fa["xmax"]) & np.isfinite(fb["xmax"])
                if mb.sum() < 20:
                    continue
                r["by_energy_bin"][f"{E_EDGES[b]:.2f}"] = {
                    "n": int(mb.sum()),
                    "xmax_paired_median_diff": boot(lambda a, b_: np.median(b_ - a), fa["xmax"][mb], fb["xmax"][mb], rng=rng),
                    "xmax_sd_ratio": boot(lambda a, b_: np.std(b_) / np.std(a), fa["xmax"][mb], fb["xmax"][mb], rng=rng),
                    "ln_nmax_paired_median_diff": boot(lambda a, b_: np.median(b_ - a), fa["ln_nmax"][mb], fb["ln_nmax"][mb], rng=rng)}
            out[f"{s}|{cn}"] = r
    return out


# ---------------------------------------------------------------- figures

COLORS = {"gh4": "C0", "gh4lr": "C5", "gh4lr_signed": "C6", "gh3": "C1", "age3": "C2", "greisen2": "C3", "rise3": "C4"}


def fig_examples(data, fits, rng):
    tr = data["train"]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.5), sharex=True)
    for i, s in enumerate(SPECIES_NAMES):
        picks = [j for c in CLASS for j in rng.choice(np.nonzero(tr["meta"]["labels"] == c)[0], 2, replace=False)]
        for ax, j in zip(axes[i], picks):
            X, Y = tr["X"][j], tr["species"][s]["Y"][j]
            ax.plot(X, Y / np.log(10), "k.", ms=5)
            for name, f in fits["train"][s].items():
                yh = fam.evaluate(name, {k: v[j:j + 1] for k, v in f.items()}, X[None])[0]
                ax.plot(X, yh / np.log(10), color=COLORS[name], lw=1.2, label=name)
            ax.set_ylim(np.nanmin(Y) / np.log(10) - 0.5, np.nanmax(Y) / np.log(10) + 0.5)
            ax.set_title(f"{s}, {CLASS[tr['meta']['labels'][j]]}, log10 E {tr['meta']['log10_E'][j]:.2f}, "
                         f"zen {tr['meta']['zenith'][j]:.0f}", fontsize=8)
        axes[i, 0].set_ylabel("log10 E_k [GeV]")
        axes[i, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("slant depth [g/cm2]")
    return fig


def fig_r2(struct):
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True, sharey=True)
    for i, s in enumerate(SPECIES_NAMES):
        for j, cn in enumerate(CLASS.values()):
            ax = axes[i, j]
            for name in fam.FOR_SPECIES[s]:
                r = struct[f"train|{s}|{cn}|{name}"]
                ax.plot(r["planes"], [r["r2_cond"][k] for k in r["planes"]], "-o", ms=3, color=COLORS[name], label=name)
            ax.axhline(structure.R2_GO, color="0.5", ls="--", lw=1)
            ax.set_ylim(-0.5, 1.02)
            ax.set_title(f"{s}, {cn}: per-plane R2 at fixed energy and zenith", fontsize=9)
        axes[i, 0].set_ylabel("R2 (conditional)")
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("plane")
    return fig


def fig_cov(mats, sel):
    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    for i, s in enumerate(SPECIES_NAMES):
        for j, cn in enumerate(CLASS.values()):
            name = sel[f"{s}|{cn}"]["family"]
            C, Cm, _ = mats[f"train|{s}|{cn}|{name}"]
            off = ~np.eye(len(C), dtype=bool)
            ax = axes[i, j]
            ax.plot(C[off], Cm[off], ".", ms=2, color=COLORS[name])
            lim = np.nanmax(np.abs(np.concatenate([C[off], Cm[off]])))
            ax.plot([-lim, lim], [-lim, lim], color="0.5", lw=1)
            ax.set_aspect("equal")
            ax.set_title(f"{s}, {cn}, {name}: off-diagonal covariance", fontsize=9)
            ax.set_xlabel("CORSIKA")
            ax.set_ylabel("rebuilt from fitted profile")
    return fig


def fig_xmax(data, fits, sel):
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    centres = (E_EDGES[:-1] + E_EDGES[1:]) / 2
    for i, s in enumerate(SPECIES_NAMES):
        for j, (c, cn) in enumerate(CLASS.items()):
            name, ax = sel[f"{s}|{cn}"]["family"], axes[i, j]
            for d, style in zip(DATASETS, ("-", "--", ":")):
                f, meta = fits[d][s][name], data[d]["meta"]
                ok = np.isfinite(f["xmax"]) & (meta["labels"] == c)
                b = np.digitize(meta["log10_E"], E_EDGES) - 1
                q = np.array([np.percentile(f["xmax"][ok & (b == k)], [16, 50, 84]) if np.sum(ok & (b == k)) >= 10
                              else [np.nan] * 3 for k in range(len(centres))])
                ax.plot(centres, q[:, 1], style, color="k", label=d)
                if d == "train":
                    ax.fill_between(centres, q[:, 0], q[:, 2], color="0.85")
            ax.set_title(f"{s}, {cn}, {name}: fitted Xmax", fontsize=9)
        axes[i, 0].set_ylabel("Xmax [g/cm2]")
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("log10 E [GeV]")
    return fig


# ---------------------------------------------------------------- stages

def stage_fit(out, args, log):
    rng = np.random.default_rng(args.seed)
    data = load(args.limit, args.seed, log)
    with Pool(args.workers) as pool:
        fits, lam = fit_everything(data, pool, log)
    struct, mats = structure_all(data, fits, rng, log)
    sel = select(struct, log)
    phys = physics(data, fits, sel, rng)
    for k, r in phys.items():
        log.info("phys %-28s inside %.3f  ER/decade %s  Xmax per cos(zen) %s  log10 Nmax slope %s", k, r["frac_inside"],
                 np.round(r.get("elongation_per_decade_inside", [np.nan] * 3), 1).tolist(),
                 np.round(r.get("xmax_per_unit_cos_zenith_inside", [np.nan] * 3), 1).tolist(),
                 np.round(r.get("log10_nmax_energy_slope_all", [np.nan] * 3), 3).tolist())
    gvc = generator_vs_corsika(data, fits, sel, rng)
    for k, r in gvc.items():
        log.info("gen-vs-corsika %-17s cov distance %s  inside cors %.3f gen %.3f", k, np.round(r["cov_distance"], 3).tolist(),
                 r["frac_inside"]["corsika"], r["frac_inside"]["generated"])
        for b, v in r["by_energy_bin"].items():
            log.info("    E %s n %4d  dXmax %s  sd ratio %s  d ln Nmax %s", b, v["n"], np.round(v["xmax_paired_median_diff"], 1).tolist(),
                     np.round(v["xmax_sd_ratio"], 2).tolist(), np.round(v["ln_nmax_paired_median_diff"], 3).tolist())
    for d in DATASETS:
        arrays = {**{f"meta.{k}": v for k, v in data[d]["meta"].items()}, "X": data[d]["X"]}
        for s in SPECIES_NAMES:
            arrays.update({f"{s}.{k}": v for k, v in data[d]["species"][s].items()})
            for name, f in fits[d][s].items():
                arrays.update({f"{s}.{name}.{k}": v for k, v in f.items()})
        path = os.path.join(out, f"fits_{d}.npz")
        np.savez(path, **arrays)
        log.info("wrote %s (%d bytes)", path, os.path.getsize(path))
    plots.save(fig_examples(data, fits, rng), out, "profile_examples", log)
    plots.save(fig_r2(struct), out, "plane_r2", log)
    plots.save(fig_cov(mats, sel), out, "covariance", log)
    plots.save(fig_xmax(data, fits, sel), out, "xmax_vs_energy", log)
    return {"n_train": len(data["train"]["meta"]["rows"]), "n_test": len(data["test"]["meta"]["rows"]),
            "gh3_lambda": {s: {CLASS[c]: v for c, v in d.items()} for s, d in lam.items()},
            "thresholds": {"r2": structure.R2_GO, "plane_frac": structure.PLANE_FRAC_GO, "cov": structure.COV_GO},
            "selected": sel, "structure": struct, "physics": phys, "generator_vs_corsika": gvc}


def totals_summary(total, cap, labels):
    r = {}
    for tag, m in (("all", np.ones(len(total), bool)), *((cn, labels == c) for c, cn in CLASS.items())):
        t = total[m] / cap
        r[tag] = {"at_or_above_cap": float(np.mean(t >= 1.0)), "above_1.1_cap": float(np.mean(t > 1.1)),
                  "within_1pct_of_cap": float(np.mean(np.abs(t - 1.0) <= 0.01)), "below_0.9_cap": float(np.mean(t < 0.9)),
                  "total_over_cap_q16_50_84": np.percentile(t, [16, 50, 84]).round(4).tolist()}
    return r


def stage_pcfm(out, args, log):
    meta = conditional.test_primaries(0)
    rows = rows_for("test", args.limit, args.seed)
    E, D, L = (torch.as_tensor(meta[k][rows]) for k in ("energies", "directions", "labels"))
    labels = meta["labels"][rows]
    s0 = sources.stage0("fork")
    if (s0.MAX_CLIP_FRAC, s0.MAX_PCFM_RETRIES) != (0.10, 10):
        raise RuntimeError(f"pipeline re-roll settings changed: {s0.MAX_CLIP_FRAC}, {s0.MAX_PCFM_RETRIES}")
    res, hist = {}, {}
    for k, s in enumerate(SPECIES_NAMES):
        cap = sources.CAP[s]
        with h5py.File(sources.corsika_path(s, "test"), "r") as f:
            cor = f["num_points"][...][rows].astype(np.float64)
        g = torch.load(os.path.join(GEN_DIR, f"generated_match_{s}.pt"), mmap=True)
        gen = np.concatenate([(g[torch.as_tensor(rows[lo:lo + 512])][..., 3] > 0).sum(1).numpy()
                              for lo in range(0, len(rows), 512)]).astype(np.float64)
        if not args.limit:
            gate_cap_rates(s, cor, gen, meta["log10_E"][rows], log)
        res[s] = {"corsika": totals_summary(cor, cap, labels), "generated_delivered": totals_summary(gen, cap, labels)}
        pcfm = os.path.join(sources.STAGED, s, "pcfm_compiled.pt")
        for seed in range(N_PCFM_SEEDS):
            torch.manual_seed(seed * 1000 + k)
            raw = s0.run_point_count_fm(model_path=pcfm, energies=E, directions=D, labels=L)
            if raw.shape != (len(rows), N_PLANES) or (raw < 0).any():
                raise ValueError(f"{s}: PointCountFM returned shape {tuple(raw.shape)}")
            rer = s0.resample_overclip(pcfm, E, D, L, raw.clone(), cap=cap)
            t_raw, t_rer = raw.sum(1).double().numpy(), rer.sum(1).double().numpy()
            below = (np.clip(t_raw - cap, 0, None) / np.maximum(t_raw, 1)) <= s0.MAX_CLIP_FRAC
            if not torch.equal(raw[torch.as_tensor(below)], rer[torch.as_tensor(below)]):
                raise RuntimeError(f"{s}: the re-roll changed showers under its threshold")
            res[s][f"pcfm_raw_seed{seed}"] = totals_summary(t_raw, cap, labels)
            res[s][f"pcfm_rerolled_seed{seed}"] = totals_summary(t_rer, cap, labels)
            res[s][f"pcfm_rerolled_seed{seed}"]["n_rerolled"] = int((~below).sum())
            if seed == 0:
                hist[s] = {"CORSIKA": cor, "PointCountFM": t_raw, "PointCountFM + re-roll": t_rer,
                           "AllShowers delivered": gen}
        for src, r in res[s].items():
            log.info("%-8s %-22s at cap %.3f  >1.1 cap %.3f  within 1%% %.3f  <0.9 cap %.3f  total/cap q16/50/84 %s", s, src,
                         r["all"]["at_or_above_cap"], r["all"]["above_1.1_cap"], r["all"]["within_1pct_of_cap"],
                         r["all"]["below_0.9_cap"], r["all"]["total_over_cap_q16_50_84"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    bins = np.linspace(0, 1.6, 81)
    for ax, s in zip(axes, SPECIES_NAMES):
        for (src, t), st in zip(hist[s].items(), ("-", "--", "-.", ":")):
            ax.hist(t / sources.CAP[s], bins=bins, histtype="step", ls=st, color="k", density=True, label=src)
        ax.set_yscale("log")
        ax.set_title(f"{s}: total kept cells / cap", fontsize=9)
        ax.set_xlabel("total / cap")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7)
    plots.save(fig, out, "cap_totals", log)
    return {"n_rows": len(rows), "re_roll": {"max_clip_frac": s0.MAX_CLIP_FRAC, "max_retries": s0.MAX_PCFM_RETRIES},
            "counts": res}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["fit", "pcfm"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke run: LIMIT training and LIMIT test rows")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER + ("_smoke" if args.limit else ""), args.stage), args)
    stage = {"fit": stage_fit, "pcfm": stage_pcfm}[args.stage]
    logs.save(out, stage(out, args, log), args, log)


if __name__ == "__main__":
    main()
