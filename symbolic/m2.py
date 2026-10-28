"""The chosen profile shape (Gaisser-Hillas in width and asymmetry) fitted to every CORSIKA training shower.

Stages: part (CPU array; parts already on disk are skipped), merge (checks against m1's fits), latent (summaries of
the fitted numbers' distribution; diagnostic).
"""
import argparse
import os
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib.pyplot as plt
import numpy as np

from analysis_core import logs, plots
from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere, latent, m1, planes

FOLDER = "m2"
NAME = "gh4lr_signed"
SIZES = [len(latent.NAMES[NAME])] * len(SPECIES_NAMES)          # latent numbers per species
MIN_BIN_ROWS = 60
FIT_RTOL = 1e-6
MAX_FIT_MISMATCH = 1e-3       # share of m1 rows allowed to land in another local minimum
LOG10_E_MAX = 6.25            # the training showers end here
E_EDGES = m1.E_EDGES
N_COS_CELLS = 6
FIG_E = np.arange(5.0, 6.2501, 0.125)


def part_rows(n, n_parts, i):
    """Rows of part i when n training rows are split into n_parts contiguous slices."""
    return np.array_split(np.arange(n), n_parts)[i]


def check_cover(rows, n):
    """The concatenated part rows must be exactly 0..n-1 in order."""
    if len(rows) != n or not np.array_equal(rows, np.arange(n)):
        raise ValueError(f"parts cover {len(np.unique(rows))} distinct of {n} rows ({len(rows)} entries)")


def part_path(out, i, n_parts):
    return os.path.join(out, "parts", f"part_{i:04d}_of_{n_parts:04d}.npz")


def fit_part(rows, workers):
    """Per-plane data and profile fits of the given training rows, as a flat dict of arrays."""
    d, meta = planes.from_corsika("train", rows)
    X = atmosphere.plane_depths(meta["zenith"], N_PLANES)
    arrays = {**{f"meta.{k}": v for k, v in meta.items()}, "X": X}
    with Pool(workers) as pool:
        for s in SPECIES_NAMES:
            E = d[s]["E"]
            Y = np.where(E > 0, np.log(np.where(E > 0, E, 1.0)), np.nan)
            f = m1.fit(pool, NAME, X, Y)
            arrays.update({f"{s}.{k}": v for k, v in d[s].items()})
            arrays.update({f"{s}.{NAME}.{k}": v for k, v in f.items()})
    return arrays


def stage_part(out, args, log):
    n = args.limit or m1._n_rows("train")
    rows = part_rows(n, args.n_parts, args.part)
    path = part_path(out, args.part, args.n_parts)
    if os.path.exists(path):
        with np.load(path) as z:
            if np.array_equal(z["meta.rows"], rows) and str(z["family"]) == NAME:
                log.info("part %d already done (%d rows), skipped", args.part, len(rows))
                return {"part": args.part, "rows": len(rows), "skipped": True}
        raise ValueError(f"{path} exists with other rows or another family")
    arrays = {**fit_part(rows, args.workers), "family": np.array(NAME)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    log.info("part %d: rows %d-%d, wrote %s (%d bytes)", args.part, rows[0], rows[-1], path, os.path.getsize(path))
    return {"part": args.part, "rows": len(rows), "skipped": False}


def gate_m1(merged, m1_fits, log):
    """Gate: on m1's training slice the energies match exactly and the fits to FIT_RTOL."""
    z = np.load(m1_fits)
    r = z["meta.rows"]
    r = r[r < len(merged["meta.rows"])]
    i = np.searchsorted(z["meta.rows"], r)
    res = {"n_rows": int(len(r))}
    for s in SPECIES_NAMES:
        if not np.array_equal(merged[f"{s}.E"][r], z[f"{s}.E"][i]):
            raise ValueError(f"{s}: per-plane energies differ from M1 on its training slice")
        a = np.column_stack([merged[f"{s}.{NAME}.ln_n"][r], merged[f"{s}.{NAME}.theta"][r]])
        b = np.column_stack([z[f"{s}.{NAME}.ln_n"][i], z[f"{s}.{NAME}.theta"][i]])
        same = np.all(np.isclose(a, b, rtol=FIT_RTOL, atol=0, equal_nan=True), 1)
        frac = float(1 - same.mean())
        res[s] = {"frac_fit_mismatch": frac, "n_mismatch": int((~same).sum())}
        log.info("gate M1 %-8s: energies equal on %d rows; fits differ beyond %.0e on %d rows (%.4f)", s, len(r),
                 FIT_RTOL, (~same).sum(), frac)
        if frac > MAX_FIT_MISMATCH:
            raise ValueError(f"{s}: {frac:.4f} of M1 rows fit differently")
    return res


def stage_merge(out, args, log):
    n = args.limit or m1._n_rows("train")
    paths = [part_path(out, i, args.n_parts) for i in range(args.n_parts)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise ValueError(f"{len(missing)} parts missing, first {missing[0]}")
    parts = [dict(np.load(p)) for p in paths]
    if any(str(p.pop("family")) != NAME for p in parts):
        raise ValueError(f"a part was fitted with another family than {NAME}")
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    merged["family"] = np.array(NAME)
    check_cover(merged["meta.rows"], n)
    log.info("merged %d parts: %d rows, EM %d hadronic %d", len(parts), n, np.sum(merged["meta.labels"] == 0),
             np.sum(merged["meta.labels"] == 1))
    gate = gate_m1(merged, os.path.join(args.out, m1.FOLDER, "fit", "fits_train.npz"), log)
    summary = {}
    X = merged["X"]
    for s in SPECIES_NAMES:
        f = {k: merged[f"{s}.{NAME}.{k}"] for k in ("ln_n", "theta", "mse", "xmax", "ln_nmax")}
        for c, cn in m1.CLASS.items():
            m = merged["meta.labels"] == c
            ok = np.isfinite(f["ln_n"]) & m
            ins = ok & m1.inside(f, X)
            if ins.sum() == 0:      # the training file is sorted by class, so a smoke slice can hold one class only
                log.info("%-8s %-8s n %6d  no fitted showers peaking inside, skipped", s, cn, m.sum())
                continue
            summary[f"{s}|{cn}"] = r = {
                "n": int(m.sum()), "frac_fitted": float(ok.sum() / m.sum()), "frac_peak_inside": float(ins.sum() / ok.sum()),
                "rms_resid_median": float(np.sqrt(np.nanmedian(f["mse"][ok]))),
                "xmax_16_50_84": np.percentile(f["xmax"][ins], [16, 50, 84]).round(1).tolist(),
                "log10_nmax_16_50_84": np.percentile(f["ln_nmax"][ok] / np.log(10), [16, 50, 84]).round(3).tolist()}
            log.info("%-8s %-8s n %6d  fitted %.4f  peak inside %.3f  rms resid %.3f  Xmax %s  log10 Nmax %s", s, cn,
                     r["n"], r["frac_fitted"], r["frac_peak_inside"], r["rms_resid_median"], r["xmax_16_50_84"],
                     r["log10_nmax_16_50_84"])
    path = os.path.join(out, "fits_train_full.npz")
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **merged)
    os.replace(tmp, path)
    log.info("wrote %s (%d bytes)", path, os.path.getsize(path))
    return {"n_rows": n, "n_parts": args.n_parts, "gate_m1": gate, "summary": summary}


def load_latent(path, log):
    """Profile numbers and flags per species from a fit file. Clean = fitted, off the bounds, peak within the planes
    (outside them Xmax is not determined)."""
    z = np.load(path)
    keep = z["meta.log10_E"] <= LOG10_E_MAX
    meta = {k: z[f"meta.{k}"][keep] for k in ("log10_E", "cos_zenith", "labels")}
    X = z["X"][keep]
    out = {"meta": meta}
    rng = np.random.default_rng(0)
    for s in SPECIES_NAMES:
        E = z[f"{s}.E"][keep]
        Y = np.where(E > 0, np.log(np.where(E > 0, E, 1.0)), np.nan)
        f = {k: z[f"{s}.{NAME}.{k}"][keep] for k in ("ln_n", "theta", "ln_nmax")}
        fl = latent.flags(f, X, Y)
        out[s] = {"Z": latent.to_latent(f, meta["log10_E"]), "start": latent.start_depth(E, X, rng), **fl,
                  "clean": ~(fl["unfit"] | fl["bound"] | fl["outside"])}
    log.info("%s: %d of %d rows with log10 E <= %.2f", os.path.basename(path), keep.sum(), len(keep), LOG10_E_MAX)
    return out


def species_class(tr, te, s, c, log):
    """Summaries for one species and class: flags, trend, spread, tails."""
    mt, mv = tr["meta"], te["meta"]
    a = (mt["labels"] == c) & tr[s]["clean"]
    b = (mv["labels"] == c) & te[s]["clean"]
    Za, Zb = tr[s]["Z"][a], te[s]["Z"][b]
    la, ca, lb, cb = mt["log10_E"][a], mt["cos_zenith"][a], mv["log10_E"][b], mv["cos_zenith"][b]
    beta = latent.fit_trend(Za, la, ca)
    Ra, Rb = latent.residuals(Za, la, ca, beta), latent.residuals(Zb, lb, cb, beta)
    c_edges = np.quantile(ca, np.linspace(0, 1, N_COS_CELLS + 1))
    n_cells = (len(E_EDGES) - 1) * N_COS_CELLS
    means = latent.binned_means(Za, latent.cell_index(la, ca, E_EDGES, c_edges), n_cells)
    binned = means[latent.cell_index(lb, cb, E_EDGES, c_edges)]
    m_all = mt["labels"] == c
    q = np.quantile(ca, [0, 1 / 3, 2 / 3, 1])
    r = {"n_train": int(m_all.sum()), "n_test": int((mv["labels"] == c).sum()),
         **{f"frac_{k}": float(tr[s][k][m_all].mean()) for k in ("unfit", "bound", "before", "after", "r_zero") if k in tr[s]},
         "r2_trend_test": latent.r2(Zb, Zb - Rb).tolist(), "r2_binned_test": latent.r2(Zb, binned).tolist(),
         "resid_sd_train": Ra.std(0).tolist(), "resid_sd_test": Rb.std(0).tolist(),
         "sd_by_energy_bin": [Ra[np.digitize(la, E_EDGES) - 1 == k].std(0).tolist() for k in range(len(E_EDGES) - 1)],
         "sd_by_cos_tercile": [Ra[(ca >= q[i]) & (ca <= q[i + 1])].std(0).tolist() for i in range(3)],
         "tails_train": latent.tail_stats(Ra), "_beta": beta}
    log.info("%-8s %-8s clean %.3f (bound %.3f, peak before %.3f, after %.3f) | R2 trend %s binned %s | sd %s | kurtosis %s",
             s, m1.CLASS[c], a.sum() / m_all.sum(), r["frac_bound"], r["frac_before"], r["frac_after"],
             np.round(r["r2_trend_test"], 3).tolist(), np.round(r["r2_binned_test"], 3).tolist(),
             np.round(r["resid_sd_test"], 3).tolist(), np.round(r["tails_train"]["excess_kurtosis"], 2).tolist())
    return r


def joint_labels():
    """Species and number name of each column of the joint latent."""
    return [f"{s} {n}" for s in SPECIES_NAMES for n in latent.NAMES[NAME]]


def joint(tr, te, per, c, log):
    """The three species together: correlations and held-out log-likelihood."""
    def stacked(d):
        m = (d["meta"]["labels"] == c) & np.all([d[s]["clean"] for s in SPECIES_NAMES], 0)
        R = np.hstack([latent.residuals(d[s]["Z"][m], d["meta"]["log10_E"][m], d["meta"]["cos_zenith"][m],
                                        per[f"{s}|{m1.CLASS[c]}"]["_beta"]) for s in SPECIES_NAMES])
        return R, d["meta"]["log10_E"][m]
    Ra, la = stacked(tr)
    Rb, lb = stacked(te)
    cov = np.cov(Ra.T)
    sd = np.sqrt(np.diag(cov))
    models = {"full": cov, "species_independent": latent.block_diagonal(cov, SIZES), "diagonal": np.diag(np.diag(cov))}
    ll = {k: latent.gaussian_loglik(Rb, v) for k, v in models.items()}
    ka, kb = np.digitize(la, E_EDGES) - 1, np.digitize(lb, E_EDGES) - 1
    tot, fallback = 0.0, []
    for k in range(len(E_EDGES) - 1):
        if (kb == k).any():
            enough = (ka == k).sum() >= MIN_BIN_ROWS
            fallback += [] if enough else [k]
            tot += latent.gaussian_loglik(Rb[kb == k], np.cov(Ra[ka == k].T) if enough else cov) * (kb == k).sum()
    ll["full_per_energy_bin"] = tot / len(Rb)
    corr = cov / np.outer(sd, sd)
    cross = np.abs(corr - latent.block_diagonal(corr, SIZES))
    i, j = np.unravel_index(np.argmax(cross), cross.shape)
    labels = joint_labels()
    r = {"n_train": int(len(Ra)), "n_test": int(len(Rb)), "test_loglik_per_shower": ll, "corr": corr.tolist(),
         "energy_bins_on_full_covariance": fallback,
         "max_cross_species_corr": {"value": float(corr[i, j]), "pair": [labels[i], labels[j]]},
         "mean_abs_cross_species_corr": float(cross[cross > 0].mean())}
    log.info("joint %-8s train %d test %d | held-out loglik/shower %s | strongest cross-species corr %.2f (%s / %s)",
             m1.CLASS[c], len(Ra), len(Rb), {k: round(v, 3) for k, v in ll.items()}, corr[i, j], labels[i], labels[j])
    return r


ROWS = [(s, c) for c in m1.CLASS for s in SPECIES_NAMES]


def latent_grid(xlabel):
    """One row per (species, class) in ROWS, one column per latent number."""
    fig, axes = plt.subplots(len(ROWS), len(latent.NAMES[NAME]), figsize=(14, 16), sharex=True)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    return fig, axes


def fig_trends(tr, per):
    fig, axes = latent_grid("log10 E [GeV]")
    mid = 0.5 * (FIG_E[1:] + FIG_E[:-1])
    for r_, (s, c) in enumerate(ROWS):
        m = (tr["meta"]["labels"] == c) & tr[s]["clean"]
        Z, l, u = tr[s]["Z"][m], tr["meta"]["log10_E"][m], tr["meta"]["cos_zenith"][m]
        T = Z - latent.residuals(Z, l, u, per[f"{s}|{m1.CLASS[c]}"]["_beta"])
        k = np.digitize(l, FIG_E) - 1
        for j, name in enumerate(latent.NAMES[NAME]):
            ax = axes[r_, j]
            q = np.array([np.percentile(Z[k == b, j], [16, 50, 84]) if (k == b).sum() >= 30 else [np.nan] * 3
                          for b in range(len(mid))])
            tm = np.array([T[k == b, j].mean() if (k == b).sum() >= 30 else np.nan for b in range(len(mid))])
            ax.fill_between(mid, q[:, 0], q[:, 2], color="#b5b5b5", alpha=0.6, lw=0, label="training showers, 16-84%")
            ax.plot(mid, q[:, 1], "o", color="#222222", ms=4, label="median")
            ax.plot(mid, tm, "-", color="#2a78d6", lw=2, label="quadratic trend (bin mean)")
            ax.set_title(f"{s}, {m1.CLASS[c]}: {name}", fontsize=9)
            ax.grid(alpha=0.3)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def fig_spread(per):
    fig, axes = latent_grid("log10 E [GeV]")
    mid = 0.5 * (E_EDGES[1:] + E_EDGES[:-1])
    for r_, (s, c) in enumerate(ROWS):
        sd = np.array(per[f"{s}|{m1.CLASS[c]}"]["sd_by_energy_bin"])
        for j, name in enumerate(latent.NAMES[NAME]):
            ax = axes[r_, j]
            ax.plot(mid, sd[:, j], "o-", color="#2a78d6", lw=1.5, ms=4)
            ax.set_ylim(0, np.nanmax(sd[:, j]) * 1.25)
            ax.set_title(f"{s}, {m1.CLASS[c]}: sd of {name}", fontsize=9)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_tails(tr, per):
    fig, axes = latent_grid("standardized residual")
    x = np.linspace(-6, 6, 200)
    for r_, (s, c) in enumerate(ROWS):
        m = (tr["meta"]["labels"] == c) & tr[s]["clean"]
        R = latent.residuals(tr[s]["Z"][m], tr["meta"]["log10_E"][m], tr["meta"]["cos_zenith"][m],
                             per[f"{s}|{m1.CLASS[c]}"]["_beta"])
        S = (R - R.mean(0)) / R.std(0)
        for j, name in enumerate(latent.NAMES[NAME]):
            ax = axes[r_, j]
            ax.hist(np.clip(S[:, j], -6, 6), bins=60, range=(-6, 6), density=True, color="#2a78d6", alpha=0.6)
            ax.plot(x, np.exp(-x * x / 2) / np.sqrt(2 * np.pi), color="#222222", lw=1.2)
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1)
            ax.set_title(f"{s}, {m1.CLASS[c]}: {name}", fontsize=9)
    fig.tight_layout()
    return fig


def fig_corr(jt):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    labels = [f"{s[:2]} {n}" for s in SPECIES_NAMES for n in latent.NAMES[NAME]]
    for ax, (cn, r) in zip(axes, jt.items()):
        im = ax.imshow(np.array(r["corr"]), cmap="RdBu_r", vmin=-1, vmax=1)
        for k in np.cumsum(SIZES[:-1]) - 0.5:
            ax.axhline(k, color="#222222", lw=1)
            ax.axvline(k, color="#222222", lw=1)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(f"{cn} primaries: correlation of residuals ({r['n_train']} showers)", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlation")
    return fig


def stage_latent(out, args, log):
    tr = load_latent(os.path.join(args.out, FOLDER, "fit", "fits_train_full.npz"), log)
    te = load_latent(os.path.join(args.out, m1.FOLDER, "fit", "fits_test.npz"), log)
    per = {f"{s}|{cn}": species_class(tr, te, s, c, log) for c, cn in m1.CLASS.items() for s in SPECIES_NAMES}
    jt = {cn: joint(tr, te, per, c, log) for c, cn in m1.CLASS.items()}
    plots.save(fig_trends(tr, per), out, "latent_trends", log)
    plots.save(fig_spread(per), out, "latent_spread", log)
    plots.save(fig_tails(tr, per), out, "latent_tails", log)
    plots.save(fig_corr(jt), out, "latent_corr", log)
    arrays = {**{f"meta.{k}": v for k, v in tr["meta"].items()}}
    for s in SPECIES_NAMES:
        arrays.update({f"{s}.{k}": tr[s][k] for k in ("Z", "unfit", "bound", "before", "after", "clean", "r_zero") if k in tr[s]})
    arrays.update({f"beta.{k}": v["_beta"] for k, v in per.items()})
    path = os.path.join(out, "latent_train.npz")
    np.savez(path, **arrays)
    log.info("wrote %s (%d bytes)", path, os.path.getsize(path))
    return {"family": NAME, "names": list(latent.NAMES[NAME]), "log10_e_max": LOG10_E_MAX,
            "species_class": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in per.items()},
            "joint": jt}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["part", "merge", "latent"])
    ap.add_argument("--n-parts", type=int, default=0)
    ap.add_argument("--part", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)))
    ap.add_argument("--limit", type=int, default=0, help="smoke run: only the first LIMIT training rows")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    if args.stage in ("part", "merge") and args.n_parts < 1:
        raise ValueError("--n-parts is required for the part and merge stages")
    if args.stage == "part" and not 0 <= args.part < args.n_parts:
        raise ValueError(f"--part {args.part} outside 0..{args.n_parts - 1}")
    if args.stage == "latent":
        out, log = logs.start(os.path.join(FOLDER, "latent"), args)
        logs.save(out, stage_latent(out, args, log), args, log)
        return
    folder = os.path.join(FOLDER + ("_smoke" if args.limit else ""), "fit")
    if args.stage == "merge":
        out, log = logs.start(folder, args)
        logs.save(out, stage_merge(out, args, log), args, log)
    else:
        log_dir, log = logs.start(os.path.join(folder, "logs", f"part_{args.part:04d}"), args)
        logs.save(log_dir, stage_part(os.path.join(args.out, folder), args, log), args, log)


if __name__ == "__main__":
    main()
