"""Per-plane (longitudinal) shower development, for every model in SPECIES.

Bins each generated shower by its native `layer_index` (cloud column 2, an exact
integer in [0, N_PLANES)) and shows how much lands on each of the 24 CORSIKA
observation planes:

    longitudinal profile   per-plane value, median with a p16/p84 band
    N_max                  each shower's PEAK-plane value, distributed
    X_max                  which plane that peak sits on, distributed
    per-plane extremes     max / mean / median across the sample, per plane

Two metrics, because "number of particles" is ambiguous for this corpus:

    energy   sum of column 3 over the plane. The physical one -- a stored point
             is a weighted CLUSTER of secondaries, not one particle (the
             generator writes a clustered representation, ~10 m cells near the
             core), so the energy column is what carries particle content.
    points   how many stored points land on the plane. A resolution artifact of
             the clustering, and capped per species (electron 4096, muon 25088),
             so it saturates where energy does not. Shown because a plane at the
             cap is exactly where truncation ("rod -> blob") starts.

Both species are driven through the SAME primaries, mirroring the paired corpus
`00_generate_data_dual_species.py` builds, so the electron and muon panels are
comparable event for event.

Usage:
    python plots/showers/plane_profile.py                       # both species, 256 showers
    python plots/showers/plane_profile.py --n 64 --species muon
    python plots/showers/plane_profile.py --from-corpus         # read the cached corpus instead
"""
import argparse
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = _HERE
while _V6 != os.path.dirname(_V6) and not os.path.exists(os.path.join(_V6, "_pathfix.py")):
    _V6 = os.path.dirname(_V6)
for _p in (_V6, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modules  # noqa: F401 — package import; keeps modules on the path
from modules.constants import (
    N_PLANES, LOG_E_MIN, LOG_E_MAX, ZENITH_MIN, ZENITH_MAX,
    AZIMUTH_MIN, AZIMUTH_MAX, LAYER_EAST_DX, DUAL_SHOWER_CACHE_PATH,
    EDA_OUTPUTS_DIR,
)

# Same backend as the corpus generator, loaded by path (its module name starts
# with a digit) -- so this EDA uses the exact checkpoints, staging and anti-clip
# policy that produced the training data.
_spec = importlib.util.spec_from_file_location(
    "gen_dual_species", os.path.join(_V6, "scripts", "00_generate_data_dual_species.py"))
gen_dual = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_dual)
SPECIES = gen_dual.SPECIES

SPECIES_COLOR = {"electron": "#1f77b4", "muon": "#d62728", "photon": "#2ca02c"}


# ── Per-plane reduction ──────────────────────────────────────────────────────

def plane_table(points: np.ndarray, e_prim: np.ndarray = None) -> dict:
    """Reduce (n_showers, P, 5) native clouds to per-plane (n_showers, N_PLANES).

    Padding rows carry energy 0 and are dropped, so a plane a shower never
    reached scores a true zero rather than a padded one. `e_prim` (the primary
    energy per shower) is carried through for `filter_by_plane_ratio`.
    """
    energy = np.zeros((points.shape[0], N_PLANES), dtype=np.float64)
    count = np.zeros((points.shape[0], N_PLANES), dtype=np.int64)
    for i, cloud in enumerate(points):
        real = cloud[cloud[:, 3] > 0]
        if not real.size:
            continue
        lay = np.clip(np.rint(real[:, 2]).astype(np.int64), 0, N_PLANES - 1)
        energy[i] = np.bincount(lay, weights=real[:, 3], minlength=N_PLANES)
        count[i] = np.bincount(lay, minlength=N_PLANES)
    out = {"energy": energy, "points": count}
    if e_prim is not None:
        out["e_prim"] = np.asarray(e_prim, dtype=np.float64).reshape(-1)
    return out


METRICS = ("points", "energy")


def filter_by_plane_ratio(tables, max_ratio: float):
    """Drop whole showers whose LARGEST plane exceeds `max_ratio` x the primary energy.

    The generator emits rare finite-but-corrupt showers -- measured up to 1.17e6
    on this statistic, against a p99.9 of 8.3 -- and they are not caught by the
    non-finite sanitization (they are finite) nor by the Step-0 anti-clip re-roll
    (their point counts are normal). They inflate a mean by 6 orders of magnitude
    and a std by more, so no summary statistic is usable without this.

    Whole showers, because the corruption is not localised: on the worst row the
    peak plane holds only 23% of the total and the MEDIAN plane is already
    inflated 4.8e4x. Zeroing a plane would leave the rest wrong.

    `max_ratio` is provisional. Measured on the 07 corpus: electron p99.9=8.3 /
    p99.99=38.7, and both species keep shedding rows from 10x to 100x without a
    clean gap, so the number is a judgement inside 10-25x rather than a boundary
    the data draws for us. 0 disables.
    """
    if not max_ratio:
        return tables, {}
    kept, report = {}, {}
    for nm, tab in tables.items():
        if "e_prim" not in tab:
            kept[nm] = tab
            continue
        ratio = tab["energy"].max(axis=1) / np.maximum(tab["e_prim"], 1e-30)
        keep = ratio <= max_ratio
        report[nm] = (int((~keep).sum()), len(keep), float(ratio.max()))
        kept[nm] = {k: v[keep] for k, v in tab.items()}
    return kept, report


def _band(ax, per_plane, color, label):
    """Median with a p16/p84 band over showers, one point per plane."""
    planes = np.arange(N_PLANES)
    med = np.median(per_plane, axis=0)
    lo, hi = np.percentile(per_plane, [16, 84], axis=0)
    ax.fill_between(planes, lo, hi, color=color, alpha=0.22, lw=0, label=f"{label} p16–p84")
    ax.plot(planes, med, color=color, lw=2.2, marker="o", ms=3.5, label=f"{label} median")


def _plane_axis(ax):
    """Label the plane axis, with the physical depth it corresponds to."""
    ax.set_xlabel(f"observation plane index   (Δ = {LAYER_EAST_DX:g} m along the axis)")
    ax.set_xlim(-0.5, N_PLANES - 0.5)
    ax.grid(alpha=0.3)


def figure_for_species(tables, metric, out_path, n_showers):
    """The four per-plane panels, all species overlaid on each."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    unit = "summed energy" if metric == "energy" else "stored points"

    ax = axes[0, 0]
    for name, tab in tables.items():
        _band(ax, tab[metric], SPECIES_COLOR.get(name, None), name)
    ax.set_yscale("log")
    ax.set_ylabel(f"{unit} on the plane")
    ax.set_title("Longitudinal profile")
    _plane_axis(ax)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name, tab in tables.items():
        nmax = tab[metric].max(axis=1)
        nmax = nmax[nmax > 0]
        ax.hist(np.log10(nmax), bins=40, histtype="step", lw=2,
                color=SPECIES_COLOR.get(name, None), label=f"{name}  (median {np.median(nmax):.3g})")
    ax.set_xlabel(f"log10  peak-plane {unit}")
    ax.set_ylabel("showers")
    ax.set_title("$N_{max}$ — each shower's peak plane")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for name, tab in tables.items():
        xmax = tab[metric].argmax(axis=1)
        ax.hist(xmax, bins=np.arange(N_PLANES + 1) - 0.5, histtype="step", lw=2,
                color=SPECIES_COLOR.get(name, None),
                label=f"{name}  (median plane {int(np.median(xmax))})")
    ax.set_ylabel("showers")
    ax.set_title("$X_{max}$ — which plane the peak sits on")
    _plane_axis(ax)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    planes = np.arange(N_PLANES)
    for name, tab in tables.items():
        c = SPECIES_COLOR.get(name, None)
        ax.plot(planes, tab[metric].max(axis=0), color=c, lw=2.2, label=f"{name} max")
        ax.plot(planes, tab[metric].mean(axis=0), color=c, lw=1.5, ls="--", label=f"{name} mean")
    if metric == "points":
        # The cap is what truncation acts on: a plane sitting at it is where the
        # "rod -> blob" clipping this corpus guards against would begin.
        for name in tables:
            cap = int(SPECIES[name]["max_points"])
            ax.axhline(cap, color=SPECIES_COLOR.get(name, None), ls=":", lw=1.4,
                       label=f"{name} cap ({cap})")
    ax.set_yscale("log")
    ax.set_ylabel(f"{unit} on the plane")
    ax.set_title("Per-plane extremes across the sample")
    _plane_axis(ax)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"Per-plane shower development — {metric}  "
                 f"(N={n_showers:,} showers/species)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def trimmed_mean(per_plane, trim=0.02):
    """Mean over showers after dropping the top `trim` by TOTAL content.

    The plain mean is not usable for every species here: the muon model emits
    rare finite-but-absurd showers (measured 4 of 256 at >100x the median,
    carrying 100.00% of the summed energy; max/median ~ 2e8). They are finite,
    so the pipeline's non-finite sanitization does not remove them. Trimming by
    whole showers -- not per plane -- keeps a shower's longitudinal shape intact.
    """
    tot = per_plane.sum(axis=1)
    keep = tot <= np.quantile(tot, 1.0 - trim) if len(tot) > 1 else np.ones(len(tot), bool)
    return per_plane[keep].mean(axis=0), int((~keep).sum())


def figure_mean_comparison(tables, out_path, n_showers, trim=0.02):
    """Mean per plane per species, over ALL showers, against a robust mean.

    Averages include showers that never reached a plane (a real 0), so the
    curves are per-shower expectations, additive across species: their sum is
    the mean complete event the surrogate has to predict.

    Both the plain and the trimmed mean are drawn because they disagree by
    orders of magnitude for muon -- see `trimmed_mean`. The composition panel
    uses the TRIMMED mean; on the plain mean it degenerates to 0/100/0 at every
    plane past 9, which describes four showers rather than the population.
    """
    planes = np.arange(N_PLANES)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    for col, metric in enumerate(("energy", "points")):
        unit = "summed energy" if metric == "energy" else "stored points"
        means = {nm: tab[metric].mean(axis=0) for nm, tab in tables.items()}
        trimmed, dropped = {}, {}
        for nm, tab in tables.items():
            trimmed[nm], dropped[nm] = trimmed_mean(tab[metric], trim)

        ax = axes[0, col]
        for nm in means:
            c = SPECIES_COLOR.get(nm)
            ratio = means[nm].sum() / max(trimmed[nm].sum(), 1e-30)
            ax.plot(planes, means[nm], color=c, lw=2.2, marker="o", ms=3.5,
                    label=f"{nm} mean  (x{ratio:,.0f} the trimmed)" if ratio > 2
                          else f"{nm} mean")
            ax.plot(planes, trimmed[nm], color=c, lw=1.6, ls="--",
                    label=f"{nm} trimmed mean (-{dropped[nm]} showers)")
        ax.set_yscale("log")
        ax.set_ylabel(f"mean {unit} per shower")
        ax.set_title(f"Mean {metric} per plane   (N={n_showers:,}/species)")
        _plane_axis(ax)
        ax.legend(fontsize=7, ncol=1)

        ax = axes[1, col]
        total = np.sum(list(trimmed.values()), axis=0)
        share = {nm: 100.0 * mu / np.maximum(total, 1e-30) for nm, mu in trimmed.items()}
        ax.stackplot(planes, *share.values(), labels=list(share),
                     colors=[SPECIES_COLOR.get(nm) for nm in share], alpha=0.85)
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"share of mean {unit} [%]")
        ax.set_title(f"Composition of the mean event ({metric}, trimmed)")
        _plane_axis(ax)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"Mean per-plane content by species, over all showers "
                 f"(dashed = top {100*trim:.0f}% of showers dropped)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def print_mean_table(tables, metric, trim=0.02):
    """The numbers behind the mean figure, so they can be quoted directly."""
    print(f"\n--- {metric}: outlier check (plain mean is only usable where this is ~1) ---")
    for nm, tab in tables.items():
        tot = tab[metric].sum(axis=1)
        med = float(np.median(tot))
        big = int((tot > 100 * med).sum())
        print(f"  {nm:<9} mean/median={tot.mean()/max(med,1e-30):>12.4g}  "
              f"max/median={tot.max()/max(med,1e-30):>12.4g}  "
              f">100x median: {big}/{len(tot)} showers "
              f"carrying {100*tot[tot > 100*med].sum()/max(tot.sum(),1e-30):.2f}%")
    means = {nm: trimmed_mean(tab[metric], trim)[0] for nm, tab in tables.items()}
    total = np.sum(list(means.values()), axis=0)
    names = list(means)
    print(f"\n=== TRIMMED mean {metric} per plane (top {100*trim:.0f}% of showers dropped) ===")
    print("  plane " + "".join(f"{nm:>14}" for nm in names) + f"{'total':>14}   share%")
    for p in range(N_PLANES):
        shares = "/".join(f"{100*means[nm][p]/max(total[p],1e-30):.0f}" for nm in names)
        print(f"  {p:>5} " + "".join(f"{means[nm][p]:>14.4g}" for nm in names)
              + f"{total[p]:>14.4g}   {shares}")
    print("  " + "-" * (6 + 14 * (len(names) + 1)))
    print(f"  {'Σ':>5} " + "".join(f"{means[nm].sum():>14.4g}" for nm in names)
          + f"{total.sum():>14.4g}"
          + f"   {'/'.join(f'{100*means[nm].sum()/max(total.sum(),1e-30):.0f}' for nm in names)}")


def summary_stats(tables):
    """mean and std over showers, per plane, for the three requested quantities.

    particles  stored points on the plane            -- absolute, per shower
    energy     summed energy on the plane            -- absolute, per shower
    profile    each shower's plane share of its OWN total (fractions summing to
               1), so shape is separated from scale: two showers with the same
               development but different energies give the same profile. Its std
               is therefore shower-to-shower SHAPE variation, which the absolute
               std conflates with the energy spread.

    std is the population std (ddof=0) over showers, including showers that never
    reached a plane -- they contribute a real 0, as in the mean.
    """
    out = {}
    for nm, tab in tables.items():
        st = {}
        for metric in METRICS:
            v = tab[metric].astype(np.float64)
            st[metric] = (v.mean(axis=0), v.std(axis=0))
        tot = tab["energy"].sum(axis=1, keepdims=True)
        prof = tab["energy"] / np.maximum(tot, 1e-30)
        st["profile"] = (prof.mean(axis=0), prof.std(axis=0))
        st["n"] = tab["energy"].shape[0]
        out[nm] = st
    return out


QUANTITY_LABEL = {
    "points":  ("particles per plane", "stored points"),
    "energy":  ("energy per plane", "summed energy"),
    "profile": ("shower profile (plane share of own total)", "fraction of shower"),
}


def figure_mean_std(stats, out_path, dataset):
    """One row per quantity: mean with a +/-1 std band, all species overlaid."""
    planes = np.arange(N_PLANES)
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12))

    for row, q in enumerate(("points", "energy", "profile")):
        title, unit = QUANTITY_LABEL[q]
        # Log panels need a positive floor for the band. std exceeds the mean on
        # most planes here, so a naive clip to ~0 makes every band fill the whole
        # axis down to the decade floor and the panel goes solid. Floor at a
        # decade below the smallest positive mean instead, and pin ylim to it.
        pos = np.concatenate([st[q][0][st[q][0] > 0] for st in stats.values()]
                             or [np.array([1.0])])
        floor = float(pos.min()) / 10.0 if pos.size else 1e-3
        for col, logy in enumerate((True, False)):
            ax = axes[row, col]
            for nm, st in stats.items():
                mu, sd = st[q]
                c = SPECIES_COLOR.get(nm)
                lo = np.maximum(mu - sd, floor) if logy else mu - sd
                ax.fill_between(planes, lo, mu + sd, color=c, alpha=0.20, lw=0)
                ax.plot(planes, mu, color=c, lw=2.2, marker="o", ms=3.5,
                        label=f"{nm}  (n={st['n']})")
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(bottom=floor)
            ax.set_ylabel(f"mean {unit}  ±1σ")
            ax.set_title(f"{title}" + ("  [log]" if logy else "  [linear]"))
            _plane_axis(ax)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f"Per-plane mean ± std by species — {dataset}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def print_mean_std(stats):
    """The three tables, mean ± std per plane per species."""
    for q in ("points", "energy", "profile"):
        title, unit = QUANTITY_LABEL[q]
        fmt = "{:>9.4f}±{:<9.4f}" if q == "profile" else "{:>9.4g}±{:<9.4g}"
        print(f"\n=== {title} — mean ± std over showers ===")
        print("  plane " + "".join(f"{nm:^21}" for nm in stats))
        for pl in range(N_PLANES):
            cells = "".join(fmt.format(stats[nm][q][0][pl], stats[nm][q][1][pl])
                            for nm in stats)
            print(f"  {pl:>5} {cells}")
        print("  " + "-" * (6 + 21 * len(stats)))
        tot = "".join(fmt.format(stats[nm][q][0].sum(), np.hypot.reduce(stats[nm][q][1]))
                      for nm in stats)
        print(f"  {'Σ':>5} {tot}   (std column = quadrature sum over planes)")


# ── Sample sources ───────────────────────────────────────────────────────────

def generate_samples(names, n, seed, device, batch, compile_gen=False):
    """Run each species' model on the SAME primaries, as the paired corpus does."""
    from allshowers.generate_showers import sample_primary_particles, run_point_count_fm
    from allshowers.generator import generate

    prim = sample_primary_particles(
        e_min=10 ** LOG_E_MIN, e_max=10 ** LOG_E_MAX,
        zenith_min=ZENITH_MIN, zenith_max=ZENITH_MAX,
        azimuth_min=AZIMUTH_MIN, azimuth_max=AZIMUTH_MAX, n=n, seed=seed,
    )
    energies, directions = prim["energies"], prim["directions"]
    labels = prim["labels"].to(torch.int64)
    print(f"[primaries] {n} shared events  "
          f"E in [{float(energies.min()):.2e}, {float(energies.max()):.2e}] GeV")

    out = {}
    for name in names:
        cfg = SPECIES[name]
        print(f"[{name}] staging (max_points={cfg['max_points']})")
        staged_dir, pcfm = gen_dual.stage_run_dir(name, cfg)
        # compile=False by default: inductor fails lowering the generator's
        # flex_attention mask subgraph on this torch build, and compilation only
        # pays for itself over a full corpus -- an EDA sample of a few hundred
        # showers spends longer compiling than generating. --compile to opt in.
        gen = gen_dual.Generator(run_dir=staged_dir, num_timesteps=gen_dual.NUM_TIMESTEPS,
                                 compile=compile_gen, solver=gen_dual.SOLVER)
        gen.max_points = int(cfg["max_points"])
        num_points = run_point_count_fm(
            model_path=pcfm, energies=energies, directions=directions, labels=labels)
        num_points = gen_dual.resample_overclip(
            pcfm, energies, directions, labels, num_points, cap=int(cfg["max_points"]))
        # Scale the batch by the species' point cap. Uncompiled flex_attention
        # materialises the full score matrix, so its memory follows max_points,
        # and the caps differ 6x (electron 4096, muon 25088). One flat batch
        # either OOMs on muon or wastes the GPU on electron; `--batch` is
        # therefore the budget AT the electron cap and is scaled from there.
        sp_batch = max(1, int(batch * 4096 / int(cfg["max_points"])))
        if sp_batch != batch:
            print(f"[{name}] batch {batch} -> {sp_batch} (cap {cfg['max_points']})")
        # gpu_test hands out mixed cards (20 GB and 40 GB seen), and the score
        # matrix is the dominant allocation, so halve and retry rather than
        # pinning a GPU model or guessing a batch that fits everywhere.
        while True:
            try:
                samples = generate(generator=gen, energies=energies, num_points=num_points,
                                   angles=directions, batch_size=sp_batch, device=device,
                                   labels=labels).float().cpu().numpy()
                break
            except torch.OutOfMemoryError:
                if sp_batch == 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                sp_batch = max(1, sp_batch // 2)
                print(f"[{name}] CUDA OOM -> retrying at batch {sp_batch}")
        print(f"[{name}] generated {samples.shape}")
        out[name] = plane_table(samples, energies.numpy())
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def corpus_samples(names, n, corpus_path):
    """Read the cached corpus instead of generating.

    Block boundaries come from the Step-0 species sidecar rather than assuming
    equal halves: SPECIES has grown past the two the current corpus was built
    with, so a species can legitimately have no rows in it.
    """
    import showerdata
    from modules.surrogates.fnn import _species_sidecar_path

    sidecar = _species_sidecar_path(corpus_path)
    if not os.path.exists(sidecar):
        raise FileNotFoundError(f"species sidecar not found: {sidecar}")
    species_ids = torch.load(sidecar).long().numpy()

    out = {}
    for i, name in enumerate(SPECIES):
        if name not in names:
            continue
        rows = np.flatnonzero(species_ids == i)
        if not rows.size:
            print(f"[{name}] no rows in the corpus (species id {i}); "
                  f"drop --from-corpus to run its model instead")
            continue
        lo, hi = int(rows[0]), int(rows[0]) + (rows.size if n <= 0 else min(n, rows.size))
        sub = showerdata.load(corpus_path, start=lo, stop=hi)
        pts = np.asarray(sub.points)
        e_prim = np.asarray(sub.energies, dtype=np.float64).reshape(-1)
        print(f"[{name}] corpus rows {lo}..{hi}  {pts.shape}")
        out[name] = plane_table(pts, e_prim)
    if not out:
        raise SystemExit("no requested species present in the corpus")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", nargs="+", default=list(SPECIES),
                    choices=list(SPECIES), help="models to profile (default: all)")
    ap.add_argument("--n", type=int, default=256, help="showers per species")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8,
                    help="generation batch AT the electron cap (4096); scaled down "
                         "per species by its own max_points. 8 fits a 40 GB A100; "
                         "halve it on a 20 GB card")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the generator (see generate_samples)")
    ap.add_argument("--metric", choices=("energy", "points", "both"), default="both")
    ap.add_argument("--from-corpus", action="store_true",
                    help="read a corpus instead of running the models")
    ap.add_argument("--corpus", type=str, default=DUAL_SHOWER_CACHE_PATH,
                    help="corpus .pt to read with --from-corpus "
                         "(default: constants.DUAL_SHOWER_CACHE_PATH)")
    ap.add_argument("--max-plane-ratio", type=float, default=20.0,
                    help="drop a shower if any plane exceeds this multiple of the "
                         "primary energy; 0 disables. See filter_by_plane_ratio -- "
                         "the value is provisional")
    ap.add_argument("--out", type=str, default=EDA_OUTPUTS_DIR)
    ap.add_argument("--from-cache", action="store_true",
                    help="reuse the per-plane tables saved by the last run "
                         "(plane_tables.npz in --out) instead of generating again")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "plane_tables.npz")
    if args.from_cache:
        z = np.load(cache)
        tables = {nm: {m: z[f"{nm}/{m}"]
                       for m in ("energy", "points", "e_prim") if f"{nm}/{m}" in z}
                  for nm in args.species if f"{nm}/energy" in z}
        if not tables:
            raise SystemExit(f"no cached species from {args.species} in {cache}")
        args.n = next(iter(tables.values()))["energy"].shape[0]
        print(f"[cache] {cache}  species={list(tables)}  n={args.n}")
    else:
        tables = (corpus_samples(args.species, args.n, args.corpus) if args.from_corpus
                  else generate_samples(args.species, args.n, args.seed, args.device,
                                        args.batch, compile_gen=args.compile))
        # The tables are tiny (n x 24 per metric); generation is what costs.
        np.savez(cache, **{f"{nm}/{m}": tab[m]
                           for nm, tab in tables.items() for m in tab})
        print(f"[cache] wrote {cache}")

    tables, dropped = filter_by_plane_ratio(tables, args.max_plane_ratio)
    if dropped:
        print(f"\n[filter] any plane > {args.max_plane_ratio:g}x the primary energy "
              f"-> drop the shower")
        for nm, (k, tot, worst) in dropped.items():
            print(f"  {nm:<9} dropped {k}/{tot} ({100*k/max(tot,1):.3f}%)  "
                  f"largest ratio seen: {worst:.4g}")

    dataset = os.path.basename(args.corpus) if args.from_corpus else "generated models"
    stats = summary_stats(tables)
    print_mean_std(stats)
    figure_mean_std(stats, os.path.join(args.out, "plane_mean_std.png"), dataset)

    tag = "corpus" if args.from_corpus else "models"
    for metric in (("energy", "points") if args.metric == "both" else (args.metric,)):
        figure_for_species(tables, metric,
                           os.path.join(args.out, f"plane_profile_{metric}_{tag}.png"),
                           n_showers=args.n)
    figure_mean_comparison(
        tables, os.path.join(args.out, f"plane_mean_by_species_{tag}.png"),
        n_showers=args.n)
    for metric in ("energy", "points"):
        print_mean_table(tables, metric)

    # The numbers behind the figures, so they can be quoted without re-reading a plot.
    for metric in ("energy", "points"):
        print(f"\n=== {metric} ===")
        for name, tab in tables.items():
            v = tab[metric]
            nmax, xmax = v.max(axis=1), v.argmax(axis=1)
            empty = float(np.mean(v.sum(axis=1) == 0))
            print(f"  {name:<9} N_max median={np.median(nmax):>12.4g}  "
                  f"p95={np.percentile(nmax, 95):>12.4g}   "
                  f"X_max median plane={int(np.median(xmax)):>2d}  "
                  f"empty showers={100 * empty:.1f}%")
            if metric == "points":
                cap = int(SPECIES[name]["max_points"])
                print(f"  {'':<9} points/shower max={int(v.sum(axis=1).max()):,} "
                      f"vs cap {cap:,}  ({100 * v.sum(axis=1).max() / cap:.0f}% of cap)")


if __name__ == "__main__":
    main()
