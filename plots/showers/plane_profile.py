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

def plane_table(points: np.ndarray) -> dict:
    """Reduce (n_showers, P, 5) native clouds to per-plane (n_showers, N_PLANES).

    Padding rows carry energy 0 and are dropped, so a plane a shower never
    reached scores a true zero rather than a padded one.
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
    return {"energy": energy, "points": count}


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
        out[name] = plane_table(samples)
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def corpus_samples(names, n):
    """Read the cached corpus instead of generating.

    Block boundaries come from the Step-0 species sidecar rather than assuming
    equal halves: SPECIES has grown past the two the current corpus was built
    with, so a species can legitimately have no rows in it.
    """
    import showerdata
    from modules.surrogates.fnn import _species_sidecar_path

    sidecar = _species_sidecar_path(DUAL_SHOWER_CACHE_PATH)
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
        lo, hi = int(rows[0]), int(rows[0]) + min(n, rows.size)
        pts = np.asarray(showerdata.load(DUAL_SHOWER_CACHE_PATH, start=lo, stop=hi).points)
        print(f"[{name}] corpus rows {lo}..{hi}  {pts.shape}")
        out[name] = plane_table(pts)
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
                    help="read the cached corpus instead of running the models")
    ap.add_argument("--out", type=str, default=EDA_OUTPUTS_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tables = (corpus_samples(args.species, args.n) if args.from_corpus
              else generate_samples(args.species, args.n, args.seed, args.device,
                                    args.batch, compile_gen=args.compile))

    tag = "corpus" if args.from_corpus else "models"
    for metric in (("energy", "points") if args.metric == "both" else (args.metric,)):
        figure_for_species(tables, metric,
                           os.path.join(args.out, f"plane_profile_{metric}_{tag}.png"),
                           n_showers=args.n)

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
