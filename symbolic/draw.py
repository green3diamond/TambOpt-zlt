"""Particle draw: symbolic showers as particles through the preprocessing model, on the
scored primaries (chain.primaries), in the suite's cloud format.

Per primary and species: plane energies E_k and Xmax from the Part-1 sampler (one draw, fixed seed); per plane with
energy, inputs annuli.features; then, with the laws of particles.py (fitted on the v2 raw showers):
  1. energy before the cut   E_k / stored share (stored law)
  2. ring energies           shares from the rings law stretched in radius by the width law along the shower
                             (width_sr.py), AR(1) from plane to plane
  3. particles per ring      Poisson(ring energy / mean particle energy (tanh-sum law of tiny_net.py, or the
                             ring-by-ring mean_energy law with --energy-law none)), at most ring / E_MIN
  4. particle energies       E_MIN each plus the ring's excess split by Burr XII draws (spread law): the ring energy
                             is exact and no particle falls under CORSIKA's threshold
  5. positions               uniform in area within the ring (the open last ring up to particles.R_OPEN x its inner
                             edge), uniform azimuth
  6. times                   axis light time + lead + Exp(mean delay - lead) (lead and delay laws)
  6b. offset (--offset)     every particle of plane k shifted in the plane by kappa x r_half,k x u_k: r_half,k the
                             radius holding half the plane's ring shares, u an AR(1) in plane index (unit variance per
                             axis, rho), from its own generator so the particles are those drawn without it
  7. render                  the preprocessing with the OLD caps (render.render, sources.CAP); each plane's stored
                             energy rescaled to E_k. With --caps new: the caps of the v2 and 150k per-simulation
                             preprocessing (sources.CAP_NEW) and no rescale, the render decides what is stored, as
                             for CORSIKA (E_k then only sets the energy before the cut)
With energies --fits (a sampler fitted on TOTAL plane energy, route B): E_k is the energy before the cut, step 1 is
skipped, the laws must be fitted on total plane energy and the caps must be new.
Gate: every drawn shower's stored plane energies equal the sampled E_k (old caps), or stay at or below the drawn
energy before the cut (new caps).
Stages: energies (sample once and save), draw (task i of N takes every N-th primary, workers within a task, one seed
per primary so the split does not matter), merge (padded clouds.npz with the suite's keys).
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
from scipy.special import softmax

from symbolic import logs
from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import annuli, atmosphere, chain, fit_profiles, lateral, longitudinal, particles, planes, energy_sr, times_sr, width_sr, \
    render, sampler, sources

FOLDER = os.path.join("particles", "draw")
LAWS = os.path.join("particles", "fit_v2", "particle_laws.npz")
WIDTH_LAW = os.path.join("particles", "sr_width", width_sr.LAW)
TIME_LAW = os.path.join("particles", "sr_times", times_sr.LAW)
ENERGY_LAW = os.path.join("particles", "judge", "sr_energy_tanh", energy_sr.LAW)
LAW_NAMES = ("stored", "rings", "mean_energy", "lead", "delay", "spread")


def fill_rings(B):
    """Ring-wise law rows left unfitted (NaN) take the nearest fitted ring's row."""
    B = B.copy()
    fitted = np.flatnonzero(np.isfinite(B).all(1))
    for j in np.flatnonzero(~np.isfinite(B).all(1)):
        B[j] = B[fitted[np.argmin(np.abs(fitted - j))]]
    return B


def load_laws(root, path=LAWS):
    z = np.load(os.path.join(root, path))
    out = {}
    for s in SPECIES_NAMES:
        for cn in planes.CLASS.values():
            law = {w: z[f"{s}.{cn}.{w}"] for w in LAW_NAMES}
            law["plane_energy"] = fitted_on(z)
            for w in ("mean_energy", "lead", "delay"):
                law[w] = fill_rings(law[w])
            out[(s, cn)] = law
    return out


def fitted_on(z):
    """The plane energy a law file's inputs were fitted on (files from before the choice: stored)."""
    return str(z["plane_energy"]) if "plane_energy" in z.files else "stored"


def width_law_path(laws, width_law):
    """The width law to draw with. Laws fitted on part of the showers need their own, or a judged shower could leak in."""
    if width_law:
        return width_law
    if laws != LAWS:
        raise ValueError("--laws is not the default one: give the --width-law fitted on the same showers")
    return WIDTH_LAW


def time_law_path(laws, time_law):
    """The time law to draw with, None for the ring-by-ring laws. Laws fitted on part of the showers need their own, or a judged shower could leak in."""
    if time_law == "none":
        return None
    if time_law:
        return time_law
    if laws != LAWS:
        raise ValueError("--laws is not the default one: give the --time-law fitted on the same showers, or none")
    return TIME_LAW


def energy_law_path(laws, energy_law):
    """The mean energy law to draw with, None for the ring-by-ring law. Laws fitted on part of the showers need their own, or a judged shower could leak in."""
    if energy_law == "none":
        return None
    if energy_law:
        return energy_law
    if laws != LAWS:
        raise ValueError("--laws is not the default one: give the --energy-law fitted on the same showers, or none")
    return ENERGY_LAW


def ring_times(law, x, k):
    """log10(1 + t / ns) per ring of the lead and the delay: from a common form if the laws carry one, else ring by ring."""
    if "time" in law:
        return tuple(times_sr.ring_times(law["time"][w], law["time"]["form"], x, k) for w in times_sr.TIMES)
    return law["lead"] @ x, law["delay"] @ x


def ring_means(law, x, k):
    """ln mean particle energy per ring: from a common form if the laws carry one, else ring by ring."""
    if "energy" in law:
        return energy_sr.ring_means(law["energy"]["theta"], law["energy"]["form"], x, k)
    return law["mean_energy"] @ x


def half_radius(shares):
    """Radius [m] where the cumulative ring shares cross 1/2, linear in r within the ring (open ring to R_OPEN x edge)."""
    cum = np.cumsum(shares)
    j = min(int(np.argmax(cum >= 0.5)), particles.N_RING - 1)
    lo, hi = particles.EDGES[j], particles.EDGES[j + 1]
    hi = hi if np.isfinite(hi) else particles.R_OPEN * lo
    before = cum[j - 1] if j else 0.0
    return float(lo + (hi - lo) * np.clip((0.5 - before) / max(shares[j], 1e-300), 0.0, 1.0))


def plane_particles(rng, k, x, e_k, law, s, log_lam=None, x_stored=None, total=False, shift=None):
    """Particles (n, 5) [plane k, across, along, energy, time] of one plane before the preprocessing; log_lam: the
    plane's log10 width, None for the rings law unstretched; x_stored: the inputs of the stored law if not x;
    total: e_k is already the plane's energy before the cut (no stored law); shift: (across, along) offset of every
    particle in units of the plane's half radius, None for none."""
    e_tot = e_k if total else e_k / softmax((x if x_stored is None else x_stored) @ law["stored"])[0]
    shares = softmax(x @ law["rings"])
    shares = shares if log_lam is None else particles.stretch(shares, log_lam)
    lead_y, delay_y = ring_times(law, x, k)
    lead = np.maximum(10.0 ** lead_y - 1, 0.0)
    tau = np.maximum(10.0 ** delay_y - 1 - lead, 0.0)
    p = ring_particles(rng, k, e_tot * shares, np.exp(ring_means(law, x, k)), law["spread"],
                       particles.E_MIN[s], lead, tau)
    if shift is not None and len(p):
        p[:, 1:3] += np.asarray(shift, np.float64) * half_radius(shares)
    return p


def ring_particles(rng, k, e_ring, mean, spread, e_min, lead, tau):
    """Particles of one plane from its ring energies, mean particle energies, spread (c, k) and delays [ns] (steps
    3-6 of the module docstring)."""
    mean = np.maximum(mean, e_min * 1.001)
    n = np.minimum(rng.poisson(e_ring / mean), np.floor(e_ring / e_min)).astype(int)
    if not n.any():
        return np.zeros((0, 5))
    j = np.repeat(np.arange(particles.N_RING), n)
    v = np.concatenate([particles.burr_sample(rng, *spread[r], m) for r, m in enumerate(n) if m])
    excess = e_ring - n * e_min
    e = e_min + excess[j] * v / np.bincount(j, v, particles.N_RING)[j]
    lo, hi = particles.EDGES[j], particles.EDGES[j + 1]
    hi = np.where(np.isfinite(hi), hi, particles.R_OPEN * lo)
    r, phi = np.sqrt(rng.uniform(lo ** 2, hi ** 2)), rng.uniform(0, 2 * np.pi, len(j))
    t = (k + 1) * atmosphere.PLANE_SPACING / annuli.C_LIGHT + (lead[j] + tau[j] * rng.exponential(size=len(j))) * 1e-9
    return np.column_stack([np.full(len(j), k), r * np.cos(phi), r * np.sin(phi), e, t])


def plane_inputs(X, xmax, l, u, e_k, ks, law, kind="stored"):
    """Inputs of the stored law (from the stored plane energies e_k) and of every other law (from the plane energy
    the laws were fitted on: stored, or total = stored / stored share). kind "total": e_k are total energies,
    the inputs of every law, and no stored law is applied."""
    age, r_ref = longitudinal.age(X, xmax), np.array([lateral.r_m_ref(k, u) for k in ks])
    feats = lambda e: annuli.features(age, np.full(len(ks), l), np.full(len(ks), u), np.log(e), r_ref)
    x_s = feats(e_k)
    if kind == "total":
        if law["plane_energy"] != "total":
            raise ValueError("total plane energies need laws fitted on total plane energy")
        return x_s, x_s
    if law["plane_energy"] == "stored":
        return x_s, x_s
    return x_s, feats(e_k / softmax(x_s @ law["stored"], axis=1)[:, 0])


def shower(job):
    """Stored cells per species of one primary -> (index, {species: cells}, seconds, particles drawn)."""
    i, seed, l, u, cn, X, d, E, xmax, laws, *rest = job
    caps = rest[0] if rest else "old"
    kind = rest[1] if len(rest) > 1 else "stored"
    offset = rest[2] if len(rest) > 2 else None
    if kind == "total" and caps != "new":
        raise ValueError("total plane energies need --caps new (old caps rescale to stored energies)")
    rng = np.random.default_rng([seed, i])
    t0, out, n_part = time.time(), {}, 0
    for si, s in enumerate(SPECIES_NAMES):
        ks = np.nonzero(E[s] > 0)[0]
        kappa, rho = (offset or {}).get(f"{s} {cn}", (0.0, 0.0))
        off = kappa * sampler.ar1(2, N_PLANES, rho, np.random.default_rng([seed, i, 1, si])) if kappa > 0 else None
        law, w = laws[(s, cn)], laws[(s, cn)]["along"]
        x_s, x = plane_inputs(X[ks], xmax[s], l, u, E[s][ks], ks, law, kind)
        ll = width_sr.law_mean(w, ks, x[:, 3], x[:, 1]) + width_sr.law_sd(w, ks) * sampler.ar1(1, N_PLANES, w[-1], rng)[0][ks]
        p = [plane_particles(rng, k, x[a], E[s][k], law, s, log_lam=ll[a], x_stored=x_s[a], total=kind == "total",
                             shift=None if off is None else off[:, k]) for a, k in enumerate(ks)]
        p = np.concatenate(p) if p else np.zeros((0, 5))
        n_part += len(p)
        cells = render.render(p, d, sources.CAPS[caps][s]) if len(p) else np.zeros((0, 5))
        k = cells[:, 2].astype(int)
        if caps == "new":
            drawn = np.bincount(p[:, 0].astype(int), p[:, 3], N_PLANES) if len(p) else np.zeros(N_PLANES)
            if (np.bincount(k, cells[:, 3], N_PLANES) > drawn * (1 + 1e-9) + 1e-9).any():
                raise ValueError(f"primary {i} {s}: stored energy above the drawn energy")
            out[s] = cells
            continue
        stored = np.bincount(k, cells[:, 3], N_PLANES)
        scale = np.divide(E[s], stored, out=np.zeros(N_PLANES), where=stored > 0)
        cells[:, 3] *= scale[k]
        if not np.allclose(np.bincount(k, cells[:, 3], N_PLANES)[stored > 0], E[s][stored > 0]):
            raise ValueError(f"primary {i} {s}: stored energies differ from the sampled plane energies")
        out[s] = cells
    return i, out, time.time() - t0, n_part


def scored_primaries(args):
    """rows, log10 E, cos z, classes, depths, unit directions of chain.primaries (--smoke: an even subset)."""
    rows = chain.primaries()[0]
    rows = rows[:: max(1, len(rows) // args.smoke)] if args.smoke else rows
    zt = np.load(os.path.join(args.out, fit_profiles.TEST))
    dirs = zt["meta.directions"][rows] / np.linalg.norm(zt["meta.directions"][rows], axis=1, keepdims=True)
    return rows, zt["meta.log10_E"][rows], zt["meta.cos_zenith"][rows], zt["meta.labels"][rows], zt["X"][rows], dirs


def given_primaries(args):
    """The same from a primaries file of judge.py, each primary --draws times in a row."""
    z = np.load(args.primaries)
    idx = np.repeat(np.arange(len(z["rows"])), args.draws)
    return tuple(z[k][idx] for k in ("rows", "l", "u", "labels", "X", "dirs"))


def sample_energies(root, l, u, X, lab, rng, fits=None, noise="plane"):
    """Plane energies {species: (n, planes)}, Xmax {species: (n,)} and the plane energy kind ("stored" for the
    default training file; a fits file of fit_profiles.py --stage tables: "total", fitted on its own showers, all
    of them up to its top) from the Part-1 sampler, class by class."""
    E = {s: np.zeros((len(l), N_PLANES)) for s in SPECIES_NAMES}
    xmax = {s: np.zeros(len(l)) for s in SPECIES_NAMES}
    if fits is None:
        kind, tr, top = "stored", sampler.load(os.path.join(root, fit_profiles.TRAIN)), sampler.TOP
    else:
        z = np.load(fits)
        kind, e_max = str(z["plane_energy"]), float(z["top"])
        tr, top = sampler.load(fits, e_max=e_max), np.inf
    for c in planes.CLASS:
        m = lab == c
        e, lat = sampler.sample(sampler.fit(tr, c, top=top, noise=noise), l[m], u[m], X[m], rng)
        for j, s in enumerate(SPECIES_NAMES):
            E[s][m] = e[s]
            xmax[s][m] = longitudinal.from_latent(sampler.species_block(lat, j)[0], l[m])[0]
    return E, xmax, kind


def energy_kind(z):
    """The plane energy an energies.npz holds (files from before the choice: stored)."""
    return str(z["plane_energy"]) if "plane_energy" in z.files else "stored"


def stage_energies(out, args, log):
    rows, l, u, lab, X, dirs = given_primaries(args) if args.primaries else scored_primaries(args)
    E, xmax, kind = sample_energies(args.out, l, u, X, lab, np.random.default_rng(args.seed), args.fits, args.noise)
    np.savez(os.path.join(out, "energies.npz"), rows=rows, labels=lab, l=l, u=u, X=X, dirs=dirs,
             plane_energy=np.array(kind), noise=np.array(args.noise), **{f"{s}.E": E[s] for s in SPECIES_NAMES},
             **{f"{s}.xmax": xmax[s] for s in SPECIES_NAMES})
    log.info("sampled %s plane energies of %d primaries, %s noise", kind, len(rows), args.noise)
    return {"primaries": int(len(rows)), "seed": args.seed, "plane_energy": kind, "fits": args.fits, "noise": args.noise}


def draw_laws(root, laws_path=LAWS, width_law=None, time_law=None, energy_law=None):
    """The laws of every (species, class) with the width, time and mean energy laws the draw uses."""
    laws = load_laws(root, laws_path)
    z_w = np.load(os.path.join(root, width_law_path(laws_path, width_law)))
    on = {"particle laws": next(iter(laws.values()))["plane_energy"], "width law": fitted_on(z_w)}
    for (s, cn), law in laws.items():
        law["along"] = z_w[f"{s}.{cn}"]
    time_law = time_law_path(laws_path, time_law)
    if time_law:
        z_t = np.load(os.path.join(root, time_law))
        on["time law"] = fitted_on(z_t)
        for (s, cn), law in laws.items():
            law["time"] = {"form": str(z_t["form"]), **{w: z_t[f"{s}.{cn}.{w}"] for w in times_sr.TIMES}}
    energy_law = energy_law_path(laws_path, energy_law)
    if energy_law:
        z_e = np.load(os.path.join(root, energy_law))
        on["energy law"] = fitted_on(z_e)
        for (s, cn), law in laws.items():
            law["energy"] = {"form": str(z_e["form"]), "theta": z_e[f"{s}.{cn}"]}
    if len(set(on.values())) > 1:
        raise ValueError(f"laws fitted on different plane energies: {on}")
    return laws


def stage_draw(out, args, log):
    z = np.load(os.path.join(out, "energies.npz"))
    laws = draw_laws(args.out, args.laws, args.width_law, args.time_law, args.energy_law)
    mine = np.arange(len(z["rows"]))[args.task::args.tasks]
    if args.only:
        mine = mine[np.isin(mine, np.load(args.only))]
    offset = None
    if args.offset:
        with open(args.offset) as f:
            offset = {k: (float(v[0]), float(v[1])) for k, v in json.load(f).items()}
        log.info("offset (kappa, rho): %s", offset)
    jobs = [(i, args.seed, float(z["l"][i]), float(z["u"][i]), planes.CLASS[int(z["labels"][i])], z["X"][i],
             z["dirs"][i], {s: z[f"{s}.E"][i] for s in SPECIES_NAMES}, {s: float(z[f"{s}.xmax"][i]) for s in SPECIES_NAMES},
             laws, args.caps, energy_kind(z), offset) for i in mine]
    cells = {s: {} for s in SPECIES_NAMES}
    secs, n_part = [], []
    with Pool(args.workers) as pool:
        for i, c, dt, n in pool.imap_unordered(shower, jobs):
            for s in SPECIES_NAMES:
                cells[s][i] = c[s].astype(np.float32)
            secs.append(dt)
            n_part.append(n)
            if len(secs) % 25 == 0:
                log.info("task %d: %d / %d primaries", args.task, len(secs), len(mine))
    part = {"index": mine}
    for s in SPECIES_NAMES:
        per = [cells[s][i] for i in mine]
        part[f"{s}.cells"] = np.concatenate(per)
        part[f"{s}.offsets"] = np.r_[0, np.cumsum([len(p) for p in per])]
    np.savez(os.path.join(out, f"part_{args.task:03d}.npz"), **part)
    log.info("task %d: %d primaries, seconds per primary median %.1f max %.1f, particles median %.3g max %.3g",
             args.task, len(mine), np.median(secs), np.max(secs), np.median(n_part), np.max(n_part))
    return {"task": args.task, "primaries": int(len(mine)), "seconds median / max": [float(np.median(secs)),
            float(np.max(secs))], "particles median / max": [float(np.median(n_part)), float(np.max(n_part))]}


def stage_merge(out, args, log):
    z = np.load(os.path.join(out, "energies.npz"))
    n = len(z["rows"])
    sel = np.sort(np.load(args.only)) if args.only else np.arange(n)
    per = {s: [None] * n for s in SPECIES_NAMES}
    for f in sorted(glob.glob(os.path.join(out, "part_*.npz"))):
        p = np.load(f)
        for s in SPECIES_NAMES:
            o, cells = p[f"{s}.offsets"], p[f"{s}.cells"]                 # read each array once
            for a, i in enumerate(p["index"]):
                per[s][i] = cells[o[a]:o[a + 1]]
    missing = [i for i in sel if per[SPECIES_NAMES[0]][i] is None]
    if missing:
        raise ValueError(f"{len(missing)} primaries have no drawn cells (first {missing[:5]})")
    clouds = {}
    for s in SPECIES_NAMES:
        mine = [per[s][i] for i in sel]
        cap = max(len(c) for c in mine)
        clouds[f"{s}.points"] = np.stack([np.pad(c, ((0, cap - len(c)), (0, 0))) for c in mine]).astype(np.float32)
        clouds[f"{s}.E"] = z[f"{s}.E"][sel]
        log.info("%s: cells per shower median %d, max %d", s, int(np.median([len(c) for c in mine])), cap)
    np.savez(os.path.join(out, args.clouds), rows=z["rows"][sel], labels=z["labels"][sel], **clouds)
    return {"primaries": int(len(sel)), "clouds": args.clouds}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("energies", "draw", "merge"))
    ap.add_argument("--out", required=True, help="the symbolic_trim root")
    ap.add_argument("--name", default="run", help="output folder under particles/draw/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="energies stage: this many primaries, evenly spread")
    ap.add_argument("--primaries", help="energies stage: primaries file of judge.py instead of the scored primaries")
    ap.add_argument("--draws", type=int, default=1, help="energies stage: draws per primary of --primaries")
    ap.add_argument("--fits", help="energies stage: fits file of fit_profiles.py --stage tables (total plane energy) "
                    "instead of the default training fits")
    ap.add_argument("--noise", choices=("plane", "peak", "skew", "table"), default="plane",
                    help="energies stage: plane noise per plane number (adopted), or per position relative to the peak: normal (peak), Johnson SU (skew) or the real leftovers as a table (table)")
    ap.add_argument("--laws", default=LAWS, help="draw stage: particle laws, relative to --out")
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--clouds", default="clouds.npz", help="merge stage: output file name")
    ap.add_argument("--only", help="draw and merge stages: .npy of entry indices of energies.npz to draw / merge "
                    "(indices and seeds unchanged)")
    ap.add_argument("--offset", help="draw stage: JSON {\"species class\": [kappa, rho]} of the plane offset "
                    "(module docstring, step 6b); none if not given")
    ap.add_argument("--caps", choices=("old", "new"), default="old",
                    help="draw stage: render caps, old (h5_files_v3, rescaled to E_k) or new (v2 / 150k, no rescale)")
    ap.add_argument("--width-law", help="draw stage: width_sr.py law, relative to --out; required with a non-default --laws")
    ap.add_argument("--time-law", help="draw stage: times_sr.py law, relative to --out, or none for the ring-by-ring laws; required with a non-default --laws")
    ap.add_argument("--energy-law", help="draw stage: energy_sr.py or tiny_net.py law, relative to --out, or none for the ring-by-ring laws; required with a non-default --laws")
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, args.name), args)
    stage = {"energies": stage_energies, "draw": stage_draw, "merge": stage_merge}[args.stage]
    res = stage(out, args, log)
    logs.save(out, res, args, log)


if __name__ == "__main__":
    main()
