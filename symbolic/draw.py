"""Particle draw (study_plan.md 2.4 step 3): symbolic showers as particles through the preprocessing model, on the
scored primaries (chain.primaries), in the suite's cloud format.

Per primary and species: plane energies E_k and Xmax from the Part-1 sampler (one draw, fixed seed); per plane with
energy, inputs annuli.features; then, with the laws of particles.py (fitted on the v2 raw showers):
  1. energy before the cut   E_k / stored share (stored law)
  2. ring energies           shares from the rings law with the plane-to-plane variety: --variety along (default:
                             the law stretched in radius by the width law along the shower of width_sr.py, AR(1) along
                             the planes), width (stretched by the linear width law of particles.py), shape
                             (stretched and tilted by the shape law, two AR(1) numbers per plane) or k0
                             (K=0 plane noise of traits.py on the logits of the rings inside 1100 m)
  3. particles per ring      Poisson(ring energy / mean particle energy (mean_energy law)), at most ring / E_MIN
  4. particle energies       E_MIN each plus the ring's excess split by Burr XII draws (spread law): the ring energy
                             is exact and no particle falls under CORSIKA's threshold
  5. positions               uniform in area within the ring (the open last ring up to generator.R_OPEN x its inner
                             edge), uniform azimuth
  6. times                   axis light time + lead + Exp(mean delay - lead) (lead and delay laws)
  7. render                  the preprocessing with the OLD caps (render.render, sources.CAP); each plane's stored
                             energy rescaled to E_k
Gate: every drawn shower's stored plane energies equal the sampled E_k.
Stages: energies (sample once and save), draw (task i of N takes every N-th primary, workers within a task, one seed
per primary so the split does not matter), merge (padded clouds.npz with the suite's keys).
"""
import argparse
import glob
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
from symbolic import annuli, atmosphere, chain, fit_profiles, generator, lateral, longitudinal, particles, planes, width_sr, \
    render, sampler, sources, traits

FOLDER = os.path.join("particles", "draw")
LAWS = os.path.join("particles", "fit_v2", "particle_laws.npz")
WIDTH_LAW = os.path.join("particles", "sr_width", width_sr.LAW)
NOISE = os.path.join("traits", "run_{0}_K0", "traits_{0}_K0.pt")
LAW_NAMES = ("stored", "rings", "width", "shape", "mean_energy", "lead", "delay", "spread")


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
            for w in ("mean_energy", "lead", "delay"):
                law[w] = fill_rings(law[w])
            out[(s, cn)] = law
    return out


def width_law_path(laws, width_law):
    """The width law to draw with. Laws fitted on part of the showers need their own, or a judged shower could leak in."""
    if width_law:
        return width_law
    if laws != LAWS:
        raise ValueError("--laws is not the default one: give the --width-law fitted on the same showers")
    return WIDTH_LAW


def load_noise(root, cn):
    """The K=0 plane-noise network as numpy layers [(W, b)], checked against traits' sizes."""
    import torch
    st = torch.load(os.path.join(root, NOISE.format(cn)), map_location="cpu")["state"]
    pre = next(k for k in st if k.endswith("plane.0.weight"))[: -len("0.weight")]
    idx = sorted({int(k[len(pre):].split(".")[0]) for k in st if k.startswith(pre)})
    net = [(st[f"{pre}{i}.weight"].numpy().T.astype(np.float64), st[f"{pre}{i}.bias"].numpy().astype(np.float64))
           for i in idx]
    if net[0][0].shape[0] != traits.P + 8 + len(SPECIES_NAMES) or net[-1][0].shape[1] != particles.OUTER:
        raise ValueError(f"unexpected noise network sizes {[W.shape for W, _ in net]}")
    return net


def noise(net, x, s, rng):
    """Ring-logit offsets (planes, OUTER) of the plane-noise network for species s, one fresh e per plane."""
    one = np.zeros((len(x), len(SPECIES_NAMES)))
    one[:, SPECIES_NAMES.index(s)] = 1.0
    h = np.column_stack([rng.normal(size=(len(x), traits.P)), x, one])
    for j, (W, b) in enumerate(net):
        h = h @ W + b
        if j < len(net) - 1:
            h = np.tanh(h)
    return h


def plane_particles(rng, k, x, e_k, law, s, off=None, log_lam=None, tilt=0.0):
    """Particles (n, 5) [plane k, across, along, energy, time] of one plane before the preprocessing; the variety is
    either K=0 logit offsets `off` or the plane's log10 width `log_lam` and tilt `tilt`."""
    e_tot = e_k / softmax(x @ law["stored"])[0]
    logit = x @ law["rings"]
    if off is not None:
        logit[: particles.OUTER] += off
    shares = softmax(logit) if log_lam is None else particles.stretch(softmax(logit), log_lam, tilt)
    lead = np.maximum(10.0 ** (law["lead"] @ x) - 1, 0.0)
    tau = np.maximum(10.0 ** (law["delay"] @ x) - 1 - lead, 0.0)
    return ring_particles(rng, k, e_tot * shares, np.exp(law["mean_energy"] @ x), law["spread"],
                          particles.E_MIN[s], lead, tau)


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
    hi = np.where(np.isfinite(hi), hi, generator.R_OPEN * lo)
    r, phi = np.sqrt(rng.uniform(lo ** 2, hi ** 2)), rng.uniform(0, 2 * np.pi, len(j))
    t = (k + 1) * atmosphere.PLANE_SPACING / annuli.C_LIGHT + (lead[j] + tau[j] * rng.exponential(size=len(j))) * 1e-9
    return np.column_stack([np.full(len(j), k), r * np.cos(phi), r * np.sin(phi), e, t])


def shower(job):
    """Stored cells per species of one primary -> (index, {species: cells}, seconds, particles drawn)."""
    i, seed, l, u, cn, X, d, E, xmax, laws, nets, variety = job
    rng = np.random.default_rng([seed, i])
    t0, out, n_part = time.time(), {}, 0
    for s in SPECIES_NAMES:
        ks = np.nonzero(E[s] > 0)[0]
        x = annuli.features(generator.age(X[ks], xmax[s]), np.full(len(ks), l), np.full(len(ks), u),
                            np.log(E[s][ks]), np.array([lateral.r_m_ref(k, d[2]) for k in ks]))
        law = laws[(s, cn)]
        if variety == "k0":
            off = noise(nets[cn], x, s, rng)
            p = [plane_particles(rng, k, x[a], E[s][k], law, s, off=off[a]) for a, k in enumerate(ks)]
        else:
            w = law[variety]
            sample = lambda v: x @ v[:-2] + v[-2] * sampler.ar1(1, N_PLANES, v[-1], rng)[0][ks]
            if variety == "along":
                ll = width_sr.law_mean(w, ks, x[:, 3], x[:, 1]) + width_sr.law_sd(w, ks) * sampler.ar1(1, N_PLANES, w[-1], rng)[0][ks]
            else:
                ll = sample(w[: len(w) // 2] if variety == "shape" else w)
            tt = sample(w[len(w) // 2:]) if variety == "shape" else np.zeros(len(ks))
            p = [plane_particles(rng, k, x[a], E[s][k], law, s, log_lam=ll[a], tilt=tt[a])
                 for a, k in enumerate(ks)]
        p = np.concatenate(p) if p else np.zeros((0, 5))
        n_part += len(p)
        cells = render.render(p, d, sources.CAP[s]) if len(p) else np.zeros((0, 5))
        k = cells[:, 2].astype(int)
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


def stage_energies(out, args, log):
    rows, l, u, lab, X, dirs = given_primaries(args) if args.primaries else scored_primaries(args)
    rng = np.random.default_rng(args.seed)
    E = {s: np.zeros((len(rows), N_PLANES)) for s in SPECIES_NAMES}
    xmax = {s: np.zeros(len(rows)) for s in SPECIES_NAMES}
    tr = sampler.load(os.path.join(args.out, fit_profiles.TRAIN))
    for c in planes.CLASS:
        m = lab == c
        e, lat = sampler.sample(sampler.fit(tr, c), l[m], u[m], X[m], rng)
        for j, s in enumerate(SPECIES_NAMES):
            E[s][m] = e[s]
            xmax[s][m] = longitudinal.from_latent(sampler.species_block(lat, j)[0], l[m])[0]
    np.savez(os.path.join(out, "energies.npz"), rows=rows, labels=lab, l=l, u=u, X=X, dirs=dirs,
             **{f"{s}.E": E[s] for s in SPECIES_NAMES}, **{f"{s}.xmax": xmax[s] for s in SPECIES_NAMES})
    log.info("sampled plane energies of %d primaries", len(rows))
    return {"primaries": int(len(rows)), "seed": args.seed}


def stage_draw(out, args, log):
    z = np.load(os.path.join(out, "energies.npz"))
    laws = load_laws(args.out, args.laws)
    if args.variety == "along":
        z_w = np.load(os.path.join(args.out, width_law_path(args.laws, args.width_law)))
        for (s, cn), law in laws.items():
            law["along"] = z_w[f"{s}.{cn}"]
    nets = {cn: load_noise(args.out, cn) for cn in planes.CLASS.values()}
    mine = np.arange(len(z["rows"]))[args.task::args.tasks]
    jobs = [(i, args.seed, float(z["l"][i]), float(z["u"][i]), planes.CLASS[int(z["labels"][i])], z["X"][i],
             z["dirs"][i], {s: z[f"{s}.E"][i] for s in SPECIES_NAMES}, {s: float(z[f"{s}.xmax"][i]) for s in SPECIES_NAMES},
             laws, nets, args.variety) for i in mine]
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
    per = {s: [None] * n for s in SPECIES_NAMES}
    for f in sorted(glob.glob(os.path.join(out, "part_*.npz"))):
        p = np.load(f)
        for s in SPECIES_NAMES:
            o, cells = p[f"{s}.offsets"], p[f"{s}.cells"]                 # read each array once
            for a, i in enumerate(p["index"]):
                per[s][i] = cells[o[a]:o[a + 1]]
    missing = [i for i in range(n) if per[SPECIES_NAMES[0]][i] is None]
    if missing:
        raise ValueError(f"{len(missing)} primaries have no drawn cells (first {missing[:5]})")
    clouds = {}
    for s in SPECIES_NAMES:
        cap = max(len(c) for c in per[s])
        clouds[f"{s}.points"] = np.stack([np.pad(c, ((0, cap - len(c)), (0, 0))) for c in per[s]]).astype(np.float32)
        clouds[f"{s}.E"] = z[f"{s}.E"]
        log.info("%s: cells per shower median %d, max %d", s, int(np.median([len(c) for c in per[s]])), cap)
    np.savez(os.path.join(out, args.clouds), rows=z["rows"], labels=z["labels"], **clouds)
    return {"primaries": n, "clouds": args.clouds}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("energies", "draw", "merge"))
    ap.add_argument("--out", required=True, help="the symbolic_trim root")
    ap.add_argument("--name", default="run", help="output folder under particles/draw/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="energies stage: this many primaries, evenly spread")
    ap.add_argument("--primaries", help="energies stage: primaries file of judge.py instead of the scored primaries")
    ap.add_argument("--draws", type=int, default=1, help="energies stage: draws per primary of --primaries")
    ap.add_argument("--laws", default=LAWS, help="draw stage: particle laws, relative to --out")
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--clouds", default="clouds.npz", help="merge stage: output file name")
    ap.add_argument("--variety", choices=("along", "width", "shape", "k0"), default="along",
                    help="draw stage: plane-to-plane variety")
    ap.add_argument("--width-law", help="draw stage, variety along: width_sr.py law, relative to --out; required with a non-default --laws")
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, args.name), args)
    stage = {"energies": stage_energies, "draw": stage_draw, "merge": stage_merge}[args.stage]
    res = stage(out, args, log)
    logs.save(out, res, args, log)


if __name__ == "__main__":
    main()
