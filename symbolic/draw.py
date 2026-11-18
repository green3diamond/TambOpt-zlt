"""Draw showers from a model folder (model.py), as particles through the preprocessing, in the cloud format.

Per primary and species: total energy per plane E_k and Xmax from part 1 (sampler.py), then per plane with energy:
  1. ring energies     E_k x the rings law's shares, stretched in radius by the width law (AR(1) from plane to plane)
  2. particles         per ring Poisson(ring energy / mean particle energy (energy.py)), at most ring / E_MIN
  3. their energies    E_MIN each plus the ring's excess split by Burr XII draws (spread.py): the ring energy is exact
  4. positions         uniform in area within the ring (the open last ring up to R_OPEN x its inner edge), any angle
  5. times             light along the axis + the ring's lead + Exp(mean delay - lead) (times.py)
  6. render            the per-simulation preprocessing (render.py, caps data.CAP), which decides what is stored.

Stages: energies (part 1 for every primary of a primaries file, DRAWS times each, one seed), draw (task i of N takes
every N-th entry, one seed per entry, so the split does not matter), merge (padded clouds.npz).
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

from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere, data, energy, lateral, longitudinal, model, render, rings, rows, sampler, spread, times, width

R_OPEN = 1.4


def sample_energies(part1, l, u, X, labels, rng):
    """Plane energies {species: (n, planes)} and Xmax {species: (n,)} from part 1, class by class."""
    E = {s: np.zeros((len(l), N_PLANES)) for s in SPECIES_NAMES}
    xmax = {s: np.zeros(len(l)) for s in SPECIES_NAMES}
    for c, cn in data.CLASS.items():
        m = labels == c
        e, Z, redrawn = sampler.sample(part1[cn], l[m], u[m], X[m], rng)
        print(f"part 1 {cn}: {m.sum()} primaries, {redrawn} redrawn", flush=True)
        for j, s in enumerate(SPECIES_NAMES):
            E[s][m] = e[s]
            xmax[s][m] = longitudinal.from_latent(sampler.species_block(Z, j)[0], l[m])[0]
    return E, xmax


def ring_particles(rng, k, e_ring, mean, ck, e_min, lead, tau):
    """Particles (n, 5) [plane k, across, along, energy, time] of one plane from its ring energies, mean particle
    energies, Burr (c, k) and times [ns] (steps 2-5)."""
    mean = np.maximum(mean, e_min * 1.001)
    n = np.minimum(rng.poisson(e_ring / mean), np.floor(e_ring / e_min)).astype(int)
    if not n.any():
        return np.zeros((0, 5))
    j = np.repeat(np.arange(data.N_RING), n)
    v = np.concatenate([spread.sample(rng, *ck[r], m) for r, m in enumerate(n) if m])
    excess = e_ring - n * e_min
    e = e_min + excess[j] * v / np.bincount(j, v, data.N_RING)[j]
    lo, hi = data.EDGES[j], data.EDGES[j + 1]
    hi = np.where(np.isfinite(hi), hi, R_OPEN * lo)
    r, phi = np.sqrt(rng.uniform(lo ** 2, hi ** 2)), rng.uniform(0, 2 * np.pi, len(j))
    t = (k + 1) * atmosphere.PLANE_SPACING / rows.C_LIGHT + (lead[j] + tau[j] * rng.exponential(size=len(j))) * 1e-9
    return np.column_stack([np.full(len(j), k), r * np.cos(phi), r * np.sin(phi), e, t])


def plane_particles(rng, k, x, e_k, law, s, log_lam):
    """Particles of one plane of species s; x: its feature row, log_lam: its log10 width."""
    shares = rings.stretch(softmax(x @ law["rings"]), log_lam)
    lead = np.maximum(10.0 ** times.ring_times(law["lead"], x, k) - 1, 0.0)
    tau = np.maximum(10.0 ** times.ring_times(law["delay"], x, k) - 1 - lead, 0.0)
    return ring_particles(rng, k, e_k * shares, np.exp(energy.ring_means(law["energy"], x, k)), law["spread"],
                          spread.E_MIN[s], lead, tau)


def shower(job):
    """Stored cells per species of one entry -> (index, {species: cells}, seconds, particles drawn)."""
    i, seed, l, u, cn, X, d, E, xmax, laws = job
    rng = np.random.default_rng([seed, i])
    t0, out, n_part = time.time(), {}, 0
    for s in SPECIES_NAMES:
        ks = np.nonzero(E[s] > 0)[0]
        law, w = laws[(s, cn)], laws[(s, cn)]["width"]
        r_ref = np.array([lateral.r_m_ref(k, u) for k in ks])
        x = rows.features(longitudinal.age(X[ks], xmax[s]), np.full(len(ks), l), np.full(len(ks), u), np.log(E[s][ks]), r_ref)
        ll = width.mean(w, ks, x[:, 3], x[:, 1]) + width.sd(w, ks) * sampler.ar1(1, N_PLANES, w[-1], rng)[0][ks]
        p = [plane_particles(rng, k, x[a], E[s][k], law, s, ll[a]) for a, k in enumerate(ks)]
        p = np.concatenate(p) if p else np.zeros((0, 5))
        n_part += len(p)
        cells = render.render(p, d, data.CAP[s]) if len(p) else np.zeros((0, 5))
        drawn = np.bincount(p[:, 0].astype(int), p[:, 3], N_PLANES) if len(p) else np.zeros(N_PLANES)
        if (np.bincount(cells[:, 2].astype(int), cells[:, 3], N_PLANES) > drawn * (1 + 1e-9) + 1e-9).any():
            raise ValueError(f"entry {i} {s}: stored energy above the drawn energy")
        out[s] = cells
    return i, out, time.time() - t0, n_part


def stage_energies(args):
    z = np.load(args.primaries)
    idx = np.repeat(np.arange(len(z["rows"])), args.draws)
    rows_, l, u, labels, X, dirs = (z[k][idx] for k in ("rows", "l", "u", "labels", "X", "dirs"))
    part1, _ = model.load(args.model)
    E, xmax = sample_energies(part1, l, u, X, labels, np.random.default_rng(args.seed))
    np.savez(os.path.join(args.out, "energies.npz"), rows=rows_, labels=labels, l=l, u=u, X=X, dirs=dirs,
             **{f"{s}.E": E[s] for s in SPECIES_NAMES}, **{f"{s}.xmax": xmax[s] for s in SPECIES_NAMES})
    return {"entries": len(rows_), "draws": args.draws}


def stage_draw(args):
    z = np.load(os.path.join(args.out, "energies.npz"))
    _, laws = model.load(args.model)
    mine = np.arange(len(z["rows"]))[args.task::args.tasks]
    if args.only:
        mine = mine[np.isin(mine, np.load(args.only))]
    jobs = [(i, args.seed, float(z["l"][i]), float(z["u"][i]), data.CLASS[int(z["labels"][i])], z["X"][i], z["dirs"][i],
             {s: z[f"{s}.E"][i] for s in SPECIES_NAMES}, {s: float(z[f"{s}.xmax"][i]) for s in SPECIES_NAMES}, laws)
            for i in mine]
    cells, secs, n_part = {s: {} for s in SPECIES_NAMES}, [], []
    with Pool(args.workers) as pool:
        for i, c, dt, n in pool.imap_unordered(shower, jobs):
            for s in SPECIES_NAMES:
                cells[s][i] = c[s].astype(np.float32)
            secs.append(dt)
            n_part.append(n)
    part = {"index": mine}
    for s in SPECIES_NAMES:
        per = [cells[s][i] for i in mine]
        part[f"{s}.cells"] = np.concatenate(per)
        part[f"{s}.offsets"] = np.r_[0, np.cumsum([len(p) for p in per])]
    np.savez(os.path.join(args.out, f"part_{args.task:03d}.npz"), **part)
    return {"task": args.task, "entries": int(len(mine)), "seconds median / max": [float(np.median(secs)), float(np.max(secs))],
            "particles median / max": [float(np.median(n_part)), float(np.max(n_part))]}


def stage_merge(args):
    z = np.load(os.path.join(args.out, "energies.npz"))
    sel = np.sort(np.load(args.only)) if args.only else np.arange(len(z["rows"]))
    per = {s: [None] * len(z["rows"]) for s in SPECIES_NAMES}
    for f in sorted(glob.glob(os.path.join(args.out, "part_*.npz"))):
        p = np.load(f)
        for s in SPECIES_NAMES:
            o, c = p[f"{s}.offsets"], p[f"{s}.cells"]
            for a, i in enumerate(p["index"]):
                per[s][i] = c[o[a]:o[a + 1]]
    missing = [i for i in sel if per[SPECIES_NAMES[0]][i] is None]
    if missing:
        raise ValueError(f"{len(missing)} entries have no drawn cells (first {missing[:5]})")
    clouds = {}
    for s in SPECIES_NAMES:
        mine = [per[s][i] for i in sel]
        cap = max(len(c) for c in mine)
        clouds[f"{s}.points"] = np.stack([np.pad(c, ((0, cap - len(c)), (0, 0))) for c in mine]).astype(np.float32)
        clouds[f"{s}.E"] = z[f"{s}.E"][sel]
    np.savez(os.path.join(args.out, "clouds.npz"), rows=z["rows"][sel], labels=z["labels"][sel], **clouds)
    return {"entries": int(len(sel))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("energies", "draw", "merge"))
    ap.add_argument("--model", required=True, help="model folder")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--primaries", help="energies stage: primaries file (judge.py primaries)")
    ap.add_argument("--draws", type=int, default=8, help="energies stage: draws per primary")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", help="draw and merge stages: .npy of the entries to draw (indices and seeds unchanged)")
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    res = {"energies": stage_energies, "draw": stage_draw, "merge": stage_merge}[args.stage](args)
    res.update(stage=args.stage, args=vars(args), git=data.git_head(), minutes=round((time.time() - t0) / 60, 2))
    with open(os.path.join(args.out, f"{args.stage}_{args.task:03d}.json" if args.stage == "draw" else f"{args.stage}.json"), "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
