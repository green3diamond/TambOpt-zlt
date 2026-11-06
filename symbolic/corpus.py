"""A pipeline corpus drawn by the symbolic model at the primaries of an existing Step-0 corpus, in its format: species-major
blocks of the same events (rows s*N + i are event i's species s), the EM/hadronic class in the pdg field, the source's
energies, directions and shower ids, and its _positions.pt / _species.pt sidecars, so Stages 1-4 read it unchanged.
Events outside the zenith range the model was fitted on (ZENITH_MIN-ZENITH_MAX) are drawn with their zenith clamped
into it for the plane depths and the law inputs, and with their own direction for placing the cells; they are listed
in <name>_zenith_clamped.npy.

Stages (tambo env):
  primaries  the source corpus' events (species blocks checked to agree; --limit keeps the first N) -> <name>_primaries.npz
  draw       task i of --tasks: a contiguous slice of the events, plane energies from the Part-1 sampler and cells from
             draw.shower with the default laws, one seed per task -> parts/<name>_part<i>.npz
  merge      every part into <name>.pt (padded to the source's point cap), the sidecars copied, then the written file read
             back and its primaries checked against the source row by row.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np

from modules.constants import N_PLANES, SPECIES_NAMES, ZENITH_MAX, ZENITH_MIN
from symbolic import atmosphere, draw, planes

ROOT = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/v6_runs/symbolic_trim"


def source_rows(per, n):
    """Rows of the source's first n events in every species block."""
    return np.concatenate([s * per + np.arange(n) for s in range(len(SPECIES_NAMES))])


def stage_primaries(args):
    import showerdata
    inc = showerdata.load_inc_particles(args.src)
    per = len(inc.energies) // len(SPECIES_NAMES)
    n = min(per, args.limit) if args.limit else per
    E, d, lab = (np.asarray(x).reshape(len(SPECIES_NAMES), per, -1)[:, :n] for x in (inc.energies, inc.directions, inc.pdg))
    if not (np.allclose(E, E[:1]) and np.allclose(d, d[:1], atol=1e-6) and (lab == lab[:1]).all()):
        raise ValueError(f"{args.src}: species blocks do not share their events")
    dirs = d[0] / np.linalg.norm(d[0], axis=1, keepdims=True)
    zenith = np.degrees(np.arccos(dirs[:, 2]))
    used = np.clip(zenith, ZENITH_MIN, ZENITH_MAX)
    l, u = np.log10(E[0, :, 0].astype(np.float64)), np.cos(np.radians(used))
    np.savez(os.path.join(args.dst, f"{args.name}_primaries.npz"), per=per, energies=E[0], directions=d[0], labels=lab[0, :, 0],
             l=l, u=u, dirs=dirs, X=atmosphere.plane_depths(used, N_PLANES), clamped=used != zenith)
    print(f"{args.name}: {n} of {per} events, log10 E {l.min():.2f}-{l.max():.2f}, classes {np.bincount(lab[0, :, 0])}, "
          f"zenith clamped on {int((used != zenith).sum())}", flush=True)


def task_slice(n, task, tasks):
    edges = np.linspace(0, n, tasks + 1).astype(int)
    return edges[task], edges[task + 1]


def stage_draw(args):
    z = np.load(os.path.join(args.dst, f"{args.name}_primaries.npz"))
    a, b = task_slice(len(z["l"]), args.task, args.tasks)
    l, u, X, lab, dirs = (z[k][a:b] for k in ("l", "u", "X", "labels", "dirs"))
    E, xmax = draw.sample_energies(ROOT, l, u, X, lab, np.random.default_rng([args.seed, args.task]))
    laws = draw.draw_laws(ROOT)
    jobs = [(i, args.seed * 1_000_003 + a, float(l[i]), float(u[i]), planes.CLASS[int(lab[i])], X[i], dirs[i],
             {s: E[s][i] for s in SPECIES_NAMES}, {s: float(xmax[s][i]) for s in SPECIES_NAMES}, laws)
            for i in range(b - a)]
    cells = {s: [None] * (b - a) for s in SPECIES_NAMES}
    with Pool(args.workers) as pool:
        for i, c, _, _ in pool.imap_unordered(draw.shower, jobs, chunksize=4):
            for s in SPECIES_NAMES:
                cells[s][i] = c[s][c[s][:, 3] > 0].astype(np.float32)
    os.makedirs(os.path.join(args.dst, "parts"), exist_ok=True)
    out = {"start": a, "stop": b}
    for s in SPECIES_NAMES:
        out[f"{s}.cells"] = np.concatenate(cells[s])
        out[f"{s}.offsets"] = np.r_[0, np.cumsum([len(c) for c in cells[s]])]
    path = os.path.join(args.dst, "parts", f"{args.name}_part{args.task:04d}.npz")
    np.savez(path + ".tmp.npz", **out)
    os.replace(path + ".tmp.npz", path)
    print(f"task {args.task}: events {a}-{b}, cells per event median {np.median(np.diff(out[f'{SPECIES_NAMES[0]}.offsets'])):.0f}", flush=True)


def stage_merge(args):
    import h5py
    import showerdata
    import torch
    z = np.load(os.path.join(args.dst, f"{args.name}_primaries.npz"))
    n, per = len(z["l"]), int(z["per"])
    parts = sorted(glob.glob(os.path.join(args.dst, "parts", f"{args.name}_part*.npz")))
    spans = [tuple(np.load(p)[k] for k in ("start", "stop")) for p in parts]
    if [int(a) for a, _ in spans] != [0] + [int(b) for _, b in spans[:-1]] or int(spans[-1][1]) != n:
        raise ValueError(f"parts do not cover events 0-{n} in order: {spans[:3]} ...")
    rows = source_rows(per, n)
    with h5py.File(args.src, "r") as f:
        ids = f["shower_ids"][:][rows]
    cap = showerdata.get_file_shape(args.src)[1]
    out = os.path.join(args.dst, f"{args.name}.pt")
    showerdata.create_empty_file(out, shape=(len(rows), cap, 5), overwrite=True)
    for k, s in enumerate(SPECIES_NAMES):
        for p in parts:
            q = np.load(p)
            a, b, o, c = int(q["start"]), int(q["stop"]), q[f"{s}.offsets"], q[f"{s}.cells"]
            pts = np.zeros((b - a, cap, 5), np.float32)
            for i in range(b - a):
                pts[i, : o[i + 1] - o[i]] = c[o[i]:o[i + 1]]
            sl = slice(k * n + a, k * n + b)
            showerdata.save_batch(showerdata.Showers(points=pts, energies=z["energies"][a:b], pdg=z["labels"][a:b],
                                                     directions=z["directions"][a:b], shower_ids=ids[sl]), out, start=k * n + a)
        print(f"{s}: block {k} written ({n} events)", flush=True)
    base = os.path.splitext(args.src)[0]
    for side in ("_positions", "_species"):
        torch.save(torch.load(base + side + ".pt")[torch.as_tensor(rows)], os.path.join(args.dst, f"{args.name}{side}.pt"))
    np.save(os.path.join(args.dst, f"{args.name}_zenith_clamped.npy"), z["clamped"])
    check(args.src, out, rows)
    with open(out + ".symbolic.json", "w") as f:
        json.dump({"source": args.src, "events": n, "rows": int(len(rows)), "seed": args.seed, "parts": len(parts)}, f, indent=1)


def check(src, out, rows):
    """The written corpus has the source's primaries, ids and sidecars at every row, and non-empty showers."""
    import h5py
    import showerdata
    import torch
    a, b = showerdata.load_inc_particles(src), showerdata.load_inc_particles(out)
    same = [np.array_equal(getattr(a, k)[rows], getattr(b, k)) for k in ("energies", "directions", "pdg")]
    with h5py.File(src, "r") as fa, h5py.File(out, "r") as fb:
        same.append(np.array_equal(fa["shower_ids"][:][rows], fb["shower_ids"][:]))
        n_pts = fb["num_points"][:]
    base, mine = os.path.splitext(src)[0], os.path.splitext(out)[0]
    for side in ("_positions", "_species"):
        same.append(torch.equal(torch.load(base + side + ".pt")[torch.as_tensor(rows)], torch.load(mine + side + ".pt")))
    print(f"check: energies, directions, pdg, ids, positions, species equal to the source: {same}; "
          f"rows {len(n_pts)}, empty showers {int((n_pts == 0).sum())}, points per row median {np.median(n_pts):.0f}", flush=True)
    if not all(same):
        raise ValueError("the written corpus does not match the source's primaries")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("primaries", "draw", "merge"))
    ap.add_argument("--src", help="source Step-0 corpus (.pt)")
    ap.add_argument("--dst", required=True, help="output folder")
    ap.add_argument("--name", required=True, help="output corpus name, e.g. cashed_showers_tau_dual")
    ap.add_argument("--limit", type=int, default=0, help="primaries stage: the first N events only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    {"primaries": stage_primaries, "draw": stage_draw, "merge": stage_merge}[args.stage](args)


if __name__ == "__main__":
    main()
