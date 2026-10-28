"""The deployed neural chain (PointCountFM, anti-clip re-roll, AllShowers) at the scored test primaries, for comparison.

Stages (one GPU):
  draws   M draws per primary, kept as energy per plane -> OUT/chain/draws/draw_XX.npz (score.py)
  clouds  the full cell clouds of draw 0 -> OUT/chain/clouds_draw00.npz (detector.py --chain); their plane sums must
          reproduce draws/draw_00.npz
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py
import numpy as np
import torch

from analysis_core import logs
from modules.constants import SPECIES_NAMES
from symbolic import planes, sources

FOLDER = "chain"
DRAWS = os.path.join(FOLDER, "draws")
CLOUDS = os.path.join(FOLDER, "clouds_draw00.npz")
N_PER_CLASS = 500
TOP = 6.125
SEED0 = 7000                   # draw d of species k uses torch seed SEED0 + 100 d + k
RTOL = 1e-3


def primaries():
    """N_PER_CLASS random test rows per class below TOP (fixed seed): rows, energies, directions, classes."""
    with h5py.File(sources.corsika_path(SPECIES_NAMES[0], "test"), "r") as f:
        E = f["energies"][...].reshape(-1).astype(np.float64)
        D = f["directions"][...].astype(np.float32)
        lab = f["pdg"][...].astype(np.int64)
    rng = np.random.default_rng(0)
    rows = np.sort(np.concatenate([rng.choice(np.nonzero((lab == c) & (np.log10(E) < TOP))[0], N_PER_CLASS,
                                              replace=False) for c in (0, 1)]))
    return rows, E[rows].astype(np.float32).reshape(-1, 1), D[rows], lab[rows]


def run(draws, log):
    """{draw: {species: (B, cap, 5) clouds}}, building each species' generator once."""
    rows, E, D, lab = primaries()
    s0 = sources.stage0()
    from allshowers.generator import Generator                # importable once stage0 has set up the pinned code
    energies, directions, labels = torch.as_tensor(E), torch.as_tensor(D), torch.as_tensor(lab)
    ids = torch.arange(len(rows))
    out = {d: {} for d in draws}
    for k, s in enumerate(SPECIES_NAMES):
        sources.weights(s)
        gen = Generator(run_dir=os.path.join(sources.STAGED, s), num_timesteps=s0.NUM_TIMESTEPS, compile=True,
                        solver=s0.SOLVER)
        gen.max_points = sources.CAP[s]
        for d in draws:
            torch.manual_seed(SEED0 + 100 * d + k)
            sh = s0._gen_chunk(gen, os.path.join(sources.STAGED, s, "pcfm_compiled.pt"), {"max_points": sources.CAP[s]},
                               s, energies, directions, labels, ids, sources.CAP[s])
            out[d][s] = torch.as_tensor(sh.points, dtype=torch.float32)
            log.info("%s draw %d done", s, d)
        del gen
        torch.cuda.empty_cache()
    return rows, out


def save(path, log, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    log.info("wrote %s (%d bytes)", path, os.path.getsize(path))


def stage_draws(args, log):
    rows = primaries()[0]
    todo = []
    for d in range(args.first, args.last + 1):
        path = os.path.join(args.out, DRAWS, f"draw_{d:02d}.npz")
        if os.path.exists(path):
            with np.load(path) as z:
                if np.array_equal(z["rows"], rows):
                    log.info("draw %d already done, skipped", d)
                    continue
            raise ValueError(f"{path} exists with other rows")
        todo.append(d)
    _, out = run(todo, log)
    for d in todo:
        save(os.path.join(args.out, DRAWS, f"draw_{d:02d}.npz"), log, rows=rows, draw=d,
             **{f"{s}.E": planes.from_clouds(out[d][s])[0] for s in SPECIES_NAMES})
    return {"drawn": todo}


def stage_clouds(args, log):
    ref = np.load(os.path.join(args.out, DRAWS, "draw_00.npz"))
    rows, out = run([0], log)
    if not np.array_equal(ref["rows"], rows):
        raise ValueError("draw_00 was made on other primaries")
    res = {}
    for s in SPECIES_NAMES:
        a, b = planes.from_clouds(out[0][s])[0], ref[f"{s}.E"]
        res[s] = float(np.median(np.abs(a - b)[b > 0] / b[b > 0]))
        log.info("%s: plane sums against draw_00, median relative difference %.2e", s, res[s])
        if res[s] > RTOL:
            raise ValueError(f"{s}: clouds do not reproduce draw_00")
    save(os.path.join(args.out, CLOUDS), log, rows=rows, **{f"{s}.points": out[0][s].numpy() for s in SPECIES_NAMES})
    return {"plane_sums_vs_draw_00": res}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=["draws", "clouds"])
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=7)
    args = ap.parse_args()
    log_dir, log = logs.start(os.path.join(FOLDER, "logs", args.stage), args)
    stage = {"draws": stage_draws, "clouds": stage_clouds}[args.stage]
    logs.save(log_dir, stage(args, log), args, log)


if __name__ == "__main__":
    main()
