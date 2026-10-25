"""Run 20 (Malata): copy a Step-0 corpus keeping only events whose direction is >= --min-angle-b (3-D angle) from the
+-x (East) axis, along which the C8 training showers' magnetic field points (showers within ~30 deg of it come out of
C8 shifted sideways; scripts/17_make_plane_inputs.py applies the same cut when it draws run 17-19's primaries).

Copies the main corpus and the held-out reserve with their _positions.pt / _species.pt sidecars, so every later stage
reads the filtered corpus unchanged. Rows stay species-major: block s holds the kept events in their original order.
Streams --chunk rows at a time; resumes from <out>.progress.json.

    python scripts/20_filter_corpus.py --src <run 15>/v6_run_00 --dst <run 20>/v6_run_00 [--min-angle-b 30]
"""
import argparse
import json
import os
import sys

import numpy as np
import showerdata
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from modules.constants import SPECIES_NAMES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="Step-0 folder with cashed_showers_tau_dual*.pt")
ap.add_argument("--dst", required=True)
ap.add_argument("--min-angle-b", type=float, default=30.0)
ap.add_argument("--chunk", type=int, default=2000)
args = ap.parse_args()
os.makedirs(args.dst, exist_ok=True)
N_SP = len(SPECIES_NAMES)


def keep_events(path):
    """Event-level keep mask (from species block 0) and the per-species direction check."""
    inc = showerdata.load_inc_particles(path)
    d = np.asarray(inc.directions, float)
    per = len(d) // N_SP
    blocks = d.reshape(N_SP, per, 3)
    if not np.allclose(blocks, blocks[:1], atol=1e-5):
        raise RuntimeError(f"{path}: species blocks do not share event directions")
    u = blocks[0] / np.linalg.norm(blocks[0], axis=1, keepdims=True)
    ang_b = np.degrees(np.arccos(np.clip(np.abs(u[:, 0]), 0.0, 1.0)))
    return ang_b >= args.min_angle_b, per


def filter_file(name):
    src, dst = os.path.join(args.src, name + ".pt"), os.path.join(args.dst, name + ".pt")
    keep, per = keep_events(src)
    k = int(keep.sum())
    rows = np.concatenate([s * per + np.flatnonzero(keep) for s in range(N_SP)])   # kept rows, species-major
    print(f"[{name}] {per} events -> {k} kept ({100 * k / per:.1f}%), {len(rows)} rows", flush=True)
    for side in ("_positions", "_species"):                                      # row-aligned sidecars
        torch.save(torch.load(os.path.join(args.src, name + side + ".pt"))[torch.as_tensor(rows)],
                   os.path.join(args.dst, name + side + ".pt"))
    prog_path = dst + ".progress.json"
    done = json.load(open(prog_path))["rows_done"] if os.path.exists(prog_path) else 0
    if done == 0:
        showerdata.create_empty_file(dst, shape=(len(rows), showerdata.get_file_shape(src)[1], 5), overwrite=True)
    for lo in range(done, len(rows), args.chunk):
        sel = rows[lo:lo + args.chunk]                                           # ascending, within one species
        sh = showerdata.load(src, start=int(sel[0]), stop=int(sel[-1]) + 1)
        showerdata.save_batch(sh[sel - sel[0]], dst, start=lo)
        json.dump(dict(rows_done=lo + len(sel), rows_total=len(rows)), open(prog_path + ".tmp", "w"))
        os.replace(prog_path + ".tmp", prog_path)
        if (lo // args.chunk) % 25 == 0:
            print(f"  [{name}] {lo + len(sel)}/{len(rows)} rows", flush=True)
    json.dump(dict(src=src, min_angle_b=args.min_angle_b, events_in=per, events_kept=k), open(dst + ".filter.json", "w"))
    print(f"[{name}] done -> {dst}", flush=True)


filter_file("cashed_showers_tau_dual_heldout")
filter_file("cashed_showers_tau_dual")
