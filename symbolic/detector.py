"""Detector-level check: the model's draws and CORSIKA's stored cells at the judge's primaries, placed at the run's
held-out decay vertices nearest in direction, through the pipeline's kernel on the grid layout (one GPU).

Per class and draw: lit detectors per event (all, by distance from the axis, outside the planes, share of events with
at least 5 / 10 / 20 lit), deposited energy against CORSIKA (median per-event ratio over the events both see, ratio
of means), share of events seen, median leading-edge shift. Then the noise floor over the draws: a row is a GAP when
|mean - CORSIKA| > 2 sd sqrt(1 + 1/n), sd the draw-to-draw spread.

The run (geometry, held-out corpus, kernel settings) comes from modules/constants.py and its TAMBOPT_* environment:
source slurm/run20_guard.sh first.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "plots", "layouts"))

import numpy as np
import showerdata
import torch
from true_utility import KernelDualLabels, PlacedClouds, grid_layout

from modules import constants as C
from modules.data import detectors_in_frames, place_clouds_shower, shower_frames
from modules.geometry import SurfaceUpMap, load_tr_mountain
from modules.optimize import LAYOUT_THRESHOLD
from symbolic import data, judge

KERNEL_POINTS = 3e6       # events x points per kernel chunk
BLOCK = 25                # events per kernel block
DIST_BINS = np.array([0, 50, 100, 200, 400, 800, 1600, np.inf])
DIST_LABELS = ["0-50", "50-100", "100-200", "200-400", "400-800", "800-1600", ">1600"]


def padded(clouds):
    out = np.zeros((len(clouds), max(1, max(len(c) for c in clouds)), 5), np.float32)
    for i, c in enumerate(clouds):
        out[i, :len(c)] = c
    return out


def vertices(dirs):
    """Decay vertices of the run's held-out events nearest in direction."""
    pos_all = torch.load(C.HELDOUT_POSITIONS_PATH)
    n_pairs = pos_all.shape[0] // len(C.SPECIES_NAMES)
    inc = showerdata.load_inc_particles(C.HELDOUT_SHOWER_CACHE_PATH, start=0, stop=n_pairs)
    hdir = np.asarray(inc.directions, np.float64)
    hdir /= np.linalg.norm(hdir, axis=1, keepdims=True)
    match = np.argmax(dirs @ hdir.T, 1)
    return pos_all[:n_pairs][torch.as_tensor(match)].float()


def response(per_species, pos, dirs, surface, xy, device):
    """Deposited energy and leading-edge time [s] per event and detector (0 where nothing arrived)."""
    n = len(per_species[C.SPECIES_NAMES[0]])
    E, T = [], []
    for lo in range(0, n, BLOCK):
        sl = slice(lo, min(n, lo + BLOCK))
        placed, frames = [], None
        for k, s in enumerate(C.SPECIES_NAMES):
            c = torch.as_tensor(padded(per_species[s][sl]))
            fr = place_clouds_shower(c, pos[sl], torch.as_tensor(dirs[sl], dtype=torch.float32),
                                     event_ids=torch.arange(sl.start, sl.stop), species=k)
            frames = fr if frames is None else frames
            placed.append(c)
        placed = PlacedClouds(placed, frames)
        chunk = max(1, int(KERNEL_POINTS // max(c.shape[1] for c in placed)))
        lab = KernelDualLabels(placed, surface, device, chunk=chunk)(None, xy)
        E.append(torch.expm1(lab[..., 0]).clamp(min=0).cpu().numpy())
        T.append((torch.expm1(lab[..., 1].clamp(min=0)) / C.T_LOG_SCALE).cpu().numpy())
        del placed, lab
        torch.cuda.empty_cache()
    return np.concatenate(E), np.concatenate(T)


def axis_distance(dirs, pos, e_det, n_det, surface):
    """Distance of each detector from the shower axis in the kernel's shower frame (NaN outside the planes)."""
    frames = shower_frames(pos, torch.as_tensor(dirs, dtype=torch.float32))
    with torch.no_grad():
        rho, z = detectors_in_frames(e_det.float(), n_det.float(), surface, frames)
    return np.where((z.numpy() >= 1) & (z.numpy() <= C.N_PLANES), rho.numpy(), np.nan)


def metrics(E, T, ref_E, ref_T, dist):
    on = E > LAYOUT_THRESHOLD
    lit = on.sum(1)
    both = on & (ref_E > LAYOUT_THRESHOLD) & (T > 0) & (ref_T > 0)
    dt = (T - ref_T)[both] * 1e9
    seen, ref_seen = E.sum(1) > LAYOUT_THRESHOLD, ref_E.sum(1) > LAYOUT_THRESHOLD
    j = np.digitize(np.where(np.isfinite(dist), dist, -1), DIST_BINS) - 1
    out = {"lit per event": float(lit.mean()),
           **{f"lit {b} m": float((on & (j == i)).sum() / len(E)) for i, b in enumerate(DIST_LABELS[:5])},
           "lit outside the planes": float((on & (j < 0)).sum() / len(E)),
           **{f"frac lit >= {n}": float(np.mean(lit >= n)) for n in (5, 10, 20)},
           "deposit, per-event ratio, events both see": float(10 ** np.median(np.log10(E.sum(1)[seen & ref_seen]
                                                                                     / ref_E.sum(1)[seen & ref_seen]))),
           "deposit, ratio of means": float(E.sum(1).mean() / ref_E.sum(1).mean()),
           "frac events seen": float(seen.mean()),
           "leading edge median [ns]": float(np.median(dt)) if len(dt) else float("nan")}
    return out


def noise_floor(per_draw, corsika):
    """Per row: CORSIKA, the mean and sd over draws, z = (mean - CORSIKA) / (sd sqrt(1 + 1/n)), GAP if |z| > 2."""
    n = len(per_draw)
    out = {}
    for k, ref in corsika.items():
        v = np.array([d[k] for d in per_draw])
        m, s = v.mean(), v.std(ddof=1)
        z = (m - ref) / (s * np.sqrt(1 + 1 / n)) if s > 0 else (0.0 if m == ref else np.inf * np.sign(m - ref))
        out[k] = {"CORSIKA": ref, "mean": float(m), "sd": float(s), "z": float(z), "gap": bool(abs(z) > 2), "draws": v.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primaries", required=True, help="judge.py primaries file")
    ap.add_argument("--clouds", required=True, help="the model's clouds, --draws entries per primary")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.primaries)
    g = np.load(args.clouds)
    draw = judge.draw_index(g["rows"])
    prim = np.unique(g["rows"])
    if draw.max() + 1 != args.draws or (np.bincount(g["rows"])[prim] != args.draws).any():
        raise ValueError(f"expected {args.draws} draws at every primary of the clouds")
    dirs = z["dirs"][prim] / np.linalg.norm(z["dirs"][prim], axis=1, keepdims=True)
    labels = z["labels"][prim]
    truth = [data.stored_cells(str(z["sims"][i]), z["dirs"][i]) for i in prim]
    corsika = {s: [t[s].astype(np.float32) for t in truth] for s in C.SPECIES_NAMES}
    device = torch.device("cuda")
    mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                                layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain)
    e_det, n_det = grid_layout(mountain)
    xy = torch.stack([e_det, n_det], -1)[None].to(device)
    pos = vertices(dirs)
    dist = axis_distance(dirs, pos, e_det, n_det, surface)
    surface = surface.to(device)
    ref = response(corsika, pos, dirs, surface, xy, device)
    res = {"run": C.RUN_LOCATION, "threshold": float(LAYOUT_THRESHOLD), "clouds": args.clouds, "primaries": len(prim)}
    per_draw = {cn: [] for cn in data.CLASS.values()}
    for k in range(args.draws):
        idx = np.nonzero(draw == k)[0]
        idx = idx[np.argsort(g["rows"][idx])]
        mine = {s: [p[p[:, 3] > 0] for p in g[f"{s}.points"][idx]] for s in C.SPECIES_NAMES}
        E, T = response(mine, pos, dirs, surface, xy, device)
        for c, cn in data.CLASS.items():
            m = labels == c
            per_draw[cn].append(metrics(E[m], T[m], ref[0][m], ref[1][m], dist[m]))
        print(f"draw {k} done", flush=True)
    for c, cn in data.CLASS.items():
        m = labels == c
        corsika_rows = metrics(ref[0][m], ref[1][m], ref[0][m], ref[1][m], dist[m])
        res[cn] = {"events": int(m.sum()), "noise floor": noise_floor(per_draw[cn], corsika_rows)}
        print(f"\n{cn} ({m.sum()} events): row | CORSIKA | mean | sd | z | gap", flush=True)
        for row, q in res[cn]["noise floor"].items():
            print(f"  {row:44s} {q['CORSIKA']:8.3f} {q['mean']:8.3f} {q['sd']:7.3f} {q['z']:6.2f} {'GAP' if q['gap'] else 'ok'}")
    with open(os.path.join(args.out, "results.json"), "w") as h:
        json.dump(res, h, indent=1)


if __name__ == "__main__":
    main()
