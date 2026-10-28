"""Detector-level check: generated showers, repaired CORSIKA and optionally the neural chain, through the pipeline
kernel on the grid layout at the same test primaries. Reports lit detectors, deposited energy and arrival time,
also by distance from the shower axis. One GPU.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py
import numpy as np
import torch

import analysis_core.run  # noqa: E402,F401  (puts plots/layouts on the path)
import showerdata  # noqa: E402
from true_utility import KernelDualLabels, grid_layout  # noqa: E402
from data_lane import sources  # noqa: E402
from modules.constants import (DET_KEY, EAST_ENTRY, GEOMETRY_GROUP, GEOMETRY_PATH_RESOLVED,  # noqa: E402
                               HELDOUT_SHOWER_CACHE_PATH, LAYER_EAST_DX, N_PLANES,
                               SPECIES_NAMES, T_LOG_SCALE)
from modules.data import place_clouds_enu  # noqa: E402
from modules.geometry import SurfaceUpMap, load_tr_mountain  # noqa: E402
from modules.optimize import LAYOUT_THRESHOLD  # noqa: E402
from symbolic import atmosphere, m1, unwrap  # noqa: E402

KERNEL_POINTS = 3e6       # events x points per kernel chunk
BLOCK = 25                # events per kernel block
DIST_BINS = np.array([0, 50, 100, 200, 400, 800, 1600, np.inf])
DIST_LABELS = ["0-50", "50-100", "100-200", "200-400", "400-800", "800-1600", ">1600"]


def corsika_clouds(rows, dirs):
    """Repaired CORSIKA clouds per species at the test rows."""
    out = {}
    for s in SPECIES_NAMES:
        with h5py.File(sources.corsika_path(s, "test"), "r") as f:
            raw = [np.asarray(f["showers"][r], np.float64).reshape(-1, 5) for r in rows]
        out[s] = [unwrap.unwrap_shower(c[c[:, 3] > 0], d, N_PLANES)[0] for c, d in zip(raw, dirs)]
    return out


def padded(clouds):
    out = np.zeros((len(clouds), max(len(c) for c in clouds), 5), np.float32)
    for i, c in enumerate(clouds):
        out[i, :len(c)] = c
    return out


def vertices(dirs, heldout):
    """Decay vertices taken from the held-out corpus primary nearest in direction."""
    pos_all = torch.load(os.path.splitext(heldout)[0] + "_positions.pt")
    n_pairs = pos_all.shape[0] // len(SPECIES_NAMES)
    inc = showerdata.load_inc_particles(heldout, start=0, stop=n_pairs)
    hdir = np.asarray(inc.directions, np.float64)
    hdir /= np.linalg.norm(hdir, axis=1, keepdims=True)
    match = np.argmax(dirs @ hdir.T, 1)
    ang = np.degrees(np.arccos(np.clip(np.sum(dirs * hdir[match], 1), -1, 1)))
    print(f"vertex match: angle median {np.median(ang):.2f} deg, 90% {np.percentile(ang, 90):.2f}", flush=True)
    return pos_all[:n_pairs][torch.as_tensor(match)].float()


def response(per_species, pos, dirs, surface, xy, device):
    """Deposited energy and leading-edge time [s] per event and detector (0 where nothing arrived)."""
    n = len(per_species[SPECIES_NAMES[0]])
    E, T = [], []
    for lo in range(0, n, BLOCK):
        sl = slice(lo, min(n, lo + BLOCK))
        placed = []
        for s in SPECIES_NAMES:
            v = per_species[s][sl]
            c = torch.as_tensor(padded(v) if isinstance(v, (list, tuple)) else np.asarray(v, np.float32))
            place_clouds_enu(c, pos[sl], torch.as_tensor(dirs[sl], dtype=torch.float32), east_entry=EAST_ENTRY,
                             layer_east_dx=LAYER_EAST_DX)
            placed.append(c)
        chunk = max(1, int(KERNEL_POINTS // max(c.shape[1] for c in placed)))
        lab = KernelDualLabels(placed, surface, device, chunk=chunk)(None, xy)
        E.append(torch.expm1(lab[..., 0]).clamp(min=0).cpu().numpy())
        T.append((torch.expm1(lab[..., 1].clamp(min=0)) / T_LOG_SCALE).cpu().numpy())
        del placed, lab
        torch.cuda.empty_cache()
    return np.concatenate(E), np.concatenate(T)


def axis_distance(dirs, pos, det_e, det_n, det_u):
    """Distance of each detector from the shower axis in the kernel's frame (NaN outside the planes)."""
    k = np.arange(N_PLANES)
    axis = np.zeros((len(dirs), N_PLANES, 5), np.float32)
    axis[:, :, :2] = ((k + 1) * atmosphere.PLANE_SPACING)[None, :, None] * dirs[:, None, :2]
    axis[:, :, 2], axis[:, :, 3] = k, 1.0
    placed = place_clouds_enu(torch.as_tensor(axis), pos.cpu(), torch.as_tensor(dirs, dtype=torch.float32),
                              east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX).numpy()
    z_det = (EAST_ENTRY - det_e) / LAYER_EAST_DX               # the kernel's continuous plane coordinate
    out = np.full((len(dirs), len(det_e)), np.nan)
    for i, p in enumerate(placed):
        o = np.argsort(p[:, 2])
        z, n, u = p[o, 2], p[o, 0], p[o, 1]
        inside = (z_det >= z[0]) & (z_det <= z[-1])
        out[i, inside] = np.hypot(det_n[inside] - np.interp(z_det[inside], z, n),
                                  det_u[inside] - np.interp(z_det[inside], z, u))
    return out


def metrics(E, T, ref_E, ref_T, dist):
    lit, ref_lit = (E > LAYOUT_THRESHOLD).sum(1), (ref_E > LAYOUT_THRESHOLD).sum(1)
    diff = lit - ref_lit
    both = (E > LAYOUT_THRESHOLD) & (ref_E > LAYOUT_THRESHOLD) & (T > 0) & (ref_T > 0)
    dt = (T - ref_T)[both] * 1e9
    seen = ref_E.sum(1) > LAYOUT_THRESHOLD
    j = np.digitize(np.where(np.isfinite(dist), dist, -1), DIST_BINS) - 1
    on = E > LAYOUT_THRESHOLD
    return {"lit per event mean": round(float(lit.mean()), 2),
            "lit rel diff to CORSIKA": round(float(diff.mean() / ref_lit.mean()), 4),
            "lit paired diff mean +- se": [round(float(diff.mean()), 2), round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 2)],
            **{f"frac lit >= {n}": round(float(np.mean(lit >= n)), 3) for n in (5, 10, 20)},
            "deposit, median of per-event ratio to CORSIKA": round(float(np.median(E.sum(1)[seen] / ref_E.sum(1)[seen])), 3),
            "log10 deposit per event 16/50/84": np.percentile(np.log10(np.maximum(E.sum(1), 1e-3)), [16, 50, 84]).round(3).tolist(),
            "leading-edge shift [ns], median and 16/84": ([round(float(np.median(dt)), 2)]
                                                           + np.percentile(dt, [16, 84]).round(2).tolist() if len(dt) else None),
            "lit per event by distance from the axis [m]": dict(zip(DIST_LABELS, [round(float((on & (j == b)).sum() / len(E)), 3)
                                                                                  for b in range(len(DIST_LABELS))])),
            "lit per event outside the planes": round(float((on & (j < 0)).sum() / len(E)), 3),
            "P(lit) by distance from the axis [m]": dict(zip(DIST_LABELS, [round(float(on[j == b].mean()), 4)
                                                                           if np.any(j == b) else None
                                                                           for b in range(len(DIST_LABELS))]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="output folder of m1.py and generator.py (--out there)")
    ap.add_argument("--clouds", required=True, help="generator clouds (.npz with rows and <species>.points)")
    ap.add_argument("--chain", default="", help="the neural chain's clouds at the same or more primaries (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default=HELDOUT_SHOWER_CACHE_PATH,
                    help="held-out shower corpus whose decay vertices place the showers (default: the pipeline's run)")
    args = ap.parse_args()
    for k in ("clouds", "chain", "out"):
        if getattr(args, k):
            setattr(args, k, os.path.join(args.root, getattr(args, k)))
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    gen = np.load(args.clouds)
    rows = gen["rows"]
    zt = np.load(os.path.join(args.root, m1.FOLDER, "fit", "fits_test.npz"))
    lab = zt["meta.labels"][rows]
    with h5py.File(sources.corsika_path(SPECIES_NAMES[0], "test"), "r") as f:
        dirs = f["directions"][...][rows].astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pos = vertices(dirs, args.heldout)
    clouds = {"CORSIKA": corsika_clouds(rows, dirs), "generator": {s: gen[f"{s}.points"] for s in SPECIES_NAMES}}
    if args.chain:
        ch = np.load(args.chain)
        sel = np.searchsorted(ch["rows"], rows)
        if not np.array_equal(ch["rows"][np.minimum(sel, len(ch["rows"]) - 1)], rows):
            raise ValueError("generator clouds are on primaries the chain does not have")
        clouds["neural chain"] = {s: ch[f"{s}.points"][sel].astype(np.float32) for s in SPECIES_NAMES}
    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, east_entry=EAST_ENTRY,
                                layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain)
    e_det, n_det = grid_layout(mountain)
    with torch.no_grad():
        u_det = surface(n_det.float(), e_det.float()).numpy().astype(np.float64)
    dist = axis_distance(dirs, pos, e_det.numpy().astype(np.float64), n_det.numpy().astype(np.float64), u_det)
    surface = surface.to(device)
    xy = torch.stack([e_det, n_det], -1)[None].to(device)
    resp = {k: response(per, pos, dirs, surface, xy, device) for k, per in clouds.items()}
    res = {"threshold": float(LAYOUT_THRESHOLD), "clouds": args.clouds, "chain": args.chain}
    for c, cn in m1.CLASS.items():
        m = lab == c
        ref_E, ref_T = resp["CORSIKA"][0][m], resp["CORSIKA"][1][m]
        res[cn] = {"n_events": int(m.sum()),
                   **{k: metrics(resp[k][0][m], resp[k][1][m], ref_E, ref_T, dist[m]) for k in clouds}}
        print(cn, json.dumps(res[cn], indent=1), flush=True)
    np.savez_compressed(os.path.join(args.out, "response.npz"), rows=rows, labels=lab, distance=dist,
                        **{f"{k}.E": v[0] for k, v in resp.items()}, **{f"{k}.T": v[1] for k, v in resp.items()})
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
