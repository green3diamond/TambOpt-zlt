"""Detector-level check: generated showers, repaired CORSIKA and optionally the neural chain, through the pipeline's
kernel and shower-frame placement on the grid layout, at the same test primaries placed at held-out decay vertices.
Reports lit detectors, deposited energy and arrival time, also by distance from the shower axis, for all events and
for those at least FIELD_CUT deg from the East-West axis (the run 20 corpus cut). One GPU.

The run (geometry, held-out corpus, kernel settings) comes from modules/constants.py and its TAMBOPT_* environment:
source slurm/run20_guard.sh (or run18_guard.sh) first.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "plots", "layouts")):
    sys.path.insert(0, _p)

import h5py
import numpy as np
import torch

import showerdata  # noqa: E402
from true_utility import KernelDualLabels, PlacedClouds, grid_layout  # noqa: E402
from modules import constants as C  # noqa: E402
from modules.data import detectors_in_frames, place_clouds, shower_frames  # noqa: E402
from modules.geometry import SurfaceUpMap, load_tr_mountain  # noqa: E402
from modules.optimize import LAYOUT_THRESHOLD  # noqa: E402
from symbolic import fit_profiles, planes, sources, unwrap  # noqa: E402

KERNEL_POINTS = 3e6       # events x points per kernel chunk
BLOCK = 25                # events per kernel block
FIELD_CUT = 30.0          # deg from the +-East axis, along which the C8 training showers' magnetic field points
DIST_BINS = np.array([0, 50, 100, 200, 400, 800, 1600, np.inf])
DIST_LABELS = ["0-50", "50-100", "100-200", "200-400", "400-800", "800-1600", ">1600"]


def run_settings():
    return {k: getattr(C, k) for k in ("RUN_LOCATION", "GEOMETRY_PATH_RESOLVED", "GEOMETRY_GROUP", "KERNEL_FRAME",
                                       "KERNEL_TRANSVERSE", "SIGMA_SPATIAL", "KERNEL_INTERP_K", "KERNEL_INTERP_MAX")}


def corsika_clouds(rows, dirs):
    """Repaired CORSIKA clouds per species at the test rows."""
    out = {}
    for s in C.SPECIES_NAMES:
        with h5py.File(sources.corsika_path(s, "test"), "r") as f:
            raw = [np.asarray(f["showers"][r], np.float64).reshape(-1, 5) for r in rows]
        out[s] = [unwrap.unwrap_shower(c[c[:, 3] > 0], d, C.N_PLANES)[0] for c, d in zip(raw, dirs)]
    return out


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
    ang = np.degrees(np.arccos(np.clip(np.sum(dirs * hdir[match], 1), -1, 1)))
    print(f"vertex match: angle median {np.median(ang):.2f} deg, 90% {np.percentile(ang, 90):.2f}", flush=True)
    return pos_all[:n_pairs][torch.as_tensor(match)].float()


def place(per_species, pos, dirs, event_ids):
    """Native clouds of a block of events -> PlacedClouds in the run's kernel frame."""
    placed, frames = [], None
    for k, s in enumerate(C.SPECIES_NAMES):
        v = per_species[s]
        c = torch.as_tensor(padded(v) if isinstance(v, (list, tuple)) else np.asarray(v, np.float32))
        fr = place_clouds(c, pos, torch.as_tensor(dirs, dtype=torch.float32), event_ids=event_ids, species=k,
                          east_entry=C.EAST_ENTRY, layer_east_dx=C.LAYER_EAST_DX)
        frames = fr if frames is None else frames
        placed.append(c)
    return PlacedClouds(placed, frames)


def response(per_species, pos, dirs, surface, xy, device, event_ids=None):
    """Deposited energy and leading-edge time [s] per event and detector (0 where nothing arrived)."""
    n = len(per_species[C.SPECIES_NAMES[0]])
    ids = torch.arange(n) if event_ids is None else torch.as_tensor(event_ids)
    E, T = [], []
    for lo in range(0, n, BLOCK):
        sl = slice(lo, min(n, lo + BLOCK))
        placed = place({s: per_species[s][sl] for s in C.SPECIES_NAMES}, pos[sl], dirs[sl], ids[sl])
        chunk = max(1, int(KERNEL_POINTS // max(c.shape[1] for c in placed)))
        lab = KernelDualLabels(placed, surface, device, chunk=chunk)(None, xy)
        E.append(torch.expm1(lab[..., 0]).clamp(min=0).cpu().numpy())
        T.append((torch.expm1(lab[..., 1].clamp(min=0)) / C.T_LOG_SCALE).cpu().numpy())
        del placed, lab
        torch.cuda.empty_cache()
    return np.concatenate(E), np.concatenate(T)


def axis_distance(dirs, pos, e_det, n_det, surface):
    """Distance of each detector from the shower axis, in the kernel's shower frame (NaN outside the planes)."""
    frames = shower_frames(pos, torch.as_tensor(dirs, dtype=torch.float32))
    with torch.no_grad():
        rho, _, z = detectors_in_frames(e_det.float(), n_det.float(), surface, frames, ring=True)
    rho, z = rho.numpy(), z.numpy()
    return np.where((z >= 1) & (z <= C.N_PLANES), rho, np.nan)


def outside_field_cut(dirs):
    return np.degrees(np.arccos(np.clip(np.abs(dirs[:, 0]), 0, 1))) >= FIELD_CUT


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


def setup(device):
    """Mountain surface, grid layout (E, N on the device) and the layout's Up, for the run's geometry."""
    mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                                layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain)
    e_det, n_det = grid_layout(mountain)
    return surface, e_det, n_det, torch.stack([e_det, n_det], -1)[None].to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="output folder of fit_profiles.py and generator.py (--out there)")
    ap.add_argument("--clouds", required=True, help="generator clouds (.npz with rows and <species>.points)")
    ap.add_argument("--chain", default="", help="the neural chain's clouds at the same or more primaries (optional)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for k in ("clouds", "chain", "out"):
        if getattr(args, k):
            setattr(args, k, os.path.join(args.root, getattr(args, k)))
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    gen = np.load(args.clouds)
    rows = gen["rows"]
    lab = np.load(os.path.join(args.root, fit_profiles.TEST))["meta.labels"][rows]
    with h5py.File(sources.corsika_path(C.SPECIES_NAMES[0], "test"), "r") as f:
        dirs = f["directions"][...][rows].astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pos = vertices(dirs)
    clouds = {"CORSIKA": corsika_clouds(rows, dirs), "generator": {s: gen[f"{s}.points"] for s in C.SPECIES_NAMES}}
    if args.chain:
        ch = np.load(args.chain)
        sel = np.searchsorted(ch["rows"], rows)
        if not np.array_equal(ch["rows"][np.minimum(sel, len(ch["rows"]) - 1)], rows):
            raise ValueError("generator clouds are on primaries the chain does not have")
        clouds["neural chain"] = {s: ch[f"{s}.points"][sel].astype(np.float32) for s in C.SPECIES_NAMES}
    surface, e_det, n_det, xy = setup(device)
    dist = axis_distance(dirs, pos, e_det, n_det, surface)
    surface = surface.to(device)
    resp = {k: response(per, pos, dirs, surface, xy, device) for k, per in clouds.items()}
    keep = outside_field_cut(dirs)
    res = {"run": run_settings(), "threshold": float(LAYOUT_THRESHOLD), "clouds": args.clouds, "chain": args.chain}
    for c, cn in planes.CLASS.items():
        for tag, m in (("all", lab == c), (f">= {FIELD_CUT:.0f} deg from the field axis", (lab == c) & keep)):
            ref_E, ref_T = resp["CORSIKA"][0][m], resp["CORSIKA"][1][m]
            res.setdefault(cn, {})[tag] = {"n_events": int(m.sum()),
                                           **{k: metrics(v[0][m], v[1][m], ref_E, ref_T, dist[m]) for k, v in resp.items()}}
        print(cn, json.dumps(res[cn], indent=1), flush=True)
    np.savez_compressed(os.path.join(args.out, "response.npz"), rows=rows, labels=lab, distance=dist,
                        **{f"{k}.E": v[0] for k, v in resp.items()}, **{f"{k}.T": v[1] for k, v in resp.items()})
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
