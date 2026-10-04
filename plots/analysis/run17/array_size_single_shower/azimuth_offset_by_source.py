"""Is the clouds' azimuth offset (cloud_axis_check.py: +9 deg) in the AllShowers generator or in its
training showers? For each species, the same horizontal fit on (a) the C8 training showers the
generator learned from (first --n of combined_<species>.h5) and (b) the run 17 generated held-out
showers: an energy-weighted principal axis of the raw (East, North) offsets through the decay point,
its azimuth minus the labelled direction's. Showers within 20 deg of vertical are skipped (azimuth
ill-defined). Photons carry no charge: a geomagnetic origin would leave them unrotated.

    python azimuth_offset_by_source.py [--n 300]
"""
import argparse
import sys

import numpy as np
import showerdata
import torch

sys.path.insert(0, "/n/home05/zdimitrov/tambo/TambOpt")   # branch run17-bwcap50 (run 17 constants)
import modules.constants as C  # noqa: E402

TRAIN = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_{}s.h5"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
n = ap.parse_args().n


def offsets(path, start, stop):
    sh = showerdata.load(path, start=start, stop=stop)
    d = np.asarray(sh.directions, dtype=float)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pts = np.asarray(sh.points, dtype=float)
    out, outu = [], []
    for i in range(len(d)):
        x = pts[i]
        w = x[:, 3]
        k = w > 0
        if k.sum() < 30 or np.hypot(d[i, 0], d[i, 1]) < np.sin(np.radians(20)):
            continue
        xy, w = x[k, :2], w[k]
        az_t = np.arctan2(d[i, 1], d[i, 0])
        for wt, dst in ((w, out), (np.ones_like(w), outu)):
            v = np.linalg.eigh((xy * wt[:, None]).T @ xy)[1][:, -1]
            v = v if v @ d[i, :2] > 0 else -v
            dst.append((np.degrees(np.arctan2(v[1], v[0]) - az_t) + 180) % 360 - 180)
    return np.array(out), np.array(outu)


def row(name, a, au):
    print(f"{name:34s} energy-weighted median {np.median(a):+6.1f} deg (16-84% {np.percentile(a, 16):+6.1f} .."
          f" {np.percentile(a, 84):+6.1f})   unweighted median {np.median(au):+6.1f} deg   [{len(a)} showers]",
          flush=True)


n_pairs = showerdata.get_file_length(C.HELDOUT_SHOWER_CACHE_PATH) // len(C.SPECIES_NAMES)
for s, sp in enumerate(C.SPECIES_NAMES):
    row(f"{sp:8s} C8 training showers", *offsets(TRAIN.format(sp), 0, n))
    row(f"{sp:8s} generated (run 17)", *offsets(C.HELDOUT_SHOWER_CACHE_PATH, s * n_pairs, s * n_pairs + n))
