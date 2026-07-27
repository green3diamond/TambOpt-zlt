"""Check #2 — can voxel-compacted clouds reproduce full-cloud kernel labels?

Kernel-direct optimization on the WHOLE corpus (design doc A1/A2) is only
affordable if the placed clouds compress to something GPU-resident. The Gaussian
kernel (sigma = SIGMA_SPATIAL) cannot resolve structure much finer than sigma,
so points are pooled per (North, Up, layer) voxel:

    energy channel : per-voxel [energy-weighted centroid, sum(energy)] — the
                     kernel is evaluated at the centroid, exact to O((delta/sigma)^2).
    time channel   : the kernel's arrival_time divides by the PADDED point count
                     P, so the per-voxel time sum is pre-scaled by (Vmax / P);
                     the mean over Vmax voxels then reproduces (1/P)*sum(t*kernel).

For each candidate voxel size delta this prints label fidelity (R^2 in the
surrogate's log1p spaces, worst-case relative errors), the compression ratio +
extrapolated full-corpus resident size, and the utility error |U_voxel - U_full|
through the frozen recon on grid and center layouts.

    cd TambOpt/detector_optimization_v6
    python plots/check_compaction_fidelity.py --n-events 512 --deltas 125,62.5,31.25
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4
import showerdata
from modules_v6.fnn_surrogate_ne import compute_labels_batch, place_clouds_enu
from modules_v6.dual_surrogate import combine_species_outputs
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v6.tr_geometry_ne import sample_initial_layout_ne, project_to_mountain_ne
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.opt_core import utility_of_xy, load_models
from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, T_LOG_SCALE, SIGMA_SPATIAL,
    DUAL_SHOWER_CACHE_PATH, DUAL_POSITIONS_PATH, TRAINING_DATASET_FOLDER,
)
from modules_v6.voxel_corpus import voxelize, pad_ragged, LAYER_BIN


def layout(mountain, scheme):
    e, n = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme=scheme)
    e, n = project_to_mountain_ne(mountain,
                                  torch.as_tensor(np.asarray(e)).float().reshape(-1),
                                  torch.as_tensor(np.asarray(n)).float().reshape(-1))
    return e.float(), n.float()


def load_placed(B, device):
    positions_all = torch.load(DUAL_POSITIONS_PATH)
    n_pairs = positions_all.shape[0] // 2
    B = min(B, n_pairs)
    e_sub = showerdata.load(DUAL_SHOWER_CACHE_PATH, start=0, stop=B)
    m_sub = showerdata.load(DUAL_SHOWER_CACHE_PATH, start=n_pairs, stop=n_pairs + B)
    elec = torch.as_tensor(e_sub.points, dtype=torch.float32)
    muon = torch.as_tensor(m_sub.points, dtype=torch.float32)
    dirs = torch.as_tensor(e_sub.directions, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    pos = positions_all[:B].float()
    place_clouds_enu(elec, pos, dirs, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX)
    place_clouds_enu(muon, pos, dirs, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX)
    return elec.to(device), muon.to(device), B, n_pairs


def voxelize_padded(clouds, delta):
    """(B, P, 5) placed clouds -> (B, Vmax, 5) kernel-ready voxels + counts (B,).
    Thin wrapper over the shared modules_v6.voxel_corpus voxelize/pad_ragged."""
    vals, counts = voxelize(clouds, delta)
    starts = torch.cumsum(counts, 0) - counts
    return pad_ragged(vals, counts, starts, clouds.shape[1]), counts


@torch.no_grad()
def labels(clouds, e_det, n_det, surface, chunk=128):
    Es, Ts = [], []
    for i in range(0, clouds.shape[0], chunk):
        E, T = compute_labels_batch(clouds[i:i + chunk], e_det, n_det, surface)
        Es.append(E); Ts.append(T)
    return torch.cat(Es), torch.cat(Ts)


def to_pred(E_e, T_e, E_m, T_m):
    pred_e = torch.stack([torch.log1p(E_e), torch.log1p(T_e * T_LOG_SCALE)], dim=-1)
    pred_m = torch.stack([torch.log1p(E_m), torch.log1p(T_m * T_LOG_SCALE)], dim=-1)
    return combine_species_outputs(pred_e, pred_m)


class StaticLabels:
    def __init__(self, pred):
        self.pred = pred

    def __call__(self, primary_batch, xy_batch):
        return self.pred


def r2(a, b):
    """R^2 of b against a (a = reference), flattened."""
    a, b = a.reshape(-1), b.reshape(-1)
    ss_res = ((a - b) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum().clamp(min=1e-30)
    return float(1.0 - ss_res / ss_tot)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-events", type=int, default=512)
    ap.add_argument("--deltas", type=str, default="125,62.5,31.25",
                    help="comma-separated voxel sizes [m] (sigma_spatial = %.0f)"
                         % SIGMA_SPATIAL)
    args = ap.parse_args()
    deltas = [float(d) for d in args.deltas.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("check #2 — voxel compaction fidelity vs full-cloud kernel labels")
    print("=" * 72)
    print(f"device {device}   sigma_spatial {SIGMA_SPATIAL} m   layer bin {LAYER_BIN}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain).to(device)
    elec, muon, B, n_pairs = load_placed(args.n_events, device)
    P = elec.shape[1]
    real_pts = float((elec[..., 3] > 0).sum() + (muon[..., 3] > 0).sum()) / B
    print(f"events {B} of {n_pairs} pairs   padded P {P}   "
          f"real pts/pair {real_pts:.0f}")

    _, recon = load_models(device)
    prim = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")
                      ).float()[:B].to(device)
    layouts = {s: tuple(t.to(device) for t in layout(mountain, s))
               for s in ("grid", "center")}

    # ── full-cloud reference labels + U per layout ───────────────────────────
    ref = {}
    for name, (x, y) in layouts.items():
        E_e, T_e = labels(elec, x, y, surface)
        E_m, T_m = labels(muon, x, y, surface)
        pred = to_pred(E_e, T_e, E_m, T_m)
        with torch.no_grad():
            U, _, _ = utility_of_xy(x, y, prim, StaticLabels(pred), recon)
        ref[name] = (E_e, T_e, E_m, T_m, pred, float(U.item()))
        print(f"full clouds  U_true({name:6s}) = {float(U.item()):9.4f}")

    # ── per-delta fidelity ───────────────────────────────────────────────────
    for delta in deltas:
        ve, ce = voxelize_padded(elec, delta)
        vm, cm = voxelize_padded(muon, delta)
        vox_per_pair = float(ce.float().mean() + cm.float().mean())
        ratio = real_pts / max(vox_per_pair, 1.0)
        full_gib = n_pairs * vox_per_pair * 5 * 4 / 2 ** 30    # ragged CSR estimate
        print()
        print(f"-- delta = {delta:g} m (sigma/{SIGMA_SPATIAL / delta:.0f})  "
              f"voxels/pair {vox_per_pair:.0f}  compression x{ratio:.1f}  "
              f"full-corpus ~{full_gib:.2f} GiB (ragged fp32)")
        for name, (x, y) in layouts.items():
            E_e, T_e, E_m, T_m, pred_full, U_full = ref[name]
            vE_e, vT_e = labels(ve, x, y, surface)
            vE_m, vT_m = labels(vm, x, y, surface)
            pred_vox = to_pred(vE_e, vT_e, vE_m, vT_m)
            with torch.no_grad():
                U_vox, _, _ = utility_of_xy(x, y, prim, StaticLabels(pred_vox), recon)

            lE  = torch.log1p(E_e + E_m);   lvE = torch.log1p(vE_e + vE_m)
            r2_E = r2(lE, lvE)
            r2_T = r2(pred_full[..., 1], pred_vox[..., 1])
            rel = ((vE_e + vE_m) - (E_e + E_m)).abs() / (E_e + E_m).clamp(min=1e-6)
            rel = rel[(E_e + E_m) > 1e-3]                       # where counts exist
            print(f"   {name:6s}  R2 log1p(E) {r2_E:.6f}   R2 log1p(T) {r2_T:.6f}   "
                  f"relerr(E) p95 {rel.quantile(0.95):.4f} max {rel.max():.4f}   "
                  f"U {float(U_vox.item()):9.4f} (dU {float(U_vox.item()) - U_full:+.4f})")

    print()
    print("PASS target: R2 > 0.99 on both channels and |dU| << the ~1-2 unit")
    print("differences the optimizer is chasing, at a delta whose full-corpus")
    print("size fits GPU memory.")


if __name__ == "__main__":
    main()
