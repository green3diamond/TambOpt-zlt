"""Check #1 — cost and existence of exact kernel gradients w.r.t. the layout.

Kernel-direct optimization (design doc A1/A2) needs one number before anything
else: the wall-clock cost of ONE full-corpus forward+backward through the
plane-aware kernel, gradients accumulated into the 60-detector (E, N) tensor —
i.e. the cost of a single L-BFGS closure call with the surrogate removed.

Two measurements:
  1. TIMING — stream the whole corpus in chunks, place, run
     `compute_labels_batch(differentiable=True)` per species, backprop a scalar
     touching both label channels; report disk/place vs GPU time per event and
     the extrapolated per-closure cost.
  2. TRUE GRADIENT AT THE GRID (--recon-events) — exact ∂U_true/∂(E, N) through
     the FROZEN recon with kernel labels on a resident event subset (chunked
     backward is exact: the composite U is a per-event mean). Answers whether
     the grid layout sits on a true-U plateau (|grad| ~ 0) or there is real
     signal to ascend.

    cd TambOpt/detector_optimization_v6
    python plots/check_kernel_grad_timing.py --n-events 0    # 0 = whole corpus
"""
import argparse
import os
import sys
import time

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
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, T_LOG_SCALE,
    DUAL_SHOWER_CACHE_PATH, DUAL_POSITIONS_PATH, TRAINING_DATASET_FOLDER,
)


def grid_layout(mountain):
    e, n = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme="grid")
    e, n = project_to_mountain_ne(mountain,
                                  torch.as_tensor(np.asarray(e)).float().reshape(-1),
                                  torch.as_tensor(np.asarray(n)).float().reshape(-1))
    return e.float(), n.float()


def load_placed_chunk(positions_all, n_pairs, start, stop):
    """Load + place electron and muon clouds for pair rows [start, stop)."""
    e_sub = showerdata.load(DUAL_SHOWER_CACHE_PATH, start=start, stop=stop)
    m_sub = showerdata.load(DUAL_SHOWER_CACHE_PATH, start=n_pairs + start, stop=n_pairs + stop)
    elec = torch.as_tensor(e_sub.points, dtype=torch.float32)
    muon = torch.as_tensor(m_sub.points, dtype=torch.float32)
    dirs = torch.as_tensor(e_sub.directions, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    pos = positions_all[start:stop].float()
    place_clouds_enu(elec, pos, dirs, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX)
    place_clouds_enu(muon, pos, dirs, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX)
    return elec, muon


class KernelDualGrad:
    """Grad-enabled, chunk-internal kernel label source with the dual FNN's
    `(primary_batch, xy_batch) -> (B, n_det, 2)` signature, for events whose
    placed clouds are held resident. Rows of a call share one layout."""

    def __init__(self, elec, muon, surface, chunk):
        self.elec, self.muon, self.surface, self.chunk = elec, muon, surface, chunk

    def __call__(self, primary_batch, xy_batch):
        e_det, n_det = xy_batch[0, :, 0], xy_batch[0, :, 1]
        outs = []
        for i in range(0, self.elec.shape[0], self.chunk):
            E_e, T_e = compute_labels_batch(self.elec[i:i + self.chunk], e_det, n_det,
                                            self.surface, differentiable=True)
            E_m, T_m = compute_labels_batch(self.muon[i:i + self.chunk], e_det, n_det,
                                            self.surface, differentiable=True)
            pred_e = torch.stack([torch.log1p(E_e), torch.log1p(T_e * T_LOG_SCALE)], dim=-1)
            pred_m = torch.stack([torch.log1p(E_m), torch.log1p(T_m * T_LOG_SCALE)], dim=-1)
            outs.append(combine_species_outputs(pred_e, pred_m))
        return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-events", type=int, default=0,
                    help="pairs to stream for the timing pass; 0 = whole corpus")
    ap.add_argument("--chunk", type=int, default=128,
                    help="events per kernel fwd+bwd (graph memory scales with it)")
    ap.add_argument("--recon-events", type=int, default=4096,
                    help="resident events for the exact U_true gradient; 0 skips")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("check #1 — grad-enabled kernel: full-corpus fwd+bwd timing")
    print("=" * 72)
    print(f"device : {device}   chunk : {args.chunk}")
    if device.type != "cuda":
        print("WARNING: no GPU — timings will not answer the A100 question.")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain).to(device)
    e_g, n_g = grid_layout(mountain)
    x = e_g.clone().to(device).requires_grad_(True)
    y = n_g.clone().to(device).requires_grad_(True)

    positions_all = torch.load(DUAL_POSITIONS_PATH)
    n_pairs = positions_all.shape[0] // 2
    N = n_pairs if args.n_events == 0 else min(args.n_events, n_pairs)
    print(f"corpus : {n_pairs} pairs, streaming {N}")

    # ── warmup (CUDA allocator + kernels) ────────────────────────────────────
    elec_w, muon_w = load_placed_chunk(positions_all, n_pairs, 0, min(args.chunk, N))
    for cl in (elec_w.to(device), muon_w.to(device)):
        E, T = compute_labels_batch(cl, x, y, surface, differentiable=True)
        (E.sum() + T.sum()).backward()
    x.grad = None; y.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    # ── timing pass ──────────────────────────────────────────────────────────
    t_load = t_gpu = 0.0
    t_all0 = time.perf_counter()
    for start in range(0, N, args.chunk):
        stop = min(start + args.chunk, N)
        t0 = time.perf_counter()
        elec, muon = load_placed_chunk(positions_all, n_pairs, start, stop)
        t1 = time.perf_counter()
        elec, muon = elec.to(device), muon.to(device)
        E_e, T_e = compute_labels_batch(elec, x, y, surface, differentiable=True)
        E_m, T_m = compute_labels_batch(muon, x, y, surface, differentiable=True)
        loss = (torch.log1p(E_e + E_m).sum()
                + torch.log1p((T_e + T_m).clamp(min=0) * T_LOG_SCALE).sum())
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_load += t1 - t0
        t_gpu += t2 - t1
    t_all = time.perf_counter() - t_all0

    gx, gy = x.grad.detach(), y.grad.detach()
    print()
    print(f"streamed {N} pairs in {t_all:.1f} s "
          f"(disk+place {t_load:.1f} s | transfer+kernel fwd+bwd {t_gpu:.1f} s)")
    print(f"per-pair: {1e3 * t_all / N:.3f} ms total, {1e3 * t_gpu / N:.3f} ms GPU")
    if N < n_pairs:
        print(f"extrapolated full corpus ({n_pairs}): "
              f"{t_all / N * n_pairs:.1f} s total, {t_gpu / N * n_pairs:.1f} s GPU "
              f"per closure call")
    else:
        print(f"one full-corpus closure call: {t_all:.1f} s "
              f"({t_gpu:.1f} s GPU-side)")
    if device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"scalar-probe grad norms: |dE| {gx.norm():.3e}  |dN| {gy.norm():.3e} "
          f"(nonzero = gradients flow)")

    # ── exact true-U gradient at the grid through the frozen recon ───────────
    if args.recon_events > 0:
        B = min(args.recon_events, n_pairs)
        print()
        print(f"exact dU_true/d(E,N) at the grid layout, {B} events, "
              f"chunked backward (exact: U is a per-event mean)")
        elec, muon = load_placed_chunk(positions_all, n_pairs, 0, B)
        elec, muon = elec.to(device), muon.to(device)
        kernel_fnn = KernelDualGrad(elec, muon, surface, args.chunk)
        _, recon = load_models(device)
        prim = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")
                          ).float()[:B].to(device)

        x2 = e_g.clone().to(device).requires_grad_(True)
        y2 = n_g.clone().to(device).requires_grad_(True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        U_total = 0.0
        for i in range(0, B, args.chunk):
            j = min(i + args.chunk, B)
            sub = KernelDualGrad(elec[i:j], muon[i:j], surface, args.chunk)
            U, _, _ = utility_of_xy(x2, y2, prim[i:j], sub, recon)
            w = (j - i) / B
            (U * w).backward()
            U_total += float(U.item()) * w
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        g = torch.stack([x2.grad, y2.grad], dim=-1)           # (n_det, 2)
        per_det = g.norm(dim=-1)
        print(f"U_true(grid, {B} events) = {U_total:.4f}   [{t1 - t0:.1f} s per U+grad]")
        print(f"|dU/d(E,N)| total {g.norm():.4e}   per-detector "
              f"max {per_det.max():.4e}  median {per_det.median():.4e}  "
              f"min {per_det.min():.4e}")
        print("interpretation: per-detector max ~ median >> min≈0 patterns show which")
        print("detectors still see true-U signal; all ≈ 0 -> the grid is a true plateau.")


if __name__ == "__main__":
    main()
