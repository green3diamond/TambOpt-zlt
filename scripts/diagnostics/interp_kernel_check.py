"""Does the interpolated (adaptive-bandwidth) kernel fix the sigma=10 label collapse?

At sigma=10 m, 78% of E and 95% of T detector slots in run 14 read exactly 0,
because a detector only sees AllShowers macro-particles that land within ~20 m.
`adaptive_bandwidths` widens each point's Gaussian to its k-th nearest in-plane
neighbour distance, so a detector between sparse points reads their local density.

Two measurements on the current run's HELD-OUT events and the 7 layout strategies:

  (1) point sampling: per species, the in-plane k-NN distance r_k of points near
      the array, and the fraction with r_k > sigma (where interpolation acts);
  (2) labels, for raw and each k: frac E == 0, frac T == 0, lit-slot log-space
      mean/std, and the deposit these 100-detector layouts collect relative to
      raw. That ratio need not be 1: interpolation moves deposit from between
      points onto detectors. Per-point conservation holds over a fine grid and is
      checked in tests/test_interp_kernel.py.

Read-only: writes stdout and an optional JSON.

    python scripts/diagnostics/interp_kernel_check.py --n-events 256 --json out.json
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _V6)
from _pathfix import V6_ROOT  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import modules  # noqa: F401,E402
from modules.constants import (  # noqa: E402
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX,
    N_PLANES, N_DETECTORS, SIGMA_SPATIAL, SPECIES_NAMES, T_LOG_SCALE, RUN_LOCATION,
)
from modules.data import compute_labels_batch  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.layouts.strategies import _STRATEGIES, _STRATEGY_FNS  # noqa: E402
from modules.showers import GetCounts_planeaware, adaptive_bandwidths  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def labels(clouds, e_det, n_det, surface, bandwidth):
    """compute_labels_batch's exact mapping, with a precomputed bandwidth so it is
    not recomputed for every layout. Parity with compute_labels_batch is asserted
    once per chunk for the raw kernel."""
    up = surface(n_det, e_det)
    z_cont = (EAST_ENTRY - e_det) / LAYER_EAST_DX
    with torch.no_grad():
        return GetCounts_planeaware(clouds, n_det, up, z_cont, None, None, None,
                                    sigma=SIGMA_SPATIAL, bandwidth=bandwidth)


def near_array_mask(clouds, mesh_nu, mesh_z, radius):
    """Real points within `radius` (transverse) of a mesh centroid and within one
    layer of its depth — the points a detector could actually see."""
    real = clouds[..., 3] > 0
    pts = clouds[..., :2]                                                  # (B, P, 2)
    d = torch.cdist(pts.reshape(-1, 2), mesh_nu).reshape(*pts.shape[:2], -1)  # (B, P, M)
    dl = (clouds[..., 2].unsqueeze(-1) - mesh_z).abs()                     # (B, P, M)
    return real & ((d < radius) & (dl < 1.0)).any(dim=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-events", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--near-radius", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    print(f"run={RUN_LOCATION}\nsigma={SIGMA_SPATIAL}  ks={args.ks}  device={DEVICE}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                               east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    C = torch.as_tensor(mountain.centroids_ENU, dtype=torch.float32, device=DEVICE)   # (M, 3) E,N,U
    mesh_nu = C[:, [1, 2]]                                    # kernel transverse = (North, Up)
    mesh_z = (EAST_ENTRY - C[:, 0]) / LAYER_EAST_DX

    t0 = time.time()
    species_clouds, B, _, _ = load_events(args.n_events, torch.device("cpu"), mountain)
    print(f"[events] {B} held-out events, species={list(SPECIES_NAMES)} ({time.time() - t0:.0f}s)")

    modes = ["raw"] + [f"k{k}" for k in args.ks]
    rng = np.random.default_rng(args.seed)
    layouts = []
    for s_name, fn_name, kw in _STRATEGIES:
        e, n = _STRATEGY_FNS[fn_name](mountain, n_det=N_DETECTORS, rng=rng, **kw)
        layouts.append((s_name, e.float().to(DEVICE), n.float().to(DEVICE)))

    out = {"config": dict(vars(args), run=RUN_LOCATION, sigma=SIGMA_SPATIAL)}
    for sp, clouds_all in zip(SPECIES_NAMES, species_clouds):
        rk = {k: [] for k in args.ks}
        acc = {m: dict(E=[], T=[]) for m in modes}
        for lo in range(0, B, args.chunk):
            cl = clouds_all[lo:lo + args.chunk].to(DEVICE)
            near = near_array_mask(cl, mesh_nu, mesh_z, args.near_radius)
            bw = {"raw": None}
            for k in args.ks:
                r = adaptive_bandwidths(cl, k, 0.0)                    # unfloored r_k
                rk[k].append(r[near].cpu())
                bw[f"k{k}"] = r.clamp(min=SIGMA_SPATIAL)
            for li, (s_name, e_det, n_det) in enumerate(layouts):
                for m in modes:
                    E, T = labels(cl, e_det, n_det, surface, bw[m])
                    acc[m]["E"].append(torch.nan_to_num(E).cpu())
                    acc[m]["T"].append(torch.nan_to_num(T).cpu())
                if lo == 0 and li == 0:
                    E_ref, T_ref = compute_labels_batch(cl, e_det, n_det, surface, interp_k=None)
                    E_raw, T_raw = acc["raw"]["E"][-1], acc["raw"]["T"][-1]
                    assert torch.equal(E_ref.cpu(), E_raw) and torch.equal(T_ref.cpu(), T_raw), \
                        "mapping drifted from compute_labels_batch"
            del cl
        print(f"\n[{sp}]  ({time.time() - t0:.0f}s)")

        res = {"points": {}, "labels": {}}
        for k in args.ks:
            r = torch.cat(rk[k])
            res["points"][f"k{k}"] = dict(
                n_near=int(r.numel()),
                r_k_p10_p50_p90=[float(q) for q in torch.quantile(r[:1_000_000], torch.tensor([.1, .5, .9]))],
                frac_r_k_gt_sigma=float((r > SIGMA_SPATIAL).float().mean()))
            p = res["points"][f"k{k}"]
            print(f"  points near array (k={k:2d}): n={p['n_near']:,}  r_k p10/p50/p90 = "
                  f"{[round(v, 1) for v in p['r_k_p10_p50_p90']]} m   frac r_k > sigma = "
                  f"{100 * p['frac_r_k_gt_sigma']:.1f}%")

        E_raw_sum = float(torch.cat(acc["raw"]["E"]).sum())
        print(f"  {'mode':>5s} {'E==0':>8s} {'T==0':>8s} {'lit log1p(E) mean/std':>24s} "
              f"{'lit log-T mean/std':>20s} {'deposit/raw':>12s}")
        for m in modes:
            E, T = torch.cat(acc[m]["E"]), torch.cat(acc[m]["T"])
            lE, lT = torch.log1p(E[E > 0]), torch.log1p(T[T > 0] * T_LOG_SCALE)
            r = dict(frac_E_zero=float((E == 0).float().mean()),
                     frac_T_zero=float((T == 0).float().mean()),
                     lit_logE=[float(lE.mean()), float(lE.std())] if lE.numel() else None,
                     lit_logT=[float(lT.mean()), float(lT.std())] if lT.numel() else None,
                     deposit_vs_raw=float(E.sum()) / E_raw_sum if E_raw_sum else float("nan"))
            res["labels"][m] = r
            fmt = lambda v: f"{v[0]:.3f}/{v[1]:.3f}" if v else "n/a"  # noqa: E731
            print(f"  {m:>5s} {100 * r['frac_E_zero']:7.2f}% {100 * r['frac_T_zero']:7.2f}% "
                  f"{fmt(r['lit_logE']):>24s} {fmt(r['lit_logT']):>20s} {r['deposit_vs_raw']:12.4f}")
        out[sp] = res

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[json] wrote {args.json}")


if __name__ == "__main__":
    main()
