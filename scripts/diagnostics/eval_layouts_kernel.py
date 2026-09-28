"""Score saved layouts against the plane-aware KERNEL on held-out events.

For each layout: the utility terms (theta, phi, E), mean reconstructability r,
the lit-detector distribution, and the frozen-recon errors, all computed from
the kernel's response (ground truth), not the surrogate's. Also the elevation
and shower-depth (z_cont) spread of the detectors, to tell what a structured
layout is lining up along.

Usage:
    python scripts/diagnostics/eval_layouts_kernel.py --n-events 2048 \
        --layout init=path.npz:init_upr0 --layout lam0=path/layout_best.pt ...
"""
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts", "diagnostics"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import upr_saturation as us  # noqa: E402  (also patches the mean-only head slicing)
from modules.constants import (  # noqa: E402
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
)
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.surrogates import build_recon_from_ckpt  # noqa: E402
from modules.constants import RECON_FOLDER, N_DETECTORS  # noqa: E402
from plots.layouts.true_utility import load_events, KernelDualLabels  # noqa: E402

DEVICE = us.DEVICE


def read_layout(spec):
    name, path = spec.split("=", 1)
    if ":" in path and path.split(":", 1)[0].endswith(".npz"):
        path, key = path.split(":", 1)
        A = np.load(path)[key]
        return name, torch.as_tensor(A[:, 0]).float(), torch.as_tensor(A[:, 1]).float()
    L = torch.load(path, map_location="cpu", weights_only=False)
    return name, torch.as_tensor(L["x"]).float(), torch.as_tensor(L["y"]).float()


def override_run(run_location, sigma, reconstruct_threshold):
    """Score against a different run than modules/constants.py points at.

    Only the kernel and the recon are involved (no surrogate), so three things
    need repointing: the held-out event files, the kernel's sigma, and the r
    gate's threshold. Each is patched in the namespace that actually reads it.
    """
    import functools
    import modules.constants as C
    import plots.layouts.true_utility as tu
    recon_dir = RECON_FOLDER + "_deepsets"
    if run_location:
        swap = lambda p: p.replace(C.RUN_LOCATION, run_location)  # noqa: E731
        tu.HELDOUT_SHOWER_CACHE_PATH = swap(C.HELDOUT_SHOWER_CACHE_PATH)
        tu.HELDOUT_POSITIONS_PATH = swap(C.HELDOUT_POSITIONS_PATH)
        recon_dir = swap(recon_dir)
        print(f"[override] run={run_location}\n           heldout={tu.HELDOUT_SHOWER_CACHE_PATH}")
    if sigma is not None:
        tu.compute_labels_batch = functools.partial(tu.compute_labels_batch, sigma_spatial=sigma)
        print(f"[override] kernel sigma={sigma} m")
    if reconstruct_threshold is not None:
        us.RECONSTRUCT_THRESHOLD = reconstruct_threshold
        print(f"[override] RECONSTRUCT_THRESHOLD={reconstruct_threshold}")
    return recon_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--layout", action="append", required=True)
    ap.add_argument("--run-location", default=None,
                    help="score against this run's held-out events and recon")
    ap.add_argument("--sigma", type=float, default=None, help="kernel sigma [m]")
    ap.add_argument("--reconstruct-threshold", type=float, default=None)
    args = ap.parse_args()
    recon_dir = override_run(args.run_location, args.sigma, args.reconstruct_threshold)

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                               east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                               n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    ck = torch.load(os.path.join(recon_dir, "recon.pt"),
                    map_location=DEVICE, weights_only=False)
    print(f"[load] recon <- {recon_dir}  config={ck.get('config')}")
    recon = build_recon_from_ckpt(ck, N_DETECTORS, DEVICE)

    clouds, B, _, prim = load_events(args.n_events, DEVICE, mountain)
    kern = KernelDualLabels(clouds, surface, DEVICE, chunk=args.chunk, expected_primary=prim)
    print(f"[events] {B} held-out events")

    for spec in args.layout:
        name, x, y = read_layout(spec)
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            t, r, soft = us.terms(x, y, prim, kern, recon)
            up = surface(y, x)
        rn, sn = r.cpu().numpy(), soft.cpu().numpy()
        z = (EAST_ENTRY - x.cpu().numpy()) / LAYER_EAST_DX
        U = t["u_theta"] + t["u_phi"] + t["u_e"]
        print(f"\n[{name}]  U_res(theta+phi+E)={U.item():.3f}  "
              + "  ".join(f"{k}={v.item():.3f}" for k, v in t.items() if k != "u_pr"))
        print(f"    mean r={rn.mean():.4f}  frac r<0.5={(rn < 0.5).mean():.3f}  "
              f"lit p10/p50/p90={np.percentile(sn, [10, 50, 90]).round(1)}")
        print(f"    detector Up p10/p50/p90={np.percentile(up.cpu().numpy(), [10, 50, 90]).round(0)}  "
              f"std={up.std().item():.0f} m   z_cont p10/p50/p90="
              f"{np.percentile(z, [10, 50, 90]).round(2)} std={z.std():.2f}")


if __name__ == "__main__":
    main()
