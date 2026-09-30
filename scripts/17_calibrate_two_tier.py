"""Calibrate the two-tier utility for the run constants.py points at.

Through the surrogate the recon looks more accurate than on the kernel, so kernel-space
targets ("good" = angle < THETA and |dlog10 E| < E_GOOD) are translated by quantile
matching on the grid layout and held-out events: the surrogate threshold passes the same
share of showers as the kernel one does. Step widths and partial-credit references scale
with the same factors (1 deg / 0.05 and 20 deg / 0.5 at the 5 deg / 0.2 defaults).

Writes <RUN_LOCATION>/two_tier_calibration_<THETA>deg.json, read by modules/optimize/
objective.py when TAMBOPT_UTILITY=two_tier and TAMBOPT_TWO_TIER_CALIB points at it.

    python scripts/17_calibrate_two_tier.py [--theta-good 5 --e-good 0.2 --n-events 1536]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import modules.constants as C  # noqa: E402
import modules.optimize.objective as OBJ  # noqa: E402
from modules.geometry import (  # noqa: E402
    load_tr_mountain, SurfaceUpMap, sample_initial_layout_ne, project_to_mountain_ne,
)
from modules.optimize import load_models  # noqa: E402
from modules.optimize.utility import angle_between_deg  # noqa: E402
from plots.layouts.true_utility import load_events, KernelDualLabels  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def errors(x, y, prim, fwd, recon):
    with torch.no_grad():
        _, E_pred, th_p, ph_p = OBJ._predict_primary(x, y, prim, fwd, recon, sample=False)
        E_true, th_t, ph_t = OBJ.primary_to_physical_labels(prim)
    a = angle_between_deg(OBJ._sph(th_p, ph_p), OBJ._sph(th_t, ph_t))
    return a, (torch.log10(E_pred) - torch.log10(E_true)).abs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-good", type=float, default=5.0)
    ap.add_argument("--e-good", type=float, default=0.2)
    ap.add_argument("--n-events", type=int, default=1536)
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()
    mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                                layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    fnn, recon = load_models(DEVICE)
    clouds, _, _, prim = load_events(args.n_events, DEVICE, mountain)
    kern = KernelDualLabels(clouds, surface, DEVICE, chunk=args.chunk, expected_primary=prim)
    torch.manual_seed(0)
    e, n = sample_initial_layout_ne(mountain, n_units=C.N_DETECTORS, scheme="grid")
    e, n = project_to_mountain_ne(mountain, torch.as_tensor(e).float(), torch.as_tensor(n).float())
    e, n = e.float().to(DEVICE), n.float().to(DEVICE)
    a_k, e_k = errors(e, n, prim, kern, recon)
    a_s, e_s = errors(e, n, prim, fnn, recon)
    tg, eg = args.theta_good, args.e_good
    fa, fe = float((a_k < tg).float().mean()), float((e_k < eg).float().mean())
    ka = float(torch.quantile(a_s, fa)) / tg
    ke = float(torch.quantile(e_s, fe)) / eg
    out = dict(theta_good=tg * ka, e_good=eg * ke, s_theta=0.2 * tg * ka, s_e=0.25 * eg * ke,
               theta_ref=20.0 * ka, e_ref=0.5 * ke, kernel_theta_good=tg, kernel_e_good=eg,
               kernel_frac_angle=fa, kernel_frac_energy=fe,
               kernel_frac_good=float(((a_k < tg) & (e_k < eg)).float().mean()),
               kernel_angle_median=float(a_k.median()), surrogate_angle_median=float(a_s.median()),
               kernel_dE_median=float(e_k.median()), surrogate_dE_median=float(e_s.median()),
               n_events=int(prim.shape[0]))
    path = os.path.join(C.RUN_LOCATION, f"two_tier_calibration_{tg:g}deg.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
