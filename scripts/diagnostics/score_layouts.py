"""Score saved layouts on held-out kernel showers with detector_count_scan's yardstick, at a
chosen 'well reconstructed' threshold. For re-scoring runs that were judged at another one.

    python scripts/diagnostics/score_layouts.py --layout name=path.pt [...] --theta-good 2.5 --out scores.json
"""
import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location("dcs", os.path.join(_HERE, "detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DCS)
GS = DCS.GS

import modules.constants as C  # noqa: E402
import modules.optimize.objective as OBJ  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.optimize import load_models  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", action="append", required=True, help="name=path.pt")
    ap.add_argument("--theta-good", type=float, default=5.0)
    ap.add_argument("--e-good", type=float, default=0.2)
    ap.add_argument("--n-kernel", type=int, default=1536)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--recon-dir", default=None,
                    help="recon checkpoint folder (default: RECON_FOLDER + '_deepsets')")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY,
                                east_entry=C.EAST_ENTRY, layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DCS.DEVICE)
    _, recon = load_models(DCS.DEVICE, recon_dir=args.recon_dir)
    gate = GS.settings(float(OBJ.RECONSTRUCT_THRESHOLD))[0]
    clouds, _, _, prim_k = load_events(args.n_kernel, DCS.DEVICE, mountain)
    kern = GS.CachedKernel(clouds, surface, DCS.DEVICE, chunk=args.chunk, expected_primary=prim_k)
    out = {}
    for spec in args.layout:
        name, path = spec.split("=", 1)
        L = torch.load(path, map_location="cpu")
        e, n = torch.as_tensor(L["x"]).float().to(DCS.DEVICE), torch.as_tensor(L["y"]).float().to(DCS.DEVICE)
        out[name] = DCS.evaluate(e, n, len(e), prim_k, kern, recon, gate, args.theta_good, args.e_good)
        print(f"[{name}] good@{args.theta_good:g}deg {out[name]['frac_good_t']:.3f}  good@5deg "
              f"{out[name]['frac_good_5']:.3f}  angle {out[name]['ang_med']:.2f}  U {out[name]['U_kernel']:.2f}",
              flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
