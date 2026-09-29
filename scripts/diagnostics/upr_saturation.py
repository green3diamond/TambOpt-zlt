"""Is the acceptance term U_PR usable as a counterweight in TambOpt?

SWGOLO's layout structure comes from resolution terms (reward packing) fighting
an acceptance term (reward catching more showers). TambOpt computes U_PR but
leaves it out of the composite. Before adding it back, check two things:

  1. Saturation: the per-event reconstructability r. If r ~ 1 for nearly every
     event, U_PR = sqrt(sum r) is flat in the layout and cannot pull anything.
  2. Scale: |dU_term/d(x,y)| for each term, so a weight can be picked that makes
     U_PR's gradient comparable to the resolution terms' instead of swamping them.

Both are measured through the frozen SURROGATE (what the optimizer reads) and
through the plane-aware KERNEL on held-out events (the ground truth), for a
grid layout, a center layout and an optimized layout.

Usage:
    python scripts/diagnostics/upr_saturation.py [--n-surr 20000] [--n-kernel 256]
"""
import argparse
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import modules  # noqa: E402,F401
from modules.constants import (  # noqa: E402
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX,
    N_PLANES, N_DETECTORS, TRAINING_DATASET_FOLDER, RUN_LOCATION,
)
from modules.geometry import (  # noqa: E402
    load_tr_mountain, SurfaceUpMap, sample_initial_layout_ne, project_to_mountain_ne,
)
from modules.optimize import load_models  # noqa: E402
from modules.optimize.objective import (  # noqa: E402
    _predict_primary, primary_to_physical_labels, W_THETA, W_PHI, W_E, W_PR, W_DIV,
    LAYOUT_THRESHOLD, TAU_LAYOUT, RECONSTRUCT_THRESHOLD, TAU_RECONSTRUCT,
    CAP_THETA, CAP_PHI, CAP_E,
)
from modules.optimize.utility import reconstructability, U_E, U_angle, U_PR  # noqa: E402
from plots.layouts.true_utility import load_events, KernelDualLabels  # noqa: E402

import modules.surrogates.dual as _dual  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run-14 surrogates were trained with the old heteroscedastic head (decoder emits
# mu_E, mu_T, logvar_E, logvar_T); this branch's DeepSets is mean-only (2 outputs).
# Rows 0-1 of the final linear ARE the mean readout, so keeping them reproduces
# the old checkpoint's mean forward exactly. Patched here, not in modules/.
_build = _dual.build_surrogate_from_ckpt


def _build_mean_only(ckpt, *a, **kw):
    # Only when the model can't take the checkpoint as-is: NLL-head code (e.g.
    # branch run12-nll-sigma50) loads 4-output checkpoints unchanged.
    try:
        return _build(ckpt, *a, **kw)
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise
    sd = ckpt["state_dict"]
    dec = sorted({int(k.split(".")[1]) for k in sd if k.startswith("decoder.")})
    if dec:
        w, b = f"decoder.{dec[-1]}.weight", f"decoder.{dec[-1]}.bias"
        if sd[w].shape[0] == 4:
            sd = dict(sd)
            sd[w], sd[b] = sd[w][:2], sd[b][:2]
            ckpt = dict(ckpt, state_dict=sd)
            print("[load] sliced 4-output (mu, logvar) head to its 2 mean rows")
    return _build(ckpt, *a, **kw)


_dual.build_surrogate_from_ckpt = _build_mean_only
OPT_12 = ("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/"
          "12_10pct_mean_analytic/test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_center/"
          "layout_best.pt")


def layouts(mountain):
    out = {}
    for scheme in ("grid", "center"):
        torch.manual_seed(0)
        e, n = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme=scheme)
        e, n = project_to_mountain_ne(mountain, torch.as_tensor(e, dtype=torch.float32),
                                      torch.as_tensor(n, dtype=torch.float32))
        out[scheme] = (e.float(), n.float())
    if os.path.exists(OPT_12):
        L = torch.load(OPT_12, map_location="cpu", weights_only=False)
        out["opt_run12_center"] = (torch.as_tensor(L["x"]).float(), torch.as_tensor(L["y"]).float())
    return out


RT = RECONSTRUCT_THRESHOLD   # overridden per --reconstruct-thresholds entry
SUMMARY = []


def terms(x, y, primary, fnn, recon):
    """Each weighted utility term as its own scalar, plus per-event r and soft count."""
    E_det, E_pred, th_pred, ph_pred = _predict_primary(x, y, primary, fnn, recon,
                                                         sample=False)
    E_true, th_true, ph_true = primary_to_physical_labels(primary)
    counts = torch.expm1(E_det)
    soft = torch.sigmoid(TAU_LAYOUT * (counts - LAYOUT_THRESHOLD)).sum(1)
    r = reconstructability(counts, layout_threshold=LAYOUT_THRESHOLD, tau_layout=TAU_LAYOUT,
                           reconstruct_threshold=RT,
                           tau_reconstruct=TAU_RECONSTRUCT)
    t = dict(
        u_theta=W_THETA * U_angle(th_pred, th_true, r, cap=CAP_THETA) / W_DIV,
        u_phi=W_PHI * U_angle(ph_pred, ph_true, r, cap=CAP_PHI, period=2 * math.pi) / W_DIV,
        u_e=W_E * U_E(E_pred, E_true, r, cap=CAP_E) / W_DIV,
        u_pr=W_PR * U_PR(r) / W_DIV,
    )
    return t, r, soft


def report(tag, name, x0, y0, primary, fnn, recon):
    x = x0.clone().to(DEVICE).requires_grad_(True)
    y = y0.clone().to(DEVICE).requires_grad_(True)
    t, r, soft = terms(x, y, primary, fnn, recon)
    rn = r.detach().cpu().numpy()
    sn = soft.detach().cpu().numpy()
    print(f"\n[{tag}] layout={name}  B={len(rn)}  reconstruct_threshold={RT}")
    print(f"    soft lit-detector count p10/p50/p90 = {np.percentile(sn, [10, 50, 90]).round(1)}")
    print(f"    r p5/p25/p50 = {np.percentile(rn, [5, 25, 50]).round(3)}  "
          f"frac r>0.95 = {(rn > 0.95).mean():.3f}  frac r<0.5 = {(rn < 0.5).mean():.3f}")
    print(f"    mean r = {rn.mean():.4f}   (U_PR's dependence on the layout is only through this)")
    gn = {}
    for k, v in t.items():
        gx, gy = torch.autograd.grad(v, (x, y), retain_graph=True, allow_unused=True)
        gx = torch.zeros_like(x) if gx is None else gx
        gy = torch.zeros_like(y) if gy is None else gy
        gn[k] = torch.cat([gx, gy]).norm().item()
    # U_PR = W sqrt(sum r): its per-event-normalised form is what a batch-size-free weight needs.
    for k, v in t.items():
        print(f"    {k:8s} value={v.item():12.4f}   |grad_xy|={gn[k]:.3e}")
    res = gn["u_theta"] + gn["u_phi"] + gn["u_e"]
    if gn["u_pr"] > 0:
        print(f"    -> weight on u_pr for gradient parity with resolution terms: "
              f"x{res / gn['u_pr']:.3e} (current weight 1)")
    SUMMARY.append((RT, tag, name, rn.mean(), (rn < 0.5).mean(),
                    sum(t[k].item() for k in ("u_theta", "u_phi", "u_e")), res))
    return dict(r=rn, soft=sn, grad=gn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-surr", type=int, default=20000)
    ap.add_argument("--n-kernel", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--n-kernel-r", type=int, default=1024,
                    help="events for the no-grad kernel r distribution")
    ap.add_argument("--skip-surrogate", action="store_true")
    ap.add_argument("--layout", action="append", default=[],
                    help="extra layout name=path.pt or name=file.npz:key (x=East, y=North)")
    ap.add_argument("--reconstruct-thresholds", type=float, nargs="+",
                    default=[RECONSTRUCT_THRESHOLD],
                    help="lit-detector count at which r = 0.5; each value is a separate pass")
    args = ap.parse_args()
    global RT
    print(f"RUN_LOCATION={RUN_LOCATION}  device={DEVICE}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                               east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                               n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    fnn, recon = load_models(DEVICE)
    L = layouts(mountain)
    for spec in args.layout:
        name, path = spec.split("=", 1)
        if ".npz:" in path:
            f, key = path.split(":", 1)
            import numpy as _np
            A = _np.load(f)[key]
            L[name] = (torch.as_tensor(A[:, 0]).float(), torch.as_tensor(A[:, 1]).float())
        else:
            d = torch.load(path, map_location="cpu", weights_only=False)
            L[name] = (torch.as_tensor(d["x"]).float(), torch.as_tensor(d["y"]).float())

    if not args.skip_surrogate:
        for RT in args.reconstruct_thresholds:
            surrogate_pass(args, L, fnn, recon)
    del fnn
    torch.cuda.empty_cache()

    # No-grad kernel r on more events: saturation needs no gradient.
    clouds, B, _, prim_k = load_events(args.n_kernel_r, DEVICE, mountain)
    kern = KernelDualLabels(clouds, surface, DEVICE, chunk=args.chunk, expected_primary=prim_k)
    with torch.no_grad():
        E_true, th_true, ph_true = primary_to_physical_labels(prim_k)
        for name, (x, y) in L.items():
            E_det, E_pred, th_pred, ph_pred = _predict_primary(
                x.to(DEVICE), y.to(DEVICE), prim_k, kern, recon, sample=False)
            counts = torch.expm1(E_det)
            soft = torch.sigmoid(TAU_LAYOUT * (counts - LAYOUT_THRESHOLD)).sum(1).cpu().numpy()
            print(f"\n[kernel no-grad] layout={name}  B={B}")
            print(f"    soft lit-detector count p10/p50/p90 = {np.percentile(soft, [10, 50, 90]).round(1)}")
            for rt in args.reconstruct_thresholds:
                rr = reconstructability(counts, layout_threshold=LAYOUT_THRESHOLD, tau_layout=TAU_LAYOUT,
                                        reconstruct_threshold=rt,
                                        tau_reconstruct=TAU_RECONSTRUCT)
                ures = (W_THETA * U_angle(th_pred, th_true, rr, cap=CAP_THETA)
                        + W_PHI * U_angle(ph_pred, ph_true, rr, cap=CAP_PHI, period=2 * math.pi)
                        + W_E * U_E(E_pred, E_true, rr, cap=CAP_E)).item() / W_DIV
                r = rr.cpu().numpy()
                print(f"    threshold={rt:g}: r p5/p25/p50 = {np.percentile(r, [5, 25, 50]).round(3)}  "
                      f"frac r>0.95 = {(r > 0.95).mean():.3f}  frac r<0.5 = {(r < 0.5).mean():.3f}  "
                      f"mean r = {r.mean():.4f}  U_res = {ures:.3f}")
                SUMMARY.append((rt, "kern1024", name, r.mean(), (r < 0.5).mean(), ures, float("nan")))
    del kern, clouds
    torch.cuda.empty_cache()

    clouds, B, _, prim_k = load_events(args.n_kernel, DEVICE, mountain)
    kern = KernelDualLabels(clouds, surface, DEVICE, chunk=args.chunk,
                            differentiable=True, expected_primary=prim_k)
    for RT in args.reconstruct_thresholds:
        for name, (x, y) in L.items():
            report("kernel", name, x, y, prim_k, kern, recon)
            torch.cuda.empty_cache()

    print("\n[summary] threshold  source     layout             mean_r  frac_r<0.5  U_res   |grad U_res|")
    for rt, tag, name, mr, fd, ures, g in SUMMARY:
        print(f"[summary] {rt:9g}  {tag:9s}  {name:17s}  {mr:.4f}  {fd:.3f}      {ures:7.3f}  {g:.3e}")


def surrogate_pass(args, L, fnn, recon):
    prim_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, prim_all.shape[0], (args.n_surr,), generator=g)
    prim = prim_all[idx].to(DEVICE)
    for name, (x, y) in L.items():
        report("surrogate", name, x, y, prim, fnn, recon)


if __name__ == "__main__":
    main()
