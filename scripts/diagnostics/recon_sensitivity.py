"""Does the reconstruction care how many detectors see a shower, and how accurate
their signals are? Tests the premise behind layout optimization.

Held-out events, kernel (ground-truth) detector response, the run's frozen recon:

  A. natural   : recon error vs the event's lit-detector count (confounded with
                 energy/geometry, but what the optimizer actually faces)
  B. dropout   : same events, keep only n of the lit detectors (the rest set dark);
                 recon error vs n, with nothing else changing
  C. E noise   : multiply lit-detector E by exp(s * N(0,1)); error vs s
  D. T jitter  : add N(0, j) ns to lit-detector arrival times; error vs j

Reference points: the recon fed the Stage-2 surrogate instead of the kernel, and
the surrogate's actual lit-slot residual in log1p(E). The reconstructability gate
r(n) = sigmoid(TAU_RECONSTRUCT * (n - RECONSTRUCT_THRESHOLD)) is printed next to B
so its threshold and slope can be compared with where the recon actually improves.

Usage:
    python scripts/diagnostics/recon_sensitivity.py [--n-events 2048]
"""
import argparse
import json
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import modules  # noqa: E402,F401
from modules.constants import (  # noqa: E402
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX,
    N_PLANES, N_DETECTORS, RUN_LOCATION, LOG_E_MIN, LOG_E_MAX, T_LOG_SCALE,
)
from modules.geometry import (  # noqa: E402
    load_tr_mountain, SurfaceUpMap, sample_initial_layout_ne, project_to_mountain_ne,
)
from modules.optimize import load_models  # noqa: E402
from modules.optimize.objective import (  # noqa: E402
    LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD, TAU_RECONSTRUCT,
)
from plots.layouts.true_utility import load_events, KernelDualLabels  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b"


def errors(recon, xy, pred):
    """Per-event 3D angle error [deg] and |dlog10 E| from recon on (B, n_det, 2) log inputs."""
    feats = torch.cat([xy, pred], dim=-1)
    with torch.no_grad():
        out = recon(feats)
    return out


def score(out, prim):
    d_p = out[:, :3] / out[:, :3].norm(dim=1, keepdim=True).clamp(min=1e-12)
    d_t = prim[:, :3] / prim[:, :3].norm(dim=1, keepdim=True).clamp(min=1e-12)
    ang = torch.rad2deg(torch.arccos((d_p * d_t).sum(1).clamp(-1, 1)))
    de = (out[:, 3] - prim[:, 3]).abs() * (LOG_E_MAX - LOG_E_MIN)
    return ang.cpu().numpy(), de.cpu().numpy()


def summary(ang, de):
    return dict(n=int(ang.size), ang_med=float(np.median(ang)), ang_p68=float(np.percentile(ang, 68)),
                de_med=float(np.median(de)), de_p68=float(np.percentile(de, 68)))


def layout(mountain, path=None):
    if path:
        d = torch.load(path, map_location="cpu", weights_only=False)
        e, n = torch.as_tensor(d["x"]).float(), torch.as_tensor(d["y"]).float()
    else:
        torch.manual_seed(0)
        e, n = sample_initial_layout_ne(mountain, n_units=N_DETECTORS, scheme="grid")
        e, n = project_to_mountain_ne(mountain, torch.as_tensor(e).float(), torch.as_tensor(n).float())
    return e.float().to(DEVICE), n.float().to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--min-lit-dropout", type=int, default=30)
    ap.add_argument("--opt-layout", default=os.path.join(
        RUN_LOCATION, "test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_grid", "layout_best.pt"))
    ap.add_argument("--out", default=os.path.join(RUN_LOCATION, "diagnostics"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    print(f"RUN_LOCATION={RUN_LOCATION}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    fnn, recon = load_models(DEVICE)
    recon.eval()
    clouds, B, _, prim = load_events(args.n_events, DEVICE, mountain)
    kern = KernelDualLabels(clouds, surface, DEVICE, chunk=args.chunk, expected_primary=prim)

    layouts = {"grid": layout(mountain)}
    if os.path.exists(args.opt_layout):
        layouts["optimized"] = layout(mountain, args.opt_layout)
    res = {"gate": {}, "natural": {}, "dropout": {}, "E_noise": {}, "T_jitter": {}, "reference": {}}

    bins = [(0, 0), (1, 2), (3, 4), (5, 7), (8, 11), (12, 19), (20, 29), (30, 49), (50, 100)]
    for name, (e, n) in layouts.items():
        xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
        with torch.no_grad():
            kpred = kern(prim, xy)                                          # (B, n_det, 2) log1p
            spred = fnn(prim, xy)
        counts = torch.expm1(kpred[..., 0])
        lit = counts > LAYOUT_THRESHOLD
        n_lit = lit.sum(1).cpu().numpy()

        ang, de = score(errors(recon, xy, kpred), prim)
        ang_s, de_s = score(errors(recon, xy, spred), prim)
        hit = kpred[..., 0] > 0
        resid = (spred[..., 0] - kpred[..., 0])[hit]
        res["reference"][name] = dict(kernel=summary(ang, de), surrogate=summary(ang_s, de_s),
                                      surrogate_logE_resid_std=float(resid.std()),
                                      surrogate_logE_resid_mean=float(resid.mean()))
        print(f"\n[{name}] kernel input: {summary(ang, de)}")
        print(f"[{name}] surrogate input: {summary(ang_s, de_s)}")
        print(f"[{name}] surrogate log1p(E) residual on hit slots: mean {resid.mean():+.3f}  std {resid.std():.3f}")

        # A. natural
        res["natural"][name] = {}
        for lo, hi in bins:
            m = (n_lit >= lo) & (n_lit <= hi)
            if m.sum() >= 10:
                res["natural"][name][f"{lo}-{hi}"] = summary(ang[m], de[m])
        print(f"[{name}] natural (lit count bin -> median angle deg, median |dlog10E|, n):")
        for k, v in res["natural"][name].items():
            print(f"    {k:>6}: {v['ang_med']:7.2f}  {v['de_med']:.3f}  {v['n']}")

        # B. dropout on events with many lit detectors
        sel = torch.as_tensor(n_lit >= args.min_lit_dropout, device=DEVICE)
        if int(sel.sum()) >= 20:
            kp, pr, lt = kpred[sel], prim[sel], lit[sel]
            xs = xy[: kp.shape[0]]
            res["dropout"][name] = {}
            for keep in (1, 2, 3, 5, 8, 12, 20, 30):
                score_rand = torch.rand(lt.shape, device=DEVICE).masked_fill(~lt, 2.0)
                rank = score_rand.argsort(1).argsort(1)                       # random order among lit
                drop = lt & (rank >= keep)
                p2 = kp.masked_fill(drop.unsqueeze(-1), 0.0)
                a2, d2 = score(errors(recon, xs, p2), pr)
                res["dropout"][name][keep] = summary(a2, d2)
            a2, d2 = score(errors(recon, xs, kp), pr)
            res["dropout"][name]["all"] = summary(a2, d2)
            print(f"[{name}] dropout on {int(sel.sum())} events with >= {args.min_lit_dropout} lit "
                  f"(kept lit -> median angle deg, median |dlog10E|):")
            for k, v in res["dropout"][name].items():
                print(f"    {str(k):>4}: {v['ang_med']:7.2f}  {v['de_med']:.3f}")

        # C. E noise, D. T jitter (all events, lit slots only)
        res["E_noise"][name], res["T_jitter"][name] = {}, {}
        for s in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
            x = kpred[..., 0]
            noisy = torch.log1p(torch.expm1(x) * torch.exp(s * torch.randn_like(x)))
            p2 = kpred.clone()
            p2[..., 0] = torch.where(hit, noisy, x)
            res["E_noise"][name][s] = summary(*score(errors(recon, xy, p2), prim))
        for j in (0.0, 5.0, 20.0, 50.0, 100.0, 300.0):
            t = kpred[..., 1]
            t_us = torch.expm1(t) + j * 1e-3 * torch.randn_like(t)             # log1p(T[s] * 1e6): microseconds
            p2 = kpred.clone()
            p2[..., 1] = torch.where(t > 0, torch.log1p(t_us.clamp(min=1e-6)), t)
            res["T_jitter"][name][j] = summary(*score(errors(recon, xy, p2), prim))
        print(f"[{name}] E noise s -> median angle deg, |dlog10E|: " +
              "  ".join(f"{s}:{v['ang_med']:.2f}/{v['de_med']:.3f}" for s, v in res["E_noise"][name].items()))
        print(f"[{name}] T jitter ns -> median angle deg, |dlog10E|: " +
              "  ".join(f"{j:g}:{v['ang_med']:.2f}/{v['de_med']:.3f}" for j, v in res["T_jitter"][name].items()))

    ns = np.arange(0, 51)
    res["gate"] = {int(k): float(1 / (1 + math.exp(-TAU_RECONSTRUCT * (k - RECONSTRUCT_THRESHOLD)))) for k in ns}
    print(f"\n[gate] r(n) with threshold {RECONSTRUCT_THRESHOLD}, tau {TAU_RECONSTRUCT}: " +
          "  ".join(f"{k}:{res['gate'][k]:.2f}" for k in (0, 1, 2, 3, 5, 8, 12, 20, 30, 50)))

    with open(os.path.join(args.out, "recon_sensitivity.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    plot(res, os.path.join(args.out, "recon_sensitivity.png"))


def plot(res, path):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    cols = {"grid": BLUE, "optimized": ORANGE}
    for name, c in cols.items():
        if name not in res["dropout"]:
            continue
        d = res["dropout"][name]
        keeps = [k for k in d if k != "all"]
        ax[0, 0].plot(keeps, [d[k]["ang_med"] for k in keeps], "o-", color=c, lw=2, label=f"{name} layout")
        ax[0, 1].plot(keeps, [d[k]["de_med"] for k in keeps], "o-", color=c, lw=2, label=f"{name} layout")
        ax[0, 0].axhline(d["all"]["ang_med"], color=c, ls=":", lw=1.5)
        ax[0, 1].axhline(d["all"]["de_med"], color=c, ls=":", lw=1.5)
        en = res["E_noise"][name]
        ax[1, 0].plot(list(en), [v["ang_med"] for v in en.values()], "o-", color=c, lw=2, label=f"{name} layout")
        ax[1, 1].plot(list(en), [v["de_med"] for v in en.values()], "o-", color=c, lw=2, label=f"{name} layout")
        tj = res["T_jitter"][name]
        ax[1, 2].plot(list(tj), [v["ang_med"] for v in tj.values()], "o-", color=c, lw=2, label=f"{name} layout")
        ref = res["reference"][name]
        for a, key in ((ax[1, 0], "ang_med"), (ax[1, 1], "de_med")):
            a.axhline(ref["surrogate"][key], color=c, ls="--", lw=1.2)
            a.axvline(ref["surrogate_logE_resid_std"], color=c, ls=":", lw=1.2)
    g = res["gate"]
    ax[0, 2].plot(list(g), list(g.values()), color=GREY, lw=2)
    titles = [["Angle error vs lit detectors kept", "Energy error vs lit detectors kept",
               f"Reconstructability gate r(n)"],
              ["Angle error vs E noise", "Energy error vs E noise", "Angle error vs T jitter"]]
    xl = [["lit detectors kept (dropout)", "lit detectors kept (dropout)", "lit detectors n"],
          ["E noise s (log-normal sigma)", "E noise s (log-normal sigma)", "T jitter [ns]"]]
    yl = [["median angle error [deg]", "median |Δlog10 E|", "r"],
          ["median angle error [deg]", "median |Δlog10 E|", "median angle error [deg]"]]
    for i in range(2):
        for j in range(3):
            a = ax[i, j]
            a.set_title(titles[i][j], fontsize=12, color=INK)
            a.set_xlabel(xl[i][j], color=GREY)
            a.set_ylabel(yl[i][j], color=GREY)
            a.grid(alpha=0.25, lw=0.5)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    for a in (ax[0, 0], ax[0, 1]):
        a.set_xscale("log")
    ax[0, 0].legend(fontsize=9, frameon=False)
    ax[1, 0].legend(fontsize=9, frameon=False)
    fig.text(0.5, 0.005, "dotted horizontal: all lit detectors kept · dashed horizontal: recon fed the "
             "surrogate · dotted vertical: surrogate's actual log1p(E) residual std",
             ha="center", fontsize=10, color=GREY)
    fig.suptitle("Recon sensitivity to detector multiplicity and signal accuracy (kernel input, held-out events)",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=130)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
