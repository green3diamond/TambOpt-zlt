"""How does reconstruction quality depend on the number of ACTIVATED detectors, and
how does the activation (trigger) threshold control that?

Held-out events, kernel (ground-truth) response on the grid layout, the run's
frozen recon. For each trigger threshold t, a detector is activated when its
count >= t; detectors below t are set dark (E = T = 0), as a real trigger would.

  natural : per-event recon error binned by that event's number of activated
            detectors (how the events come; bigger showers light more)
  matched : events with >= 15 activated detectors; keep only 3 / 5 / 10 / 15 of
            them (random), so the SAME events are compared at 3 vs 15

    python scripts/diagnostics/activation_threshold.py [--n-events 4096]
(Legacy, runs 12-18; kept for the figures in 02_findings_after_run13.pdf.)
"""
import argparse
import json
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
import modules.constants as C  # noqa: E402
from modules.geometry import (  # noqa: E402
    load_tr_mountain, SurfaceUpMap, sample_initial_layout_ne, project_to_mountain_ne,
)
from modules.optimize import load_models  # noqa: E402
from plots.layouts.true_utility import load_events, KernelDualLabels  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLDS = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0)       # 0 = any nonzero signal; 1 = current trigger
BINS = [(1, 2), (3, 4), (5, 9), (10, 14), (15, 29), (30, 59), (60, 100)]
KEEP = (3, 5, 10, 15)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")


def errors(recon, xy, pred, prim):
    with torch.no_grad():
        out = recon(torch.cat([xy, pred], dim=-1))
    d_p = out[:, :3] / out[:, :3].norm(dim=1, keepdim=True).clamp(min=1e-12)
    d_t = prim[:, :3] / prim[:, :3].norm(dim=1, keepdim=True).clamp(min=1e-12)
    ang = torch.rad2deg(torch.arccos((d_p * d_t).sum(1).clamp(-1, 1)))
    de = (out[:, 3] - prim[:, 3]).abs() * (C.LOG_E_MAX - C.LOG_E_MIN)
    return ang.cpu().numpy(), de.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--recon-dir", default=None)
    ap.add_argument("--out", default=os.path.join(C.RUN_LOCATION, "diagnostics"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    print(f"RUN_LOCATION={C.RUN_LOCATION}  sigma={C.SIGMA_SPATIAL}", flush=True)

    mountain = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY,
                                east_entry=C.EAST_ENTRY, layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(DEVICE)
    _, recon = load_models(DEVICE, recon_dir=args.recon_dir)
    recon.eval()
    clouds, B, _, prim = load_events(args.n_events, DEVICE, mountain)
    kern = KernelDualLabels(clouds, surface, DEVICE, chunk=args.chunk, expected_primary=prim)
    e, n = sample_initial_layout_ne(mountain, n_units=C.N_DETECTORS, scheme="grid")
    e, n = project_to_mountain_ne(mountain, torch.as_tensor(e).float(), torch.as_tensor(n).float())
    xy = torch.stack([e.float(), n.float()], -1).to(DEVICE)[None].expand(B, -1, -1)
    with torch.no_grad():
        kpred = kern(prim, xy)                                        # (B, n_det, 2) log1p
    counts = torch.expm1(kpred[..., 0])

    res = {}
    for t in THRESHOLDS:
        act = counts > 0 if t == 0 else counts >= t
        pred = kpred.masked_fill(~act.unsqueeze(-1), 0.0)
        n_act = act.sum(1).cpu().numpy()
        ang, de = errors(recon, xy, pred, prim)
        r = dict(n_act_median=float(np.median(n_act)), frac_lt3=float((n_act < 3).mean()),
                 frac_ge15=float((n_act >= 15).mean()),
                 all=dict(ang=float(np.median(ang)), dE=float(np.median(de))), natural={}, matched={})
        for lo, hi in BINS:
            m = (n_act >= lo) & (n_act <= hi)
            if m.sum() >= 20:
                r["natural"][f"{lo}-{hi}"] = dict(n=int(m.sum()), ang=float(np.median(ang[m])),
                                                  dE=float(np.median(de[m])))
        sel = torch.as_tensor(n_act >= 15, device=DEVICE)
        if int(sel.sum()) >= 50:
            a_s, p_s, x_s, pr_s = act[sel], pred[sel], xy[: int(sel.sum())], prim[sel]
            rank = torch.rand(a_s.shape, device=DEVICE).masked_fill(~a_s, 2.0).argsort(1).argsort(1)
            for k in KEEP:
                drop = a_s & (rank >= k)
                ak, dk = errors(recon, x_s, p_s.masked_fill(drop.unsqueeze(-1), 0.0), pr_s)
                r["matched"][k] = dict(n=int(sel.sum()), ang=float(np.median(ak)), dE=float(np.median(dk)))
        res[t] = r
        tag = "any signal" if t == 0 else f"count >= {t:g}"
        print(f"\n[{tag}] activated per event: median {r['n_act_median']:.0f}, "
              f"<3: {r['frac_lt3']:.1%}, >=15: {r['frac_ge15']:.1%}  |  all events: angle {r['all']['ang']:.2f} deg, "
              f"|dlog10E| {r['all']['dE']:.3f}")
        print("    natural  (activated -> median angle deg, |dlog10E|, events): " +
              "  ".join(f"{k}: {v['ang']:.1f}/{v['dE']:.2f}/{v['n']}" for k, v in r["natural"].items()))
        if r["matched"]:
            print(f"    matched  ({r['matched'][KEEP[0]]['n']} events with >=15 activated, keep k -> angle, |dlog10E|): " +
                  "  ".join(f"{k}: {v['ang']:.1f}/{v['dE']:.3f}" for k, v in r["matched"].items()))

    with open(os.path.join(args.out, "activation_threshold.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(res, os.path.join(args.out, "activation_threshold.png"))


def plot(res, path):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    labels = [f"{lo}-{hi}" for lo, hi in BINS]
    for c, (t, r) in zip(COLORS, res.items()):
        name = "any signal" if t == 0 else f"count ≥ {t:g}"
        xs = [i for i, k in enumerate(labels) if k in r["natural"]]
        ax[0].plot(xs, [r["natural"][labels[i]]["ang"] for i in xs], "o-", color=c, lw=2, label=name)
        ax[1].plot(xs, [r["natural"][labels[i]]["dE"] for i in xs], "o-", color=c, lw=2, label=name)
        if r["matched"]:
            ks = list(r["matched"])
            ax[2].plot(ks, [r["matched"][k]["ang"] for k in ks], "o-", color=c, lw=2, label=name)
    for a in ax[:2]:
        a.set_xticks(range(len(labels)))
        a.set_xticklabels(labels)
        a.set_xlabel("activated detectors in the event")
    ax[0].set_ylabel("median angle error [deg]")
    ax[1].set_ylabel("median |Δlog10 E|")
    ax[2].set_xlabel("activated detectors kept (same events, ≥15 activated)")
    ax[2].set_ylabel("median angle error [deg]")
    ax[0].set_title("Angle error vs activated detectors")
    ax[1].set_title("Energy error vs activated detectors")
    ax[2].set_title("Same events: keep 3 / 5 / 10 / 15")
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(title="trigger threshold", fontsize=9, frameon=False)
    fig.suptitle(f"Recon vs activated detectors and trigger threshold — {os.path.basename(C.RUN_LOCATION)}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
