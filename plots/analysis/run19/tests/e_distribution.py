"""Run 19: distribution of the kernel's E (expected particles per detector, species combined) over detector x shower
pairs, for the grid, the 6 km disk and r = 25 m triplets (100 / 99 detectors, 5,000 held-out showers). Also what a
Poisson reading of E would give: P(>= 1 particle) = 1 - exp(-E), summed per shower = expected lit detectors.
Writes tests/e_distribution.png and tests/e_distribution.json. GPU (kernel).

    python plots/analysis/run19/tests/e_distribution.py
"""
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

_s = importlib.util.spec_from_file_location("dcs", os.path.join(REPO, "scripts/diagnostics/detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_s); _s.loader.exec_module(DCS)
GS = DCS.GS
D = os.path.join(C.RUN_LOCATION, "diagnostics")
T = os.path.join(D, "tests")


def paths(f):
    return dict(l.strip().split("=", 1) for l in open(f) if l.strip())


P0, PT = paths(os.path.join(T, "layouts.txt")), paths(os.path.join(D, "triplets", "layouts.txt"))
LAYS = [("grid (100)", P0["grid_n100"], "#2a78d6"), ("6 km disk (100)", P0["disk_r6000_n100"], "#1baf7a"),
        ("triplets r = 25 m (99)", PT["tri33_r25_n99"], "#c0392b")]
dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
clouds, B, _, prim = load_events(5000, dev, m)
kern = GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)

EDGES = [0, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1, 3, 10, 100, 1e3, np.inf]
res, Es = {}, {}
for lab, path, _ in LAYS:
    L = torch.load(path, map_location="cpu")
    xy = torch.stack([torch.as_tensor(L["x"]).float(), torch.as_tensor(L["y"]).float()], -1).to(dev)[None].expand(B, -1, -1)
    with torch.no_grad():
        E = torch.expm1(kern(prim, xy)[..., 0]).clamp(min=0).cpu().numpy()            # (B, nd) expected particles
    Es[lab] = E
    h = np.histogram(E.ravel(), bins=EDGES)[0] / E.size
    res[lab] = dict(
        bins={f"[{a:g}, {b:g})": float(v) for a, b, v in zip(EDGES[:-1], EDGES[1:], h)},
        median_all=float(np.median(E)), median_nonzero=float(np.median(E[E > 0])),
        per_shower_median=dict(lit_gt1e3=float(np.median((E > 1e-3).sum(1))), lit_gt1=float(np.median((E > 1).sum(1))),
                               poisson_expected_lit=float(np.median((1 - np.exp(-E)).sum(1))),
                               max_E=float(np.median(E.max(1))), sum_E=float(np.median(E.sum(1)))),
        share_showers_poisson_lt3=float(np.mean((1 - np.exp(-E)).sum(1) < 3)),
        share_showers_gt1e3_lt3=float(np.mean((E > 1e-3).sum(1) < 3)))
    r = res[lab]["per_shower_median"]
    print(f"{lab:24s} median E (all) {res[lab]['median_all']:.3g}, (E>0) {res[lab]['median_nonzero']:.3g} | per shower: "
          f"lit E>0.001 {r['lit_gt1e3']:.0f}, E>1 {r['lit_gt1']:.0f}, Poisson expected {r['poisson_expected_lit']:.1f}, "
          f"max E {r['max_E']:.3g}, sum E {r['sum_E']:.3g}", flush=True)
    for k, v in res[lab]["bins"].items():
        print(f"    {k:>16s} {100 * v:6.2f}%")
json.dump(dict(n_events=B, layouts={l: p for l, p, _ in LAYS}, results=res), open(os.path.join(T, "e_distribution.json"), "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(21, 6))
lb = np.linspace(-6, 4, 61)
for lab, _, c in LAYS:
    E = Es[lab]
    ax[0].hist(np.log10(E[E > 0]), bins=lb, histtype="step", lw=2, color=c, weights=np.full((E > 0).sum(), 100 / E.size),
               label=f"{lab}: E = 0 in {100 * np.mean(E == 0):.0f}%")
    ax[1].hist(np.log10(np.clip(E.max(1), 1e-6, None)), bins=lb, histtype="step", lw=2, color=c, label=lab)
    pe = (1 - np.exp(-E)).sum(1)
    ax[2].hist([(E > 1e-3).sum(1), pe, (E > 1).sum(1)][1], bins=np.arange(0, 41), histtype="step", lw=2, color=c, label=f"{lab}: Poisson expected")
    ax[2].hist((E > 1e-3).sum(1), bins=np.arange(0, 41), histtype="step", lw=1.2, ls="--", color=c, label=f"{lab}: E > 0.001")
for x in (-3, 0):
    ax[0].axvline(x, color="#888", ls=":", lw=1)
ax[0].set(xlabel="log10 E (expected particles, detector × shower, E > 0)", ylabel="% of all detector × shower pairs",
          title="Kernel E per detector")
ax[1].set(xlabel="log10 max E over the layout's detectors", ylabel="showers", title="Brightest detector per shower")
ax[2].set(xlabel="detectors per shower", ylabel="showers", title="Lit per shower: Σ (1 − e^−E) (solid) vs E > 0.001 (dashed)")
for a in ax:
    a.grid(alpha=0.25, lw=0.5); a.legend(fontsize=9, frameon=False)
fig.suptitle(f"{os.path.basename(C.RUN_LOCATION)} · kernel E (species combined) · {B} held-out showers · dotted: E = 0.001 and E = 1")
fig.tight_layout(); fig.savefig(os.path.join(T, "e_distribution.png"), dpi=120, facecolor="white")
print("[write]", os.path.join(T, "e_distribution.png"), os.path.join(T, "e_distribution.json"))
