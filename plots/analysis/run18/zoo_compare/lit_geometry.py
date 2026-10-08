"""Where are the lit detectors when the recon gets a shower right? (run 18, relative-time recon by default)

Same 112 distinct test layouts x 1,536 held-out showers as lit_vs_angle.py. Per pair, from the detectors with
>= 1 count (kernel): n_lit, mean pairwise distance, median nearest-neighbour spacing, and the baseline (largest
distance), horizontal East-North in metres; plus the recon's angle error. Writes lit_geometry_pairs.npz,
lit_geometry.png (median error and share within 1 deg vs each distance, per n_lit group, bands from 200
shower resamples) and lit_geometry.json (a within-1 deg vs > 5 deg comparison table).

    python lit_geometry.py [--recon-dir <dir>]
"""
import argparse
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = "/n/home05/zdimitrov/tambo/TambOpt"
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.optimize import load_models  # noqa: E402
from modules.optimize.objective import primary_to_physical_labels  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

_s = importlib.util.spec_from_file_location("dcs", os.path.join(REPO, "scripts/diagnostics/detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_s); _s.loader.exec_module(DCS)
GS = DCS.GS
Z = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(Z)
R18 = os.path.dirname(D)

ap = argparse.ArgumentParser()
ap.add_argument("--recon-dir", default=os.path.join(R18, "test_v6_run_03_recentered_deepsets_zoo_reltime"))
ap.add_argument("--n-events", type=int, default=1536)
args = ap.parse_args()
LISTS = {"core_test_ndet": f"{D}/core_test_ndet/layouts.txt", "clusters": f"{D}/clusters/layouts.txt",
         "angleonly": f"{Z}/layouts_angleonly.txt", "disks_n10": f"{Z}/layouts_disks_n10.txt"}

dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, recon_dir=args.recon_dir, with_fnn=False)
clouds, B, _, prim = load_events(args.n_events, dev, m)
kern = GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)
E_true, th_t, ph_t = primary_to_physical_labels(prim)


def lit_geometry(xy, lit):
    """xy (nd, 2), lit (B, nd) bool -> mean pairwise, median NN, max pairwise distance per row (nan if < 2 lit)."""
    d = torch.cdist(xy, xy)                                                    # (nd, nd)
    pair = lit[:, :, None] & lit[:, None, :] & ~torch.eye(xy.shape[0], dtype=torch.bool, device=xy.device)
    npair = pair.sum((1, 2)).float()
    mean = torch.where(npair > 0, (d * pair).sum((1, 2)) / npair.clamp(min=1), torch.full_like(npair, float("nan")))
    mx = torch.where(npair > 0, (d * pair).amax((1, 2)), torch.full_like(npair, float("nan")))
    dn = torch.where(pair, d, torch.full_like(d.expand_as(pair), float("inf"))).amin(2)   # NN of each lit detector
    dn = torch.where(lit & torch.isfinite(dn), dn, torch.full_like(dn, float("nan")))
    return mean, torch.nanmedian(dn, dim=1).values, mx


rows = {k: [] for k in ("lit", "mean", "nn", "base", "ang", "ev", "lay")}
seen, seen_xy, names = set(), set(), []
for s, f in LISTS.items():
    for line in open(f):
        if not line.strip():
            continue
        name, path = line.strip().split("=", 1)
        if path in seen:
            continue
        seen.add(path)
        L = torch.load(path, map_location="cpu")
        e, n = torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev)
        key = tuple(sorted(zip(np.round(e.cpu().numpy(), 1), np.round(n.cpu().numpy(), 1))))
        if key in seen_xy:
            continue
        seen_xy.add(key)
        names.append(name)
        nd = e.numel()
        pos = torch.stack([e, n], -1)
        xy = pos[None].expand(B, -1, -1)
        with torch.no_grad():
            lab = kern(prim, xy)
            feats = torch.zeros(B, C.N_DETECTORS, 4, device=dev)
            feats[:, :nd, :2] = xy
            feats[:, :nd, 2:] = lab
            E_p, th_p, ph_p = primary_to_physical_labels(recon(feats))
            a, _ = DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t))
            lit = torch.expm1(lab[..., 0]) >= 1
            mean, nn_, base = lit_geometry(pos, lit)
        for k, v in (("lit", lit.sum(1)), ("mean", mean), ("nn", nn_), ("base", base), ("ang", a)):
            rows[k].append(v.cpu().numpy())
        rows["ev"].append(np.arange(B))
        rows["lay"] += [len(names) - 1] * B
        print(f"[{name:28s}] n={nd:3d} lit med {np.median(rows['lit'][-1]):.0f}  lit NN med "
              f"{np.nanmedian(rows['nn'][-1]):6.0f} m  within 1° {np.mean(rows['ang'][-1] < 1):.3f}", flush=True)
R = {k: (np.concatenate(v) if k != "lay" else np.array(v)) for k, v in rows.items()}
np.savez(f"{Z}/lit_geometry_pairs.npz", names=np.array(names), **R)
print(f"{len(names)} distinct layouts, {R['ang'].size} shower-layout pairs")

EDGES = [0, 50, 100, 200, 300, 450, 600, 800, 1200, 1e9]
LABS = ["<50", "50-100", "100-200", "200-300", "300-450", "450-600", "600-800", "800-1200", "1200+"]
GROUPS = (("3–5 lit", (3, 6), "#eb6834"), ("6–15 lit", (6, 16), "#2a78d6"), ("16+ lit", (16, 10 ** 6), "#8a5cd1"))
rng = np.random.default_rng(0)
BOOT = [np.bincount(rng.integers(0, B, B), minlength=B) for _ in range(200)]


def binned(x, sel):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        k = sel & (x >= lo) & (x < hi)
        a, ev = R["ang"][k], R["ev"][k]
        if a.size < 50:
            out.append(None); continue
        bs = []
        for w in BOOT:
            ww = w[ev].astype(float)
            if ww.sum() == 0:
                continue
            o = np.argsort(a); cw = np.cumsum(ww[o]) / ww.sum()
            bs.append((a[o][np.searchsorted(cw, 0.5)], np.sum(ww * (a < 1)) / ww.sum()))
        lo_, hi_ = np.percentile(np.array(bs), [16, 84], axis=0)
        out.append(dict(n=int(a.size), showers=int(np.unique(ev).size), med=float(np.median(a)), f1=float(np.mean(a < 1)),
                        med_band=[float(lo_[0]), float(hi_[0])], f1_band=[float(lo_[1]), float(hi_[1])]))
    return out


res = {}
fig, axs = plt.subplots(2, 3, figsize=(21, 11))
x = np.arange(len(LABS))
for j, (key, title) in enumerate((("mean", "mean distance between lit detectors"),
                                  ("nn", "median nearest-neighbour spacing of lit detectors"),
                                  ("base", "baseline: largest distance between lit detectors"))):
    for gname, (lo, hi), col in GROUPS:
        r = binned(R[key], (R["lit"] >= lo) & (R["lit"] < hi))
        res[f"{key}|{gname}"] = r
        ok = [i for i, s in enumerate(r) if s]
        if not ok:
            continue
        axs[0, j].plot(x[ok], [r[i]["med"] for i in ok], "o-", color=col, lw=2, label=gname)
        axs[0, j].fill_between(x[ok], *np.array([r[i]["med_band"] for i in ok]).T, color=col, alpha=0.15, lw=0)
        axs[1, j].plot(x[ok], [100 * r[i]["f1"] for i in ok], "o-", color=col, lw=2, label=gname)
        axs[1, j].fill_between(x[ok], *(100 * np.array([r[i]["f1_band"] for i in ok])).T, color=col, alpha=0.15, lw=0)
    for i, (yl, yscale) in enumerate((("median angle error [deg]", "log"), ("share within 1° [%]", "linear"))):
        ax = axs[i, j]
        ax.set_xticks(x); ax.set_xticklabels(LABS, rotation=45); ax.set_xlabel(f"{title} [m]")
        ax.set(ylabel=yl, yscale=yscale); ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=9)
    axs[0, j].set_title(title.capitalize())
axs[0, 0].set_ylim(0.9, 20); axs[0, 1].set_ylim(0.9, 20); axs[0, 2].set_ylim(0.9, 20)
for a_ in axs[1]:
    a_.set_ylim(0, 25)
fig.suptitle(f"Relative-time recon · {len(names)} distinct layouts × {B} held-out showers · lit = ≥ 1 count · "
             "bands: 16–84% over shower resamples", fontsize=12.5)
fig.tight_layout()
fig.savefig(f"{Z}/lit_geometry.png", dpi=120, facecolor="white", bbox_inches="tight")

tab = {}
for lab_, sel in (("within 1°", R["ang"] < 1), ("1–2.5°", (R["ang"] >= 1) & (R["ang"] < 2.5)),
                  ("2.5–5°", (R["ang"] >= 2.5) & (R["ang"] < 5)), ("> 5°", R["ang"] >= 5)):
    s = sel & (R["lit"] >= 2)
    tab[lab_] = dict(pairs=int(s.sum()), lit=float(np.median(R["lit"][s])), mean=float(np.nanmedian(R["mean"][s])),
                     nn=float(np.nanmedian(R["nn"][s])), base=float(np.nanmedian(R["base"][s])))
json.dump(dict(bins=LABS, table=tab, res=res), open(f"{Z}/lit_geometry.json", "w"), indent=1)
print("showers with >= 2 lit, medians:")
for k, v in tab.items():
    print(f"  {k:10s} pairs {v['pairs']:6d}  n_lit {v['lit']:4.0f}  mean dist {v['mean']:5.0f} m  NN {v['nn']:5.0f} m  "
          f"baseline {v['base']:5.0f} m")
for key in ("mean", "nn", "base"):
    r = res[f"{key}|6–15 lit"]
    print(f"6–15 lit vs {key}: " + "  ".join(f"{l}: {s['med']:.2f}°/{100 * s['f1']:.0f}%" for l, s in zip(LABS, r) if s))
print("[write] lit_geometry.png lit_geometry.json lit_geometry_pairs.npz")
