"""How many detectors must a shower light to be reconstructed within 1 / 2.5 / 5 deg (zoo recon)?

For every test layout (core_test_ndet, clusters, angleonly, 10-det. disks) and each of the 1,536 held-out showers:
the number of detectors with >= 1 count (kernel) and the recon's angle error. Pooled over layouts, binned by
the lit count: share within 1 / 2.5 / 5 deg, for all layouts and split into spread layouts (grids, disks
r >= 2 km, 500 m clusters) and compact ones (cores, disks r <= 500 m, clusters r <= 200 m).
Also the same vs the summed counts. Writes lit_vs_angle.png and lit_vs_angle.json.

    python lit_vs_angle.py [--recon-dir <dir>]
"""
import argparse
import importlib.util
import json
import os
import re
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
ap.add_argument("--recon-dir", default=os.path.join(R18, "test_v6_run_03_recentered_deepsets_zoo"))
ap.add_argument("--n-events", type=int, default=1536)
args = ap.parse_args()

LISTS = {"core_test_ndet": f"{D}/core_test_ndet/layouts.txt", "clusters": f"{D}/clusters/layouts.txt",
         "angleonly": f"{Z}/layouts_angleonly.txt", "disks_n10": f"{Z}/layouts_disks_n10.txt"}


def kind(name):
    if name.startswith("grid") or name.startswith("kern_"):
        return "spread"
    m = re.match(r"disk_r(\d+)", name)
    if m:
        return "spread" if int(m.group(1)) >= 2000 else ("compact" if int(m.group(1)) <= 500 else "mid")
    m = re.match(r"clu\d+x\d+_r(\d+)", name)
    if m:
        return "spread" if int(m.group(1)) >= 500 else ("compact" if int(m.group(1)) <= 200 else "mid")
    if re.match(r"core_r\d+", name):
        return "compact"
    return "mid"


dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, recon_dir=args.recon_dir, with_fnn=False)
clouds, B, _, prim = load_events(args.n_events, dev, m)
kern = GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)
E_true, th_t, ph_t = primary_to_physical_labels(prim)

rows = {"lit": [], "lit5": [], "sum": [], "ang": [], "kind": [], "n": [], "ev": [], "lay": []}
seen, seen_xy = set(), set()
names = []
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
        if key in seen_xy:                       # same detector positions listed under another set
            continue
        seen_xy.add(key)
        names.append(name)
        nd = e.numel()
        xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
        with torch.no_grad():
            lab = kern(prim, xy)                                               # (B, nd, 2) log1p E, T
            feats = torch.zeros(B, C.N_DETECTORS, 4, device=dev)
            feats[:, :nd, :2] = xy
            feats[:, :nd, 2:] = lab
            E_p, th_p, ph_p = primary_to_physical_labels(recon(feats))
            a, _ = DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t))
            cnt = torch.expm1(lab[..., 0])
        rows["lit"].append((cnt >= 1).sum(1).cpu().numpy())
        rows["lit5"].append((cnt >= 5).sum(1).cpu().numpy())
        rows["sum"].append(cnt.sum(1).cpu().numpy())
        rows["ang"].append(a.cpu().numpy())
        rows["kind"] += [kind(name)] * B
        rows["n"] += [nd] * B
        rows["ev"].append(np.arange(B))
        rows["lay"] += [len(names) - 1] * B
        print(f"[{name:28s}] n={nd:3d} lit med {np.median(rows['lit'][-1]):.0f}  within 1° {np.mean(rows['ang'][-1] < 1):.3f}",
              flush=True)
lit, lit5, tot, ang = (np.concatenate(rows[k]) for k in ("lit", "lit5", "sum", "ang"))
kinds, ns = np.array(rows["kind"]), np.array(rows["n"])
evs, lays = np.concatenate(rows["ev"]), np.array(rows["lay"])
np.savez(f"{Z}/lit_vs_angle_pairs.npz", lit=lit, lit5=lit5, tot=tot, ang=ang, kind=kinds, n=ns, ev=evs, lay=lays,
         names=np.array(names))
print(f"{len(names)} distinct layouts, {ang.size} shower-layout pairs")

BINS = list(range(0, 13)) + [15, 20, 30]
lab_bins = [str(b) if b < 12 else f"{b}+" for b in range(0, 12)] + ["12-14", "15-19", "20-29", "30+"]


def binned(x, sel, th):
    out = []
    for lo, hi in zip(BINS, BINS[1:] + [10 ** 9]):
        k = sel & (x >= lo) & (x < hi)
        out.append((float(np.mean(ang[k] < th)) if k.sum() >= 30 else np.nan, int(k.sum())))
    return out


res = {}
COL = {"all": "#1f2328", "spread": "#2a78d6", "compact": "#eb6834"}
fig, axs = plt.subplots(2, 3, figsize=(21, 11))
xs = np.arange(len(BINS))
for j, th in enumerate((1, 2.5, 5)):
    ax = axs[0, j]
    for grp, sel in (("all", np.ones_like(ang, bool)), ("spread", kinds == "spread"), ("compact", kinds == "compact")):
        b = binned(lit, sel, th)
        res[f"{grp}|{th:g}"] = b
        ax.plot(xs, [100 * v for v, _ in b], "o-", color=COL[grp], lw=2 if grp == "all" else 1.5, label=grp)
    ax.set_xticks(xs); ax.set_xticklabels(lab_bins, rotation=45, fontsize=8)
    ax.set(xlabel="detectors with ≥ 1 count", ylabel=f"share within {th:g}° [%]",
           title=f"Within {th:g}° vs detectors lit (≥ 1 count)", ylim=(0, 100))
    ax.grid(alpha=0.25, lw=0.5)
axs[0, 0].legend(frameon=False)
# counts
ax = axs[1, 0]
cb = np.array([0, 1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 1e9])
for th, ls in ((1, "-"), (2.5, "--"), (5, ":")):
    v = [np.mean(ang[(tot >= lo) & (tot < hi)] < th) if ((tot >= lo) & (tot < hi)).sum() >= 30 else np.nan
         for lo, hi in zip(cb[:-1], cb[1:])]
    ax.plot(np.arange(len(v)), 100 * np.array(v), "o" + ls, color="#1f2328", lw=1.8, label=f"within {th:g}°")
ax.set_xticks(np.arange(len(cb) - 1)); ax.set_xticklabels([f"{int(a)}–" if a < 1e9 else "" for a in cb[:-1]], rotation=45, fontsize=8)
ax.set(xlabel="summed counts over all detectors (bin lower edge)", ylabel="share [%]", title="Vs total signal", ylim=(0, 100))
ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False)
# >= 5 counts
ax = axs[1, 1]
for th, ls in ((1, "-"), (2.5, "--"), (5, ":")):
    b = binned(lit5, np.ones_like(ang, bool), th)
    ax.plot(xs, [100 * v for v, _ in b], "o" + ls, color="#1baf7a", lw=1.8, label=f"within {th:g}°")
ax.set_xticks(xs); ax.set_xticklabels(lab_bins, rotation=45, fontsize=8)
ax.set(xlabel="detectors with ≥ 5 counts", ylabel="share [%]", title="Vs detectors with ≥ 5 counts (all layouts)",
       ylim=(0, 100))
ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False)
# population
ax = axs[1, 2]
ax.bar(xs, [n for _, n in binned(lit, np.ones_like(ang, bool), 1)], color="#9aa4b1")
ax.set_xticks(xs); ax.set_xticklabels(lab_bins, rotation=45, fontsize=8)
ax.set(xlabel="detectors with ≥ 1 count", ylabel="shower-layout pairs", title="How often (all layouts pooled)")
ax.grid(alpha=0.25, lw=0.5, axis="y")
fig.suptitle(f"Zoo recon · {len(names)} distinct test layouts × {B} held-out showers pooled · spread = grids, optimized, disks "
             f"≥ 2 km, 500 m clusters; compact = cores, disks ≤ 500 m, clusters ≤ 200 m", fontsize=12.5)
fig.tight_layout()
fig.savefig(f"{Z}/lit_vs_angle.png", dpi=120, facecolor="white", bbox_inches="tight")
json.dump({"bins": lab_bins, "res": {k: [[None if np.isnan(v) else v, c] for v, c in b] for k, b in res.items()}},
          open(f"{Z}/lit_vs_angle.json", "w"), indent=1)
for th in (1, 2.5, 5):
    print(f"within {th:g}°: " + "  ".join(f"{l}:{100 * v:.0f}%" for l, (v, _) in zip(lab_bins, res[f'all|{th:g}'])
                                         if not np.isnan(v)))
print("[write] lit_vs_angle.png lit_vs_angle.json")
