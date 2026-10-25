"""Run 19 quick test, kNN 2-of-3 gate: every detector is read out only if at least 2 of {itself, its 2 nearest
neighbours (in the layout, East-North)} are hit (hit = E' > 0.001); otherwise E = 0, untimed, before the recon.
Same noise draws for every readout ('none' = all read out, 'knn' = gated; tri* also 'group' = 2-of-3 within its own
triangle). Clean run 19 recon.
Layouts: grid_n99, disk_r6000_n99, all tri33_* (triplets/layouts.txt) and clu20x5_r1000_n100 (tests/layouts.txt).
Writes knn_scores.json: {layout: {readout: {W: metrics}}} (angle and energy: *_e = |dlog10 E| < 0.2, *_5e = both). A background muon's E uses the detector's distance from
the shower axis. --counting poisson draws particle counts per species before the species combination
(modules/optimize/counting.py; hit = any counted particle) and writes knn_scores_poisson.json unless --out is given.

    python score_knn.py [--windows 0 128 1000 10000] [--counting poisson --eps-gamma 0.02]
"""
import argparse
import importlib.util
import json
import os
import re
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.optimize import load_models  # noqa: E402
from modules.optimize.objective import primary_to_physical_labels  # noqa: E402
from modules.optimize.background import window_centres, add_background, axis_distance  # noqa: E402
from modules.optimize.counting import CountingTransform, EPS_GAMMA, trigger_prob  # noqa: E402
from plots.layouts.true_utility import load_events  # noqa: E402

_s = importlib.util.spec_from_file_location("dcs", os.path.join(REPO, "scripts/diagnostics/detector_count_scan.py"))
DCS = importlib.util.module_from_spec(_s); _s.loader.exec_module(DCS)
T_DIR = os.path.join(C.RUN_LOCATION, "diagnostics", "triplets")
MIN_HIT = 3
E_GOOD = 0.2                                                                       # |dlog10 E| for "energy good" (two-tier e_good)
HIT_MIN = 1e-3                                                                     # the kernel's timing threshold (HIT_DEPOSIT_MIN)
READOUTS = ("none", "knn")

ap = argparse.ArgumentParser()
ap.add_argument("--windows", type=float, nargs="+", default=[0, 128, 1000, 10000])
ap.add_argument("--n-events", type=int, default=5000)
ap.add_argument("--event-start", type=int, default=0, help="first held-out event (score the held-out set in chunks)")
ap.add_argument("--recon-dir", default=C.RECON_FOLDER + "_deepsets")
ap.add_argument("--out", default=None, help="default knn_scores.json, or knn_scores_poisson.json with --counting poisson")
ap.add_argument("--counting", choices=("none", "poisson"), default="none")
ap.add_argument("--eps-gamma", type=float, default=EPS_GAMMA, help="photon detection probability (counting only)")
ap.add_argument("--count-seed", type=int, default=7)
ap.add_argument("--mu-min", type=float, default=1.0, help="expected-count trigger: a detector counts if mu >= this")
ap.add_argument("--k-min", type=int, default=3, help="expected-count trigger: >= this many such read-out detectors")
ap.add_argument("--layouts", default=None, help="name=path.pt lines instead of the default triplet / grid / disk set")
ap.add_argument("--drop-test", type=float, default=0.0,
                help="robustness test: this fraction of the detectors, chosen at random per shower (seeded), is missing "
                     "(E = T = 0, mu = 0) after background and before the gate; 0 = off")
ap.add_argument("--axis-within", type=float, default=0.0,
                help="keep only showers whose axis passes within this distance [m] of the plane centre (0 = all)")
ap.add_argument("--core-in-layout", action="store_true",
                help="score each layout only on showers whose axis passes within the layout's radius of its centroid")
args = ap.parse_args()
if args.out is None:
    args.out = os.path.join(T_DIR, "knn_scores.json" if args.counting == "none" else "knn_scores_poisson.json")

dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, recon_dir=args.recon_dir, with_fnn=False)
CENTRE = torch.as_tensor(m.centroids_ENU, dtype=torch.float32).mean(0)            # plane centre (E, N, U)


def near_centre(pos, dirs):
    v = CENTRE - pos
    return (v - (v * dirs).sum(1, keepdim=True) * dirs).norm(dim=1) < args.axis_within


clouds, B, _, prim = load_events(args.n_events, dev, m, select=near_centre if args.axis_within > 0 else None,
                                 start=args.event_start)
ct = CountingTransform(surface, None, mode="poisson" if args.counting == "poisson" else "expected", eps_gamma=args.eps_gamma)
kern = DCS.GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim, species_transform=ct)
E_true, th_t, ph_t = primary_to_physical_labels(prim)
T_FALLBACK = None

layouts = {}
for f, pat in ([(args.layouts, r".+")] if args.layouts else []) or ((os.path.join(T_DIR, "layouts.txt"), r"grid_n99|disk_r6000_n99|tri33_r\d+_n99"),
               (os.path.join(C.RUN_LOCATION, "diagnostics", "tests", "layouts.txt"), r"clu20x5_r1000_n100")):
    for line in open(f):
        k, p = line.strip().split("=", 1)
        if re.fullmatch(pat, k):
            layouts[k] = p
print(f"{len(layouts)} triplet layouts, windows {args.windows} ns, recon {args.recon_dir}", flush=True)

out = {}
for li, (name, path) in enumerate(layouts.items()):
    L = torch.load(path, map_location="cpu")
    e, n = torch.as_tensor(L["x"]).float().to(dev), torch.as_tensor(L["y"]).float().to(dev)
    xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
    nd = e.numel()
    d = torch.cdist(torch.stack([e, n], -1), torch.stack([e, n], -1)).fill_diagonal_(float("inf"))
    nn = d.topk(2, largest=False).indices                                          # (nd, 2) nearest neighbours
    tri = name.startswith("tri")                                                   # stored triangle by triangle
    rds = READOUTS + (("group",) if tri else ())
    res = {r: {} for r in rds}
    if tri:
        mates = torch.arange(nd, device=dev).view(-1, 3).repeat_interleave(3, 0)  # (nd, 3) own triangle
        same = float((torch.sort(nn, 1).values == torch.sort(mates[mates != torch.arange(nd, device=dev)[:, None]]
                                                              .view(nd, 2), 1).values).all(1).float().mean())
        res["knn_is_triangle"] = same                                              # share of detectors whose kNN = mates
    r_axis = axis_distance(e, n, surface, kern.frames)                           # (B, nd) m, background muon scaling
    if args.core_in_layout:                                                      # axis passes inside the layout
        ce, cn = e.mean().view(1), n.mean().view(1)
        sub = axis_distance(ce, cn, surface, kern.frames)[:, 0] < float(torch.hypot(e - ce, n - cn).max())
    else:
        sub = torch.ones(B, dtype=torch.bool, device=dev)
    ct.gen = torch.Generator(device=dev); ct.gen.manual_seed(1000003 * args.count_seed + li); ct.n_detected, ct.mu = [], []
    with torch.no_grad():
        lab = kern(prim, xy)
        n_counted = float(torch.cat(ct.n_detected)[sub].mean()) if ct.n_detected else float("nan")
        mu = torch.cat(ct.mu)                                                      # (B, nd) expected detected particles
        E = torch.expm1(lab[..., 0].clamp(min=0)); t = torch.expm1(lab[..., 1].clamp(min=0)); timed = lab[..., 1] > 0
        if T_FALLBACK is None:
            T_FALLBACK = float(t[timed].median()) if timed.any() else 10.0
        t_c = window_centres(t, timed, T_FALLBACK)
        for W in args.windows:
            g = torch.Generator(device=dev); g.manual_seed(int(1000003 * li + W))
            En, Tn, Nb, hasbg = add_background(E, t, timed, t_c, W, g, r_axis=r_axis)
            ghost = hasbg & ~timed                                                 # background-only detectors
            alive = torch.ones_like(En, dtype=torch.bool)
            if args.drop_test > 0:                                                 # missing detectors
                gd = torch.Generator(device=dev); gd.manual_seed(int(7919 * li + W) + 1)
                alive = torch.rand(En.shape, device=dev, generator=gd) >= args.drop_test
                En, Tn = En * alive, Tn * alive
            for rd in rds:
                h = (En > HIT_MIN).float()
                if rd == "none":
                    keep = torch.ones_like(En, dtype=torch.bool)
                elif rd == "knn":
                    keep = (h + h[:, nn[:, 0]] + h[:, nn[:, 1]]) >= 2                # kNN 2-of-3
                else:
                    keep = (h.view(B, -1, 3).sum(-1, keepdim=True) >= 2).expand(B, -1, 3).reshape(B, nd)   # own triangle
                Ek, Tk = torch.where(keep, En, 0.0), torch.where(keep, Tn, 0.0)
                E_p, th_p, ph_p = primary_to_physical_labels(recon(torch.cat([xy, torch.log1p(Ek)[..., None],
                                                                             torch.log1p(Tk)[..., None]], -1)))
                a, de = (x[sub].cpu().numpy() for x in DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t)))
                trig = ((Ek[sub] > 1).sum(1) >= MIN_HIT).cpu().numpy()
                r = dict(n_det=nd, n_showers=int(sub.sum()), n_counted=n_counted, triggered=float(trig.mean()),
                         ang_med=float(np.median(a)),
                         read_det=float(keep[sub].float().sum(1).mean()), ghosts=float((ghost & keep)[sub].float().sum(1).mean()),
                         lit_det=float(((Ek > 0) | (Tk > 0))[sub].float().sum(1).mean()))
                # notebook-style trigger on EXPECTED counts: >= k_min read-out detectors with mu >= mu_min (hard),
                # and the smooth r_b = sigmoid(5 (sum_i sigmoid(5 (mu_i - mu_min)) - k_min)) as a weight
                mk = (mu * keep * alive)[sub]
                trig_mu = ((mk >= args.mu_min).sum(1) >= args.k_min).cpu().numpy()
                rb = torch.sigmoid(5 * (torch.sigmoid(5 * (mk - args.mu_min)).mul(keep[sub]).sum(1) - args.k_min)).cpu().numpy()
                r.update(mu_det=float((mk >= args.mu_min).float().sum(1).mean()), trig_mu=float(trig_mu.mean()),
                         r_soft=float(rb.mean()))
                # energy: |dlog10 E| < E_GOOD (the two-tier e_good); "5e" = within 5 deg and energy good
                eg = de < E_GOOD
                # The Stage 4 utilities, unscaled (the optimiser logs them x30):
                #   u_two_tier   mean[g + 0.3 (1 - g) q], g = soft(a < 5) soft(|dlog10 E| < 0.2), q = 1/(1 + (a/20)^2 + (e/0.5)^2)
                #   u_swgo       mean[r_b (g_a + 0.3 (1 - g_a) q_a)], angle only; r_b = P(>= 3 read-out detectors with
                #                >= 1 particle) (Poisson-binomial) or, u_swgo_step, sigmoid(5 (sum sigmoid(10 (mu - 1)) - 3))
                at, et = torch.as_tensor(a), torch.as_tensor(de)
                g2 = torch.sigmoid(5.0 - at) * torch.sigmoid((0.2 - et) / 0.05)
                q2 = 1.0 / (1.0 + (at / 20.0) ** 2 + (et / 0.5) ** 2)
                ga, qa = torch.sigmoid(5.0 - at), 1.0 / (1.0 + (at / 20.0) ** 2)
                rp = trigger_prob(mk.double().cpu(), 3).float()
                rs = torch.sigmoid(5.0 * (torch.sigmoid(10.0 * (mk.cpu() - 1.0)).sum(1) - 3.0))
                # drawn trigger (--counting poisson): >= 3 read-out detectors with a drawn particle (E' > 1 kernel units:
                # one counted particle is >> 1; background muons count too), the trigger a real array would apply
                td = torch.as_tensor(trig).float()
                r.update(ang_mean_trig=float(a[trig].mean()) if trig.any() else float("nan"),   # angle error [deg]
                         ang_med_trig=float(np.median(a[trig])) if trig.any() else float("nan"))  # over triggered
                r.update(u_swgo_drawn=float((td * (ga + 0.3 * (1 - ga) * qa)).mean()))
                r.update(u_two_tier=float((g2 + 0.3 * (1 - g2) * q2).mean()),
                         u_swgo=float((rp * (ga + 0.3 * (1 - ga) * qa)).mean()),
                         u_swgo_step=float((rs * (ga + 0.3 * (1 - ga) * qa)).mean()),
                         trig_poisson=float(rp.mean()), trig_step=float(rs.mean()))
                r.update(notrig_e=float(eg.mean()), allmu_e=float(np.mean(eg & trig_mu)),
                         notrig_5e=float(np.mean(eg & (a < 5))), allmu_5e=float(np.mean(eg & (a < 5) & trig_mu)),
                         de_med=float(np.median(de)), de_med_trig=float(np.median(de[trig_mu])) if trig_mu.any() else float("nan"))
                for th, kk in ((1, "1"), (2.5, "2.5"), (5, "5")):
                    r[f"notrig_{kk}"] = float(np.mean(a < th))
                    r[f"all_{kk}"] = float(np.mean((a < th) & trig))
                    r[f"allmu_{kk}"] = float(np.mean((a < th) & trig_mu))
                    r[f"rw_{kk}"] = float(np.sum(rb * (a < th)) / max(rb.sum(), 1e-9))
                res[rd][f"{W:g}"] = r
    out[name] = res
    s = "  ".join(f"W={w}: " + " ".join(f"{rd} {100*res[rd][w]['notrig_1']:.1f}/{100*res[rd][w]['all_1']:.1f} g{res[rd][w]['ghosts']:.2f}"
                                       for rd in rds) for w in res["none"])
    print(f"[{name:16s}] (1 deg no trig / trig, ghosts) {s}", flush=True)
json.dump(dict(recon_dir=args.recon_dir, event_start=args.event_start, drop_test=args.drop_test, windows_ns=args.windows, min_hit=MIN_HIT, n_events=B, counting=args.counting,
               mu_min=args.mu_min, k_min=args.k_min,
               eps_gamma=args.eps_gamma if args.counting == "poisson" else None, count_seed=args.count_seed,
               bg_muon_scaling="per detector, distance from the axis", scores=out),
          open(args.out, "w"), indent=1)
print("[write]", args.out)
