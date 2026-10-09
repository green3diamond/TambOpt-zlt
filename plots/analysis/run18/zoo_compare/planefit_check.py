"""Gate for the attention recon: does a plain plane-front fit on the kernel times beat the recon?

Same 112 distinct test layouts x 1,536 held-out showers as lit_geometry.py (kernel labels, relative-time
recon by default). Per shower-layout pair, from the timed detectors with >= 0.1 count: a least-squares fit
    t_i = t0 + a (p_i . u1) + b (p_i . u2)
in the plane of the mesh (u1, u2 from a PCA of the mesh centroids, normal m), p_i = (E, N, Up) in metres,
t in us. The in-plane part of the direction is c (a u1 + b u2) (capped at length 1); the out-of-plane part
is +-sqrt(1 - |.|^2) along m, with ONE global sign (the sign most true directions have; also reported:
the per-shower best sign, an upper bound). Weighted (log1p E) and unweighted fits. Fit only with >= 3
timed detectors whose in-plane spread is >= 10 m (smallest singular value of the centred coordinates).
Writes planefit_check.json and planefit_check.png (median error and share within 1 deg vs the number of
timed detectors: recon vs fit, on the same pairs).

    python planefit_check.py [--recon-dir <dir>]
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
C_M_PER_US = 299.792458

ap = argparse.ArgumentParser()
ap.add_argument("--recon-dir", default=os.path.join(R18, "test_v6_run_03_recentered_deepsets_zoo_reltime"))
ap.add_argument("--n-events", type=int, default=1536)
ap.add_argument("--e-min", type=float, default=0.1, help="counts a timed detector needs to enter the fit")
args = ap.parse_args()
LISTS = (f"{D}/core_test_ndet/layouts.txt", f"{D}/clusters/layouts.txt", f"{Z}/layouts_angleonly.txt",
         f"{Z}/layouts_disks_n10.txt")

dev = DCS.DEVICE
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
_, recon = load_models(dev, recon_dir=args.recon_dir, with_fnn=False)
clouds, B, _, prim = load_events(args.n_events, dev, m)
kern = GS.CachedKernel(clouds, surface, dev, chunk=64, expected_primary=prim)
E_true, th_t, ph_t = primary_to_physical_labels(prim)
n_true = DCS.sph(th_t, ph_t)                                                   # (B, 3) unit, ENU

# Plane basis from the mesh: u1, u2 in the plane, m_hat its normal.
cen = torch.as_tensor(np.asarray(m.centroids_ENU), dtype=torch.float32, device=dev)
_, _, Vt = torch.linalg.svd(cen - cen.mean(0), full_matrices=False)
u1, u2, m_hat = Vt[0], Vt[1], Vt[2]
frac_pos = float(((n_true @ m_hat) > 0).float().mean())
SIGN = 1.0 if frac_pos >= 0.5 else -1.0
print(f"[plane] normal {m_hat.tolist()}  true directions with n.m > 0: {frac_pos:.3f} -> global sign {SIGN:+.0f}")


def angle(u, v):
    return torch.rad2deg(2 * torch.atan2((u - v).norm(dim=-1), (u + v).norm(dim=-1)))


def plane_fit(P, t, sel, w):
    """P (nd, 3) m, t (B, nd) us, sel (B, nd) bool, w (B, nd) weights -> (n_fit (B, 3), ok (B,))."""
    q = torch.stack([P @ u1, P @ u2], -1)                                      # (nd, 2) in-plane coords
    w = w * sel
    sw = w.sum(1, keepdim=True).clamp(min=1e-12)
    qm = (w[..., None] * q).sum(1) / sw                                        # (B, 2) weighted centre
    tm = (w * t).sum(1, keepdim=True) / sw
    dq = q[None] - qm[:, None, :]                                              # (B, nd, 2)
    A = torch.einsum("bn,bni,bnj->bij", w, dq, dq)                             # (B, 2, 2)
    rhs = torch.einsum("bn,bni,bn->bi", w, dq, t - tm)                         # (B, 2)
    # In-plane spread (unweighted) for the collinearity cut.
    ns = sel.sum(1)
    cq = torch.where(sel[..., None], q[None] - (q[None] * sel[..., None]).sum(1, keepdim=True) / ns.clamp(min=1)[:, None, None], 0.0)
    smin = torch.linalg.svdvals(cq)[:, -1] / ns.clamp(min=1).sqrt()
    ok = (ns >= 3) & (smin >= 10.0)
    A = torch.where(ok[:, None, None], A, torch.eye(2, device=dev).expand_as(A))   # unfittable rows: dummy solve
    ab = torch.linalg.solve(A, rhs[..., None]).squeeze(-1)                     # (B, 2) us / m
    npar = C_M_PER_US * (ab[:, :1] * u1 + ab[:, 1:] * u2)                      # (B, 3)
    L = npar.norm(dim=-1, keepdim=True)
    npar = torch.where(L > 1, npar / L, npar)
    nperp = (1 - npar.pow(2).sum(-1, keepdim=True)).clamp(min=0).sqrt()
    return npar, nperp, ok


rows = {k: [] for k in ("nt", "ok", "rec", "fw", "fu", "fbest", "lay")}
seen, seen_xy, names = set(), set(), []
for f in LISTS:
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
        xy = torch.stack([e, n], -1)[None].expand(B, -1, -1)
        with torch.no_grad():
            lab = kern(prim, xy)
            feats = torch.zeros(B, C.N_DETECTORS, 4, device=dev)
            feats[:, :nd, :2] = xy
            feats[:, :nd, 2:] = lab
            E_p, th_p, ph_p = primary_to_physical_labels(recon(feats))
            a_rec, _ = DCS.ang_err((E_p.clamp(min=1.0), th_p, ph_p, E_true, th_t, ph_t))
            P = torch.stack([e, n, surface(n, e)], -1)                         # (nd, 3) ENU m
            Ecnt = torch.expm1(lab[..., 0])
            t_us = torch.expm1(lab[..., 1])
            sel = (lab[..., 1] > 0) & (Ecnt >= args.e_min)
            out = {}
            for tag, w in (("fw", torch.log1p(Ecnt)), ("fu", torch.ones_like(Ecnt))):
                npar, nperp, ok = plane_fit(P, t_us, sel, w)
                out[tag] = angle(npar + SIGN * nperp * m_hat, n_true)
                if tag == "fw":
                    alt = angle(npar - SIGN * nperp * m_hat, n_true)
                    out["fbest"] = torch.minimum(out[tag], alt)
                    out["ok"] = ok
                    out["agree"] = ((npar * n_true).sum(-1) > 0)[ok].float().mean() if ok.any() else torch.tensor(float("nan"))
        rows["nt"].append(sel.sum(1).cpu().numpy()); rows["ok"].append(out["ok"].cpu().numpy())
        rows["rec"].append(a_rec.cpu().numpy())
        for k in ("fw", "fu", "fbest"):
            rows[k].append(out[k].cpu().numpy())
        rows["lay"] += [len(names) - 1] * B
        okm = out["ok"]
        print(f"[{name:28s}] n={nd:3d} fit ok {okm.float().mean():.2f}  on fitted: recon med "
              f"{a_rec[okm].median() if okm.any() else float('nan'):5.2f}°  fit med "
              f"{out['fw'][okm].median() if okm.any() else float('nan'):5.2f}°  in-plane sign agrees {float(out['agree']):.2f}", flush=True)
R = {k: (np.concatenate(v) if k != "lay" else np.array(v)) for k, v in rows.items()}
np.savez(f"{Z}/planefit_pairs.npz", names=np.array(names), **R)

BINS = ((3, 4, "3"), (4, 6, "4–5"), (6, 11, "6–10"), (11, 21, "11–20"), (21, 10 ** 6, "21+"))
res = {}
for lo, hi, lab_ in BINS:
    k = R["ok"].astype(bool) & (R["nt"] >= lo) & (R["nt"] < hi)
    if k.sum() < 50:
        continue
    res[lab_] = dict(pairs=int(k.sum()), **{
        f"{t}_{s}": float(f(R[t][k])) for t in ("rec", "fw", "fu", "fbest")
        for s, f in (("med", np.median), ("f1", lambda a: np.mean(a < 1)), ("f25", lambda a: np.mean(a < 2.5)),
                     ("f5", lambda a: np.mean(a < 5)))})
allok = R["ok"].astype(bool)
summary = dict(pairs=int(R["rec"].size), fitted=int(allok.sum()), sign=SIGN, frac_pos=frac_pos, e_min=args.e_min,
               recon_dir=args.recon_dir, bins=res,
               all_fitted={t: dict(med=float(np.median(R[t][allok])), f1=float(np.mean(R[t][allok] < 1)))
                           for t in ("rec", "fw", "fu", "fbest")})
json.dump(summary, open(f"{Z}/planefit_check.json", "w"), indent=1)

labs = list(res)
x = np.arange(len(labs))
SER = (("rec", "recon (relative-time)", "#2a78d6"), ("fw", "plane fit, log1p E weights", "#eb6834"),
       ("fu", "plane fit, unweighted", "#1baf7a"), ("fbest", "plane fit, best sign (bound)", "#9b9a95"))
fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
for t, lab_, col in SER:
    axs[0].plot(x, [res[b][f"{t}_med"] for b in labs], "o-", color=col, lw=2, label=lab_,
                ls="--" if t == "fbest" else "-")
    axs[1].plot(x, [100 * res[b][f"{t}_f1"] for b in labs], "o-", color=col, lw=2, label=lab_,
                ls="--" if t == "fbest" else "-")
for ax, yl in zip(axs, ("median angle error [deg]", "share within 1° [%]")):
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n({res[b]['pairs']:,})" for b in labs])
    ax.set_xlabel("timed detectors with ≥ 0.1 count (pairs)"); ax.set_ylabel(yl)
    ax.grid(alpha=0.25, lw=0.5); ax.legend(frameon=False, fontsize=9)
axs[0].set_yscale("log")
fig.suptitle(f"Plane-front fit on kernel times vs the recon · {len(names)} layouts × {B} held-out showers · "
             f"{allok.sum():,} of {R['rec'].size:,} pairs fittable", fontsize=12)
fig.tight_layout()
fig.savefig(f"{Z}/planefit_check.png", dpi=120, facecolor="white", bbox_inches="tight")
for b in labs:
    r = res[b]
    print(f"{b:6s} pairs {r['pairs']:7d}  median recon {r['rec_med']:5.2f}°  fit {r['fw_med']:5.2f}° "
          f"(unw {r['fu_med']:5.2f}°, best sign {r['fbest_med']:5.2f}°)  within 1°: recon {100 * r['rec_f1']:4.1f}%  "
          f"fit {100 * r['fw_f1']:4.1f}%")
print("[write] planefit_check.json planefit_check.png planefit_pairs.npz")
