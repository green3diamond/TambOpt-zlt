"""Expected response (modules/optimize/expected_response.py, Jay's symbolic model) against the kernel on CORSIKA 8
test showers, Malata uniform grid. Showers and placement as score_all.py SRC=c8 (each C8 test shower at the decay
vertex of the run-20 tau with the closest direction); the expected response uses the same primaries (energy,
direction, class from the pdg) and frames.

Per source: hits per shower (mu >= 1 with a time), trigger share (>= 3 hits), mean mu vs distance from the axis,
the arrival delay behind the plane front (T - depth / c) of detectors hit in both, and the analytical recon v1
(share within 1 deg and median, over each source's own triggered showers).

    python check_expected.py <out.json> [n_showers]          (under slurm/run20_guard.sh, SRC is set here)
"""
import importlib.util, json, os, sys, time
import numpy as np, torch

OUT = sys.argv[1]; N_EV = sys.argv[2] if len(sys.argv) > 2 else "1000"
os.environ["SRC"] = "c8"
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R)
sys.argv = ["score_all.py", os.devnull + ".none", N_EV]
spec = importlib.util.spec_from_file_location("score_all", f"{R}/scripts/paper2/score_all.py")
SA = importlib.util.module_from_spec(spec); spec.loader.exec_module(SA)
S4, C, dev, surface, mountain = SA.S4, SA.C, SA.dev, SA.surface, SA.mountain
from modules.optimize import analytic_reco as AR
from modules.optimize import expected_response as XR
from modules.data.dataset_builder import detectors_in_frames

t0 = time.time()
prim = SA.prim
dirs = prim[:, :3].numpy().astype(np.float64)
log10_e = (prim[:, 3] * (C.LOG_E_MAX - C.LOG_E_MIN) + C.LOG_E_MIN).numpy().astype(np.float64)
pdg = prim[:, 4].round().long().numpy()
cls = pdg.clip(0, 1)                       # the C8 files' pdg field is the class: 0 EM, 1 hadronic
part1, laws = XR.load_model()
logd, tconst, loge = XR.tables(part1, laws, log10_e, dirs, cls, seed=1, energy_density=True)
print(f"tables {logd.shape}, {time.time() - t0:.0f}s; classes {np.bincount(cls)}", flush=True)

L = torch.load(f"{C.RUN_LOCATION}/diagnostics/paper1/ugrid.pt", map_location="cpu", weights_only=False)
x = torch.as_tensor(L["x"]).float().to(dev); y = torch.as_tensor(L["y"]).float().to(dev)
xy = torch.stack([x, y], -1)
n_plane = AR.plane_normal(torch.as_tensor(mountain.centroids_ENU, dtype=torch.float32)).to(dev)
X3 = torch.stack([x, y, surface(y, x)], -1)
acc = {k: {w: [] for w in ("mu", "t_us", "rho", "depth", "ang")} for k in ("kernel", "expected")}
Esp = {k: [] for k in ("kernel", "expected")}       # per-species detector energy [kernel units: GeV / 628 m^2]
with torch.no_grad():
    for lo in range(0, SA.N, SA.CH):
        hi = min(lo + SA.CH, SA.N)
        fr = SA.frames[lo:hi].to(dev); p = prim[lo:hi].to(dev)
        kern = S4.BatchKernel([c[lo:hi].to(dev) for c in SA.clouds], [None] * len(SA.clouds), surface,
                              differentiable=False, frames=fr)
        exp_ = XR.ExpectedResponse(torch.as_tensor(logd[lo:hi]).to(dev), torch.as_tensor(tconst[lo:hi]).to(dev), fr,
                                   surface)
        rho, z = detectors_in_frames(x, y, surface, fr)
        # per-species detector energy: kernel E_s, and the model's energy density x 2 pi sigma0^2 (= 628 m^2) read as mu
        ks_ = [S4.compute_labels_batch(c[lo:hi].to(dev), x, y, surface, differentiable=False, frames=fr)[0] for c in SA.clouds]
        Esp["kernel"].append(torch.stack(ks_, 1).float().cpu())
        ee = XR.ExpectedResponse(torch.as_tensor(loge[lo:hi]).to(dev) + float(np.log(2 * np.pi * 100.0 / XR.AREA)),
                                 torch.as_tensor(tconst[lo:hi]).to(dev), fr, surface)
        ee(p, xy[None].expand(hi - lo, -1, -1))
        Esp["expected"].append(torch.stack([mm / float(XR.EFF[a]) for a, mm in enumerate(ee._mu_s)], 1).float().cpu())
        for name, resp in (("kernel", kern), ("expected", exp_)):
            out = resp(p, xy[None].expand(hi - lo, -1, -1))
            t_us = torch.expm1(out[..., 1].clamp(min=0)); mu = resp.mu
            hit = (mu >= 1) & (t_us > 0)
            rr = AR.RUNS["v1"](X3[None].expand(hi - lo, -1, -1), t_us, mu, hit, n_plane)
            dtrue = torch.nn.functional.normalize(p[:, :3], dim=-1)
            a = torch.rad2deg(torch.arccos((rr["dir"] * dtrue).sum(-1).clamp(-1, 1)))
            a = torch.where(rr["valid"] & (hit.sum(1) >= 3), a, torch.full_like(a, float("nan")))
            for w, v in (("mu", mu), ("t_us", t_us), ("rho", rho), ("depth", z * 500.0), ("ang", a)):
                acc[name][w].append(v.float().cpu())
A = {k: {w: torch.cat(v).numpy() for w, v in d.items()} for k, d in acc.items()}

res = dict(n_showers=int(SA.N), classes=np.bincount(cls).tolist(), smooth_h=XR.SMOOTH_H)
EK, EX = torch.cat(Esp["kernel"]).numpy(), torch.cat(Esp["expected"]).numpy()        # (N, 3, nd)
rho_all = A["kernel"]["rho"]
res["energy_vs_rho"] = {s: {f"{rb_[i]:g}-{rb_[i + 1]:g}": [float(EK[:, a][(rho_all >= rb_[i]) & (rho_all < rb_[i + 1])].mean()),
                                                         float(EX[:, a][(rho_all >= rb_[i]) & (rho_all < rb_[i + 1])].mean())]
                            for i in range(len(rb_) - 1) if ((rho_all >= rb_[i]) & (rho_all < rb_[i + 1])).any()}
                        for a, s in enumerate(C.SPECIES_NAMES)
                        for rb_ in [np.array([0, 20, 50, 100, 200, 400, 800, 1600, 1e5])]}
res["energy_vs_rho_note"] = "mean detector energy [GeV per 628 m^2]: [kernel, model energy density x 628 m^2]"
rb = np.array([0, 20, 50, 100, 200, 400, 800, 1600, 1e5])
for k, d in A.items():
    hit = (d["mu"] >= 1) & (d["t_us"] > 0); nh = hit.sum(1); trig = nh >= 3
    ang = d["ang"][trig]
    r = dict(hits_mean=float(nh.mean()), hits_median=float(np.median(nh)), trigger=float(trig.mean()),
             v1_within1=float(np.mean(ang < 1)) if trig.any() else None,
             v1_median=float(np.nanmedian(ang)) if trig.any() else None,
             mu_mean_vs_rho={f"{rb[i]:g}-{rb[i + 1]:g}": float(d["mu"][(d["rho"] >= rb[i]) & (d["rho"] < rb[i + 1])].mean())
                             for i in range(len(rb) - 1) if ((d["rho"] >= rb[i]) & (d["rho"] < rb[i + 1])).any()})
    res[k] = r
hk = (A["kernel"]["mu"] >= 1) & (A["kernel"]["t_us"] > 0)
he = (A["expected"]["mu"] >= 1) & (A["expected"]["t_us"] > 0)
both = hk & he
lag = {k: (A[k]["t_us"] * 1e3 - A[k]["depth"] / 0.299792458)[both] for k in A}      # ns behind the plane front
res["both_hit"] = dict(n=int(both.sum()), hit_agreement=float((hk == he).mean()),
                       jaccard=float(both.sum() / max(1, (hk | he).sum())),
                       delay_kernel_median_ns=float(np.median(lag["kernel"])) if both.any() else None,
                       delay_expected_median_ns=float(np.median(lag["expected"])) if both.any() else None,
                       delay_diff_median_ns=float(np.median(lag["expected"] - lag["kernel"])) if both.any() else None)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
print(f"done {time.time() - t0:.0f}s", flush=True)
