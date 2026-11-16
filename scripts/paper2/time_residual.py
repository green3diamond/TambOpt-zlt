"""Paper 2, Sec. analytic: how far the kernel's hit times lie from a plane front, for flow-matching (held-out) and
CORSIKA 8 test showers. Backs the sentence "the flow-matching hit times scatter by ~3 us RMS around a plane front".

Showers and placement are score_all.py's (SRC=c8 for CORSIKA 8), noise-free kernel, hits as the analytical recon
(RECO, ANA_HIT=mu): >= 1 expected particle and a time. Per shower with >= 3 hits, the RMS of the time residual
  (a) true front:     t - (X . n_true) / c, minus its mean (only t0 free);
  (b) best plane:     least-squares fit t = t0 + g . X (free slowness g; the best any plane front can do; >= 5 hits).
X: detector ENU position [m]; n_true: shower propagation direction; c = 299.792458 m/us; times in us.

    [SRC=c8] python time_residual.py <out.json> <n_showers> [layout.pt]      (under slurm/run20_guard.sh)
"""
import importlib.util, json, os, sys
import numpy as np, torch

OUT, N_EV = sys.argv[1], sys.argv[2]
LAYOUT = sys.argv[3] if len(sys.argv) > 3 else None
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# score_all.py with no layouts only loads the showers (clouds, prim, frames), the surface and the device
sys.argv = ["score_all.py", os.devnull + ".none", N_EV]
spec = importlib.util.spec_from_file_location("score_all", f"{R}/scripts/paper2/score_all.py")
SA = importlib.util.module_from_spec(spec); spec.loader.exec_module(SA)
S4, C, dev, surface = SA.S4, SA.C, SA.dev, SA.surface
C_LIGHT = 299.792458                                        # m / us

lay = LAYOUT or f"{C.RUN_LOCATION}/diagnostics/paper1/ugrid.pt"
L = torch.load(lay, map_location="cpu", weights_only=False)
x = torch.as_tensor(L["x"]).float().to(dev); y = torch.as_tensor(L["y"]).float().to(dev)
X0 = torch.stack([x, y, surface(y, x)], -1)                 # (nd, 3)

rms_true, rms_plane, nhit = [], [], []
with torch.no_grad():
    for lo in range(0, SA.N, SA.CH):
        hi = min(lo + SA.CH, SA.N)
        kern = S4.BatchKernel([c[lo:hi].to(dev) for c in SA.clouds], [None] * len(SA.clouds), surface,
                              differentiable=False, frames=SA.frames[lo:hi].to(dev))
        p = SA.prim[lo:hi].to(dev)
        out = kern(p, X0[None, :, :2].expand(hi - lo, -1, -1))
        t = torch.expm1(out[..., 1].clamp(min=0))           # us
        hit = (kern.mu >= 1) & (t > 0)
        n = torch.nn.functional.normalize(p[:, :3], dim=-1)
        Xc = X0 - X0.mean(0)                                # centred positions (conditioning of the fit)
        for b in range(hi - lo):
            h = hit[b]; k = int(h.sum())
            if k < 3:
                continue
            tb, Xb = t[b, h].double(), Xc[h].double()
            r = tb - Xb @ n[b].double() / C_LIGHT
            rms_true.append(float((r - r.mean()).pow(2).mean().sqrt()))
            nhit.append(k)
            if k < 5:                                       # plane fit has 4 parameters: needs > 4 hits
                continue
            A = torch.cat([torch.ones(k, 1, dtype=torch.float64, device=dev), Xb], 1)
            sol = torch.linalg.lstsq(A.cpu(), tb[:, None].cpu()).solution
            rp = tb.cpu() - (A.cpu() @ sol).squeeze(1)
            rms_plane.append(float(rp.pow(2).mean().sqrt()))

rt, rp = np.array(rms_true), np.array(rms_plane)
q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (16, 50, 84)} | {"mean": float(a.mean())}
res = dict(src=os.environ.get("SRC") or "fm", layout=lay, n_showers=int(SA.N), n_used=len(rt), n_used_plane=len(rp),
           median_hits=float(np.median(nhit)) if nhit else float("nan"),
           rms_true_front_us=q(rt), rms_best_plane_us=q(rp))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
