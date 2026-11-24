"""Localise a gradient mismatch of the analytical-recon utility (TAMBOPT_TIME_MODEL=mean/edge): for one detector and
axis, autograd vs central finite difference of scalar summaries at each stage of the chain, same showers and time
draws (timing.reseed) every evaluation:
  E_j, mu_j (kernel counts at the detector), t_plane_j, T_j (new arrival time, timed showers), lag_j = T_j - t_plane_j,
  ang (sum of the fitted-direction angle to truth over reconstructible showers), U.
The first stage that disagrees is where the gradient goes wrong.

    python grad_localize.py <out.json> <det> <axis E|N> [n_showers] [h]
"""
import importlib.util, json, os, sys
import torch
import torch.nn.functional as F
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
spec = importlib.util.spec_from_file_location("s4", f"{R}/scripts/04_optimize_lbfgs_kernel.py")
S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
import modules.constants as C
import modules.optimize.timing as TIMING
from modules.optimize.objective import utility_of_xy
from modules.optimize import analytic_reco as AR
from modules.optimize.analytic_diff import AnalyticRecon

OUT, J, AX = sys.argv[1], int(sys.argv[2]), sys.argv[3]
B = int(sys.argv[4]) if len(sys.argv) > 4 else 64; H = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
dev = S4.DEVICE
m = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                        layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
pool = S4.C8Pool(m, B, 4200)
clouds, _, prim, frames = pool.load(0, pool.n)
cl = [c.to(dev) for c in clouds]; p_ = prim.to(dev); f_ = frames.to(dev)
g = torch.load(f"{C.RUN_LOCATION}/diagnostics/paper1/ugrid.pt", map_location="cpu")
e0, n0 = g["x"].float().to(dev), g["y"].float().to(dev)
n_plane = AR.plane_normal(torch.as_tensor(m.centroids_ENU, dtype=torch.float32)).to(dev)
recon = AnalyticRecon(surface, n_plane, kappa=10.0)

CAP = {}
_kcall = S4.BatchKernel.__call__
def _k(self, pb, xy):
    o = _kcall(self, pb, xy)
    CAP["out"] = o; CAP["mu"] = self.mu; CAP["E"] = torch.expm1(o[..., 0].clamp(min=0))
    return o


import modules.data.dataset_builder as DB
_clb = DB.compute_labels_batch
def _clb_cap(*a, **kw):
    r = _clb(*a, **kw)
    if kw.get("moments") is not None:
        CAP.setdefault("moms", []).append(kw["moments"])
    return r
S4.compute_labels_batch = _clb_cap
S4.BatchKernel.__call__ = _k
_rcall = AnalyticRecon.__call__
def _r(self, feats):
    o = _rcall(self, feats); CAP["d"] = o[:, :3]; return o
AnalyticRecon.__call__ = _r


def stages(e, n):
    TIMING.reseed(123); CAP["moms"] = []
    k = S4.BatchKernel(cl, [None] * len(cl), surface, differentiable=True, frames=f_)
    recon.bind(k, p_)
    U, _, _ = utility_of_xy(e, n, p_, k, recon)
    T = torch.expm1(CAP["out"][..., 1].clamp(min=0))                   # us
    timed = (CAP["out"][..., 1] > 0).detach()
    from modules.data.dataset_builder import detectors_in_frames
    z = detectors_in_frames(e, n, surface, f_)[1]
    tpl = z * C.C8_PLANE_DX / 299.792458                                 # us (C8_PLANE_DX m / c)
    dt = F.normalize(p_[:, :3], dim=-1)
    cosang = (F.normalize(CAP["d"], dim=-1) * dt).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    ang = torch.rad2deg(torch.arccos(cosang))
    ok = (ang < 179.0).detach()
    tj = timed[:, J]
    sp = {}
    for si, mo in enumerate(CAP["moms"][:3]):
        sp[f"gw{si}_j"] = (mo["gw"][:, J] * tj).sum()
        sp[f"mean{si}_j"] = (mo["mean"][:, J] * tj).sum()
        sp[f"w{si}_j"] = (mo["w"][:, J] * tj).sum()
    return dict(**sp, E_j=CAP["E"][:, J].sum(), mu_j=CAP["mu"][:, J].sum(),
                tplane_j=(tpl[:, J] * tj).sum(), T_j=(T[:, J] * tj).sum(), lag_j=((T[:, J] - tpl[:, J]) * tj).sum() * 1e3,
                ang=(ang * ok).sum(), U=U), dict(timed=timed, ok=ok)


e = e0.clone().requires_grad_(True); nn_ = n0.clone().requires_grad_(True)
S, st0 = stages(e, nn_)
v = e if AX == "E" else nn_
auto = {}
for key, val in S.items():
    gr = torch.autograd.grad(val, v, retain_graph=True, allow_unused=True)[0]
    auto[key] = 0.0 if gr is None else float(gr[J])
d = torch.zeros_like(e0); d[J] = H
with torch.no_grad():
    Sp, stp = stages(e0 + d, n0) if AX == "E" else stages(e0, n0 + d)
    Sm, stm = stages(e0 - d, n0) if AX == "E" else stages(e0, n0 - d)
res = dict(det=J, axis=AX, h=H, flips=dict(timed=int((stp["timed"] != stm["timed"]).sum()),
                                           ok=int((stp["ok"] != stm["ok"]).sum())), stages={})
print(f"det {J} {AX}, h {H} m, flips {res['flips']}")
for key in S:
    fd = (float(Sp[key]) - float(Sm[key])) / (2 * H)
    res["stages"][key] = dict(value=float(S[key]), auto=auto[key], fd=fd)
    print(f"  {key:9s} value {float(S[key]):+.5g}  auto {auto[key]:+.5g}  fd {fd:+.5g}")
json.dump(res, open(OUT, "w"), indent=1)
