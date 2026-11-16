"""Dump of the mean time model's arrival-time delay and noise at the detectors (TAMBOPT_TIME_MODEL=mean), uniform
grid on Malata, CORSIKA 8 pool showers (C8Pool seed 4200, as the optimization runs). For every timed detector of every
shower: distance from the axis r [m], expected particles mu, the noise-free delay behind the plane front (kernel-mean
lag) [ns], the noise sd sqrt(var / mu + jitter^2) [ns] and one noisy draw [ns]. Read by plot_timing.py (cleanup).

    python time_noise_dump.py <out.npz> [n_showers] [chunk]
"""
import importlib.util, os, sys
import numpy as np, torch
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
spec = importlib.util.spec_from_file_location("s4", f"{R}/scripts/04_optimize_lbfgs_kernel.py")
S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
import modules.constants as C
import modules.optimize.timing as TIMING
from modules.data.dataset_builder import detectors_in_frames

OUT = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024; CH = int(sys.argv[3]) if len(sys.argv) > 3 else 64
assert TIMING.MODEL == "mean", "run with TAMBOPT_TIME_MODEL=mean"
dev = S4.DEVICE
m = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                        layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)
pool = S4.C8Pool(m, N, 4200)
g = torch.load(f"{C.RUN_LOCATION}/diagnostics/paper1/ugrid.pt", map_location="cpu")
e0, n0 = g["x"].float().to(dev), g["y"].float().to(dev)

CAP = {}
_arr = TIMING.arrival
def _cap(moms, mus, timed):
    CAP.update(moms=moms, mus=mus, timed=timed)
    return _arr(moms, mus, timed)
TIMING.arrival = _cap

rows = {k: [] for k in ("shower", "r", "mu", "lag", "sd", "noisy")}
TIMING.reseed(1)
gz = torch.Generator(device=dev).manual_seed(7)
with torch.no_grad():
    for lo in range(0, pool.n, CH):
        clouds, bws, prim, frames = pool.load(lo, min(lo + CH, pool.n))
        f_ = frames.to(dev)
        k = S4.BatchKernel([c.to(dev) for c in clouds], [w.to(dev) if w is not None else None for w in bws], surface,
                           differentiable=False, frames=f_)
        k(prim.to(dev), torch.stack([e0, n0], -1).unsqueeze(0))
        mean, var, mu = TIMING.mix_mean(CAP["moms"], CAP["mus"])
        sd = torch.sqrt(var / mu.clamp(min=1.0) + TIMING.JITTER_NS ** 2)
        noisy = mean + sd * torch.randn(mean.shape, generator=gz, device=dev)
        r = detectors_in_frames(e0, n0, surface, f_)[0]
        t = CAP["timed"]
        b, d = torch.nonzero(t, as_tuple=True)
        for key, v in (("shower", b + lo), ("r", r[b, d]), ("mu", mu[b, d]), ("lag", mean[b, d]), ("sd", sd[b, d]),
                       ("noisy", noisy[b, d])):
            rows[key].append(v.float().cpu().numpy())
        print(f"{min(lo + CH, pool.n)}/{pool.n} showers, {sum(len(x) for x in rows['r'])} timed detectors", flush=True)
np.savez(OUT, **{k: np.concatenate(v) for k, v in rows.items()}, jitter_ns=TIMING.JITTER_NS)
print("wrote", OUT)
