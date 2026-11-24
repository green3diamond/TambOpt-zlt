"""How many detectors give the DeepSets reconstruction an arrival time without seeing a particle? (audit: thresholds far
below one particle). Same held-out showers, kernel pass and triggered selection as scripts/paper2/score_all.py
(noise-free kernel, expected counts mu at 2 m^2). Per layout and triggered shower (>= 3 detectors with mu >= 1 and a
time): n1 = timed detectors with mu >= 1, n_sub = timed detectors with mu < 1 (the reconstruction reads both), and
the share of n_sub with mu < 0.1. Also by log10 E decade.

    python scripts/diagnostics/subthreshold_timed_count.py <out.json> <n_events> name=layout.pt ...
Run under the run's guard (run 20: slurm/run20_guard.sh).
"""
import importlib.util, json, os, sys
os.environ["TAMBOPT_UTILITY"] = "swgo_trigger"            # BatchKernel then also computes mu
import numpy as np, torch
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, R)
spec = importlib.util.spec_from_file_location("s4", f"{R}/scripts/04_optimize_lbfgs_kernel.py")
S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
import modules.constants as C  # noqa: E402
from modules.data.dataset_builder import _load_corpus_metadata  # noqa: E402

out_path, n_ev, pairs = sys.argv[1], int(sys.argv[2]), [a.split("=", 1) for a in sys.argv[3:]]
dev = S4.DEVICE
mountain = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                               layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)


class HeldOut(S4.MainPool):                                   # as score_all.py
    def __init__(self):
        self.path = C.HELDOUT_SHOWER_CACHE_PATH
        self.meta = _load_corpus_metadata(mountain, self.path, 0)
        self.n, self.bw = self.meta.k_sp, None

    def clouds(self, s, a, b, with_frames=False):
        import showerdata
        m = self.meta
        pts = showerdata.load(self.path, start=s * m.per_sp + a, stop=s * m.per_sp + b).points
        c = torch.as_tensor(pts, dtype=torch.float32)
        fr = S4.place_clouds_shower(c, m.positions[a:b], m.dirs[a:b], event_ids=torch.arange(a, b), species=s)
        return (c, fr) if with_frames else c


pool = HeldOut()
clouds, bws, prim, frames = pool.load(0, min(n_ev, pool.n))
N = prim.shape[0]; CH = 32
le = (prim[:, 3] * (C.LOG_E_MAX - C.LOG_E_MIN) + C.LOG_E_MIN).numpy()
print(f"{N} held-out showers", flush=True)

res = {}
for name, path in pairs:
    L = torch.load(path, map_location="cpu", weights_only=False)
    x = torch.as_tensor(L["x"]).float().to(dev); y = torch.as_tensor(L["y"]).float().to(dev)
    n1, ns, nvs, trig = [], [], [], []
    with torch.no_grad():
        for lo in range(0, N, CH):
            hi = min(lo + CH, N)
            kern = S4.BatchKernel([c[lo:hi].to(dev) for c in clouds], [None] * len(clouds), surface,
                                  differentiable=False, frames=frames[lo:hi].to(dev))
            out = kern(prim[lo:hi].to(dev), torch.stack([x, y], -1)[None].expand(hi - lo, -1, -1))
            timed = out[..., 1] > 0
            m1 = (kern.mu >= 1) & timed
            sub = (kern.mu < 1) & timed
            n1.append(m1.sum(1).cpu()); ns.append(sub.sum(1).cpu()); nvs.append((sub & (kern.mu < 0.1)).sum(1).cpu())
            trig.append((m1.sum(1) >= 3).cpu())
    n1, ns, nvs, trig = (torch.cat(v).numpy() for v in (n1, ns, nvs, trig))
    t = trig.astype(bool)
    q = lambda v: [float(np.percentile(v, p)) for p in (10, 50, 90)] if len(v) else []
    r = dict(triggered=float(t.mean()), n1_p10_50_90=q(n1[t]), nsub_p10_50_90=q(ns[t]), n1_mean=float(n1[t].mean()),
             nsub_mean=float(ns[t].mean()), sub_share_of_timed=float(ns[t].sum() / max(1, (ns[t] + n1[t]).sum())),
             very_sub_share=float(nvs[t].sum() / max(1, ns[t].sum())),
             share_with_nsub_gt_n1=float((ns[t] > n1[t]).mean()), by_decade={})
    for d in range(int(np.floor(le.min())), int(np.ceil(le.max()))):
        k = t & (le >= d) & (le < d + 1)
        if k.sum() >= 20:
            r["by_decade"][f"1e{d}"] = dict(n=int(k.sum()), n1_median=float(np.median(n1[k])),
                                            nsub_median=float(np.median(ns[k])))
    res[name] = r
    print(f"[{name}] triggered {100 * r['triggered']:.1f}%; per triggered shower: mu>=1 timed {r['n1_mean']:.1f} "
          f"(p10/50/90 {r['n1_p10_50_90']}), mu<1 timed {r['nsub_mean']:.1f} (p10/50/90 {r['nsub_p10_50_90']}); "
          f"sub share of timed {100 * r['sub_share_of_timed']:.0f}%, of those mu<0.1 {100 * r['very_sub_share']:.0f}%; "
          f"showers with more sub than real {100 * r['share_with_nsub_gt_n1']:.0f}%", flush=True)
    for k, v in r["by_decade"].items():
        print(f"    {k}: n={v['n']} median mu>=1 {v['n1_median']:.0f}, mu<1 {v['nsub_median']:.0f}", flush=True)
json.dump(res, open(out_path, "w"), indent=1)
print("[write]", out_path)
