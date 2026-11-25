"""Analytical reconstruction on saved hits (score_all.py RECO=dump HITS_DIR=...), on CPU: same fits, batches and
metrics as score_all.py RECO=v1|v2|v3, without the kernel pass. TAMBOPT_ANA_MODEL=tdr|llhama picks the model
(modules/optimize/analytic_reco.py); llhama uses the per-shower local plane (mean terrain normal of the hit detectors).

    TAMBOPT_ANA_MODEL=llhama python scripts/paper2/ana_fit_hits.py <out.json> <v1|v2|v3> <hits.pt> ... [--batch 32]
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.optimize import analytic_reco as AR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("version", choices=list(AR.RUNS)); ap.add_argument("hits", nargs="+")
ap.add_argument("--batch", type=int, default=32)          # score_all.py's CH: L-BFGS shares its line search per batch
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for f in args.hits:
    name = os.path.basename(f)[:-3]
    if name in res:
        continue
    H = torch.load(f, map_location="cpu", weights_only=False)
    N = H["t"].shape[0]
    a_all = []
    for lo in range(0, N, args.batch):
        hi = min(lo + args.batch, N)
        X = H["X"][None].expand(hi - lo, -1, -1)
        hit = H["hit"][lo:hi]
        npl = AR.local_plane_normal(H["normals"][None].expand(hi - lo, -1, -1), hit) if AR.LL else H["n_plane"]
        rr = AR.RUNS[args.version](X, H["t"][lo:hi], H["n"][lo:hi], hit, npl)
        a = torch.rad2deg(torch.arccos((rr["dir"] * H["dtrue"][lo:hi]).sum(-1).clamp(-1, 1)))
        a_all.append(torch.where(rr["valid"], a, torch.full_like(a, 180.0)))
    a = torch.cat(a_all).numpy()
    r = dict(path=H["path"], n=int(N), model=AR.MODEL, reco=args.version, batch=args.batch, med=float(np.median(a)),
             det=float(H["det"].float().mean()), trig=float(H["trig"].mean()),
             **{f"w{t:g}": float(np.mean(a < t)) for t in (1, 2.5, 5)})
    for tag, m in (("cnt", H["cnt"]), ("det", H["det"])):
        m = m.numpy().astype(bool); am = a[m]
        r[f"n_{tag}"] = int(m.sum()); r[f"med_{tag}"] = float(np.median(am)) if len(am) else float("nan")
        r.update({f"w{t:g}_{tag}": float(np.mean(am < t)) if len(am) else float("nan") for t in (1, 2.5, 5)})
    res[name] = r
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[{name}] " + " ".join(f"{k}={v:.4g}" for k, v in r.items() if isinstance(v, float)), flush=True)
print("done", flush=True)
