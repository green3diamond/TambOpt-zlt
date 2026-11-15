"""Fine-tune the run's DeepSets recon on CORSIKA 8 showers (Hamza's combined_<species>.h5 training set, disjoint from
the *_test.h5 showers used in the CORSIKA scoring).

Showers are placed like score_all.corsika_events: each at the decay vertex of the run's tau with the closest
direction (kept within 3 deg), own direction and energy. Per step: B showers, one layout drawn from the stage-1
strategy zoo (modules/layouts/strategies.py, random strategy), noise-free kernel (Stage 4's BatchKernel) -> recon
inputs (x, y, log E, log T) -> Stage 3's per-axis loss. Starts from <RECON_FOLDER>_deepsets; writes
<OUT>/recon.pt (best validation on 5 % held-out training showers, fixed layouts).

    python ft_c8.py <out_dir> [n_showers] [epochs]
Run under the run's guard (run 19: slurm/run19_guard.sh, run 20: slurm/run20_guard.sh).
"""
import importlib.util, json, os, sys, time
import numpy as np, torch, h5py
R = "/n/home05/zdimitrov/tambo/TambOpt"; sys.path.insert(0, R); sys.path.insert(0, R + "/scripts")
spec = importlib.util.spec_from_file_location("s4", f"{R}/scripts/04_optimize_lbfgs_kernel.py")
S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
spec3 = importlib.util.spec_from_file_location("s3", f"{R}/scripts/03_train_recon_deepsets.py")
S3 = importlib.util.module_from_spec(spec3); spec3.loader.exec_module(S3)
import modules.constants as C
from modules.optimize import load_models
from modules.surrogates import encode_primary
from modules.layouts.strategies import _STRATEGIES, _STRATEGY_FNS

OUT = sys.argv[1]; N_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
B, LR, PATIENCE, MAX_DEG = 32, S3.LR / 10, 4, 3.0
os.makedirs(OUT, exist_ok=True)
dev = S4.DEVICE
torch.manual_seed(0); rng = np.random.default_rng(0)
mountain = S4.load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                               layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = S4.SurfaceUpMap.from_mountain(mountain, grid_h=256, grid_w=256).to(dev)

# ── CORSIKA training showers, matched to the run's taus ──────────────────────
H = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3"
FILES = {"electron": "combined_electrons.h5", "muon": "combined_muons.h5", "photon": "combined_photons.h5"}
t0 = time.time()
with h5py.File(f"{H}/{FILES['electron']}", "r") as f:
    dirs = torch.as_tensor(f["directions"][:], dtype=torch.float32); en = torch.as_tensor(f["energies"][:, 0])
    pdg = torch.as_tensor(f["pdg"][:].astype(np.int64))
dirs = dirs / dirs.norm(dim=1, keepdim=True)
with h5py.File(C.TAU_WHOLESKY_PATH, "r") as f:
    tpos = np.asarray(f["position"][...]).T; tdir = np.asarray(f["direction"][...]).T; te = np.asarray(f["energy"][...])
k = (te >= 10 ** C.LOG_E_MIN) & (te <= 10 ** C.LOG_E_MAX)
tpos = torch.as_tensor(tpos[k], dtype=torch.float32)
tdir = torch.as_tensor(tdir[k], dtype=torch.float32).to(dev); tdir = tdir / tdir.norm(dim=1, keepdim=True)
d = dirs.to(dev); bc = torch.full((len(d),), -2.0, device=dev); bi = torch.zeros(len(d), dtype=torch.long, device=dev)
for lo in range(0, tdir.shape[0], 5_000):
    c, j = (d @ tdir[lo:lo + 5_000].T).max(1)
    u = c > bc; bc[u] = c[u]; bi[u] = j[u] + lo
ang = torch.rad2deg(torch.arccos(bc.clamp(-1, 1))).cpu()
keep = torch.nonzero(ang < MAX_DEG).squeeze(1)
print(f"{len(keep)} of {len(d)} training showers within {MAX_DEG} deg of a tau direction", flush=True)
keep = keep[torch.as_tensor(rng.permutation(len(keep))[:N_MAX])].sort().values
pos = tpos[bi.cpu()[keep]]
centre = torch.as_tensor(mountain.centroids_ENU, dtype=torch.float32).mean(0)
prim = encode_primary(dirs[keep], en[keep].float(), pdg[keep], pos, centre)
clouds, frames = [], None
for si, sp in enumerate(C.SPECIES_NAMES):
    with h5py.File(f"{H}/{FILES[sp]}", "r") as f:
        P = int(f["shape"][1]); ds = f["showers"]
        arr = np.zeros((len(keep), P, 5), np.float32)
        for r, i in enumerate(keep.tolist()):
            a = np.asarray(ds[i], dtype=np.float32).reshape(-1, 5); arr[r, :len(a)] = a
    c = torch.as_tensor(arr)
    frames = S4.place_clouds_shower(c, pos, dirs[keep], event_ids=keep, species=si)
    clouds.append(c)
    print(f"[load] {sp}: {tuple(c.shape)}  {time.time() - t0:.0f}s", flush=True)
N = len(keep); target = prim[:, :4].clone()
perm = torch.as_tensor(rng.permutation(N)); n_val = max(N // 20, 256)
val_idx, tr_idx = perm[:n_val].sort().values, perm[n_val:]
print(f"train {len(tr_idx)}  val {len(val_idx)}", flush=True)


def layout(r):
    """One layout from a random stage-1 strategy."""
    s = int(r.integers(len(_STRATEGIES))); _, fn, kw = _STRATEGIES[s]
    e, n = _STRATEGY_FNS[fn](mountain, n_det=C.N_DETECTORS, rng=r, **kw)
    return torch.stack([torch.as_tensor(e).float(), torch.as_tensor(n).float()], -1).to(dev)


def inputs(idx, xy):
    kern = S4.BatchKernel([c[idx].to(dev, non_blocking=True) for c in clouds], [None] * len(clouds), surface,
                          differentiable=False, frames=frames[idx].to(dev))
    xyb = xy[None].expand(len(idx), -1, -1)
    out = kern(prim[idx].to(dev), xyb)
    return torch.stack([xyb[..., 0], xyb[..., 1], out[..., 0], out[..., 1]], -1)


VAL_LAYOUTS = [layout(np.random.default_rng(1000 + i)) for i in range(8)]   # fixed validation layouts
_, recon = load_models(dev, recon_dir=C.RECON_FOLDER + "_deepsets", with_fnn=False)
src = torch.load(os.path.join(C.RECON_FOLDER + "_deepsets", "recon.pt"), map_location="cpu", weights_only=False)
recon.requires_grad_(True)
opt = torch.optim.AdamW(recon.parameters(), lr=LR, weight_decay=0.03)


@torch.no_grad()
def validate():
    recon.eval(); s, n = 0.0, 0
    for li, xy in enumerate(VAL_LAYOUTS):
        for lo in range(0, len(val_idx), B):
            idx = val_idx[lo:lo + B]
            l = S3._per_axis_loss(recon(inputs(idx, xy)), target[idx].to(dev))[0]
            s += float(l) * len(idx); n += len(idx)
    return s / n


log = [dict(epoch=0, val=validate())]; best, best_ep = log[0]["val"], 0
print(f"[epoch   0] val {best:.4f} (source recon)", flush=True)
for ep in range(1, EPOCHS + 1):
    t1 = time.time(); recon.train(); s, n = 0.0, 0
    order = tr_idx[torch.as_tensor(rng.permutation(len(tr_idx)))]
    for lo in range(0, len(order) - B + 1, B):
        idx = order[lo:lo + B].sort().values
        with torch.no_grad():
            inp = inputs(idx, layout(rng))
        loss = S3._per_axis_loss(recon(inp), target[idx].to(dev))[0]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(recon.parameters(), max_norm=S3.GRAD_CLIP); opt.step()
        s += loss.item() * B; n += B
    v = validate(); log.append(dict(epoch=ep, train=s / n, val=v, dt=time.time() - t1))
    print(f"[epoch {ep:3d}] train {s / n:.4f}  val {v:.4f}  {time.time() - t1:.0f}s", flush=True)
    if v < best - 1e-6:
        best, best_ep = v, ep
        torch.save(dict(src, state_dict=recon.state_dict(), epoch=ep, val_total=v,
                        c8_finetune=dict(n_showers=N, lr=LR, max_deg=MAX_DEG)), os.path.join(OUT, "recon.pt"))
        print(f"  [save] {OUT}/recon.pt", flush=True)
    json.dump(dict(log=log, best=best, best_epoch=best_ep, n_showers=N), open(os.path.join(OUT, "log.json"), "w"),
              indent=1)
    if ep - best_ep >= PATIENCE:
        print(f"[stop] best val {best:.4f} at epoch {best_ep}"); break
print("done", flush=True)
