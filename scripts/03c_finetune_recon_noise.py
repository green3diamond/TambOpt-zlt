"""Stage 3c: fine-tune the run's recon on inputs with cosmic-ray background added on the fly.

Starts from <RECON_FOLDER>_deepsets/recon.pt (left untouched) and writes a NEW checkpoint to
<RECON_FOLDER>_deepsets_noise/recon.pt (same format, so load_models(recon_dir=...) reads it).

Same corpus, shower-level split and blob filter as 03_train_recon_deepsets.py (kernel labels, --recon-input kernel).
Each training row gets its own window W drawn uniformly from --windows (default 0 / 128 ns / 1 us / 10 us; 0 keeps
clean rows in the mix), and background from modules/optimize/background.py: E' = E + Poisson(lambda), T' = min(T,
earliest background time), times uniform in W around the true front. Normalisation stats are kept from the clean
recon. Validation: a fixed 200k-row slice of the val split with fixed noise (row i gets window i mod len(windows)),
reported per window; the checkpoint keeps the best mean val over the windows.

    python scripts/03c_finetune_recon_noise.py [--windows 0 128 1000 10000] [--epochs 60] [--patience 8]
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.optimize import load_models  # noqa: E402
from modules.optimize.background import front_times, add_background  # noqa: E402

_s = importlib.util.spec_from_file_location("stage3", os.path.join(REPO, "scripts", "03_train_recon_deepsets.py"))
S3 = importlib.util.module_from_spec(_s); _s.loader.exec_module(S3)

ap = argparse.ArgumentParser()
ap.add_argument("--windows", type=float, nargs="+", default=[0, 128, 1000, 10000], help="background windows [ns]")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--patience", type=int, default=8, help="epochs with no mean-val improvement before stopping")
ap.add_argument("--rows-per-epoch", type=int, default=2_000_000, help="random training rows per epoch")
ap.add_argument("--n-val", type=int, default=200_000)
ap.add_argument("--lr", type=float, default=S3.LR)
ap.add_argument("--weight-decay", type=float, default=0.03)
ap.add_argument("--src", default=C.RECON_FOLDER + "_deepsets")
ap.add_argument("--out", default=C.RECON_FOLDER + "_deepsets_noise")
args = ap.parse_args()

dev = S3.DEVICE
D = C.TRAINING_DATASET_FOLDER
os.makedirs(args.out, exist_ok=True)
WS = torch.tensor(args.windows, dtype=torch.float32)
print(f"src {args.src}\nout {args.out}\nwindows {args.windows} ns  lr {args.lr}  wd {args.weight_decay}  "
      f"rows/epoch {args.rows_per_epoch}", flush=True)

m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(dev)

t0 = time.time()
primary = torch.load(os.path.join(D, "primary.pt")).float()
xy = torch.load(os.path.join(D, "xy.pt")).float()                                # (N, nd, 2) = (East, North)
strat_ids = torch.load(os.path.join(D, "strategy_ids.pt")).long()
species_ids = torch.load(os.path.join(D, "species_ids.pt")).long()
keep = ~torch.load(os.path.join(D, "blob_ids.pt")).bool()
E_k, T_k = S3.load_kernel_responses(D)                                           # log1p E, log1p(t [us])
target = primary[:, :4].clone()
print(f"[load] {time.time() - t0:.0f}s  rows {len(primary)}", flush=True)

train_idx, val_idx = S3.shower_level_split(strat_ids, species_ids, S3.VAL_FRAC, S3.SEED)
train_idx, val_idx = train_idx[keep[train_idx]], val_idx[keep[val_idx]]
g = torch.Generator().manual_seed(S3.SEED)
val_idx = val_idx[torch.randperm(len(val_idx), generator=g)[:args.n_val]]
print(f"[split] train {len(train_idx)}  val {len(val_idx)} (fixed slice)", flush=True)

timed_all = T_k > 0
T_FALLBACK = float(torch.expm1(T_k[train_idx[:200_000]][timed_all[train_idx[:200_000]]]).median())
print(f"[t_fallback] {T_FALLBACK:.3f} us (median timed T)", flush=True)

ds = TensorDataset(torch.arange(len(primary)), xy, E_k, T_k, target)
tr_loader = DataLoader(Subset(ds, train_idx.tolist()), batch_size=S3.BATCH_SIZE, pin_memory=True,
                       sampler=RandomSampler(range(len(train_idx)), replacement=False, num_samples=args.rows_per_epoch))
va_loader = DataLoader(Subset(ds, val_idx.tolist()), batch_size=4096, shuffle=False, pin_memory=True)

_, recon = load_models(dev, recon_dir=args.src, with_fnn=False)
src_ckpt = torch.load(os.path.join(args.src, "recon.pt"), map_location="cpu", weights_only=False)
recon.requires_grad_(True)
opt = torch.optim.AdamW(recon.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def noisy(idx, xy_b, E_b, T_b, tgt_b, W, gen):
    """Kernel inputs (log1p) -> inputs with background in window W (per row), as the recon reads them."""
    E = torch.expm1(E_b.clamp(min=0)); t = torch.expm1(T_b.clamp(min=0)); timed = T_b > 0
    B, nd = E.shape
    up = surface(xy_b[..., 1].reshape(-1), xy_b[..., 0].reshape(-1)).view(B, nd)
    P = torch.cat([xy_b, up[..., None]], -1)
    n = tgt_b[:, :3] / tgt_b[:, :3].norm(dim=1, keepdim=True).clamp(min=1e-12)   # propagation direction (ENU)
    En, Tn, _, _ = add_background(E, t, timed, front_times(t, timed, P, n, T_FALLBACK), W, gen)
    return torch.stack([xy_b[..., 0], xy_b[..., 1], torch.log1p(En), torch.log1p(Tn)], -1)


@torch.no_grad()
def validate():
    recon.eval()
    gen = torch.Generator(device=dev); gen.manual_seed(12345)
    sums = torch.zeros(len(WS)); cnt = torch.zeros(len(WS))
    for k, (idx, xy_b, E_b, T_b, tgt_b) in enumerate(va_loader):
        xy_b, E_b, T_b, tgt_b = (x.to(dev, non_blocking=True) for x in (xy_b, E_b, T_b, tgt_b))
        wi = torch.arange(len(idx), device=dev) % len(WS)
        inp = noisy(idx, xy_b, E_b, T_b, tgt_b, WS.to(dev)[wi], gen)
        l = S3._per_axis_loss(recon(inp), tgt_b, reduction="none")[0].detach()
        sums += torch.zeros(len(WS), device=dev).index_add_(0, wi, l).cpu()
        cnt += torch.bincount(wi, minlength=len(WS)).cpu().float()
    return (sums / cnt).tolist()


resume = os.path.join(args.out, "finetune_resume.pt")
log, best, best_ep, start = [], float("inf"), 0, 0
if os.path.exists(resume):
    r = torch.load(resume, map_location=dev, weights_only=False)
    recon.load_state_dict(r["state_dict"]); opt.load_state_dict(r["optimizer"])
    log, best, best_ep, start = r["log"], r["best"], r["best_ep"], r["epoch"]
    print(f"[resume] epoch {start}  best mean val {best:.5f} (epoch {best_ep})", flush=True)
else:
    v0 = validate()
    log.append(dict(epoch=0, train=None, val=v0, val_mean=sum(v0) / len(v0)))
    print(f"[epoch   0] clean recon  val per window {['%.4f' % v for v in v0]}  mean {log[0]['val_mean']:.4f}", flush=True)

gen = torch.Generator(device=dev); gen.manual_seed(1000 + start)
for ep in range(start, args.epochs):
    t0 = time.time(); recon.train(); s, n = 0.0, 0
    for idx, xy_b, E_b, T_b, tgt_b in tr_loader:
        xy_b, E_b, T_b, tgt_b = (x.to(dev, non_blocking=True) for x in (xy_b, E_b, T_b, tgt_b))
        W = WS.to(dev)[torch.randint(len(WS), (len(idx),), device=dev, generator=gen)]
        with torch.no_grad():
            inp = noisy(idx, xy_b, E_b, T_b, tgt_b, W, gen)
        loss = S3._per_axis_loss(recon(inp), tgt_b)[0]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(recon.parameters(), max_norm=S3.GRAD_CLIP); opt.step()
        s += loss.item() * len(idx); n += len(idx)
    v = validate(); vm = sum(v) / len(v)
    log.append(dict(epoch=ep + 1, train=s / n, val=v, val_mean=vm, dt=time.time() - t0))
    print(f"[epoch {ep + 1:3d}] train {s / n:.4f}  val per window {['%.4f' % x for x in v]}  mean {vm:.4f}  "
          f"{time.time() - t0:.0f}s", flush=True)
    if vm < best - 1e-6:
        best, best_ep = vm, ep + 1
        ck = dict(src_ckpt, state_dict=recon.state_dict(), epoch=best_ep, val_total=vm,
                  noise_finetune=dict(src=args.src, windows_ns=args.windows, val_per_window=v, lr=args.lr,
                                      weight_decay=args.weight_decay, rows_per_epoch=args.rows_per_epoch,
                                      t_fallback_us=T_FALLBACK))
        torch.save(ck, os.path.join(args.out, "recon.pt.tmp")); os.replace(os.path.join(args.out, "recon.pt.tmp"),
                                                                            os.path.join(args.out, "recon.pt"))
        print(f"  [save] {args.out}/recon.pt", flush=True)
    torch.save(dict(state_dict=recon.state_dict(), optimizer=opt.state_dict(), log=log, best=best, best_ep=best_ep,
                    epoch=ep + 1), resume + ".tmp")
    os.replace(resume + ".tmp", resume)
    json.dump(dict(log=log, best_val_mean=best, best_epoch=best_ep, windows_ns=args.windows, src=args.src),
              open(os.path.join(args.out, "finetune_log.json"), "w"), indent=1)
    if ep + 1 - best_ep >= args.patience:
        print(f"[stop] {args.patience} epochs without improvement; best mean val {best:.5f} at epoch {best_ep}")
        break
print(f"[done] best mean val {best:.5f} at epoch {best_ep} -> {args.out}/recon.pt")
