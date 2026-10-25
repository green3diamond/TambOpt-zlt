"""Stage 3c: fine-tune the run's recon on inputs with optional Poisson counting and cosmic background, on the fly.

Starts from --src (<RECON_FOLDER>_deepsets, untouched) and writes a new recon.pt to --out (default
<RECON_FOLDER>_deepsets_count with --counting poisson, else _deepsets_noise_kunits). Same rows, shower-level split and
blob filter as 03_train_recon_deepsets.py. Per batch: the per-species kernel labels of each row (E.pt / T.pt) ->
counted per species (--counting poisson, modules/optimize/counting.py) -> combined -> background in a window drawn from
--windows (modules/optimize/background.py; 0 = clean). Validation: a fixed --n-val slice with fixed noise (row i gets
window i mod len(windows)); the checkpoint keeps the best mean over the windows. Resumes from finetune_resume.pt.

    python scripts/03c_finetune_recon_noise.py [--windows 0 128 1000 10000] [--epochs 60] [--patience 8]
                                               [--counting poisson --eps-gamma 0.02]
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
from modules.optimize import load_models  # noqa: E402
from modules.optimize.background import window_centres, add_background, axis_distance_rows, BG_UNITS  # noqa: E402
from modules.optimize.counting import count_species, EPS_GAMMA  # noqa: E402
from modules.surrogates import combine_species_outputs  # noqa: E402
from modules.geometry import load_tr_mountain, SurfaceUpMap  # noqa: E402
from modules.data.dataset_builder import _load_corpus_metadata, shower_frames  # noqa: E402

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
ap.add_argument("--out", default=None)
ap.add_argument("--counting", choices=("none", "poisson"), default="none")
ap.add_argument("--eps-gamma", type=float, default=EPS_GAMMA, help="photon detection probability (counting only)")
ap.add_argument("--dropout", type=float, default=0.0,
                help="per row, a fraction f ~ U(0, dropout) of the detectors goes missing (E = T = 0), after background: "
                     "trains the recon to reconstruct with missing inputs; 0 = off")
ap.add_argument("--min-hit", type=int, default=0,
                help="train and validate only on rows with >= this many read-out detectors with signal (E' > 0, after "
                     "counting and background): the trigger; 0 = every row")
args = ap.parse_args()
if args.out is None:
    args.out = C.RECON_FOLDER + ("_deepsets_count" if args.counting == "poisson" else
                                 "_deepsets_dropout" if args.dropout > 0 else "_deepsets_noise_kunits")

dev = S3.DEVICE
D = C.TRAINING_DATASET_FOLDER
os.makedirs(args.out, exist_ok=True)
WS = torch.tensor(args.windows, dtype=torch.float32)
print(f"src {args.src}\nout {args.out}\nwindows {args.windows} ns  lr {args.lr}  wd {args.weight_decay}  "
      f"rows/epoch {args.rows_per_epoch}  counting {args.counting}"
      + (f" (eps_gamma {args.eps_gamma})" if args.counting == "poisson" else ""), flush=True)


t0 = time.time()
primary = torch.load(os.path.join(D, "primary.pt")).float()
xy = torch.load(os.path.join(D, "xy.pt")).float()                                # (N, nd, 2) = (East, North)
strat_ids = torch.load(os.path.join(D, "strategy_ids.pt")).long()
species_ids = torch.load(os.path.join(D, "species_ids.pt")).long()
keep = ~torch.load(os.path.join(D, "blob_ids.pt")).bool()
E_raw = torch.load(os.path.join(D, "E.pt")).float()                             # per-species rows, log1p E
T_raw = torch.log1p(torch.load(os.path.join(D, "T.pt")).float() * C.T_LOG_SCALE)  # log1p(t [us]), as S3
N_ROWS, n_sp, n_st = len(primary), int(species_ids.max()) + 1, int(strat_ids.max()) + 1
n_ev = N_ROWS // (n_sp * n_st)
if not (torch.equal(species_ids, torch.arange(n_sp).repeat_interleave(n_ev).repeat(n_st))
        and torch.equal(strat_ids, torch.arange(n_st).repeat_interleave(n_sp * n_ev))):
    raise RuntimeError("rows are not strategy-major, species-major, event order (see S3.load_kernel_responses)")
ROW_BASE = strat_ids * (n_sp * n_ev) + torch.arange(N_ROWS) % n_ev                # row of species 0, same (strat, event)
SP_OFF = torch.arange(n_sp) * n_ev
target = primary[:, :4].clone()
m = load_tr_mountain(C.GEOMETRY_PATH_RESOLVED, C.GEOMETRY_GROUP, C.DET_KEY, east_entry=C.EAST_ENTRY,
                     layer_east_dx=C.LAYER_EAST_DX, n_planes=C.N_PLANES)
surface = SurfaceUpMap.from_mountain(m, grid_h=256, grid_w=256).to(S3.DEVICE)
max_showers = 0 if C.DATASET_FRACTION >= 1.0 else int(C.DATASET_FRACTION * len(C.SPECIES_NAMES) * C.NUM_SHOWERS)
meta = _load_corpus_metadata(m, C.DUAL_SHOWER_CACHE_PATH, max_showers)
if meta.k_sp != n_ev:
    raise RuntimeError(f"corpus metadata keeps {meta.k_sp} events per species, the training rows have {n_ev}")
FRAMES = shower_frames(meta.positions[:n_ev], meta.dirs[:n_ev])                 # (n_ev, 12), event order
d_err = float((torch.nn.functional.normalize(meta.dirs[:n_ev], dim=1) - primary[:n_ev, :3]).abs().max())
print(f"[frames] {n_ev} events; max |dir(meta) - dir(primary.pt)| = {d_err:.2e}", flush=True)
if d_err > 1e-3:
    raise RuntimeError("corpus metadata and primary.pt disagree on the event directions: frames would be mis-paired")
print(f"[load] {time.time() - t0:.0f}s  rows {len(primary)}", flush=True)

train_idx, val_idx = S3.shower_level_split(strat_ids, species_ids, S3.VAL_FRAC, S3.SEED)
train_idx, val_idx = train_idx[keep[train_idx]], val_idx[keep[val_idx]]
g = torch.Generator().manual_seed(S3.SEED)
val_idx = val_idx[torch.randperm(len(val_idx), generator=g)[:args.n_val]]
print(f"[split] train {len(train_idx)}  val {len(val_idx)} (fixed slice)", flush=True)

def species_inputs(idx, xy_b, gen, mode):
    """Rows idx (CPU) -> combined (log1p E, log1p t) on the device, after counting (mode poisson) or not, and the
    rows' detector distances from the shower axis."""
    sib = ROW_BASE[idx][:, None] + SP_OFF                                          # (B, n_sp) same strat / event
    Es, Ts = E_raw[sib].to(dev, non_blocking=True), T_raw[sib].to(dev, non_blocking=True)
    r = axis_distance_rows(xy_b, surface, FRAMES[idx % n_ev].to(dev))
    preds = [torch.stack([Es[:, k], Ts[:, k]], -1) for k in range(n_sp)]
    preds, _ = count_species(preds, r, gen, mode=mode, eps_gamma=args.eps_gamma)
    comb = combine_species_outputs(*preds)
    return comb[..., 0], comb[..., 1], r


MODE = "poisson" if args.counting == "poisson" else "expected"
with torch.no_grad():
    i0 = train_idx[:200_000]
    _, T0, _ = species_inputs(i0, xy[i0].to(dev), None, "expected")
T_FALLBACK = float(torch.expm1(T0[T0 > 0]).median())
print(f"[t_fallback] {T_FALLBACK:.3f} us (median timed T)", flush=True)

ds = TensorDataset(torch.arange(len(primary)), xy, target)
tr_loader = DataLoader(Subset(ds, train_idx.tolist()), batch_size=S3.BATCH_SIZE, pin_memory=True,
                       sampler=RandomSampler(range(len(train_idx)), replacement=False, num_samples=args.rows_per_epoch))
va_loader = DataLoader(Subset(ds, val_idx.tolist()), batch_size=4096, shuffle=False, pin_memory=True)

_, recon = load_models(dev, recon_dir=args.src, with_fnn=False)
src_ckpt = torch.load(os.path.join(args.src, "recon.pt"), map_location="cpu", weights_only=False)
recon.requires_grad_(True)
opt = torch.optim.AdamW(recon.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def noisy(idx, xy_b, W, gen):
    """Rows -> recon inputs: per-species labels, counted (--counting poisson), combined, background in window W."""
    E_b, T_b, r = species_inputs(idx, xy_b, gen, MODE)
    E = torch.expm1(E_b.clamp(min=0)); t = torch.expm1(T_b.clamp(min=0)); timed = T_b > 0
    En, Tn, _, _ = add_background(E, t, timed, window_centres(t, timed, T_FALLBACK), W, gen, r_axis=r)
    if args.dropout > 0:                                                         # missing detectors
        f = torch.rand(En.shape[0], 1, device=En.device, generator=gen) * args.dropout
        alive = torch.rand(En.shape, device=En.device, generator=gen) >= f
        En, Tn = En * alive, Tn * alive
    return torch.stack([xy_b[..., 0], xy_b[..., 1], torch.log1p(En), torch.log1p(Tn)], -1)


def triggered(inp):
    """(B,) 1.0 for rows with >= --min-hit detectors with signal (E' > 0), all 1.0 with --min-hit 0."""
    if not args.min_hit:
        return torch.ones(inp.shape[0], device=inp.device)
    return ((inp[..., 2] > 0).sum(1) >= args.min_hit).float()


@torch.no_grad()
def validate():
    recon.eval()
    gen = torch.Generator(device=dev); gen.manual_seed(12345)
    sums = torch.zeros(len(WS)); cnt = torch.zeros(len(WS))
    for k, (idx, xy_b, tgt_b) in enumerate(va_loader):
        xy_b, tgt_b = (x.to(dev, non_blocking=True) for x in (xy_b, tgt_b))
        wi = torch.arange(len(idx), device=dev) % len(WS)
        inp = noisy(idx, xy_b, WS.to(dev)[wi], gen)
        l = S3._per_axis_loss(recon(inp), tgt_b, reduction="none")[0].detach()
        trig = triggered(inp)
        sums += torch.zeros(len(WS), device=dev).index_add_(0, wi, l * trig).cpu()
        cnt += torch.zeros(len(WS), device=dev).index_add_(0, wi, trig).cpu()
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
    for idx, xy_b, tgt_b in tr_loader:
        xy_b, tgt_b = (x.to(dev, non_blocking=True) for x in (xy_b, tgt_b))
        W = WS.to(dev)[torch.randint(len(WS), (len(idx),), device=dev, generator=gen)]
        with torch.no_grad():
            inp = noisy(idx, xy_b, W, gen)
        if args.min_hit:
            keep = triggered(inp).bool()
            if not keep.any():
                continue
            inp, tgt_b, idx = inp[keep], tgt_b[keep], idx[keep.cpu()]
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
                                      t_fallback_us=T_FALLBACK, counting=args.counting,
                                      eps_gamma=args.eps_gamma if args.counting == "poisson" else None,
                                      bg_units=BG_UNITS))
        torch.save(ck, os.path.join(args.out, "recon.pt.tmp")); os.replace(os.path.join(args.out, "recon.pt.tmp"),
                                                                            os.path.join(args.out, "recon.pt"))
        print(f"  [save] {args.out}/recon.pt", flush=True)
    torch.save(dict(state_dict=recon.state_dict(), optimizer=opt.state_dict(), log=log, best=best, best_ep=best_ep,
                    epoch=ep + 1), resume + ".tmp")
    os.replace(resume + ".tmp", resume)
    json.dump(dict(log=log, best_val_mean=best, best_epoch=best_ep, windows_ns=args.windows, src=args.src,
                   counting=args.counting, eps_gamma=args.eps_gamma if args.counting == "poisson" else None,
                   min_hit=args.min_hit, dropout=args.dropout),
              open(os.path.join(args.out, "finetune_log.json"), "w"), indent=1)
    if ep + 1 - best_ep >= args.patience:
        print(f"[stop] {args.patience} epochs without improvement; best mean val {best:.5f} at epoch {best_ep}")
        break
print(f"[done] best mean val {best:.5f} at epoch {best_ep} -> {args.out}/recon.pt")
