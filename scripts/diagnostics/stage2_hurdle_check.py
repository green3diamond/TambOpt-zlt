"""Does a lit/dark split help run 14's (sigma=10) Stage 2 surrogate?

docs/sigma10-stage2-collapse.md, Measurement 4 notes: an MSE head trained on lit
and dark slots together predicts P(hit|x) * E[y|hit,x], so on slots that are lit
only some of the time its predictions are biased low whenever the detector IS
lit. A hurdle model — a lit/dark classifier plus a regressor trained on lit
slots only — predicts E[y|hit,x] directly. This script tests whether that helps.

Two models, identical architecture (DeepSets, run 14's config), recipe (Adam +
OneCycle, 02's LR schedule, batch, clip) and epoch budget, trained on the same
rows of one species:

  baseline : production mean-only head, z-scored MSE on every slot (02's loss)
  hurdle   : same trunk, decoder emits (logit_E, logit_T, mu_E, mu_T); loss =
             BCE(logit, lit) on every slot + MSE(mu, y) on lit slots only, with
             mu z-scored by lit-only stats

Scored on the SAME validation rows and in the SAME global-z units as
Measurement 4 (per-species stats over all slots, log-T T channel), together
with the production checkpoint. Because a dark slot's label is exactly 0 in
both channels' log space, the MSE-optimal hurdle prediction is p * mu; the
script also reports mu alone on lit slots and a hard 0.5 gate.

    python scripts/diagnostics/stage2_hurdle_check.py --epochs 15 --json out.json
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _V6)
sys.path.insert(0, _HERE)
from _pathfix import V6_ROOT  # noqa: F401,E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import modules  # noqa: F401,E402
from modules.constants import SPECIES_NAMES, T_LOG_SCALE, N_DETECTORS, RUN_LOCATION  # noqa: E402
from modules.surrogates import compute_normalization, DeepSetsSurrogate  # noqa: E402
from modules.surrogates.blocks import _mlp  # noqa: E402
import stage2_zero_inflation as zi  # noqa: E402  (split, constants, legacy ckpt loader)

_spec02 = importlib.util.spec_from_file_location(
    "train02", os.path.join(_V6, "scripts", "02_train_fnn_deepsets.py"))
_train02 = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(_train02)
T2 = _train02  # LR, LR_MAX, LR_MIN, ONECYCLE_PCT_START, GRAD_CLIP, BATCH_SIZE, DS_*, mse_mean_only

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── data: exactly Measurement 4's preparation ─────────────────────────────────

def load_species(run_dir, species):
    d1 = os.path.join(run_dir, zi.STAGE1_SUBDIR)
    ld = lambda n: torch.load(os.path.join(d1, n), map_location="cpu")  # noqa: E731
    primary, xy = ld("primary.pt").float(), ld("xy.pt").float()
    E, T = ld("E.pt").float(), ld("T.pt").float()
    strat, sp, blob = ld("strategy_ids.pt").long(), ld("species_ids.pt").long(), ld("blob_ids.pt").bool()

    idx = torch.nonzero(sp == SPECIES_NAMES.index(species)).squeeze(-1)
    primary, xy, E, T, strat, blob = (t[idx] for t in (primary, xy, E, T, strat, blob))

    keep = ~blob
    stats = compute_normalization(primary[keep], xy[keep], E[keep], T[keep])
    T = torch.log1p(T * T_LOG_SCALE)          # dark stays exactly 0; lit > 0
    n = T.shape[1]
    stats["out_mean"][n:] = float(T.mean())
    stats["out_std"][n:] = max(float(T.std()), 1e-6)

    tr, va = zi.shower_level_split(strat, zi.VAL_FRAC, zi.SEED)
    tr, va = tr[keep[tr]], va[keep[va]]
    return dict(primary=primary, xy=xy, E=E, T=T, stats=stats, train=tr, val=va)


# ── models ────────────────────────────────────────────────────────────────────

def make_model(stats, n_out):
    """02's DeepSets config; the hurdle variant only widens the last layer to 4."""
    m = DeepSetsSurrogate(n_det=N_DETECTORS, primary_dim=stats["in_mean"].numel() - 2 * N_DETECTORS,
                          hidden=T2.DS_HIDDEN, context=T2.DS_CONTEXT,
                          n_enc=T2.DS_N_ENC, n_dec=T2.DS_N_DEC, dropout=T2.DS_DROPOUT)
    if n_out != 2:
        m.decoder = _mlp(T2.DS_HIDDEN + T2.DS_CONTEXT, T2.DS_HIDDEN, n_out, T2.DS_N_DEC, T2.DS_DROPOUT)
    m.set_normalization(stats)
    return m.to(DEVICE)


def hit_stats(data):
    """Lit-only mean/std per channel, from training rows — the hurdle's regression scale."""
    E, T = data["E"][data["train"]], data["T"][data["train"]]
    e, t = E[E > 0], T[T > 0]
    return torch.tensor([e.mean(), t.mean()]), torch.tensor([e.std(), t.std()])


def hurdle_loss(out, E, T, h_mean, h_std):
    logit, mu_z = out[..., :2], out[..., 2:]
    y = torch.stack([E, T], dim=-1)
    lit = y > 0
    bce = F.binary_cross_entropy_with_logits(logit, lit.float(), reduction="none").mean(dim=(0, 1))
    tz = (y - h_mean) / h_std
    se = (mu_z - tz) ** 2 * lit
    mse = se.sum(dim=(0, 1)) / lit.sum(dim=(0, 1)).clamp(min=1)
    return 0.5 * bce.sum() + 0.5 * mse.sum(), bce.detach(), mse.detach()


# ── training (02's recipe, fixed epoch budget) ────────────────────────────────

def train(kind, data, epochs, seed, h_mean=None, h_std=None):
    torch.manual_seed(seed)
    stats = data["stats"]
    model = make_model(stats, 2 if kind == "baseline" else 4)
    out_mean, out_std = stats["out_mean"].to(DEVICE), stats["out_std"].to(DEVICE)
    if h_mean is not None:
        h_mean, h_std = h_mean.to(DEVICE), h_std.to(DEVICE)

    tr = data["train"]
    P, XY, E, T = (data[k][tr].to(DEVICE) for k in ("primary", "xy", "E", "T"))
    n, B = len(tr), T2.BATCH_SIZE
    steps = epochs * math.ceil(n / B)
    opt = torch.optim.Adam(model.parameters(), lr=T2.LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=T2.LR_MAX, total_steps=steps, pct_start=T2.ONECYCLE_PCT_START,
        div_factor=T2.LR_MAX / T2.LR, final_div_factor=T2.LR / T2.LR_MIN)
    print(f"\n[train:{kind}] rows={n:,} epochs={epochs} steps={steps:,}", flush=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for lo in range(0, n, B):
            b = perm[lo:lo + B]
            if kind == "baseline":
                loss, _, _ = T2.mse_mean_only(model(P[b], XY[b]), E[b], T[b], out_mean, out_std)
            else:
                loss, _, _ = hurdle_loss(model._forward_z(P[b], XY[b]), E[b], T[b], h_mean, h_std)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=T2.GRAD_CLIP)
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        print(f"  [{kind}] epoch {ep + 1:3d}/{epochs}  train loss {tot / n:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    del P, XY, E, T
    torch.cuda.empty_cache()
    return model


# ── evaluation in Measurement 4's global-z units ──────────────────────────────

class Acc:
    """Accumulates squared global-z error on lit / dark slots, per channel,
    plus the signed residual on lit slots (the shrinkage-bias test)."""

    def __init__(self):
        self.d = {c: dict(lit_se=0., lit_n=0., dark_se=0., dark_n=0., lit_res=0.) for c in "ET"}

    def add(self, pred_z, tgt_z, lit):
        for i, c in enumerate("ET"):
            se = (pred_z[..., i] - tgt_z[..., i]) ** 2
            l = lit[..., i]
            a = self.d[c]
            a["lit_se"] += float(se[l].sum()); a["lit_n"] += float(l.sum())
            a["dark_se"] += float(se[~l].sum()); a["dark_n"] += float((~l).sum())
            a["lit_res"] += float((tgt_z[..., i] - pred_z[..., i])[l].sum())

    def summary(self, lit_var):
        out = {}
        for c, a in self.d.items():
            mse_lit = a["lit_se"] / a["lit_n"]
            bias = a["lit_res"] / a["lit_n"]
            out[c] = dict(mse_lit=mse_lit, mse_dark=a["dark_se"] / a["dark_n"],
                          mse_all=(a["lit_se"] + a["dark_se"]) / (a["lit_n"] + a["dark_n"]),
                          bias_lit=bias, bias2_share=bias ** 2 / mse_lit if mse_lit else float("nan"),
                          var_explained_lit=1 - mse_lit / lit_var[c])
        out["total"] = 0.5 * (out["E"]["mse_all"] + out["T"]["mse_all"])
        return out


@torch.no_grad()
def evaluate(data, models, h_mean, h_std, batch=4096):
    stats = data["stats"]
    om, osd = stats["out_mean"].to(DEVICE), stats["out_std"].to(DEVICE)
    n = N_DETECTORS
    gm = torch.stack([om[0], om[n]]); gs = torch.stack([osd[0], osd[n]])     # per-channel global z
    hm, hs = h_mean.to(DEVICE), h_std.to(DEVICE)
    va = data["val"]

    names = ["production", "baseline", "hurdle_soft", "hurdle_cond", "hurdle_hard"]
    accs = {k: Acc() for k in names}
    cls = dict(tp=torch.zeros(2), fp=torch.zeros(2), fn=torch.zeros(2), n=torch.zeros(2), brier=torch.zeros(2))
    lit_sum = torch.zeros(2, dtype=torch.float64); lit_sq = torch.zeros(2, dtype=torch.float64); lit_n = torch.zeros(2)
    bins = torch.linspace(-1, 12, 27)
    binned = {k: dict(p=torch.zeros(26, dtype=torch.float64), y=torch.zeros(26, dtype=torch.float64),
                      n=torch.zeros(26, dtype=torch.float64)) for k in ("production", "baseline", "hurdle_soft")}

    for lo in range(0, len(va), batch):
        b = va[lo:lo + batch]
        P, XY = data["primary"][b].to(DEVICE), data["xy"][b].to(DEVICE)
        y = torch.stack([data["E"][b], data["T"][b]], dim=-1).to(DEVICE)       # raw log space
        lit = y > 0
        yz = (y - gm) / gs

        preds = {"production": models["production"](P, XY), "baseline": models["baseline"](P, XY)}
        out = models["hurdle"]._forward_z(P, XY)
        p = torch.sigmoid(out[..., :2])
        mu = out[..., 2:] * hs + hm
        preds["hurdle_soft"] = p * mu
        preds["hurdle_cond"] = mu
        preds["hurdle_hard"] = mu * (p > 0.5)
        for k, pr in preds.items():
            pz = (pr - gm) / gs
            accs[k].add(pz, yz, lit)
            if k in binned:                                   # E channel, lit slots only
                l = lit[..., 0]
                pe, ye = pz[..., 0][l], yz[..., 0][l]
                bi = (torch.bucketize(pe, bins.to(DEVICE)).clamp(1, 26) - 1).cpu()
                binned[k]["p"].index_add_(0, bi, pe.double().cpu())
                binned[k]["y"].index_add_(0, bi, ye.double().cpu())
                binned[k]["n"].index_add_(0, bi, torch.ones_like(bi, dtype=torch.float64))

        pred_lit = p > 0.5
        for i in range(2):
            cls["tp"][i] += float((pred_lit[..., i] & lit[..., i]).sum())
            cls["fp"][i] += float((pred_lit[..., i] & ~lit[..., i]).sum())
            cls["fn"][i] += float((~pred_lit[..., i] & lit[..., i]).sum())
            cls["n"][i] += lit[..., i].numel()
            cls["brier"][i] += float(((p[..., i] - lit[..., i].float()) ** 2).sum())
            lz = yz[..., i][lit[..., i]].double()
            lit_sum[i] += float(lz.sum()); lit_sq[i] += float((lz ** 2).sum()); lit_n[i] += lz.numel()

    lit_mean = lit_sum / lit_n
    lit_var = {c: float(lit_sq[i] / lit_n[i] - lit_mean[i] ** 2) for i, c in enumerate("ET")}
    res = {k: a.summary(lit_var) for k, a in accs.items()}
    res["lit_variance"] = lit_var
    res["classifier"] = {c: dict(precision=float(cls["tp"][i] / (cls["tp"][i] + cls["fp"][i]).clamp(min=1)),
                                 recall=float(cls["tp"][i] / (cls["tp"][i] + cls["fn"][i]).clamp(min=1)),
                                 brier=float(cls["brier"][i] / cls["n"][i]))
                         for i, c in enumerate("ET")}
    res["binned_E_lit"] = {k: [dict(pred=float(v["p"][j] / v["n"][j]), true=float(v["y"][j] / v["n"][j]), n=int(v["n"][j]))
                               for j in range(26) if v["n"][j] > 0] for k, v in binned.items()}
    return res


def report(res):
    print("\n" + "=" * 92)
    print("Validation, global-z units (Measurement 4's). 'bias' = mean(y - pred) on lit slots.")
    print("=" * 92)
    print(f"{'model':>13s} {'ch':>3s} {'MSE lit':>9s} {'MSE dark':>9s} {'MSE all':>9s} "
          f"{'bias lit':>9s} {'bias² share':>12s} {'var expl lit':>13s}")
    for k in ("production", "baseline", "hurdle_soft", "hurdle_cond", "hurdle_hard"):
        for c in "ET":
            r = res[k][c]
            print(f"{k:>13s} {c:>3s} {r['mse_lit']:9.4f} {r['mse_dark']:9.4f} {r['mse_all']:9.4f} "
                  f"{r['bias_lit']:9.4f} {100 * r['bias2_share']:11.1f}% {100 * r['var_explained_lit']:12.1f}%")
        print(f"{k:>13s} total = 0.5*(E+T) = {res[k]['total']:.4f}")
    print(f"\nlit-slot variance (global z): {res['lit_variance']}")
    print(f"classifier (0.5 threshold): {res['classifier']}")
    print("\nE, lit slots binned by prediction (global z): pred -> true")
    for k, rows in res["binned_E_lit"].items():
        print(f"  {k}: " + "  ".join(f"{r['pred']:.2f}->{r['true']:.2f}(n={r['n']})" for r in rows if r["n"] > 1000))


OUT_SUBDIR = "test_v6_run_02_hurdle_check"


def checkpoint(model, kind, species, stats, epochs, seed, results, h_mean=None, h_std=None):
    """Same keys a Stage-2 checkpoint carries (state_dict, norm_stats, config), plus
    the hurdle's lit-only regression stats. `config["n_out"]` = 4 marks the hurdle
    decoder (logit_E, logit_T, mu_E, mu_T); production loaders expect 2 and will
    refuse it, which is intended."""
    ck = dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
              norm_stats={k: v.cpu() for k, v in stats.items()},
              epoch=epochs,
              config=dict(model_type="deepsets", kind=kind, species=species,
                          n_det=N_DETECTORS, primary_dim=int(stats["in_mean"].numel() - 2 * N_DETECTORS),
                          hidden=T2.DS_HIDDEN, context=T2.DS_CONTEXT, n_enc=T2.DS_N_ENC,
                          n_dec=T2.DS_N_DEC, dropout=T2.DS_DROPOUT,
                          n_out=2 if kind == "baseline" else 4, seed=seed,
                          trained_by="scripts/diagnostics/stage2_hurdle_check.py"),
              val=results)
    if h_mean is not None:
        ck["hit_mean"], ck["hit_std"] = h_mean.cpu(), h_std.cpu()
    return ck


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="electron")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="subsample the training rows (smoke tests); val is always full")
    ap.add_argument("--json", default="", help="extra copy of the results JSON")
    ap.add_argument("--no-save", action="store_true", help="skip checkpoints (smoke tests)")
    ap.add_argument("--out-dir", default=os.path.join(RUN_LOCATION, OUT_SUBDIR))
    args = ap.parse_args()
    run_dir = RUN_LOCATION
    print(f"run={run_dir}  species={args.species}  device={DEVICE}")

    # Fail before hours of training, not after: never write into the production
    # Stage-2 folder, never overwrite an earlier run's files.
    out_paths = {k: os.path.join(args.out_dir, f"fnn_{args.species}_{k}.pt") for k in ("baseline", "hurdle")}
    out_paths["results"] = os.path.join(args.out_dir, f"results_{args.species}.json")
    if not args.no_save:
        prod = os.path.realpath(os.path.join(run_dir, zi.STAGE2_SUBDIR))
        if os.path.realpath(args.out_dir) == prod:
            sys.exit(f"refusing to write into the production Stage-2 folder {prod}")
        clash = [p for p in out_paths.values() if os.path.exists(p)]
        if clash:
            sys.exit(f"refusing to overwrite existing files: {clash}")
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[save] checkpoints -> {args.out_dir}")

    t0 = time.time()
    data = load_species(run_dir, args.species)
    if args.train_frac < 1.0:
        g = torch.Generator().manual_seed(args.seed)
        k = max(1, int(args.train_frac * len(data["train"])))
        data["train"] = data["train"][torch.randperm(len(data["train"]), generator=g)[:k]]
    print(f"[data] train={len(data['train']):,} val={len(data['val']):,} rows  ({time.time() - t0:.0f}s)")
    h_mean, h_std = hit_stats(data)
    print(f"[hit stats] lit-only mean={h_mean.tolist()} std={h_std.tolist()} (E log1p, log-T)")

    ckpt = torch.load(os.path.join(run_dir, zi.STAGE2_SUBDIR, f"fnn_{args.species}.pt"),
                      map_location=DEVICE, weights_only=False)
    # Normalization buffers come in with the state dict (they are registered buffers).
    production = zi._load_mean_rows(ckpt, data["primary"].shape[1], DEVICE)
    production.eval()

    models = dict(production=production,
                  baseline=train("baseline", data, args.epochs, args.seed),
                  hurdle=train("hurdle", data, args.epochs, args.seed, h_mean, h_std))
    res = evaluate(data, models, h_mean, h_std)
    res["config"] = dict(vars(args), run=run_dir, prod_epoch=ckpt.get("epoch"), prod_val_total=ckpt.get("val_total"))
    report(res)
    if not args.no_save:
        torch.save(checkpoint(models["baseline"], "baseline", args.species, data["stats"],
                              args.epochs, args.seed, res["baseline"]), out_paths["baseline"])
        torch.save(checkpoint(models["hurdle"], "hurdle", args.species, data["stats"],
                              args.epochs, args.seed,
                              {k: res[k] for k in ("hurdle_soft", "hurdle_cond", "hurdle_hard")},
                              h_mean, h_std), out_paths["hurdle"])
        with open(out_paths["results"], "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[save] wrote {out_paths['baseline']}\n       {out_paths['hurdle']}\n       {out_paths['results']}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    main()
