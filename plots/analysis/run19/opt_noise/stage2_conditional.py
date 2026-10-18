"""Run 19 Stage 2 (hurdle surrogate) diagnostics on the validation showers (SPLIT_SEED = 1, as trained), conditional
plots of what the noise-aware optimiser actually reads from the surrogate:
    stage2_conditional_<species>.png   rows: value head mu | lit target;  thresholded readout (P(lit) >= 0.5 -> mu,
                                       else 0) | lit target;  P(lit) calibration (predicted vs actual lit fraction)
    stage2_conditional_event.png       the species-combined thresholded readout (forward_thresholded) vs the
                                       species-combined kernel label (what the recon gets), lit targets
    *_hit6.png                         the same on (event, layout) samples whose combined kernel label has >= 6
                                       detectors with E > 1
Columns: log1p(E) and log1p(T [us]). Writes stage2_diag.json (lit/dark confusion per species and channel).
"""
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = "/n/home05/zdimitrov/tambo/TambOpt"
sys.path.insert(0, REPO)
import modules.constants as C  # noqa: E402
from modules.surrogates import load_dual_surrogate, combine_species_outputs  # noqa: E402

def _mod(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
TVP = _mod("tvp", os.path.join(REPO, "plots", "training", "02_nn_target_vs_pred.py"))
S3 = _mod("s3", os.path.join(REPO, "scripts", "03_train_recon_deepsets.py"))

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D, OUT = C.TRAINING_DATASET_FOLDER, os.path.join(C.RUN_LOCATION, "diagnostics", "opt_noise")
N_EV, P_MIN = 150_000, 0.5                                                  # samples (events x layouts)
primary = torch.load(os.path.join(D, "primary.pt")).float()
xy = torch.load(os.path.join(D, "xy.pt")).float()
E = torch.load(os.path.join(D, "E.pt")).float()                                  # log1p(E), per species row
T = torch.log1p(torch.load(os.path.join(D, "T.pt")).float() * C.T_LOG_SCALE)       # log1p(t [us])
st = torch.load(os.path.join(D, "strategy_ids.pt")).long(); sp = torch.load(os.path.join(D, "species_ids.pt")).long()
keep = ~torch.load(os.path.join(D, "blob_ids.pt")).bool()
n_sp, n_st = int(sp.max()) + 1, int(st.max()) + 1
n_ev = len(sp) // (n_sp * n_st)
_, val_idx = S3.shower_level_split(st, sp, S3.VAL_FRAC, 1)                       # the recon's (= Stage 2's) val showers
# val rows of species 0, then the same (strategy, event) for every species
v0 = val_idx[sp[val_idx] == 0]
v0 = v0[torch.randperm(len(v0), generator=torch.Generator().manual_seed(0))[:N_EV]]
rows = torch.stack([v0 + s * n_ev for s in range(n_sp)])                           # (n_sp, N_EV)
ok = keep[rows].all(0); rows = rows[:, ok]
print(f"{rows.shape[1]} validation (event, layout) samples x {n_sp} species", flush=True)

sur = load_dual_surrogate(C.FNN_FOLDER, dev)
P, MU = [], []
with torch.no_grad():
    for s, m in enumerate(sur.models):
        ps, ms = [], []
        for lo in range(0, rows.shape[1], 2048):
            r = rows[s, lo:lo + 2048]
            p, mu = m.forward_parts(primary[r].to(dev), xy[r].to(dev))
            ps.append(p.cpu()); ms.append(mu.cpu())
        P.append(torch.cat(ps)); MU.append(torch.cat(ms))

# species-combined kernel label (what the recon was trained on) and the >= 6-hit selection on it (hit = E > 1)
kern = combine_species_outputs(*[torch.stack([E[rows[s]], T[rows[s]]], -1) for s in range(n_sp)])
HIT6 = (torch.expm1(kern[..., 0]) > 1).sum(1) >= 6                                # (N,) samples hitting >= 6 detectors
print(f">= 6 hits (E > 1): {int(HIT6.sum())} of {len(HIT6)} samples ({100*HIT6.float().mean():.1f}%)", flush=True)
n_tot, t_min = 0, None
for s in range(n_sp):
    p, mu = P[s], MU[s]
    n_tot = n_tot + torch.where(p[..., 0] >= P_MIN, torch.expm1(mu[..., 0]).clamp(min=0), torch.zeros_like(mu[..., 0]))
    ts = torch.where(p[..., 1] >= P_MIN, mu[..., 1].clamp(min=0), torch.full_like(mu[..., 1], float("inf")))
    t_min = ts if t_min is None else torch.minimum(t_min, ts)
sur_ev = torch.stack([torch.log1p(n_tot), torch.where(torch.isfinite(t_min), t_min, torch.zeros_like(t_min))], -1)


def figures(sel, tag, what):
    """Per-species and event figures on the samples `sel` (bool, (N,)); file suffix `tag`."""
    out = {}
    for s, name in enumerate(C.SPECIES_NAMES):
        tgt = torch.stack([E[rows[s]], T[rows[s]]], -1)[sel]                    # (n, nd, 2)
        p, mu = P[s][sel], MU[s][sel]
        thr = torch.where(p >= P_MIN, mu.clamp(min=0), torch.zeros_like(mu))
        fig, axs = plt.subplots(3, 2, figsize=(13, 16))
        out[name] = {}
        for c, ch in enumerate(("log1p(E)", "log1p(T·1e6)")):
            t_, lit = tgt[..., c].flatten().numpy(), (tgt[..., c] > 0).flatten().numpy()
            TVP._conditional_panel(axs[0, c], t_, mu[..., c].flatten().numpy(), f"value head μ | lit · {ch}")
            TVP._conditional_panel(axs[1, c], t_, thr[..., c].flatten().numpy(), f"readout (P ≥ {P_MIN} → μ, else 0) | lit · {ch}")
            pp = p[..., c].flatten().numpy()
            bins = np.linspace(0, 1, 21); idx = np.digitize(pp, bins) - 1
            frac = [lit[idx == i].mean() if (idx == i).sum() > 100 else np.nan for i in range(20)]
            ax = axs[2, c]
            ax.plot(0.5 * (bins[1:] + bins[:-1]), frac, "o-", color="#2a78d6"); ax.plot([0, 1], [0, 1], "r--", lw=1)
            hit_ok = float(((pp >= P_MIN) & lit).sum() / max(lit.sum(), 1))
            dark_ok = float(((pp < P_MIN) & ~lit).sum() / max((~lit).sum(), 1))
            out[name][ch] = dict(n_samples=int(sel.sum()), lit_frac=float(lit.mean()), recall_lit=hit_ok, recall_dark=dark_ok,
                                 false_lit=float(((pp >= P_MIN) & ~lit).sum() / max((pp >= P_MIN).sum(), 1)))
            ax.set(xlabel="predicted P(lit)", ylabel="actual lit fraction",
                   title=f"{ch}: lit {100*lit.mean():.1f}% · lit found {100*hit_ok:.1f}% · dark found {100*dark_ok:.1f}%")
            ax.grid(alpha=.3)
        fig.suptitle(f"Run 19 Stage 2, {name}: what the noise-aware optimiser reads ({what})", fontsize=13)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"stage2_conditional_{name}{tag}.png"), dpi=100, facecolor="white"); plt.close(fig)
        print(name, tag or "all", out[name], flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
    for c, ch in enumerate(("log1p(E)", "log1p(T·1e6)")):
        k_, s_ = kern[sel][..., c], sur_ev[sel][..., c]
        TVP._conditional_panel(axs[c], k_.flatten().numpy(), s_.flatten().numpy(), f"event {ch} | lit")
        lit, slit = (k_ > 0).flatten(), (s_ > 0).flatten()
        out.setdefault("event", {})[ch] = dict(n_samples=int(sel.sum()), lit_frac=float(lit.float().mean()),
                                               recall_lit=float((lit & slit).sum() / lit.sum()),
                                               recall_dark=float((~lit & ~slit).sum() / (~lit).sum()))
    fig.suptitle(f"Run 19 Stage 2, species combined: surrogate readout vs the kernel label ({what})", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"stage2_conditional_event{tag}.png"), dpi=100, facecolor="white"); plt.close(fig)
    print("event", tag or "all", out["event"], flush=True)
    return out


diag = dict(all=figures(torch.ones_like(HIT6), "", "all validation showers"),
            hit6=figures(HIT6, "_hit6", "showers hitting ≥ 6 detectors with E > 1"))
json.dump(diag, open(os.path.join(OUT, "stage2_diag.json"), "w"), indent=1)
print("[write] stage2_conditional_*.png (+ _hit6) stage2_diag.json")
