"""Stage 4 with a component-balanced utility: equal θ/φ/E shares.

Runs `04_optimize_lbfgs_ensemble.py` unchanged (perturbed Adam chains, chunked
L-BFGS with scoring-set accept/rollback, aligned ensemble) but maximizes

    U_bal = Σ_k w_k · u_k + u_offmesh,      w_k = mean_j(ū_j) / ū_k

over the weighted parts u_theta / u_phi / u_e of `utility_of_xy`, where ū is a
detached EMA of those parts. At equilibrium every term contributes an equal
third and Σ_k w_k ū_k = Σ ū, so U_bal keeps the magnitude of raw U (GRAD_CLIP,
the off-mesh penalty and the accept tolerance stay calibrated). Because ū is
measured on the already-W-scaled parts, the W_THETA/W_PHI/W_E constants cancel.

The weights adapt every Adam step, then are frozen to one shared w* (from the
tails of all chains' Adam logs) for L-BFGS, scoring-set accept/rollback and the
ensemble ranking, so the line search sees a deterministic objective and scores
stay comparable across chunks and chains.

The ensemble script is loaded by path and its module globals patched rather
than copied. Artifacts land in ``<OPT_FOLDER>_lbfgs_balanced{--opt_suffix}_{scheme}/``
(same files as the ensemble, plus balance_weights.json / balance_weights.pt).

    python scripts/04_optimize_lbfgs_balanced.py --chains 6
"""
import argparse
import importlib.util
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers v6 root

import torch

_spec = importlib.util.spec_from_file_location(
    "_ens04", os.path.join(_HERE, "04_optimize_lbfgs_ensemble.py"))
_ens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ens)

from modules.constants import OPT_FOLDER

for _name in ("utility_of_xy", "adam_warm_start", "_run_one_scheme",
              "OPT_DIR_TEMPLATE", "ADAM_LOG_EVERY", "DEVICE", "main"):
    if not hasattr(_ens, _name):
        raise AttributeError(f"04_optimize_lbfgs_ensemble.py no longer defines "
                             f"{_name!r}; update the patches in {__file__}")


# ── Config ───────────────────────────────────────────────────────────────────
COMPONENTS          = ("u_theta", "u_phi", "u_e")
BALANCE_EMA         = 0.99     # ~100-step horizon over Adam steps
BALANCE_FLOOR       = 1e-3     # ū_k floor, relative to Σ|ū|
BALANCE_W_MAX       = 10.0     # weight cap (early steps have φ ≈ 1)
BALANCE_FREEZE_TAIL = 500      # Adam log entries per chain averaged into w*

_raw_utility_of_xy = _ens.utility_of_xy


def weights_from_means(u_bar: torch.Tensor) -> torch.Tensor:
    """Equal-share weights w_k = mean(ū) / ū_k, with ū floored and w capped."""
    floor = BALANCE_FLOOR * max(float(u_bar.abs().sum()), 1.0)
    u = u_bar.clamp_min(floor)
    return (u.mean() / u).clamp_max(BALANCE_W_MAX)


class Balancer:
    """Weight state shared by every patched utility call.

    "ema": each grad-enabled call with finite parts updates ū (Adam steps);
    no_grad calls read the current weights without updating.
    "frozen": fixed w* (L-BFGS closures, rescoring, scoring set)."""

    def __init__(self):
        self.mode = "ema"
        self.u_bar = None
        self.w_frozen = None
        self.history = []       # per chain: [step, w_theta, w_phi, w_e] rows
        self._step = 0

    def start_chain(self):
        self.mode, self.u_bar, self._step = "ema", None, 0
        self.history.append([])

    def freeze(self, w: torch.Tensor):
        self.mode, self.w_frozen = "frozen", w

    def weights(self, parts: dict) -> torch.Tensor:
        u = torch.stack([parts[k].detach() for k in COMPONENTS]).float()
        if self.mode == "frozen":
            return self.w_frozen.to(u.device)
        if torch.is_grad_enabled() and torch.isfinite(u).all():
            self.u_bar = u if self.u_bar is None else (
                BALANCE_EMA * self.u_bar + (1.0 - BALANCE_EMA) * u)
            self._step += 1
            if self.history and (self._step == 1
                                 or self._step % _ens.ADAM_LOG_EVERY == 0):
                self.history[-1].append(
                    [self._step, *weights_from_means(self.u_bar).tolist()])
        if self.u_bar is None:
            return torch.ones(len(COMPONENTS), device=u.device)
        return weights_from_means(self.u_bar)


BAL = Balancer()


def balanced_utility_of_xy(*args, **kwargs):
    """Drop-in for `utility_of_xy` returning (U_bal, r, parts).

    `parts` keeps the unweighted u_theta/u_phi/u_e (what the logs and
    utility_components.png show) and gains u_unbalanced and w_theta/w_phi/w_e."""
    U, r, parts = _raw_utility_of_xy(*args, **kwargs)
    w = BAL.weights(parts)
    U_bal = sum(w[i] * parts[k] for i, k in enumerate(COMPONENTS))
    if "u_offmesh" in parts:
        U_bal = U_bal + parts["u_offmesh"]
    parts["u_unbalanced"] = U
    for i, k in enumerate(COMPONENTS):
        parts["w_" + k[2:]] = w[i]
    return U_bal, r, parts


def frozen_weights(adam_logs):
    """w* from the mean over chains of each chain's last BALANCE_FREEZE_TAIL
    finite component log entries. Returns (w*, ū*, per-chain tail means)."""
    tails = []
    for log in adam_logs:
        rows = [[e[k] for k in COMPONENTS] for e in log
                if all(math.isfinite(e[k]) for k in COMPONENTS)]
        if rows:
            tails.append(torch.tensor(rows[-BALANCE_FREEZE_TAIL:]).mean(dim=0))
    if not tails:
        return torch.ones(len(COMPONENTS)), None, []
    u_star = torch.stack(tails).mean(dim=0)
    return weights_from_means(u_star), u_star, tails


_orig_adam_warm_start = _ens.adam_warm_start
_orig_run_one_scheme = _ens._run_one_scheme


def adam_warm_start(*args, **kwargs):
    BAL.start_chain()
    return _orig_adam_warm_start(*args, **kwargs)


def _run_one_scheme(scheme, mountain, fnn, recon, primary_all, n_total_primaries,
                    per_source):
    logs = [lg for _bests, chain_logs, *_rest in per_source.values() for lg in chain_logs]
    w_star, u_star, tails = frozen_weights(logs)
    BAL.freeze(w_star.to(_ens.DEVICE))
    fmt = lambda t: "  ".join(f"{k[2:]}={float(v):.3f}" for k, v in zip(COMPONENTS, t))
    print(f"[balance] scheme={scheme}  frozen w*: {fmt(w_star)}")
    if u_star is not None:
        print(f"[balance]   from ū* (tail {BALANCE_FREEZE_TAIL}, {len(tails)} chains): "
              f"{fmt(u_star)}  ->  contributions {fmt(w_star * u_star)}")

    result = _orig_run_one_scheme(scheme, mountain, fnn, recon, primary_all,
                                  n_total_primaries, per_source)

    opt_dir = _ens.OPT_DIR_TEMPLATE.format(scheme=scheme)
    summary = dict(
        components=list(COMPONENTS),
        w_star=w_star.tolist(),
        u_star=None if u_star is None else u_star.tolist(),
        per_chain_tail_means=[t.tolist() for t in tails],
        balance_ema=BALANCE_EMA, balance_floor=BALANCE_FLOOR,
        balance_w_max=BALANCE_W_MAX, balance_freeze_tail=BALANCE_FREEZE_TAIL,
    )
    with open(os.path.join(opt_dir, "balance_weights.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({**summary, "adam_weight_history": BAL.history},
               os.path.join(opt_dir, "balance_weights.pt"))
    BAL.history = []
    return result


_ens.utility_of_xy = balanced_utility_of_xy
_ens.adam_warm_start = adam_warm_start
_ens._run_one_scheme = _run_one_scheme
_ens.OPT_DIR_TEMPLATE = OPT_FOLDER + "_lbfgs_balanced_{scheme}"


def main():
    # The ensemble's own --opt_suffix handler would rewrite the template back
    # to _lbfgs_ensemble…, so consume it here and hide it from _ens.main().
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--opt_suffix", type=str, default="")
    known, rest = ap.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    _ens.OPT_DIR_TEMPLATE = OPT_FOLDER + "_lbfgs_balanced" + known.opt_suffix + "_{scheme}"

    print("=" * 72)
    print("v6/04_optimize_lbfgs_balanced.py — equal-share θ/φ/E utility")
    print(f"  w_k = mean(ū)/ū_k   EMA={BALANCE_EMA}  floor={BALANCE_FLOOR}  "
          f"w_max={BALANCE_W_MAX}  freeze tail={BALANCE_FREEZE_TAIL}")
    print(f"  output: {_ens.OPT_DIR_TEMPLATE}")
    _ens.main()


if __name__ == "__main__":
    main()
