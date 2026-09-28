"""Run 04_optimize_lbfgs_ensemble.py with an acceptance term added to U.

Tests the SWGOLO mechanism: resolution terms (reward packing) against an
acceptance term (reward lighting enough detectors on more events). Production
leaves U_PR out of the composite; this adds

    U' = U + lam * mean(r)

where r is the per-event reconstructability gate. mean(r) is used instead of
U_PR = sqrt(sum r) so the term's strength does not change with the batch size
(5k Adam / 6k L-BFGS). upr_saturation.py measured lam ~ 200 as gradient parity
with the theta/phi/E terms on the grid layout.

The production script runs unmodified: everything is patched in before it is
executed. Also slices the run-14 surrogates' old (mu, logvar) head to its mean
rows, which this branch's mean-only DeepSets needs to load them.

Usage:
    python scripts/diagnostics/ensemble_with_upr.py --lam 200 -- \
        --chains 1 --schemes grid --opt_suffix _upr200
"""
import os
import runpy
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import modules  # noqa: E402,F401
import modules.optimize as _opt  # noqa: E402
import modules.optimize.objective as _obj  # noqa: E402
import modules.surrogates.dual as _dual  # noqa: E402

argv = sys.argv[1:]
lam = 0.0
if "--lam" in argv:
    i = argv.index("--lam")
    lam = float(argv[i + 1])
    del argv[i:i + 2]
if argv and argv[0] == "--":
    argv = argv[1:]

_build = _dual.build_surrogate_from_ckpt


def _build_mean_only(ckpt, *a, **kw):
    # Only when the model can't take the checkpoint as-is: NLL-head code (e.g.
    # branch run12-nll-sigma50) loads 4-output checkpoints unchanged.
    try:
        return _build(ckpt, *a, **kw)
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise
    sd = ckpt["state_dict"]
    dec = sorted({int(k.split(".")[1]) for k in sd if k.startswith("decoder.")})
    if dec:
        w, b = f"decoder.{dec[-1]}.weight", f"decoder.{dec[-1]}.bias"
        if sd[w].shape[0] == 4:
            sd = dict(sd)
            sd[w], sd[b] = sd[w][:2], sd[b][:2]
            ckpt = dict(ckpt, state_dict=sd)
    return _build(ckpt, *a, **kw)


_dual.build_surrogate_from_ckpt = _build_mean_only

_utility = _obj.utility_of_xy


def utility_with_acceptance(*a, **kw):
    U, r, parts = _utility(*a, **kw)
    acc = lam * r.mean()
    parts["u_acc"] = acc
    return U + acc, r, parts


_obj.utility_of_xy = utility_with_acceptance
_opt.utility_of_xy = utility_with_acceptance

print(f"[ensemble_with_upr] lam={lam}  (U' = U + lam * mean(r))  argv={argv}", flush=True)
sys.argv = [os.path.join(_REPO, "scripts", "04_optimize_lbfgs_ensemble.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")
