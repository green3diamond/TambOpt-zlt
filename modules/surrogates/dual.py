"""Dual-species surrogate: the e and mu models combined into one physical event.

The per-species files are the same simulated events split by secondary species,
so a physical event needs both models run on the same primary and layout.

Combination happens in PHYSICAL space, not elementwise, because both channels
are log-compressed (E = log1p(counts), T = log1p(T_phys * T_LOG_SCALE)):

    N_tot = N_e + N_mu                        counts add
    t_tot = (N_e*t_e + N_mu*t_mu) / N_tot     count-weighted, as the kernel defines T

then re-encoded to the same log channels. Keeps the single-surrogate contract
`fnn(primary, xy) -> (B, n_det, 2)`, and both branches stay in the autograd
graph so Step 4 backprops through both. See docs/THEORY.md §3.6 and §5.6.
"""

import os
import time

import torch
import torch.nn as nn

from ..constants import N_DETECTORS, PRIMARY_DIM, SPECIES_NAMES, T_LOG_SCALE
from .deepsets import build_surrogate_from_ckpt

def species_ckpt_name(species: str) -> str:
    """Checkpoint filename Step 2 writes for one species."""
    return f"fnn_{species}.pt"


ELECTRON_CKPT = species_ckpt_name("electron")   # kept: named in older run logs
MUON_CKPT     = species_ckpt_name("muon")


def combine_species_outputs(*preds: torch.Tensor) -> torch.Tensor:
    """Physically combine per-species (B, n_det, 2) predictions into one event.

    Counts are extensive so they add, and the kernel defines T as a
    count-weighted mean, so both generalise to any number of components:

        n_tot = Σ n_s              t_tot = Σ (n_s · t_s) / n_tot

    Differentiable everywhere; negative model outputs are clamped to zero
    counts / zero time before combining (a detector with no predicted signal
    contributes nothing, matching the kernel's behavior on empty clouds).

    Variadic, and exactly equivalent to the old two-argument form for two
    inputs — the sum is accumulated in the given order, so `(e, mu)` still
    produces bit-identical results.
    """
    if not preds:
        raise ValueError("combine_species_outputs needs at least one prediction")

    n_tot = None
    nt_sum = None
    for pred in preds:
        n_s = torch.expm1(pred[..., 0]).clamp(min=0.0)                  # counts
        t_s = torch.expm1(pred[..., 1]).clamp(min=0.0) / T_LOG_SCALE    # seconds
        n_tot = n_s if n_tot is None else n_tot + n_s
        nt_sum = n_s * t_s if nt_sum is None else nt_sum + n_s * t_s

    t_tot = nt_sum / n_tot.clamp(min=1e-12)
    E_out = torch.log1p(n_tot)
    T_out = torch.log1p(t_tot * T_LOG_SCALE)
    return torch.stack([E_out, T_out], dim=-1)


class MultiSpeciesSurrogate(nn.Module):
    """N frozen per-species surrogates behind the single-surrogate contract.

    forward(primary, xy) evaluates EVERY per-species model on the SAME primary
    (whose pdg feature is the real EM/hadronic class each was trained on) and
    combines their outputs physically — a primary describes one complete event,
    and every component is always part of it. Routing is by model identity, not
    by the pdg feature.

    Models are held in `constants.SPECIES_NAMES` order.
    """

    def __init__(self, models, names=SPECIES_NAMES):
        super().__init__()
        models = list(models)
        if len(models) != len(names):
            raise ValueError(f"got {len(models)} models for species {tuple(names)}")
        self.species = tuple(names)
        self.models = nn.ModuleList(models)
        self.n_det = getattr(models[0], "n_det", N_DETECTORS)

    def __getattr__(self, name):
        """`self.electron` / `self.muon` / … resolve to that species' model.

        nn.Module routes attribute lookup through its own _parameters/_modules
        dicts, so this only runs for names it did not find -- the per-species
        aliases older callers (and the paper-figure scripts) use by name.
        """
        try:
            species = super().__getattribute__("species")
            models = super().__getattribute__("_modules")["models"]
        except (AttributeError, KeyError):
            return super().__getattr__(name)
        if name in species:
            return models[species.index(name)]
        return super().__getattr__(name)

    def forward(self, primary: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            primary : (B, PRIMARY_DIM) — passed unchanged to every model.
            xy      : (B, n_det, 2) — shared layout, stays in the autograd graph
                      of ALL branches.
        Returns:
            (B, n_det, 2) combined event response — col 0 = log1p(N_tot),
            col 1 = log1p(t_tot * T_LOG_SCALE).
        """
        return combine_species_outputs(*(m(primary, xy) for m in self.models))

    def forward_with_var(self, primary: torch.Tensor, xy: torch.Tensor):
        """(mean, var) — mean is identical to forward(). var sums the per-species
        raw-unit variances (independent noise sources); the physical combination
        (count-weighted average, log1p) is nonlinear, so this is not a full
        delta-method propagation, just a reasonable per-detector uncertainty
        signal for recon/optimizer consumption.
        """
        mean = self.forward(primary, xy)
        var = None
        for m in self.models:
            v = m.forward_var(primary, xy)
            var = v if var is None else var + v
        return mean, var

    def forward_sample(self, primary: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """One stochastic draw from the predicted (mean, var) distribution,
        same (B, n_det, 2) contract as forward() — a fresh noisy realization
        each call instead of the mean point estimate, so downstream
        training/optimization sees the surrogate's learned aleatoric spread
        directly rather than being handed mean and variance as separate,
        discardable inputs. Reparameterized (mean + eps*std) so gradients
        into (primary, xy) still flow for stage-4's L-BFGS/Adam.
        """
        mean, var = self.forward_with_var(primary, xy)
        eps = torch.randn_like(mean)
        return mean + eps * var.clamp(min=0.0).sqrt()


# The pipeline ran on exactly two species for its first three generations, and
# the name is in run logs, notebooks and plots/layouts/true_utility.py.
DualSpeciesSurrogate = MultiSpeciesSurrogate


def _ckpt_provenance(path: str) -> str:
    """Absolute path + mtime of a checkpoint, for the load line.

    The load line names only the FILE, which is identical in every run world, so
    nothing in a log distinguishes one run's checkpoint from another's.
    """
    ap = os.path.abspath(path)
    try:
        st = os.stat(ap)
    except OSError as e:
        return f"{ap}  [UNREADABLE: {e.strerror}]"
    return (f"{ap}  mtime="
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")


def load_dual_surrogate(folder: str,
                        device: torch.device,
                        n_det: int = N_DETECTORS,
                        primary_dim: int = PRIMARY_DIM,
                        species=SPECIES_NAMES) -> MultiSpeciesSurrogate:
    """Load one `fnn_<species>.pt` per entry in `species` into a frozen wrapper.

    Each checkpoint is built via `build_surrogate_from_ckpt` (flat-MLP or
    DeepSets, chosen by its saved config), gets its own norm stats from the
    checkpoint, and is frozen in eval mode.

    Keeps its name across ~10 call sites even though it is no longer only dual.
    """
    models = []
    for name in species:
        fname = species_ckpt_name(name)
        path = os.path.join(folder, fname)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        models.append(build_surrogate_from_ckpt(ckpt, n_det, primary_dim, device))
        cfg = ckpt.get("config", {})
        print(f"[load] {fname}  model={cfg.get('model_type', 'fnn')}  "
              f"epoch={ckpt.get('epoch', '?')}  val={ckpt.get('val_total', '?')}\n"
              f"       {_ckpt_provenance(path)}", flush=True)
    dual = MultiSpeciesSurrogate(models, species).to(device)
    dual.eval()
    return dual
