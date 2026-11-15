"""Primary encoding and normalization statistics, shared by every stage.

The surrogate itself is `deepsets.DeepSetsSurrogate` (the flat-MLP FNNSurrogate was removed 2026-11-02).
"""

import os

import torch

from ..constants import LOG_E_MIN, LOG_E_MAX

# ── Primary encoding ─────────────────────────────────────────────────────────

def encode_primary(directions:   torch.Tensor,
                   energies:     torch.Tensor,
                   pdg:          torch.Tensor,
                   positions:    torch.Tensor,
                   array_center: torch.Tensor) -> torch.Tensor:
    """Raw primary encoding: kinematics + decay vertex.

    Returns (N, PRIMARY_DIM):
    ``[dir_x, dir_y, dir_z, log_e_norm, pdg, rel_E, rel_N, rel_U]``, with
    log_e_norm = (log10(E) - LOG_E_MIN) / (LOG_E_MAX - LOG_E_MIN) ∈ [0, 1] and
    rel_* the decay vertex relative to `array_center` in metres (left unscaled —
    `compute_normalization` z-scores every primary column).

    The vertex is included because tau_wholesky.jl aims every surviving tau at the
    array, so direction alone barely discriminates — without it, two taus with the
    same direction and energy give different labels from identical input
    (aleatoric floor R² 0.49 without, 0.56 with).

    Cols 0-3 keep their meaning, so Step 3's ``primary[:, :4]`` target is unaffected.

    Args:
        directions   : (N, 3) unit vectors (sin θ cos φ, sin θ sin φ, cos θ).
        energies     : (N,) or (N, 1) primary energies [GeV], range ~[1e5, 1e8].
        pdg          : (N,) EM/hadronic primary class ids (0 or 1) — NOT the e/µ species.
        positions    : (N, 3) ENU decay vertices, from the Step-0 `_positions.pt` sidecar.
        array_center : (3,) ENU centre of the detector region. Passed in, not global,
                       so the encoding cannot go stale when the mesh changes.
    """
    dirs = torch.as_tensor(directions, dtype=torch.float32)
    eng  = torch.as_tensor(energies,   dtype=torch.float32).reshape(-1, 1)
    pdg  = torch.as_tensor(pdg,        dtype=torch.float32).reshape(-1, 1)
    pos  = torch.as_tensor(positions,  dtype=torch.float32)
    ctr  = torch.as_tensor(array_center, dtype=torch.float32).reshape(1, 3)
    log_e = torch.log10(eng)
    log_e_norm = (log_e - LOG_E_MIN) / (LOG_E_MAX - LOG_E_MIN)

    rel = pos - ctr                                       # decay → array-centre frame
    return torch.cat([dirs, log_e_norm, pdg, rel], dim=1)



def _species_sidecar_path(shower_cache_path: str) -> str:
    """Path of the Step-0 e/µ species sidecar paired with a dual corpus .pt
    (`…_dual_<2N>.pt` -> `…_dual_<2N>_species.pt`). Written by
    00_generate_data_dual_species.py; row-aligned with the corpus."""
    base, ext = os.path.splitext(shower_cache_path)
    return base + "_species" + ext


def _load_species_sidecar(shower_cache_path: str, keep_idx) -> torch.Tensor:
    """Load the Step-0 e/µ species sidecar (0=electron, 1=muon) and index it by
    `keep_idx` (the same rows used for the corpus metadata). Raises if missing —
    regenerate the corpus with the updated Step 0 (which writes the sidecar)."""
    path = _species_sidecar_path(shower_cache_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"species sidecar not found: {path}\n"
            f"Regenerate the corpus with the updated 00_generate_data_dual_species.py "
            f"(it writes the e/µ species sidecar; the corpus `pdg` is now the EM/hadronic class).")
    return torch.load(path)[torch.as_tensor(keep_idx)].long()


def compute_normalization(primary:   torch.Tensor,
                          xy:        torch.Tensor,
                          E:         torch.Tensor,
                          T:         torch.Tensor) -> dict:
    """Per-feature z-score statistics for inputs and outputs.

    Input vector layout (for the FNN) is `[primary (PRIMARY_DIM), xy_flat (2*n_det)]`.
    Output vector layout is `[E (100), T (100)]` = 200 (first E then T).

    For the input, primary features keep per-feature stats, but all xy
    slots share one (mean_x, std_x, mean_y, std_y) pair — otherwise
    permutation augmentation creates a train/val z-score mismatch.
    Likewise the output uses one shared (mean, std) for all E slots and
    one for all T slots.
    """
    n_det = E.shape[1]

    # ── Input: primary per-feature, xy shared across detectors ──────────
    p_mean = primary.mean(dim=0)                                    # (PRIMARY_DIM,)
    p_std  = primary.std(dim=0).clamp(min=1e-10)                     # (PRIMARY_DIM,)

    # One scalar mean/std for x coords, one for y coords
    xy_x = xy[..., 0]                                               # (B, n_det)
    xy_y = xy[..., 1]                                               # (B, n_det)
    x_mean = xy_x.mean();  x_std = xy_x.std()
    y_mean = xy_y.mean();  y_std = xy_y.std()

    # Broadcast to the full (205,) layout: [primary(5), x0,y0,x1,y1,...,x99,y99]
    xy_mean = torch.stack([x_mean.expand(n_det),
                           y_mean.expand(n_det)], dim=-1).reshape(-1)   # (200,)
    xy_std  = torch.stack([x_std.expand(n_det),
                           y_std.expand(n_det)], dim=-1).reshape(-1)    # (200,)

    in_mean = torch.cat([p_mean, xy_mean])                          # (205,)
    in_std  = torch.cat([p_std,  xy_std])                           # (205,)

    # ── Output: one shared stat for all E slots, one for all T slots ────
    E_mean = E.mean();  E_std = E.std()
    T_mean = T.mean();  T_std = T.std()

    out_mean = torch.cat([E_mean.expand(n_det),
                          T_mean.expand(n_det)])                    # (200,)
    out_std  = torch.cat([E_std.expand(n_det),
                          T_std.expand(n_det)])                     # (200,)

    return dict(
        in_mean=in_mean,   in_std=in_std,
        out_mean=out_mean, out_std=out_std,
    )
