"""Reconstruction net: per-detector response -> primary labels.

Maps flattened `(x, y, E_pred, T_pred)` per detector to the primary encoding
`[dir_x, dir_y, dir_z, log_e_norm]`. `Reconstruction` mirrors `FNNSurrogate`
one-for-one — same hidden widths, same baked-in input z-score and output
denormalization, same `set_normalization()` contract and buffer names.
`DeepSetsRecon` is the permutation-equivariant version Step 3 trains.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import N_DETECTORS
from .blocks import _mlp


class Reconstruction(nn.Module):
    """Flat MLP: (layout + predicted E/T) → primary encoding.

    Input layout (flattened row-major per sample, length `n_det * input_features`):
        [x_0, y_0, E_0, T_0, x_1, y_1, E_1, T_1, ...]
    Output (length `output_dim`, typically 4):
        [dir_x, dir_y, dir_z, log_e_norm]   (v6 primary encoding)

    Normalization buffers are initialized to identity (zeros/ones) and populated
    via `set_normalization(...)`. The FNN pattern is followed exactly: input is
    z-scored on the way in, output is de-z-scored on the way out, so the forward
    pass works in raw units from the caller's perspective.
    """

    def __init__(self,
                 n_det:          int = N_DETECTORS,
                 input_features: int = 4,
                 output_dim:     int = 4,
                 hidden:         int = 512,
                 dropout:        float = 0.1):
        super().__init__()
        self.n_det          = n_det
        self.input_features = input_features
        self.output_dim     = output_dim

        in_dim = n_det * input_features

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

        self.register_buffer("in_mean",  torch.zeros(in_dim))
        self.register_buffer("in_std",   torch.ones(in_dim))
        self.register_buffer("out_mean", torch.zeros(output_dim))
        self.register_buffer("out_std",  torch.ones(output_dim))

    def set_normalization(self,
                          in_mean:  torch.Tensor,
                          in_std:   torch.Tensor,
                          out_mean: torch.Tensor,
                          out_std:  torch.Tensor) -> None:
        """Copy z-score buffers. Shapes must match the registered buffers."""
        self.in_mean.copy_(in_mean)
        self.in_std.copy_(in_std)
        self.out_mean.copy_(out_mean)
        self.out_std.copy_(out_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: `x` of shape `(B, n_det * input_features)` in raw units.
        Returns: `(B, output_dim)` in raw primary-encoding units.
        """
        x_n   = (x - self.in_mean) / self.in_std
        out_n = self.net(x_n)
        return out_n * self.out_std + self.out_mean


RELTIME_FEATURES = 9


def reltime_features(x: torch.Tensor, ref_e_min: float = 0.0) -> torch.Tensor:
    """(B, n, 4) raw [x, y, log1p E, log1p(t * 1e6)] -> (B, n, 9) relative-time features.

        [x - x0, y - y0, log1p E, dt in ns, lit, timed, x0, y0, log1p(t0 in us)]

    lit   = E > 0: the detector saw any deposit.
    timed = T > 0: it has a leading-edge time. The kernel times a detector only when one point's deposit
            clears HIT_DEPOSIT_MIN (kernel.py), so most lit detectors have E > 0 but T = 0 (the sentinel).
    Reference (x0, y0, t0): the earliest timed detector; with none timed, the highest-E lit detector
    (t0 = 0); with nothing lit, the array's mean position (t0 = 0). dt = t - t0 for timed detectors,
    0 otherwise (`timed` tells "first hit" from "no time"). x0, y0, log1p(t0) are broadcast to every
    detector so nothing absolute is lost. The raw log-time input squashes the ~10-100 ns differences
    that set the direction into ~0.005 of a unit; dt keeps them linear. Differentiable a.e. in x, y.

    ref_e_min > 0 ("reltime1"): the reference is the earliest timed detector with >= ref_e_min counts, falling
    back to the rules above when there is none. With 0 the earliest timed detector is often a faint one
    (82% of run 18 training rows: E < 0.1 count, timed by a single deposit), a noisy anchor for every dt.
    """
    pos, E, T = x[..., :2], x[..., 2], x[..., 3]
    lit, timed = E > 0, T > 0
    t_us = torch.expm1(T.clamp(min=0.0))
    big = torch.finfo(t_us.dtype).max
    k_t = torch.where(timed, t_us, torch.full_like(t_us, big)).argmin(dim=1)           # earliest timed
    if ref_e_min > 0:
        strong = timed & (torch.expm1(E.clamp(min=0.0)) >= ref_e_min)
        k_s = torch.where(strong, t_us, torch.full_like(t_us, big)).argmin(dim=1)      # earliest timed and bright
        k_t = torch.where(strong.any(dim=1), k_s, k_t)
    k_e = torch.where(lit, E, torch.full_like(E, -big)).argmax(dim=1)                  # brightest lit
    any_t, any_l = timed.any(dim=1), lit.any(dim=1)
    k = torch.where(any_t, k_t, k_e)
    p_k = pos.gather(1, k[:, None, None].expand(-1, 1, 2)).squeeze(1)                  # (B, 2)
    p0 = torch.where((any_t | any_l)[:, None], p_k, pos.mean(dim=1))
    t0 = torch.where(any_t, t_us.gather(1, k[:, None]).squeeze(1), torch.zeros_like(t_us[:, 0]))
    timedf, litf = timed.to(x.dtype), lit.to(x.dtype)
    dt_ns = (t_us - t0[:, None]) * 1e3 * timedf
    n = x.shape[1]
    return torch.cat([pos - p0[:, None, :], E[..., None], dt_ns[..., None], litf[..., None], timedf[..., None],
                      p0[:, None, :].expand(-1, n, -1), torch.log1p(t0)[:, None, None].expand(-1, n, 1)], dim=-1)


# input_mode -> what the encoder sees. "reltime1": reltime_features with the reference at the earliest timed
# detector with >= 1 count.
_REF_E_MIN = {"reltime": 0.0, "reltime1": 1.0}
INPUT_MODES = ("raw",) + tuple(_REF_E_MIN)


def encoder_features(x: torch.Tensor, input_mode: str) -> torch.Tensor:
    """(B, n, 4) raw -> the encoder input for `input_mode` (raw: unchanged; reltime*: (B, n, 9))."""
    return x if input_mode == "raw" else reltime_features(x, _REF_E_MIN[input_mode])


def encoder_in_dim(input_mode: str, input_features: int = 4) -> int:
    return input_features if input_mode == "raw" else RELTIME_FEATURES


class DeepSetsRecon(nn.Module):
    """Permutation-invariant set→4-vector reconstruction.

    token_i = [x_i, y_i, E_i, T_i]         per-detector features
    h_i     = encoder(token_i)              shared encoder
    c       = context_proj(mean_i h_i)      invariant aggregate
    out     = decoder(c)                    single global decode → primary encoding

    Input: (B, n_det, input_features) in raw units.
    """

    def __init__(self,
                 n_det:          int = N_DETECTORS,
                 input_features: int = 4,
                 output_dim:     int = 4,
                 hidden:         int = 256,
                 context:        int = 128,
                 n_enc:          int = 3,
                 n_dec:          int = 2,
                 pool:           str = "mean",
                 input_mode:     str = "raw"):
        super().__init__()
        self.n_det          = n_det
        self.input_features = input_features
        self.output_dim     = output_dim
        self.pool           = pool
        # "raw": the encoder sees the 4 raw features; "reltime": reltime_features() first (9 features).
        # Callers always pass the raw (B, n, 4) input either way.
        self.input_mode     = input_mode
        enc_in = encoder_in_dim(input_mode, input_features)

        pool_dim = hidden * 2 if pool == "maxmean" else hidden
        self.encoder      = _mlp(enc_in, hidden, hidden, n_enc)
        self.context_proj = nn.Linear(pool_dim, context)
        self.decoder      = _mlp(context, hidden, output_dim, n_dec)

        self.register_buffer("in_mean",  torch.zeros(enc_in))
        self.register_buffer("in_std",   torch.ones(enc_in))
        self.register_buffer("out_mean", torch.zeros(output_dim))
        self.register_buffer("out_std",  torch.ones(output_dim))

    def set_normalization(self,
                          in_mean:  torch.Tensor,
                          in_std:   torch.Tensor,
                          out_mean: torch.Tensor,
                          out_std:  torch.Tensor) -> None:
        self.in_mean.copy_(in_mean)
        self.in_std.copy_(in_std)
        self.out_mean.copy_(out_mean)
        self.out_std.copy_(out_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (B, n_det, input_features) in raw units.
        Returns: (B, output_dim) in raw primary-encoding units.
        """
        x = encoder_features(x, self.input_mode)
        x_n   = (x - self.in_mean) / self.in_std        # broadcast (features,)
        h     = self.encoder(x_n)                        # (B, n_det, hidden)
        if self.pool == "maxmean":
            pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)  # (B, 2*hidden)
        else:
            pooled = h.mean(dim=1)                       # (B, hidden)
        c     = self.context_proj(pooled)                # (B, context)  invariant pool
        out_n = self.decoder(c)                          # (B, output_dim)
        return out_n * self.out_std + self.out_mean


PAIR_FEATURES = 6


def pair_features(x: torch.Tensor) -> torch.Tensor:
    """(B, n, 4) raw [x, y, log1p E, log1p(t * 1e6)] -> (B, n, n, 6), [b, i, j] = detector j relative to i:
    dx, dy, dist [km], dt [us] (0 unless both timed), both_timed, d log1p E."""
    pos, E, T = x[..., :2], x[..., 2], x[..., 3]
    timed = (T > 0).to(x.dtype)
    t_us = torch.expm1(T.clamp(min=0.0))
    d = (pos[:, None, :, :] - pos[:, :, None, :]) / 1000.0
    dist = (d.pow(2).sum(-1) + 1e-6).sqrt()
    both = timed[:, :, None] * timed[:, None, :]
    dt = (t_us[:, None, :] - t_us[:, :, None]) * both
    dE = E[:, None, :] - E[:, :, None]
    return torch.stack([d[..., 0], d[..., 1], dist, dt, both, dE], dim=-1)


class _PairBiasBlock(nn.Module):
    """Pre-LN self-attention + FFN; `bias` (B, heads, n, n) is added to the attention logits."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, z, bias):
        B, n, d = z.shape
        q, k, v = self.qkv(self.ln1(z)).view(B, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)              # (B, heads, n, d/heads)
        z = z + self.out(a.transpose(1, 2).reshape(B, n, d))
        return z + self.ffn(self.ln2(z))


class SetTransformerRecon(nn.Module):
    """DeepSetsRecon with attention between detectors before the pool.

    token_i = encoder(features_i)                       as DeepSetsRecon (reltime_features by default)
    bias_ij = pair_mlp(pair_features_ij) + log w_j      per head; w_j = soft "hit" weight from E_j
    z       = n_blocks x [attention(z, bias) + FFN]     no positional encoding: permutation-equivariant
    c       = context_proj([sum w z / sum w, max(z + log w)])
    out     = decoder(c)

    Detectors below ~hit_e_min counts get w -> 0 (log w >= -30): they are not attended to and not
    pooled. Everything is smooth in x, y. Input: (B, n_det, 4) raw, as DeepSetsRecon.
    """

    def __init__(self,
                 n_det:          int = N_DETECTORS,
                 input_features: int = 4,
                 output_dim:     int = 4,
                 hidden:         int = 256,
                 context:        int = 128,
                 n_enc:          int = 4,
                 n_dec:          int = 4,
                 d_model:        int = 128,
                 n_heads:        int = 4,
                 n_blocks:       int = 3,
                 pair_hidden:    int = 32,
                 hit_e_min:      float = 0.1,
                 hit_tau:        float = 0.5,
                 input_mode:     str = "reltime"):
        super().__init__()
        self.n_det, self.input_features, self.output_dim = n_det, input_features, output_dim
        self.n_heads, self.hit_e_min, self.hit_tau, self.input_mode = n_heads, hit_e_min, hit_tau, input_mode
        enc_in = encoder_in_dim(input_mode, input_features)
        self.encoder      = _mlp(enc_in, hidden, d_model, n_enc)
        self.pair_mlp     = _mlp(PAIR_FEATURES, pair_hidden, n_heads, 2)
        self.blocks       = nn.ModuleList(_PairBiasBlock(d_model, n_heads) for _ in range(n_blocks))
        self.context_proj = nn.Linear(2 * d_model, context)
        self.decoder      = _mlp(context, hidden, output_dim, n_dec)

        self.register_buffer("in_mean",  torch.zeros(enc_in))
        self.register_buffer("in_std",   torch.ones(enc_in))
        self.register_buffer("out_mean", torch.zeros(output_dim))
        self.register_buffer("out_std",  torch.ones(output_dim))

    set_normalization = DeepSetsRecon.set_normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Soft hit weight on log(counts): ~1 above hit_e_min, 1/2 at it, -> 0 below (hit_tau in e-folds of E).
        log_e = torch.log(torch.expm1(x[..., 2].clamp(min=0.0)) + 1e-12)
        logw = F.logsigmoid((log_e - math.log(self.hit_e_min)) / self.hit_tau).clamp(min=-30.0)          # (B, n)
        f = encoder_features(x, self.input_mode)
        z = self.encoder((f - self.in_mean) / self.in_std)                                # (B, n, d)
        bias = self.pair_mlp(pair_features(x)).permute(0, 3, 1, 2) + logw[:, None, None, :]   # (B, h, n, n)
        for blk in self.blocks:
            z = blk(z, bias)
        w = logw.exp()[..., None]
        pooled = torch.cat([(w * z).sum(1) / w.sum(1).clamp(min=1e-6), (z + logw[..., None]).amax(1)], dim=-1)
        out_n = self.decoder(self.context_proj(pooled))
        return out_n * self.out_std + self.out_mean


def build_recon_from_ckpt(ckpt: dict, n_det: int, device=None):
    """Polymorphic recon loader — dispatches on ckpt["config"]["model_type"].

    Returns a frozen eval-mode model with normalization applied.
    Handles both "mlp" (Reconstruction) and "deepsets" (DeepSetsRecon).
    """
    cfg   = ckpt.get("config", {})
    mtype = cfg.get("model_type", "mlp")
    n_det_ckpt      = int(ckpt.get("num_detectors", n_det))
    input_features  = int(ckpt.get("input_features", 4))

    if mtype == "deepsets":
        model = DeepSetsRecon(
            n_det=n_det_ckpt,
            input_features=input_features,
            hidden=int(cfg.get("hidden", 256)),
            context=int(cfg.get("context", 128)),
            n_enc=int(cfg.get("n_enc", 3)),
            n_dec=int(cfg.get("n_dec", 2)),
            pool=cfg.get("pool", "mean"),
            input_mode=cfg.get("input_mode", "raw"),
        )
    elif mtype == "settf":
        model = SetTransformerRecon(
            n_det=n_det_ckpt, input_features=input_features,
            hidden=int(cfg["hidden"]), context=int(cfg["context"]), n_enc=int(cfg["n_enc"]), n_dec=int(cfg["n_dec"]),
            d_model=int(cfg["d_model"]), n_heads=int(cfg["n_heads"]), n_blocks=int(cfg["n_blocks"]),
            pair_hidden=int(cfg["pair_hidden"]), hit_e_min=float(cfg["hit_e_min"]), hit_tau=float(cfg["hit_tau"]),
            input_mode=cfg.get("input_mode", "reltime"),
        )
    else:
        model = Reconstruction(
            n_det=n_det_ckpt,
            input_features=input_features,
            hidden=int(cfg.get("hidden", 512)),
            dropout=float(cfg.get("dropout", 0.1)),
        )

    model.load_state_dict(ckpt["state_dict"])
    model.set_normalization(
        in_mean  = ckpt["input_mean"],
        in_std   = ckpt["input_std"],
        out_mean = ckpt["target_mean"],
        out_std  = ckpt["target_std"],
    )
    if device is not None:
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model