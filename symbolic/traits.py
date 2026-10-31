"""Per-shower traits of the in-plane shape: a noise-input network trained against CORSIKA, to be distilled into
formulas afterwards (study_plan.md Part 2; user 10-30).

The variety candidates (variety.py) all failed: the shape varies mostly as a trait of the whole shower, shared by
electrons and photons, which no plane-wise form produces. Here a plane's ring shares are softmax(x B + A(z, x,
species) + N(e, x, species)) on all rings (the outer rings scatter too, and that moves the inner cuts): x B the ring
law, z K numbers per shower (shared by its planes and species), e ~ N(0, I_P) drawn per plane, A and N small
networks (zero at the start, so the model starts at the law). Everything is measured on the log-odds of every dense
plane's cumulative across-track shares, each bin divided by its spread in the training data (so every bin weighs
alike, as the rule judges each cut against its own noise). Two stages, because a score-trained network puts the
shared variety into per-plane noise (results.md 10-30):
1. traits: every training shower gets its own z, fitted with A by least squares on its planes plus a N(0, I) prior
   on z (MAP, full batch), so a shower's shared shape can only be fitted through its z;
2. plane noise: N trained with A's offsets of each shower's own z fixed, by energy score (M_SAMPLES draws against the
   CORSIKA shower: whole shower plus the sum of the planes' own scores).
New showers draw z from a Gaussian fitted to the training traits. K = 0 has stage 2 only.

Rule (fixed before the run), per class, on held-out showers, K = 0..K_MAX:
1. spread and 2. memory exactly as variety.py, every species;
3. species: correlation of the 30 m offsets of two species on the same plane within 2 se of CORSIKA's (bootstrap by
   shower, both sides), every species pair.
The answer is the smallest K that passes 1-3 for every species of the class. Reported: the 2.3 pooled check.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
from torch import nn

from symbolic import logs
from modules.constants import SPECIES_NAMES
from symbolic import planes, radial, radial_bins, variety

FOLDER = "traits"
K_MAX = 3
P = 4                                    # per-plane noise inputs
M_SAMPLES = 8                            # model draws per shower in the energy score
HIDDEN = 64
DEPTH = 2                                # hidden layers
ACT = "tanh"
ACTS = {"tanh": nn.Tanh, "silu": nn.SiLU, "relu": nn.ReLU}
EPOCHS = 1000
PATIENCE = 50
Z_STEPS = 5000                           # full-batch steps of the trait fit (stage 1)
Z_TOL = 1e-5                             # stage 1 stops when 100 steps improve the loss by less than this share
BATCH = 64
LR = 1e-3
VAL = 0.1
EPS = 1e-5                               # clip of the cumulative shares before the log-odds
S = len(SPECIES_NAMES)


def tensors(data, laws, c):
    """Per class: showers x species x planes arrays of observed shares, features, law logits and a dense-plane mask."""
    keep = {s: (d["label"] == c) & (d["n"] >= variety.MIN_CELLS) for s, d in data.items()}
    sh = np.unique(np.concatenate([data[s]["i"][m] for s, m in keep.items()]))
    n_k = 1 + max(int(d["k"].max()) for d in data.values())
    n_f = data[SPECIES_NAMES[0]]["x"].shape[1]
    t = {"i": sh, "O": np.zeros((len(sh), S, n_k, radial_bins.N_BIN)), "X": np.zeros((len(sh), S, n_k, n_f)),
         "base": np.zeros((len(sh), S, n_k, radial_bins.N_RING)), "mask": np.zeros((len(sh), S, n_k), bool)}
    for j, s in enumerate(SPECIES_NAMES):
        d, m = data[s], keep[s]
        r, k = np.searchsorted(sh, d["i"][m]), d["k"][m]
        t["O"][r, j, k], t["X"][r, j, k], t["mask"][r, j, k] = d["obs"][m], d["x"][m], True
        t["base"][r, j, k] = d["x"][m] @ laws[s]
    return t


def take(t, rows):
    return {key: val[rows] for key, val in t.items()}


def as_torch(t):
    return {key: torch.as_tensor(t[key], dtype=torch.float64) for key in ("O", "X", "base")} | \
           {"mask": torch.as_tensor(t["mask"])}


def mlp(n_in, width, depth, act):
    """Small network with its last layer zero, so it starts at no offset."""
    sizes = [n_in] + [width] * depth
    layers = [m for a, b in zip(sizes[:-1], sizes[1:]) for m in (nn.Linear(a, b), ACTS[act]())]
    net = nn.Sequential(*layers, nn.Linear(width, radial_bins.N_RING))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


def features(X, lead):
    """Plane features and species one-hot of showers X [B, S, n_k, n_f], broadcast to lead + (S, n_k, n_f + S)."""
    B, _, n_k, n_f = X.shape
    pad = (1,) * (len(lead) - 1)
    return torch.cat([X.view(B, *pad, S, n_k, n_f).expand(*lead, S, n_k, n_f),
                      torch.eye(S).view(1, *pad, S, 1, S).expand(*lead, S, n_k, S)], -1)


def shower_offsets(A, z, X):
    """A's ring-logit offsets for traits z [B, ..., K] of showers X: [B, ..., S, n_k, rings]."""
    lead, K = z.shape[:-1], z.shape[-1]
    n_k = X.shape[2]
    return A(torch.cat([z.view(*lead, 1, 1, K).expand(*lead, S, n_k, K), features(X, lead)], -1))


class Traits(nn.Module):
    """Plane noise N(e, x, species) -> logit offsets on every ring, e per plane (P); `shift` adds offsets fixed from
    outside (the shower part)."""

    def __init__(self, n_f, width=HIDDEN, depth=DEPTH, act=ACT):
        super().__init__()
        self.plane = mlp(P + n_f + S, width, depth, act)
        self.register_buffer("scale", torch.ones(S, radial_bins.N_BIN - 1))     # per species and bin, set by train

    def forward(self, X, base, m, gen=None, shift=0.0):
        """Ring shares of m draws of every (shower, species, plane): [B, m, S, n_k, rings]."""
        B, _, n_k, _ = X.shape
        e = torch.randn(B, m, S, n_k, P, generator=gen)
        off = self.plane(torch.cat([e, features(X, (B, m))], -1))
        return torch.softmax(base.unsqueeze(1) + shift + off, -1)


class ShowerTraits(nn.Module):
    """The two-stage model: z ~ N(mean, cov) of the fitted training traits (kept as `z_fit`, the data to distil A
    from), A's offsets, then the plane noise."""

    def __init__(self, A, z, noise):
        super().__init__()
        self.A, self.noise = A, noise
        K = z.shape[1]
        self.register_buffer("z_fit", z)
        self.register_buffer("z_mean", z.mean(0))
        self.register_buffer("z_chol", torch.linalg.cholesky(torch.cov(z.T).reshape(K, K)))

    def forward(self, X, base, m, gen=None):
        B = X.shape[0]
        z = self.z_mean + torch.randn(B, m, len(self.z_mean), generator=gen) @ self.z_chol.T
        return self.noise(X, base, m, gen, shift=shower_offsets(self.A, z, X))


def across(q, Mn):
    p = q @ Mn
    return p / p.sum(-1, keepdim=True)


def log_odds(p):
    """Log-odds of the cumulative shares, last bin (always 1) dropped."""
    c = torch.cumsum(p, -1)[..., :-1].clamp(EPS, 1 - EPS)
    return torch.log(c) - torch.log1p(-c)


def bin_scale(T):
    """Spread of every bin's log-odds over the dense planes of each species: [S, bins - 1]."""
    lo = log_odds(T["O"])
    return torch.stack([lo[:, j][T["mask"][:, j]].std(0) for j in range(S)])


def _es(F, y):
    """Energy score of the draws F [B, m, ..., D] against y [B, ..., D], per leading index and position."""
    dist = lambda a, b: torch.sqrt(((a - b) ** 2).sum(-1) + 1e-12)
    m = F.shape[1]
    return dist(F, y.unsqueeze(1)).mean(1) - 0.5 * dist(F.unsqueeze(2), F.unsqueeze(1)).sum((1, 2)) / (m * (m - 1))


def energy_score(model, T, Mn, gen=None):
    """Score of the showers in T on the log-odds of the cumulative across-track shares of their dense planes: the
    energy score of the whole shower (all planes and species; carries the shared trait) plus the sum of every plane's own energy score
    (its scatter, which the whole-shower score alone barely sees). Both are proper, so is the sum."""
    mask = T["mask"].to(T["O"].dtype)
    sc = model.scale.unsqueeze(1)                                                            # [S, 1, bins - 1]
    y = log_odds(T["O"]) / sc * mask.unsqueeze(-1)                                           # [B, S, n_k, bins - 1]
    F = log_odds(across(model(T["X"], T["base"], M_SAMPLES, gen), Mn)) / sc * mask.unsqueeze(1).unsqueeze(-1)
    whole = _es(F.flatten(2), y.flatten(1))
    planes_ = (_es(F, y) * mask).sum((1, 2))
    return (whole + planes_).mean()


def plane_loss(off, T, Mn, scale):
    """Half the squared error of the standardised log-odds over every dense plane, given ring-logit offsets off."""
    r = (log_odds(across(torch.softmax(T["base"] + off, -1), Mn)) - log_odds(T["O"])) / scale.unsqueeze(1)
    return 0.5 * ((r ** 2).sum(-1) * T["mask"].to(r.dtype)).sum()


def fit_traits(K, t, M, seed, steps=Z_STEPS, lr=LR, width=HIDDEN, depth=DEPTH, act=ACT):
    """Stage 1: the K traits of every shower of t and the shower network A, MAP under a N(0, I) prior on z."""
    torch.manual_seed(seed)
    Mn = torch.as_tensor(M / M.sum(1, keepdims=True), dtype=torch.float64)
    T = as_torch(t)
    scale = bin_scale(T)
    A = mlp(K + t["X"].shape[-1] + S, width, depth, act)
    z = nn.Parameter(0.1 * torch.randn(len(t["i"]), K))
    opt = torch.optim.Adam([*A.parameters(), z], lr=lr)
    hist = []
    for step in range(steps):
        opt.zero_grad()
        loss = (plane_loss(shower_offsets(A, z, T["X"]), T, Mn, scale) + 0.5 * (z ** 2).sum()) / len(z)
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if step >= 100 and hist[-101] - hist[-1] < Z_TOL * abs(hist[-1]):
            break
    return A, z.detach(), {"trait steps": step + 1, "trait loss": hist[-1]}


def fit(K, t, M, seed, epochs=EPOCHS, steps=Z_STEPS, **net):
    """The model for K traits on the showers of t: stage 1 (K > 0), then the plane noise on top of it."""
    if K == 0:
        return train(t, M, seed, epochs, **net)
    A, z, info = fit_traits(K, t, M, seed, steps, **net)
    with torch.no_grad():
        shift = shower_offsets(A, z, torch.as_tensor(t["X"])).numpy()
    noise, fit2 = train({**t, "base": t["base"] + shift}, M, seed, epochs, **net)
    zn = (z - z.mean(0)) / z.std(0)
    info |= {"trait z mean": z.mean(0).tolist(), "trait z sd": z.std(0).tolist(),
             "trait z skew": (zn ** 3).mean(0).tolist(), "trait z excess kurtosis": ((zn ** 4).mean(0) - 3).tolist()}
    return ShowerTraits(A, z, noise), {**fit2, **info}


def train(t, M, seed, epochs=EPOCHS, lr=LR, **net):
    """Stage 2, the plane noise, fitted on the showers of t (a VAL share held back for early stopping, scored with
    fixed noise); net: width, depth, act of the network."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Mn = torch.as_tensor(M / M.sum(1, keepdims=True), dtype=torch.float64)
    model = Traits(t["X"].shape[-1], **net)
    T = as_torch(t)
    model.scale.copy_(bin_scale(T))
    perm = rng.permutation(len(t["i"]))
    n_val = max(1, int(VAL * len(perm)))
    val, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, state, wait = np.inf, None, 0
    for ep in range(epochs):
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // BATCH)):
            opt.zero_grad()
            loss = energy_score(model, {key: v[b] for key, v in T.items()}, Mn)
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(energy_score(model, {key: x[val] for key, x in T.items()}, Mn,
                                   torch.Generator().manual_seed(seed)))
        if v < best - 1e-4 * abs(v):
            best, state, wait = v, {key: x.clone() for key, x in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait > PATIENCE:
                break
    model.load_state_dict(state)
    return model, {"epochs": ep + 1, "val energy score": best}


def corr_boot(x, y, unit, rng):
    sh, inv = np.unique(unit, return_inverse=True)
    boots = [variety.wcorr(x, y, rng.multinomial(len(sh), np.full(len(sh), 1 / len(sh)))[inv])
             for _ in range(variety.BOOT)]
    return float(np.corrcoef(x, y)[0, 1]), float(np.std(boots))


def judge(model, t, M, rng):
    """Rules 1-3 on the held-out showers t: per species as variety.judge, then the species pairs."""
    T = as_torch(t)
    Mn = torch.as_tensor(M / M.sum(1, keepdims=True), dtype=torch.float64)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(int(rng.integers(2 ** 31)))
        P_mod = across(model(T["X"], T["base"], variety.DRAWS, gen), Mn).double().numpy()     # [n, draws, S, n_k, bins]
        P_law = across(torch.softmax(T["base"], -1), Mn).double().numpy()
    res, u = {}, {}
    for j, s in enumerate(SPECIES_NAMES):
        r, k = np.nonzero(t["mask"][:, j])
        if len(r) == 0:
            continue
        obs, p0 = t["O"][r, j, k], P_law[r, j, k]
        pm = np.concatenate([P_mod[r, d, j, k] for d in range(variety.DRAWS)])
        i_m, k_m = np.concatenate([r * variety.DRAWS + d for d in range(variety.DRAWS)]), np.tile(k, variety.DRAWS)
        s_obs, s_law, s_mod = variety.cut_shares(obs), variety.cut_shares(p0), variety.cut_shares(pm)
        sp = variety.spread(s_obs, s_mod, r, rng)
        u_obs, u_mod = s_obs[:, 1] - s_law[:, 1], s_mod[:, 1] - np.tile(s_law[:, 1], variety.DRAWS)
        mem = {lag: (variety.memory(u_obs, r, k, lag, rng), variety.memory(u_mod, i_m, k_m, lag, rng))
               for lag in variety.LAGS}
        win = lambda p: radial.pooled(radial_bins.window(p))
        res[s] = {"planes": len(r), "spread W1 model / noise per cut": {int(c): sp[n] for n, c in enumerate(variety.CUTS_M)},
                  "memory corr CORSIKA (se) / model (se)": mem,
                  "median per-plane KL to the law, CORSIKA / model": [variety.median_kl(obs, p0),
                                                                      variety.median_kl(pm[: len(r)], p0)],
                  "2.3 pooled W1 with traits / law alone / noise": [radial.w1(win(pm), win(obs)), radial.w1(win(p0), win(obs)),
                                                                   radial.split_half(radial_bins.window(obs), r, rng)],
                  "rule 1 spread": all(w <= 2 * n for w, n in sp),
                  "rule 2 memory": all(abs(mo[0] - mc[0]) <= 2 * np.hypot(mc[1], mo[1]) for mc, mo in mem.values())}
        u[j] = (r * 64 + k, u_obs, u_mod.reshape(variety.DRAWS, -1))
    pairs = {}
    for a in u:
        for b in u:
            if b <= a:
                continue
            (ka, oa, ma), (kb, ob, mb) = u[a], u[b]
            _, ia, ib = np.intersect1d(ka, kb, return_indices=True)
            if len(ia) <= 10:
                continue
            c_obs = corr_boot(oa[ia], ob[ib], ka[ia] // 64, rng)
            unit = np.concatenate([ka[ia] // 64 * variety.DRAWS + d for d in range(variety.DRAWS)])
            c_mod = corr_boot(ma[:, ia].ravel(), mb[:, ib].ravel(), unit, rng)
            pairs[f"{SPECIES_NAMES[a]}-{SPECIES_NAMES[b]}"] = {
                "planes": len(ia), "corr CORSIKA (se) / model (se)": [c_obs, c_mod],
                "pass": bool(abs(c_obs[0] - c_mod[0]) <= 2 * np.hypot(c_obs[1], c_mod[1]))}
    ok = all(v["rule 1 spread"] and v["rule 2 memory"] for v in res.values()) and all(v["pass"] for v in pairs.values())
    return {"species": res, "species pairs (rule 3)": pairs, "verdict": "PASS" if ok else "FAIL"}


def summary_line(r):
    out = []
    for s, v in r["species"].items():
        worst = max(w / n for w, n in v["spread W1 model / noise per cut"].values())
        mem = " ".join(f"{mc[0]:.2f}/{mo[0]:.2f}" for mc, mo in v["memory corr CORSIKA (se) / model (se)"].values())
        out.append(f"{s} spread {worst:.1f}x mem {mem} {'ok' if v['rule 1 spread'] and v['rule 2 memory'] else 'X'}")
    out += [f"{p} {v['corr CORSIKA (se) / model (se)'][0][0]:.2f}/{v['corr CORSIKA (se) / model (se)'][1][0]:.2f} "
            f"{'ok' if v['pass'] else 'X'}" for p, v in r["species pairs (rule 3)"].items()]
    return " | ".join(out)


def stage_run(out, args, log):
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    M = radial_bins.ring_matrix()
    data = radial_bins.plane_table(args.out)
    rng = np.random.default_rng(args.seed)
    res = {"K max": K_MAX, "plane noise": P, "samples": M_SAMPLES, "min cells": variety.MIN_CELLS, "hidden": HIDDEN,
           "trait steps max": Z_STEPS,
           "rule": __doc__.split("Rule")[1].strip(), "results": {}}
    for c, cn in planes.CLASS.items():
        laws = {s: np.load(os.path.join(args.out, radial_bins.FOLDER, f"conv_{s}", "ring_laws.npz"))[f"{s}.{cn}.rings"]
                for s in SPECIES_NAMES}
        t = tensors(data, laws, c)
        test = t["i"] % 5 == 0
        tr, te = take(t, ~test), take(t, test)
        res["results"][cn] = {"showers train / held out": [int((~test).sum()), int(test.sum())]}
        for K in range(K_MAX + 1):
            model, info = fit(K, tr, M, args.seed)
            r = {**info, **judge(model, te, M, rng)}
            res["results"][cn][f"K={K}"] = r
            torch.save({"state": model.state_dict(), "K": K}, os.path.join(out, f"traits_{cn}_K{K}.pt"))
            log.info("%-8s K=%d epochs %d | %s | %s", cn, K, info["epochs"], summary_line(r), r["verdict"])
            logs.save(out, res, args, log)
        passing = [K for K in range(K_MAX + 1) if res["results"][cn][f"K={K}"]["verdict"] == "PASS"]
        res["results"][cn]["smallest passing K"] = passing[0] if passing else None
        log.info("%-8s smallest passing K: %s", cn, res["results"][cn]["smallest passing K"])
        logs.save(out, res, args, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="run", help="output folder under traits/")
    args = ap.parse_args()
    out, log = logs.start(os.path.join(FOLDER, args.name), args)
    stage_run(out, args, log)


if __name__ == "__main__":
    main()
