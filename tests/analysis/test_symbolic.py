"""CPU self-test of the along-the-shower model: slant depth, profile fits, plane sums, the sampler (margins, copula,
plane residuals), the energy score and the CORSIKA repair."""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import h5py
import numpy as np
import torch
from scipy import special

from symbolic import (annuli, atmosphere as atm, copula as cp, evaluate as ev, fit_profiles, lateral,
                      longitudinal, planes, sampler, score, sources, unwrap as uw)

DIST = atm.PLANE_SPACING * np.arange(1, 25)


def analytic_vertical(h0, h1):
    """Exact vertical depth between two altitudes inside the exponential layers."""
    out, lo = 0.0, 0.0
    for top, b, c in atm.LAYERS:
        a, z = max(h0, lo), min(h1, top)
        if z > a:
            out += b * (np.exp(-a / c) - np.exp(-z / c))
        lo = top
    return out


def test_vertical_depth_matches_analytic():
    X = atm.slant_depth([0.0], DIST)[0]
    exact = [analytic_vertical(atm.INJECTION_HEIGHT, atm.INJECTION_HEIGHT + d) for d in DIST]
    assert np.allclose(X, exact, rtol=1e-5), np.max(np.abs(X / exact - 1))


def test_depth_geometry():
    rho0 = atm.density(atm.INJECTION_HEIGHT)
    X = atm.slant_depth([90.0], [1.0, 2.0])[0]
    assert np.allclose(X, rho0 * 100.0 * np.array([1.0, 2.0]), rtol=1e-3)
    X = atm.plane_depths([60.0, 90.0, 100.0], 24)
    assert np.all(np.diff(X, axis=1) > 0) and X[0, -1] < X[1, -1] < X[2, -1]
    try:
        atm.slant_depth([180.0], [3000.0])
        raise AssertionError("an axis below the Earth radius was accepted")
    except ValueError:
        pass


def test_layer_table_is_nearly_continuous():
    for top, *_ in atm.LAYERS[:3]:
        lo, hi = atm.density(top - 1e-3), atm.density(top + 1e-3)
        assert abs(hi / lo - 1) < 0.02, (top, lo, hi)
    assert abs(atm.density(0.0) - 1.24e-3) < 1e-5


def toy_depths():
    return atm.plane_depths([80.0], 24)[0]


def test_profile_fit_recovers_both_skews_and_rejects_scrambled_planes():
    rng = np.random.default_rng(0)
    X = toy_depths()
    for R in (0.25, -0.15):
        y = 3.0 + longitudinal.shape(X, 450.0, 220.0, R) + rng.normal(0, 0.02, len(X))
        ln_n, th, mse, n_fit = longitudinal.fit_one(X, y)
        assert n_fit == 24 and mse < 0.02 ** 2 * 2 and abs(th[2] - R) < 0.02 and abs(th[0] - 450.0) < 5, (R, th, mse)
        assert longitudinal.fit_one(X, y[rng.permutation(len(y))])[2] > 100 * mse


def test_profile_is_gaisser_hillas_with_a_gaussian_limit():
    X = toy_depths()
    xmax, x0, lam = 400.0, -20.0, 70.0
    gh = (xmax - x0) / lam * np.log((X - x0) / (xmax - x0)) + (xmax - X) / lam
    L, R = np.sqrt(lam * (xmax - x0)), np.sqrt(lam / (xmax - x0))
    assert np.allclose(longitudinal.shape(X, xmax, L, R), gh, atol=1e-9)
    z = (X - xmax) / 200.0
    for r in (0.0, 1e-7, 1e-5):
        assert np.allclose(longitudinal.shape(X, xmax, 200.0, r), -0.5 * z * z + r * z ** 3 / 3, atol=1e-6)
    ln_n, th, mse, _ = longitudinal.fit_one(X, 3.0 + longitudinal.shape(X, xmax, 250.0, 0.0))
    assert mse < 1e-8 and abs(th[0] - xmax) < 1 and abs(th[1] - 250.0) < 2 and abs(th[2]) < 0.01, (th, mse)


def test_empty_planes_are_skipped_and_too_few_planes_refused():
    X = toy_depths()
    y = 3.0 + longitudinal.shape(X, 400.0, 200.0, 0.2)
    y[:3] = np.nan
    assert longitudinal.fit_one(X, y)[3] == 21
    y[3:19] = np.nan
    assert longitudinal.fit_one(X, y) is None


def test_plane_sums_and_clouds_agree():
    pts = np.array([[0, 0, 2, 1.0, 0], [1, 0, 2, 2.0, 0], [0, 1, 23, 4.0, 0], [0, 0, 5, 0.0, 0]])
    E, N = planes.plane_sums(pts)
    assert E[2] == 3.0 and E[23] == 4.0 and E.sum() == 7.0 and N[2] == 2 and N.sum() == 3
    c = torch.zeros(2, 6, 5, dtype=torch.float64)
    c[0, :4] = torch.as_tensor(pts)
    Ec, Nc = planes.from_clouds(c)
    assert np.array_equal(Ec[0], E) and np.array_equal(Nc[0], N) and Ec[1].sum() == 0
    for bad in ([[0, 0, 24, 1.0, 0]], [[0, 0, 2.5, 1.0, 0]], [[0, 0, 2, -1.0, 0]]):
        try:
            planes.plane_sums(np.array(bad, float))
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


def write_toy(folder, misalign=None, short=None):
    rng = np.random.default_rng(3)
    n = 4
    for s in sources.SPECIES_NAMES:
        with h5py.File(os.path.join(folder, f"{s}.h5"), "w") as f:
            npts = rng.integers(1, 6, n)
            data = f.create_dataset("showers", (n,), dtype=h5py.vlen_dtype(np.float32))
            for i in range(n):
                p = np.zeros((npts[i], 5), np.float32)
                p[:, 2], p[:, 3] = np.arange(npts[i]) % 24, i + 1
                data[i] = p.ravel()
            ids = np.arange(n) + (s == misalign)
            num = npts - (s == short)
            for k, v in (("shower_ids", ids), ("energies", np.full((n, 1), 1e6, np.float32)),
                         ("directions", np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)),
                         ("pdg", np.array([0, 1, 1, 0])), ("num_points", num)):
                f[k] = v
    return lambda species, split: os.path.join(folder, f"{species}.h5")


def test_corsika_planes_check_alignment_and_counts():
    with tempfile.TemporaryDirectory() as d:
        d_, meta = planes.from_corsika("test", [2, 0], path=write_toy(d))
        assert meta["rows"].tolist() == [0, 2] and np.allclose(meta["zenith"], 0.0)
        for s in sources.SPECIES_NAMES:
            assert np.array_equal(d_[s]["N"].sum(1), d_[s]["num_points"])
            assert np.allclose(d_[s]["E"].sum(1), d_[s]["num_points"] * np.array([1, 3]))
    for kw in ({"misalign": "muon"}, {"short": "photon"}):
        with tempfile.TemporaryDirectory() as d:
            try:
                planes.from_corsika("test", [0, 1], path=write_toy(d, **kw))
                raise AssertionError(f"accepted {kw}")
            except ValueError:
                pass


def test_parts_cover_rows_and_gate_rejects_gaps():
    n = 1003
    rows = np.concatenate([fit_profiles.part_rows(n, 7, i) for i in range(7)])
    fit_profiles.check_cover(rows, n)
    for bad in (rows[:-1], np.concatenate([rows, rows[:1]]), np.concatenate([rows[:10], rows[11:], rows[10:11]])):
        try:
            fit_profiles.check_cover(bad, n)
        except ValueError:
            continue
        raise AssertionError("check_cover accepted a broken cover")


def test_latent_round_trip_and_flags():
    rng = np.random.default_rng(3)
    n = 400
    X = np.tile(atm.plane_depths([80.0], 24), (n, 1))
    E = np.ones((n, 24))
    xmax, L, lnn, le = rng.uniform(200, 600, n), rng.uniform(100, 300, n), rng.uniform(10, 12, n), rng.uniform(5, 6, n)
    pos, neg = longitudinal.r_caps(xmax, L, X[:, 0], X[:, -1])
    R = rng.uniform(-0.5, 0.5, n) * np.where(rng.uniform(size=n) < 0.5, pos, neg)
    R = np.where(R >= 0, np.minimum(R, 0.5 * pos), np.maximum(R, -0.5 * neg))
    f = {"ln_n": lnn, "theta": np.column_stack([xmax, L, R])}
    back = longitudinal.from_latent(longitudinal.to_latent(f, le), le)
    assert all(np.allclose(a, b) for a, b in zip(back, (xmax, L, R, lnn)))
    fl = longitudinal.flags(f, X, E)
    assert not fl["bound"].any() and not fl["unfit"].any()
    f["theta"][7, 2], f["theta"][8, 2] = 0.999 * pos[7], -0.999 * neg[8]
    f["theta"][9, 0] = X[0, -1] + 100.0
    f["theta"][10, 0] = X[0, 0] - 50.0
    f["ln_n"][11] = np.nan
    fl = longitudinal.flags(f, X, E)
    assert fl["bound"][7] and fl["bound"][8] and fl["unfit"][11] and fl["unfit"].sum() == 1
    assert fl["outside"][9] and fl["outside"][10] and fl["outside"].sum() == 2


def toy_conditions(n, rng):
    l, u = rng.uniform(5.0, 6.2, n), rng.uniform(-0.2, 0.5, n)
    design = cp.Design(l, u)
    return design(l, u), design(l, u)[:, 1]


def test_johnson_su_margin_recovers_a_skewed_heavy_tailed_law():
    rng = np.random.default_rng(5)
    D, a = toy_conditions(20000, rng)
    xi, lam, gam, dlt = 1.0 + 0.5 * a, 0.8, -0.6, 1.3
    x = xi + lam * np.sinh((rng.normal(size=len(a)) - gam) / dlt)
    tr, te = slice(0, 15000), slice(15000, None)
    jsu, nor = cp.Margin("jsu").fit(D[tr], x[tr]), cp.Margin("normal").fit(D[tr], x[tr])
    gain = jsu.loglik(D[te], x[te]).mean() - nor.loglik(D[te], x[te]).mean()
    rate_j, rate_n = cp.tail_rate(jsu, D[te], x[te]), cp.tail_rate(nor, D[te], x[te])
    print(f"      JSU minus normal {gain:.3f} nats; tail rate JSU {rate_j:.4f} normal {rate_n:.4f}")
    assert gain > 0.05 and abs(rate_j - 0.0027) < 0.0025 and rate_n > 2 * rate_j, (gain, rate_j, rate_n)
    err = np.abs(jsu.ppf(D[te], np.full(len(x[te]), 0.5)) - (xi[te] + lam * np.sinh(-gam / dlt)))
    assert err.mean() < 0.02 and err.max() < 0.08, (err.mean(), err.max())
    z = jsu.score(D[te], x[te])
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05
    assert np.allclose(jsu.ppf(D[te], special.ndtr(z)), x[te], atol=1e-6, rtol=1e-6)


def test_mlp_margin_follows_a_curved_law_the_linear_one_misses():
    rng = np.random.default_rng(6)
    D, a = toy_conditions(12000, rng)
    x = np.sin(2.5 * a) + 0.2 * rng.normal(size=len(a))
    tr, te = slice(0, 9000), slice(9000, None)
    lin, mlp = cp.Margin("jsu").fit(D[tr], x[tr]), cp.MLPMargin(seed=0).fit(D[tr], x[tr])
    truth = (-0.5 * np.log(2 * np.pi * 0.04) - (x[te] - np.sin(2.5 * a[te])) ** 2 / 0.08).mean()
    ll_mlp, ll_lin = mlp.loglik(D[te], x[te]).mean(), lin.loglik(D[te], x[te]).mean()
    z = mlp.score(D[te], x[te])
    print(f"      held out, nats below the true law: MLP {truth - ll_mlp:.3f} ({mlp.steps} steps), linear {truth - ll_lin:.3f}")
    assert truth - ll_mlp < 0.05 and abs(z.mean()) < 0.1 and abs(z.std() - 1) < 0.05, (truth - ll_mlp, z.mean(), z.std())
    assert np.allclose(mlp.ppf(D[te], special.ndtr(z)), x[te], atol=1e-6, rtol=1e-6)


def test_evaluation_metrics_separate_a_wrong_model_from_the_truth():
    E = np.array([[0.0, 0.5, 3.0, 9.0, 2.0, 0.0], [5.0, 4.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6])
    sm = ev.summaries(E)
    assert sm["onset"].tolist() == [2, 0, 6] and sm["end"].tolist() == [4, 1, -1] and sm["peak"].tolist()[:2] == [3, 0]
    rng = np.random.default_rng(8)
    n = 1500
    cond = np.column_stack([rng.uniform(5, 6, n), rng.uniform(-0.2, 0.5, n)])
    law = lambda: 10 ** (2 + cond[:, :1] + 0.3 * rng.normal(size=(n, 24)))
    Ec = {s: law() for s in sources.SPECIES_NAMES}
    same = ev.all_metrics({s: law()[:, None] for s in sources.SPECIES_NAMES}, Ec, cond, cond)
    off = ev.all_metrics({s: 3 * law()[:, None] for s in sources.SPECIES_NAMES}, Ec, cond, cond)
    j0, j1 = same["joint"], off["joint"]
    print(f"      AUC same {j0['auc']:.3f} shifted {j1['auc']:.3f}; mean diff {j0['mean_abs_diff']:.3f} / {j1['mean_abs_diff']:.3f}")
    assert abs(j0["auc"] - 0.5) < 0.06 and j1["auc"] > 0.9 and j1["mean_abs_diff"] > 5 * j0["mean_abs_diff"]
    assert {"joint.mean_abs_diff", "joint.auc", "electron.w1_total"} <= set(ev.worse(off, same))


def test_torch_profile_shape_matches_numpy_and_has_finite_gradients():
    rng = np.random.default_rng(12)
    X = np.tile(atm.plane_depths([80.0], 24), (200, 1))
    xmax, L = rng.uniform(300, 900, (200, 1)), rng.uniform(100, 400, (200, 1))
    R = np.r_[rng.uniform(-0.6, 0.6, 190), np.zeros(5), np.full(5, 1e-6)][:, None]
    ref = longitudinal.shape(X, xmax, L, R)
    Rt = torch.tensor(R, requires_grad=True)
    got = longitudinal.shape_torch(torch.tensor(X), torch.tensor(xmax), torch.tensor(L), Rt)
    assert np.allclose(np.nan_to_num(got.detach().numpy(), neginf=-1e9), np.nan_to_num(ref, neginf=-1e9), atol=1e-9)
    ok = torch.isfinite(got)
    torch.exp(torch.where(ok, got, torch.zeros_like(got))).where(ok, torch.zeros_like(got)).sum().backward()
    assert torch.all(torch.isfinite(Rt.grad)) and np.any(~np.isfinite(ref))




def test_gaussian_copula_and_species_blocks():
    rng = np.random.default_rng(7)
    lam = np.array([0.8, 0.7, 0.6, 0.0, 0.0, 0.3])
    C = np.outer(lam, lam)
    np.fill_diagonal(C, 1.0)
    Za, Zb = rng.multivariate_normal(np.zeros(6), C, 8000), rng.multivariate_normal(np.zeros(6), C, 8000)
    Ch = cp.corr(Za)
    g, blocks = cp.gaussian_logc(Zb, Ch).mean(), cp.gaussian_logc(Zb, cp.block_corr(Ch, [3, 3])).mean()
    info = -0.5 * np.linalg.slogdet(C)[1]               # expected log c of a Gaussian copula under its own C
    assert abs(g - info) < 0.03 and g > blocks + 0.01, (g, info, blocks)


def test_sampler_plane_residuals_and_profiles():
    rng = np.random.default_rng(9)
    n = 3000
    X = np.tile(atm.plane_depths([80.0], 24), (n, 1))
    xmax, L, R, lnn = rng.uniform(300, 500, n), rng.uniform(150, 250, n), rng.uniform(-0.1, 0.2, n), rng.uniform(10, 12, n)
    curve = lnn[:, None] + longitudinal.shape(X, xmax[:, None], L[:, None], R[:, None])
    bias, sd, rho = 0.05 * np.sin(np.arange(24)), 0.1, 0.6
    noisy = curve + bias + sd * sampler.ar1(n, 24, rho, rng)
    d = {"X": X, **{s: {"E": np.exp(noisy), "ln_n": lnn, "theta": np.column_stack([xmax, L, R])}
                    for s in sources.SPECIES_NAMES}}
    p = sampler.plane_residuals(d, np.arange(n))["electron"]
    assert np.max(np.abs(p["bias"] - bias)) < 0.01 and np.max(np.abs(p["sd"] - sd)) < 0.01 and abs(p["rho"] - rho) < 0.03, p
    le = rng.uniform(5, 6, n)
    Zlat = np.tile(np.column_stack([lnn / np.log(10) - le, xmax, np.log(L), R, np.zeros(n)]), (1, 3))
    prof = sampler.to_profiles(Zlat, le, X)
    assert np.allclose(np.nan_to_num(prof["muon"], nan=-1.0), np.nan_to_num(np.where(np.isfinite(curve), curve, np.nan), nan=-1.0))
    # a start depth between planes 3 and 4 empties planes 1-3 and nothing else
    Zlat[:, 2 * sampler.BLOCK - 1] = 0.5 * (X[:, 2] + X[:, 3])
    late = sampler.to_profiles(Zlat, le, X)["muon"]
    assert np.all(np.isnan(late[:, :3])) and np.allclose(np.nan_to_num(late[:, 3:], nan=-1.0), np.nan_to_num(prof["muon"][:, 3:], nan=-1.0))
    # a curve with R < 0 ends at xmax + L/|R|: flagged only when that end falls before the last plane
    z = np.tile(np.array([[-0.3, 400.0, np.log(200.0), 0.1, 0.0]]), (3, 3))
    z[1, 3], z[2, 3] = -0.5, -0.1
    assert sampler.ends_inside(z, np.full(3, 5.5), X[:3]).tolist() == [False, True, False]


def test_tables_stage_fits_total_energies_and_the_sampler_loads_them():
    import argparse
    import logging
    rng = np.random.default_rng(5)
    n = 12
    with tempfile.TemporaryDirectory() as d:
        names, truth = [], {}
        for i in range(n):
            cls, le = ("EM", "hadronic")[i % 2], 5.2 + 0.15 * i
            zen = np.radians(rng.uniform(65, 85))
            direction = np.array([np.sin(zen), 0.0, np.cos(zen)]) * 2.0                 # not unit on purpose
            X = atm.plane_depths([np.degrees(zen)], 24)[0]
            E = {s: np.exp(11 + j + longitudinal.shape(X, 450.0 + 20 * j, 200.0, 0.1)) for j, s in enumerate(sources.SPECIES_NAMES)}
            name = f"{cls}_{i}.npz"
            np.savez(os.path.join(d, name), direction=direction, log10E=le, **{"class": np.array(cls)},
                     **{f"{s}.E_total": E[s] for s in sources.SPECIES_NAMES})
            names.append(name)
            truth[name] = (le, cls, np.cos(zen), E, X)
        lst = os.path.join(d, "list.txt")
        open(lst, "w").write("\n".join(names))
        args = argparse.Namespace(out=d, tables=d, list=lst, top=6.5, name="t", workers=1)
        res = fit_profiles.stage_tables(args, logging.getLogger("test_tables"))
        z = np.load(res["fits"])
        kept = sorted(nm for nm in names if truth[nm][0] <= 6.5)
        assert z["meta.names"].tolist() == kept and res["kept"] == len(kept) < n
        assert str(z["plane_energy"]) == "total" and float(z["top"]) == 6.5
        for r, nm in enumerate(kept):
            le, cls, u, E, X = truth[nm]
            assert z["meta.log10_E"][r] == le and z["meta.labels"][r] == fit_profiles.LABEL[cls]
            assert np.isclose(z["meta.cos_zenith"][r], u) and np.allclose(z["X"][r], X)
            for s in sources.SPECIES_NAMES:
                assert np.array_equal(z[f"{s}.E"][r], E[s])
        for s in sources.SPECIES_NAMES:
            ref = longitudinal.fit_block((z["X"], longitudinal.log_energy(z[f"{s}.E"])))
            assert all(np.array_equal(z[f"{s}.{k}"], v, equal_nan=True) for k, v in ref.items())
        tr = sampler.load(res["fits"], e_max=6.0)
        assert len(tr["meta"]["log10_E"]) == (z["meta.log10_E"] <= 6.0).sum()
        try:
            fit_profiles.stage_tables(args, logging.getLogger("test_tables"))
        except ValueError as exc:
            assert "refusing" in str(exc)
        else:
            raise AssertionError("an existing fits file was overwritten")


def test_peak_noise_keeps_early_scatter_away_from_the_peak():
    rng = np.random.default_rng(21)
    n = 4000
    X = np.tile(atm.plane_depths([80.0], 24), (n, 1))
    xmax, L, R, lnn = rng.uniform(300, 1500, n), np.full(n, 200.0), np.zeros(n), rng.uniform(10, 12, n)
    curve = lnn[:, None] + longitudinal.shape(X, xmax[:, None], L[:, None], R[:, None])
    t = (X - xmax[:, None]) / L[:, None]
    sd_true = np.where(t < -1, 2.0, 0.05)                                        # wild before the peak, tight at it
    d = {"X": X, **{s: {"E": np.exp(curve + sd_true * rng.normal(size=curve.shape)), "ln_n": lnn,
                        "theta": np.column_stack([xmax, L, R])} for s in sources.SPECIES_NAMES}}
    peak = sampler.plane_residuals(d, np.arange(n), "peak")["photon"]
    plane = sampler.plane_residuals(d, np.arange(n), "plane")["photon"]
    assert np.all(np.diff(peak["at"]) > 0), peak["at"]
    near = (peak["at"] > -0.5) & (peak["at"] < 1.0)
    early = peak["at"] < -1.5
    assert near.any() and np.all(np.abs(peak["sd_t"][near] - 0.05) < 0.01), peak
    assert early.any() and np.all(np.abs(peak["sd_t"][early] - 2.0) < 0.2), peak
    k = int(np.argmin(np.abs(X[0] - 900.0)))                                     # a plane at the peak for some, early for others
    assert plane["sd"][k] > 0.5 and np.all(np.abs(sampler.at_peak(peak, np.array([0.0, 0.5]))[1] - 0.05) < 0.01)
    try:
        sampler.plane_residuals(d, np.arange(n), "other")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown noise model was accepted")


def test_skew_noise_keeps_the_energy_of_low_skewed_leftovers():
    rng = np.random.default_rng(23)
    n = 4000
    X = np.tile(atm.plane_depths([80.0], 24), (n, 1))
    xmax, L, R, lnn = rng.uniform(300, 1500, n), np.full(n, 200.0), np.zeros(n), rng.uniform(10, 12, n)
    curve = lnn[:, None] + longitudinal.shape(X, xmax[:, None], L[:, None], R[:, None])
    r = 1.0 - rng.exponential(1.0, curve.shape)                                  # mean 0, sd 1, skewed low
    d = {"X": X, **{s: {"E": np.exp(curve + r), "ln_n": lnn, "theta": np.column_stack([xmax, L, R])}
                    for s in sources.SPECIES_NAMES}}
    t = sampler.peak_position(X, xmax, L)
    ok = np.isfinite(curve)
    true = np.exp(r[ok]).mean()                                                  # e / 2
    skew = sampler.plane_residuals(d, np.arange(n), "skew")["photon"]
    peak = sampler.plane_residuals(d, np.arange(n), "peak")["photon"]
    z = rng.normal(size=t.shape)
    e_skew = np.exp(sampler.skew_noise(skew, t, z)[ok]).mean()
    bias, sd = sampler.at_peak(peak, t)
    e_peak = np.exp((bias + sd * z)[ok]).mean()
    assert abs(e_skew / true - 1) < 0.05 and e_peak / true > 1.1, (e_skew / true, e_peak / true)
    at = skew["at"][1:-1]                                                        # a bin's own median t: its law alone
    tt, rr = np.repeat(at[:, None], 200, 1), np.tile(np.linspace(-3.0, 0.9, 200), (len(at), 1))
    back = sampler.skew_noise(skew, tt, sampler.skew_scores(skew, tt, rr))
    assert np.allclose(back, rr, atol=1e-6), np.max(np.abs(back - rr))
    assert abs(skew["rho"]) < 0.05                                                # independent leftovers


def test_table_noise_keeps_energy_and_stays_within_the_leftovers_seen():
    rng = np.random.default_rng(24)
    n = 4000
    X = np.tile(atm.plane_depths([80.0], 24), (n, 1))
    xmax, L, R, lnn = rng.uniform(300, 1500, n), np.full(n, 200.0), np.zeros(n), rng.uniform(10, 12, n)
    curve = lnn[:, None] + longitudinal.shape(X, xmax[:, None], L[:, None], R[:, None])
    t = sampler.peak_position(X, xmax, L)
    r = np.where(t < -1, 3.0 * rng.standard_t(3, curve.shape), 1.0 - rng.exponential(1.0, curve.shape))  # heavy early
    d = {"X": X, **{s: {"E": np.exp(curve + r), "ln_n": lnn, "theta": np.column_stack([xmax, L, R])}
                    for s in sources.SPECIES_NAMES}}
    ok = np.isfinite(curve)
    p = sampler.plane_residuals(d, np.arange(n), "table")["photon"]
    assert p["noise"] == "table" and p["q"].shape == (len(sampler.PEAK_EDGES) - 1, len(sampler.Z_GRID))
    z = rng.normal(size=t.shape) * 1.5                                         # wide on purpose: the clip must hold
    noise = sampler.table_noise(p, t, z)
    assert noise[ok].max() <= r[ok].max() + 1e-12 and noise[ok].min() >= r[ok].min() - 1e-12
    near = ok & (t > -0.5) & (t < 1.0)
    e_true = np.exp(r[near]).mean()
    e_table = np.exp(sampler.table_noise(p, t, rng.normal(size=t.shape))[near]).mean()
    assert abs(e_table / e_true - 1) < 0.05, e_table / e_true
    at = p["at"][1:-1]
    tt, rr = np.repeat(at[:, None], 200, 1), np.tile(np.linspace(-2.0, 0.9, 200), (len(at), 1))
    back = sampler.table_noise(p, tt, sampler.table_scores(p, tt, rr))
    inside = (rr >= p["q"][1:-1, :1]) & (rr <= p["q"][1:-1, -1:])
    assert np.allclose(back[inside], rr[inside], atol=1e-9), np.max(np.abs(back[inside] - rr[inside]))


def test_bright_noise_recovers_the_brightness_slope_and_keeps_energy_per_bin():
    rng = np.random.default_rng(31)
    n = 6000
    X = np.tile(atm.plane_depths([80.0], 24), (n, 1))
    xmax, L, R = rng.uniform(300, 1500, n), np.full(n, 200.0), np.zeros(n)
    l10 = rng.uniform(5.0, 6.0, n)
    lnn = np.log(10.0) * l10 + rng.normal(0, 0.5, n)                           # curves bright or dim for their energy
    curve = lnn[:, None] + longitudinal.shape(X, xmax[:, None], L[:, None], R[:, None])
    v = curve - np.log(10.0) * l10[:, None]
    r = -0.6 * (v - np.nanmean(v)) + 0.2 * rng.normal(size=curve.shape)       # planes pull back toward the mean
    d = {"X": X, "meta": {"log10_E": l10},
         **{s: {"E": np.exp(curve + r), "ln_n": lnn, "theta": np.column_stack([xmax, L, R])} for s in sources.SPECIES_NAMES}}
    t = sampler.peak_position(X, xmax, L)
    p = sampler.plane_residuals(d, np.arange(n), "bright")["photon"]
    near = np.isfinite(curve) & (t > -1.0) & (t < 2.0)
    assert np.all(np.abs(p["ab"][3:9, 1] + 0.6) < 0.05), p["ab"][:, 1]
    tf = np.where(np.isfinite(t), t, 0.0)
    drawn = curve + sampler.bright_mean(p, tf, np.where(np.isfinite(v), v, 0.0)) + sampler.table_noise(p, tf, rng.normal(size=t.shape))
    ratio = np.exp(drawn[near]).sum() / np.exp(curve[near] + r[near]).sum()
    tab = sampler.plane_residuals(d, np.arange(n), "table")["photon"]
    plain = np.exp(curve[near] + sampler.table_noise(tab, tf, rng.normal(size=t.shape))[near]).sum() / np.exp(curve[near] + r[near]).sum()
    assert abs(ratio - 1) < 0.05 and plain > ratio + 0.05, (ratio, plain)


def test_peak_and_plane_noise_draw_the_same_curve_without_noise():
    rng = np.random.default_rng(22)
    n = 1500
    X = np.tile(atm.plane_depths([80.0], 24), (n, 1))
    le = rng.uniform(5.0, 6.0, n)
    xmax, L, R, lnn = rng.uniform(300, 500, n), rng.uniform(150, 250, n), rng.uniform(0.0, 0.2, n), rng.uniform(10, 12, n)
    curve = lnn[:, None] + longitudinal.shape(X, xmax[:, None], L[:, None], R[:, None])
    f = {"ln_n": lnn, "theta": np.column_stack([xmax, L, R])}
    d = {"X": X, "meta": {"log10_E": le, "cos_zenith": rng.uniform(0.1, 0.5, n), "labels": np.zeros(n, int)},
         **{s: {"E": np.exp(curve + 0.1 * rng.normal(size=curve.shape)), **f, "Z": longitudinal.to_latent(f, le),
                "start": rng.uniform(0, 50, n), "clean": np.ones(n, bool)} for s in sources.SPECIES_NAMES}}
    plane, peak = sampler.fit(d, 0, top=np.inf), sampler.fit(d, 0, top=np.inf, noise="peak")
    assert all(peak["planes"][s]["noise"] == "peak" and "noise" not in plane["planes"][s] for s in sources.SPECIES_NAMES)
    for s in sources.SPECIES_NAMES:
        plane["planes"][s].update(bias=np.zeros(24), sd=np.zeros(24))
        peak["planes"][s].update(bias_t=np.zeros_like(peak["planes"][s]["at"]), sd_t=np.zeros_like(peak["planes"][s]["at"]))
    peak.update(design=plane["design"], margins=plane["margins"], C=plane["C"])  # same profile numbers drawn
    a, _ = sampler.sample(plane, le[:200], np.full(200, 0.3), X[:200], np.random.default_rng(5))
    b, _ = sampler.sample(peak, le[:200], np.full(200, 0.3), X[:200], np.random.default_rng(5))
    assert all(np.array_equal(a[s], b[s]) for s in sources.SPECIES_NAMES)


def test_sampler_fit_uses_only_showers_below_the_cut():
    rng = np.random.default_rng(13)
    n = 1500
    X = np.tile(atm.plane_depths([80.0], 24), (n, 1))
    le = rng.uniform(5.0, 6.2, n)
    xmax, L, R, lnn = rng.uniform(300, 500, n), rng.uniform(150, 250, n), rng.uniform(-0.1, 0.2, n), rng.uniform(10, 12, n)
    curve = lnn[:, None] + longitudinal.shape(X, xmax[:, None], L[:, None], R[:, None])
    d = {"X": X, "meta": {"log10_E": le, "cos_zenith": rng.uniform(-1, 1, n), "labels": np.zeros(n, int)},
         **{s: {"E": np.exp(curve + 0.1 * rng.normal(size=curve.shape)), "ln_n": lnn,
                "theta": np.column_stack([xmax, L, R]), "Z": rng.normal(size=(n, 4)), "start": rng.uniform(0, 50, n),
                "clean": np.ones(n, bool)} for s in sources.SPECIES_NAMES}}
    below = le < 5.9
    m = sampler.fit(d, 0, top=5.9)
    assert m["n_train"] == below.sum() and np.isclose(m["design"].m[0], le[below].mean())
    assert sampler.fit(d, 0)["n_train"] == (le < sampler.TOP).sum()


def test_start_depth_lies_between_the_last_leading_empty_plane_and_the_first_lit_one():
    X = np.tile(np.arange(1, 7) * 50.0, (4, 1))
    E = np.array([[5.0, 9, 9, 9, 9, 9], [0.0, 0.5, 9, 9, 9, 9], [0.0, 0, 0, 0, 0, 2], [0.0, 0, 0, 0, 0, 0]])
    st = longitudinal.start_depth(E, X, np.random.default_rng(0))
    assert 0 <= st[0] < 50 and 100 <= st[1] < 150 and 250 <= st[2] < 300 and np.isnan(st[3]), st


def test_energy_score_is_zero_for_a_point_mass_and_prefers_the_true_law():
    rng = np.random.default_rng(11)
    y = rng.normal(size=(2000, 5))
    assert np.allclose(score.energy_score(np.repeat(y[:, None], 4, 1), y), 0.0)
    true = rng.normal(size=(2000, 8, 5))
    wide = 1.6 * rng.normal(size=(2000, 8, 5))
    shifted = rng.normal(size=(2000, 8, 5)) + 0.5
    es = {k: score.energy_score(v, y).mean() for k, v in (("true", true), ("wide", wide), ("shifted", shifted))}
    assert es["true"] < es["wide"] and es["true"] < es["shifted"], es


UNWRAP_DIR = np.array([0.3, -0.9, 0.3]) / np.linalg.norm([0.3, -0.9, 0.3])


def wrap_plane(rng, k=6, n=4000):
    """Cells of one synthetic plane about its axis point (core 30 m, halo 150 m) plus a stray cell far South."""
    axis = uw.axis_point(k, UNWRAP_DIR)
    xy = axis + np.concatenate([rng.normal(0, 30, (n, 2)), rng.normal(0, 150, (n // 4, 2))])
    e = np.concatenate([rng.gamma(2.0, 5.0, n), rng.gamma(1.0, 0.5, n // 4)])
    idx, inv = np.unique(np.floor(xy / uw.CELL).astype(np.int64), axis=0, return_inverse=True)
    cells, w = idx * uw.CELL + uw.CELL / 2, np.bincount(inv.ravel(), e)
    stride = int(np.ptp(idx[:, 0])) + 2
    stray = np.floor(axis / uw.CELL) * uw.CELL + uw.CELL / 2 - [0.0, 1.5 * stride * uw.CELL]
    return np.vstack([cells, stray]), np.append(w, 0.01), axis, stride


def grid_overlap(xy_a, e_a, xy_b, e_b):
    """Energy share two clouds put on the same cells."""
    keys = [np.rint((p - uw.CELL / 2) / uw.CELL).astype(np.int64) for p in (xy_a, xy_b)]
    ka, ia = np.unique(keys[0], axis=0, return_inverse=True)
    kb, ib = np.unique(keys[1], axis=0, return_inverse=True)
    da = dict(zip(map(tuple, ka), np.bincount(ia.ravel(), e_a)))
    db = dict(zip(map(tuple, kb), np.bincount(ib.ravel(), e_b)))
    return sum(min(v, db.get(q, 0.0)) for q, v in da.items()) / sum(e_a)


def test_unwrap_restores_a_plane_whose_core_overflowed():
    rng = np.random.default_rng(5)
    xy, e, axis, stride = wrap_plane(rng)
    bad, w = uw.encode(xy, e, stride)
    assert np.isclose(w.sum(), e.sum()) and uw.moved(bad, w, axis) and not uw.moved(xy, e, axis)
    fixed, s = uw.unwrap_plane(bad, w, axis, UNWRAP_DIR)
    ov = grid_overlap(xy, e, fixed, w)
    print(f"      stride {stride} found {s}; energy on its own cell {ov:.4f}; near axis {uw.near_share(fixed, w, axis):.3f}"
          f" (truth {uw.near_share(xy, e, axis):.3f})")
    assert s == stride and ov > 0.98
    # a stride wider than the y range leaves the plane as it was, and the repair leaves it alone
    same, w2 = uw.encode(xy, e, int(np.ptp(np.floor(xy[:, 1] / uw.CELL))) + 2)
    assert grid_overlap(xy, e, same, w2) > 1 - 1e-12
    kept, s2 = uw.unwrap_plane(same, w2, axis, UNWRAP_DIR)
    assert s2 == 0 and np.array_equal(kept, same)
    try:
        uw.encode(xy, e, stride - 1)
        raise AssertionError("encode accepted a stride below the x range + 2")
    except ValueError:
        pass


def test_unwrap_shower_and_restore_agree():
    rng = np.random.default_rng(6)
    layers = []
    for k in (3, 6):
        xy, e, axis, stride = wrap_plane(rng, k=k, n=1500)
        bad, w = uw.encode(xy, e, stride)
        layers.append(np.column_stack([bad, np.full(len(w), k), w, np.zeros(len(w))]))
    cloud = np.vstack(layers + [np.zeros((5, 5))])
    fixed, s_k = uw.unwrap_shower(cloud, UNWRAP_DIR, 24)
    print(f"      strides per plane {s_k[[3, 6]].tolist()}")
    assert np.count_nonzero(s_k) == 2 and np.array_equal(fixed[:, 2:], cloud[:, 2:])
    assert np.array_equal(uw.restore(cloud, UNWRAP_DIR, s_k), fixed)




if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:
                failed += 1
                print(f"FAIL  {name}: {exc!r}")
    sys.exit(1 if failed else 0)
