"""CPU self-test of the symbolic pieces: slant depth, profile fits, structure metrics, plane sums, the latent, the
sampler (margins, copula, plane residuals) and the energy score."""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import h5py
import numpy as np
import torch

from data_lane import sources
from scipy import special

from symbolic import (atmosphere as atm, copula as cp, families as fam, latent, m1, m2, planes, sampler,
                      score, structure, unwrap as uw)
from symbolic import annuli, generator

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


TRUE = {"gh4": ((400.0, -20.0, 70.0), None), "gh4lr": ((400.0, 200.0, 0.35), None),
        "gh4lr_signed": ((400.0, 200.0, -0.2), None), "gh3": ((400.0, -20.0, 70.0), 70.0), "age3": ((400.0, 0.4), None),
        "greisen2": ((12.0,), None), "rise3": ((150.0, 900.0), None)}


def test_families_recover_known_profiles():
    rng = np.random.default_rng(0)
    X = toy_depths()
    for name, (theta, extra) in TRUE.items():
        f = fam.FAMILIES[name]
        y = 3.0 + f.shape(X, *theta) + rng.normal(0, 0.02, len(X))
        ln_n, got, mse, n_fit = fam.fit_one(f, X, y, extra)
        assert n_fit == 24 and mse < 0.02 ** 2 * 2, (name, mse)
        assert abs(f.xmax(*got) / f.xmax(*theta) - 1) < 0.05, (name, f.xmax(*got), f.xmax(*theta))
        scrambled = fam.fit_one(f, X, y[rng.permutation(len(y))], extra)[2]
        assert scrambled > 100 * mse, (name, scrambled, mse)


def test_empty_planes_are_skipped_and_too_few_planes_refused():
    X = toy_depths()
    f = fam.FAMILIES["gh4"]
    y = 3.0 + f.shape(X, *TRUE["gh4"][0])
    y[:3] = np.nan
    assert fam.fit_one(f, X, y)[3] == 21
    y[3:19] = np.nan
    assert fam.fit_one(f, X, y) is None


def population(n, rng, wave=0.0):
    """GH showers whose Xmax and size follow energy and fluctuate; `wave` adds cross-plane structure no profile carries."""
    l, cz = rng.uniform(5.0, 6.2, n), rng.uniform(-0.17, 0.5, n)
    X = atm.plane_depths(np.degrees(np.arccos(cz)), 24)
    xmax = 350 + 80 * (l - 5.5) + rng.normal(0, 40, n)
    shape = fam.FAMILIES["gh3"].shape
    Y = np.stack([2.3 * l[i] + shape(X[i], xmax[i], -20.0, 70.0) for i in range(n)])
    k = np.arange(24)
    Y += wave * np.sin(2 * np.pi * k / 6 + rng.uniform(0, 2 * np.pi, n)[:, None]) + rng.normal(0, 0.03, Y.shape)
    return X, Y, l, cz


def test_structure_passes_low_dim_and_fails_extra_structure():
    rng = np.random.default_rng(1)
    for wave, should_pass in ((0.0, True), (0.4, False)):
        X, Y, l, cz = population(400, rng, wave)
        f = fam.fit_block(("gh3", X, Y, np.full(len(Y), 70.0)))
        r = structure.summarize(Y, fam.evaluate("gh3", f, X), l, cz, 3, rng, n_boot=10)
        print(f"      wave {wave}: planes passing R2 {r['frac_planes_r2_go']:.2f}, cov explained {r['cov_explained'][0]:.3f}")
        assert r["go_r2"] == should_pass and r["go_cov"] == should_pass, (wave, r["frac_planes_r2_go"], r["cov_explained"])


def test_matched_covariance_distance():
    rng = np.random.default_rng(2)
    _, Y, l, cz = population(400, rng)
    same = structure.compare_covariance(Y, Y.copy(), l, cz, rng, n_boot=5)
    assert same["cov_distance"][0] < 1e-12
    wider = structure.compare_covariance(Y, Y + rng.normal(0, 0.5, Y.shape), l, cz, rng, n_boot=5)
    assert wider["cov_distance"][0] > 0.2 and all(v > 1 for v in wider["sd_ratio"] if v is not None)


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


def test_totals_summary():
    r = m1.totals_summary(np.array([100.0, 95.0, 120.0, 50.0]), 100, np.array([0, 0, 1, 1]))
    assert r["all"]["at_or_above_cap"] == 0.5 and r["all"]["above_1.1_cap"] == 0.25 and r["EM"]["below_0.9_cap"] == 0.0
    assert r["hadronic"]["below_0.9_cap"] == 0.5 and r["all"]["within_1pct_of_cap"] == 0.25

def test_m2_parts_cover_rows_and_gate_rejects_gaps():
    n = 1003
    rows = np.concatenate([m2.part_rows(n, 7, i) for i in range(7)])
    m2.check_cover(rows, n)
    for bad in (rows[:-1], np.concatenate([rows, rows[:1]]), np.concatenate([rows[:10], rows[11:], rows[10:11]])):
        try:
            m2.check_cover(bad, n)
        except ValueError:
            continue
        raise AssertionError("check_cover accepted a broken cover")

def test_latent_round_trip_flags_and_gaussian_models():
    rng = np.random.default_rng(3)
    n = 400
    X = np.tile(atm.plane_depths([80.0], 24), (n, 1))
    Y = np.zeros((n, 24))
    xmax, L, lnn, le = rng.uniform(200, 600, n), rng.uniform(100, 300, n), rng.uniform(10, 12, n), rng.uniform(5, 6, n)
    pos, neg = fam.r_caps(xmax, L, X[:, 0], X[:, -1])
    R = rng.uniform(-0.5, 0.5, n) * np.where(rng.uniform(size=n) < 0.5, pos, neg)
    R[:7] = 0.0
    R = np.where(R >= 0, np.minimum(R, 0.5 * pos), np.maximum(R, -0.5 * neg))
    f = {"ln_n": lnn, "theta": np.column_stack([xmax, L, R]), "ln_nmax": lnn}
    back = latent.from_latent(latent.to_latent(f, le), le)
    assert all(np.allclose(a, b) for a, b in zip(back, (xmax, L, R, lnn)))
    fl = latent.flags(f, X, Y)
    assert fl["r_zero"][:7].all() and not fl["r_zero"][7:].any() and not fl["bound"].any() and not fl["unfit"].any()
    f["theta"][7, 2], f["theta"][8, 2] = 0.999 * pos[7], -0.999 * neg[8]
    f["theta"][9, 0] = X[0, -1] + 100.0
    f["theta"][10, 0] = X[0, 0] - 50.0
    fl = latent.flags(f, X, Y)
    assert fl["bound"][7] and fl["bound"][8]
    assert fl["after"][9] and not fl["before"][9] and fl["before"][10] and not fl["after"][10]
    assert fl["outside"][9] and fl["outside"][10] and fl["outside"].sum() == 2
    # correlated Gaussian residuals: the full covariance must beat the diagonal one held out; independent: a tie
    C = np.array([[1.0, 0.8], [0.8, 1.0]])
    A, B = rng.multivariate_normal([0, 0], C, 4000), rng.multivariate_normal([0, 0], C, 4000)
    full, diag = np.cov(A.T), np.diag(np.diag(np.cov(A.T)))
    assert latent.gaussian_loglik(B, full) > latent.gaussian_loglik(B, diag) + 0.3
    I, J = rng.normal(size=(4000, 2)), rng.normal(size=(4000, 2))
    assert abs(latent.gaussian_loglik(J, np.cov(I.T)) - latent.gaussian_loglik(J, np.diag(np.diag(np.cov(I.T))))) < 0.01
    assert np.allclose(latent.block_diagonal(np.ones((12, 12)), [4, 4, 4]).sum(), 48)


def test_signed_r_fits_both_skews_where_gh4lr_cannot():
    X = toy_depths()
    for R in (0.25, -0.15):
        y = 3.0 + fam._gh_lr_shape(X, 450.0, 220.0, R)
        th = fam.fit_one(fam.FAMILIES["gh4lr_signed"], X, y)[1]
        assert abs(th[2] - R) < 0.01 and abs(th[0] - 450.0) < 2, (R, th)
    # with R >= 0 forced, a profile skewed the other way sits at R = 0 with a worse fit
    lr = fam.fit_one(fam.FAMILIES["gh4lr"], X, y)
    assert lr[1][2] < 1e-6 and lr[2] > 100 * fam.fit_one(fam.FAMILIES["gh4lr_signed"], X, y)[2]


def test_gh4lr_is_gh4_reparameterized_with_a_gaussian_limit():
    X = toy_depths()
    xmax, x0, lam = 400.0, -20.0, 70.0
    L, R = np.sqrt(lam * (xmax - x0)), np.sqrt(lam / (xmax - x0))
    assert np.allclose(fam._gh_lr_shape(X, xmax, L, R), fam._gh_shape(X, xmax, x0, lam), atol=1e-9)
    z = (X - xmax) / 200.0
    for r in (0.0, 1e-7, 1e-5):
        assert np.allclose(fam._gh_lr_shape(X, xmax, 200.0, r), -0.5 * z * z + r * z ** 3 / 3, atol=1e-6)
    # a symmetric profile is recovered with R near 0 and no bound hit, where gh4 needs its start depth floor
    y = 3.0 + fam._gh_lr_shape(X, xmax, 250.0, 0.0)
    ln_n, th, mse, n_fit = fam.fit_one(fam.FAMILIES["gh4lr"], X, y)
    assert mse < 1e-8 and abs(th[0] - xmax) < 1 and abs(th[1] - 250.0) < 2 and th[2] < 0.01, (th, mse)


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
    # the normal score is exact in the tails and inverts through ppf
    z = jsu.score(D[te], x[te])
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05
    assert np.allclose(jsu.ppf(D[te], special.ndtr(z)), x[te], atol=1e-6, rtol=1e-6)


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
    curve = lnn[:, None] + fam.FAMILIES[m2.NAME].shape(X, xmax[:, None], L[:, None], R[:, None])
    bias, sd, rho = 0.05 * np.sin(np.arange(24)), 0.1, 0.6
    noisy = curve + bias + sd * sampler.ar1(n, 24, rho, rng)
    fits = {"X": X}
    for s in sources.SPECIES_NAMES:
        fits.update({f"{s}.E": np.exp(noisy), f"{s}.{m2.NAME}.ln_n": lnn,
                     f"{s}.{m2.NAME}.theta": np.column_stack([xmax, L, R])})
    p = sampler.plane_residuals(fits, np.arange(n))["electron"]
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



def test_start_depth_lies_between_the_last_leading_empty_plane_and_the_first_lit_one():
    X = np.tile(np.arange(1, 7) * 50.0, (4, 1))
    E = np.array([[5.0, 9, 9, 9, 9, 9], [0.0, 0.5, 9, 9, 9, 9], [0.0, 0, 0, 0, 0, 2], [0.0, 0, 0, 0, 0, 0]])
    st = latent.start_depth(E, X, np.random.default_rng(0))
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
    planes = []
    for k in (3, 6):
        xy, e, axis, stride = wrap_plane(rng, k=k, n=1500)
        bad, w = uw.encode(xy, e, stride)
        planes.append(np.column_stack([bad, np.full(len(w), k), w, np.zeros(len(w))]))
    cloud = np.vstack(planes + [np.zeros((5, 5))])
    fixed, s_k = uw.unwrap_shower(cloud, UNWRAP_DIR, 24)
    print(f"      strides per plane {s_k[[3, 6]].tolist()}")
    assert np.count_nonzero(s_k) == 2 and np.array_equal(fixed[:, 2:], cloud[:, 2:])
    assert np.array_equal(uw.restore(cloud, UNWRAP_DIR, s_k), fixed)



def test_annulus_nkg_fit_recovers_its_own_profile():
    r_ref = 120.0
    for ln_r50, s, r_hi in ((np.log(0.04), 1.8, 3000.0), (np.log(0.5), 1.2, 450.0), (np.log(2.0), 0.7, 9500.0)):
        n = int(np.searchsorted(annuli.EDGES, r_hi))
        edges = np.minimum(annuli.EDGES, r_hi)[:n + 1]
        w = np.zeros(annuli.N_BIN)
        w[:n] = 1e4 * annuli.nkg_shares(edges, ln_r50, s, r_ref)
        got = annuli.fit_profile(w, r_hi, r_ref)
        print(f"      truth ({ln_r50:.3f}, {s}) fitted ({got[0]:.3f}, {got[1]:.3f})")
        assert got[2] and abs(got[0] - ln_r50) < 0.02 and abs(got[1] - s) < 0.01



def test_pattern_fit_recovers_a_planted_pattern_and_its_correlation():
    rng = np.random.default_rng(12)
    n_sh, n_pl, cols = 2000, 10, 20
    g_true = np.linspace(-1.0, 2.0, cols)
    a = sampler.ar1(n_sh, n_pl, 0.8, rng).ravel()
    unit, plane = np.repeat(np.arange(n_sh), n_pl), np.tile(np.arange(n_pl), n_sh)
    w = rng.uniform(1, 50, size=(len(a), cols)) * (rng.uniform(size=(len(a), cols)) < 0.6)   # cell counts, 40% empty
    r = np.where(w > 0, a[:, None] * g_true + 0.3 * rng.normal(size=w.shape), 0.0)
    fit = unit % 5 != 0
    g = generator.fit_pattern(r[fit], w[fit], rng)
    score = generator.pattern_scores(r[fit], w[fit], g)
    cos = abs(g @ g_true) / np.linalg.norm(g) / np.linalg.norm(g_true)
    rho = generator.lag1(score, unit[fit], plane[fit])
    print(f"      cos to truth {cos:.4f}, |g| / |truth| {np.linalg.norm(g) / np.linalg.norm(g_true):.3f}, rho {rho:.3f}, "
          f"score sd {score.std():.3f}, held-out removed {generator.removed(r[~fit], w[~fit], g):.3f}")
    assert cos > 0.999 and abs(rho - 0.8) < 0.03 and abs(score.std() - 1) < 1e-6


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

