"""CPU self-test of part 1, the draw, the judge and the corpus bookkeeping on toy showers."""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import h5py
import numpy as np

from modules.constants import N_PLANES, SPECIES_NAMES
from symbolic import atmosphere, corpus, data, draw, judge, lateral, longitudinal, rows, sampler

N_CELLS = 30


def toy_fits(rng, n=240):
    """A profiles file of showers made from known curves plus noise, half EM, half hadronic."""
    l, cz = rng.uniform(5.0, 6.5, n), rng.uniform(0.05, 0.4, n)
    X = atmosphere.plane_depths(np.degrees(np.arccos(cz)), N_PLANES)
    out = {"meta.names": np.array([f"t{i}" for i in range(n)]), "meta.log10_E": l, "meta.cos_zenith": cz,
           "meta.labels": np.arange(n) % 2, "X": X}
    for s in SPECIES_NAMES:
        xmax = rng.normal(500, 60, n)
        y = (l * np.log(10) - 1.0)[:, None] + longitudinal.shape(X, xmax[:, None], rng.uniform(200, 300, n)[:, None], 0.2)
        E = np.where(np.isfinite(y), np.exp(y + rng.normal(0, 0.1, y.shape)), 0.0)
        fit = longitudinal.fit_block((X, longitudinal.log_energy(E)))
        out.update({f"{s}.E": E, f"{s}.ln_n": fit["ln_n"], f"{s}.theta": fit["theta"]})
    return out


def test_part1_survives_the_model_file_and_draws_curves_that_never_end_inside():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        np.savez(os.path.join(tmp, "fits.npz"), **toy_fits(rng))
        d = sampler.load(os.path.join(tmp, "fits.npz"))
    m = sampler.fit(d, 1)
    back = sampler.from_arrays(sampler.to_arrays(m, "hadronic"), "hadronic")
    l, u = np.full(50, 6.0), np.full(50, 0.2)
    X = atmosphere.plane_depths(np.degrees(np.arccos(u)), N_PLANES)
    a, Za, _ = sampler.sample(m, l, u, X, np.random.default_rng(1))
    b, Zb, _ = sampler.sample(back, l, u, X, np.random.default_rng(1))
    assert np.array_equal(Za, Zb) and all(np.array_equal(a[s], b[s]) for s in SPECIES_NAMES)
    assert not sampler.ends_inside(Za, l, X).any()
    true_peak = np.median(d["electron"]["E"][d["meta"]["labels"] == 1].max(1) / 10 ** d["meta"]["log10_E"][d["meta"]["labels"] == 1])
    assert abs(np.log10(np.median(a["electron"].max(1)) / 1e6 / true_peak)) < 0.3


def test_table_noise_reads_each_bins_quantiles_at_its_median():
    p = {"at": np.array([-1.0, 0.0, 2.0]), "q": np.stack([np.linspace(-1, 1, len(sampler.Z_GRID)) * w for w in (1, 2, 3)])}
    z = sampler.Z_GRID[::20]
    for k in range(3):
        assert np.allclose(sampler.table_noise(p, np.full(len(z), p["at"][k]), z), p["q"][k][::20])


def toy_laws():
    """Every in-plane law of one group: equal ring shares, width 1, ln mean energy 0 (1 GeV), Burr (2.5, 0.9)."""
    theta = np.zeros(9 * 3 + 1)
    return {"rings": np.zeros((8, data.N_RING)), "width": np.array([1.0, 1.0, 0, 0, 0, 0, -5.0, 0, 0]),
            "lead": np.array([-4.0, 1.6, 0.7]), "delay": np.array([-3.0, 1.6, 0.7]), "energy": theta,
            "spread": np.tile([2.5, 0.9], (data.N_RING, 1))}


def test_ring_particles_keep_ring_energy_and_land_in_their_rings():
    rng = np.random.default_rng(2)
    e_ring = np.r_[np.zeros(3), np.full(data.N_RING - 3, 50.0)]
    lead, tau = np.full(data.N_RING, 5.0), np.full(data.N_RING, 20.0)
    p = draw.ring_particles(rng, 4, e_ring, np.full(data.N_RING, 1.0), np.tile([2.5, 0.9], (data.N_RING, 1)), 0.01, lead, tau)
    j = np.searchsorted(data.EDGES, np.hypot(p[:, 1], p[:, 2]), side="right") - 1
    assert np.allclose(np.bincount(j, p[:, 3], data.N_RING), e_ring) and p[:, 3].min() >= 0.01
    t0 = 5 * atmosphere.PLANE_SPACING / rows.C_LIGHT
    assert (p[:, 4] >= t0 + 5e-9 - 1e-15).all() and abs((p[:, 4] - t0).mean() - 25e-9) < 2e-9


def test_a_shower_stores_at_most_what_it_drew():
    rng = np.random.default_rng(3)
    laws = {(s, cn): toy_laws() for s in SPECIES_NAMES for cn in data.CLASS.values()}
    E = {s: np.r_[np.zeros(4), np.full(6, 2e4), np.zeros(N_PLANES - 10)] for s in SPECIES_NAMES}
    X = atmosphere.plane_depths([80.0], N_PLANES)[0]
    d = np.array([0.6, 0.77, np.cos(np.radians(80.0))])
    i, cells, _, n = draw.shower((0, 0, 6.0, d[2], "EM", X, d / np.linalg.norm(d), E, {s: 400.0 for s in SPECIES_NAMES}, laws))
    for s in SPECIES_NAMES:
        k = cells[s][:, 2].astype(int)
        assert 0 < len(cells[s]) <= data.CAP[s] and set(k) <= set(range(4, 10))
        assert (np.bincount(k, cells[s][:, 3], N_PLANES) <= E[s] * (1 + 1e-9)).all()


def toy_dirs(rng, n):
    zen, az = np.radians(rng.uniform(60, 89, n)), rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([np.sin(zen) * np.cos(az), np.sin(zen) * np.sin(az), np.cos(zen)])


def toy_clouds(rng, dirs, shift=0.0):
    """Cells about the axis: across-track Gaussian (40 m), energy peaking mid-shower, time growing with offset."""
    out = []
    for d in dirs:
        h, perp, c = lateral.frame(d)
        cells = []
        for k in range(N_PLANES):
            a, b = rng.normal(shift, 40.0, N_CELLS), rng.normal(0, 40.0, N_CELLS)
            xy = (k + 1) * atmosphere.PLANE_SPACING * d[:2] / np.linalg.norm(d) + a[:, None] * perp + (b * c)[:, None] * h
            e = rng.gamma(2.0, 1.0 + 5 * np.exp(-0.5 * ((k - 10) / 4) ** 2), N_CELLS)
            t = (k + 1) * atmosphere.PLANE_SPACING / rows.C_LIGHT + np.abs(a) * 1e-10 + rng.exponential(2e-9, N_CELLS)
            cells.append(np.column_stack([xy, np.full(N_CELLS, k), e, t]))
        out.append(np.concatenate(cells))
    return out


def test_judge_scores_a_good_source_at_its_floor_and_a_shifted_one_loses_clearly():
    rng = np.random.default_rng(4)
    n, m = 12, 4
    dirs = toy_dirs(rng, n)
    r = np.repeat(np.arange(n), m)
    target = judge.observables(toy_clouds(rng, dirs), dirs, np.arange(n))
    good = judge.observables(toy_clouds(rng, dirs[r]), dirs, r)
    bad = judge.observables(toy_clouds(rng, dirs[r], shift=150.0), dirs, r)
    res = judge.score(target["abs_t1"], good["abs_t1"], bad["abs_t1"], np.arange(n), rng, "other", n_boot=50)
    assert res["symbolic"] <= 3 * res["floor symbolic"] and res["other"] >= 5 * res["floor other"], res
    assert judge.verdict(res, "abs_t1", "other") == "PASS clear" and judge.verdict(res, "lead_time", "other") == "reported"
    swapped = judge.score(target["abs_t1"], bad["abs_t1"], good["abs_t1"], np.arange(n), rng, "other", n_boot=50)
    assert judge.verdict(swapped, "abs_t1", "other") == "FAIL clear"
    v = np.arange(24.0)
    one, zero = np.ones(24), np.zeros(24, int)
    a, b = (v, one, zero, zero, np.arange(24)), (v, one, zero, zero, np.arange(24)[::-1])
    assert judge.dist(a, b) == 0.0 and judge.per_plane(a, b) > 0                # per plane sees plane order


def toy_h5(sim, cells, total, d, cap=None):
    os.makedirs(sim, exist_ok=True)
    for s in SPECIES_NAMES:
        with h5py.File(os.path.join(sim, f"x_{data.H5_NAME[s]}.h5"), "w") as f:
            f.attrs["nmax"] = cap or data.CAP[s]
            v = f.create_dataset("showers", (1,), dtype=h5py.vlen_dtype(np.float32))
            v[0] = cells.astype(np.float32).ravel()
            f["num_points"] = np.array([len(cells)])
            f["directions"] = np.array([d], np.float32)
            f.create_group("plane_stats")["energy"] = total.astype(np.float32)


def test_stored_cells_are_read_and_inconsistent_files_refused():
    d = np.array([0.0, 0.6, 0.8])
    cells = np.array([[10.0, 20.0, 3, 5.0, 1e-6], [0.0, 0.0, 3, 2.0, 1e-6], [5.0, 5.0, 7, 1.0, 2e-6]])
    total = np.zeros(N_PLANES)
    total[3], total[7] = 8.0, 1.0
    with tempfile.TemporaryDirectory() as tmp:
        toy_h5(os.path.join(tmp, "a"), cells, total, d)
        out = data.stored_cells(os.path.join(tmp, "a"), d)
        assert all(np.allclose(out[s], cells, atol=1e-6) for s in SPECIES_NAMES)
        for i, bad in enumerate((dict(total=total * 0.5), dict(cap=4096), dict(d=np.array([0.6, 0.0, 0.8])))):
            sim = os.path.join(tmp, f"b{i}")
            toy_h5(sim, **{**dict(cells=cells, total=total, d=d), **bad})
            try:
                data.stored_cells(sim, d)
            except ValueError:
                continue
            raise AssertionError(f"accepted an inconsistent file: {list(bad)}")


def test_corpus_task_slices_cover_every_event_once_and_rows_are_species_major():
    for n, tasks in ((1000, 2), (464008, 32), (7, 3)):
        spans = [corpus.task_slice(n, t, tasks) for t in range(tasks)]
        assert spans[0][0] == 0 and spans[-1][1] == n and all(spans[i][1] == spans[i + 1][0] for i in range(tasks - 1))
    assert corpus.source_rows(per=10, n=3).tolist() == [0, 1, 2, 10, 11, 12, 20, 21, 22]


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
