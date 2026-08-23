"""Unit test for the Step-0 degenerate-shower re-roll. No GPU, no checkpoints.

`regenerate_degenerate` is the only guard that sees the inflated-energy showers
(they are finite and normally sized), so its selection and stop conditions are
worth pinning down. The AllShowers generator is replaced by a stub that returns
a scripted sequence of draws, which lets the test assert exactly WHICH rows were
re-rolled and how many times.

    python tests/test_blob_guard.py
"""
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def _load_step0():
    """Import the Step-0 script without its heavy top-level imports.

    It pulls in allshowers + the surrogate stack and hardcodes a CUDA device at
    import time, none of which this test needs; stubbing the modules keeps the
    test runnable on a login shell or a CPU-only node."""
    for name, attrs in {
        "showerdata": ("Showers", "save_batch", "create_empty_file"),
        "allshowers": (),
        "allshowers.generate_showers": ("sample_primary_particles", "run_point_count_fm"),
        "allshowers.generator": ("Generator", "generate"),
    }.items():
        mod = types.ModuleType(name)
        for a in attrs:
            setattr(mod, a, lambda *a, **k: None)
        sys.modules.setdefault(name, mod)

    import importlib.util
    path = os.path.join(os.path.dirname(_HERE), "scripts",
                        "00_generate_data_dual_species.py")
    spec = importlib.util.spec_from_file_location("gen00", path)
    mod = importlib.util.module_from_spec(spec)
    # DEVICE = torch.device("cuda") at import time is fine (no allocation), but
    # the module also validates SPECIES against constants at import.
    spec.loader.exec_module(mod)
    return mod


def _shower(n_points, cap, per_point_energy):
    """One (cap, 5) cloud with `n_points` real points at a fixed energy each."""
    s = torch.zeros(cap, 5)
    s[:n_points, 3] = per_point_energy
    return s


def main():
    g = _load_step0()
    cap, n = 64, 6
    e_prim = torch.full((n, 1), 100.0)

    # Rows 1, 3, 4 are degenerate. Row 4 is inf (the corpus holds one such muon
    # row); rows 1 and 3 are finite but inflated, which is the common mode.
    good = _shower(20, cap, 10.0)            # total 200 -> ratio 2.0
    def blob(scale):
        return _shower(20, cap, 10.0 * scale)
    samples = torch.stack([good, blob(1e4), good, blob(1e6), good, good]).clone()
    samples[4, :20, 3] = float("inf")
    bad_rows = {1, 3, 4}

    # The stub returns clean showers, but records every row index it was asked
    # to redraw so the test can assert the guard re-rolled exactly the bad ones.
    calls = []
    def fake_generate(generator, energies, num_points, angles, batch_size,
                      device, labels):
        calls.append(int(energies.shape[0]))
        return torch.stack([good.clone() for _ in range(energies.shape[0])])
    g.generate = fake_generate

    ratio_before = g._deposited_over_primary(samples, e_prim)
    flagged = set(torch.nonzero(
        ~torch.isfinite(ratio_before) | (ratio_before > 20.0)).flatten().tolist())
    assert flagged == bad_rows, f"selection: {flagged} != {bad_rows}"
    print(f"  selection            OK  flagged {sorted(flagged)}")

    out, sp_batch = g.regenerate_degenerate(
        gen=None, name="muon", samples=samples,
        energies=e_prim, num_points=torch.full((n,), 20),
        directions=torch.zeros(n, 3), labels=torch.zeros(n, dtype=torch.int64),
        sp_batch=4, max_ratio=20.0, max_retries=3)

    ratio_after = g._deposited_over_primary(out, e_prim)
    assert torch.isfinite(ratio_after).all(), "non-finite survived"
    assert (ratio_after <= 20.0).all(), f"still degenerate: {ratio_after}"
    print(f"  all rows clean       OK  max ratio {ratio_after.max():.4g}")

    assert calls == [3], f"expected one re-roll of 3 rows, got {calls}"
    print(f"  one pass, bad only   OK  generate called with {calls}")

    # Untouched rows must be bit-identical -- the guard must not perturb showers
    # it did not flag.
    for i in sorted(set(range(n)) - bad_rows):
        assert torch.equal(out[i], good), f"row {i} was modified"
    print("  good rows untouched  OK")

    # Budget exhaustion: a generator that never recovers must stop after
    # max_retries and keep the last draw rather than loop forever.
    calls.clear()
    def stuck_generate(**kw):
        calls.append(int(kw["energies"].shape[0]))
        return torch.stack([blob(1e4) for _ in range(kw["energies"].shape[0])])
    g.generate = stuck_generate
    stuck = torch.stack([good, blob(1e4)]).clone()
    out2, _ = g.regenerate_degenerate(
        gen=None, name="muon", samples=stuck, energies=e_prim[:2],
        num_points=torch.full((2,), 20), directions=torch.zeros(2, 3),
        labels=torch.zeros(2, dtype=torch.int64), sp_batch=4,
        max_ratio=20.0, max_retries=3)
    assert calls == [1, 1, 1], f"expected exactly 3 retries, got {calls}"
    assert g._deposited_over_primary(out2, e_prim[:2])[0] <= 20.0, "good row lost"
    print(f"  budget exhausts      OK  {len(calls)} retries then gave up")

    # max_retries=0 disables the guard entirely.
    calls.clear()
    untouched = torch.stack([blob(1e4), good]).clone()
    out3, _ = g.regenerate_degenerate(
        gen=None, name="muon", samples=untouched.clone(), energies=e_prim[:2],
        num_points=torch.full((2,), 20), directions=torch.zeros(2, 3),
        labels=torch.zeros(2, dtype=torch.int64), sp_batch=4,
        max_ratio=20.0, max_retries=0)
    assert calls == [] and torch.equal(out3, untouched), "max_retries=0 not a no-op"
    print("  max_retries=0 no-op  OK")

    print("\nall blob-guard tests passed")


if __name__ == "__main__":
    main()
