"""Diagnostic of expected_response.tables() on CORSIKA 8 test showers (no GPU): per shower and species, the model's
energy per plane (one part-1 draw) against the stored cells' energy per plane, and the model's particles per plane
(ring energy / mean particle energy, summed over rings) with the implied mean particle energy, and the model's energy
within 50 / 200 m of the axis against the cells' (radius |t1| of the cell, the kernel's ring radius).

    python diag_planes.py [n_showers]
"""
import os, sys
import numpy as np, h5py
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, R)
from scipy.special import softmax
from modules.constants import N_PLANES, SPECIES_NAMES, ZENITH_MAX, ZENITH_MIN
from modules.optimize import expected_response as XR
from symbolic import atmosphere, data, draw, energy, lateral, longitudinal, rings, rows, times, width

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
H = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3"
FILES = {"electron": "combined_electrons_test.h5", "muon": "combined_muons_test.h5", "photon": "combined_photons_test.h5"}
part1, laws = XR.load_model()
ln_r, area = XR.ring_geometry()
rc = np.exp(ln_r)
with h5py.File(f"{H}/{FILES['electron']}") as f:
    dirs = f["directions"][:N].astype(np.float64); en = f["energies"][:N, 0].astype(np.float64); cls = f["pdg"][:N].astype(int)
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
zen = np.clip(np.degrees(np.arccos(dirs[:, 2])), ZENITH_MIN, ZENITH_MAX); u = np.cos(np.radians(zen))
X = atmosphere.plane_depths(zen, N_PLANES)
E, xmax = draw.sample_energies(part1, np.log10(en), u, X, cls, np.random.default_rng(1))
for i in range(N):
    cn = data.CLASS[cls[i]]
    print(f"\n=== shower {i}: {cn}, E {en[i]:.3g} GeV, zenith {zen[i]:.1f}")
    e1 = np.array([-dirs[i, 1], dirs[i, 0], 0.0]); e1 /= np.linalg.norm(e1)
    for s in SPECIES_NAMES:
        with h5py.File(f"{H}/{FILES[s]}") as f:
            c = np.asarray(f["showers"][i], dtype=np.float64).reshape(-1, 5)
        c = c[c[:, 3] > 0]
        k_c = c[:, 2].astype(int); r_c = np.abs(c[:, 0] * e1[0] + c[:, 1] * e1[1])
        Ec = np.bincount(k_c, c[:, 3], N_PLANES)
        law = laws[(s, cn)]
        ks = np.nonzero(E[s][i] > 0)[0]
        r_ref = np.array([lateral.r_m_ref(k, u[i]) for k in ks])
        x = rows.features(longitudinal.age(X[i, ks], xmax[s][i]), np.full(len(ks), np.log10(en[i])), np.full(len(ks), u[i]),
                          np.log(E[s][i, ks]), r_ref)
        ll = width.mean(law["width"], ks, x[:, 3], x[:, 1])
        q = softmax(x @ law["rings"], axis=1)
        sh = np.stack([rings.stretch(q[j], ll[j]) for j in range(len(ks))])
        me = np.exp(energy.tanh_sum(law["energy"], times.expand(rows.plane_inputs(x, ks)))).reshape(len(ks), -1)
        e_ring = E[s][i, ks][:, None] * sh
        npart = (e_ring / me).sum(1)
        print(f"  {s:8s} sum E: model {E[s][i].sum():.3g}  cells {Ec.sum():.3g}  | particles (model) {npart.sum():.3g}, "
              f"mean E {E[s][i, ks].sum() / npart.sum():.3g} GeV | width log10 {np.round(ll[:3], 2)}")
        for k in (4, 8, 12):
            if k in ks:
                j = list(ks).index(k)
                m50 = e_ring[j][rc < 50].sum(); m200 = e_ring[j][rc < 200].sum()
                c50 = c[(k_c == k) & (r_c < 50), 3].sum(); c200 = c[(k_c == k) & (r_c < 200), 3].sum()
                print(f"    plane {k:2d}: E model {E[s][i, k]:.3g} cells {Ec[k]:.3g} | <50 m model {m50:.3g} cells {c50:.3g} "
                      f"| <200 m model {m200:.3g} cells {c200:.3g} | ring-0..4 mean E {np.round(me[j][:5], 3)}")
