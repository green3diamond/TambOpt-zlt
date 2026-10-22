# SWGO clustering, particle counting and triggers vs TambOpt (2026-10-22)

Branch `bg-kernel-units`. Sources: `~/tambo/SWGOLO` (`SWGOLO7.ipynb`, `swgolo148.cpp`), Dorigo et al.,
"Toward the end-to-end optimization of the SWGO array layout", arXiv 2310.01857 (Nucl. Phys. B 1017 (2025) 116934),
and run 19 counting tests in `<run 19>/diagnostics/counting/`. Earlier comparison: `docs/swgolo-comparison.md`
(branch `swgolo-comparison-run12`).

## 1. Where the "triplets" come from

### Notebook (`SWGOLO7.ipynb`): the reconstructability step
Per shower b and unit i (cells 5, 8, 11, 17–20, 59):

```
mu_bi = TankArea * rho(R_bi; E_b, theta_b) * cos(theta_b)          # expected particles, TankArea = 68.59 pi = 215 m^2
N_bi  = sigmoid(10 (mu_bi - 0.1)) * (mu_bi + 0.05 mu_bi xi)        # SmearN: soft 0.1-particle gate, 5 % Gaussian
n_b   = sum_i sigmoid(5 (N_bi - 5))                                # units with >~ 5 particles
r_b   = sigmoid(5 (n_b - 3))                                       # >~ 3 such units
U     = 0.01 sum_b r_b / ((th^ - th)^2 + 1e-5)                     # U_TH
      +      sum_b r_b / ((E^ - E)^2 + 0.01)                       # U_E
      + sqrt(sum_b r_b + 1e-6) / sqrt(rho_cores)                   # U_PR, cores thrown over +-3 km
```

- r_b steps from 0.007 (n = 2) to 0.5 (n = 3) to 0.993 (n = 4) and multiplies **every** term. Where a shower's
  density is just enough for one unit to pass 5 particles, putting units next to it makes them pass together: r_b goes
  ~0 → ~1. The gradient, dr/dx_i = 25 r(1 − r) σ'(5(N_i − 5)) dN_i/dR dR/dx_i, is non-zero only for units near the
  5-particle edge in showers near the 3-unit edge, and pulls them toward units that already pass.
- U_PR (√Σ r_b, cores over ±3 km while the array starts within 1 km) rewards catching more showers: many separate
  groups beat one dense core.
- `push_apart(min_dist = 2 * TankRadius)` = 16.6 m: a group is a touching blob.
- No Poisson draw (the "poisson" comment in `GetCounts` is not what the code does): the counts are smooth
  expectations. The clustering is a property of the threshold structure of the objective, not of noise averaging.
- The optimisation cell unpacks 7 values from `GenerateShowers`, the saved `GenerateShowers` returns 10: the cell was
  written for an older version and does not run as saved; which N it thresholds (e + μ or one channel) is undefined.

### C++ (`swgolo148.cpp`) and the paper: symmetry and aggregates, built in
- `CommonMode = 3` (l. 280, 10099): one sector of units is optimised and copied at 120° and 240° (l. 3910–3923).
  The paper: "units move in triplets, so final layouts show triangular symmetry". Imposed, not found.
- Each unit is a 19-tank hexagonal macro-aggregate (`TankNumber = 19`, `TankArea = π·1.91²·19` = 218 m², l. 10621),
  ≥ 0.6 m between tanks, ≥ 22 m between aggregates; the paper's reason: computing load.
- Trigger `PActive` = Poisson P(≥ `Ntrigger` = 50 tanks with ≥ 1 particle), per tank P(≥ 1) = 1 − e^(−λ) (l. 4743, 4499).

## 2. What the optimisation finds

| | SWGO (C++, paper §9) | TambOpt (runs 18, 19) |
|---|---|---|
| Free result | Expands outward, empties the centre ("a donut is to be preferred to a disc"), units ~uniform inside the boundary; three different starts converge to the same layout; utility +40–75 % | Even coverage (grid / wide disk) best; optimisations spread out from any start, never form a dense core, gain ≤ noise at 1° from the grid |
| Imposed | Triangular symmetry (`CommonMode = 3`), 19-tank aggregates | Detectors confined to the plane |
| Notebook | Blobs of ~3 units from the r_b step | — |

### Why TambOpt does not hollow out the centre
SWGO's empty centre is an acceptance effect: cores land uniformly over the array **plus 2 km** (`R_slack`), the
footprints of 0.1–10 PeV showers at 4.7 km altitude are hundreds of metres across, and units in the middle of a
symmetric array are redundant for showers landing there (their neighbours already measure them) while units moved
outward catch showers that would otherwise be lost (U_GF, U_PR count events). TambOpt has none of these pressures:
- every tau axis hits the plane (illumination ~uniform over the array, no cores outside it), so moving outward buys
  no acceptance and the plane boundary stops it anyway;
- at run 19's ~1 km spacing the footprint (~1 km) is not redundantly covered: a hole in the centre loses the showers
  landing there;
- the objective is resolution (share within 1°), not an event count. When an acceptance term was added (run 12,
  λ·mean(r)), detectors moved to the rim — the same tendency, without a gain against the kernel.

## 3. Expected particles: SWGO vs TAMBO

| | Area per unit | Units | Total area | Photons | Typical core distance |
|---|---|---|---|---|---|
| SWGOLO notebook | 215 m² (19-tank unit) | 90 | ~19,400 m² | counted (water) | 0–1 km (array), cores ±3 km |
| SWGOLO C++ default | 218 m² | 150 (36–330 in the paper) | ~8,000–72,000 m² | counted | array + 2 km |
| SWGO benchmarks | 11.5 m² (one tank) | 3,805–8,371 | ~44,000–96,000 m² | counted | — |
| TAMBO model (run 19) | 2 m² | 100 | 200 m² | ε_γ = 0.02 (assumed scintillator) | 0.5–5 km on a 144 km² plane |

TambOpt's kernel E is summed kinetic energy (GeV) over a 10 m Gaussian (peak 1, total 2πσ² ≈ 628 m²), not particles.
Per species and distance r from the axis, expected particles in a detector of area A:
`mu_s = E_s · A / (2πσ² KE_s(r)) / ratio_s(r)` (`modules/optimize/background.py`, `particles_per_E`; KE_s and the
ring/true ratio from 16 CORSIKA 8 π⁻ showers, `modules/optimize/c8_scaling.json`). Band values (100 m – 3 km):
KE e 0.081, μ 3.23, γ 0.043 GeV; ring ratio ≈ 0.6.

Measured with Poisson counting (run 19 recon trained without counting, `<run 19>/diagnostics/counting/`):

| Layout (100 det.) | Spacing | Showers (axis inside) | Detectors with ≥ 1 particle / shower | Within 1° / 5°, no counting | Within 1° / 5°, counting |
|---|---|---|---|---|---|
| Grid, 12 km plane | ~1 km | 5,000 | 0.59 | 10.4 / 73.5 % | 0.2 / 4.9 % |
| Disk r = 1.5 km | ~270 m | 2,155 | 3.4 | 12.3 / 80.3 % | 0.5 / 9.0 % |
| Disk r = 1 km | ~180 m | 1,035 | 6.1 | 11.1 / 76.2 % | 0.4 / 10.6 % |
| Disk r = 750 m | ~130 m | 579 | 9.9 | 7.8 / 70.6 % | 0.5 / 13.1 % |
| Disk r = 500 m (Malata-like) | ~90 m | 256 | 18.4 | 6.6 / 65.6 % | 0.4 / 12.9 % |

An SWGO unit expects tens to thousands of particles where a 2 m² TAMBO detector at ~1 km expects ~0.01–0.5: ~100×
the area per unit for charged particles, ×4–5 from counting photons, and ~10× closer to the core (density ∝ r^−3
there). At Malata-like density (r = 500 m disk) ~18 of 100 detectors see a particle. The recon was trained on noise-free
inputs, so the counted columns are a lower bound until the fine-tune on counted inputs (job 54580014) is scored.

## 4. Poisson triggers: kNN 2-of-3 vs a ≥ 3-detector threshold

Triplets vs an evenly filled disk of the same 99 detectors, Poisson counting, showers whose axis lands inside
(`region_triplets_poisson.json`; 257–1,172 showers, so ±~2 % at 5°):

| Region | Even disk, within 5° | Triangles r = 10 / 25 / 50 / 100 m, within 5° | Detectors with ≥ 1 particle (disk / triangles) |
|---|---|---|---|
| R = 500 m | 16.0 % | 15.6 / 16.4 / 17.0 / 12.6 % | 18.3 / 16–18 |
| R = 750 m | 13.2 % | 11.1 / 10.4 / 10.9 / 12.9 % | 9.8 / 8.8–9.8 |
| R = 1 km | 11.0 % | 9.8 / 9.5 / 10.1 / 10.2 % | 6.1 / 5.9–6.1 |

Without counting the same comparison is also within noise (e.g. R = 1 km: disk 11.3 %, triangles 9.2–11.5 % within 1°).
So with no trigger rule, tight triangles do not beat even coverage, counted or not: the cluster itself adds no
direction information (three units ~17 m apart sample one point of the front).

What the two rules do (not yet run with counting in the region; `score_knn.py` has no region selection):
- **kNN 2-of-3 gate** (per detector): read out iff ≥ 2 of {itself, its 2 nearest neighbours} are hit. It removes
  isolated hits — robust against background ghosts (10 µs: 3.2 → 0.3–0.5 ghosts per shower, no counting) — and with
  counting pays for having a close partner: for a triangle of equal μ, P(≥ 2 of 3) = 3p² − 2p³ with p = 1 − e^(−μ),
  while spread detectors rarely see the same shower. It implies "≥ 2 nearby detectors with signal", not 3 and no
  particle threshold. In an objective it acts only through the recon's loss of signal (smooth), or through the
  straight-through gradient of the soft gate in Stage 4.
- **≥ k detectors with ≥ 1 particle** (per event, Poisson, as SWGOLO's `PActive`):
  r_b = P(≥ k) ≈ 1 − Σ_{j<k} e^(−λ_b) λ_b^j / j!, λ_b = Σ_i (1 − e^(−μ_bi)); smooth in the positions. As a weight on
  every shower's reward (k = 3) it is the Poisson version of the notebook's step and the term that would create
  clusters in an optimisation. It rejects showers carried by one background muon but does not clean individual ghosts.
- Robustness: the gate is local and acts on every detector, so it survives a few dead or noisy detectors and long
  windows; the event threshold is global and blind to where the hits are, so ghosts count toward k. Combined (gate,
  then ≥ k gated detectors) gives both. Either rule buys real acceptance only if the hardware / analysis trigger is
  the same rule; otherwise it is a definition the optimiser satisfies cheaply by clustering.

## 5. Open
- Fine-tune on counted inputs (54580014) and rescore (54580015); counted scoring of all layouts (54580013).
- kNN gate with counting in the Malata-like region; Stage 4 with counting, the gate and r_b (k = 3) to see whether
  clusters form and whether they score better on the recon with counted inputs.
- TAMBO detector area, type and photon efficiency (currently 2 m², ε_γ = 0.02, assumptions).
- Calibration from π⁻ showers only; ring kernel uses |t1| (geometry bias corrected on average by ratio_s(r)).
