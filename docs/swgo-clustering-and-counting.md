# SWGO clustering, particle counting and triggers vs TambOpt (2026-10-22)

Sources: `~/tambo/SWGOLO` (`SWGOLO7.ipynb`, `swgolo148.cpp`); Dorigo et al., arXiv 2310.01857 (Nucl. Phys. B 1017
(2025) 116934); run 19 counting tests in `<run 19>/diagnostics/counting/`.

## 1. Triplets

**Notebook** (cells 5, 8, 11, 17–20, 58–59). Per shower b, unit i:

```
mu_bi = 215 m^2 * rho(R_bi; E, theta) * cos(theta)        # expected particles (TankArea = 68.59 pi)
N_bi  = sigmoid(10 (mu_bi - 0.1)) * mu_bi * (1 + 0.05 xi)  # SmearN: soft 0.1-particle gate, 5 % noise; no Poisson
n_b   = sum_i sigmoid(5 (N_bi - 5))                        # units with >~ 5 particles
r_b   = sigmoid(5 (n_b - 3))                               # >~ 3 such units
U     = 0.01 sum r_b/(dth^2 + 1e-5) + sum r_b/(dE^2 + 0.01) + sqrt(sum r_b)/sqrt(rho_cores)   # cores over +-3 km
```

r_b (0.007 at n = 2, 0.5 at n = 3, 0.993 at n = 4) multiplies every term, so units placed together cross the
5-particle edge together and switch the shower on. √Σr_b with cores beyond the array spreads the groups out;
`push_apart` (16.6 m) makes them touching blobs. The counts are smooth expectations: the clustering comes from the
objective, not from noise. (The optimisation cell unpacks 7 values from a `GenerateShowers` that returns 10; it does
not run as saved.)

**C++.** `CommonMode = 3` (l. 280, 3910–3923) optimises one sector and copies it at 120°/240°: "units move in
triplets" (paper) = imposed triangular symmetry. Each unit is a 19-tank aggregate (218 m², l. 10621), chosen for
computing cost. Trigger: Poisson P(≥ 50 tanks with ≥ 1 particle), per tank 1 − e^(−λ) (l. 4499, 4743).

## 2. What the optimisation finds

| | Free result | Imposed |
|---|---|---|
| SWGO C++ (paper §9) | Expands, empties the centre ("a donut is to be preferred to a disc"), ~uniform inside the boundary; different starts converge | Triangular symmetry, 19-tank aggregates |
| SWGO notebook | Groups of ~3 units | The r_b step |
| TambOpt (runs 18, 19) | Even coverage; spreads from any start, no dense core, no gain over the grid beyond noise | Detectors on the plane |

SWGO hollows the centre for acceptance: cores are thrown over the array + 2 km (`R_slack`), so units moved outward
gain triggered showers, while at SWGO's density central units largely see the same showers (inference, not stated in
the paper). Its utility is flux precision, energy and pointing resolution (U_GF, U_IR, U_PR); acceptance enters through
the number of triggered showers. In TambOpt every axis hits the plane, so moving outward gains nothing; the run 19
objective (share of all showers within 1°) does include acceptance, but there is none to win outside the array. Both
optimisers spread out and avoid dense cores (TambOpt's even coverage is the analogue of SWGO's expansion). Run 12's
acceptance term (σ = 50) pushed detectors to the rim with no gain; run 14's (σ = 10) gave +3.1 % (not bootstrapped);
both Malata-era kernels.

Hollow-centre test, run 19 (100 detectors evenly over an annulus r_in – 6 km, all 5,000 showers, no counting;
`annuli_none.json`): a hole loses showers.

| r_in | 0 (6 km disk) | 2 km | 3 km | 4 km | grid |
|---|---|---|---|---|---|
| Within 1° / 5° | 11.9 / 77.4 % | 9.9 / 73.0 % | 8.4 / 65.0 % | 5.2 / 52.1 % | 10.4 / 73.5 % |
| Median error | 2.76° | 3.05° | 3.56° | 4.76° | 3.05° |

With counting all are 4–5 % within 5° (0.5 detectors per shower with a particle), so the test is only informative
without counting.

## 3. Expected particles

| | Area / unit | Units | Total | Photons | Distance to core |
|---|---|---|---|---|---|
| Notebook | 215 m² | 90 | ~19,000 m² | counted (water) | ≤ 1 km array, cores ±3 km |
| C++ / paper | 218 m² | 36–330 | 8,000–72,000 m² | counted | array + 2 km |
| SWGO benchmarks | 11.5 m² | 3,805–8,371 | 44,000–96,000 m² | counted | — |
| TAMBO model | 2 m² | 100 | 200 m² | ε_γ = 0.02 | 0.5–5 km on 144 km² |

TambOpt's E is summed kinetic energy (GeV) over a 10 m Gaussian (total weight 628 m²). Expected particles per species
at distance r: `mu_s = E_s · A / (2πσ² KE_s(r)) / ratio_s(r)` (`particles_per_E`, table `c8_scaling.json` from 16
CORSIKA 8 π⁻ showers; 100 m – 3 km: KE e 0.081, μ 3.23, γ 0.043 GeV, ring ratio ≈ 0.6). Per unit SWGO has ~100× the
area, ×4–5 from photons, and sits ~10× closer to cores (density ∝ r^−3): tens–thousands of particles vs ~0.01–0.5.

Poisson counting, current recon (trained without counting — a lower bound until the counted fine-tune):

| 100 detectors | Spacing | Showers (axis inside) | Det. with ≥ 1 particle | 1° / 5°, no counting | 1° / 5°, counting |
|---|---|---|---|---|---|
| Grid, 12 km | ~1 km | 5,000 | 0.6 | 10.4 / 73.5 % | 0.2 / 4.9 % |
| Disk 1.5 km | ~270 m | 2,155 | 3.4 | 12.3 / 80.3 % | 0.5 / 9.0 % |
| Disk 1 km | ~180 m | 1,035 | 6.1 | 11.1 / 76.2 % | 0.4 / 10.6 % |
| Disk 750 m | ~130 m | 579 | 9.9 | 7.8 / 70.6 % | 0.5 / 13.1 % |
| Disk 500 m (Malata-like) | ~90 m | 256 | 18.4 | 6.6 / 65.6 % | 0.4 / 12.9 % |

## 4. Poisson triggers

Triangles (33 × 3) vs an even disk of 99 detectors, showers landing inside, within 5° (±~2 %):

| Region | Even disk, counting | Triangles r = 10 / 25 / 50 / 100 m, counting |
|---|---|---|
| R = 500 m | 16.0 % | 15.6 / 16.4 / 17.0 / 12.6 % |
| R = 750 m | 13.2 % | 11.1 / 10.4 / 10.9 / 12.9 % |
| R = 1 km | 11.0 % | 9.8 / 9.5 / 10.1 / 10.2 % |

No trigger rule, counted or not: triangles do not beat even coverage (no counting, R = 1 km: disk 11.3 %, triangles
9.2–11.5 % within 1°).

With a Poisson event trigger, ≥ 3 detectors with ≥ 1 particle (with counting, one particle already gives E′ > 1;
no background; showers whose axis lands inside):

| 99–100 detectors | Showers | Triggered | Within 5°, triggered | Within 5°, all |
|---|---|---|---|---|
| Grid, 12 km | 2,453 | 4.9 % | 16.0 % | 3.8 % |
| Disk 1.5 km | 2,155 | 38.5 % | 18.6 % | 9.0 % |
| Disk 1 km | 1,031–1,035 | 60–61 % | 16.4 % | 10.6–11.0 % |
| Triangles in R = 1 km (r = 10–100 m) | 1,031–1,172 | 58–63 % | 14–16 % | 9.5–10.2 % |
| Disk 750 m | 576–579 | 74–75 % | 16–17 % | 13.1–13.2 % |
| Triangles in R = 750 m | 579–689 | 71–73 % | 14–17 % | 10.4–12.9 % |
| Disk 500 m (Malata-like) | 256–257 | 88–90 % | 14–17 % | 12.9–16.0 % |
| Triangles in R = 500 m | 257–326 | 84–88 % | 15–19 % | 12.6–17.0 % |

The trigger rate rises from 5 % to ~90 % with density; triangles trigger no more often than the even disk (usually
1–3 points less) and are as accurate within noise. Within 1° stays ≤ 1.3 % (recon not trained on counted inputs).

| | kNN 2-of-3 gate (per detector) | ≥ k detectors with ≥ 1 particle (per event) |
|---|---|---|
| Rule | Read out iff ≥ 2 of {itself, 2 nearest} hit | r_b = P(≥ k) ≈ 1 − Σ_{j<k} e^(−λ) λ^j/j!, λ = Σ_i (1 − e^(−μ_i)) |
| Implies | ≥ 2 nearby detectors with signal; no particle threshold | Notebook step (k = 3) in Poisson form (SWGOLO `PActive`) |
| Clustering pressure | Pairs/triplets: a triangle of equal μ passes with 3p² − 2p³; acts via recon loss or straight-through gradient | Strong, if used as a weight on every shower's reward |
| Background | Removes isolated ghosts (10 µs, no counting: 3.2 → 0.3–0.5 per shower) | Rejects single-muon showers; ghosts still count toward k |
| Robustness | Local: tolerates dead/noisy detectors and long windows | Global, blind to where hits are |

Combining both (gate, then ≥ k gated detectors) gives both properties. Either rule only buys real acceptance if the
actual trigger is that rule.

Malata-like region, kNN 2-of-3 gate and the notebook-style trigger together (current recon, showers whose axis lands
inside, triangles r = 10–100 m, ±1–4 %; `region_mutrig_{none,poisson}.json`). Gate columns: Poisson counting, within 5°
of all showers. Expected-count trigger: ≥ 3 read-out detectors with μ ≥ 1, 0 µs (the smooth r_b agrees within ~1 point).

| Region | Layout | Showers | Det. with ≥ 1 drawn particle | Kept by the gate | All read out, 0 / 10 µs | kNN gate, 0 / 10 µs | Det. with μ ≥ 1 | Triggered (μ ≥ 1, ≥ 3 det.) | Triggered and within 5°, no counting / counting |
|---|---|---|---|---|---|---|---|---|---|
| R = 500 m | even disk | 257 | 18.3 | 93 % | 16.0 / 2.7 % | 12.8 / 6.2 % | 14.5 | 81 % | 56.4 / 14.8 % |
| | triangles | 257–326 | 16.2–18.0 | 92–95 % | 12.6–17.0 / 3.1–4.7 % | 8.9–11.7 / 4.7–7.4 % | 12.7–14.1 | 74–81 % | 52.5–58.7 / 10.7–15.6 % |
| R = 750 m | even disk | 576 | 9.8 | 87 % | 13.2 / 2.1 % | 8.5 / 4.7 % | 7.3 | 58 % | 42.5 / 10.2 % |
| | triangles | 579–689 | 8.8–9.8 | 88–91 % | 10.4–12.9 / 2.8–4.0 % | 7.8–8.7 / 5.2–7.4 % | 6.6–7.3 | 55–63 % | 40.3–45.1 / 8.2–10.4 % |
| R = 1 km | even disk | 1,031 | 6.1 | 82 % | 11.0 / 3.4 % | 8.2 / 5.8 % | 4.3 | 43 % | 32.8 / 7.6 % |
| | triangles | 1,031–1,172 | 5.8–6.1 | 83–89 % | 9.5–10.2 / 2.9–4.0 % | 6.4–7.8 / 5.1–5.5 % | 4.2–4.5 | 41–55 % | 31.2–41.1 / 6.7–7.9 % |

- Gate: at 10 µs it removes ghosts (3.0–3.4 → 0.3–0.6 per shower) and roughly doubles the share within 5°; at 0 µs it
  costs (lone real hits dropped). Triangles pass it only slightly more often than the even disk; accuracy equal within
  noise. For r ≤ 50 m the in-triangle 2-of-3 equals the kNN gate.
- Expected-count trigger reproduces the notebook mechanism: at lower density tight triangles cross the μ ≥ 1 edge
  together and trigger more often (R = 1 km, r = 25 m: 51 vs 43 %, ~4σ), with about the same accuracy per triggered
  shower, so an objective counting only triggered, well-reconstructed showers prefers triangles (no counting: up to
  41 vs 33 %). With counting the advantage disappears (6.7–7.9 vs 7.6 %): drawn counts smear the hard edge.
- Within 1° ≤ 2 % with counting (recon not trained on counted inputs). Without counting the gate keeps 88–98 % of
  detectors and matters only at 10 µs.

## 5. Status and open (2026-10-22)

**Jobs** (all pending on gpu_requeue, none started):

| Jobs | What | When done |
|---|---|---|
| 54580013 | Counting scoring of all run 19 layouts (background test + kNN gate), current recon | rescore and plot |
| 54580014 → 54580015 | Fine-tune the recon on counted inputs (2 days, 200 GB), then rescore | the real counted results |
| 16 + 4 `ko19x_*` | Noise-aware layout optimisations: extended Adam + L-BFGS; old background, main checkout | rescore, update `opt_noise_findings`, rebuild `run19_12km.pdf` |

**Open:**
- Stage 4 with counting, the kNN gate and the event trigger r_b (k = 3): do clusters form?
- Counted results vs shower energy.
- Gradient direction check on the kernel grid runs.
- √2·|t1| ring radius (branch `ring-sqrt2`): new run folder, Stage 1 relabel, recon retrain.
- Calibration from π⁻ showers only.

**Done:** kNN gate with counting in the Malata-like region (§ 4); `--counting notebook` mode dropped; detector model
2 m² scintillator, ε_γ = 0.02 (decided); ring-radius fix implemented on `ring-sqrt2`; memory notes updated.

**Code state:**
- `bg-kernel-units`: not pushed, not merged; merge into the main checkout only after the 20 `ko19x` jobs have started
  (they import the main checkout's background code).
- `ring-sqrt2`: 1 commit on top of `bg-kernel-units`, not pushed.
- Main checkout, uncommitted: lit-threshold option (`score_all.py`, `analyze.py`, `detector_count_grid.py`),
  best-per-family and triplet-split figures, `e_distribution.py`, two slurm scripts. The summary deck builder lives in
  `detector_optimization_v6/figures/` (outside the repo).
- Housekeeping: old worktree `.claude/worktrees/r19-nandebug`.
