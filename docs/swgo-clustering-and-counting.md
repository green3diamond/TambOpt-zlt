# SWGO clustering, particle counting and triggers vs TambOpt (2026-10-22)

Sources: `~/tambo/SWGOLO` (`SWGOLO7.ipynb`, `swgolo148.cpp`); Dorigo et al., arXiv 2310.01857 (Nucl. Phys. B 1017
(2025) 116934); run 19 counting tests in `<run 19>/diagnostics/counting/`.

**Disclaimer.** Every result with counted inputs ("counting" columns, § 3–4) uses the run 19 recon, trained only on
noise-free kernel inputs; no recon trained on counted inputs exists (the zoo fine-tune 54580014 was cancelled, § 7). These numbers are
lower bounds, and the layout rankings under counting may change once the recon is retrained. Results without counting
use the recon on the inputs it was trained for.

**Background amplitude (changed 2026-10-23).** The 10 µs results in § 4 used a background muon whose kernel E depended
on the detector's distance from the shower axis (1 / particles_per_E('mu', r): ~3,900 at 30 m, ~650 band, ~90 at 3 km),
which is unphysical for a cosmic muon. The default is now `TAMBOPT_BG_UNITS=cosmic`: each muon gets its own random KE
from a sea-level spectrum, (KE + 2.6 GeV)^−2.7 on 0.1 GeV – 1 TeV (mean 3.85 GeV), entered as KE · 2πσ² / 2 m²
(mean ~1,210 per muon), independent of the shower. The § 4 background columns predate this.

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

SWGO hollows the centre for acceptance (cores land over the array + 2 km). In TambOpt every axis hits the plane, so
there is nothing to gain outward (donuts at the scale of the site: § 5).

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

Poisson counting (lower bounds, see the disclaimer):

| 100 detectors | Spacing | Showers (axis inside) | Det. with ≥ 1 particle | 1° / 5°, no counting | 1° / 5°, counting |
|---|---|---|---|---|---|
| Grid, 12 km | ~1 km | 5,000 | 0.6 | 10.4 / 73.5 % | 0.2 / 4.9 % |
| Disk 1.5 km | ~270 m | 2,155 | 3.4 | 12.3 / 80.3 % | 0.5 / 9.0 % |
| Disk 1 km | ~180 m | 1,035 | 6.1 | 11.1 / 76.2 % | 0.4 / 10.6 % |
| Disk 750 m | ~130 m | 579 | 9.9 | 7.8 / 70.6 % | 0.5 / 13.1 % |
| Disk 500 m (Malata-like) | ~90 m | 256 | 18.4 | 6.6 / 65.6 % | 0.4 / 12.9 % |

## 4. Poisson triggers

| | kNN 2-of-3 gate (per detector) | ≥ k detectors with ≥ 1 particle (per event) |
|---|---|---|
| Rule | Read out iff ≥ 2 of {itself, 2 nearest} hit | r_b = P(≥ k) ≈ 1 − Σ_{j<k} e^(−λ) λ^j/j!, λ = Σ_i (1 − e^(−μ_i)) |
| Implies | ≥ 2 nearby detectors with signal | Notebook step (k = 3) in Poisson form (SWGOLO `PActive`) |
| Clustering pressure | Pairs/triplets (a triangle of equal μ passes with 3p² − 2p³) | Strong, if it weights every shower's reward |
| Background | Removes isolated ghosts | Rejects single-muon showers; ghosts still count |
| Robustness | Local: tolerates dead/noisy detectors | Global, blind to where hits are |

Either rule buys real acceptance only if the actual trigger is that rule.

Tests: 99 detectors in a disk of radius R (even, or 33 triangles of r = 10–100 m), Malata-like density, showers whose
axis lands inside (257–1,172 per layout, ±1–4 %); `region_mutrig_{none,poisson}.json`. Triangles: 33 tight triangles
of 3 detectors (side 17–173 m), centres spread evenly over the disk.

**Detectors per shower** (counting, 0 µs)

| Region | Layout | Showers | With ≥ 1 drawn particle | Read out by the kNN gate | With μ ≥ 1 (expected) |
|---|---|---|---|---|---|
| R = 500 m | even disk | 257 | 18.3 | 17.0 | 14.5 |
| | triangles | 257–326 | 16.2–18.0 | 15.4–16.7 | 12.7–14.1 |
| R = 750 m | even disk | 576 | 9.8 | 8.5 | 7.3 |
| | triangles | 579–689 | 8.8–9.8 | 7.8–8.8 | 6.6–7.3 |
| R = 1 km | even disk | 1,031 | 6.1 | 5.0 | 4.3 |
| | triangles | 1,031–1,172 | 5.8–6.1 | 4.9–5.4 | 4.2–4.5 |

**Readout: within 5° of all showers** (counting)

| Region | Layout | All read out, 0 µs | All read out, 10 µs | kNN gate, 0 µs | kNN gate, 10 µs |
|---|---|---|---|---|---|
| R = 500 m | even disk | 16.0 % | 2.7 % | 12.8 % | 6.2 % |
| | triangles | 12.6–17.0 % | 3.1–4.7 % | 8.9–11.7 % | 4.7–7.4 % |
| R = 750 m | even disk | 13.2 % | 2.1 % | 8.5 % | 4.7 % |
| | triangles | 10.4–12.9 % | 2.8–4.0 % | 7.8–8.7 % | 5.2–7.4 % |
| R = 1 km | even disk | 11.0 % | 3.4 % | 8.2 % | 5.8 % |
| | triangles | 9.5–10.2 % | 2.9–4.0 % | 6.4–7.8 % | 5.1–5.5 % |

**Event trigger: ≥ 3 detectors** (all read out, 0 µs)

| Region | Layout | ≥ 1 drawn particle: triggered | … and within 5° | μ ≥ 1 (notebook): triggered | … and within 5°, no counting | … and within 5°, counting |
|---|---|---|---|---|---|---|
| R = 500 m | even disk | 90 % | 15.6 % | 81 % | 56.4 % | 14.8 % |
| | triangles | 84–88 % | 12.6–16.4 % | 74–81 % | 52.5–58.7 % | 10.7–15.6 % |
| R = 750 m | even disk | 75 % | 12.2 % | 58 % | 42.5 % | 10.2 % |
| | triangles | 71–73 % | 9.9–12.2 % | 55–63 % | 40.3–45.1 % | 8.2–10.4 % |
| R = 1 km | even disk | 61 % | 10.0 % | 43 % | 32.8 % | 7.6 % |
| | triangles | 58–63 % | 8.7–9.4 % | 41–55 % | 31.2–41.1 % | 6.7–7.9 % |

Columns:
- **≥ 1 drawn particle: triggered:** share of showers where ≥ 3 detectors actually received at least one particle
  (particles drawn at random around the expected number).
- **… and within 5°:** share of all showers that triggered by that rule and were also reconstructed within 5° (inputs
  counted).
- **μ ≥ 1 (notebook): triggered:** share of showers where ≥ 3 detectors *expect* at least one particle (the SWGOLO
  notebook's rule with threshold 1 instead of 5; no randomness).
- **… and within 5°, no counting:** share of all showers that pass the μ ≥ 1 trigger and are reconstructed within 5°
  from the noise-free kernel input.
- **… and within 5°, counting:** the same, with the drawn particle counts as recon input.

## 5. Getting a SWGO-like layout

The layout comes from the utility. The shower setup decides whether the utility has anything to trade.

| | SWGO | TambOpt (run 19) | Effect |
|---|---|---|---|
| Cores vs allowed detector area | Cores ±3 km, array ≤ 1 km (notebook); array + 2 km (C++) | Both = the 12 km plane | SWGO: spreading gains showers → expansion, hollow centre. TambOpt: covering the plane is optimal → even grid |
| Detection threshold | ≥ 5 particles per unit, ≥ 3 units (r_b) | None: every detector reads the noise-free expectation | Without a threshold, density cannot pay off |
| Clusters | r_b step + `push_apart` (notebook); imposed (C++) | — | Triplets need a hard step on expected counts (§ 4) |
| Recon | Likelihood fit, any layout | DeepSets on the 12 km layout zoo (incl. 30–600 m clusters, 400/800 m blobs) | — |

Why the 500 m disk does not beat the grid (§ 3 table):
- **Over all showers:** the disk sees 256 of 5,000 axes, so it is capped at ~5 % of showers whatever its quality.
- **Per shower it covers, no counting:** 6.6 / 65.6 % vs the grid's 10.4 / 73.5 % within 1° / 5°. Grid detectors get
  a usable time below one particle (run 19: the ≥ 3-detector trigger drops the grid from 10.4 to 4.4 % within 1°), and
  a 12 km baseline times better than 1 km.
- **Per shower it covers, counting:** the disk wins, 12.9 vs 4.9 % within 5° (18 vs 0.6 detectors with a particle).
  This is where density pays, but the recon is not yet trained on counted inputs.

To give the optimiser SWGO's trade-off:
1. Counted inputs and a recon fine-tuned on them (54580014).
2. Detectors confined to the site (Malata-like, R ≈ 0.5–1 km); showers whose axis lands within the site + a ~2 km
   margin (run 19 showers, selected); utility averaged over all of them, missed showers = 0, times r_b (k = 3).
   Spreading then buys acceptance and packing buys resolution.

Check without training (job 54622674, `swgoobj_S{1500,3000,0}.json`): fixed layouts scored with the SWGO-like
objective, U5 = share of all showers in a fixed set (axis within S of the plane centre) that pass the expected-count
trigger (≥ 3 detectors with μ ≥ 1) and are reconstructed within 5°; noise-free recon input, so the recon is used as
trained. 99 detectors.

| Layout | U5, S = 1.5 km (518 showers) | U5, S = 3 km (1,784) | U5, whole plane (5,000) | Within 1°, no trigger, whole plane |
|---|---|---|---|---|
| Disk 300 m | 5.4 % | 1.7 % | 0.8 % | 0.6 % |
| Disk 500 m | 9.8 % | 3.3 % | 1.5 % | 1.0 % |
| Disk 1 km | 19.9 % | 6.5 % | 2.5 % | 1.8 % |
| Disk 1.5 km | 20.1 % | 7.3 % | 2.8 % | 2.3 % |
| Disk 2 km | 14.5 % | 7.1 % | 2.7 % | 3.8 % |
| Disk 3 km | 5.6 % | 4.5 % | 1.9 % | 6.4 % |
| Disk 6 km | 0.6 % | 1.3 % | 1.3 % | 11.6 % |
| Grid | 1.0 % | 1.5 % | 1.3 % | 9.7 % |
| Triangles r = 10 m, R = 1 km | 23.7 % | 7.2 % | 2.8 % | 1.6 % |

- With the trigger the objective peaks at an intermediate radius (1–2 km), on every shower set, the whole plane
  included; without it the score grows with the radius (grid / 6 km disk best). The trigger alone creates the
  density–area trade-off; the shower region shifts the optimum (1–1.5 km for S = 1.5 km, 1.5–2 km for S ≥ 3 km).
- Triangles are level with even disks (+3.8 pts at S = 1.5 km, within noise for 518 showers; equal elsewhere).
- Only ≤ 31 % of showers trigger even at the best radius (≤ 4 % on the whole plane): with 2 m² detectors μ ≥ 1 is rare.
- Noise-free recon input favours sparse layouts (sub-particle signals), so the peak is conservative.
- With energy also required (|Δlog₁₀E| < 0.2, the two-tier cut; job 54627380) every score drops by ~⅓ and the peak
  stays where it is (S = 1.5 km: disk 1–1.5 km 13.7 %, grid 0.4 %).

**Donuts and detector area** (jobs 54626904, 54627380, 54628915, 54629473; files `swgoobjE_S…`, `area_mu0p1_S…`,
`area_mu0p01_S…`). Donuts: 99 detectors evenly over r_in – R at the scale of the best disk. Area: the trigger cut μ ≥ 1 /
0.1 / 0.01 on the 2 m² expected counts = a 2 / 20 / 200 m² detector (recon input unchanged). Triggered and within 5° /
within 1°, showers within 1.5 km:

| Layout | 2 m² | 20 m² | 200 m² |
|---|---|---|---|
| Disk 1.5 km | 20.1 / 2.5 % | 43.1 / 6.0 % | 68.5 / 9.5 % |
| Donut 0.25–1.5 km | 19.1 / 4.6 % | 42.5 / 8.7 % | 68.5 / 11.4 % |
| Donut 0.5–1.5 km | 20.5 / 3.1 % | 40.5 / 6.6 % | 67.8 / 9.8 % |
| Donut 1–1.5 km | 18.0 / 2.3 % | 36.3 / 4.6 % | 60.8 / 8.9 % |
| Disk 2 km | 14.5 / 2.1 % | 37.8 / 7.7 % | 68.3 / 14.7 % |
| Donut 0.25–2 km | 13.9 / 2.9 % | 37.1 / 9.3 % | 66.8 / 16.0 % |
| Donut 1.5–2 km | 8.7 / 0.2 % | 22.8 / 1.9 % | 42.3 / 4.6 % |
| Triggered, disk 1.5 km | 25 % | 51 % | 81 % |

Best disk (triggered and within 5°) by area:

| Shower set | 2 m² | 20 m² | 200 m² | Grid, 200 m² |
|---|---|---|---|---|
| Within 1.5 km | 1–1.5 km (20 %) | 1.5 km (43 %) | 1.5–2 km (68 %) | 26 % |
| Within 3 km | 1.5–2 km (7 %) | 2 km (19 %) | 3 km (42 %) | 24 % |
| Whole plane | 1.5 km (3 %) | 2–3 km (8 %) | 4.5 km (20 %) | 21 % |

- A hole does not pay at 5° at any area, even at 200 m² where 81 % of showers trigger (~13 detectors with μ ≥ 0.01).
  A small hole (0.25 km) is ahead at 1° by 1–3 pts in every case (each ~2σ, same sign everywhere); larger holes lose.
- More area spreads the best layout out (whole plane at 200 m²: 4.5 km disk ≈ grid), towards even coverage, not a
  donut. Detector area alone does not give SWGO's donut; the remaining difference is the geometry (SWGO cores uniform
  beyond a compact array; here showers selected by axis distance on a tilted plane, km-long footprints).

## 6. Status and open (2026-10-22)

**Stage 4 with the SWGO-like objective**

Branch `swgo-trigger` (commit df7ae14, worktree `.claude/worktrees/swgo-trigger`). `TAMBOPT_UTILITY=swgo_trigger`: per shower r_b × (g + 0.3 (1 − g) q), g = soft within
5°, q the two-tier partial credit; r_b = P(≥ 3 detectors with ≥ 1 particle), exact Poisson-binomial over
p_i = 1 − e^(−μ_i) (`counting.trigger_prob`; the sampled trigger rate in expectation, smooth). μ_i = expected detected
particles at the detector's distance from the axis, with gradients, scaled to `TAMBOPT_TRIG_AREA` m²; a shower that
does not trigger scores 0. Recon input unchanged (noise-free), so the current recon is used as trained. Options
`TRIG_K`, `TRIG_AREA`, `TRIG_ENERGY` (`kernel_opt_run19.sh … swgotrig`). Tests: trigger_prob vs Poisson draws, gradient.
Smoke (54634710, 30 steps from the grid): runs, 2.5 s/step, 3 % of showers trigger (as in the check).

| Jobs (gpu_requeue) | Start | Area | Steps × showers | Expected if the objective works |
|---|---|---|---|---|
| 54634922 `swgotrig_grid_a2` | grid | 2 m² | 5,000 × 2,048 | contracts towards a ~1.5 km disk |
| 54634923 `swgotrig_center_a2` | centre | 2 m² | 5,000 × 2,048 | expands to ~1.5 km |
| 54634924 `swgotrig_grid_a20` | grid | 20 m² | 5,000 × 2,048 | ~2–3 km disk (whole plane) |

Risk: from the grid only ~3 % of showers trigger, so the trigger gradient is weak and Adam may random-walk (run 19).

**Open:**
- The three `swgotrig` runs above: do they contract to a ~1.5 km disk? Then score the result as in § 5.
- Recon fine-tuned on counted inputs, only on triggered rows (≥ 3 detectors hit) of site-density layouts.
- Counted results vs shower energy.
- Gradient direction check on the kernel grid runs.
- √2·|t1| ring radius (branch `ring-sqrt2`): new run folder, Stage 1 relabel, recon retrain.
- Calibration from π⁻ showers only.

**Code state:**
- `bg-kernel-units`: not pushed, not merged; free to merge (the `ko19x` jobs that needed the old main checkout are
  cancelled).
- `ring-sqrt2`: 1 commit on top of `bg-kernel-units`, not pushed.
- `swgo-trigger`: 1 commit on top of `bg-kernel-units` (the swgo_trigger utility), not pushed.
- Main checkout, uncommitted: lit-threshold option (`score_all.py`, `analyze.py`, `detector_count_grid.py`),
  best-per-family and triplet-split figures, `e_distribution.py`, two slurm scripts. The summary deck builder lives in
  `detector_optimization_v6/figures/` (outside the repo).
- Housekeeping: old worktree `.claude/worktrees/r19-nandebug`.

## 7. Cancelled jobs (2026-10-22)

Cancelled before they started (none had run):

| Jobs | What | Why cancelled |
|---|---|---|
| 54580014 → 54580015 `r19_count_ft`, `r19_count_rescore` | Fine-tune the recon on counted inputs (rows of the 12 km zoo), then rescore | Most rows have 0–2 hits (~3 % reach 3 detectors), so it would mostly learn the prior; replaced by the swgo_trigger optimisation with the current recon |
| 54580013 `r19_count_score` | Counting scores of all run 19 layouts with the current recon | A lower bound only (recon not trained on counted inputs); 54580015 rescores the same with the fine-tuned recon |
| 16 `ko19x_{ker,sur}_{grid,center}_l{0,0p3}_bg{1000,10000}` | Noise-aware optimisations (two-tier λ 0 / 0.3, kNN gate, 1 / 10 µs background), kernel and surrogate | Same objective that only finds the grid (§ 5); old background units; likely the grid again |
| 4 `ko19x_ker_center_…_b` (54069522–27) | Continuations of the kernel centre starts (afternotok) | Their parents were cancelled |

To restart (exact submit lines in `<run 19>/diagnostics/cancelled_jobs_2026-10-22.txt`; from the main checkout; after merging `bg-kernel-units`, add `TAMBOPT_BG_UNITS=count` to keep the background they were meant to test):

```
cd ~/tambo/TambOpt
R19=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/19_plane45S_12km_showerframe_400k
X=BATCH=4096,GATE=knn23,FREEZE_UNSEEN=1,CHUNK=64,LBFGS_CHUNKS=17,LBFGS_EVENTS=6000
for init in grid center; do N=$([ $init = grid ] && echo 5000 || echo 10000)
 for lam in 0 0.3; do for bg in 1000 10000; do t=${lam/./p}
  sbatch -J ko19x_ker_${init}_l${t}_bg$bg --export=ALL,$X,BG_WINDOW=$bg,LBFGS_MAXITER=150 \
    slurm/kernel_opt_run19.sh $init twotier $lam 10 $N
  sbatch -J ko19x_sur_${init}_l${t}_bg$bg \
    --export=ALL,$X,BG_WINDOW=$bg,RESPONSE=surrogate,FNN_DIR=$R19/test_v6_run_02_recentered,LBFGS_MAXITER=1500 \
    slurm/kernel_opt_run19.sh $init twotier $lam 10 $N
done; done; done
# continuation of a kernel centre start <ID> (same -J with _b, same --export, plus):
#   sbatch -t 1-12:00:00 -d afternotok:<ID> --export=ALL,PREV_JOB=<ID>,... slurm/kernel_opt_run19.sh center ...
# counting score with the current recon (from this worktree):
#   sbatch -J r19_count_score slurm/counting_run19.sh score
```
