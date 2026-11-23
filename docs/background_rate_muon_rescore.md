# Background rate: rescoring with cosmic muons only (23 Nov 2026)

Paper 2 uses 1,826 /m^2/s (SWGO all-particle rate at ~4.8 km). Here every layout with background was rescored at
the cosmic-muon rate at 3-4 km, 400 /m^2/s (`TAMBOPT_BG_RATE=4e-7`), as a lower bound. Everything else as res3
(score_all.py, 5,000 FM showers or all C8 test showers). Branch `bg-muon-rate`; `scripts/paper2/bgrate_rescore.sh`;
scores in `tmp_rings/res4` (res3 with the 16 background families replaced). Check: at the default rate the grid
reproduces res3 exactly.

Background hits per detector and window (2 m^2 x rate x W): 0.0037 (1 us) and 0.037 (10 us) now; 0.0008 and 0.008
with muons. All charged particles at 3-4 km (~700-1,000 /m^2/s, rough) would lie in between.

## Results: % of showers within 1 deg, current rate -> muon rate

kNN 2-of-3 gate (paper 2 default):

| Site | Layout | No bg | 1 us | 10 us |
|---|---|---|---|---|
| Malata FM | grid | 34.8 | 23.5 -> 29.8 | 3.5 -> 18.7 |
| Malata FM | opt current, ugrid start | 36.8 | 24.6 -> 30.7 | 4.0 -> 19.1 |
| Malata FM | opt swgotrig, grid start | 36.3 | 23.6 -> 29.9 | 3.7 -> 18.7 |
| Malata FM | opt swgotrig, centre start | 36.4 | 23.8 -> 29.5 | 3.6 -> 18.5 |
| Malata FM | disk r400 | 24.2 | 17.4 -> 21.6 | 2.5 -> 14.4 |
| Malata FM | disk r150 | 11.7 | 9.2 -> 11.1 | 1.7 -> 7.3 |
| Malata C8 | grid | 30.8 | 20.3 -> 25.9 | 3.1 -> 15.3 |
| Malata C8 | opt swgotrig, grid start | 32.1 | 20.7 -> 26.4 | 2.6 -> 16.4 |
| 12 km FM | ugrid | - | 8.9 -> 9.3 | 4.6 -> 8.2 |
| 12 km FM | opt current, ugrid start | - | 8.9 -> 9.4 | 4.8 -> 8.3 |
| 12 km FM | opt two-tier, grid start | - | 7.6 -> 8.0 | 4.3 -> 7.1 |
| 12 km C8 | ugrid | 21.8 | 15.2 -> 16.2 | 7.9 -> 13.8 |
| 12 km C8 | opt current, ugrid start | 20.6 | 16.1 -> 17.3 | 8.9 -> 14.7 |
| 12 km C8 | opt two-tier, grid start | 19.4 | 14.3 -> 15.1 | 7.3 -> 12.9 |
| 12 km C8 | core r300 | 1.0 | 0.8 -> 0.9 | 0.3 -> 0.5 |

No gate:

| Site | Layout | 1 us | 10 us |
|---|---|---|---|
| Malata FM | grid | 25.0 -> 32.6 | 2.9 -> 19.2 |
| Malata C8 | grid | 22.0 -> 28.6 | 1.9 -> 15.8 |
| 12 km C8 | ugrid | 14.3 -> 20.4 | 0.7 -> 9.6 |
| 12 km C8 | opt current, ugrid start | 14.3 -> 19.3 | 0.5 -> 9.2 |

Rank correlation of all layouts (current vs muon rate), with the gate: 1 us 0.96-1.00; 10 us 0.97-0.98 on the 12 km
plane, 0.46 (C8) and 0.57 (FM, all 65 layouts) on Malata, where the current-rate 10 us scores are all ~3 % and barely ordered.
Without the gate: 0.35-0.99.

## What changes

- 1 us: the background loss roughly halves on Malata, < 1 point on the 12 km plane; layout order unchanged.
- 10 us: now every Malata layout collapses to ~3 %; with muons they keep about half their no-background accuracy
  (15-19 %) and the no-background order returns (optimized ~ grid > disk r400 > disk r150).
- kNN gate: with muons it no longer helps on Malata (no gate is 1-3 points better) or on the 12 km plane at 1 us
  (~4 points worse); it still helps on the 12 km plane at 10 us (13.8 vs 9.6 %).
- Qualitative conclusions of paper 2 hold (grid and optimized layouts on top, background ranking = no-background
  ranking). No re-optimization needed: the `sur5_1us_*` rankings are stable.

All 16 families done; res4 built (872 records replaced).
