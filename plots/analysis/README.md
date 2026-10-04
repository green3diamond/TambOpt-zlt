# Analysis figure scripts (runs 15 and 17)

Copies of the one-off figure and job scripts behind `figures/findings_after_run13.pdf`, kept here
as examples. The live copies sit next to their data in
`/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/<run>/...`
(path given per folder below). Most of them find their inputs relative to their own location
(`os.path.dirname(__file__)`) or through absolute data paths, so run the live copy, or copy one
back next to its data, rather than running it from here.

The reusable pieces they call are in the repo proper: `scripts/diagnostics/score_layouts.py`,
`detector_count_scan.py`, `gate_scan.py`, `plots/layouts/true_utility.py`, `modules/`.

| Folder | Live copy | Makes |
|---|---|---|
| `run15/` | `15_sigma10_interp_hurdle_10pct/diagnostics/` | `plot_det20*.py`, `plot_det100.py`, `plot_detector_count_scan.py`: detector-count scan summaries (20 / 50 / 100 detectors, 5° and 2.5°); `norm_factor.py`: kernel normalising-factor stats (run 16) |
| `run15/plot_kernel_utility_vs_epoch.py` | `15_sigma10_interp_hurdle_10pct/` | utility vs step for the kernel-in-the-loop runs |
| `run15/core_test/` | `15_sigma10_interp_hurdle_10pct/diagnostics/core_test/` | hand-made cores / hybrids vs grid and optimized layouts |
| `run17/core_test/` | `17_plane45_interp_hurdle_100k/diagnostics/core_test/` | same core test on the 45° plane, kernel-optimized layouts, kernel-trained recon; `score_core_test.sh` scores and plots (split into cores / hybrids / kernel-optimized pages) |
| `run17/core_test_bwcap50/` | `17_plane45_interp_hurdle_100k/diagnostics/core_test_bwcap50/` | the core-test layouts rescored with kernel widths capped at 50 m (`TAMBOPT_KERNEL_BW_MAX=50`); utility only, uncapped vs capped |
| `run17/array_size_single_shower/` | `17_plane45_interp_hurdle_100k/diagnostics/array_size_single_shower/` | one shower (core at the centre / ~600 m off) seen by 100-detector disks of growing radius + grid: deposit and arrival time per detector, recon errors, uncapped vs 50 m-capped widths; `cloud_axis_check.py`: placed clouds' axis vs labelled direction |
| `run17/zenith_cap100/` | `17_plane45_interp_hurdle_100k/diagnostics/zenith_cap100/` | zenith <= 100 test: cloud-axis tilt vs angle from West (C8 vs generated), grid recon by zenith band (uncapped / 50 m cap), run 18 geometry with the cut; array-size figures for two zenith <= 100 showers |
| `run17/kernel_frame/` | `17_plane45_interp_hurdle_100k/diagnostics/kernel_frame/` | global-frame kernel (as used) vs layer-depth fix vs shower-frame kernel, footprints on the run 17 plane and Malata for 3 showers (`--cap`); `kernel_schematic.py`: how a recorded point is stretched now vs in the shower frame |
| `run17/array_size_fixed_kernel/` | `17_plane45_interp_hurdle_100k/diagnostics/array_size_fixed_kernel/` | the array-size test redone (zenith <= 100, >= 30 deg from the C8 field, disks on the core) with the shower-frame kernel, the layer depth fixed and optionally interpolated arrival times; `array_size_ablation.py`: recon with times / energies / all signals removed |
| `run17/stage4_surrogate/` | `17_plane45_interp_hurdle_100k/archive/stage4_surrogate_2026-10-01/` | Stage 4 through the surrogate summary (layouts + metrics); `plot_decays_3d.py`: 3-D decays and directions over the plane |
| `run17/recon_snapshot/` | `17_plane45_interp_hurdle_100k/archive/stage4_surrogate_2026-10-01/diagnostics/recon_snapshot/` | recon conditional plots from a mid-training checkpoint |
| `findings_pdf/` | `figures/` | `make_findings_pdf.py` builds the findings PDF from existing PNGs; `build_findings_pdf.sh` runs it as a job |

Run 17 scripts need the main checkout on a run 17 branch (`run17-plane` or `run17-bwcap50`), since
`modules/constants.py` there points at the run 17 folder.
