"""PDF: run 19 (12 x 12 km South-facing plane): setup, recon, all layout-test results, density vs area at
1 / 2.5 / 5 deg, and the comparison with run 18. Same page style as make_zoo_recon_pdf.py.

    python figures/make_run19_pdf.py   (any Python with Pillow + matplotlib)
"""
import glob
import json
import os
import re

import matplotlib
from PIL import Image, ImageDraw, ImageFont

B = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/"
R19 = B + "19_plane45S_12km_showerframe_400k/"
T = R19 + "diagnostics/tests/"
P = R19 + "diagnostics/pdf/"
SNAP = R19 + "diagnostics/recon_snapshot_ep230/"
REC = R19 + "test_v6_run_03_recentered_deepsets/"
Z18 = B + "18_plane45S_6km_showerframe_200k/diagnostics/zoo_compare/reltime1_wd003/"
OUT = B + "figures/run19_12km.pdf"
W, H, M = 3200, 1800, 90
IMG_H = 1400
TTF = os.path.join(matplotlib.get_data_path(), "fonts/ttf/")
FONT = ImageFont.truetype(TTF + "DejaVuSans.ttf", 46)
HEAD = ImageFont.truetype(TTF + "DejaVuSans-Bold.ttf", 72)
TITLE = ImageFont.truetype(TTF + "DejaVuSans-Bold.ttf", 110)
MONO = ImageFont.truetype(TTF + "DejaVuSansMono.ttf", 40)
INK = "#1F2328"


def page(paths, bullets):
    p = Image.new("RGB", (W, H), "white")
    imgs = [Image.open(x).convert("RGB") for x in paths]
    gap = 80
    img_h = IMG_H - max(0, len(bullets) - 3) * 66
    avail_w = W - 2 * M - gap * (len(imgs) - 1)
    h = min(img_h, int(avail_w / sum(i.width / i.height for i in imgs)))
    rs = [i.resize((int(i.width * h / i.height), h), Image.LANCZOS) for i in imgs]
    x = (W - (sum(r.width for r in rs) + gap * (len(rs) - 1))) // 2
    for r in rs:
        p.paste(r, (x, M))
        x += r.width + gap
    d = ImageDraw.Draw(p)
    y = M + h + 50
    for b in bullets:
        d.text((M + 60, y), "•  " + b, fill=INK, font=FONT)
        y += 66
    return p


def text_page(heading, bullets):
    p = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(p)
    d.text((200, 160), heading, fill=INK, font=HEAD)
    for k, b in enumerate(bullets):
        d.text((210, 330 + 88 * k), ("•  " + b) if b else "", fill=INK, font=FONT)
    return p


def table_page(heading, rows, notes):
    p = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(p)
    d.text((200, 150), heading, fill=INK, font=HEAD)
    y = 320
    for k, row in enumerate(rows):
        d.text((210, y), row, fill=INK if k else "#52514e", font=MONO)
        y += 58
    y += 40
    for b in notes:
        d.text((210, y), "•  " + b, fill=INK, font=FONT)
        y += 66
    return p


def section_page(title, subtitle):
    p = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(p)
    d.text((200, 700), title, fill=INK, font=TITLE)
    d.text((210, 880), subtitle, fill="#52514e", font=FONT)
    return p


def full_page(path):
    img = Image.open(path).convert("RGB")
    k = min((W - 2 * M) / img.width, (H - 2 * M) / img.height)
    img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
    p = Image.new("RGB", (W, H), "white")
    p.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return p


S = json.load(open(T + "scores.json"))["scores"]
KEY = {1: "within1", 2.5: "within2p5", 5: "within5"}
DA = json.load(open(T + "density_area.json"))
G18 = json.load(open(Z18 + "detcount/detector_count_grid.json"))
G19 = json.load(open(T + "detcount/detector_count_grid.json"))
L = json.load(open(REC + "recon_train_log.json"))


def best(n, key="within1"):
    xs = [x for x in S if S[x]["n_det"] == n]
    return max(xs, key=lambda x: S[x][key])


pages = [section_page("Run 19: 12 × 12 km plane", "2026-10-13 · branch run19-12km · 421,200 showers · 100 detectors · 125 test layouts × 5,000 held-out showers")]

b100, b50, b20, b10 = (best(n) for n in (100, 50, 20, 10))
pages.append(text_page("Summary", [
    f"100 det.: the 6 km disk (covers 92% of the plane) edges out the full grid: "
    f"{100 * S[b100]['within1']:.1f} vs {100 * S['grid_n100']['within1']:.1f}% within 1°",
    "With fewer detectors a partial-coverage disk wins clearly: 4.5 km (62% of the plane) for 50, 3 km (28%) for 20",
    "The full-plane grid ranks 2nd of 34 (100 det.), 5th of 34 (50), 9th of 40 (20), 9th of 14 (10)",
    "Density vs area (1°): growing the disk with N (+0.8 / +3.7 / +4.6 pts) beats denser disks (+0.9 / +2.8 / +3.4)",
    "Lit-detector spacing that works best: ~450–800 m (run 18: 300–600 m), ~6–9 lit detectors per shower",
    "Everything is lower than run 18 (100-det. grid 10.4% vs 14.6% within 1°): 4× the area for the same 100 detectors",
    "At the same ~1 km spacing (run 18: 25 det., run 19: 100 det.), run 19 does better: 10.4% vs 4.2% within 1°",
    "Recon: direction is good where showers hit ≥ 6 detectors; 58% of training rows hit none (compact training layouts)",
]))

# ── Setup ────────────────────────────────────────────────────────────────────
pages.append(section_page("Setup", "Plane, showers, training layouts, recon and the layout tests"))
pages.append(text_page("Setup: plane and showers", [
    "Plane: 12 × 12 km at 45°, facing South (horizontal: East ±5.9 km, North 0–8.4 km); run 18 was 6 × 6 km",
    "Showers: 421,200 tau decays (2× run 18), 5% held out; electron, muon and photon components (C8 / AllShowers)",
    "Decays: Latin hypercube in a box 0.5–6.5 km in front of the plane, aimed at uniform points on it",
    "     cuts: zenith 60–100°, ≥ 30° from the C8 field axis, decay–target distance ≤ 11.5 km (C8 covers 12 km)",
    "     distance p5 / p50 / p95: 4.5 / 9.0 / 11.3 km (run 18 median 6.5 km) · zenith p5 / p50 / p95: 65 / 86 / 99°",
    "     energy: log10(E / GeV) uniform in 5–7",
    "Detector response: shower-frame kernel, ring showers, times interpolated between C8 planes, widths k = 8 (≤ 50 m)",
    "Detectors: 100 (the full-plane grid is ~1.2 km × 0.84 km apart)",
    "Commands: scripts/17_make_plane_inputs.py --side 12000 --box-depth 6000 --max-dist 11500 --n-events 421200",
    "     --facing south --zenith 60 100 --min-angle-b 30 --seed 19; slurm/run19_pipeline.sh gen → build → recon → tests",
]))
pages.append(page([R19 + "inputs/plane_primaries.png"], [
    "Inputs: decay vertices, targets on the plane, distances and directions of the 421,200 primaries"]))
pages.append(text_page("Setup: training layouts, recon and tests", [
    "Training layouts: 11 strategies; every shower is labelled under each (one layout per batch of 60 showers):",
    "     centre blobs σ 400 / 800 m · uniform random ×2 · Latin hypercube ×2 · clusters ×2 (30–600 m)",
    "     disk (radius up to 8 km) · grid patch ×2 (side 0.5–12 km, random position)",
    "Recon: DeepSets (431k parameters), relative-time input reltime1 (dt from the earliest timed detector with ≥ 1 count)",
    "     AdamW wd 0.03, lr 3e-5, batch 256, 11.8M training rows; Adam stopped by hand at epoch 252 (val flat since ~165)",
    f"     then L-BFGS from the Adam best (epoch 230): val {L['adam_best_val']:.4f} → {L['best_val_total']:.4f} in 218 closure calls",
    "Tests: 125 layouts (grids N = 4–100; centred disks N = 10/20/50/100, r 0.1–12 km; cores; hybrids; clusters)",
    "     scored on 5,000 held-out showers (±0.4–0.5 pts at 1°), each layout's own N detectors given to the recon",
    "Metrics: share of showers with angle error < 1 / 2.5 / 5° (angle only); median angle error",
    "     lit = detector with ≥ 1 count; NN spacing / baseline = medians over showers with ≥ 2 lit",
]))
pages.append(full_page(P + "training_strategies_batch0.png"))

# ── Recon ────────────────────────────────────────────────────────────────────
pages.append(section_page("Recon", "Training and target vs prediction on the validation rows (kernel input)"))
pages.append(page([P + "recon_training.png"], [
    "Val loss 0.1323 vs 0.0769 in run 18: the direction part is 2.5× higher (0.079 vs 0.031), the energy part 0.054 vs 0.046",
    "Adam kept 'improving' by tiny amounts (any gain > 1e-6 resets the patience): stopped at 252, then L-BFGS (−0.6%)"]))
pages.append(page([REC + "recon_target_vs_pred_kernel.png", REC + "recon_conditional_kernel.png"], [
    "All 1.32M val rows, final recon: a tight diagonal plus flat bands at the average answer (dir_x ≈ 0, dir_z ≈ 0.07, log E ≈ 0.49)",
    "The bands are showers that hit no detector: 57.8% of val rows (compact training layouts miss most of the 12 km plane)"]))
pages.append(table_page("How many detectors the training showers hit", [
    "hit detectors (> 1 particle)     share of val rows",
    "0                                57.8%",
    "≥ 1                              42.2%",
    "≥ 3                              23.5%",
    "≥ 6                              11.6%",
    "≥ 10                              6.5%",
    "≥ 20                              3.0%",
], ["Val rows = 1.32M shower × training-layout pairs (11 strategies); Adam-best recon (epoch 230)",
    "Most rows carry no signal, so the val loss mixes 'cannot know' rows with real reconstructions"]))
pages.append(page([SNAP + "recon_target_vs_pred_kernel_lit6.png", SNAP + "recon_conditional_kernel_lit6.png"], [
    "Only val rows with ≥ 6 detectors hit (153k): the bands are gone; dir_x and dir_y on the diagonal, dir_z slightly wider",
    "Energy tracks the truth above log_e_norm ≈ 0.4; low energies are biased upward and the top saturates near 0.9",
    "Adam-best recon (epoch 230, before L-BFGS)"]))

# ── Tests ────────────────────────────────────────────────────────────────────
pages.append(section_page("Layout tests", "125 layouts × 5,000 held-out showers, the run's final recon"))
pages.append(page([T + "disks_vs_radius.png"], [
    "Centred disks: the best radius grows with N: 1.5–2 km (10), 3 km (20), 4.5 km (50), 6 km (100)",
    "Past the best radius the share falls fast: spacing passes the ~1 km footprint and showers light too few detectors",
    "100 det.: 6 km disk 11.9 / 45.1 / 77.4% at 1 / 2.5 / 5°; full grid 10.4 / 40.3 / 73.5%"]))

RADII = (1000, 1500, 2000, 3000, 4500, 6000, 9000)
for th in (1, 2.5, 5):
    g = {(int(n), int(r)): v for n, dd in DA["grid"][f"{th:g}"].items() for r, v in dd.items()}
    rows = ["detectors \\ radius   " + "".join(f"{r / 1e3:g} km".ljust(9) for r in RADII),
            "plane covered          " + "".join(f"{100 * DA['cover'][str(r)]:.0f}%".ljust(9) for r in RADII)]
    for n in (10, 20, 50, 100):
        bb = max(RADII, key=lambda r: g[(n, r)])
        rows.append(f"{n:<23d}" + "".join((f"[{g[(n, r)]:.1f}]" if r == bb else f"{g[(n, r)]:.1f}").ljust(9) for r in RADII))
    dens = [(10, 20, 2000), (20, 50, 3000), (50, 100, 4500)]
    area = [((10, 1500), (20, 2000), "1.5 → 2 km"), ((20, 2000), (50, 3000), "2 → 3 km"), ((50, 3000), (100, 4500), "3 → 4.5 km")]
    rows += ["", "step        more density, same area             more area, same density (radius ∝ √N)"]
    for (a, b, r), ((n0, r0), (n1, r1), lab) in zip(dens, area):
        rows.append(f"{a} → {b}".ljust(12) + f"r {r / 1e3:g} km: {g[(a, r)]:.1f} → {g[(b, r)]:.1f} ({g[(b, r)] - g[(a, r)]:+.1f})".ljust(36)
                    + f"{lab}: {g[(n0, r0)]:.1f} → {g[(n1, r1)]:.1f} ({g[(n1, r1)] - g[(n0, r0)]:+.1f})")
    notes = ["[ ] = best radius per detector count · 4.5 km and larger exceed the plane's 4.2 km N–S half-width (detectors snap to the edge)",
             "Plane covered: share of the plane within r of its centre · 5,000 held-out showers, run 19 recon"]
    if th == 1:
        notes.insert(0, "Unlike run 18, area still pays at 100 detectors (3 → 4.5 km); both paths gain at every step in the table")
    pages.append(table_page(f"Disks: more density or more area? · within {th:g}°", rows, notes))

pages.append(page([T + "best_per_family.png"], [
    "Best of each family per detector count, within 1° (top) and 5° (bottom): disk > grid ≈ cluster > hybrid > core at every N",
    "Compact cores (r ≤ 500 m) stay at ≤ 1% within 1° even with 50–100 detectors"]))
pages.append(page([T + "detcount/detector_count_grid.png"], [
    "Full-plane grid vs detector count: 100 det. (~1 km apart) light a median 2 detectors per shower; 66% of showers light < 3",
    "Within 5°: 11% (10 det.) → 53% (50) → 74% (100)"]))
pages.append(page([T + "lit_resolution.png"], [
    "Spread layouts: median error falls to ~2.8° at 6–9 lit detectors, then rises (many lit = tight clusters near the core)",
    "Share within 1° peaks at ~12% with 8–9 lit; most showers light 0–3 detectors on this plane"]))
pages.append(page([T + "lit_geometry.png"], [
    "Best nearest-neighbour spacing of the lit detectors: ~450–800 m (6–15 lit: ~12.5% within 1°, median ~2.6–2.9°)",
    "Run 18 had 300–600 m; likely because the decays are farther here (median 9 vs 6.5 km), so footprints are wider"]))
pages.append(section_page("Test layouts", "All 125 layouts by detector count, ranked best to worst by the share within 1°"))
for f in sorted(glob.glob(T + "layout_catalog_p*.png")):
    pages.append(full_page(f))

# ── Run 18 vs run 19 ─────────────────────────────────────────────────────────
pages.append(section_page("Run 18 vs run 19", "6 km vs 12 km plane, same recon recipe (DeepSets, reltime1, wd 0.03)"))
pages.append(page([P + "run18_vs_run19.png"], [
    "Top: disks vs radius (solid run 19, dashed run 18): the optimum moves out from ~3 km to ~4.5–6 km and drops in height",
    "Bottom: full-plane grid; orange = run 18 with N/4 detectors, i.e. the same detectors per area as run 19 at N"]))
rows = ["                  run 18 (6 km)                      run 19 (12 km)",
        "detectors         grid 1°/2.5°/5°     best 1°           grid 1°/2.5°/5°     best 1°"]


def s18(n):
    d = {}
    for st in ("disks_n10", "core_test_ndet", "angleonly"):
        for k, v in json.load(open(f"{Z18}{st}/scores_pad_1deg.json")).items():
            if (S18 := re.fullmatch(r"disk_r(\d+)(?:_n(\d+))?", k)) and int(S18.group(2) or 100) == n:
                d[int(S18.group(1))] = 100 * v["frac_good_t"]
    r = max(d, key=d.get)
    return r, d[r]


for n in (10, 20, 50, 100):
    a, b = G18[str(n)], G19[str(n)]
    r18, v18 = s18(n)
    bx = best(n)
    rows.append(f"{n:<18d}" + f"{100 * a['good1_angle']:.1f} / {100 * a['good25_angle']:.1f} / {100 * a['good5_angle']:.1f}".ljust(20)
                + f"{v18:.1f} (disk {r18 / 1e3:g} km)".ljust(18)
                + f"{100 * b['good1_angle']:.1f} / {100 * b['good25_angle']:.1f} / {100 * b['good5_angle']:.1f}".ljust(20)
                + f"{100 * S[bx]['within1']:.1f} ({bx.replace('_n%d' % n, '')})")
rows += ["", "same spacing (~1 km)   run 18, 25 det.: 4.2 / 19.4 / 45.5%, lit 1     run 19, 100 det.: 10.4 / 40.3 / 73.5%, lit 2"]
pages.append(table_page("Run 18 vs run 19: grid and best layout per detector count", rows, [
    "Best 1° in run 18 = best centred disk (1,536 showers); run 19 = best of all 125 layouts (5,000 showers)",
    "Run 18 at 50 det.: a 3 km disk (≈ the whole 6 km plane, edge-snapped) also beat the grid; at 100 they tie",
    "At equal detector spacing run 19 does better: more lit detectors per shower (wider footprints from farther decays,",
    "     fewer showers near the array edge); at equal detector count it does worse: 4× the area to cover",
]))

# ── Optimisations ────────────────────────────────────────────────────────────
pages.append(section_page("Layout optimisations", "Kernel-in-the-loop L-BFGS on detector positions · scored on the 5,000 held-out test showers"))
pages.append(table_page("Optimisation runs (100 detectors, run 19 recon)", [
    "utility                                        grid start      centre start",
    "current (weighted 1/err² composite)            52546049        52546052",
    "two-tier, good = 5° and |Δlog E| < 0.2, λ = 0   52546050        52546053",
    "two-tier, same, λ = 0.3                         52546051        52546054",
    "three-tier: 0.9 / 0.3 / 0.1 (1° / 5° / seen)    52546510        52547481",
], ["Same settings as run 18's runs: lr 10 m, 20,000 steps, 512 fresh showers per step from the Stage 0 pool,",
    "     best layout picked on a separate 1,024-shower scoring set every 250 steps; kernel widths cached first (52546048)",
    "Two-tier: U = mean[g + λ (1 − g) q], g = soft 'good', q = 1 / (1 + (a / 20°)² + (e / 0.5)²)",
    "Scored with the run's recon on the 5,000 held-out test showers (the layout tests' set) at 1 / 2.5 / 5°",
]))
pages.append(page([P + "three_tier_sigmoids.png"], [
    "Three-tier: each shower scores its best level: 0.9 within 1°, 0.3 within 5°, 0.1 if detected (≥ 3 detectors > 1 particle)",
    "     u = 0.9 g1 + 0.3 (1 − g1) g5 + 0.1 (1 − g1)(1 − g5) d,  U = 30 · mean(u) · angle only, no energy condition",
    "g1 = σ((1° − a) / 0.2°) drops 0.92 → 0.08 between 0.5° and 1.5°; g5 = σ((5° − a) / 1°) drops 0.88 → 0.12 between 3° and 7°",
    "The sigmoids never cross (they meet only at 0°): a clean staircase, but the 0.3 level already slopes from ~2° (0.29 → 0.25 at 4°)",
    "Soft 'detected' d: detectors above 1 particle, count thresholded at 2.5 → 2 hits 0.27, 3 hits 0.73, 4 hits 0.95",
]))

OPT = R19 + "diagnostics/opt/"
OS = json.load(open(OPT + "opt_scores.json"))
LAB = {"start": "start", "current": "current", "twotier_l0": "two-tier λ 0", "twotier_l0p3": "two-tier λ 0.3",
       "threetier": "three-tier"}
pages.append(full_page(OPT + "opt_layouts.png"))
rows = ["layout                       all showers 1° / 2.5° / 5°   median   detected   detected only 1° / 2.5° / 5°   median"]
for scheme in ("grid", "center"):
    for k, lab in LAB.items():
        r = OS.get(f"{k}_{scheme}")
        if r is None:
            rows.append(f"{lab + ' · ' + scheme:<29s}(still running)"); continue
        rows.append(f"{lab + ' · ' + scheme:<29s}" + f"{100*r['all_1']:.1f} / {100*r['all_2.5']:.1f} / {100*r['all_5']:.1f}".ljust(29)
                    + f"{r['ang_med']:.2f}°".ljust(9) + f"{100*r['detected']:.1f}%".ljust(11)
                    + f"{100*r['det_1']:.1f} / {100*r['det_2.5']:.1f} / {100*r['det_5']:.1f}".ljust(31) + f"{r['ang_med_det']:.2f}°")
    rows.append("")
done = [k for k in OS if not k.startswith("start")]
bo = max(done, key=lambda k: OS[k]["all_1"]); g = OS["start_grid"]
pages.append(table_page("Optimised layouts: reconstruction accuracy", rows[:-1], [
    "Detected = ≥ 3 detectors with > 1 particle (as in three-tier) · 5,000 held-out showers: ±0.45 pts at 10% (all showers)",
    f"Best over all showers: {LAB[bo.rsplit('_', 1)[0]]} from the {bo.rsplit('_', 1)[1]} ({100*OS[bo]['all_1']:.1f} / {100*OS[bo]['all_2.5']:.1f} / "
    f"{100*OS[bo]['all_5']:.1f}%) vs the grid start ({100*g['all_1']:.1f} / {100*g['all_2.5']:.1f} / {100*g['all_5']:.1f}%)",
    "Reference: best test layout (6 km disk) 11.9 / 45.1 / 77.4% · from the centre: 9.2–9.6% at 1° (three-tier: 6.9%)",
]))
pages.append(page([OPT + "opt_accuracy.png"], [
    "All showers: every optimisation from the grid gains at 2.5° and 5° (+0.7 to +5.6 pts); at 1° only two-tier and three-tier gain",
    "From the centre (4% of showers detected at the start) all utilities spread out: 9.2–9.6% within 1°, three-tier only 6.9%",
    "No run produced a dense core: the optimiser always spreads the 100 detectors over the 12 km plane"]))
pages.append(page([OPT + "opt_accuracy_detected.png"], [
    "Detected showers only (≥ 3 detectors with > 1 particle): the optimised layouts reconstruct 12.9–17.1% within 1° and 79–85% within 5°",
    "     vs 12.9 / 49.9 / 81.3% for the grid start: they detect fewer showers (27–33% vs 34%) but mostly reconstruct them better",
    "Each bar uses that layout's own detected showers (~1,500–1,700 of 5,000), so differences of ~1 pt are within noise"]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=200)
print(f"[write] {OUT}  ({len(pages)} pages)")
