"""PDF: run 18 recon retrained on a layout zoo (what changed, training, old vs zoo recon on the same test layouts).
Same page style as make_findings_pdf.py.

    python figures/make_zoo_recon_pdf.py   (any Python with Pillow + matplotlib)
"""
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib

B = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/"
R18 = B + "18_plane45S_6km_showerframe_200k/"
D18 = R18 + "diagnostics/"
ZC = D18 + "zoo_compare/"
ZR = R18 + "test_v6_run_03_recentered_deepsets_zoo/"
OUT = B + "figures/run18_zoo_recon.pdf"
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
    d.text((200, 180), heading, fill=INK, font=HEAD)
    for k, b in enumerate(bullets):
        d.text((210, 360 + 92 * k), "•  " + b, fill=INK, font=FONT)
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


pages = [section_page("Run 18: recon trained on a layout zoo",
                      "2026-10-07 · branch run18-agnostic-recon · 6 km South-facing plane, shower-frame kernel")]

pages.append(text_page("What changed", [
    "Goal: a recon that cares less about the layout; the old one only saw spread-out 100-detector layouts",
    "4 new training layout types appended (ids 5–8); the 5 old ones (ids 0–4) are unchanged:",
    "     clusters ×2 (2–50 clusters, radius 30–600 m) · disk (radius 0.2–4 km) · grid patch (10 × 10 over a 0.5–6 km square)",
    "A new layout every 60 showers per type (~3,300 per type); all 3 species of an event share it",
    "Stage 1: --strategies labels only chosen types; 01b_merge_strategy_blocks.py appends them (same events checked)",
    "Dataset ..._northeast_zoo: 9 types × 600,210 rows = 5.4M rows (was 5 × 600,210 = 3.0M)",
    "Recon: same model, inputs, loss and training as before, only more layouts → ..._deepsets_zoo",
    "Unchanged: showers (Stage 0), kernel, held-out test showers, scoring",
    "Old recon and Stage 4 runs moved to <run18>/archive/",
    "Incident: the first build ignored the folder switch and overwrote the old 5-type dataset;",
    "     fixed (output folder from constants) and rebuilt; layout draws are seeded, rebuild verified identical",
]))

pages.append(page([D18 + "zoo/zoo_layouts.png"], [
    "The 4 new training layout types, 3 draws each (100 detectors on the 6 × 6 km plane)",
    "Clusters: K a random divisor of 100 (2–50), one radius per draw · disks and grid patches at random places and sizes"]))

pages.append(table_page("Recon training", [
    "                       old recon        zoo recon",
    "layout types           5                9",
    "training rows          2.7M             4.85M",
    "val loss (own val set) 0.063            0.089",
    "Adam                   –                217 epochs, best 0.0908 @ 167 (8.1 h)",
    "L-BFGS                 505 steps        484 steps, best 0.0893 (stopped, 150 w/o gain)",
    "total time             ~3.7 h           15.6 h",
], [
    "Val losses are not comparable: the zoo val set includes the harder cluster, disk and patch layouts",
    "Fair comparison on the next pages: both recons on the same 1,536 held-out showers and the same test layouts",
]))

pages.append(page([ZR + "recon_train_curves.png", ZR + "recon_conditional_kernel.png"], [
    "Zoo recon: training curves (left) and reconstruction across showers on kernel responses (right)"]))

pages.append(page([ZC + "compare_scatter.png"], [
    "116 test layouts (cores, hybrids, disks, clusters at 20 / 50 / 100 det., grids, 6 optimized), both recons",
    "Zoo better on 78% of layouts within 1° (median +0.5 pts), 96% within 2.5° (+2.1), 98% within 5° (+4.5)",
    "Largest gains for clusters and cores (+5 pts within 5°), smallest for the grid (+1.6) and optimized layouts (+0.9)"]))

pages.append(page([ZC + "compare_best.png"], [
    "Best layout per family: the ranking is unchanged · even coverage (grid, ~3 km disk, optimized) stays on top",
    "100 det. within 1°: grid 12.3%, disk 4.5 km 13.1%, optimized 13.1%, best cluster 9.9%, best core 1.7%",
    "20 det. within 5°: grid 35.6 → 40.6%, 3 km disk 46.0 → 50.6%, best cluster 43.4 → 47.5%"]))

pages.append(page([ZC + "best_layouts_1deg.png"], [
    "The layouts behind the previous page, within 1°: zoo recon's best per family in blue, old recon's best in grey if different",
    "The winners are the spread-out members of each family (500 m clusters, widest disks, cores with a wide halo)"]))
pages.append(page([ZC + "best_layouts_5deg.png"], ["Same, within 5°"]))

pages.append(page([ZC + "disks_vs_radius.png"], [
    "Disks of 10 / 20 / 50 / 100 detectors centred on the plane, radius 0.1–12 km; solid = zoo recon, dashed = old recon",
    "Peaks at r ≈ 3 km (disk just spans the 6 km plane); 10 det. peak smaller (1.5–2 km), 100 det. within 1° at 4.5 km",
    "Peaks (zoo), 10 / 20 / 50 / 100 det.: within 1° 2.2 / 4.4 / 9.2 / 13.1%, 2.5° 10 / 21 / 38 / 47%, 5° 28 / 51 / 71 / 79%",
    "Zoo gains most for compact disks (≤ 0.5 km: +3–7 pts within 5°) and for disks wider than the plane"]))

pages.append(page([ZC + "compare_detcount.png"], [
    "Grid with N detectors: zoo recon better at every N, most at small N (within 5°: N = 20 39 → 43%, N = 10 26 → 29%)",
    "At 100 detectors almost the same (within 5° 76.8 → 77.8%, within 1° 12.2 → 12.3%)"]))

pages.append(text_page("Test: how many lit detectors does 1° need?", [
    "Question: how many detectors must a shower light for the recon to reach 1° resolution?",
    "Layouts: every test layout above (grids, disks, clusters, cores, hybrids, optimized; 10–100 detectors),",
    "     each counted once (identical detector positions listed in several test sets are dropped): 112 layouts",
    "Showers: the same 1,536 held-out showers on every layout → 172,032 shower–layout pairs",
    "Per pair: lit detectors = kernel response ≥ 1 count; angle error of the zoo recon (true vs reconstructed direction)",
    "Binned by lit count: median and 68% angle error, share within 1°, number of distinct showers in the bin",
    "Error bands: 200 resamples of the 1,536 showers (all pairs of a shower move together, so a few big showers",
    "     cannot carry a bin); bands = 16–84% of the resamples",
    "Groups: all layouts, and spread layouts (grids, optimized, disks ≥ 2 km, 500 m clusters)",
]))
pages.append(page([ZC + "lit_resolution.png"], [
    "Spread layouts: median error 9.1° (0 lit) → 5.3° (1) → 3.8° (2) → 3.0° (4) → 2.6° at 6–7 lit, then flat to 15 lit (± 0.1°)",
    "68% error flattens at ~3.9°; share within 1° tops out at 12–15%: no lit count reaches 1° resolution",
    "Worse again above ~16 lit, clearly at 30+ (4.7°, 104 showers): large showers on compact layouts, short baselines",
    "So the 1° limit is not the number of hits: it is set by the recon, the timing information or the kernel's arrival times"]))

pages.append(text_page("Conclusions", [
    "Training on more layout types helps every layout a little, the unusual ones most (clusters, cores: +5 pts at 5°)",
    "So part of the old gap was the recon's training range, but only a few points of it",
    "The ranking does not change: compact cores and small clusters still lose to even coverage at 1°, 2.5° and 5°",
    "Within 1° the changes are small (−1.2 to +2.8 pts per layout); the best layouts stay at 12–13% for 100 detectors",
    "Supports 'why nothing beats the grid': with uniform illumination, even coverage wins, independent of the recon",
    "Resolution stops improving at ~6 lit detectors (median 2.6°); 1° is out of reach at any lit count with this recon",
    "Next: a direct timing-plane fit on lit detectors (is 2.6° the recon or the signal?); uneven, Malata-like illumination",
]))

# ── Relative-time recon (snapshot of the running job) ───────────────────────
pages.append(section_page("Relative-time recon",
                          "Same zoo data and training; new input encoding · snapshot at L-BFGS step 25 (job still running)"))
pages.append(text_page("What changed: the recon's input", [
    "Old input per detector: x, y (absolute), log1p E, log1p(t in µs since the decay); unlit detectors had t = 0",
    "     → a 1° tilt over 1 km (~60 ns) was ~0.005 in log units on a 3–30 µs offset; t = 0 read as a very early hit",
    "New input (9 features, built inside the model, callers unchanged; branch run18-reltime-recon):",
    "     dx, dy, dt relative to the reference detector (dt linear, ns) · log1p E · lit (E > 0) · timed (T > 0)",
    "     · reference x0, y0, log1p(t0) broadcast to every detector, so nothing absolute is lost",
    "Reference = earliest timed detector (else brightest lit, else array centre); untimed detectors get dt = 0",
    "Most lit detectors have no time: the kernel times a detector only if one point clears HIT_DEPOSIT_MIN",
    "     (training rows: 36% of slots have E > 0, 5.6% have a time)",
    "Same zoo dataset (9 layout types), model size, loss, Adam (patience 50) and L-BFGS as the zoo recon",
    "Val loss: Adam best 0.0788 at epoch 43 (zoo: 0.0908 at 167); L-BFGS step 25: 0.0782 (zoo final: 0.0893)",
]))
pages.append(page([ZC + "compare_scatter_reltime_lbfgs25_vs_zoo.png"], [
    "Same 129 test layouts and 1,536 held-out showers; x = zoo recon, y = relative-time recon (snapshot)",
    "Better on 84% of layouts within 1° (median +0.5 pts), 94% within 2.5° (+2.3), 96% within 5° (+3.5)",
    "Median angle error per layout: −0.56° (median change); largest gains for optimized, clusters and cores"]))
pages.append(table_page("Key layouts: zoo → relative-time recon", [
    "layout                 within 1°      within 2.5°    within 5°      median error",
    "grid (100)             12.3 → 13.9    47.4 → 50.9    77.8 → 80.7    2.65° → 2.45°",
    "disk 3 km (100)        12.0 → 14.4    46.7 → 50.9    78.8 → 82.8    2.68° → 2.45°",
    "disk 4.5 km (100)      13.1 → 12.4    46.6 → 49.0    76.9 → 81.6    2.67° → 2.55°",
    "clusters 20×5, 500 m   9.9 → 12.0     39.1 → 43.0    72.1 → 74.7    3.13° → 2.95°",
    "core 500 m (100)       1.7 → 2.4      9.2 → 11.3     24.7 → 30.3    10.1° → 8.6°",
    "grid 50                7.3 → 8.8      32.7 → 37.1    66.9 → 70.1    3.58° → 3.26°",
    "grid 20                3.1 → 2.8      15.0 → 16.4    40.6 → 41.6    6.19° → 6.07°",
], [
    "Grid N = 4 / 5 / 10 / 20 / 50 / 100 within 5°: 9.2 / 12.8 / 28.5 / 42.7 / 69.0 / 77.8% → 10.2 / 12.6 / 30.5 / 44.1 / 71.0 / 80.7%",
    "Ranking unchanged: even coverage (grid, ~3 km disk) still on top; cores and small clusters still far behind",
]))
pages.append(page([ZC + "disks_vs_radius_reltime_lbfgs25_vs_zoo.png"], [
    "Disks of 10 / 20 / 50 / 100 detectors centred on the plane; dashed = zoo recon, solid = relative-time recon",
    "Peaks (relative-time), 10 / 20 / 50 / 100 det.: within 1° 2.4 / 5.7 / 10.3 / 14.4%, 2.5° 12 / 25 / 42 / 51%, 5° 30 / 53 / 76 / 83%",
    "Zoo: 2.2 / 4.4 / 9.2 / 13.1%, 10 / 21 / 38 / 47%, 28 / 51 / 71 / 79% · higher at almost every radius and N",
    "Same shape: every N still peaks at r ≈ 3 km (10 det. at 1.5 km); the 100-det. 1° peak moves from 4.5 to 3 km"]))
pages.append(page([ZC + "lit_resolution_reltime_lbfgs25_vs_zoo.png"], [
    "Spread layouts: median error floor 2.62–2.69° → 2.41–2.47° (6–15 lit); 68% error 3.8–4.0° → 3.6–3.7°",
    "Within 1° at 6–12 lit: 12–14% → 15–16%; median error lower at every lit count from 2 to 29, same flat shape",
    "A real but modest gain (~0.2° on the median): the encoding was part of the limit, not all of it",
    "Final scores after L-BFGS finishes; next candidates: angle loss, attention over detector pairs, timing-plane fit"]))

pages.append(page([ZC + "lit_geometry.png"], [
    "Same 112 layouts × 1,536 showers; per shower, the lit detectors' (≥ 1 count) mean distance, nearest-neighbour spacing and baseline",
    "Best (6–15 lit): nearest-neighbour spacing 300–600 m → median 2.4°, 15–16% within 1°; < 100 m → 3.7–4.0°, 6–7%",
    "Mean distance 600–1200 m and baseline ≥ 800 m are best; within 1° vs > 5° showers: baseline 1.1 km vs 0.7 km, NN 278 vs 226 m",
    "Above ~800 m spacing too few lit detectors sit in one footprint (~900 m across): the optimum spacing is ~300–600 m"]))

pages.append(section_page("Test layouts",
                          "The 112 distinct layouts by detector count, ranked best to worst by the share within 1° (relative-time recon)"))


def full_page(path):
    """One figure filling the slide (margins M)."""
    img = Image.open(path).convert("RGB")
    k = min((W - 2 * M) / img.width, (H - 2 * M) / img.height)
    img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
    p = Image.new("RGB", (W, H), "white")
    p.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return p


import glob as _glob  # noqa: E402
for _f in sorted(_glob.glob(ZC + "layout_catalog_p*.png")):
    pages.append(full_page(_f))

# ── Weight decay ───────────────────────────────────────────────────────────
WD = R18 + "diagnostics/wd_sweep/"
pages.append(section_page("Weight decay", "Does regularising the relative-time recon help? · AdamW sweep on run 18 data"))
pages.append(text_page("Weight-decay sweep: setup", [
    "Why: the relative-time recon overfits after epoch 43 (val 0.0788 then rises while train keeps falling); no dropout, no decay",
    "Unchanged: relative-time DeepSets recon (431k parameters), run 18 zoo data (9 layout types, 4.85M training rows)",
    "Each run is a full retrain from scratch, only the optimiser changes: AdamW (decoupled decay), lr 3e-5, batch 256",
    "Weight decay wd = 0.01 / 0.03 / 0.1 / 0.3 (per step the weights shrink by lr × wd: 0.06% … 16% per epoch)",
    "Shortened schedule: ≤ 80 epochs, stop after 30 without improvement, no L-BFGS (the full run had 600 / 50 / L-BFGS)",
    "Baseline wd 0: the relative-time recon's own Adam phase (first 80 epochs), same data, seed and model",
    "4 GPU jobs in parallel (~2.5–3.5 h each, with requeues resumed from per-epoch checkpoints)",
    "Next: the wd 0.03 recon is finished with L-BFGS (job running), then scored on the test layouts",
]))
pages.append(table_page("Weight-decay sweep: results (Adam phase only, ≤ 80 epochs)", [
    "wd      best val  @epoch  direction  energy  rise by ep 80   first epoch with val ≤ 0.0790 / 0.0785 / 0.0780",
    "0       0.0788      43     0.0314    0.0474    +0.0008        40 / never / never",
    "0.01    0.0783      53     0.0314    0.0470    +0.0005        38 / 53 / never",
    "0.03    0.0777      78     0.0311    0.0467    +0.0001        38 / 53 / 69",
    "0.1     0.0783      80     0.0315    0.0469     0.0000        62 / 67 / never",
    "0.3     0.0834      66     0.0332    0.0502    +0.0004        never",
], [
    "direction = dx + dy + dz and energy = logE terms of the val loss at the best epoch",
    "rise by ep 80 = the run's own val at epoch 80 minus its own best val: how much it got worse after its best epoch (overfitting)",
    "     e.g. no decay: best 0.0788 at epoch 43, 0.0796 at epoch 80 → +0.0008 · wd 0.03: 0.0777 → 0.0778 → +0.0001",
    "wd 0.03 is best: −0.0011 vs no decay (−1.4%), mostly energy; direction only −1%; no overfitting by epoch 80",
    "It reaches val ≤ 0.0785 in 53 epochs; no decay never gets there with Adam (0.0782 only after L-BFGS, ~4.2 h)",
    "0.1 learns slower but its val is still falling at 80; 0.3 is too strong · Default for future trainings: wd 0.03",
]))
pages.append(page([WD + "wd_sweep.png"], [
    "Val loss per epoch (left) and the train–val gap (right): without decay the gap keeps growing; 0.03 and 0.1 hold val down"]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=200)
print(f"[write] {OUT}  ({len(pages)} pages)")
