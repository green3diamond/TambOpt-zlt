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
    "Disks of 20 / 50 / 100 detectors centred on the plane, radius 0.1–12 km; solid = zoo recon, dashed = old recon",
    "Every curve peaks at r ≈ 3 km (the disk just spans the 6 km plane); only 100 det. within 1° peaks at 4.5 km (13.1%)",
    "Peaks (zoo): within 1° 4.4 / 9.2 / 13.1%, within 2.5° 21 / 38 / 47%, within 5° 51 / 71 / 79% for 20 / 50 / 100 det.",
    "Zoo gains most for compact disks (≤ 0.5 km: +3–7 pts within 5°) and for disks wider than the plane"]))

pages.append(page([ZC + "compare_detcount.png"], [
    "Grid with N detectors: zoo recon better at every N, most at small N (within 5°: N = 20 39 → 43%, N = 10 26 → 29%)",
    "At 100 detectors almost the same (within 5° 76.8 → 77.8%, within 1° 12.2 → 12.3%)"]))

pages.append(text_page("Conclusions", [
    "Training on more layout types helps every layout a little, the unusual ones most (clusters, cores: +5 pts at 5°)",
    "So part of the old gap was the recon's training range, but only a few points of it",
    "The ranking does not change: compact cores and small clusters still lose to even coverage at 1°, 2.5° and 5°",
    "Within 1° the changes are small (−1.2 to +2.8 pts per layout); the best layouts stay at 12–13% for 100 detectors",
    "Supports 'why nothing beats the grid': with uniform illumination, even coverage wins, independent of the recon",
    "Next: optimize with the zoo recon; train on fewer detectors (random subsets); uneven, Malata-like illumination",
]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=200)
print(f"[write] {OUT}  ({len(pages)} pages)")
