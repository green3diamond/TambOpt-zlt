"""PDF: run 20b (Malata, 1 deg under 10 us background + kNN gate): setup, tests, the first 8 h of the three Stage 4
runs, the lr scan and the gradient checks (why the runs do not beat the grid). Same page style as
make_summary_2wk_pdf.py. Figures from plots/analysis/run20/plot_r20b_tests.py.

    python figures/make_run20b_pdf.py   (any Python with Pillow + matplotlib)
"""
import os

import matplotlib
from PIL import Image, ImageDraw, ImageFont, ImageOps

B = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/"
R20 = B + "20_malata_showerframe/"
F = R20 + "diagnostics/r20b/"
OUT = B + "figures/07_run20b_1deg_background.pdf"
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
    """Monospace table with borders: an outer frame, a line under every row (darker under the header rows and at blank
    group separators) and column rules in the gutters, i.e. character columns that are blank in every row for >= 2
    characters. Header = the first row plus any following rows that start with a space."""
    p = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(p)
    d.text((200, 150), heading, fill=INK, font=HEAD)
    cw, x0, top = MONO.getlength("M"), 210, 320
    n = max(len(r) for r in rows)
    body = [r.ljust(n) for r in rows if r.strip()]
    gut = [all(r[c] == " " for r in body) for c in range(n)]
    cols, c = [], 0
    while c < n:
        if gut[c]:
            s = c
            while c < n and gut[c]:
                c += 1
            if c - s >= 2 and s > 0 and c < n:
                cols.append(x0 + cw * (s + c) / 2)
        else:
            c += 1
    nh = 1
    while nh < len(rows) and rows[nh][:1] == " ":
        nh += 1
    x1, y, LIGHT, DARK = x0 + cw * n + 14, top, "#c9c8c3", "#52514e"
    d.line([(x0 - 14, y - 10), (x1, y - 10)], fill=DARK, width=3)
    for k, row in enumerate(rows):
        if not row.strip():
            d.line([(x0 - 14, y - 10), (x1, y - 10)], fill=DARK, width=3)
            y += 14
            continue
        d.text((x0, y), row, fill=INK if k >= nh else "#52514e", font=MONO)
        y += 58
        d.line([(x0 - 14, y - 10), (x1, y - 10)], fill=DARK if k == nh - 1 else LIGHT, width=3 if k == nh - 1 else 1)
    d.line([(x0 - 14, y - 10), (x1, y - 10)], fill=DARK, width=3)
    for x in [x0 - 14, x1] + cols:
        frame = x in (x0 - 14, x1)
        d.line([(x, top - 10), (x, y - 10)], fill=DARK if frame else LIGHT, width=3 if frame else 1)
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


def pending_page(heading, what):
    """Placeholder for results that are still being computed."""
    p = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(p)
    d.text((200, 160), heading, fill=INK, font=HEAD)
    d.text((W // 2 - 420, H // 2 - 60), "… in progress …", fill="#9b9a95", font=TITLE)
    d.text((210, H - 260), what, fill="#52514e", font=FONT)
    return p


def full_page(path):
    img = Image.open(path).convert("RGB")
    k = min((W - 2 * M) / img.width, (H - 2 * M) / img.height)
    img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
    p = Image.new("RGB", (W, H), "white")
    p.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return p




def stack_page(paths, bullets):
    """Figures stacked top to bottom (same width), bullets below."""
    p = Image.new("RGB", (W, H), "white")
    imgs = [Image.open(x).convert("RGB") for x in paths]
    imgs = [i.crop(ImageOps.invert(i).getbbox()) for i in imgs]                 # trim white margins
    gap = 30
    img_h = IMG_H - max(0, len(bullets) - 3) * 66 + 40
    w = min(W - 2 * M, int((img_h - gap * (len(imgs) - 1)) / sum(i.height / i.width for i in imgs)))
    y = M - 40
    for i in imgs:
        r = i.resize((w, int(i.height * w / i.width)), Image.LANCZOS)
        p.paste(r, ((W - w) // 2, y))
        y += r.height + gap
    d = ImageDraw.Draw(p)
    y += 20
    for b in bullets:
        d.text((M + 60, y), "•  " + b, fill=INK, font=FONT)
        y += 66
    return p


def side_page(heading, path, bullets):
    """One figure on the left, heading and bullets on the right."""
    p = Image.new("RGB", (W, H), "white")
    img = Image.open(path).convert("RGB")
    h = H - 2 * M
    w = int(img.width * h / img.height)
    if w > W * 0.45:
        w = int(W * 0.45); h = int(img.height * w / img.width)
    p.paste(img.resize((w, h), Image.LANCZOS), (M, (H - h) // 2))
    d = ImageDraw.Draw(p)
    x = M + w + 100
    d.text((x, 200), heading, fill=INK, font=HEAD)
    for k, b in enumerate(bullets):
        d.text((x + 10, 370 + 88 * k), ("    " + b.strip()) if b.startswith(" ") else ("•  " + b), fill=INK, font=FONT)
    return p


pages = [section_page("Run 20b: 1° under background, Malata",
                      "2026-10-28 – 10-31 · Stage 4 with 10 µs background, kNN gate and drawn counts · "
                      "gradient checks")]

pages.append(text_page("Summary", [
    "Run 20 (no background): no Stage 4 run beat the grid; the gradient at the grid is mostly batch noise",
    "Run 20b: optimise 1° under 10 µs background + kNN gate, with the drawn-count recon and drawn counts",
    "     3 runs from the grid: 1° utility with Poisson trigger, with step trigger, and Old U (no trigger term)",
    "     8 × 2,048 showers per step, step size 3 m halved when the scoring-set U stalls (floor 0.03 m)",
    "Tests passed: without the new options Stage 4 is bit-identical to before; plateau halving and resume work",
    "After 8 h (500–580 steps) all three runs are 2–6% below the grid on the scoring set: cancelled",
    "Gradient check: under background + drawn counts the gradient is noise (cosine between batches ≈ 0.01)",
    "     even on the same showers, redrawing only background and counts gives unrelated gradients (≈ 0.04)",
    "     without the noise the same utility has a signal (cosine 0.17, 32% of detectors with SNR > 2)",
    "Next: take the gradient on expected counts without background; keep drawn counts + background for validation",
]))

pages.append(table_page("Run 20b setup", [
    "run            reward per shower                                          trigger",
    "1deg_poisson   sigmoid((1° − a) / 0.2°) + 0.3 (1 − g) / (1 + (a / 5°)²)   Poisson, read-out detectors only",
    "1deg_step      same                                                       step (SWGOLO notebook), read-out only",
    "oldu           soft(a < 5°) × soft(|Δlog10 E| < 0.2), partial credit      none",
], [
    "Common: run 20 pool and scoring set (1,024 showers), grid start, 100 detectors of 2 m², kNN 2-of-3 gate",
    "Background: cosmic muons in a 10 µs window; counts Poisson-drawn per species (straight-through gradient)",
    "Recon: run 20 drawn-count fine-tune (test_v6_run_03_recentered_deepsets_count)",
    "Steps: 8 sub-batches of 2,048 averaged per Adam step; checkpoint every 10; jobs of 8 h with 2 continuations",
    "Branch r20b-1deg-bg (commits e670176, 2b7cb9a)",
]))

pages.append(page([F + "r20b_lr_scan.png"], [
    "lr scan (run 20 settings, no background, from the grid; stopped at step 150 to save GPU time)",
    "lr 1 and 3 both reach U ≈ 27.27 (+0.9% over the grid's 27.03); lr 10 falls back to the grid by step 100",
    "Chosen: lr 3, halved on plateaus down to lr / 100",
]))

pages.append(page([F + "r20b_training.png"], [
    "Scoring set: every run drops below the grid within 75 steps and does not recover (Poisson best = grid)",
    "Training batches drift down too: 4–7% below the grid after 500 steps; the step size halves 4–5 times",
    "Step trigger's +1.8% at step 50 and Old U's +0.1% are no larger than its step-to-step swings (±2–4%)",
]))

pages.append(page([F + "r20b_layouts.png"], [
    "Detectors moved a median 52–69 m (max ~200 m) in every direction, with no common pattern between runs",
    "Consistent with a random walk: Adam moves ~ step size per step whatever the gradient's direction",
]))

pages.append(page([F + "r20b_grad_check.png"], [
    "Blue: 8 different batches of 2,048 showers; grey: one batch, 8 new background / count draws",
    "Under 10 µs + kNN + drawn counts: cosine ≈ 0 and |mean| / |g| at the pure-noise line → averaging 8 does not help",
    "The draws alone randomise the gradient: the straight-through count / gate gradient changes with each draw",
    "Without noise the 1° Poisson gradient has a signal (0.17; 32–34% of detectors with SNR > 2)",
]))

pages.append(table_page("Gradient check numbers (grid start)", [
    "configuration                          U      cos batches   cos draws   |mean|/|g|   SNR > 2",
    "run 20 utility, no noise, default rec  26.96  +0.033        –           0.38         22%",
    "1° Poisson, no noise, count recon       0.94  +0.174        –           0.53         32%",
    "1° Poisson, 10 µs + kNN + drawn         7.42  +0.014        +0.040      0.37         17%",
    "1° step, 10 µs + kNN + drawn            7.26  +0.013        +0.042      0.37         16%",
    "Old U, 10 µs + kNN + drawn             13.50  −0.013        +0.003      0.34         11%",
], [
    "Pure noise with 8 batches gives |mean| / |g| = 0.35; a fully consistent gradient gives 1",
    "Read out 20% of detectors under 10 µs + kNN; 0.54 ghost (background-only) read-outs per shower",
    "1° Poisson with noise loses 10% to untriggered showers and 65% among triggered (mean reward 0.25)",
    "The count recon on expected counts scores U 0.94: fine-tuned on drawn counts, it needs them as input",
]))

pages.append(text_page("Conclusions and next steps", [
    "8× bigger batches cannot fix it: the noise is per draw, not per shower sample",
    "Straight-through drawn counts + background make the layout gradient uninformative at this batch size",
    "Option A: gradient from expected counts, no background (signal 0.17), default recon; validate with noise",
    "Option B: fine-tune a recon on both expected and drawn counts, so one recon serves gradient and validation",
    "Option C: average the gradient over many draws per shower (cost grows linearly with the draws)",
    "The run 20 conclusion stands under background: the grid is not beaten by gradient-based Stage 4",
]))

pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=200)
print(OUT, len(pages), "pages")
