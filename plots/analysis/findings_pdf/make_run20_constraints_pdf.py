"""PDF: run 20 (Malata) energy bands and construction constraints: setup, the surrogate / trigger-only / penalty-only /
kernel optimisations, the Latin-hypercube repeats, the kernel validation tables and the final layouts. Same page style
as make_run20b_pdf.py. Tables are read from the result files (diagnostics/{sur5,sur5_1us,trg,pen,ker5}/quick_tables_notion.md,
lhs/lhs_summary.md, ker5/compare_sur_kernel.md).

    python figures/make_run20_constraints_pdf.py   (any Python with Pillow + matplotlib)
"""
import os
import re

import matplotlib
from PIL import Image, ImageDraw, ImageFont, ImageOps

B = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/"
R20 = B + "20_malata_showerframe/"
DG = R20 + "diagnostics/"
F = DG + "r20b/"
OUT = B + "figures/08_run20_energy_constraints.pdf"
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


def md_sections(path):
    """{section heading: {row label: [cells]}} of a markdown table file (Notion colorbox marks stripped)."""
    t = re.sub(r"\$\\colorbox\{#[0-9a-f]+\}\{\\textbf\{([^}]*)\}\}\$", r"\1", open(path).read())
    out = {}
    for part in t.split("\n### ")[1:]:
        head, body = part.split("\n", 1)
        rows = [l for l in body.splitlines() if l.startswith("| ") and not l.startswith("| Layout")
                and not l.startswith("| Run")]
        out[head.strip()] = {r.split("|")[1].strip(): [c.strip() for c in r.split("|")[2:-1]] for r in rows}
    return out


def q(tag):
    """per layout: (trig, 1 deg, 5 deg, New U) no bg all read out, New U at 1 us all read out, construction metrics"""
    s = md_sections(DG + tag + "/quick_tables_notion.md")
    a, b, c = s["No background, all read out"], s["1 µs background, all read out"], s["Construction metrics"]
    return {k: dict(trig=a[k][-5], d1=a[k][-4], d5=a[k][-2], u=a[k][-1], u1=b[k][-1], slope=c[k][0], acc=c[k][2],
                    pad=c[k][3], row=c[k][4], cable=c[k][5]) for k in a}


def row(cells, widths):
    return "".join(str(x).ljust(w) for x, w in zip(cells, widths)).rstrip()


SUR, SUR1, TRG, PEN, KER = q("sur5"), q("sur5_1us"), q("trg"), q("pen"), q("ker5")
G = SUR["Grid (100, start)"]
BANDS = [("E 5.00-5.67", "E 5.00-5.67"), ("E 5.67-6.33", "E 5.67-6.33"), ("E 6.33-7.00", "E 6.33-7.00"),
         ("all E", "all E")]
PENS = [("+ slope limit", "+ slope limit"), ("+ access", "+ access"), ("+ pads", "+ pads"),
        ("+ contour rows", "+ contour rows")]


def find(d, part):
    return next((v for k, v in d.items() if k.endswith(part)), None)


pages = [section_page("Run 20: energy and construction",
                      "Malata · energy bands and construction constraints · surrogate and kernel Stage 4, construction penalties, "
                      "Latin-hypercube repeats, kernel validation")]

pages.append(text_page("Summary", [
    "Question: does the best layout depend on shower energy, and what do construction preferences cost?",
    "Optimised layouts never beat the 100-detector grid on the kernel (New U 0.832); the best ones tie with it",
    "Energy bands: optimising on one band gives no layout that is better in that band (all within ±0.006)",
    "Construction: contour rows (terraces) cost < 1% of New U and save 15-21% cable; pads ~5%; access 10-12%",
    "Without the angle term (trigger only / penalty only) layouts get compact and cheaper, at a growing cost",
    "     penalty only, access: half the showers no longer trigger (New U 0.32)",
    "5 Latin-hypercube starts converge to the same score (±0.001-0.005): a broad plateau just below the grid",
    "Kernel vs surrogate: same direction only where a penalty sets it; the kernel's noisier gradient wanders",
    "     further and does worse (E 5.00-5.67 collapses to clusters, New U 0.61)",
    "The slope penalty did not move detectors at its weight (lambda too small against the utility)",
]))

pages.append(table_page("Setup", [
    row(["Run set", "Response", "Objective", "Start", "Runs"], [16, 14, 44, 18, 10]),
    row(["sur5", "surrogate", "5° Poisson utility (+ penalty)", "grid (100)", "8"], [16, 14, 44, 18, 10]),
    row(["sur5_1us", "surrogate", "same, 1 µs background, no gate", "grid (100)", "8"], [16, 14, 44, 18, 10]),
    row(["trg", "surrogate", "trigger probability only (+ penalty)", "grid (100)", "8"], [16, 14, 44, 18, 10]),
    row(["pen", "surrogate", "penalty only (utility 0)", "grid (100)", "4"], [16, 14, 44, 18, 10]),
    row(["lhs", "surrogate", "5° utility (8) and penalty only (4)", "LHS x 5 seeds", "60"], [16, 14, 44, 18, 10]),
    row(["ker5", "kernel", "5° Poisson utility (+ penalty)", "grid (100)", "8"], [16, 14, 44, 18, 10]),
], [
    "All: SGD, step 1 m for a gradient as large as the first one, 1,000 steps, Poisson trigger (>= 3 detectors with >= 1 particle)",
    "5° utility: 30 x mean[ P_trig x (sigmoid(5° - a) + 0.3 (1 - g) / (1 + (a / 20°)^2)) ], default recon, expected counts",
    "Energy-band runs: training and scoring showers with log10 E [GeV] in 5.00-5.67, 5.67-6.33, 6.33-7.00, or all",
    "Stage 2 surrogate: trained on run 20's 464k showers x ~11 layouts (5.1 M pairs per particle type)",
    "Validation: kernel, 24,366 held-out showers, drawn counts, both recons, 0 / 1 / 10 µs, all read out and kNN gate",
]))

pages.append(table_page("Construction penalties", [
    row(["Penalty", "What it penalises", "lambda (5° / trigger, penalty)"], [10, 70, 30]),
    row(["slope", "relu(slope - 33°)^2 under each detector (33° = median under the grid)", "0.006"], [10, 70, 30]),
    row(["access", "3-D distance to the access point (lowest point, 229 / -928 m)", "0.6"], [10, 70, 30]),
    row(["pads", "squared distance to the nearest of 20 pad centres (k-means)", "0.04"], [10, 70, 30]),
    row(["contour", "squared height off the nearest of 8 constant-height rows", "0.02"], [10, 70, 30]),
], [
    "Each penalty is normalised to the grid: 30 x lambda x P(layout) / P(grid), i.e. 30 lambda U at the grid",
    "lambda chosen so the penalty's pull at the grid is about half the 5° utility's gradient",
    "Metrics in the tables: mean slope, mean distance to the access point, RMS distance to the pad centre, RMS height",
    "off the nearest row, 3-D cable tree length (minimum spanning tree)",
]))

W8 = [26, 9, 9, 9, 9, 13, 11, 12]
rows = [row(["Layout", "Trig %", "1° %", "5° %", "New U", "U at 1 µs", "Cable km", "opt. 1 µs U"], W8)]
rows.append(row(["Grid (100, start)", G["trig"], G["d1"], G["d5"], G["u"], G["u1"], G["cable"], "-"], W8))
for lab, part in BANDS + PENS:
    a, b = find(SUR, part), find(SUR1, part)
    rows.append(row([lab, a["trig"], a["d1"], a["d5"], a["u"], a["u1"], a["cable"], b["u1"]], W8))
pages.append(table_page("Surrogate, 5° utility: kernel validation", rows, [
    "Kernel validation, drawn-count recon, all energies; Trig = share of all showers; 1° / 5° = share of triggered",
    "New U at no background and at 1 µs (all read out) of the layout optimised without background;",
    "last column: New U at 1 µs of the layout optimised at 1 µs (no gain from optimising with the background)",
]))

W7 = [18, 15, 15, 15, 38]
rows = [row(["Penalty", "5° utility", "trigger only", "penalty only", "metric: 5° / trigger / penalty"], W7)]
MET = {"+ slope limit": ("slope", "°"), "+ access": ("acc", " m"), "+ pads": ("pad", " m"),
       "+ contour rows": ("row", " m")}
for lab, part in PENS:
    vs = [find(d, part) for d in (SUR, TRG, PEN)]
    key, unit = MET[part]
    rows.append(row([lab.replace("+ ", ""), *[v["u"] if v else "locked" for v in vs],
                     " / ".join(v[key] if v else "-" for v in vs) + unit], W7))
    rows.append(row(["  cable [km]", *[v["cable"] if v else "-" for v in vs], ""], W7))
rows.append(row(["grid", G["u"], G["u"], G["u"], f"{G['slope']}° / {G['acc']} m / {G['pad']} m / {G['row']} m"], W7))
pages.append(table_page("Same penalty, three objectives (New U, no background)", rows, [
    "Without the angle term the penalties dominate: layouts get compact, the cable shorter, reconstruction worse",
    "Contour rows are nearly free whatever the objective; access on its own halves the triggering",
    "Trigger only + pads is missing: its run folder is locked on the file system (validation scores saved)",
]))

W5 = [16, 14, 14, 14]
rows = [row(["Band", "5° utility", "trigger only", "kernel 5°"], W5)]
for lab, part in BANDS:
    rows.append(row([lab, find(SUR, part)["u"], find(TRG, part)["u"], find(KER, part)["u"]], W5))
rows.append(row(["grid", G["u"], G["u"], G["u"]], W5))
pages.append(table_page("Energy bands (New U on all energies, no background)", rows, [
    "Each run is trained and scored on one band only; validation on all energies and per band (per-band: same picture)",
    "No band-specific layout is better than the grid in its own band (tables in diagnostics/<tag>)",
    "Trigger only on the highest band bunches detectors downslope (cable 6.3 km) and loses 13%",
]))

lhs = [l for l in open(DG + "lhs/lhs_summary.md").read().splitlines() if l.startswith("| ") and "seed 1" not in l]
W6 = [26, 18, 12, 12, 12, 12]
rows = [row(["Run", "New U mean ± std", "Access m", "Pad m", "Row m", "Cable km"], W6)]
for l in lhs:
    c = [x.strip() for x in l.split("|")[1:-1]]
    rows.append(row([c[0], c[6], c[8], c[9], c[10], c[11]], W6))
pages.append(table_page("Five Latin-hypercube starts (surrogate)", rows, [
    "Starts: LHS on the detector region (no stacked detectors); grid New U 0.831",
    "Every 5° run climbs from the random starts (0.815) to 0.822-0.829 with a tiny spread: the result does not",
    "depend on the start, and the plateau sits just below the grid; penalties also converge to the same score",
]))

cmp_rows = [l for l in open(DG + "ker5/compare_sur_kernel.md").read().splitlines() if l.startswith("| ") and "Move" not in l]
W9 = [16, 15, 15, 11, 9, 13, 13]
rows = [row(["Run", "Moved sur. m", "Moved ker. m", "Apart m", "cos", "New U sur.", "New U ker."], W9)]
for l in cmp_rows:
    c = [x.strip() for x in l.split("|")[1:-1]]
    rows.append(row(c, W9))
pages.append(table_page("Surrogate vs kernel (same start and settings)", rows, [
    "cos = cosine between the two displacement fields (1 = same direction, 0 = unrelated)",
    "Agreement only where a penalty sets the direction (access, slope); the utility-driven moves are unrelated",
    "The kernel gradient is noisier, so the same step setting wanders further: E 5.00-5.67 collapses (cable 3.9 km)",
]))

pages.append(section_page("Utility over iterations", "Training batches and the fixed scoring set, all 1,000-step runs"))
pages.append(page([F + "utility_curves_5deg.png"], [
    "Surrogate runs (blue) rise in the first ~100 steps and then stay flat: converged on a plateau",
    "Kernel runs (orange) peak by step 100-300 and then decline: constant-step SGD on a noisy gradient random-walks away",
    "     (E 5.00-5.67: 27.9 -> 22.0); the final layouts are worse than the best ones (layout_best.pt, near the peak)",
    "Penalty runs: U - penalty rises while the penalty is paid off; access still rising at step 1,000",
]))
pages.append(page([F + "utility_curves_other.png"], [
    "Trigger only: P(trigger) gains < 0.1 of 30 (already ~95%); the 6.33-7.00 band slowly declines (noise)",
    "Penalty only: pads and contour converge within ~200 steps; access is still moving at step 1,000",
    "Trigger only + pads missing: its folder is locked on the file system",
]))
pages.append(page([F + "utility_curves_same_u.png"], [
    "Same snapshots scored with the optimised formula (5° reward x smooth Poisson trigger on expected counts, - penalty,",
    "     the run's own scoring set): surrogate (reproduces the run's log exactly) and kernel as the response",
    "With this formula the kernel follows the surrogate: all E +0.12, E 5.00-5.67 +0.21, pads +0.78, contour +0.62",
    "The gains are small (all E: 0.4% of U) - below the ±0.5% scatter of the held-out drawn-count New U above,",
    "     which is why the previous slide shows no trend; pads / contour gains here are mostly the penalty paid off",
]))
pages.append(page([F + "utility_curves_lhs.png"], [
    "Five Latin-hypercube starts per configuration: the five curves reach the same level within ~200 steps",
    "     (e.g. 5° all E: 25.8-26.1 at the start, 26.1-26.2 from step 200 on); access is still rising at step 1,000",
]))
pages.append(section_page("Final layouts", "Surrogate (all objectives), Latin-hypercube repeats, kernel"))
for p in (F + "all_layouts.png", DG + "lhs/lhs_layouts.png", DG + "ker5/compare_sur_kernel.png"):
    pages.append(full_page(p))
pages.append(stack_page([F + "sgd_layouts_energy_ker5deg_final.png", F + "sgd_layouts_construct_ker5deg_final.png"], [
    "Kernel runs (5° utility, 1,000 steps from the grid): energy bands (top) and construction penalties (bottom)"]))
pages.append(stack_page([F + "sgd_layouts_energy_sur5deg_1us_final.png", F + "sgd_layouts_construct_sur5deg_1us_final.png"], [
    "Surrogate runs optimised at 1 µs background (no gate): energy bands (top) and construction penalties (bottom)"]))

pages.append(text_page("Conclusions", [
    "On Malata the evenly spread grid is at (or on) a broad optimum of the 5° utility; no optimisation beat it",
    "Energy-band optimisation gives no band-specific advantage; the high band is hardest for every layout",
    "Construction: contour rows are the cheap option (< 1% New U, 15-21% less cable, detectors on terraces)",
    "     pads cost ~5%, access 10-12%; with the angle term removed these costs double or worse",
    "The angle / energy reward is what keeps the array spread; trigger or penalties alone favour compact layouts",
    "The surrogate is a usable guide for construction trade-offs; the kernel's noisy gradient needs smaller steps",
    "Open: a stronger slope penalty, a real access map (roads, camp), and the locked trigger-only pads run",
]))

pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=200)
print(OUT, len(pages), "pages")
