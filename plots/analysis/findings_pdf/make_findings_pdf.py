"""PDF: findings after run 13 (runs 14-16, SWGOLO comparison). White pages, figures on top, short bullets.

    python figures/make_findings_pdf.py   (any Python with Pillow + matplotlib)
"""
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib

B = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/"
OUT = B + "figures/findings_after_run13.pdf"
W, H, M = 3200, 1800, 90
IMG_H = 1400
TTF = os.path.join(matplotlib.get_data_path(), "fonts/ttf/")
FONT = ImageFont.truetype(TTF + "DejaVuSans.ttf", 46)
HEAD = ImageFont.truetype(TTF + "DejaVuSans-Bold.ttf", 72)
TITLE = ImageFont.truetype(TTF + "DejaVuSans-Bold.ttf", 110)
INK = "#1F2328"
R12, R14 = B + "12_10pct_mean_analytic/", B + "14_sigma10_10pct/"
R15, R16 = B + "15_sigma10_interp_hurdle_10pct/", B + "16_sigma10_interp_emean_10pct/"


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
    d.text((200, 220), heading, fill=INK, font=HEAD)
    for k, b in enumerate(bullets):
        d.text((210, 420 + 92 * k), "•  " + b, fill=INK, font=FONT)
    return p


MONO = ImageFont.truetype(TTF + "DejaVuSansMono.ttf", 40)


def table_page(heading, rows, notes, header=True):
    """Heading, a monospace table (first row = header) and short bullets under it."""
    p = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(p)
    d.text((200, 150), heading, fill=INK, font=HEAD)
    y = 320
    for k, row in enumerate(rows):
        d.text((210, y), row, fill=INK if (k or not header) else "#52514e", font=MONO)
        y += 58
    y += 40
    for b in notes:
        d.text((210, y), "•  " + b, fill=INK, font=FONT)
        y += 66
    return p


def section_page(title, subtitle):
    """Title slide that opens a new set of tests."""
    p = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(p)
    d.text((200, 700), title, fill=INK, font=TITLE)
    d.text((210, 880), subtitle, fill="#52514e", font=FONT)
    return p


cover = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(cover)
d.text((200, 520), "TambOpt · findings after run 13", fill=INK, font=TITLE)
for k, b in enumerate([
        "Runs 14 (σ = 10 m), 15 (interpolated kernel + lit/dark Stage 2), 16 (mean energy)",
        "SWGOLO comparison: why SWGO finds structure and TambOpt does not",
        "Details: docs/swgolo-comparison.md, docs/sigma10-stage2-collapse.md, docs/run15-findings.md"]):
    d.text((210, 750 + 70 * k), "•  " + b, fill=INK, font=FONT)

pages = [cover, text_page("Summary", [
    "Uniform fill comes from the setup: every tau axis hits the mesh, and detectors live on that same mesh",
    "SWGO throws cores over more area than its array covers → density vs area trade-off → core + halo",
    "Implementation choices (surrogate vs simulator, recon, optimizer, start) do not move the optimum",
    "No layout beats a uniform grid by more than ~1–3% on held-out true utility (runs 12, 14, 15)",
    "Acceptance term: no gain where acceptance is saturated (σ = 50, run 15); small, untested gain at σ = 10",
    "More lit detectors per shower → much better recon (same events: 3 lit 29°, 15 lit 11°); changing the gate does not help",
    "Next: near-miss tau corpus (requested); a two-tier utility in recon accuracy (count well-reconstructed showers first)",
])]

pages.append(section_page('SWGO comparison', 'Where showers land, and what an acceptance term does (run 12)'))
pages.append(page([B + "figures/eda_outputs/axis_impact_density/axis_impact_density.png"], [
    "Where tau axes hit the mesh: 100% of 602k taus (1e5–1e7 GeV) hit it by selection",
    "Interior illumination 0.92–1.15 of uniform; footprints stretched ~3.3× along the slope",
    "docs/swgolo-comparison.md §2"]))

pages.append(page([R12 + "test_v6_run_04_optimize_lbfgs_ensemble_upr0_grid/layout_density.png",
                   R12 + "test_v6_run_04_optimize_lbfgs_ensemble_upr640_grid/layout_density.png"], [
    "Run 12 (σ = 50 m) · Stage 4 without (left) and with (right) an acceptance term, λ = 640",
    "λ pushes detectors to the rim, but kernel utility 42.99 vs 43.03: no gain (acceptance 89% saturated)",
    "docs/swgolo-comparison.md §8"]))

pages.append(section_page('Run 14: σ = 10 m', 'Narrow kernel: detector labels, surrogate and layouts'))
pages.append(page([R14 + "test_v6_run_02_recentered/fnn_electron_conditional.png"], [
    "Run 14 · σ = 10 m · Stage 2 electron surrogate",
    "78% of E and 95% of T detector slots are exactly zero; lit-slot E error is scatter, not bias (bias² 0.4%)",
    "docs/sigma10-stage2-collapse.md"]))

pages.append(page([R14 + "test_v6_run_04_optimize_lbfgs_ensemble_upr0_grid/layout_density.png",
                   R14 + "test_v6_run_04_optimize_lbfgs_ensemble_upr200_grid/layout_density.png"], [
    "Run 14 · Stage 4 without (left) and with (right) an acceptance term, λ = 200 (\"spine\")",
    "Kernel utility at σ = 10: spine 23.95, λ = 0 23.75, grid 23.22 (no error bars); acceptance mean r 0.57",
    "docs/swgolo-comparison.md §4–5, §9"]))

pages.append(section_page('Run 15: interpolated kernel', 'Adaptive Gaussian widths + lit/dark Stage 2 surrogate'))
pages.append(page([R15 + "test_v6_run_01_northeast/kernel_bw_k8_hist.png"], [
    "Run 15 · interpolated kernel: width h = max(10 m, distance to 8th in-plane neighbour)",
    "Energy spread beyond 10 m: electron 54%, photon 61%, muon 91% (muon energy-weighted median ≈ 190 m)",
    "docs/run15-findings.md"]))

pages.append(page([R15 + "test_v6_run_02_recentered/fnn_electron_conditional.png"], [
    "Run 15 · Stage 2 electron surrogate, lit/dark head (median line)",
    "Lit detector slots: E ≈ 92% (run 14: 22%)"]))

s3 = R15 + "test_v6_run_03_recentered_deepsets/"
pages.append(page([s3 + "recon_conditional.png", s3 + "recon_conditional_kernel.png"], [
    "Run 15 · Stage 3 recon conditional: detector response model (left), kernel (right)",
    "Recon val loss 0.066 (run 14: 0.142)"]))

s4 = R15 + "test_v6_run_04_optimize_lbfgs_ensemble_full_corpus_"
pages.append(page([s4 + "grid/layout_density.png", s4 + "center/layout_density.png"], [
    "Run 15 · Stage 4: grid start (left) true U 30.48 → 30.85 (+1.2%); center start (right) 7.13 → 30.79",
    "Acceptance saturated (kernel mean r 0.88); reconstruct threshold 5 → 30 keeps layouts within 1.4–3.6%",
    "docs/run15-findings.md §2–4"]))

pages.append(section_page('Kernel-in-the-loop optimization',
                          'Optimizing directly on the kernel instead of the surrogate (run 15)'))
k4 = R15 + "test_v6_run_04_optimize_lbfgs_kernel_recon_utility_lr10_"
pages.append(page([k4 + "grid/layout_density.png", k4 + "center/layout_density.png"], [
    "Stage 4's 75k events, 512 fresh per step, Adam lr = 10, 20k steps; best layout picked on 1,024 separate events",
    "Held-out true U: grid start (left) 30.48 → 30.57 (+0.3%); center start (right) 7.13 → 30.50",
    "Surrogate optimizer for comparison: 30.85 (grid), 30.79 (center): the kernel finds nothing better",
    "Best scoring-set U already at step ~2,750 (31.75 / 31.79), then drifts back to ~31.0–31.3",
    "Earlier 64-fixed-event runs overfit (grid → 29.24); run 12's stall was setup (blob start, lr = 1, 200 steps)"]))

pages.append(section_page('Run 16: mean energy', 'Detector E as the kernel-weighted mean point energy'))
pages.append(page([R16 + "test_v6_run_02_recentered/fnn_electron_conditional.png"], [
    "Run 16 · detector E = Σ e·w / max(Σ w, 1): the weighted mean point energy (otherwise run 15)",
    "Stage 1 labels vs run 15: same lit set; only ~1% of lit E values changed (mean Δlog1pE 0.01–0.02)",
    "Normalising factor max(Σ w, 1): mean 1.04–1.08; > 1 on only 1.1–1.5% of lit detectors (there ~3×, up to 100–300×)",
    "So Stage 2 matches run 15 (val 0.341/0.393/0.359 vs 0.342/0.392/0.360): not yet a real test of the mean",
    "Real test: mean wherever the detector is lit (lit/dark still set by the sum), then rerun Stages 1–2"]))

pages.append(section_page('Reconstruction vs lit detectors', 'How many detectors a shower needs, and what the trigger threshold does'))
D15, D12 = R15 + "diagnostics/", R12 + "diagnostics/"
pages.append(page([D15 + "recon_sensitivity.png"], [
    "Run 15 · recon (fed the kernel) vs number of lit detectors and signal accuracy, held-out events",
    "Same events, keep n lit: 1 → 24°, 3 → 19°, 12 → 14°, 30 → 12°, all → 8.6° median angle error; no plateau",
    "Signal accuracy barely matters: E × up to ~1.6 at random or 300 ns T jitter change the error < 3%",
    "Recon is ~4× more accurate on surrogate input (2.0°) than on the kernel (8.6°): the surrogate is optimistic"]))

pages.append(page([D12 + "activation_threshold.png"], [
    "Run 12 (σ = 50) · detectors below the trigger set dark, as a real trigger would",
    "Same events, activated detectors kept: 3 → 29°, 5 → 20°, 10 → 14°, 15 → 11° (trigger count ≥ 1)",
    "Higher trigger: fewer activated (median 41 → 26 at ≥ 100) and worse recon overall (7.3° → 13.3°)",
    "At a fixed number of activated detectors the trigger barely matters (15 kept: 10.6–12.8°)"]))

pages.append(page([D15 + "activation_threshold.png"], [
    "Run 15 (σ = 10, interpolated kernel) · same test",
    "Same events, activated kept (count ≥ 1): 3 → 48°, 15 → 23°",
    "The recon leans on sub-threshold signals: removing counts < 1 raises the error 8.7° → 14.5°",
    "Partly out of distribution: the recon was trained on unthresholded detector responses"]))

pages.append(section_page('Gate scan', 'Trigger threshold and gate softness, optimized and judged on the kernel (runs 12, 15, 16)'))
pages.append(table_page("Would a different gate help the optimizer?", [
    "setting (1,500 Adam steps from grid)      run 12     run 15     run 16",
    "grid start (not optimized)                  5.37°      8.64°      8.72°",
    "current gate, two seeds               5.11/5.35°  8.15/8.15° 8.64/8.63°",
    "higher trigger (count > 3, 10, 30)     5.11–5.22°  8.21–8.44° 8.53–8.66°",
    "softer trigger (τ = 1, 0.3)            5.03–5.09°  8.24–8.39° 8.37–8.52°",
    "event gate > 10 / 20                   5.28° (20)  8.25–8.26° 8.48–8.84°",
    "softer event gate (τ = 0.1, 0.05)      5.14–5.19°  8.31–8.51° 8.48–8.78°",
    "strict: event > 30, trigger>3+event>20 5.17–5.27°  8.84–8.96° 8.90–9.10°",
], [
    "Median angle error of the optimized layout, recon fed the kernel, held-out events",
    "Softer or higher-trigger gates: within seed noise (~0.25°); strict event gates are worse (true U 28.6 vs ~31)",
    "High trigger makes gradients noisier (batch consistency 0.91 → 0.59 at count > 30, run 15)",
    "Every setting ranks test layouts like real recon quality (rank correlation 0.8–0.94): none removes the flatness"]))

pages.append(section_page('A new utility', 'Count well-reconstructed showers first, then accuracy for the rest'))
def utility_page():
    """The utility proposal, typeset with LaTeX (matplotlib usetex) on a white 16:9 page."""
    import io
    import matplotlib.pyplot as plt
    rc = {"text.usetex": True, "font.family": "sans-serif",
          "text.latex.preamble": r"\usepackage{amsmath}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
        fig.patch.set_facecolor("white")
        t = lambda y, s, **kw: fig.text(0.0625, y, s, color=INK, va="top", **kw)
        t(0.88, r"\textbf{A two-tier utility in reconstruction accuracy}", fontsize=30)
        t(0.77, r"For shower $i$: angle error $a_i$ [deg] and energy error $e_i = |\Delta \log_{10} E|$ (recon output vs truth)",
          fontsize=17)
        t(0.69, r"$\displaystyle g_i \;=\; \sigma\!\left(\frac{\theta_{\rm good}-a_i}{s_\theta}\right)"
                r"\,\sigma\!\left(\frac{e_{\rm good}-e_i}{s_e}\right)$", fontsize=24)
        fig.text(0.62, 0.655, r"shower $i$ reconstructed well (soft yes/no)", fontsize=16, color="#52514e", va="top")
        t(0.56, r"$\displaystyle q_i \;=\; \frac{1}{1+\left(a_i/\theta_{\rm ref}\right)^2+\left(e_i/e_{\rm ref}\right)^2}$",
          fontsize=24)
        fig.text(0.62, 0.525, r"partial accuracy: in $(0,1]$, diminishing returns", fontsize=16, color="#52514e", va="top")
        t(0.445, r"$\displaystyle U \;=\; \frac{1}{N}\sum_{i=1}^{N}\Big[\, g_i \;+\; \lambda\,(1-g_i)\,q_i \,\Big],"
                r"\qquad \lambda \approx 0.1\text{--}0.3$", fontsize=24)
        t(0.285, r"e.g.\ $\theta_{\rm good}=5^\circ$, $e_{\rm good}=0.2$, $s_\theta=1^\circ$, $s_e=0.05$, "
                r"$\theta_{\rm ref}=20^\circ$, $e_{\rm ref}=0.5$, $\sigma(x)=1/(1+e^{-x})$", fontsize=16)
        bullets = [
            r"Goal: as many showers as possible reconstructed well, then as much accuracy as possible for the rest",
            r"Today's $1/\mathrm{err}^2$ reward keeps paying for polishing good showers; $g_i$ stops paying once a shower is good",
            r"$\lambda<1$: the number of good showers dominates; the rest only decides between layouts with equal counts",
            r"Caveat: via the surrogate the recon looks $\sim\!4\times$ too accurate ($2.0^\circ$ vs $8.6^\circ$ on the kernel): calibrate thresholds",
            r"Test: optimize for a few $(\theta_{\rm good}, \lambda)$; judge by the fraction of held-out showers reconstructed well (kernel)",
        ]
        for k, b in enumerate(bullets):
            t(0.235 - 0.042 * k, r"$\bullet$\ \ " + b, fontsize=15)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, facecolor="white")
        plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((W, H), Image.LANCZOS)


pages.append(utility_page())

pages.append(table_page("Two-tier utility: test on run 16", [
    "utility (start)                      good 5°/0.2   good 3°/0.15   median angle   NN CV",
    "grid start, not optimized               13.2%          5.0%          8.72°       0.18",
    "current (grid), two seeds            13.9 / 13.2%   5.3 / 4.8%    8.62 / 8.71°   0.54–0.57",
    "two-tier λ=0.3 (grid), two seeds     13.9 / 14.7%   5.3 / 5.5%    8.56 / 8.47°   0.52",
    "two-tier λ=0.1 / λ=0 (grid)          13.2 / 14.6%   4.4 / 5.3%    8.61 / 8.41°   0.56",
    "two-tier, strict 3° (grid)              13.8%          5.5%          8.43°       0.59",
    "current (center start)                  13.4%          5.5%          8.63°       0.55",
    "two-tier λ=0.3 (center start)           13.6%          5.7%          8.40°       0.56",
], [
    "3,000 Adam steps through the FNN surrogate (run 16, lit/dark head) + run 15 recon; judged on held-out kernel events",
    "Thresholds calibrated by quantile matching: 5° on the kernel ≈ 1.6° through the surrogate (~3× optimistic)",
    "No measurable gain: 13.2–14.7% good vs 13.2–13.9% (seed noise up to 0.8 points, statistical ±0.9)",
    "No structure: every layout is a uniform fill; NN CV rises from the grid's 0.18 with either utility"]))
def two_tier_layouts_page():
    """The nine optimized layouts in one wide row, from the saved layout files."""
    import io, json
    import matplotlib.pyplot as plt
    import torch
    D = R16 + "diagnostics/"
    res = json.load(open(D + "two_tier_test.json"))["variants"]
    names = list(res)
    fig, ax = plt.subplots(2, 5, figsize=(W / 200 * 0.62, H / 200 * 0.8), dpi=200, sharex=True, sharey=True)
    ax = ax.ravel()
    ax[-1].axis("off")
    for a, name in zip(ax, names):
        L = torch.load(D + f"two_tier_layout_{name}.pt", map_location="cpu")
        a.scatter(L["x"].numpy(), L["y"].numpy(), s=9, color="#2a78d6", lw=0)
        r = res[name]
        a.set_title(f"{name.replace('two_tier', 'two-tier').replace('_', ' ')}\n"
                    f"good {r['frac_good_5']:.1%} · {r['ang_med']:.2f}°", fontsize=8, color=INK)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#d9d7d0")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200, facecolor="white"); plt.close(fig); buf.seek(0)
    img = Image.open(buf).convert("RGB")
    p = Image.new("RGB", (W, H), "white")
    h = H - 2 * M - 180
    img = img.resize((int(img.width * h / img.height), h), Image.LANCZOS)
    p.paste(img, ((W - img.width) // 2, M))
    d = ImageDraw.Draw(p)
    for k, b in enumerate(["Optimized layouts, current vs two-tier utility (run 16): all uniform fills, no core or halo",
                           "Title: variant, share of held-out showers good at 5°/0.2 (kernel), median angle error"]):
        d.text((M + 60, M + img.height + 50 + 66 * k), "•  " + b, fill=INK, font=FONT)
    return p


pages.append(two_tier_layouts_page())

pages.append(section_page('Detector-count scan', 'Is detector density what keeps the layout uniform? (run 15, 20 and 50 detectors)'))
def figure_page(path):
    """One figure alone, as large as the page allows."""
    img = Image.open(path).convert("RGB")
    k = min((W - 2 * M) / img.width, (H - 2 * M) / img.height)
    img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
    p = Image.new("RGB", (W, H), "white")
    p.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return p


pages.append(figure_page(R15 + "diagnostics/det20_summary.png"))
pages.append(figure_page(R15 + "diagnostics/det20_summary_2p5deg.png"))
pages.append(figure_page(R15 + "diagnostics/det100_summary_5deg.png"))
pages.append(figure_page(R15 + "diagnostics/det100_summary_2p5deg.png"))
pages.append(figure_page(R15 + "diagnostics/core_test/core_test.png"))

R17 = B + "17_plane45_interp_hurdle_100k/"
pages.append(section_page('Run 17: synthetic plane', 'Random tau decays next to a 45° plane: a clean geometry test'))
pages.append(text_page("Run 17 setup", [
    "Plane: 3 × 3 km, inclined 45°, facing East; the 100 detectors live on it",
    "Decays: Latin hypercube in a 3 × 3 × 3 km cube east of the plane, each aimed at a uniform point on the plane",
    "Cuts: decay–target distance ≥ 1 km, zenith 60–120°; log10 E uniform in 5–7; 105,300 events (100k train)",
    "Showers: flow-matching AllShowers (e, γ, μ), placed at the decay point",
    "Detector labels: kernel with adaptive widths, h = max(10 m, distance to 8th neighbour)",
    "Surrogate (lit/dark) and recon (30% kernel / 70% surrogate inputs) trained on those labels",
    "Optimization on top: Adam 10k steps, grid and center starts; current utility, two-tier λ = 0 and 0.3",
    "Layouts judged on kernel + recon (true utility) · Stage 4 results to follow",
]))
pages.append(page([R17 + "test_v6_run_01_northeast/decays_3d.png"], [
    "Run 17 · 300 of 105,300 decays (coloured by energy); line = direction to the target on the plane",
    "Decay–target distance p5 / p50 / p95: 1.9 / 3.6 / 5.1 km; zenith p5 / p50 / p95: 71° / 95° / 117°"]))
s2 = R17 + "test_v6_run_02_recentered/"
def grid_page(paths, bullets, ncol=2, cell_h=640, gap=50):
    """Figures in a grid (rows centred), short bullets under it."""
    p = Image.new("RGB", (W, H), "white")
    imgs = [Image.open(x).convert("RGB") for x in paths]
    rows = [imgs[i:i + ncol] for i in range(0, len(imgs), ncol)]
    y = M
    for row in rows:
        rs = [i.resize((int(i.width * cell_h / i.height), cell_h), Image.LANCZOS) for i in row]
        x = (W - (sum(r.width for r in rs) + gap * (len(rs) - 1))) // 2
        for r in rs:
            p.paste(r, (x, y))
            x += r.width + gap
        y += cell_h + gap
    d = ImageDraw.Draw(p)
    y += 10
    for b in bullets:
        d.text((M + 60, y), "•  " + b, fill=INK, font=FONT)
        y += 66
    return p


pages.append(grid_page([s2 + "fnn_electron_conditional.png", s2 + "fnn_muon_conditional.png",
                        s2 + "fnn_photon_conditional.png"], [
    "Run 17 · Stage 2 detector response model, conditional: electron, muon (top), photon (bottom)",
    "Mean lit log1p E 0.22 / 0.08 / 0.34 (run 15: 0.83 / 0.23 / 1.12); val loss 0.398 / 0.450 / 0.402",
    "Detectors 3× sparser (spacing 119 m vs 45 m): median 1–2 with ≥ 1 count per shower (run 15: 10–14)"]))
s3 = R17 + "test_v6_run_03_recentered_deepsets/"
pages.append(page([s3 + "recon_conditional_kernel.png", s3 + "recon_conditional.png"], [
    "Run 17 · Stage 3 recon conditional: kernel input (left), detector response model input (right)",
    "Val loss 0.0517 (run 15: 0.0665), but the East component is nearly free: every tau travels west",
    "Grid layout on the kernel: median angle 14.3°, 3.3% within 5° and 0.2 in log E (run 15: 8.6°, 13.2%)",
    "On surrogate input (right) recon is almost exact: median angle 2.4°, ~6× better than on the kernel (run 15: ~4×)"]))
pages.append(page([R17 + "archive/stage4_surrogate_2026-10-01/stage4_surrogate_summary.png"], [
    "Run 17 · Stage 4 through the surrogate (FNN + recon): Adam 10k steps, 3 chains; judged on 1,536 held-out kernel showers",
    "Current utility: good at 5° 3.3% → 4.3% (±0.5% stat.), kernel utility 15.5 → 17.2 (+11%); same from the center start",
    "Current utility puts a dense row of detectors along the southern edge (North ≈ −1,450 m) from both starts",
    "Two-tier no better; λ = 0 from the center start gets no gradient (almost no good showers) and stays in the blob"]))
for tag in ("cores", "hybrids", "optimized"):
    pages.append(figure_page(R17 + f"archive/stage4_surrogate_2026-10-01/diagnostics/core_test/core_test_{tag}.png"))

# Run 17, kernel in the loop: recon trained on 100% kernel responses, layouts optimized on the kernel.
# Added only once its results exist (recon_kernel_run17.sh, kernel_opt_run17.sh, score_core_test.sh).
RK = R17 + "test_v6_run_03_recentered_deepsets_kernel100/"
CT = R17 + "diagnostics/core_test/"
if all(os.path.exists(p) for p in (RK + "recon_conditional_kernel.png", CT + "core_test_scores.json",
                                   CT + "core_test_kernel_optimized.png")):
    import json
    import torch
    ck = torch.load(RK + "recon.pt", map_location="cpu", weights_only=False)
    tl = json.load(open(RK + "recon_train_log.json"))
    sc = json.load(open(CT + "core_test_scores.json"))
    g = sc["grid"]
    pages.append(section_page('Run 17: kernel in the loop',
                              'Recon trained on kernel responses only; layouts optimized directly on the kernel'))
    pages.append(page([RK + "recon_conditional_kernel.png"], [
        "Run 17 · Stage 3 recon trained on 100% kernel responses (was 30% kernel / 70% surrogate), kernel input",
        f"Val loss {tl.get('adam_best_val', tl['best_val_total']):.4f} at epoch "
        f"{tl.get('adam_best_epoch', tl['best_epoch'])} (Adam stopped at {sum('phase' not in e for e in tl['log'])}), "
        f"{float(ck.get('val_total', float('nan'))):.4f} after L-BFGS · 30/70 recon: 0.0517",
        f"Grid layout: median angle {g['ang_med']:.1f}°, {g['frac_good_5']:.1%} within 5° and 0.2 in log E "
        "(30/70 recon: 14.3°, 3.3%)"]))
    for tag in ("cores", "hybrids"):
        pages.append(figure_page(CT + f"core_test_{tag}.png"))
    ko = [v for k, v in sc.items() if k.startswith("kern_")]
    rng = lambda key, s=1: f"{min(v[key] for v in ko) * s:.1f}–{max(v[key] for v in ko) * s:.1f}"
    pages.append(page([CT + "core_test_kernel_optimized.png"], [
        "Kernel in the loop: Adam lr = 10, 20k steps, 512 fresh events per step; best layout on a separate 1,024",
        f"Good at 5°: {rng('frac_good_5', 100)}% vs grid {g['frac_good_5']:.1%} (±0.5% stat.); "
        f"median angle {rng('ang_med')}° vs {g['ang_med']:.1f}°; kernel utility {rng('U_kernel')} vs {g['U_kernel']:.1f}",
        "Center starts all leave the blob, two-tier λ = 0 included (it stayed stuck through the surrogate)",
        "Same size of gain as through the surrogate (~+1 point at 5°): optimizing on the kernel finds nothing better"]))

# Run 17 shower-geometry checks (diagnostics/array_size_single_shower, zenith_cap100, kernel_frame).
AS, ZC, KF = R17 + "diagnostics/array_size_single_shower/", R17 + "diagnostics/zenith_cap100/", R17 + "diagnostics/kernel_frame/"
if all(os.path.exists(p) for p in (AS + "array_size_shower_off_cap50.png", ZC + "zenith_cap_check.png",
                                   KF + "kernel_frame_issue_cap50.png")):
    pages.append(section_page('Shower geometry checks',
                              'One shower vs array size, shower direction vs label, kernel frame, layer placement'))
    pages.append(page([AS + "array_size_shower_center_uncapped.png", AS + "array_size_shower_center_cap50.png"], [
        "One held-out shower, core 52 m from the plane centre (zenith 109°): 100-detector disks of 50 m – 1 km + grid",
        "Left: kernel widths uncapped (up to 1.7 km) · right: capped at 50 m · recon trained on uncapped kernel responses",
        "Uncapped: all 100 detectors of the 50 m disk ≥ 1 count; angle 15–31° · capped: 13–23°, log10 E 0.14–0.8 low",
        "Deposits sit off the core ★ (where the labelled axis meets the plane): see the direction check below"]))
    pages.append(page([AS + "array_size_shower_off_uncapped.png", AS + "array_size_shower_off_cap50.png"], [
        "Same test, core 568 m from the plane centre (zenith 112°); disks stay at the centre",
        "Uncapped: the 50 m disk, 570 m from the core, still has all 100 detectors ≥ 1 count (~50 counts each)",
        "Capped: 0 detectors in the 50 m disk; best angle 4.4° with the 500 m disk · single showers are noisy",
        "Both showers have zenith > 100°, outside the AllShowers training range (60–100°)"]))
    pages.append(page([ZC + "zenith_cap_check.png"], [
        "Zenith ≤ 100°: the axis tilt does not change; it depends only on the angle from due West (panel 1)",
        "Grid, good at 5°: 4.9% (≤ 100°) vs 1.1% (> 100°) uncapped, 3.6% vs 2.2% capped · median angle 13.7° vs 15.2°",
        "Run 18 geometry with the cut at 100°: 53% of throws kept (82% at 120°), same azimuth spread",
        "A shower with 0 detectors ≥ 1 count was still reconstructed to 4.6–6°: the recon's direction prior is narrow"]))
    pages.append(table_page("The shower tilt follows the simulation's magnetic field B", [
        "angle between shower axis and B   sideways shift per km of depth   C8 test showers",
        "             0–10°                       0.38  (21°)                    220",
        "            10–20°                       0.31  (17°)                    552",
        "            20–30°                       0.22  (12°)                    804",
        "          30° and more                   0.00                         ~4,300",
    ], [
        "Hamza's C8 simulation sets B = (22.7 μT, −2.5 nT, −4.3 nT) in its root frame: along +x, our East–West axis",
        "Within ~30° of B, each layer's particles form a narrow band (20–200 m) shifted 0.85–2.8 km toward −y (South)",
        "Beyond 30°: normal lateral spread (± a few hundred m) centred on the axis",
        "Present in the C8 training showers; the generator reproduces it (run 17: e +9.5°, μ +9.9°, γ +18.5° in azimuth)",
        "A field parallel to the shower should bend it least: suspected C8 tracking issue for particles moving along B",
        "Not confirmed: raw C8 output was deleted; test = one +x and one +y shower, with B as set and with B = 0",
        "Run 17/18: ~80% of showers within 30° of West → affected · Malata (taus heading North) → not affected"]))
    pages.append(page([KF + "kernel_frame_issue.png"], [
        "Rows: run 17 shower ~due West, ~40° off West, Malata shower heading ~North · uncapped widths",
        "Columns: top view (grey: kernel's East slabs, blue: shower's layers) · footprint as used / layer fix / "
        "shower frame",
        "The kernel matches in global North–Up with East slabs; C8 records each shower on planes perpendicular to its axis",
        "The two differ for every shower; Malata: the kernel smears the shower into a ±500 m East–West line"]))
    pages.append(page([KF + "kernel_frame_issue_cap50.png"], [
        "Same, widths capped at 50 m: footprints shrink to 35–80 faces with ≥ 1 count",
        "Due West: all three versions now agree; the signal lands ~1 km South of the ★ (the tilt above)",
        "40° off West: the kernel as used draws a ~2 km strip along North; the shower frame a compact patch at the ★",
        "Malata: unchanged by the cap; the shower frame puts ~4× more signal near the ★ on fewer faces"]))
    pages.append(table_page("Layer placement is one layer short", [
        "C8 plane k sits at k × 500 m along the axis, k = 1–24  →  stored layer l at depth (l + 1) × 500 m",
        "measured on the C8 test showers (e, μ, γ): depth / 500 m = l + 1.00 for every layer 0–15",
        "place_clouds_enu: Up = decay Up + l × 500 m × d_Up   (East, North come from x/y and are right)",
    ], [
        "Up error = 500 m × d_Up: 12–45 m for near-horizontal showers, ~170 m at ±20° elevation, ~250 m at ±30°",
        "Explains the +150–190 m Up offsets seen for two run 17 showers at zenith 109–112°",
        "Every Stage 1 label built since the July placement fix carries it · fix: depth (l + 1) × 500 m, then rebuild Stage 1"],
        header=False))
    pages.append(table_page("Shower directions: training vs our runs", [
        "                     zenith of travel                  azimuth of travel",
        "AllShowers training  60–100°, flat                     all directions, flat (0–30° bin 20% low)",
        "run 17 / run 18      60–120° (35% above 100°)          80% within 30° of due West",
        "Malata (run 15)      56–120° (16–84%: 72–96°)          mostly North to North-West (30–150°)",
    ], [
        "130,000 primaries shared by the e, μ, γ training sets · energy 10^5–10^7.7 GeV · 50/50 EM / hadronic",
        "Run 17/18: the East-facing plane forces West-going showers, inside the tilted range around B",
        "A shower-frame kernel would drop the East-facing requirement and allow any shower direction"]))

# Run 17 diagnostics with the fixed kernel (diagnostics/array_size_fixed_kernel, kernel_frame/kernel_schematic).
FK = R17 + "diagnostics/array_size_fixed_kernel/"
if all(os.path.exists(p) for p in (KF + "kernel_schematic.png",
                                   FK + "array_size_ablation_off_zen100_fixed_tinterp_oncore_b30.json")):
    import json as _json
    pages.append(section_page('Run 17 diagnostics: fixed kernel',
                              'Shower-frame kernel, layer depth fixed, arrival times interpolated · decisions for run 18'))
    pages.append(figure_page(KF + "kernel_schematic.png"))
    for w, ev_txt in (("center", "shower 4287 (core 72 m from centre, zenith 92°, 36° from B)"),
                      ("off", "shower 2283 (core 531 m off, zenith 86°, 35° from B)")):
        pages.append(page([FK + f"array_size_shower_{w}_zen100_oncore_b30_cap50.png",
                           FK + f"array_size_shower_{w}_zen100_fixed_tinterp_oncore_b30_cap50.png"], [
            f"Run 17 · {ev_txt} · disks centred on the core + grid · 50 m cap",
            "Left: kernel as used (global frame) · right: shower frame, layer depth fixed, times interpolated between planes",
            "Recon trained on the old kernel for both: errors on the right are only indicative",
            "Right: footprint on the ★, smooth arrival-time gradient across the array (left: flat with stray late detectors)"]))
    def _abl(w, tag):
        return _json.load(open(FK + f"array_size_ablation_{w}_zen100_{tag}oncore_b30.json"))["cap50"]
    cols = ("r = 300 m", "r = 500 m", "r = 750 m", "r = 1000 m", "grid")
    rows = ["shower  kernel                       " + "".join(f"{c.replace('r = ', ''):>9s}" for c in cols)]
    for w, name in (("center", "4287"), ("off", "2283")):
        for tag, lab in (("", "as used (global)"), ("fixed_", "shower frame, plane times"),
                         ("fixed_tinterp_", "shower frame, interpolated")):
            a = _abl(w, tag)
            rows.append(f"{name:6s}  {lab:28s} " + "".join(f"{a[c]['full']:8.1f}°" for c in cols))
    pages.append(table_page("Angle error by array size, widths capped at 50 m", rows, [
        "Small disks (50–300 m) fail for both showers with every detector lit: no lever arm for the direction",
        "Interpolated times: error falls steadily with array size; shower 2283 within 6° from 500 m up",
        "Two showers only, recon trained on the old kernel: retrain before drawing conclusions"]))
    rows = ["shower 4287, shower frame, 50 m cap   lit   span   time spread    full  no time  no energy  no signal"]
    for tag, lab in (("fixed_", "plane times"), ("fixed_tinterp_", "interpolated")):
        a = _abl("center", tag)
        for c in ("r = 750 m", "r = 1000 m"):
            r = a[c]
            rows.append(f"{lab:13s} {c:>10s}            {r['lit']:4d} {r['span_m']:5.0f} m {r['t_spread_ns']:7.0f} ns  "
                        f"{r['full']:6.1f}° {r['no time']:7.1f}° {r['no energy']:8.1f}° {r['no signal']:9.1f}°")
    pages.append(table_page("What the recon uses, and why times were quantized", rows, [
        "No signal: 31–47° off for these showers, so their good results are not the recon's prior",
        "No time costs 10–30°: arrival times carry most of the direction",
        "Plane times jump in steps of one C8 layer (500 m / c ≈ 1.6 μs): a detector gets its earliest plane's time",
        "At 750 m the lit detectors straddle a step (1,523 ns spread), at 1 km they don't (266 ns): 22° vs 4°",
        "Interpolated, t = t_point + (detector depth − plane depth) / c: spreads grow smoothly with distance"]))
    pages.append(text_page("Decisions for run 18", [
        "Layer depth (l + 1) × 500 m · kernel in each shower's own frame (transverse t1, t2 and depth along the axis)",
        "Arrival times interpolated between planes · circular showers: t2 from the measured t1 spread of the same layer",
        "Kernel widths capped at 50 m",
        "Plane from run 17, 6 × 6 km, 45°, facing South like Malata: taus travel North into it",
        "Showers ≥ 30° from the C8 field axis, zenith 60–100° (AllShowers training range), 210,600 events, 100 detectors",
        "Retrain everything except the Stage 2 surrogate: Stage 0, Stage 1 labels, Stage 3 recon on the kernel",
        "Then 6 kernel-in-the-loop optimizations (current, two-tier λ 0 / 0.3 × grid / center), scoring and core test",
    ]))

# Run 18: South-facing 6 km plane, shower-frame kernel, no Stage 2 (results 2026-10-06).
R18 = B + "18_plane45S_6km_showerframe_200k/"
D18 = R18 + "diagnostics/"
if all(os.path.exists(p) for p in (R18 + "inputs/plane_primaries.png", D18 + "core_test/core_test_scores.json",
                                   D18 + "angleonly/angleonly_scores.png", D18 + "core_test_ndet/core_test_ndet_1deg.png")):
    import json as _json
    pages.append(section_page('Run 18: shower-frame kernel',
                              '6 km plane facing South, 210,600 showers, ring showers, no surrogate'))
    pages.append(text_page("Run 18 setup", [
        "Plane: 6 × 6 km, 45°, facing South like Malata; taus travel North into it; 100 detectors",
        "Decays: LHS box south of the plane, aimed at uniform points on it; distance ≥ 1 km, zenith 60–100°",
        "Directions ≥ 30° from the C8 field axis (x), outside the range C8 shifts sideways · 210,600 events (200k train)",
        "Kernel in each shower's own frame: layer depth (l + 1) × 500 m, arrival times interpolated between planes",
        "Ring showers: each particle a ring of radius |t1| around the axis (C8 lost the vertical offset)",
        "Adaptive widths k = 8, floor 10 m, cap 50 m · recon trained on kernel responses only (no Stage 2)",
        "Then 6 kernel-in-the-loop optimizations (current, two-tier λ 0 / 0.3 × grid / center), scoring, core test",
    ]))
    pages.append(page([R18 + "inputs/plane_primaries.png"], [
        "Run 18 · primaries: decays south of the plane, median distance 6.5 km, zenith p5/p50/p95 68 / 88 / 99°",
        "Azimuth of travel 50–130° (heading North), every shower ≥ 30° from the x axis"]))
    pages.append(page([R18 + "test_v6_run_03_recentered_deepsets/recon_conditional_kernel.png"], [
        "Run 18 · recon on kernel input: direction tight on the diagonal; val loss 0.063 (run 17 kernel recon: 0.148)",
        "Energy is the weak part: a flat band near the mean for showers with almost no signal"]))
    sc18 = _json.load(open(D18 + "core_test/core_test_scores.json"))
    g18 = sc18["grid"]
    for tag in ("cores", "hybrids", "kernel_optimized"):
        pages.append(figure_page(D18 + f"core_test/core_test_{tag}.png"))
    pages.append(table_page("Run 18 core test: angle and energy vs angle only", [
        "layout                              good 5° (+E)   within 5°   within 2.5°   within 1°   median",
    ] + [f"{lab:34s} {100 * sc18[k]['frac_good_5']:8.1f}%   {100 * a5[k]['frac_good_t']:8.1f}%   "
         f"{100 * a25[k]['frac_good_t']:8.1f}%   {100 * a1[k]['frac_good_t']:8.1f}%   {sc18[k]['ang_med']:5.1f}°"
         for a5, a25, a1 in [(_json.load(open(D18 + "core_test_ndet/scores_pad_5deg.json")),
                             _json.load(open(D18 + "core_test_ndet/scores_pad_2.5deg.json")),
                             _json.load(open(D18 + "core_test_ndet/scores_pad_1deg.json")))]
         for k, lab in (("grid", "grid"), ("core_r300", "core 300 m"), ("core_r500", "core 500 m"),
                        ("core300_40_spread60", "40 in 300 m core + 60 spread"),
                        ("core300_60_rim40", "60 in 300 m core + 40 on rim"),
                        ("kern_current_grid", "optimized, current (grid)"),
                        ("kern_twotier_l0_grid", "optimized, two-tier λ 0 (grid)"),
                        ("kern_twotier_l0p3_center", "optimized, two-tier λ 0.3 (center)"))], [
        "Run 18 vs run 17 grid: good at 5° (+E) 3.5% → 38.3%, median angle 14° → 2.8° (not a controlled comparison)",
        "Without the energy condition the grid has 77% within 5°: energy, not direction, is the main limit",
        "Grid-start optimizations end within ~1.5 points of the grid; center starts spread but don't catch up",
        "Showers with no detector ≥ 1 count still get 3.3°: the recon uses sub-1-count ring signals"]))
    pages.append(figure_page(D18 + "angleonly/angleonly_scores.png"))
    pages.append(page([D18 + "detector_count/detector_count_grid.png"], [
        "Grid over the plane with 4–100 detectors: within 5° (angle) 8% → 77%, median angle 16° → 2.8°",
        "Still rising at 100 detectors, more slowly (50 → 100: +9 points)"]))
    pages.append(page([D18 + "detector_count/recon_sensitivity.png"], [
        "Keeping 1–3 of a shower's lit detectors: ~12°; all of them: 4.5° · energy noise ×1.6 or 100 ns jitter barely matter",
        "Natural: 2.2° for showers lighting 8–11 detectors, 3.4° with none ≥ 1 count"]))
    pages.append(page([D18 + "footprint/footprint_radius.png"], [
        "Footprint on the grid: half the collected signal within ~300 m of the axis, 90% within ~440 m (medians)",
        "Grid spacing ~500 m, nearest detector ~210 m from the axis → median 4 lit detectors per shower",
        "Particles' energy-weighted radius (5.8 km) dominated by far-out points beyond the plane: to check"]))
    pages.append(figure_page(D18 + "core_test_ndet/core_test_ndet_pad.png"))
    pages.append(figure_page(D18 + "core_test_ndet/core_test_ndet_1deg.png"))
    pages.append(text_page("Run 18: why nothing beats the grid", [
        "Showers land uniformly on the plane: N detectors over area A give every layout the same mean count per shower",
        "Clustering shares them out unevenly: many detectors for the few showers landing there, none for the rest",
        "Compact cores: 5–21% within 5° and < 1.2% within 1° at any detector count; hybrids improve with spread",
        "Disks peak at 3 km radius (≈ covering the plane): 79% / 70% / 46% within 5° for 100 / 50 / 20 detectors",
        "Within 1°: grid 12.2% vs best optimized 13.2% (100 det.) · 3 km disk beats the grid at 20 and 50 detectors",
        "N < 100: empty recon slots set to 0 (repeating the detectors instead changes scores by 1–4 points)",
        "Structure should need uneven illumination (real mountain) or a different goal; recon trained on 100 detectors",
    ]))

pages.append(text_page("Status and open questions", [
    "Kernel-in-the-loop on Stage 4's 75k events: grid 30.48 → 30.57, center 7.13 → 30.50 (surrogate: 30.85 / 30.79)",
    "Gate scan (runs 12, 15, 16): no trigger or softness setting beats the current gate",
    "Run 16: labels ~99% identical to run 15; a true-mean rerun is needed to test the idea",
    "Near-miss tau corpus (axis within ~3 km of the mesh): requested",
    "Two-tier utility (run 16): no gain, no structure; next: near-miss taus, bootstrap error bars",
    "Run 18 (shower-frame kernel, ring showers): grid 77% within 5°, nothing concentrated beats even coverage",
    "Next: clustered grids, a recon trained on variable detector counts, and uneven (Malata-like) illumination",
    "C8: confirm the shift for showers within 30° of B (test run or ask Hamza); check the B units (nT vs μT)",
]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=200)
print(f"[write] {OUT}  ({len(pages)} pages)")
