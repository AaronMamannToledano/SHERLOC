"""fig1b_svg.py — Figure 1b: SHERLOC architecture, in the visual language of the PhD-defence slides.

The same ctDNA data are read along two axes: along time (ctDNA-dynamics encoder: one temporal kernel shared by all
genes, one temporal map per panel-level feature) and across genes (genomic-interaction encoder, pre-trained on
MSK-CHORD and frozen, applied at every draw). Every representation enters a Cox layer with its own coefficient;
the score is then recalibrated out of fold.
Text: TeX Gyre Heros (Helvetica); math: Latin Modern Math. Notation as in the Methods.
Usage: python fig1b_svg.py -> article/figures/fig1b_architecture.svg / .png (7.2 in wide, 600 dpi)
"""
import subprocess
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "article" / "figures"   # analyses/v2/article/figures
W, H = 1440, 420
PT = 200 / 72
SANS, MATH = "TeX Gyre Heros", "Latin Modern Math"
CRIMSON, PLUM, BLUE, AMBER = "#c60b46", "#63003c", "#1f5fae", "#c98a12"
PINK_BG, BLUE_BG, GREY_BG = "#fcf0f4", "#eef4fb", "#f3f4f6"
INK, INK2, INK3, LINE = "#1e1e1e", "#5b5b5b", "#9a9a9a", "#b9b9b9"
RISK = [("low", BLUE), ("intermediate", AMBER), ("high", CRIMSON)]

el = []
add = el.append


def fs(pt):
    return round(max(pt, 5.2) * PT * 0.84, 2)


def text(x, y, s, size=5.4, color=INK, weight=400, anchor="start", style="normal", extra=""):
    add(f'<text xml:space="preserve" x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{fs(size)}" fill="{color}" '
        f'font-weight="{weight}" font-style="{style}" text-anchor="{anchor}" {extra}>{s}</text>')


def _it(s):
    o = []
    for ch in s:
        if ch == "h": o.append("ℎ")
        elif "a" <= ch <= "z": o.append(chr(0x1D44E + ord(ch) - 97))
        elif "A" <= ch <= "Z": o.append(chr(0x1D434 + ord(ch) - 65))
        elif "α" <= ch <= "ω": o.append(chr(0x1D6FC + ord(ch) - 0x3B1))
        elif ch == "Φ": o.append("\U0001D6F7")
        else: o.append(ch)
    return "".join(o)


def _bf(s):
    o = []
    for ch in s:
        if "a" <= ch <= "z": o.append(chr(0x1D41A + ord(ch) - 97))
        elif "A" <= ch <= "Z": o.append(chr(0x1D400 + ord(ch) - 65))
        elif "α" <= ch <= "ω": o.append(chr(0x1D736 + ord(ch) - 0x3B1))
        else: o.append(ch)
    return "".join(o)


def math(x, y, parts, size=6.2, color=INK, anchor="middle"):
    """parts: (string, kind); kind in it | rm | bf | bb, optionally _sub / _sup."""
    out, cur = [], 0.0
    for s, k in parts:
        base, _, pos = k.partition("_")
        if base == "it": s = _it(s)
        elif base == "bf": s = _bf(s)
        elif base == "bb": s = s.replace("R", "ℝ")
        s = s.replace(" ", " ")
        tgt = {"sub": 0.25, "sup": -0.42}.get(pos, 0.0) * fs(size)
        sz = fs(size * (0.70 if pos else 1))
        out.append(f'<tspan dy="{tgt - cur:.2f}" font-size="{sz}">{s}</tspan>'); cur = tgt
    add(f'<text xml:space="preserve" x="{x:.1f}" y="{y:.1f}" font-family="{MATH}" fill="{color}" '
        f'text-anchor="{anchor}">{"".join(out)}</text>')


def arrow(pts, color=LINE, sw=1.0):
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    L = np.hypot(xb - xa, yb - ya); ux, uy = (xb - xa) / L, (yb - ya) / L
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts[:-1]) + f" L{xb - ux * 5:.1f},{yb - uy * 5:.1f}"
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"/>')
    px, py = -uy, ux
    add(f'<path d="M{xb},{yb} L{xb - ux * 7 + px * 3:.1f},{yb - uy * 7 + py * 3:.1f} '
        f'L{xb - ux * 7 - px * 3:.1f},{yb - uy * 7 - py * 3:.1f} Z" fill="{color}"/>')


def rect(x, y, w, h, r=4, fill="#ffffff", stroke="none", sw=1.0, extra=""):
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{sw}" {extra}/>')


def snowflake(cx, cy, s=1.0, col=BLUE):
    g = "".join(f'<g transform="rotate({a})"><line x1="0" y1="-7" x2="0" y2="7"/>'
                f'<path d="M-2.2,-5.2 L0,-3.4 L2.2,-5.2 M-2.2,5.2 L0,3.4 L2.2,5.2" fill="none"/></g>' for a in (0, 60, 120))
    add(f'<g transform="translate({cx},{cy}) scale({s})" stroke="{col}" stroke-width="1.3" stroke-linecap="round">{g}</g>')


def chip(x, y, w, parts, h=22):
    rect(x, y - h / 2, w, h, r=5, stroke=LINE, sw=0.9)
    math(x + w / 2, y + 4.5, parts, size=6, color=INK2)


def tint(a):                                       # crimson tint, a in [0, 1]
    c = (198, 11, 70)
    return "#%02x%02x%02x" % tuple(round(255 - a * (255 - v)) for v in c)


def btint(a):
    c = (31, 95, 174)
    return "#%02x%02x%02x" % tuple(round(255 - a * (255 - v)) for v in c)


rng = np.random.RandomState(11)

# ═════ Left: the same data sliced two ways ═══════════════════════════════════
text(22, 34, "Along time", size=5.6, color=PLUM, weight=700)
text(116, 34, "one row at a time", size=4.6, color=INK3)
cx0, cw = 120, 30
for j, lab in enumerate(["1", "2", "3", "4"]):
    math(cx0 + j * cw + cw / 2, 50, [("t", "it"), (lab, "rm_sub")], size=5.2, color=INK3)
rows = [("TP53", True), ("PIK3CA", True), ("KRAS", True), (None, None), ("mean VAF", False), ("driver count", False)]
ry = 58
row_y = {}
for name, gene in rows:
    if name is None:
        text(cx0 + 2 * cw, ry + 6, "⋮", size=6, color=INK3, anchor="middle"); ry += 12; continue
    text(112, ry + 10.5, name, size=5, color=INK, anchor="end", style="italic" if gene else "normal")
    vals = rng.rand(4)
    for j in range(4):
        add(f'<rect x="{cx0 + j * cw}" y="{ry}" width="{cw}" height="14" fill="{tint(0.15 + 0.8 * vals[j])}"/>')
    rect(cx0, ry, 4 * cw, 14, r=3, fill="none", stroke=CRIMSON, sw=0.8)
    row_y[name] = ry + 7; ry += 19

text(22, 196, "Across genes", size=5.6, color=PLUM, weight=700)
text(132, 196, "one draw at a time", size=4.6, color=INK3)
for k, g in enumerate(["TP53", "PIK3CA", "KRAS"]):
    text(112, 214 + k * 14, g, size=5, color=INK, anchor="end", style="italic")
for j in range(4):
    x = cx0 + 4 + j * cw
    for k in range(3):
        add(f'<rect x="{x}" y="{204 + k * 14}" width="22" height="14" fill="{tint(0.15 + 0.8 * rng.rand())}"/>')
    rect(x, 204, 22, 42, r=3, fill="none", stroke=BLUE, sw=1.0)
    math(x + 11, 262, [("v", "bf"), (str(j + 1), "rm_sub")], size=5.2, color=INK3)

rect(22, 318, 218, 26, r=6, stroke=LINE, sw=0.9)
text(131, 335, "Clinical covariates", size=5.2, color=INK, anchor="middle")

arrow([(248, 108), (284, 108)])
arrow([(248, 226), (284, 226)])
arrow([(244, 331), (284, 331)])

# ═════ ctDNA-dynamics encoder ════════════════════════════════════════════════
ex, ew = 290, 470
rect(ex, 26, ew, 162, r=8, fill=PINK_BG, stroke=CRIMSON, sw=1.3)
text(ex + 14, 46, "ctDNA-dynamics encoder", size=6.2, color=CRIMSON, weight=700)
text(ex + 14, 61, "How does each signal evolve under treatment?", size=4.8, color=INK2, style="italic")
# gene level: one kernel shared
kx, ky, kh = 398, 86, 72
text(kx + 17, 80, "one kernel ", size=4.6, color=CRIMSON, weight=700, anchor="end")
math(kx + 19, 80, [("w", "bf")], size=5.4, color=CRIMSON, anchor="start")
text(kx + 27, 80, ", shared by all genes", size=4.6, color=CRIMSON, weight=700)
traj = [[0.9, 0.6, 0.45, 0.3], [0.4, 0.42, 0.5, 0.75], [0.7, 0.3, 0.25, 0.45]]
kern = [0.55, 0.85, 0.95, 0.25]
for i, tr in enumerate(traj):
    yy = ky + 12 + i * 24
    pts = " ".join(f"{ex + 16 + j * 14:.1f},{yy + 8 - tr[j] * 16:.1f}" for j in range(4))
    add(f'<polyline points="{pts}" fill="none" stroke="#555" stroke-width="1"/>')
    for j in range(4):
        add(f'<circle cx="{ex + 16 + j * 14}" cy="{yy + 8 - tr[j] * 16:.1f}" r="1.7" fill="#555"/>')
    add(f'<line x1="{ex + 14}" y1="{yy + 10}" x2="{ex + 62}" y2="{yy + 10}" stroke="#d7cfd2" stroke-width="0.6"/>')
    arrow([(ex + 70, yy + 2), (kx - 4, yy + 2)], sw=0.8)
    for j, v in enumerate(kern):
        add(f'<rect x="{kx + 10 + j * 8}" y="{yy + 8 - v * 14:.1f}" width="6" height="{v * 14:.1f}" fill="{CRIMSON}"/>')
    arrow([(kx + 52, yy + 2), (kx + 74, yy + 2)], sw=0.8)
    math(kx + 80, yy + 6, [("h", "it"), (["TP53", "PIK3CA", "KRAS"][i], "rm_sub")], size=5.4, color=INK2, anchor="start")
rect(kx, ky, 50, kh, r=4, fill="#ffffff", stroke=CRIMSON, sw=1.2)
for i in range(3):                                  # redraw bars above the white kernel box
    yy = ky + 12 + i * 24
    for j, v in enumerate(kern):
        add(f'<rect x="{kx + 10 + j * 8}" y="{yy + 8 - v * 14:.1f}" width="6" height="{v * 14:.1f}" fill="{CRIMSON}"/>')
math(ex + 118, 178, [("h", "it"), ("g", "it_sub"), (" = ", "rm"), ("σ", "it"), ("(", "rm"), ("∑", "rm"), ("t", "it_sub"),
                     (" w", "it"), ("t", "it_sub"), (" x", "it"), ("g,t", "it_sub"), (" + ", "rm"), ("b", "it"), (")", "rm")],
     size=6.2)
add(f'<line x1="{ex + 262}" y1="74" x2="{ex + 262}" y2="180" stroke="#e3c4cf" stroke-width="0.8" stroke-dasharray="3 2"/>')
# panel level: one map per feature
px0 = ex + 276
text(px0 + 70, 80, "one map ", size=4.6, color=CRIMSON, weight=700, anchor="end")
math(px0 + 72, 80, [("W", "bf"), ("f", "it_sub")], size=5.4, color=CRIMSON, anchor="start")
text(px0 + 86, 80, " per feature", size=4.6, color=CRIMSON, weight=700)
ptraj = [[0.85, 0.55, 0.5, 0.35], [0.3, 0.35, 0.3, 0.8]]
pk = [[0.2, 0.5, 0.95, 0.4], [0.1, 0.25, 0.6, 0.95]]
for i, (tr, kk) in enumerate(zip(ptraj, pk)):
    yy = 98 + i * 34
    pts = " ".join(f"{px0 + 2 + j * 12:.1f},{yy + 8 - tr[j] * 16:.1f}" for j in range(4))
    add(f'<polyline points="{pts}" fill="none" stroke="#555" stroke-width="1"/>')
    for j in range(4):
        add(f'<circle cx="{px0 + 2 + j * 12}" cy="{yy + 8 - tr[j] * 16:.1f}" r="1.7" fill="#555"/>')
    arrow([(px0 + 46, yy + 2), (px0 + 66, yy + 2)], sw=0.8)
    rect(px0 + 68, yy - 10, 44, 24, r=3, fill="#ffffff", stroke=CRIMSON, sw=1.1)
    for j, v in enumerate(kk):
        add(f'<rect x="{px0 + 76 + j * 8}" y="{yy + 10 - v * 16:.1f}" width="6" height="{v * 16:.1f}" fill="{CRIMSON}"/>')
    arrow([(px0 + 114, yy + 2), (px0 + 132, yy + 2)], sw=0.8)
    math(px0 + 137, yy + 6, [("r", "bf"), (["VAF", "drv"][i], "rm_sub")], size=5.4, color=INK2, anchor="start")
math(px0 + 92, 178, [("r", "bf"), ("f", "it_sub"), (" = ", "rm"), ("W", "bf"), ("f", "it_sub"), (" z", "bf"), ("f", "it_sub"),
                     (" + ", "rm"), ("b", "bf"), ("f", "it_sub")], size=6.2)

# ═════ Genomic-interaction encoder ═══════════════════════════════════════════
gy = 200
rect(ex, gy, ew, 104, r=8, fill=BLUE_BG, stroke=BLUE, sw=1.3)
text(ex + 14, gy + 20, "Genomic-interaction encoder", size=6.2, color=BLUE, weight=700)
text(ex + 14, gy + 35, "Which alterations occur together at one draw?", size=4.8, color=INK2, style="italic")
snowflake(ex + ew - 16, gy + 16, 0.95)
for j in range(4):
    x = ex + 18 + j * 18
    for k in range(4):
        add(f'<rect x="{x}" y="{gy + 48 + k * 9}" width="11" height="9" fill="{tint(0.12 + 0.8 * rng.rand())}"/>')
    rect(x, gy + 48, 11, 36, r=2, fill="none", stroke=BLUE, sw=0.8)
    add(f'<path d="M{x + 11},{gy + 66} C{x + 40 - j * 6},{gy + 66} {ex + 112},{gy + 66} {ex + 120},{gy + 66}" '
        f'fill="none" stroke="{LINE}" stroke-width="0.7"/>')
arrow([(ex + 112, gy + 66), (ex + 124, gy + 66)], sw=0.8)
rect(ex + 126, gy + 50, 56, 32, r=4, fill="#ffffff", stroke=BLUE, sw=1.2)
math(ex + 150, gy + 71, [("W", "bf"), ("pre", "rm_sub")], size=6.4, color=BLUE)
snowflake(ex + 176, gy + 56, 0.45)
for j in range(4):
    x = ex + 206 + j * 20
    arrow([(ex + 184, gy + 66), (x - 3, gy + 58 + (j - 1.5) * 1)], sw=0.6) if j == 0 else None
    for k in range(2):
        add(f'<circle cx="{x + 4}" cy="{gy + 58 + k * 13}" r="4" fill="{btint(0.35 + 0.5 * rng.rand())}" stroke="{BLUE}" stroke-width="0.7"/>')
    math(x + 4, gy + 92, [("e", "bf"), (str(j + 1), "rm_sub")], size=5, color=INK3)
math(ex + 385, gy + 58, [("e", "bf"), ("t", "it_sub"), (" = ", "rm"), ("σ", "it"), ("(", "rm"), ("W", "bf"), ("pre", "rm_sub"),
                         (" v", "bf"), ("t", "it_sub"), (")", "rm")], size=6.4)
text(ex + 385, gy + 77, "applied at every draw", size=4.4, color=INK2, anchor="middle")
text(ex + 385, gy + 90, "pre-trained on MSK-CHORD", size=4.4, color=INK2, anchor="middle")

# clinical
rect(ex, 318, 100, 26, r=6, fill="#ffffff", stroke=LINE, sw=0.9)
text(ex + 50, 335, "Batch norm", size=5, color=INK2, anchor="middle")

# ═════ representations → Cox layer ══════════════════════════════════════════
ox, ow = 790, 92
outs = [(110, [("h", "bf"), (" ∈ ", "rm"), ("R", "bb"), ("G", "it_sup")]),
        (150, [("r", "bf"), (" ∈ ", "rm"), ("R", "bb"), ("2F", "it_sup")]),
        (266, [("e", "bf"), (" ∈ ", "rm"), ("R", "bb"), ("2T", "it_sup")]),
        (331, [("c", "bf"), (" ∈ ", "rm"), ("R", "bb"), ("C", "it_sup")])]
for yy, parts in outs:
    src = ex + ew if yy < 330 else ex + 100
    arrow([(src + 2, yy), (ox - 2, yy)])
    chip(ox, yy, ow, parts)
TR = 904
yc = 216
for yy, _ in outs:
    add(f'<path d="M{ox + ow},{yy} L{TR},{yy} L{TR},{yc}" fill="none" stroke="{LINE}" stroke-width="1"/>')
arrow([(TR, yc), (926, yc)])

# Cox layer
cx, cwid = 926, 186
rect(cx, yc - 62, cwid, 128, r=8, fill="#ffffff", stroke=PLUM, sw=1.5)
text(cx + cwid / 2, yc - 40, "Cox layer", size=6.2, color=PLUM, weight=700, anchor="middle")
math(cx + cwid / 2, yc - 14, [("Φ", "it"), (" = [ ", "rm"), ("h", "bf"), (", BN(", "rm"), ("r", "bf"), ("), BN(", "rm"),
                             ("e", "bf"), ("), BN(", "rm"), ("c", "bf"), (") ]", "rm")], size=5.6)
math(cx + cwid / 2, yc + 12, [("η", "it"), (" = ", "rm"), ("σ", "it"), ("(", "rm"), ("θ", "bf"), ("⊤", "rm_sup"),
                             ("Φ", "it"), (" + ", "rm"), ("θ", "it"), ("0", "rm_sub"), (")", "rm")], size=6.6, color=PLUM)
text(cx + cwid / 2, yc + 36, "one coefficient per input", size=4.3, color=INK2, anchor="middle")
text(cx + cwid / 2, yc + 49, "dropout 0.4", size=4.3, color=INK2, anchor="middle")
math(cx + cwid / 2, yc + 86, [("λ", "it"), ("(", "rm"), ("t", "it"), (" | ", "rm"), ("Φ", "it"), (") = ", "rm"),
                             ("λ", "it"), ("0", "rm_sub"), ("(", "rm"), ("t", "it"), (")", "rm"), (" e", "it"),
                             ("η", "it_sup")], size=6, color=INK2)
text(cx + cwid / 2, yc + 102, "trained end-to-end on the", size=4.4, color=INK3, anchor="middle")
text(cx + cwid / 2, yc + 114, "Cox partial likelihood", size=4.4, color=INK3, anchor="middle")

# recalibration
arrow([(cx + cwid + 2, yc), (1132, yc)])
rx, rw = 1134, 150
rect(rx, yc - 54, rw, 118, r=8, fill="#f8eef3", stroke="#c9a3b6", sw=1.1)
text(rx + rw / 2, yc - 34, "Out-of-fold", size=5.4, color=PLUM, weight=700, anchor="middle")
text(rx + rw / 2, yc - 21, "recalibration", size=5.4, color=PLUM, weight=700, anchor="middle")
math(rx + rw / 2, yc + 1, [("S", "it"), ("(", "rm"), ("t", "it"), (" | ", "rm"), ("x", "bf"), (") =", "rm")], size=6)
math(rx + rw / 2, yc + 19, [("exp(−", "rm"), ("H", "it"), ("0", "rm_sub"), ("(", "rm"), ("t", "it"), (")", "rm"),
                            (" e", "it"), ("βη", "it_sup"), (")", "rm")], size=6)
math(rx + 22, yc + 38, [("β", "it")], size=5.2, color=INK2, anchor="start")
text(rx + 38, yc + 37, "calibration slope", size=4.3, color=INK2)
math(rx + 20, yc + 52, [("H", "it"), ("0", "rm_sub")], size=5.2, color=INK2, anchor="start")
text(rx + 38, yc + 51, "Breslow", size=4.3, color=INK2)

# survival
arrow([(rx + rw + 2, yc), (1300, yc)])
spx, spy, spw, sph = 1306, yc + 44, 46, 86
add(f'<path d="M{spx},{spy - sph - 4} L{spx},{spy} L{spx + spw + 3},{spy}" fill="none" stroke="{INK2}" stroke-width="0.8"/>')
xs = np.linspace(0, 1, 50)
for (lab, c), r_ in zip(RISK, [0.35, 0.9, 1.9]):
    yv = np.exp(-r_ * xs)
    pts = " ".join(f"{spx + spw * a:.1f},{spy - sph * b:.1f}" for a, b in zip(xs, yv))
    add(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round"/>')
    text(spx + spw + 4, spy - sph * yv[-1] + 4, lab, size=4, color=c, weight=700)
math(spx + spw / 2, spy + 15, [("t", "it")], size=5.6, color=INK2)
text(spx + 36, spy - sph - 12, "individual survival", size=4.4, color=INK2, anchor="middle")

# ═════ take-home callout ═════════════════════════════════════════════════════
rect(22, 368, 1396, 42, r=3, fill=GREY_BG)
add(f'<rect x="22" y="368" width="4" height="42" fill="{CRIMSON}"/>')
add(f'<text xml:space="preserve" x="40" y="394" font-family="{SANS}" font-size="{fs(5.6)}" fill="{INK}">'
    f'Two complementary questions, two encoders, <tspan fill="{CRIMSON}" font-weight="700">one interpretable output</tspan>'
    f': every representation enters the Cox layer with its own coefficient, hence its own hazard ratio.</text>')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>' + "\n".join(el) + "</svg>")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "fig1b_architecture.svg").write_text(svg)
subprocess.run(["rsvg-convert", "-z", "3", "-o", str(OUT / "fig1b_architecture.png"),
                str(OUT / "fig1b_architecture.svg")], check=True)
print("done")
