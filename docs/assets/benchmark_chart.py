# SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
# SPDX-License-Identifier: MIT
"""Generates docs/assets/benchmark-{light,dark}.svg (the README benchmark chart).

    python docs/assets/benchmark_chart.py
"""
from pathlib import Path

# (label, ms per 80 ms frame, footnote marker). Rows 1-2 were measured with the
# 7B architecture and random weights (identical compute); rows 3-5 on the real
# PersonaPlex weights with rocm/validate_offline.py.
ROWS = [
    ("Upstream code, eager PyTorch", 211.5),
    ("+ HIP graphs", 87.9),
    ("+ GEMM tuning, codec fixes", 57.7),
    ("+ custom HIP kernels (bf16)", 32.5),
    ("+ int8 weights (optional)", 23.8),
]
BUDGET = 80.0

THEMES = {
    "light": dict(surface="#fcfcfb", border="#e4e3df", text1="#0b0b0b", text2="#52514e",
                  muted="#8a8984", grid="#ecebe7", bar="#2a78d6", budget="#52514e"),
    "dark": dict(surface="#1a1a19", border="#2e2e2c", text1="#f0efec", text2="#c3c2b7",
                 muted="#8f8e88", grid="#2b2b29", bar="#3987e5", budget="#c3c2b7"),
}

W, H = 880, 384
LEFT, RIGHT = 248, 200          # label column / value column
TOP = 116
ROW_H, BAR_H = 44, 22
XMAX = 220.0
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def x_of(ms: float) -> float:
    return LEFT + (W - LEFT - RIGHT) * ms / XMAX


def bar_path(x0: float, y: float, x1: float, h: float, r: float = 4.0) -> str:
    """Square at the baseline (left), 4px rounded data end (right)."""
    r = min(r, (x1 - x0) / 2, h / 2)
    return (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z")


def render(t: dict) -> str:
    o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             f'role="img" aria-labelledby="title desc" font-family="{FONT}">')
    o.append('<title id="title">PersonaPlex on RX 7900 XTX: time per 80 ms audio frame</title>')
    desc = "; ".join(f"{lbl}: {ms:.1f} ms ({BUDGET / ms:.2f}x real time)" for lbl, ms in ROWS)
    o.append(f'<desc id="desc">Lower is better; real time requires under {BUDGET:.0f} ms. {desc}.</desc>')
    o.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="{t["surface"]}" stroke="{t["border"]}"/>')
    o.append(f'<text x="28" y="40" font-size="19" font-weight="600" fill="{t["text1"]}">'
             'Time to generate one 80 ms audio frame</text>')
    o.append(f'<text x="28" y="64" font-size="13.5" fill="{t["text2"]}">'
             'AMD Radeon RX 7900 XTX, native Windows 11 · lower is better · real time needs &lt; 80 ms</text>')

    plot_top, plot_bot = TOP - 6, TOP + ROW_H * len(ROWS) - (ROW_H - BAR_H) + 6
    # recessive grid + axis ticks
    for v in (0, 50, 100, 150, 200):
        x = x_of(v)
        o.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bot}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{plot_bot + 18}" font-size="11.5" text-anchor="middle" fill="{t["muted"]}">{v}</text>')
    o.append(f'<text x="{x_of(XMAX):.1f}" y="{plot_bot + 18}" font-size="11.5" text-anchor="end" fill="{t["muted"]}">ms</text>')

    labels = []
    halo = f'stroke="{t["surface"]}" stroke-width="5" stroke-linejoin="round"'
    for i, (label, ms) in enumerate(ROWS):
        y = TOP + i * ROW_H
        cy = y + BAR_H / 2
        weight = "600" if i >= 3 else "400"
        o.append(f'<text x="{LEFT - 14}" y="{cy + 4.5:.1f}" font-size="13.5" text-anchor="end" '
                 f'font-weight="{weight}" fill="{t["text1"]}">{label}</text>')
        o.append(f'<path d="{bar_path(x_of(0), y, x_of(ms), BAR_H)}" fill="{t["bar"]}"/>')
        rt = BUDGET / ms
        content = (f'<tspan font-weight="600">{ms:.1f} ms</tspan>'
                   f'<tspan> · {rt:.2f}× real time</tspan>')
        pos = f'x="{x_of(ms) + 10:.1f}" y="{cy + 4.5:.1f}" font-size="13.5"'
        # halo copy underneath (portable; no paint-order needed), then the label itself
        labels.append(f'<text {pos} fill="{t["surface"]}" {halo}>{content}</text>')
        labels.append(f'<text {pos} fill="{t["text1"]}"><tspan font-weight="600">{ms:.1f} ms</tspan>'
                      f'<tspan fill="{t["text2"]}"> · {rt:.2f}× real time</tspan></text>')

    # real-time budget reference line: over the bars, under the value labels
    bx = x_of(BUDGET)
    o.append(f'<line x1="{bx:.1f}" y1="{plot_top - 10}" x2="{bx:.1f}" y2="{plot_bot}" stroke="{t["budget"]}" '
             f'stroke-width="1.5" stroke-dasharray="5 4"/>')
    o.append(f'<text x="{bx}" y="{plot_top - 16}" font-size="12" text-anchor="middle" fill="{t["text2"]}">'
             'real-time budget: 80 ms</text>')
    o.extend(labels)

    o.append(f'<text x="28" y="{H - 18}" font-size="11.5" fill="{t["muted"]}">'
             'First two rows: 7B architecture with random weights (identical compute). '
             'Others: real PersonaPlex weights, rocm/validate_offline.py.</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    here = Path(__file__).parent
    for name, theme in THEMES.items():
        (here / f"benchmark-{name}.svg").write_text(render(theme), encoding="utf-8")
        print("wrote", here / f"benchmark-{name}.svg")
