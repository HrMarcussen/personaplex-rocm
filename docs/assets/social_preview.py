# SPDX-FileCopyrightText: Copyright (c) 2026 HrMarcussen
# SPDX-License-Identifier: MIT
"""Generates docs/assets/social-preview.svg (1280x640 card for GitHub's social preview).

Convert to PNG with any SVG renderer before uploading it under
Settings -> General -> Social preview.
"""
from pathlib import Path

W, H = 1280, 640
BG, CARD, TEXT1, TEXT2, MUTED, ACCENT, BAR_DIM = "#0f1115", "#171a20", "#f0efec", "#c3c2b7", "#8f8e88", "#3987e5", "#3a3f4a"
FONT = "Helvetica, Arial, sans-serif"
ROWS = [("Upstream, eager", 211.5, BAR_DIM), ("This port (bf16)", 32.5, ACCENT), ("This port (int8)", 23.8, ACCENT)]


def render() -> str:
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="{BG}"/>',
         f'<rect x="40" y="40" width="{W - 80}" height="{H - 80}" rx="28" fill="{CARD}"/>',
         f'<text x="96" y="138" font-size="30" fill="{MUTED}">PERSONAPLEX-7B · AMD RADEON · WINDOWS 11</text>',
         f'<text x="96" y="232" font-size="68" font-weight="bold" fill="{TEXT1}">Real-time voice AI</text>',
         f'<text x="96" y="310" font-size="68" font-weight="bold" fill="{TEXT1}">you can interrupt.</text>',
         f'<text x="96" y="372" font-size="30" fill="{TEXT2}">Full-duplex speech-to-speech on an RX 7900 XTX.</text>',
         f'<text x="96" y="414" font-size="30" fill="{TEXT2}">No CUDA, no WSL: ROCm + custom HIP kernels.</text>']
    # mini benchmark: ms per 80 ms frame
    x0, y0, maxw = 96, 458, 560
    o.append(f'<text x="{x0}" y="{y0}" font-size="22" fill="{MUTED}">ms per 80 ms of speech (lower is better)</text>')
    for i, (label, ms, color) in enumerate(ROWS):
        y = y0 + 22 + i * 34
        w = max(6, maxw * ms / 220)
        o.append(f'<rect x="{x0 + 250}" y="{y}" width="{w:.0f}" height="22" rx="4" fill="{color}"/>')
        o.append(f'<text x="{x0}" y="{y + 18}" font-size="22" fill="{TEXT2}">{label}</text>')
        o.append(f'<text x="{x0 + 250 + w + 12:.0f}" y="{y + 18}" font-size="22" font-weight="bold" fill="{TEXT1}">{ms:g} ms</text>')
    # headline number
    o.append(f'<text x="{W - 96}" y="300" font-size="150" font-weight="bold" text-anchor="end" fill="{ACCENT}">2.5×</text>')
    o.append(f'<text x="{W - 96}" y="352" font-size="30" text-anchor="end" fill="{TEXT2}">faster than real time</text>')
    o.append(f'<text x="{W - 96}" y="560" font-size="26" text-anchor="end" fill="{MUTED}">github.com/HrMarcussen/personaplex-rocm</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    out = Path(__file__).with_name("social-preview.svg")
    out.write_text(render(), encoding="utf-8")
    print("wrote", out)
