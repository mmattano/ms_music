"""Generate the app icons in ms_music/gui/assets (run from the repo root).

    python installer/make_icons.py

Draws a waveform glyph on the GUI's indigo, saved as PNG (favicon and
Linux), ICO (Windows shortcuts) and ICNS (macOS app).
"""

import os

import numpy as np
from PIL import Image, ImageDraw

OUT = os.path.join(
    os.path.dirname(__file__), "..", "ms_music", "gui", "assets"
)
SIZE = 1024
INDIGO = (79, 70, 229, 255)
WHITE = (255, 255, 255, 255)


def draw(size=SIZE):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = size * 0.06
    d.rounded_rectangle(
        [pad, pad, size - pad, size - pad], radius=size * 0.22, fill=INDIGO
    )
    # Waveform bars: a spectrum-like envelope.
    n = 9
    x0, x1 = size * 0.2, size * 0.8
    heights = 0.12 + 0.55 * np.exp(-(((np.arange(n) - 4) / 2.6) ** 2))
    heights *= np.array([0.6, 0.9, 0.7, 1.0, 0.85, 1.0, 0.7, 0.9, 0.6])
    width = (x1 - x0) / n * 0.55
    for i, h in enumerate(heights):
        cx = x0 + (i + 0.5) * (x1 - x0) / n
        half = h * size * 0.5
        d.rounded_rectangle(
            [cx - width / 2, size / 2 - half, cx + width / 2, size / 2 + half],
            radius=width / 2,
            fill=WHITE,
        )
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    img = draw()
    img.resize((256, 256), Image.LANCZOS).save(os.path.join(OUT, "icon.png"))
    img.save(
        os.path.join(OUT, "icon.ico"),
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    img.save(os.path.join(OUT, "icon.icns"))
    print("Wrote icons to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
