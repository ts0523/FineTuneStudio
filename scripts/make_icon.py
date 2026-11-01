# -*- coding: utf-8 -*-
"""生成应用图标：绿底 + 白色设置转轮 + 两个白色小眼睛。"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "studio" / "assets"
SIZE = 1024
GREEN = "#159447"
WHITE = "#FFFFFF"


def gear_polygon(cx: float, cy: float, outer: float, inner: float,
                 teeth: int = 12) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    steps = teeth * 4
    for i in range(steps):
        angle = -math.pi / 2 + 2 * math.pi * i / steps
        phase = i % 4
        radius = outer if phase in (1, 2) else inner
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    scale = 4
    size = SIZE * scale
    cx = cy = size / 2
    im = Image.new("RGBA", (size, size), GREEN)
    draw = ImageDraw.Draw(im)

    # 白色齿轮直接压在绿底上，齿轮轮廓和齿牙必须保留可见。
    draw.polygon(gear_polygon(cx, cy, 448 * scale, 345 * scale), fill=WHITE)
    draw.ellipse((cx - 235 * scale, cy - 235 * scale, cx + 235 * scale, cy + 235 * scale),
                 fill=GREEN)
    # 转轮中心就是脸：两个白色小圆点作为眼睛。
    eye_y = cy - 42 * scale
    eye_r = 34 * scale
    for dx in (-78, 78):
        draw.ellipse((cx + dx * scale - eye_r, eye_y - eye_r,
                      cx + dx * scale + eye_r, eye_y + eye_r), fill=WHITE)

    im = im.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    png = ASSETS / "app.png"
    im.save(png, "PNG")
    ico = ASSETS / "app.ico"
    im.save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                        (128, 128), (256, 256)])
    print(f"icon: {ico} ({ico.stat().st_size} bytes), png: {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
