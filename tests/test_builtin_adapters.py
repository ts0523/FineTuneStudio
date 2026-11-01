"""内置能力与图标资源回归。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from studio.core import adapters

ROOT = Path(__file__).resolve().parents[1]


def test_builtin_adapters_cover_input_and_output():
    keys = [c.key for c in adapters.BUILTIN_CAPABILITIES]
    assert keys == ["text", "image", "audio", "speech"]
    assert adapters.input_modalities(keys) == ["text", "image", "audio"]
    assert adapters.supports_speech(keys)


def test_icon_is_green_with_white_gear():
    png = ROOT / "studio" / "assets" / "app.png"
    ico = ROOT / "studio" / "assets" / "app.ico"
    assert png.is_file() and ico.is_file()
    im = Image.open(png).convert("RGB")
    r, g, b = im.getpixel((0, 0))
    assert g > r and g > b, "图标应为绿色背景"
    white = sum(1 for px in im.resize((128, 128)).getdata()
                if min(px) > 230)
    green = sum(1 for px in im.resize((128, 128)).getdata()
                if px[1] > px[0] and px[1] > px[2])
    assert white > 1000 and green > 3000
