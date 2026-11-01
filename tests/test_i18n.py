"""Bilingual UI catalog and runtime translation rules."""
from __future__ import annotations

import json
from pathlib import Path

from studio.ui import i18n

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_has_english_ui_strings():
    data = json.loads((ROOT / "studio" / "assets" / "i18n_en.json").read_text(encoding="utf-8"))
    assert len(data) > 500
    assert data["设置"] == "Settings"
    assert data["开始帮训"] == "Start AutoTrain"
    assert data["界面语言 / UI language"] == "Interface language / UI language"


def test_tr_uses_catalog_and_keeps_paths():
    old = i18n.language()
    try:
        i18n.set_language("en")
        assert i18n.tr("设置") == "Settings"
        assert i18n.tr("第 2 轮") == "Round 2"
        assert i18n.tr("AI 在线 · 成功 3 次") == "AI online · 3 successes"
        assert i18n.tr("D:\\models\\minimind") == "D:\\models\\minimind"
    finally:
        i18n.set_language(old)


def test_chinese_mode_is_passthrough():
    old = i18n.language()
    try:
        i18n.set_language("zh")
        assert i18n.tr("设置") == "设置"
    finally:
        i18n.set_language(old)


def test_catalog_contains_no_local_absolute_paths():
    raw = (ROOT / "studio" / "assets" / "i18n_en.json").read_text(encoding="utf-8")
    assert "D:\\\\AI" not in raw
    assert "C:\\\\Users" not in raw
    assert "sk-or-v1" not in raw
