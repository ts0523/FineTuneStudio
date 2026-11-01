# -*- coding: utf-8 -*-
"""Runtime UI translation.

Chinese remains the source language.  The English catalog is intentionally kept
in Python so source runs and PyInstaller builds behave identically.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import ttk

LANG = "zh"
_INSTALLED = False
_UNTRANSLATED: set[str] = set()
CAPTURE = os.environ.get("FTS_I18N_CAPTURE") == "1"
_HAN = re.compile(r"[\u4e00-\u9fff]")

# Full runtime strings are translated at widget boundary, so concatenated
# messages are covered without rewriting every page.
EN: dict[str, str] = {}


def _load_catalog() -> None:
    global EN
    if EN:
        return
    direct = Path(__file__).resolve().parents[1] / "assets" / "i18n_en.json"
    try:
        from ..core import paths
        p = paths.asset_file("i18n_en.json")
        if not p.is_file():
            p = direct
    except Exception:  # noqa: BLE001
        p = direct
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            EN = {str(k): str(v) for k, v in obj.items()}
    except (OSError, ValueError):
        EN = {}


def set_language(lang: str) -> None:
    global LANG
    LANG = "en" if str(lang).lower().startswith("en") else "zh"
    if LANG == "en":
        _load_catalog()


def language() -> str:
    return LANG


def _record(text: str) -> None:
    if CAPTURE and _HAN.search(text):
        _UNTRANSLATED.add(text)


_PATTERNS = [
    (re.compile(r"^torch (.+) · 项目虚拟环境$"), r"torch \1 · project virtual environment"),
    (re.compile(r"^AI 在线 · 成功 (\d+) 次$"), r"AI online · \1 successes"),
    (re.compile(r"^AI 在线 · (\d+) 次$"), r"AI online · \1 requests"),
    (re.compile(r"^(\d+) 线程 · 内存 ([\d.]+) GB\n磁盘剩余 ([\d.]+) GB$"),
     r"\1 threads · RAM \2 GB\nDisk free \3 GB"),
    (re.compile(r"^共 ([\d.]+) GB · 内存可用 ([\d.]+) GB$"),
     r"Total \1 GB · RAM available \2 GB"),
    (re.compile(r"^CPU ([\d.]+)% · 内存 ([\d.]+)/([\d.]+) GB · 磁盘剩余 ([\d.]+) GB$"),
     r"CPU \1% · RAM \2/\3 GB · Disk free \4 GB"),
    (re.compile(r"^后台回调出错："), "Background callback error: "),
    (re.compile(r"^输出目录："), "Output directory: "),
    (re.compile(r"第 (\d+) 轮"), r"Round \1"),
    (re.compile(r"(\d+) 条"), r"\1 examples"),
    (re.compile(r"(\d+) 步"), r"\1 steps"),
    (re.compile(r"(\d+) 分"), r"\1 pts"),
    (re.compile(r"(\d+) 次"), r"\1 times"),
    (re.compile(r"(\d+) 个"), r"\1 items"),
    (re.compile(r"(\d+) 题"), r"\1 questions"),
    (re.compile(r"(\d+) 行"), r"\1 lines"),
    (re.compile(r"(\d+) 秒"), r"\1 s"),
    (re.compile(r"(\d+) 分钟"), r"\1 min"),
    (re.compile(r"(\d+) 小时"), r"\1 hours"),
    (re.compile(r"(\d+) 张"), r"\1 images"),
    (re.compile(r"(\d+) 层"), r"\1 layers"),
    (re.compile(r"(\d+) 字"), r"\1 chars"),
]

_TERMS = [
    ("全部由数据量、模型规模与本机实测速度推算",
     "Estimated from data size, model size, and measured local speed"),
    ("outputs/ 下的适配器与合并模型", "adapters and merged models under outputs/"),
    ("A：需要 SenseVoice/SigLIP/Mimi 三个子编码器和 GPU，本机只训练文本主干；模型文件里保留了这些结构。",
     "A: It needs SenseVoice/SigLIP/Mimi encoders and a GPU. This machine trains only the text backbone; those structures remain in the model file."),
    ("目标：", "Goal: "), ("基座：", "Base: "), ("数据：", "Data: "),
    ("方式：", "Method: "), ("超参：", "Hyperparameters: "), ("样本：", "Samples: "),
    ("步数：", "Steps: "), ("最终 loss：", "Final loss: "),
    ("验证 loss：", "Validation loss: "), ("耗时：", "Duration: "),
    ("吞吐 ", "Throughput "), ("产物：", "Artifacts: "),
    ("已停止", "stopped"), ("已完成", "completed"), ("正在", "in progress: "),
    ("成功", "succeeded"), ("失败", "failed"), ("取消", "cancelled"),
    ("训练完成", "training complete"), ("训练", "training"), ("验证", "validation"),
    ("模型", "model"), ("数据", "data"), ("加载", "load"), ("保存", "save"),
    ("下载", "download"), ("合并", "merge"), ("导出", "export"), ("设置", "settings"),
    ("回合", "round"), ("评分", "score"), ("检查点", "checkpoint"), ("显存", "VRAM"),
    ("内存", "memory"), ("磁盘", "disk"), ("线程", "threads"), ("文件", "file"),
    ("目录", "directory"), ("条", "items"), ("个", "items"), ("秒", "s"), ("轮", "rounds"),
]


def tr(text):
    if not isinstance(text, str) or LANG != "en" or not _HAN.search(text):
        return text
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith(("http://", "https://")):
        return text
    hit = EN.get(text)
    if hit is not None:
        return hit
    out = text
    for pattern, repl in _PATTERNS:
        out = pattern.sub(repl, out)
    if out == text:
        for zh, en in _TERMS:
            out = out.replace(zh, en)
    if out != text:
        return out
    _record(text)
    return text


def _wrap_text_option(fn):
    def wrapped(self, *args, **kwargs):
        if "text" in kwargs:
            kwargs["text"] = tr(kwargs["text"])
        return fn(self, *args, **kwargs)
    return wrapped


def _patch_widget(cls) -> None:
    if getattr(cls, "_fts_i18n_patched", False):
        return
    cls.__init__ = _wrap_text_option(cls.__init__)  # type: ignore[method-assign]
    cls.configure = _wrap_text_option(cls.configure)  # type: ignore[method-assign]
    cls.config = cls.configure  # type: ignore[attr-defined]
    cls._fts_i18n_patched = True  # type: ignore[attr-defined]


def _patch_canvas() -> None:
    if getattr(tk.Canvas, "_fts_i18n_canvas", False):
        return
    old_text = tk.Canvas.create_text
    old_item = tk.Canvas.itemconfigure

    def create_text(self, *args, **kwargs):
        if "text" in kwargs:
            kwargs["text"] = tr(kwargs["text"])
        return old_text(self, *args, **kwargs)

    def itemconfigure(self, tag, *args, **kwargs):
        if "text" in kwargs:
            kwargs["text"] = tr(kwargs["text"])
        return old_item(self, tag, *args, **kwargs)

    tk.Canvas.create_text = create_text  # type: ignore[method-assign]
    tk.Canvas.itemconfigure = itemconfigure  # type: ignore[method-assign]
    tk.Canvas._fts_i18n_canvas = True  # type: ignore[attr-defined]


def _patch_wm() -> None:
    if getattr(tk.Wm, "_fts_i18n_title", False):
        return
    old = tk.Wm.title

    def title(self, string=None):
        return old(self, tr(string) if string is not None else string)

    tk.Wm.title = title  # type: ignore[method-assign]
    tk.Wm._fts_i18n_title = True  # type: ignore[attr-defined]


def _dump() -> None:
    if not CAPTURE or not _UNTRANSLATED:
        return
    try:
        root = Path(os.environ.get("FTS_ROOT") or Path.cwd())
        p = root / "studio_state" / "i18n-untranslated.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        old: list[str] = []
        if p.is_file():
            old = json.loads(p.read_text(encoding="utf-8"))
        p.write_text(json.dumps(sorted(set(old) | _UNTRANSLATED), ensure_ascii=False,
                                indent=1), encoding="utf-8")
    except OSError:
        pass


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    for cls in (tk.Label, tk.Button, tk.Checkbutton, tk.Radiobutton,
                ttk.Label, ttk.Button, ttk.Checkbutton, ttk.Radiobutton):
        _patch_widget(cls)
    _patch_canvas()
    _patch_wm()
    atexit.register(_dump)


def untranslated() -> list[str]:
    return sorted(_UNTRANSLATED)
