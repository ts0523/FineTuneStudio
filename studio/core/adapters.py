# -*- coding: utf-8 -*-
"""内置能力适配器与训练适配器发现。

内置能力不是需要下载的模型：文本主干、视觉编码器、语音编码器、本机 TTS
都已经在应用里。这里把它们描述成可多选能力，供实验台/对话页统一展示。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import models as models_mod, paths


@dataclass(frozen=True)
class Capability:
    key: str
    title: str
    direction: str
    description: str
    default: bool = True


BUILTIN_CAPABILITIES: tuple[Capability, ...] = (
    Capability("text", "语言理解", "输入", "文本主干，理解中文问题与上下文"),
    Capability("image", "图像理解", "输入", "MobileNetV3-Small 视觉编码 + 投影层"),
    Capability("audio", "语音理解", "输入", "Whisper-tiny 语音编码 + 投影层"),
    Capability("speech", "语音输出", "输出", "系统 TTS 把回答读出来"),
)

DEFAULT_CAPABILITIES = tuple(c.key for c in BUILTIN_CAPABILITIES if c.default)


def capability(key: str) -> Capability | None:
    return next((c for c in BUILTIN_CAPABILITIES if c.key == key), None)


def normalize_capabilities(keys) -> list[str]:
    valid = {c.key for c in BUILTIN_CAPABILITIES}
    return [k for k in keys if k in valid]


def input_modalities(keys) -> list[str]:
    selected = set(normalize_capabilities(keys))
    return [k for k in ("text", "image", "audio") if k in selected]


def supports_speech(keys) -> bool:
    return "speech" in set(normalize_capabilities(keys))


def adapter_modalities(path: str | Path) -> list[str]:
    """读取训练产物的多模态类型；没有投影层返回空列表。"""
    p = Path(path)
    if not (p / "mm_projector.pt").is_file():
        return []
    meta: dict = {}
    try:
        meta = json.loads((p / "train_meta.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        pass
    mm = meta.get("multimodal") if isinstance(meta, dict) else {}
    stats = mm.get("stats") if isinstance(mm, dict) else {}
    out = []
    if not isinstance(stats, dict):
        return ["image", "audio"]
    if int(stats.get("image") or 0) > 0:
        out.append("image")
    if int(stats.get("audio") or 0) > 0:
        out.append("audio")
    return out or ["image", "audio"]


def adapter_choices() -> list[dict]:
    out = []
    for item in models_mod.scan_outputs():
        if item.kind != "adapter":
            continue
        mods = adapter_modalities(item.path)
        out.append({"name": item.name, "path": str(item.path),
                    "modalities": mods, "has_projector": bool(mods)})
    return out


def asset_path(name: str) -> Path:
    """运行期资源路径；源码 / PyInstaller 单文件都能找到。"""
    bundled = paths.bundle_dir() / "assets" / name
    if bundled.is_file():
        return bundled
    return paths.root() / "studio" / "assets" / name
