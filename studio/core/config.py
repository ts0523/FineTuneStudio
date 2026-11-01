# -*- coding: utf-8 -*-
"""配置读写（studio_config.json，原子写入）。"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from . import paths

DEFAULTS: dict[str, Any] = {
    "endpoint": "https://hf-mirror.com",
    "python": "",                 # 空 = 自动探测 .venv
    "root": "",                   # 空 = 自动探测
    "threads": 0,                 # 0 = 自动（物理核数一半）
    "theme": "forest",            # 默认「白色森林」
    "theme_custom": {},        # 用户自由调节的颜色（bg/panel/panel2/border/text/text2/accent）
    "recycle_delete": True,       # 删除模型/实验走回收站
    "log_lines": 3000,            # 日志面板最多保留行数
    "auto_check_env": True,
    # AI 能力（OpenRouter）：实验台的云端模型、数据层造数、训练层帮训都用它
    "ai": {
        "key": "",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "",
        "fallbacks": [
            "dots-studio/dots-3-note-preview:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "nvidia/nemotron-3.5-super-vl:free",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        ],
        "catalog": {},
        "blocked": {},
        "stats": {},
        "temperature": 0.6,
        "max_tokens": 1200,
        "timeout": 60,
        "voice": "",
        "voice_profile": "",
        "rate": 0,
        "asr": "auto",
        "whisper_model": "whisper-tiny",
        "referer": "",
        "title": "FineTune Studio",
        "web_sources": True,
    },
    "autotrain": {
        "goal": "",
        "rounds": 3,
        "per_round": 40,
        "eval_n": 6,
        "use_web": True,
    },
    "chat": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 128,
        "repetition_penalty": 1.1,
    },
    "train": {
        "epochs": 3.0,
        "batch_size": 2,
        "grad_accum": 2,
        "lr": 0.0002,
        "max_len": 256,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "val_ratio": 0.1,
        "seed": 42,
        "full": False,
    },
    "window": {"w": 1400, "h": 880},
    "last": {"model": "", "dataset": ""},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[str, Any] = json.loads(json.dumps(DEFAULTS))
        self._path = paths.config_path()
        self._load_error = ""
        self.load()

    # ---------------------------------------------------------------- 读
    @property
    def path(self) -> Path:
        return self._path

    @property
    def load_error(self) -> str:
        return self._load_error

    def load(self) -> None:
        p = paths.config_path()
        self._path = p
        if not p.is_file():
            return
        try:
            raw = json.loads(p.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict):
                raise ValueError("配置文件根节点必须是 JSON 对象")
            with self._lock:
                self._data = _merge(DEFAULTS, raw)
            self._load_error = ""
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"

    # ---------------------------------------------------------------- 取/存
    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any, save: bool = True) -> None:
        with self._lock:
            parts = dotted.split(".")
            cur = self._data
            for part in parts[:-1]:
                nxt = cur.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    cur[part] = nxt
                cur = nxt
            cur[parts[-1]] = value
            if save:
                self.save()

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))

    def save(self) -> str:
        """原子写；返回空串表示成功，否则返回错误描述。

        整个「取快照 → 写临时文件 → 替换」都在同一把锁里完成，避免两个线程
        各自快照后互相覆盖（丢更新）。
        """
        with self._lock:
            data = json.dumps(self._data, ensure_ascii=False, indent=2)
            target = self._path
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(prefix=".cfg-", suffix=".tmp", dir=str(target.parent))
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, target)
                return ""
            except OSError as exc:
                return f"配置写入失败：{exc}"


_CFG: Config | None = None


def cfg() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = Config()
    return _CFG
