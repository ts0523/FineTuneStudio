# -*- coding: utf-8 -*-
"""Python 运行时探测：找到训练用的解释器，检查依赖是否齐备。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import paths, procs

REQUIRED_MODULES = ["torch", "transformers", "peft", "datasets", "accelerate", "requests", "psutil"]

_PROBE_CODE = r"""
import json, os, platform, sys
info = {"python": sys.version.split()[0], "exe": sys.executable,
        "platform": platform.platform(), "arch": platform.machine(),
        "cpu_count": os.cpu_count() or 0}
def safe(fn, key):
    try:
        info[key] = fn()
    except Exception as e:
        info[key] = "ERR:" + type(e).__name__ + ":" + str(e)[:120]
def _torch_ver(): import torch; return torch.__version__
def _cuda(): import torch; return bool(torch.cuda.is_available())
def _threads(): import torch; return int(torch.get_num_threads())
safe(_torch_ver, "torch"); safe(_cuda, "cuda"); safe(_threads, "torch_threads")
def _libs():
    out = {}
    for m in ("transformers", "peft", "datasets", "accelerate", "requests", "psutil"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = ""
    return out
safe(_libs, "libs")
def _mem():
    import psutil
    vm = psutil.virtual_memory()
    return {"total": int(vm.total), "available": int(vm.available)}
safe(_mem, "mem")
def _disk():
    import psutil, os
    root = os.environ.get("FTS_PROBE_ROOT", ".")
    du = psutil.disk_usage(root)
    return {"total": int(du.total), "free": int(du.free)}
safe(_disk, "disk")
print(json.dumps(info, ensure_ascii=False))
"""


@dataclass
class Env:
    python: Path | None = None
    source: str = "未找到"
    version: str = ""
    data: dict = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    ok: bool = False
    error: str = ""

    @property
    def torch(self) -> str:
        return str(self.data.get("torch", "") or "")

    @property
    def cuda(self) -> bool:
        return bool(self.data.get("cuda", False))

    def lib_versions(self) -> dict[str, str]:
        v = self.data.get("libs") or {}
        return {k: str(x) for k, x in v.items()} if isinstance(v, dict) else {}

    def summary(self) -> str:
        if self.python is None:
            return "未找到训练环境"
        if self.missing:
            return f"缺少依赖: {', '.join(self.missing)}"
        return f"Python {self.version} · torch {self.torch}" if self.ok else (self.error or "环境异常")


def _venv_candidates() -> list[Path]:
    r = paths.root()
    return [
        r / ".venv" / "Scripts" / "python.exe",
        r / "venv" / "Scripts" / "python.exe",
        r / "env" / "Scripts" / "python.exe",
        r / "python" / "python.exe",
        r / ".venv" / "bin" / "python",
    ]


def find_python(cfg) -> tuple[Path | None, str]:
    custom = str(cfg.get("python", "") or "").strip()
    if custom:
        p = Path(custom).expanduser()
        if p.is_file():
            return p, "配置指定"
        return None, f"配置里的解释器不存在: {p}"
    for c in _venv_candidates():
        if c.is_file():
            return c, "项目虚拟环境"
    return None, "未找到 .venv（可在设置页一键初始化）"


def probe(cfg) -> Env:
    env = Env()
    py, source = find_python(cfg)
    env.python = py
    env.source = source
    if py is None:
        env.error = source
        return env
    e = procs.base_env(py, {"FTS_PROBE_ROOT": str(paths.root())})
    code, out = procs.run_text([str(py), "-c", _PROBE_CODE], cwd=paths.root(), env=e, timeout=120)
    if code != 0:
        env.error = out.strip().splitlines()[-1] if out.strip() else f"探测失败(exit {code})"
        return env
    line = ""
    for ln in reversed(out.strip().splitlines()):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            line = ln
            break
    if not line:
        env.error = "探测输出不是 JSON"
        return env
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        env.error = "探测输出解析失败"
        return env
    env.data = data
    env.version = str(data.get("python", ""))
    libs = env.lib_versions()
    torch_ver = data.get("torch", "")
    if isinstance(torch_ver, str) and torch_ver and not torch_ver.startswith("ERR:"):
        libs["torch"] = torch_ver
    env.data["libs"] = libs
    env.missing = [m for m in REQUIRED_MODULES if not libs.get(m)]
    env.ok = not env.missing
    return env


def venv_python_for_create(cfg) -> list[str] | None:
    """给"一键初始化环境"找出可用于创建 venv 的基础解释器。"""
    import shutil as _sh
    py, _ = find_python(cfg)
    if py is not None:
        return [str(py)]
    for name in ("py", "python", "python3"):
        p = _sh.which(name)
        if p:
            return [p]
    return None


def default_threads() -> int:
    n = os.cpu_count() or 4
    return max(1, n // 2)
