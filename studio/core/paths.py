# -*- coding: utf-8 -*-
"""路径解析。

定位顺序（项目根目录）：
1. 环境变量 FTS_ROOT
2. exe 所在目录向上最多 4 层，找含 .venv\\Scripts\\python.exe 或 scripts\\train_lora.py 的目录
3. 兜底：exe 所在目录
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_ID = "FineTuneStudio"
APP_NAME = "FineTune Studio"
VERSION = "2.1.0"

_cached_root: Path | None = None
_override_root: Path | None = None


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """PyInstaller 解包目录；源码运行时为项目根目录。"""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base)
    return Path(__file__).resolve().parents[2]


def source_dir() -> Path:
    """studio 包所在目录。"""
    return Path(__file__).resolve().parents[1]


def exe_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def looks_like_root(p: Path) -> bool:
    try:
        return ((p / ".venv" / "Scripts" / "python.exe").is_file()
                or (p / "scripts" / "train_lora.py").is_file()
                or (p / "requirements.txt").is_file())
    except OSError:
        return False


def project_root() -> Path:
    env = os.environ.get("FTS_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p.resolve()
    start = exe_dir()
    for p in [start, *start.parents][:5]:
        if looks_like_root(p):
            return p
    return start


def root() -> Path:
    global _cached_root
    if _override_root is not None:
        return _override_root
    if _cached_root is None:
        _cached_root = project_root()
    return _cached_root


def set_root(p: str | Path) -> Path:
    """用户手动指定项目根目录（写进配置，下次启动仍生效）。"""
    global _cached_root, _override_root
    path = Path(p).expanduser().resolve()
    _override_root = path
    _cached_root = path
    return path


def asset_file(name: str) -> Path:
    """应用资源：源码在 studio/assets，打包后在解包目录/assets。"""
    packaged = bundle_dir() / "assets" / name
    if packaged.is_file():
        return packaged
    return root() / "studio" / "assets" / name


def models_dir() -> Path:
    return root() / "models"


def data_dir() -> Path:
    return root() / "data"


def outputs_dir() -> Path:
    return root() / "outputs"


def logs_dir() -> Path:
    return root() / "logs"


def config_path() -> Path:
    return root() / "studio_config.json"


def _localappdata() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / APP_ID


def state_dir() -> Path:
    """索引/日志/worker 副本所在目录。项目根不可写时退到 LOCALAPPDATA。"""
    d = root() / "studio_state"
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".writetest"
        probe.write_text("1", encoding="utf-8")
        probe.unlink()
        return d
    except OSError:
        d = _localappdata() / "state"
        d.mkdir(parents=True, exist_ok=True)
        return d


def index_path() -> Path:
    return state_dir() / "index.json"


def workers_dir() -> Path:
    """worker 脚本目录。打包后首次运行会把脚本解包到状态目录。"""
    if is_frozen():
        src = bundle_dir() / "workers"
        if not src.is_dir():
            raise FileNotFoundError(f"安装包内缺少 workers 目录: {src}")
        target = state_dir() / "workers" / VERSION
        target.mkdir(parents=True, exist_ok=True)
        # 版本内文件可能随升级变化：逐个比对内容，不一致就覆盖
        for f in src.glob("*.py"):
            dst = target / f.name
            try:
                if dst.is_file() and dst.read_bytes() == f.read_bytes():
                    continue
            except OSError:
                pass
            shutil.copyfile(f, dst)
        return target
    return source_dir() / "workers"


def ensure_dirs() -> None:
    for d in (models_dir(), data_dir(), outputs_dir(), logs_dir(), state_dir()):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def human_size(n: float | int | None) -> str:
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{n:.0f} {unit}"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def human_time(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "-"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"
