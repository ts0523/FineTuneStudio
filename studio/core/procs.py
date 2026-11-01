# -*- coding: utf-8 -*-
"""子进程工具：隐藏控制台窗口、UTF-8 环境、按进程树强杀。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

IS_WIN = sys.platform == "win32"
CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008


def base_env(python_exe: Path | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
    # 避免 HF 的进度条与遥测干扰事件流
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TRANSFORMERS_VERBOSITY"] = "warning"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
    if python_exe is not None:
        env["PYTHONHOME"] = ""
        env.pop("PYTHONHOME", None)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def popen(args: list[str], cwd: str | Path | None = None, env: dict[str, str] | None = None,
          stdin_pipe: bool = False) -> subprocess.Popen:
    kwargs: dict = {
        "cwd": str(cwd) if cwd else None,
        "env": env or base_env(),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.PIPE if stdin_pipe else subprocess.DEVNULL,
        "bufsize": -1,
    }
    if IS_WIN:
        kwargs["creationflags"] = CREATE_NO_WINDOW
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = si
    return subprocess.Popen(args, **kwargs)


def kill_tree(pid: int) -> None:
    """强杀整个进程树（Windows 用 taskkill，其它平台用 killpg）。"""
    if pid <= 0:
        return
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=CREATE_NO_WINDOW, timeout=15)
        else:
            killpg = getattr(os, "killpg", None)
            getpgid = getattr(os, "getpgid", None)
            if killpg is not None and getpgid is not None:
                killpg(getpgid(pid), 9)
            else:
                os.kill(pid, 9)
    except Exception:  # noqa: BLE001
        try:
            os.kill(pid, 9)
        except Exception:  # noqa: BLE001
            pass


def run_text(args: list[str], cwd: str | Path | None = None, env: dict[str, str] | None = None,
             timeout: float = 120) -> tuple[int, str]:
    """同步执行并返回 (returncode, 合并输出)。失败不抛异常。"""
    try:
        p = popen(args, cwd=cwd, env=env)
        out, err = p.communicate(timeout=timeout)
        text = (out or b"").decode("utf-8", "replace") + (err or b"").decode("utf-8", "replace")
        return p.returncode, text
    except subprocess.TimeoutExpired:
        try:
            kill_tree(p.pid)
        except Exception:  # noqa: BLE001
            pass
        return -9, "命令执行超时"
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def open_in_explorer(path: str | Path) -> bool:
    p = Path(path)
    try:
        if p.is_file():
            os.startfile(str(p.parent))  # type: ignore[attr-defined]
            return True
        if p.is_dir():
            os.startfile(str(p))  # type: ignore[attr-defined]
            return True
    except Exception:  # noqa: BLE001
        return False
    return False
