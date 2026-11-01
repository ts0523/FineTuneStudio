# -*- coding: utf-8 -*-
"""硬件实时信息。优先用 psutil（打包时一并带上），不可用时退回 ctypes。"""
from __future__ import annotations

import ctypes
import os
import shutil
from dataclasses import dataclass

try:  # psutil 随安装包一起提供；源码运行时装在 .venv 里
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


@dataclass
class Snapshot:
    cpu_percent: float = 0.0
    mem_total: int = 0
    mem_available: int = 0
    disk_total: int = 0
    disk_free: int = 0
    cpu_count: int = 0


def _mem_fallback() -> tuple[int, int]:
    try:
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullTotalPhys), int(st.ullAvailPhys)
    except Exception:  # noqa: BLE001
        return 0, 0


def snapshot(path: str | None = None) -> Snapshot:
    snap = Snapshot(cpu_count=os.cpu_count() or 0)
    target = path or os.getcwd()
    try:
        du = shutil.disk_usage(target)
        snap.disk_total, snap.disk_free = du.total, du.free
    except OSError:
        pass
    if psutil is not None:
        try:
            snap.cpu_percent = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            snap.mem_total, snap.mem_available = int(vm.total), int(vm.available)
        except Exception:  # noqa: BLE001
            pass
    if not snap.mem_total:
        snap.mem_total, snap.mem_available = _mem_fallback()
    return snap


def cpu_name() -> str:
    if psutil is not None:
        return os.environ.get("PROCESSOR_IDENTIFIER", "") or ""
    return os.environ.get("PROCESSOR_IDENTIFIER", "") or ""


def machine_line() -> str:
    snap = snapshot()
    return (f"{snap.cpu_count} 线程 · {snap.mem_total/1024**3:.1f} GB 内存 · "
            f"磁盘剩余 {snap.disk_free/1024**3:.1f} GB")


def has_psutil() -> bool:
    return psutil is not None


def process_mem_mb(pid: int) -> float | None:
    if psutil is None:
        return None
    try:
        return psutil.Process(pid).memory_info().rss / 1024 ** 2
    except Exception:  # noqa: BLE001
        return None
