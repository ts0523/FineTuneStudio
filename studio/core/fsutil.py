# -*- coding: utf-8 -*-
"""文件系统安全工具：原子写入、安全命名、路径边界校验、停止标记清理。

所有「用户可控的名字」和「下载来的相对路径」都必须先经过这里的校验，
禁止借名字写出预期根目录（绝对路径 / 盘符 / UNC / ..）。
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

_BAD_NAME_CHARS = set('<>:"|?*')
_CONTROL_RE = re.compile(r"[\x00-\x1f]")
_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {
    f"lpt{i}" for i in range(1, 10)}


class PathSafetyError(ValueError):
    """名字/路径不满足安全规则。"""


def validate_component(name: str, *, max_len: int = 96, allow_dot_prefix: bool = False) -> str:
    """校验单个路径分量（目录名/文件名），不合法直接拒绝。返回去掉首尾空白后的名字。"""
    raw = str(name or "")
    if _CONTROL_RE.search(raw):
        raise PathSafetyError("名字里含有控制字符")
    value = raw.strip()
    if not value:
        raise PathSafetyError("名字不能为空")
    if value in (".", ".."):
        raise PathSafetyError("名字不能是 . 或 ..")
    if any(sep in value for sep in ("/", "\\")):
        raise PathSafetyError("名字里不能包含路径分隔符")
    if ":" in value:
        raise PathSafetyError("名字里不能包含盘符或冒号")
    if any(ch in value for ch in _BAD_NAME_CHARS):
        raise PathSafetyError("名字里含有 Windows 不允许的字符 < > : \" | ? *")
    if not allow_dot_prefix and value.startswith("."):
        raise PathSafetyError("名字不能以 . 开头")
    stem = value.split(".", 1)[0].lower()
    if stem in _RESERVED:
        raise PathSafetyError(f"名字 {value} 是 Windows 保留设备名")
    if len(value) > max_len:
        raise PathSafetyError(f"名字过长（>{max_len} 字符）")
    return value


def sanitize_component(name: str, fallback: str = "item", *, max_len: int = 72) -> str:
    """把用户显示名清理成可用目录名（不是安全边界，安全边界仍走 validate_component）。"""
    value = str(name or "").strip()
    value = _CONTROL_RE.sub("", value)
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value)
    value = value.strip(".-_")
    value = value[:max_len] or fallback
    # 去掉可能形成保留名的前缀
    if value.split(".", 1)[0].lower() in _RESERVED:
        value = "_" + value
    return value


def safe_child(root: Path, name: str, *, max_len: int = 96) -> Path:
    """在 root 下安全地构造一个子路径；name 必须是单个分量。"""
    root = Path(root)
    comp = validate_component(name, max_len=max_len)
    target = (root / comp).resolve()
    root_r = root.resolve()
    try:
        target.relative_to(root_r)
    except ValueError as exc:
        raise PathSafetyError(f"路径越界：{target} 不在 {root_r} 内") from exc
    return target


def _validate_rel_parts(rel_name: str) -> list[str]:
    """校验仓库里的相对文件名（允许 a/b/c 形式，但每段都必须安全）。"""
    raw = str(rel_name or "").replace("\\", "/")
    if not raw.strip():
        raise PathSafetyError("空文件名")
    if raw.startswith("/"):
        raise PathSafetyError("拒绝绝对路径")
    parts = raw.split("/")
    if any(not p for p in parts):
        raise PathSafetyError("拒绝空路径分量")
    if any(p in (".", "..") for p in parts):
        raise PathSafetyError("拒绝 . / .. 路径分量")
    if re.match(r"^[A-Za-z]:", raw):
        raise PathSafetyError("拒绝盘符路径")
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise PathSafetyError("拒绝 UNC 路径")
    for p in parts:
        validate_component(p, allow_dot_prefix=False, max_len=160)
    return parts


def resolve_download_target(dest: Path, rel_name: str) -> Path:
    """下载文件名 → 目标路径；解析后仍必须落在 dest 内。"""
    dest = Path(dest)
    parts = _validate_rel_parts(rel_name)
    target = dest.joinpath(*parts)
    dest_r = dest.resolve()
    try:
        target.resolve().relative_to(dest_r)
    except ValueError as exc:
        raise PathSafetyError(f"下载路径越界：{target} 不在 {dest_r} 内") from exc
    return target


# ---------------------------------------------------------------- 原子写入
def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """同目录临时文件 + flush/fsync + os.replace；失败时原文件保持不变。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".fts-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".fts-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, obj: object, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent) + "\n")


def atomic_write_lines(path: Path, lines: Iterable[str]) -> int:
    """逐行原子写入（不额外拼大字符串）。返回写出行数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".fts-", suffix=".tmp", dir=str(path.parent))
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(str(line))
                if not str(line).endswith("\n"):
                    f.write("\n")
                n += 1
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return n
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_copy_file(src: Path, dst: Path) -> None:
    """原子复制文件（导入数据集时用）。"""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".fts-", suffix=".tmp", dir=str(dst.parent))
    try:
        with src.open("rb") as rf, os.fdopen(fd, "wb") as wf:
            while True:
                chunk = rf.read(1024 * 1024)
                if not chunk:
                    break
                wf.write(chunk)
            wf.flush()
            os.fsync(wf.fileno())
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------- 停止标记
def prepare_stop_flag(path: Path) -> bool:
    """新训练/续训启动前清掉陈旧停止标记。返回是否删除了旧标记。"""
    if path is None:
        return False
    p = Path(path)
    try:
        if p.is_file():
            p.unlink()
            return True
    except OSError:
        pass
    return False
