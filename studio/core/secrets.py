# -*- coding: utf-8 -*-
"""AI Key 的本地安全存储。

顺序：
1. 环境变量 FTS_AI_KEY / OPENROUTER_API_KEY（推荐，优先级最高）
2. Windows DPAPI 加密文件 studio_state/secrets/ai_key.bin（仅当前用户可解密）
3. 旧版 studio_config.json 里的 ai.key：首次读取时迁移到 DPAPI 并从配置中清除

本模块不会把 Key 写进日志、异常消息或明文配置。
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from pathlib import Path

from . import fsutil, paths

ENV_NAMES = ("FTS_AI_KEY", "OPENROUTER_API_KEY")
_SECRET_FILE = "ai_key.bin"


class SecretError(Exception):
    pass


class _BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def secrets_dir() -> Path:
    return paths.state_dir() / "secrets"


def secret_path() -> Path:
    return secrets_dir() / _SECRET_FILE


def key_from_env() -> str:
    for name in ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _dpapi(data: bytes, protect: bool) -> bytes:
    if os.name != "nt":
        raise SecretError("DPAPI 仅在 Windows 上可用；请改用环境变量 FTS_AI_KEY")
    blob_in = _BLOB()
    buf = ctypes.create_string_buffer(data)
    blob_in.cbData = len(data)
    blob_in.pbData = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))
    blob_out = _BLOB()
    fn = ctypes.windll.crypt32.CryptProtectData if protect else ctypes.windll.crypt32.CryptUnprotectData
    args = [ctypes.byref(blob_in), "fts-ai-key", None, None, None, 1, ctypes.byref(blob_out)]
    if not fn(*args):
        raise SecretError(f"DPAPI {'加密' if protect else '解密'}失败（错误码 {ctypes.get_last_error()}）")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def read_stored_key() -> str:
    """读取 DPAPI 文件；文件不存在返回空串。"""
    p = secret_path()
    if not p.is_file():
        return ""
    try:
        raw = p.read_bytes()
        return _dpapi(raw, protect=False).decode("utf-8").strip()
    except (OSError, SecretError, UnicodeDecodeError):
        return ""


def store_key(key: str) -> None:
    key = (key or "").strip()
    if not key:
        raise SecretError("Key 不能为空")
    blob = _dpapi(key.encode("utf-8"), protect=True)
    fsutil.atomic_write_bytes(secret_path(), blob)


def clear_stored_key() -> bool:
    p = secret_path()
    try:
        if p.is_file():
            p.unlink()
            return True
    except OSError:
        pass
    return False


def _cfg(cfg):
    if cfg is not None:
        return cfg
    from . import config as config_mod

    return config_mod.cfg()


def key_source(cfg=None) -> str:
    if key_from_env():
        return "env"
    if read_stored_key():
        return "dpapi"
    if str(_cfg(cfg).get("ai.key", "") or "").strip():
        return "legacy-plaintext"
    return ""


def migrate_legacy_key(cfg=None) -> tuple[bool, str]:
    """把旧明文 ai.key 迁移进 DPAPI，并从配置文件里清除。返回 (是否迁移, 说明)。"""
    cfg = _cfg(cfg)
    legacy = str(cfg.get("ai.key", "") or "").strip()
    if not legacy:
        return False, ""
    try:
        if not read_stored_key():
            store_key(legacy)
        cfg.set("ai.key", "", save=True)
        return True, "旧明文 Key 已迁移到 Windows DPAPI，并从 studio_config.json 清除；请尽快在服务商后台撤销并轮换旧 Key。"
    except (SecretError, OSError) as exc:
        # 迁移失败时不清空明文，避免用户丢失 Key；但明确提示
        return False, f"旧明文 Key 迁移失败：{exc}。请改用环境变量 FTS_AI_KEY 并手动清空配置里的 ai.key。"


def scrub_config_key(cfg=None) -> bool:
    """仅清除配置里的明文 Key（不读取、不回显）。"""
    cfg = _cfg(cfg)
    if not str(cfg.get("ai.key", "") or "").strip():
        return False
    cfg.set("ai.key", "", save=True)
    return True


def get_api_key(cfg=None) -> str:
    """env → DPAPI；顺带完成旧明文配置迁移。"""
    cfg = _cfg(cfg)
    migrated, _note = migrate_legacy_key(cfg)
    del migrated
    env = key_from_env()
    if env:
        return env
    return read_stored_key()


def set_api_key(key: str, cfg=None) -> str:
    """保存 Key（DPAPI），同时清空配置里的明文。返回空串表示成功。"""
    cfg = _cfg(cfg)
    key = (key or "").strip()
    if not key:
        return "Key 不能为空"
    if key_from_env():
        return "检测到环境变量 FTS_AI_KEY / OPENROUTER_API_KEY，环境变量优先；这里的保存只影响本地 DPAPI 存储"
    try:
        store_key(key)
    except (SecretError, OSError) as exc:
        return str(exc)
    scrub_config_key(cfg)
    return ""


def clear_api_key(cfg=None) -> str:
    cfg = _cfg(cfg)
    cleared = clear_stored_key()
    scrub_config_key(cfg)
    if key_from_env():
        return ("已清除本地 DPAPI Key 和配置明文；但环境变量里仍有 Key，"
                "请在系统环境变量中删除 FTS_AI_KEY / OPENROUTER_API_KEY")
    return "已清除本地保存的 Key" if cleared else "没有本地保存的 Key"
