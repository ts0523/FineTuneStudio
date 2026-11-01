# -*- coding: utf-8 -*-
"""模型代码信任策略。

背景：HuggingFace 仓库里的 *.py 会在加载模型时被 import 执行，等同于在本机运行
任意代码。权重文件的 SHA-256 只能证明「这个权重文件没被改过」，**不能**证明整个
仓库（含代码）可信，二者必须分开对待。

规则：
- MiniMind-O（jingyaogong/minimind-3o）是内置可信例外，代码文件哈希固定在本模块。
- 其他仓库默认不下载、不执行 *.py；用户必须显式确认信任，提供完整 commit
  SHA（不能是 main/master/tag），下载时把代码文件哈希写入 .fts_trust.json。
- 每次加载都重新校验目录内代码哈希；对不上就拒绝执行。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import fsutil, paths

TRUST_FILE = ".fts_trust.json"

# MiniMind-O 内置可信代码（固定 SHA-256，改动任何一行都会导致校验失败）
BUILTIN_CODE = {
    "jingyaogong/minimind-3o": {
        "model_minimind.py": "c97b24b4b900dbbf5fabc0cca81a9f580011f91e032501f41684e10c6ca6ee60",
        "model_omni.py": "0fc7abb2ed621508f22682f81b18fb778ceea0bb5563c8b5ae1d389ed77c01ad",
    }
}
BUILTIN_REPOS = tuple(BUILTIN_CODE)
_MUTABLE_REFS = {"", "main", "master", "head", "latest"}


class TrustError(PermissionError):
    pass


def code_files(dirpath: Path) -> list[Path]:
    return sorted(p for p in Path(dirpath).glob("*.py") if p.is_file())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def code_hashes(dirpath: Path) -> dict[str, str]:
    return {p.name: sha256(p) for p in code_files(dirpath)}


def read_manifest(dirpath: Path) -> dict:
    p = Path(dirpath) / TRUST_FILE
    try:
        obj = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def write_manifest(dirpath: Path, repo: str, revision: str,
                   hashes: dict[str, str] | None = None) -> Path:
    hashes = hashes if hashes is not None else code_hashes(dirpath)
    obj = {"repo": repo, "revision": revision, "code_sha256": hashes,
           "trusted_at": __import__("time").time(),
           "note": "仅对列出的代码文件哈希生效；权重 SHA-256 不代表仓库可信"}
    p = Path(dirpath) / TRUST_FILE
    fsutil.atomic_write_json(p, obj, indent=1)
    return p


def trust_record_path(repo: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (repo or "").strip().strip("/"))[:60] or "repo"
    return paths.state_dir() / "trust" / f"{safe}.json"


def builtin_repo(repo: str) -> bool:
    return (repo or "").strip().strip("/").lower() in {r.lower() for r in BUILTIN_REPOS}


def builtin_match(dirpath: Path) -> bool:
    """目录内代码与 MiniMind-O 的固定哈希完全一致。"""
    actual = code_hashes(dirpath)
    if not actual:
        return False
    for pinned in BUILTIN_CODE.values():
        if set(actual) == set(pinned) and all(pinned[n] == h for n, h in actual.items()):
            return True
    return False


def check_code_trust(dirpath: Path, repo: str = "") -> tuple[bool, str]:
    """加载前调用：目录内有代码时，必须命中内置哈希或有效的信任清单。"""
    files = code_files(dirpath)
    if not files:
        return True, "无模型代码"
    if builtin_match(dirpath):
        return True, "MiniMind-O 内置可信代码哈希匹配"
    manifest = read_manifest(dirpath)
    m_repo = str(manifest.get("repo") or repo or "")
    revision = str(manifest.get("revision") or "")
    if not manifest:
        return False, (f"目录含 {len(files)} 个 .py 模型代码，代码哈希与内置固定值不匹配，"
                       f"且没有 {TRUST_FILE} 信任记录；仅为权重校验通过不代表代码可信")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision.strip()):
        return False, f"信任记录里的 revision={revision!r} 不是完整 commit SHA，拒绝执行模型代码"
    if builtin_repo(m_repo) and not builtin_match(dirpath):
        return False, "声称是 MiniMind-O，但代码哈希与内置固定哈希不一致"
    pinned = manifest.get("code_sha256") or {}
    if not isinstance(pinned, dict):
        return False, "信任记录格式错误（code_sha256 不是对象）"
    if set(pinned) != {p.name for p in files}:
        return False, "目录里的代码文件与信任记录清单不一致（多出或缺少文件）"
    for p in files:
        if str(pinned.get(p.name) or "").lower() != sha256(p):
            return False, f"{p.name} 的哈希与信任记录不一致"
    return True, f"已按固定 revision {revision} 校验 {len(files)} 个代码文件"


def assert_code_trusted(dirpath: Path, repo: str = "") -> str:
    ok, reason = check_code_trust(dirpath, repo)
    if not ok:
        raise TrustError(reason)
    return reason


def validate_revision(revision: str) -> str:
    """下载自定义代码仓库时必须使用不可变的完整 commit SHA。"""
    value = (revision or "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise TrustError("下载/执行自定义模型代码必须指定完整的 40 位 commit SHA，不能用 main/tag/短 SHA")
    return value.lower()


def repo_verified_by_weights(repo: str, weight_hashes: dict | None = None) -> bool:
    """明确：权重哈希校验通过 ≠ 仓库可信。这个函数永远只描述权重。"""
    if not weight_hashes:
        return False
    return all(bool(v) for v in weight_hashes.values())
