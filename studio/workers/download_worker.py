# -*- coding: utf-8 -*-
"""模型下载 worker：走镜像站，逐文件断点续传，实时上报速度/进度/ETA。

用法：
    python download_worker.py --config dl_cfg.json
dl_cfg.json：{"repo","dest","endpoint","allow":[...],"verify_sha":true,"cancel_flag":"..."}
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TypedDict

EVT = "@@EVT@@"
UA = "FineTuneStudio/1.0 (local desktop app)"
CHUNK = 1024 * 256

# 默认不下载 *.py：模型仓库里的代码会在加载时被 import 执行
DEFAULT_ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken",
                 "*.safetensors.index.json"]

# MiniMind-O 内置可信代码哈希（与 studio/core/trust.py 保持一致）
BUILTIN_CODE = {
    "jingyaogong/minimind-3o": {
        "model_minimind.py": "c97b24b4b900dbbf5fabc0cca81a9f580011f91e032501f41684e10c6ca6ee60",
        "model_omni.py": "0fc7abb2ed621508f22682f81b18fb778ceea0bb5563c8b5ae1d389ed77c01ad",
    }
}
_MUTABLE_REFS = {"", "main", "master", "head", "latest"}
_FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
TRUST_FILE = ".fts_trust.json"


class _FileEntry(TypedDict):
    name: str
    size: int
    sha256: str


def emit(**kw) -> None:
    kw.setdefault("t", round(time.time(), 3))
    sys.stdout.write(EVT + json.dumps(kw, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    sys.stdout.write(str(msg).rstrip() + "\n")
    sys.stdout.flush()


def validate_endpoint(endpoint: str) -> str:
    """只允许 http/https，拒绝 file://、ftp://、空协议等。"""
    ep = (endpoint or "https://hf-mirror.com").strip().rstrip("/")
    if "://" not in ep:
        ep = "https://" + ep
    parts = urllib.parse.urlsplit(ep)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"下载端点协议不允许或地址无效：{endpoint!r}（只支持 http/https）")
    return ep


def safe_rel_name(name: str) -> str:
    """仓库文件相对路径安全校验：拒绝绝对路径 / 盘符 / UNC / 空名 / .. 分量。"""
    raw = str(name or "")
    if not raw.strip():
        raise ValueError("空文件名")
    if "\x00" in raw or re.search(r"[\x01-\x1f]", raw):
        raise ValueError("文件名含控制字符")
    norm = raw.replace("\\", "/")
    if norm.startswith("/") or raw.startswith("\\\\"):
        raise ValueError("拒绝绝对路径 / UNC 路径")
    if re.match(r"^[A-Za-z]:", raw) or ":" in raw:
        raise ValueError("拒绝盘符路径")
    parts = norm.split("/")
    if any(not p or p in (".", "..") for p in parts):
        raise ValueError("拒绝空路径分量或 . / .. 分量")
    for p in parts:
        if any(ch in p for ch in '<>"|?*'):
            raise ValueError("文件名含 Windows 不允许的字符")
        if p.startswith("."):
            raise ValueError("拒绝隐藏文件名")
    return norm


def api_json(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def wanted(name: str, allow: list[str], ignore: list[str]) -> bool:
    if name.startswith(".") or name.endswith(".py"):
        return False
    for pat in ignore:
        if fnmatch.fnmatch(name, pat):
            return False
    return any(fnmatch.fnmatch(name, pat) for pat in allow)


def validate_dest(dest: Path) -> Path:
    """下载目标必须是 FTS_ROOT/models 下的目录。"""
    root_models = (Path(os.environ.get("FTS_ROOT", ".")) / "models").resolve()
    dest = Path(dest).resolve()
    try:
        dest.relative_to(root_models)
    except ValueError as exc:
        raise ValueError(f"下载目标越界：{dest} 不在 {root_models} 内") from exc
    return dest


def _target_path(dest: Path, name: str) -> Path:
    name = safe_rel_name(name)
    target = dest.joinpath(*name.split("/"))
    try:
        target.resolve().relative_to(dest.resolve())
    except ValueError as exc:
        raise ValueError("下载路径越界") from exc
    return target


def _download_once(endpoint: str, repo: str, name: str, dest: Path, size: int,
                   overall: dict, cancel_flag: Path, restart: bool = False,
                   revision: str = "main") -> tuple[bool, str]:
    """下载一次。返回 (成功, 失败原因)。

    镜像站的 Range 行为并不总是可靠（有时忽略 Range 直接回整份文件，状态码仍是 206），
    所以这里必须核对 Content-Range 起点；对不上就丢弃本地 .part 重下，绝不盲目追加。
    """
    target = _target_path(dest, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and size and target.stat().st_size == size:
        overall["done"] += size
        emit(type="file_skip", name=name, size=size, done=overall["done"], total=overall["total"])
        return True, ""

    part = target.with_name(target.name + ".part")
    if restart and part.is_file():
        part.unlink(missing_ok=True)
    have = part.stat().st_size if part.is_file() else 0
    if size and have > size:
        part.unlink(missing_ok=True)
        have = 0

    quote = urllib.parse.quote(name)
    rev = urllib.parse.quote(revision or "main", safe="")
    url = f"{endpoint}/{repo}/resolve/{rev}/{quote}?download=true"
    headers = {"User-Agent": UA, "Range": f"bytes={have}-"} if have else {"User-Agent": UA}
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and size and have == size:
            part.replace(target)
            overall["done"] += size
            return True, ""
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, f"{exc.reason}（网络中断或镜像不可达）"

    mode, skip = "wb", 0
    if have and resp.status == 206:
        cr = resp.headers.get("Content-Range", "")
        start = -1
        try:
            start = int(cr.split()[1].split("-")[0])
        except (IndexError, ValueError):
            start = -1
        if start == have:
            mode, skip = "ab", have
        else:
            resp.close()
            part.unlink(missing_ok=True)
            return False, f"续传区间不匹配（本地 {have}，服务端 {start}）"
    elif have:
        mode, skip = "wb", 0          # 服务端忽略 Range，返回整份：覆盖写

    if not size:
        cr = resp.headers.get("Content-Range", "")
        if "/" in cr:
            try:
                size = int(cr.split("/")[-1])
                overall["total"] += size
            except ValueError:
                pass
    overall["done"] += skip
    emit(type="file_start", name=name, size=size, done=overall["done"], total=overall["total"],
         resumed=bool(skip))

    t_last = time.time()
    bytes_window = 0
    written = skip
    speed = 0.0
    try:
        with part.open(mode) as f:
            while True:
                if cancel_flag.exists():
                    emit(type="cancelled", name=name, done=overall["done"], total=overall["total"])
                    return False, "用户取消"
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                n = len(chunk)
                written += n
                overall["done"] += n
                bytes_window += n
                now = time.time()
                if now - t_last >= 0.25:
                    speed = bytes_window / (now - t_last)
                    bytes_window = 0
                    t_last = now
                    eta = ((overall["total"] - overall["done"]) / speed) if speed > 0 else None
                    emit(type="progress", name=name, done=overall["done"], total=overall["total"],
                         file_done=written, file_total=size,
                         speed=round(speed), eta=round(eta, 1) if eta else None)
    except (OSError, urllib.error.URLError) as exc:
        overall["done"] -= written
        return False, f"传输中断：{exc}"
    finally:
        resp.close()

    got = part.stat().st_size if part.is_file() else 0
    if size and got != size:
        overall["done"] -= written
        if got > size:
            part.unlink(missing_ok=True)     # 多出来的字节说明服务端重复发了数据
            return False, f"字节数超出预期（{got} > {size}），已清空重下"
        return False, f"字节数不足（{got} / {size}），保留 .part 续传"
    part.replace(target)
    return True, ""


def download_file(endpoint: str, repo: str, name: str, dest: Path, size: int,
                  overall: dict, cancel_flag: Path, revision: str = "main") -> bool:
    """下载单个文件（失败自动从头重试一次），返回 True 表示成功。"""
    last = ""
    for attempt in (0, 1):
        ok, reason = _download_once(endpoint, repo, name, dest, size, overall, cancel_flag,
                                    restart=attempt == 1, revision=revision)
        if ok:
            return True
        last = reason or last
        if cancel_flag.exists():
            return False
        if attempt == 0:
            log(f"[重试] {name}：{reason}")
    emit(type="error", message=f"{name} 下载失败：{last}")
    return False


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def watch_parent() -> None:
    """父进程（界面）消失后自动退出，避免留下孤儿进程继续吃 CPU。"""
    import ctypes
    import threading

    pid = os.environ.get("FTS_PARENT_PID", "")
    if not pid.isdigit():
        return
    ppid = int(pid)

    def loop() -> None:
        k32 = ctypes.windll.kernel32
        while True:
            time.sleep(2.0)
            handle = k32.OpenProcess(0x00100000, False, ppid)  # SYNCHRONIZE
            if not handle:
                os._exit(9)
            k32.CloseHandle(handle)

    threading.Thread(target=loop, daemon=True).start()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    watch_parent()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))

    repo = str(cfg.get("repo") or "").strip().strip("/")
    parts = repo.split("/")
    if len(parts) != 2 or not all(re.match(r"^[A-Za-z0-9._-]{1,80}$", p) for p in parts):
        emit(type="error", message=f"仓库名不合法：{repo!r}（应为 组织/模型名，且不能含路径字符）")
        return 2
    try:
        endpoint = validate_endpoint(str(cfg.get("endpoint") or "https://hf-mirror.com"))
        dest = validate_dest(Path(cfg["dest"]))
    except ValueError as exc:
        emit(type="error", message=str(exc))
        return 2
    allow = cfg.get("allow") or DEFAULT_ALLOW
    ignore = cfg.get("ignore") or ["*.md", "*.gitattributes", "*.png", "*.jpg", "*.gguf", "*.h5", "*.msgpack", "*.ot"]
    cancel_flag = Path(cfg["cancel_flag"]) if cfg.get("cancel_flag") else dest / ".cancel"
    verify_sha = bool(cfg.get("verify_sha", True))
    allow_code = bool(cfg.get("allow_code", False))
    revision = str(cfg.get("revision") or "main").strip()
    builtin = BUILTIN_CODE.get(repo)

    if allow_code and not builtin and not _FULL_COMMIT.fullmatch(revision):
        emit(type="error", message="下载并执行自定义模型代码必须指定完整的 40 位 commit SHA（不能用 main/tag）")
        return 2

    emit(type="stage", name="list", message=f"读取仓库清单 {repo}（revision {revision}）")
    try:
        rev = urllib.parse.quote(revision or "main", safe="")
        info = api_json(f"{endpoint}/api/models/{repo}/revision/{rev}?blobs=true", timeout=30)
    except urllib.error.HTTPError as exc:
        msg = {401: "需要授权或仓库不存在", 403: "访问被拒绝（可能需要登录）", 404: "仓库不存在"}.get(
            exc.code, f"HTTP {exc.code}")
        emit(type="error", message=f"读取仓库失败：{msg}")
        return 3
    except Exception as exc:  # noqa: BLE001
        emit(type="error", message=f"读取仓库失败：{type(exc).__name__}: {exc}")
        return 3

    files: list[_FileEntry] = []
    skipped = 0
    code_files: list[str] = []
    for sib in info.get("siblings") or []:
        name = str(sib.get("rfilename") or "")
        if not name:
            continue
        try:
            name = safe_rel_name(name)
        except ValueError as exc:
            skipped += 1
            emit(type="warn", message=f"忽略不安全的仓库文件名 {name!r}：{exc}")
            continue
        if name.endswith(".py"):
            if builtin and name in builtin:
                pass                    # 内置可信代码：固定哈希，下载后逐个校验
            elif not allow_code:
                skipped += 1
                emit(type="warn", message=f"已跳过模型代码 {name}（未明确信任；加载时也不会执行）")
                continue
            elif cfg.get("code_allowlist"):
                expected = str((cfg.get("code_allowlist") or {}).get(name) or "").lower()
                lfs_hash = str((sib.get("lfs") or {}).get("sha256") or "").lower()
                if expected and lfs_hash and expected != lfs_hash:
                    skipped += 1
                    emit(type="warn", message=f"跳过 {name}：哈希不在允许列表内")
                    continue
            code_files.append(name)
        elif not wanted(name, allow, ignore):
            continue
        lfs = sib.get("lfs") if isinstance(sib.get("lfs"), dict) else {}
        size = int(sib.get("size") or lfs.get("size") or 0)
        files.append({"name": name, "size": size, "sha256": lfs.get("sha256") or ""})

    if not files:
        emit(type="error", message="仓库里没有符合规则的模型文件（可能只有 .gguf 或需要登录）")
        return 4

    files.sort(key=lambda x: (x["size"] == 0, -x["size"]))
    total = sum(f["size"] for f in files)
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(str(dest)).free
    if total and free < total * 1.05:
        emit(type="error", message=f"磁盘空间不足：需要 {human(total)}，剩余 {human(free)}")
        return 5
    emit(type="list", repo=repo, files=len(files), total=total, dest=str(dest),
         names=[f["name"] for f in files], skipped=skipped, code=(len(code_files) if allow_code or builtin else 0))

    overall = {"done": 0, "total": total}
    for idx, f in enumerate(files, 1):
        if cancel_flag.exists():
            emit(type="cancelled", done=overall["done"], total=overall["total"])
            return 6
        emit(type="file", index=idx, count=len(files), name=f["name"], size=f["size"])
        ok = download_file(endpoint, repo, f["name"], dest, f["size"], overall, cancel_flag,
                           revision=revision)
        if not ok:
            if cancel_flag.exists():
                emit(type="cancelled", done=overall["done"], total=overall["total"])
                return 6
            return 7
        if verify_sha and f["sha256"]:
            emit(type="stage", name="verify", message=f"校验 {f['name']}")
            p = dest / f["name"]
            try:
                got = sha256_of(p)
            except OSError as exc:
                emit(type="error", message=f"校验读取失败：{exc}")
                return 8
            if got.lower() != f["sha256"].lower():
                p.unlink(missing_ok=True)
                emit(type="error", message=f"{f['name']} 校验失败（sha256 不一致），文件已删除，请重下")
                return 8

    # ------------------------------------------------ 模型代码信任记录
    hashes: dict[str, str] = {}
    for name in code_files:
        p = _target_path(dest, name)
        if not p.is_file():
            emit(type="error", message=f"模型代码 {name} 没有下载成功")
            return 7
        hashes[name] = sha256_of(p)

    allowlist = cfg.get("code_allowlist") or {}
    for name, digest in hashes.items():
        expected = str(allowlist.get(name) or "").lower()
        if expected and digest.lower() != expected:
            _target_path(dest, name).unlink(missing_ok=True)
            emit(type="error", message=f"{name} 的实际 SHA-256 不在允许列表内，已删除；不会记录信任")
            return 8

    if builtin:
        for name, expected in builtin.items():
            p = _target_path(dest, name)
            if not p.is_file():
                emit(type="error", message=f"MiniMind-O 缺少内置代码文件 {name}")
                return 7
            if hashes.get(name, "").lower() != expected.lower():
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
                emit(type="error", message=f"{name} 与内置可信哈希不一致，已删除；不会执行该代码")
                return 8
    if hashes or builtin:
        manifest = {"repo": repo, "revision": revision, "code_sha256": hashes,
                    "trusted_at": time.time(),
                    "note": "仅对列出的代码哈希生效；权重 SHA-256 不代表整个仓库可信"}
        try:
            tmp = dest / (TRUST_FILE + ".tmp")
            tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, dest / TRUST_FILE)
        except OSError as exc:
            emit(type="error", message=f"写入代码信任记录失败：{exc}")
            return 8
        emit(type="code_policy", repo=repo, revision=revision,
             files=sorted(hashes), builtin=bool(builtin))

    total_bytes = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    emit(type="done", repo=repo, dest=str(dest), files=len(files), bytes=total_bytes)
    log(f"[完成] {repo} → {dest}（{human(total_bytes)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
