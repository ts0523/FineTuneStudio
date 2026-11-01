# -*- coding: utf-8 -*-
"""模型仓库访问（默认走 hf-mirror 镜像）。

只做只读查询：搜索、仓库文件清单。真正的下载在 workers/download_worker.py 里跑，
避免界面线程被网络阻塞。
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

UA = "FineTuneStudio/1.0 (local desktop app)"


class HubError(Exception):
    pass


def normalize_endpoint(endpoint: str) -> str:
    """只接受 http/https；拒绝 file://、ftp:// 等协议。"""
    ep = (endpoint or "https://hf-mirror.com").strip().rstrip("/")
    if "://" not in ep:
        ep = "https://" + ep
    parts = urllib.parse.urlsplit(ep)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise HubError(f"镜像地址协议不允许：{endpoint!r}（只支持 http/https）")
    return ep


def _validate_repo(repo: str) -> str:
    repo = (repo or "").strip().strip("/")
    parts = repo.split("/")
    if len(parts) != 2 or not all(re.match(r"^[A-Za-z0-9._-]{1,80}$", p) for p in parts):
        raise HubError("仓库名格式应为 组织/模型名，且不能包含路径字符")
    return repo


def _get_json(url: str, timeout: float = 20.0) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 401 or exc.code == 403:
            raise HubError(f"仓库拒绝访问（HTTP {exc.code}），可能需要授权或仓库不存在") from exc
        if exc.code == 404:
            raise HubError("仓库不存在（HTTP 404）") from exc
        raise HubError(f"HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise HubError(f"网络不可达：{exc.reason}（检查镜像地址或代理）") from exc
    except TimeoutError as exc:
        raise HubError("请求超时，稍后重试或换镜像") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HubError("返回内容不是 JSON，镜像可能被拦") from exc


def search(query: str, endpoint: str = "https://hf-mirror.com", limit: int = 20) -> list[dict]:
    ep = normalize_endpoint(endpoint)
    q = urllib.parse.quote((query or "").strip())
    url = f"{ep}/api/models?search={q}&limit={int(limit)}&sort=downloads&direction=-1"
    data = _get_json(url, timeout=20)
    if not isinstance(data, list):
        raise HubError("搜索返回格式异常")
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tags = item.get("tags") or []
        out.append({
            "id": item.get("modelId") or item.get("id") or "",
            "downloads": int(item.get("downloads") or 0),
            "likes": int(item.get("likes") or 0),
            "pipeline": item.get("pipeline_tag") or "",
            "tags": [str(t) for t in tags][:6],
            "updated": str(item.get("lastModified") or "")[:10],
        })
    return [x for x in out if x["id"]]


def repo_files(repo: str, endpoint: str = "https://hf-mirror.com") -> tuple[list[dict], str]:
    """返回 (文件清单, 仓库最后更新时间)。每项 {name,size,sha256}。"""
    ep = normalize_endpoint(endpoint)
    repo = _validate_repo(repo)
    url = f"{ep}/api/models/{repo}?blobs=true"
    data = _get_json(url, timeout=25)
    if not isinstance(data, dict):
        raise HubError("仓库信息格式异常")
    files = []
    for sib in data.get("siblings") or []:
        if not isinstance(sib, dict):
            continue
        name = str(sib.get("rfilename") or "")
        if not name:
            continue
        lfs = sib.get("lfs") if isinstance(sib.get("lfs"), dict) else {}
        size = sib.get("size") or lfs.get("size") or 0
        files.append({"name": name, "size": int(size or 0), "sha256": lfs.get("sha256") or ""})
    return files, str(data.get("lastModified") or "")


def search_datasets(query: str, endpoint: str = "https://hf-mirror.com",
                    limit: int = 12) -> list[dict]:
    """在镜像站搜索公开数据集（只读元信息，不下载，用于帮训时找合法资源）。"""
    ep = normalize_endpoint(endpoint)
    q = urllib.parse.quote((query or "").strip())
    url = f"{ep}/api/datasets?search={q}&limit={int(limit)}&sort=downloads&direction=-1"
    data = _get_json(url, timeout=25)
    if not isinstance(data, list):
        raise HubError("数据集搜索返回格式异常")
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tags = [str(t) for t in (item.get("tags") or [])][:8]
        out.append({
            "id": str(item.get("id") or item.get("datasetId") or ""),
            "downloads": int(item.get("downloads") or 0),
            "likes": int(item.get("likes") or 0),
            "tags": tags,
            "updated": str(item.get("lastModified") or "")[:10],
            "license": next((t.split(":", 1)[1] for t in tags if t.lower().startswith("license:")),
                            ""),
        })
    return [x for x in out if x["id"]]


CURATED = [
    {"repo": "jingyaogong/minimind-3o", "params": "0.1B", "lang": "中英",
     "why": "推荐：MiniMind-O 全模态 0.1B（文本主干 64M 可训，216 MB）"},
    {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "params": "0.5B", "lang": "中英",
     "why": "标准架构，CPU 可做 LoRA；内存约 2~3 GB，适合试标准模型"},
    {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "params": "1.5B", "lang": "中英",
     "why": "效果更好但 CPU 训练慢，建议只做少量步数验证流程"},
    {"repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "params": "1.1B", "lang": "英文",
     "why": "老牌小模型，架构标准，便于验证下载/加载链路"},
    {"repo": "google/gemma-2-2b-it", "params": "2B", "lang": "多语",
     "why": "质量高；需要 HF 授权，下载前先在网页同意许可"},
]


def curated() -> list[dict]:
    return [dict(x) for x in CURATED]
