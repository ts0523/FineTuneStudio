# -*- coding: utf-8 -*-
"""用户主动设定的问答数据集。

每条样本由用户手写问题和回答，可选附加一张图片或一段录音。媒体会复制到
data/qa_media/<数据集>/ 下，写进 JSONL 的 image/audio 字段，训练 worker
会自动识别并启用对应投影层。
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import dataset as ds, fsutil, paths

MEDIA_DIR = "qa_media"


class CustomDataError(ValueError):
    pass


def dataset_name(name: str) -> str:
    value = fsutil.sanitize_component(name or "自建问答", "自建问答", max_len=60)
    return value if value.endswith(".jsonl") else value + ".jsonl"


def dataset_path(name: str) -> Path:
    return fsutil.safe_child(paths.data_dir(), dataset_name(name))


def _meta_path(path: Path) -> Path:
    return Path(str(path) + ".meta.json")


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        return [ln for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    except OSError:
        return []


def count(path: Path) -> int:
    return len(_read_lines(path))


def create(name: str) -> Path:
    path = dataset_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        fsutil.atomic_write_lines(path, [])
        fsutil.atomic_write_json(_meta_path(path), {
            "kind": "custom-qa", "name": path.name, "created": time.time(),
            "note": "用户主动设定的问答，可附加图片或语音",
        }, indent=1)
    return path


def rename(src: str | Path, new_name: str) -> Path:
    src = Path(src)
    dst = dataset_path(new_name)
    if dst == src:
        return src
    if dst.exists():
        raise CustomDataError(f"数据集已存在：{dst.name}")
    if not src.is_file():
        raise CustomDataError(f"数据集不存在：{src.name}")
    src.replace(dst)
    old_meta = _meta_path(src)
    new_meta = _meta_path(dst)
    if old_meta.is_file():
        try:
            meta = json.loads(old_meta.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        meta.update({"name": dst.name, "renamed": time.time()})
        fsutil.atomic_write_json(new_meta, meta, indent=1)
        try:
            old_meta.unlink()
        except OSError:
            pass
    return dst


def _copy_media(dataset: Path, kind: str, source: str | Path | None) -> str:
    if not source:
        return ""
    src = Path(source)
    if not src.is_file():
        raise CustomDataError(f"媒体文件不存在：{src}")
    sub = paths.data_dir() / MEDIA_DIR / dataset.stem
    sub.mkdir(parents=True, exist_ok=True)
    dst = fsutil.safe_child(sub, f"{int(time.time() * 1000)}-{src.name}", max_len=120)
    shutil.copy2(src, dst)
    return str(dst.resolve())


def make_record(question: str, answer: str, image: str | Path | None = None,
                audio: str | Path | None = None, audio_text: str = "") -> dict:
    q = str(question or "").strip()
    a = str(answer or "").strip()
    if not q:
        raise CustomDataError("问题不能为空")
    if not a:
        raise CustomDataError("回答不能为空")
    if len(q) > 2000 or len(a) > 8000:
        raise CustomDataError("问题或回答过长")
    if image and audio:
        raise CustomDataError("每条自建样本只附加一种媒体（图片或语音），避免训练时占位符冲突")
    content = q
    record: dict = {"messages": [{"role": "user", "content": content},
                                 {"role": "assistant", "content": a}]}
    if image:
        record["messages"][0]["content"] = "<|image_pad|>" + q
        record["image"] = str(image)
    if audio:
        record["messages"][0]["content"] = "<|audio_pad|>" + q
        record["audio"] = str(audio)
        record["audio_text"] = str(audio_text or q)
    ds.normalize(record)
    return record


def append_pair(path: str | Path, question: str, answer: str,
                image: str | Path | None = None, audio: str | Path | None = None,
                audio_text: str = "") -> int:
    if image and audio:
        raise CustomDataError("每条自建样本只附加一种媒体（图片或语音），避免训练时占位符冲突")
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    image_path = _copy_media(dst, "image", image)
    audio_path = _copy_media(dst, "audio", audio)
    record = make_record(question, answer, image_path, audio_path, audio_text)
    lines = _read_lines(dst)
    lines.append(json.dumps(record, ensure_ascii=False))
    fsutil.atomic_write_lines(dst, lines)
    meta = {}
    if _meta_path(dst).is_file():
        try:
            meta = json.loads(_meta_path(dst).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    meta.update({"kind": "custom-qa", "name": dst.name,
                 "updated": time.time(), "samples": len(lines)})
    fsutil.atomic_write_json(_meta_path(dst), meta, indent=1)
    return len(lines)


def samples(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in _read_lines(Path(path)):
        try:
            obj = json.loads(line)
            msgs = ds.normalize(obj)
        except (json.JSONDecodeError, ds.DataError, ValueError):
            continue
        item: dict = {"messages": msgs}
        for key in ("image", "audio", "audio_text"):
            if obj.get(key):
                item[key] = str(obj[key])
        out.append(item)
    return out
