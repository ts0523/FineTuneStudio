"""用户自建问答数据集：命名、媒体复制、追加与重命名。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.core import custom_data, dataset as ds


def test_custom_pair_roundtrip_without_media(isolated_root):
    path = custom_data.create("咖啡标准问答")
    assert path.parent == isolated_root / "data"
    n = custom_data.append_pair(path, "拿铁多少钱？", "拿铁 28 元。")
    assert n == 1
    got = custom_data.samples(path)
    assert got[0]["messages"][0]["content"] == "拿铁多少钱？"
    assert ds.validate(path).ok
    meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
    assert meta["samples"] == 1 and meta["kind"] == "custom-qa"


def test_custom_pair_copies_image_and_writes_placeholder(isolated_root, tmp_path):
    src = tmp_path / "cup.png"
    src.write_bytes(b"fake-png")
    path = custom_data.create("看图点单")
    custom_data.append_pair(path, "图里是什么？", "一杯咖啡。", image=src)
    obj = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert obj["messages"][0]["content"].startswith("<|image_pad|>")
    assert Path(obj["image"]).is_file()
    assert Path(obj["image"]).parent.parent.name == custom_data.MEDIA_DIR


def test_custom_pair_rejects_both_media(isolated_root, tmp_path):
    path = custom_data.create("混合")
    image = tmp_path / "a.png"
    audio = tmp_path / "a.wav"
    image.write_bytes(b"x")
    audio.write_bytes(b"y")
    with pytest.raises(custom_data.CustomDataError, match="一种媒体"):
        custom_data.append_pair(path, "q", "a", image=image, audio=audio)
    assert custom_data.count(path) == 0


def test_custom_dataset_rename(isolated_root):
    path = custom_data.create("旧名字")
    custom_data.append_pair(path, "问", "答")
    dst = custom_data.rename(path, "新名字")
    assert dst.is_file() and not path.exists()
    assert custom_data.count(dst) == 1
    assert json.loads(Path(str(dst) + ".meta.json").read_text(encoding="utf-8"))["name"] == dst.name


def test_dataset_split_preserves_media(isolated_root, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    src = isolated_root / "data" / "source.jsonl"
    record = {"messages": [{"role": "user", "content": "<|image_pad|>看图"},
                           {"role": "assistant", "content": "图"}],
              "image": str(img)}
    src.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    train, val = ds.split(src, isolated_root / "data" / "tr.jsonl",
                          isolated_root / "data" / "va.jsonl", ratio=0)
    assert (train, val) == (1, 0)
    out = json.loads((isolated_root / "data" / "tr.jsonl").read_text(encoding="utf-8"))
    assert out["image"] == str(img)
