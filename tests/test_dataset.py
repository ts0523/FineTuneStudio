"""P2-15/16：数据角色校验、坏行计数、原子写入回滚、保存后复核。"""
from __future__ import annotations

import json

import pytest

from studio.core import dataset as ds, fsutil, nl


def _line(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def test_unknown_role_rejected_not_silently_user():
    with pytest.raises(ds.DataError):
        ds.normalize({"messages": [{"role": "robot", "content": "hi"},
                                   {"role": "assistant", "content": "yo"}]})
    with pytest.raises(ds.DataError):
        ds.normalize({"messages": [{"role": "user", "content": "hi"},
                                   {"role": "assistant", "content": "yo"},
                                   {"role": "tool", "content": "x"}]})


def test_last_message_must_be_assistant():
    with pytest.raises(ds.DataError):
        ds.normalize({"messages": [{"role": "user", "content": "q"}]})
    with pytest.raises(ds.DataError):
        ds.normalize({"messages": [{"role": "system", "content": "s"},
                                   {"role": "user", "content": "q"},
                                   {"role": "assistant", "content": "a"},
                                   {"role": "user", "content": "q2"}]})


def test_system_and_alternation_order():
    with pytest.raises(ds.DataError):
        ds.normalize({"messages": [{"role": "user", "content": "q"},
                                   {"role": "system", "content": "late"},
                                   {"role": "assistant", "content": "a"}]})
    with pytest.raises(ds.DataError):
        ds.normalize({"messages": [{"role": "user", "content": "q1"},
                                   {"role": "user", "content": "q2"},
                                   {"role": "assistant", "content": "a"}]})
    assert ds.normalize({"messages": [{"role": "system", "content": "s"},
                                      {"role": "user", "content": "q"},
                                      {"role": "assistant", "content": "a"},
                                      {"role": "user", "content": "q2"},
                                      {"role": "assistant", "content": "a2"}]})[-1]["role"] == "assistant"


def test_validate_counts_bad_empty_duplicate(isolated_root):
    p = isolated_root / "data" / "bad.jsonl"
    lines = [
        _line({"messages": [{"role": "user", "content": "q1"},
                            {"role": "assistant", "content": "a1"}]}),
        _line({"messages": [{"role": "user", "content": "q1"},
                            {"role": "assistant", "content": "a1"}]}),
        _line({"messages": [{"role": "user", "content": "q2"},
                            {"role": "assistant", "content": " "}]}),
        _line({"messages": [{"role": "user", "content": "q3"},
                            {"role": "robot", "content": "a3"}]}),
        "{not json",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = ds.validate(p)
    assert rep.total == 5
    assert rep.valid == 1
    assert rep.empty == 1
    assert rep.duplicates == 1
    assert len(rep.issues) >= 3
    chk = ds.verify_after_write(p)
    assert chk["written"] == 5
    assert chk["valid"] == 1
    assert chk["bad"] == 2          # 非 JSON + 未知角色
    assert chk["empty"] == 1
    assert chk["dup"] == 1


def test_multimodal_same_text_different_media_not_duplicate(tmp_path):
    p = tmp_path / "mm.jsonl"
    lines = [
        _line({"messages": [{"role": "user", "content": "<|image_pad|>图里是什么？"},
                            {"role": "assistant", "content": "红色圆形。"}],
               "image": "a.png"}),
        _line({"messages": [{"role": "user", "content": "<|image_pad|>图里是什么？"},
                            {"role": "assistant", "content": "红色圆形。"}],
               "image": "b.png"}),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = ds.validate(p)
    assert rep.valid == 2
    assert rep.duplicates == 0


def test_atomic_write_failure_keeps_original(isolated_root, monkeypatch):
    p = isolated_root / "data" / "keep.jsonl"
    p.write_text("OLD\n", encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(fsutil.os, "replace", boom)
    with pytest.raises(OSError):
        fsutil.atomic_write_text(p, "NEW\n")
    assert p.read_text(encoding="utf-8") == "OLD\n"
    assert not list((isolated_root / "data").glob(".fts-*.tmp"))


def test_clean_reports_and_verifies(isolated_root):
    p = isolated_root / "data" / "src.jsonl"
    lines = [
        _line({"messages": [{"role": "user", "content": "q"},
                            {"role": "assistant", "content": "a"}]}),
        _line({"messages": [{"role": "user", "content": "q"},
                            {"role": "assistant", "content": "a"}]}),
        _line({"messages": [{"role": "user", "content": "q2"},
                            {"role": "assistant", "content": ""}]}),
        _line({"messages": [{"role": "user", "content": "q3"},
                            {"role": "assistant", "content": "a" * 500}]}),
        "{oops",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = isolated_root / "data" / "clean.jsonl"
    stat = ds.clean(p, out, max_chars=200, dedupe=True, drop_empty=True)
    assert stat["in"] == 5
    assert stat["dup"] == 1
    assert stat["empty"] == 1
    assert stat["too_long"] == 1
    assert stat["bad"] == 1
    assert stat["kept"] == 1
    assert stat["verified"]["valid"] == 1


def test_append_atomic_failure_keeps_original(isolated_root, monkeypatch):
    p = isolated_root / "data" / "acc.jsonl"
    p.write_text(_line({"messages": [{"role": "user", "content": "q"},
                                     {"role": "assistant", "content": "a"}]}) + "\n",
                 encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(fsutil.os, "replace", boom)
    with pytest.raises(OSError):
        nl.append_to(p, [{"messages": [{"role": "user", "content": "q2"},
                                       {"role": "assistant", "content": "a2"}]}])
    assert "q2" not in p.read_text(encoding="utf-8")
