"""合并 worker 主流程：文件写出后重新加载/生成失败，必须保留失败终态。"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import torch

import custom_model
import merge_worker


class _Base:
    config = types.SimpleNamespace(use_cache=False)


class _Merged:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(use_cache=False)

    def merge_and_unload(self):
        return self

    def save_pretrained(self, out, safe_serialization=True) -> None:
        Path(out, "model.bin").write_bytes(b"merged-weight")

    def eval(self):
        return self

    def generate(self, **_kwargs):
        raise RuntimeError("验证生成失败（故障注入）")


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def save_pretrained(self, out) -> None:
        Path(out, "tokenizer_config.json").write_text("{}", encoding="utf-8")

    def apply_chat_template(self, *_args, **_kwargs) -> str:
        return "用户：你好\n助手："

    def __call__(self, *_args, **_kwargs) -> dict:
        return {"input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]])}

    def decode(self, *_args, **_kwargs) -> str:
        return ""


def test_merge_writes_file_then_verify_failure_is_nonzero(tmp_path, monkeypatch, capsys):
    base = tmp_path / "models" / "base"
    adapter = tmp_path / "models" / "adapter"
    base.mkdir(parents=True)
    adapter.mkdir(parents=True)
    (base / "config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    class FakePeft:
        @staticmethod
        def from_pretrained(_base, _adapter):
            return _Merged()

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeft))

    def load(path, _dtype, note=None):
        if Path(path).resolve() == base.resolve():
            return _Base(), _Tokenizer(), "transformers"
        return _Merged(), _Tokenizer(), "transformers"

    monkeypatch.setattr(custom_model, "load_causal_lm", load)
    cfg_path = tmp_path / "merge.json"
    out = tmp_path / "outputs" / "merged"
    cfg_path.write_text(json.dumps({
        "model": str(base), "adapter": str(adapter), "out": str(out), "verify": True,
    }), encoding="utf-8")
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["merge_worker.py", "--config", str(cfg_path)])

    code = merge_worker.main()

    assert code == 5
    assert (out / "model.bin").is_file(), "验证失败时仍应保留已写出的合并产物"
    events = []
    for line in capsys.readouterr().out.splitlines():
        if line.startswith(merge_worker.EVT):
            events.append(json.loads(line[len(merge_worker.EVT):]))
    assert any(e.get("type") == "merged" for e in events)
    assert any(e.get("type") == "verified" and e.get("ok") is False for e in events)
    done = [e for e in events if e.get("type") == "done"]
    assert done and done[-1]["status"] == "failed"
    assert done[-1]["written"] is True
    assert done[-1]["verified"] is False
