"""真实环境验收：默认跳过，设置 FTS_LIVE_ACCEPTANCE=1 后才发起真实请求/合并。

覆盖：
- OpenRouter 真实 AI 造数与评分（失败时 judge 会本地兜底，测试要求必须有 ai+local）
- hf-mirror 真实仓库清单与文件下载（无 mock）
- 真实 MiniMind-O + 真实 LoRA 合并，随后注入生成失败，验收 failed/nonzero 终态
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

import custom_model
import download_worker
import merge_worker
from studio.core import ai, autotrain, nl

LIVE = os.environ.get("FTS_LIVE_ACCEPTANCE") == "1"
pytestmark = [pytest.mark.live, pytest.mark.skipif(not LIVE, reason="需要 FTS_LIVE_ACCEPTANCE=1")]


def test_live_openrouter_generate_and_judge(monkeypatch):
    assert ai.enabled(), "需要可用 OpenRouter Key（环境变量或 DPAPI）"
    monkeypatch.setattr(ai, "_bump", lambda *a, **k: None)
    monkeypatch.setattr(ai, "_mark_blocked", lambda *a, **k: None)

    samples, model = nl.generate(
        "生成咖啡店点单问答，问题具体，答案不超过两句并包含价格。",
        4, kind="qa", lang="中文", batch=4)
    assert len(samples) >= 4
    assert model, "必须记录真实造数模型名"

    pairs = []
    for sample in samples[:3]:
        msgs = sample["messages"]
        pairs.append({
            "q": next(m["content"] for m in msgs if m["role"] == "user"),
            "a": next(m["content"] for m in msgs if m["role"] == "assistant"),
        })
    result = autotrain.judge(pairs, [p["a"] for p in pairs], "咖啡店点单")
    assert result["judge_source"] == "ai+local", f"AI 评分失败并落到本地兜底：{result}"
    assert result["ai_model"]


def test_live_download_from_hf_mirror(isolated_root, monkeypatch, capsys):
    dest = isolated_root / "models" / "tiny-random-gpt2"
    cfg = isolated_root / "download.json"
    cfg.write_text(json.dumps({
        "repo": "hf-internal-testing/tiny-random-gpt2",
        "dest": str(dest),
        "endpoint": "https://hf-mirror.com",
        "revision": "main",
        "allow": ["config.json"],
        "ignore": ["*.never"],
        "verify_sha": True,
        "cancel_flag": str(dest / ".cancel"),
        "allow_code": False,
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["download_worker.py", "--config", str(cfg)])

    assert download_worker.main() == 0
    downloaded = dest / "config.json"
    assert downloaded.is_file()
    assert json.loads(downloaded.read_text(encoding="utf-8"))
    events = [json.loads(line[len(download_worker.EVT):])
              for line in capsys.readouterr().out.splitlines()
              if line.startswith(download_worker.EVT)]
    assert any(e.get("type") == "done" for e in events)


def test_live_merge_then_verification_failure(project_root, tmp_path, monkeypatch, capsys):
    base = project_root / "models" / "minimind-3o"
    if not (base / "config.json").is_file():
        pytest.skip("缺少真实 MiniMind-O 基座")
    adapters = [p for p in (project_root / "outputs").iterdir()
                if p.is_dir() and (p / "adapter_config.json").is_file()]
    if not adapters:
        pytest.skip("缺少真实 LoRA 适配器产物")
    adapter = adapters[0]
    out = tmp_path / "outputs" / "merged-verify-failure"
    cfg = tmp_path / "merge.json"
    cfg.write_text(json.dumps({
        "model": str(base), "adapter": str(adapter), "out": str(out), "verify": True,
    }), encoding="utf-8")
    monkeypatch.setenv("FTS_ROOT", str(tmp_path))
    original = custom_model.load_causal_lm

    def load_with_verify_failure(path, dtype=None, note=None):
        model, tok, kind = original(path, dtype, note)
        if Path(path).resolve() == out.resolve():
            def fail(**_kwargs):
                raise RuntimeError("真实合并产物加载成功，但生成验证失败（故障注入）")
            model.generate = fail
        return model, tok, kind

    monkeypatch.setattr(custom_model, "load_causal_lm", load_with_verify_failure)
    monkeypatch.setattr(sys, "argv", ["merge_worker.py", "--config", str(cfg)])
    try:
        code = merge_worker.main()
    finally:
        # 真实合并产物接近 256 MB，验收后立即清理临时目录。
        shutil.rmtree(out, ignore_errors=True)

    assert code == 5
    events = [json.loads(line[len(merge_worker.EVT):])
              for line in capsys.readouterr().out.splitlines()
              if line.startswith(merge_worker.EVT)]
    assert any(e.get("type") == "merged" for e in events)
    assert any(e.get("type") == "verified" and e.get("ok") is False for e in events)
    assert any(e.get("type") == "done" and e.get("status") == "failed"
               for e in events)
