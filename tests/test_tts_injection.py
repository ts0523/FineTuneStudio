"""P0-4：TTS 批量脚本不得被数据集文本/音色/路径注入。"""
from __future__ import annotations

import shutil

import pytest

from studio.core import audio, multimodal as mm, paths

powershell = shutil.which("powershell")
pytestmark = pytest.mark.skipif(powershell is None, reason="需要 Windows PowerShell")


def test_audio_ps_literal_escapes_quotes():
    assert audio._ps_literal("C:/a'b.wav") == "'C:/a''b.wav'"
    assert audio._ps_literal("x\ny") == "'x\ny'"


def test_malicious_text_is_data_not_code(isolated_root):
    canary = isolated_root / "pwned.txt"
    text_evil = f"$([char]36)(New-Item -LiteralPath '{canary}' -ItemType File)"
    texts = [
        text_evil,
        "反引号 `whoami` 与双引号 \"x\"、单引号 'y'",
        "第一行\n第二行\n$(whoami)",
    ]
    out_dir = isolated_root / "audio 特殊 ' 目录"
    tmp_holder = isolated_root / "含 ` 反引号 的目录"
    tmp_holder.mkdir(parents=True, exist_ok=True)
    files, err = mm.synthesize(texts, out_dir, "")
    assert err == ""
    assert files and all(f is not None and f.stat().st_size > 2000 for f in files)
    assert not canary.exists(), "文本被执行成了命令（脚本注入未修复）"

    # 脚本本身必须固定，不含任何用户文本
    script = (paths.state_dir() / "_tts_batch.ps1").read_text(encoding="utf-8-sig")
    assert "New-Item" not in script
    assert "反引号" not in script
    assert str(out_dir) not in script
    # 临时任务 JSON 用完即删
    assert not list(paths.state_dir().glob("_tts_job_*.json"))


def test_tts_reports_nonzero_failure(isolated_root):
    out_dir = isolated_root / "audio-fail"
    files, err = mm.synthesize(["测试失败上报"], out_dir, "NoSuchVoice-DO-NOT-EXIST")
    assert files == [None]
    assert "code=" in err or "失败" in err


def test_voice_name_is_data_not_code(isolated_root):
    canary = isolated_root / "voice_pwned.txt"
    evil_voice = f"x'$([char]36)(New-Item -LiteralPath '{canary}')'"
    files, err = mm.synthesize(["音色名测试"], isolated_root / "audio-voice", evil_voice)
    assert not canary.exists()
    assert files == [None]
    assert err
