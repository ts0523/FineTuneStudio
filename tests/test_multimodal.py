"""P0-5 / P1-8：多模态占位符数量与编码器特征严格一致。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import mm_encoder
import mm_support

ROOT = Path(__file__).resolve().parents[1]


def test_placeholder_expands_to_encoder_rows():
    img = mm_support.expand_placeholder("<|image_pad|>看图", "image")
    aud = mm_support.expand_placeholder("<|audio_pad|>听语音", "audio")
    assert img.count("<|image_pad|>") == mm_encoder.VISION_TOKENS
    assert aud.count("<|audio_pad|>") == mm_encoder.AUDIO_TOKENS
    # 已有多余占位符也会被归一化，不会翻倍
    img2 = mm_support.expand_placeholder("<|image_pad|>" * 3 + "看图", "image")
    assert img2.count("<|image_pad|>") == mm_encoder.VISION_TOKENS
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "没有占位符"},
            {"role": "assistant", "content": "a"}]
    out = mm_support.expand_messages(msgs, "audio")
    assert out[1]["content"].count("<|audio_pad|>") == mm_encoder.AUDIO_TOKENS
    assert out[0]["content"] == "sys"


class _Tok:
    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|image_pad|>": 5, "<|audio_pad|>": 6}[token]


class _Host(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, mm_support.LLM_DIM)


def _adapter():
    return mm_support.MMAdapter(_Host(), _Tok())


def test_adapter_rejects_count_mismatch():
    ad = _adapter()
    ids = torch.tensor([[1, 5, 5, 1]])          # 2 个占位符
    ad.set_features([torch.zeros(3, mm_support.VISION_DIM)])   # 3 行特征
    with pytest.raises(ValueError, match="占位符"):
        ad.host.embed_tokens(ids)


def test_adapter_accepts_exact_count():
    ad = _adapter()
    ids = torch.tensor([[1] + [5] * mm_encoder.VISION_TOKENS + [1]])
    ad.set_features([torch.zeros(mm_encoder.VISION_TOKENS, mm_support.VISION_DIM)])
    out = ad.host.embed_tokens(ids)
    assert out.shape[0] == 1


def test_adapter_injects_image_and_audio_together():
    ad = _adapter()
    ids = torch.tensor([[1] + [5] * mm_encoder.VISION_TOKENS
                        + [6] * mm_encoder.AUDIO_TOKENS + [1]])
    ad.set_features([{
        "image": torch.ones(mm_encoder.VISION_TOKENS, mm_support.VISION_DIM),
        "audio": torch.ones(mm_encoder.AUDIO_TOKENS, mm_support.AUDIO_DIM),
    }])
    out = ad.host.embed_tokens(ids)
    assert out.shape == (1, ids.shape[1], mm_support.LLM_DIM)


def test_composite_projector_loads_selected_modalities(tmp_path):
    base = _adapter()
    image_ckpt = tmp_path / "image.pt"
    audio_ckpt = tmp_path / "audio.pt"
    torch.save({"vision": base.proj_v.state_dict(), "audio": base.proj_a.state_dict(),
                "meta": {"stats": {"image": 3, "audio": 0}}}, image_ckpt)
    torch.save({"vision": base.proj_v.state_dict(), "audio": base.proj_a.state_dict(),
                "meta": {"stats": {"image": 0, "audio": 2}}}, audio_ckpt)
    ad = mm_support.MMAdapter.load_composite(base.model, _Tok(), [image_ckpt, audio_ckpt])
    assert ad is not None
    assert set(ad.loaded_modalities) == {"image", "audio"}


def test_adapter_modalities_reads_train_meta(tmp_path):
    from studio.core import adapters

    out = tmp_path / "adapter"
    out.mkdir()
    (out / "mm_projector.pt").write_bytes(b"x")
    (out / "train_meta.json").write_text(
        json.dumps({"multimodal": {"stats": {"image": 4, "audio": 0}}}), encoding="utf-8")
    assert adapters.adapter_modalities(out) == ["image"]


@pytest.mark.skipif(not (ROOT / "models" / "_encoders" / "whisper-tiny").is_dir()
                    or not (ROOT / "data" / "mm_data" / "images").is_dir(),
                    reason="缺少 whisper-tiny 编码器或已清空多模态样本")
def test_single_image_and_audio_smoke():
    imgs = sorted((ROOT / "data" / "mm_data" / "images").glob("*.png"))
    wavs = sorted((ROOT / "data" / "mm_data" / "audio").glob("*.wav"))
    assert imgs and wavs, "缺少现成的图像/语音冒烟样本"

    vmodel, preprocess = mm_encoder.load_vision_encoder()
    feat = mm_encoder.encode_image(vmodel, preprocess, imgs[0])
    assert feat.shape == (mm_encoder.VISION_TOKENS, mm_encoder.VISION_DIM)

    amodel = mm_encoder.load_audio_encoder(ROOT / "models" / "_encoders" / "whisper-tiny")
    feat2 = mm_encoder.encode_audio(amodel, wavs[0])
    assert feat2.shape == (mm_encoder.AUDIO_TOKENS, mm_encoder.AUDIO_DIM)


def test_auto_asr_prefers_whisper(monkeypatch, isolated_root):
    from studio.core import audio

    calls = []
    monkeypatch.setattr(audio, "asr_whisper", lambda _wav: calls.append("whisper") or ("中文结果", ""))
    monkeypatch.setattr(audio, "asr_sapi", lambda _wav: calls.append("sapi") or ("错误结果", 0.99))

    text, engine, error = audio.transcribe(Path("sample.wav"), engine="auto")

    assert (text, error) == ("中文结果", "")
    assert engine == "Whisper（本地）"
    assert calls == ["whisper"]


def test_auto_asr_falls_back_to_sapi(monkeypatch, isolated_root):
    from studio.core import audio

    monkeypatch.setattr(audio, "asr_whisper", lambda _wav: ("", "Whisper 不可用"))
    monkeypatch.setattr(audio, "asr_sapi", lambda _wav: ("系统结果", 0.4))

    text, engine, error = audio.transcribe(Path("sample.wav"), engine="auto")

    assert (text, error) == ("系统结果", "")
    assert engine == "系统识别（回退）"
