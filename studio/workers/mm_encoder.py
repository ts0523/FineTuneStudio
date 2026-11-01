# -*- coding: utf-8 -*-
"""多模态编码器：轻量视觉（MobileNetV3-Small）+ 轻量音频（Whisper-tiny encoder）。

设计取舍（为什么要"冻结 + 预缓存"）：
- 本机 torch 是 CPU 版，Intel 核显（UHD 730）在 PyTorch 里不可用（实测 torch.xpu.is_available()=False），
  所以不指望 GPU 加速；
- 编码器参数量小（视觉 2.5M、音频 8M）且**冻结**，每条样本只需前向一次；
  把特征缓存成 .npz 后，训练时只跑投影层 + LoRA，速度与纯文本训练几乎一致。

输出的 token 数与 MiniMind 的配置对齐：
- 视觉：4x4=16 个 token（768 维），占位符 <|image_pad|>
- 音频：32 个 token（768 维），占位符 <|audio_pad|>
"""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import torch

VISION_TOKENS = 49          # MobileNet 原生 7x7，保留局部颜色信息
AUDIO_TOKENS = 32
VISION_DIM = 576            # MobileNetV3-Small 特征维度
AUDIO_DIM = 384             # Whisper-tiny encoder 维度
LLM_DIM = 768               # MiniMind 主干 hidden


# ---------------------------------------------------------------- 视觉
def load_vision_encoder(cache_dir: Path | None = None):
    """返回 (model, preprocess)。MobileNetV3-Small，ImageNet 预训练，2.5M 参数。"""
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights).features.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, weights.transforms()


@torch.no_grad()
def encode_image(model, preprocess, image_path: str | Path) -> np.ndarray:
    """图片 → [VISION_TOKENS, VISION_DIM] 特征。"""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    feats = model(x)                       # [1, 576, 7, 7]
    return feats[0].flatten(1).transpose(0, 1).numpy().astype(np.float32)   # [49, 576]


# ---------------------------------------------------------------- 音频
def load_audio_encoder(model_dir: str | Path):
    """Whisper-tiny 的 encoder 部分（8M 参数），输入 log-mel。"""
    from transformers import WhisperModel

    model = WhisperModel.from_pretrained(str(model_dir),
                                         local_files_only=True).encoder.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        n = f.getnframes()
        ch = f.getnchannels()
        raw = f.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def resample(data: np.ndarray, sr: int, target: int = 16000) -> np.ndarray:
    if sr == target:
        return data
    n = int(round(len(data) * target / sr))
    x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)


def log_mel(data: np.ndarray, sr: int = 16000, n_mel: int = 80, n_frames: int = 3000):
    """简易 log-mel：25ms 窗 / 10ms 步，与 Whisper 的输入形状对齐（80 x 3000）。"""
    if sr != 16000:
        data = resample(data, sr, 16000)
    win, hop = 400, 160
    if len(data) < win:
        data = np.pad(data, (0, win - len(data)))
    frames = 1 + (len(data) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(frames)[:, None]
    spec = np.abs(np.fft.rfft(data[idx] * np.hanning(win), axis=1)) ** 2      # [T, 201]

    # mel 滤波器组（HTK 近似）
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(0), hz2mel(sr / 2), n_mel + 2)
    bins = np.floor((win + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mel, spec.shape[1]), dtype=np.float32)
    for i in range(n_mel):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center > left:
            fb[i, left:center] = (np.arange(left, center) - left) / max(1, center - left)
        if right > center:
            fb[i, center:right] = ((right - np.arange(center, right))
                                   / max(1, right - center))
    mel = np.log10(np.maximum(spec @ fb.T, 1e-10))
    mel = (mel + 4.0) / 4.0                       # 粗略归一化到 [-1,1]
    if mel.shape[0] < n_frames:
        mel = np.pad(mel, ((0, n_frames - mel.shape[0]), (0, 0)))
    else:
        mel = mel[:n_frames]
    return mel.astype(np.float32)                 # [3000, 80]


@torch.no_grad()
def encode_audio(model, wav_path: str | Path) -> np.ndarray:
    data, sr = read_wav(wav_path)
    mel = log_mel(data, sr)                       # [3000, 80]
    x = torch.from_numpy(mel).transpose(0, 1).unsqueeze(0)   # [1, 80, 3000]
    feats = model(x).last_hidden_state            # [1, 1500, 384]
    feats = torch.nn.functional.adaptive_avg_pool1d(feats.transpose(1, 2), AUDIO_TOKENS).transpose(1, 2)
    return feats[0].numpy().astype(np.float32)    # [32, 384]


# ---------------------------------------------------------------- 投影层
class MMProjector(torch.nn.Module):
    """把编码器特征映射到 LLM 的 hidden 空间，可训练。"""

    def __init__(self, in_dim: int, out_dim: int = LLM_DIM):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.GELU(),
            torch.nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------- 特征缓存
def extract_features(manifest: list[dict], out_dir: Path, log=print,
                     audio_dir: Path | None = None) -> dict:
    """给 manifest 里每条样本算特征并缓存。返回 {sample_id: 相对路径}。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = audio_dir or Path("models") / "_encoders" / "whisper-tiny"
    vision = audio = None
    cache: dict[str, str] = {}
    for i, item in enumerate(manifest, 1):
        sid = str(item.get("id") or i)
        npz = out_dir / f"{sid}.npz"
        payload = {}
        if item.get("image"):
            if vision is None:
                log("加载视觉编码器 MobileNetV3-Small（2.5M）")
                vision = load_vision_encoder()
            payload["image"] = encode_image(vision[0], vision[1], item["image"])
        if item.get("audio"):
            if audio is None:
                log("加载音频编码器 Whisper-tiny encoder")
                audio = load_audio_encoder(audio_dir)
            payload["audio"] = encode_audio(audio, item["audio"])
        if payload:
            np.savez_compressed(npz, **payload)
            cache[sid] = npz.name
        if i % 10 == 0:
            log(f"特征提取 {i}/{len(manifest)}")
    return cache


def describe() -> dict:
    return {"vision": {"encoder": "MobileNetV3-Small", "params_m": 2.5, "tokens": VISION_TOKENS,
                       "dim": VISION_DIM},
            "audio": {"encoder": "Whisper-tiny encoder", "params_m": 8.0, "tokens": AUDIO_TOKENS,
                      "dim": AUDIO_DIM},
            "llm_dim": LLM_DIM}


if __name__ == "__main__":
    print(json.dumps(describe(), ensure_ascii=False, indent=1))
    print("mel frames", log_mel(np.zeros(16000, dtype=np.float32)).shape)
    print("sqrt", math.sqrt(768))
