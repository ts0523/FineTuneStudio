# -*- coding: utf-8 -*-
"""音色克隆（本地轻量版）。

没有 GPU、也不想让用户为 TTS 模型再装 2 GB 依赖，所以走「声学特征贴合」：
- 录一段 6~8 秒的样本，估计用户的中位基频（F0）与说话速度；
- 用系统 SAPI 合成一段校准语音，估计底音色的 F0 与语速；
- 把比值换算成 SSML 的 pitch / rate 百分比，保存成音色档案；
- 朗读时用档案里的 pitch / rate，让系统音色听起来更接近用户。
诚实地说：它克隆的是**音高与语速**，不是声纹本身。真要 1:1 声纹需要
CosyVoice/GPT-SoVITS 这类模型（数百 MB~数 GB），不适合随 exe 自带。
"""
from __future__ import annotations

import json
import shutil
import time
import wave
from pathlib import Path

from . import audio, paths

CALIB_TEXT = "你好，这是音色校准，今天天气不错，我们出去走走吧。"
CALIB_FILE = "voice_calibration.json"


def _root() -> Path:
    d = paths.state_dir() / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _profiles_path() -> Path:
    return paths.state_dir() / "voice_profiles.json"


# ---------------------------------------------------------------- 读写
def load() -> list[dict]:
    p = _profiles_path()
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
        return [x for x in (raw or []) if isinstance(x, dict) and x.get("id")]
    except Exception:  # noqa: BLE001
        return []


def save(items: list[dict]) -> None:
    try:
        _profiles_path().write_text(json.dumps(items, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    except OSError:
        pass


def find(pid: str) -> dict | None:
    return next((x for x in load() if x.get("id") == pid), None)


def remove(pid: str) -> None:
    items = [x for x in load() if x.get("id") != pid]
    save(items)


# ---------------------------------------------------------------- 声学特征
def duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate() or 1
            return w.getnframes() / float(rate)
    except Exception:  # noqa: BLE001
        return 0.0


def estimate_f0(path: Path) -> float:
    """中位基频（Hz）。用自相关，60~400 Hz；读不到返回 0。"""
    try:
        import numpy as np
        with wave.open(str(path), "rb") as w:
            rate, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(n)
    except Exception:  # noqa: BLE001
        return 0.0
    try:
        if width == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        elif width == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) * 256
        elif width == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 65536
        else:
            return 0.0
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        frame = int(rate * 0.05)
        hop = int(rate * 0.02)
        lo, hi = int(rate / 400), int(rate / 60)
        vals = []
        for start in range(0, max(1, len(data) - frame), hop):
            seg = data[start:start + frame]
            if float(np.sqrt((seg ** 2).mean())) < 250:
                continue
            seg = seg - seg.mean()
            corr = np.correlate(seg, seg, "full")[frame - 1:]
            if hi >= len(corr):
                continue
            idx = lo + int(np.argmax(corr[lo:hi]))
            if corr[idx] > 0.25 * corr[0]:
                vals.append(rate / idx)
        if not vals:
            return 0.0
        vals.sort()
        mid = len(vals) // 2
        return float(vals[mid])
    except Exception:  # noqa: BLE001
        return 0.0


def _calibration(voice: str) -> dict:
    """底音色的 F0 / 语速，缓存到 state/voice_calibration.json。"""
    path = paths.state_dir() / CALIB_FILE
    cache: dict = {}
    try:
        if path.is_file():
            cache = json.loads(path.read_text(encoding="utf-8-sig")) or {}
    except Exception:  # noqa: BLE001
        cache = {}
    key = voice or audio.pick_voice() or "default"
    if key in cache:
        return cache[key]
    wav = audio.speak_wav(CALIB_TEXT, voice=voice, rate=0)
    f0 = estimate_f0(wav) if wav else 0.0
    secs = duration(wav) if wav else 0.0
    info = {"f0": f0, "seconds": secs,
            "chars_per_sec": round(len(CALIB_TEXT) / secs, 3) if secs > 0.2 else 0.0}
    cache[key] = info
    try:
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
    return info


def clone(name: str, sample: Path, sample_text: str = "", voice: str = "") -> tuple[dict | None, str]:
    """用一段录音建音色档案。返回 (档案, 错误)。"""
    sample = Path(sample)
    if not sample.is_file():
        return None, "录音文件不存在"
    dur = duration(sample)
    if dur < 1.0:
        return None, "样本太短，至少说 3 秒"
    base = _calibration(voice)
    user_f0 = estimate_f0(sample)
    base_f0 = float(base.get("f0") or 0)
    pitch = 0
    if user_f0 and base_f0:
        ratio = user_f0 / base_f0
        # 实测 SSML +25% 约等于 +46% F0；负值在 SAPI 上几乎不降，故只向上贴
        pitch = int(max(0, min(55, round((ratio - 1) * 55))))
    rate_pct = 0
    base_cps = float(base.get("chars_per_sec") or 0)
    if sample_text.strip() and base_cps > 0:
        user_cps = len(sample_text.strip()) / max(0.4, dur)
        rate_pct = int(max(-35, min(30, round((user_cps / base_cps - 1) * 100))))
    pid = f"v{int(time.time())}"
    dst = _root() / f"{pid}.wav"
    try:
        shutil.copy2(sample, dst)
    except OSError as exc:
        return None, f"保存样本失败：{exc}"
    prof = {
        "id": pid, "name": (name or "我的音色").strip()[:20], "voice": voice or audio.pick_voice(),
        "pitch": pitch, "rate_pct": rate_pct,
        "f0": round(user_f0, 1), "base_f0": round(base_f0, 1),
        "seconds": round(dur, 2), "sample": str(dst),
        "sample_text": sample_text.strip()[:200], "created": time.time(),
    }
    items = load()
    items.insert(0, prof)
    save(items[:20])
    return prof, ""


def speak(text: str, profile: dict | None = None, rate: int = 0) -> Path | None:
    """按档案朗读，返回 wav（失败返回 None）。"""
    if not text.strip():
        return None
    if profile:
        return audio.speak_wav(text, voice=str(profile.get("voice") or ""),
                               rate=rate, pitch=int(profile.get("pitch") or 0),
                               rate_pct=int(profile.get("rate_pct") or 0))
    return audio.speak_wav(text, rate=rate)
