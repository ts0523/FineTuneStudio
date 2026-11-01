# -*- coding: utf-8 -*-
"""多模态数据生成：合成图像集 + 用系统 TTS 把文本问答转成语音问答。

为什么这样造数据：
- 语音：直接用 Windows 自带 SAPI（Microsoft Huihui，zh-CN）把**你自己的文本数据**里
  的问题读出来，得到「语音问题 → 文本答案」的真实训练样本，不用外部数据集。
- 图像：没有图文语料，就用程序合成「形状 × 颜色 × 数量」的图，配一句标准答案。
  这是教学级数据，目的是让模型学会"看图回答"这件事本身。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from . import fsutil, paths

MM_DIR = "mm_data"
IMG_SUBDIR = "images"
WAV_SUBDIR = "audio"

SHAPES = [("圆形", "ellipse"), ("方形", "rect"), ("三角形", "tri")]
COLORS = [("红色", (214, 48, 48)), ("蓝色", (48, 84, 214)), ("绿色", (40, 168, 92)),
          ("黄色", (232, 186, 40))]
COUNTS = [1, 2]

# 固定脚本：不包含任何用户文本/音色/路径的插值；数据只走独立 UTF-8 JSON 文件。
_TTS_SCRIPT = r"""param([string]$JobFile)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$raw = [System.IO.File]::ReadAllText($JobFile, [System.Text.Encoding]::UTF8)
$job = $raw | ConvertFrom-Json
$resultPath = [string]$job.result
$rows = @()
$failed = 0
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  if ([string]$job.voice) { $synth.SelectVoice([string]$job.voice) }
  foreach ($item in @($job.items)) {
    $ok = $true
    $err = ''
    try {
      $synth.SetOutputToWaveFile([string]$item.out)
      $synth.Speak([string]$item.text)
      $synth.SetOutputToNull()
    } catch {
      $ok = $false
      $err = $_.Exception.Message
      $failed = $failed + 1
    }
    $rows += [pscustomobject]@{ index = [int]$item.index; ok = $ok; error = $err }
  }
} finally {
  $synth.Dispose()
}
$payload = [pscustomobject]@{ items = $rows; failed = $failed }
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($resultPath, ($payload | ConvertTo-Json -Depth 5 -Compress), $utf8)
if ($failed -gt 0) { exit 1 }
exit 0
"""



def mm_root() -> Path:
    d = paths.data_dir() / MM_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _draw(draw, kind: str, box, color) -> None:
    x0, y0, x1, y1 = box
    if kind == "ellipse":
        draw.ellipse(box, fill=color)
    elif kind == "rect":
        draw.rectangle(box, fill=color)
    else:
        draw.polygon([(x0, y1), (x1, y1), ((x0 + x1) // 2, y0)], fill=color)


def make_image_dataset(per_combo: int = 6, seed: int = 0) -> tuple[Path, int]:
    """合成图像问答集：位置、大小、背景都随机，逼模型看「形状/颜色/数量」而不是背句式。

    per_combo 越大，同一组合的变体越多，模型越难靠"记住答案分布"蒙过去。
    """
    import random

    from PIL import Image, ImageDraw

    rnd = random.Random(seed)
    out_dir = mm_root() / IMG_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    idx = 0
    for sname, skind in SHAPES:
        for cname, cval in COLORS:
            for count in COUNTS:
                for _rep in range(per_combo):
                    idx += 1
                    im = Image.new("RGB", (224, 224),
                                   rnd.choice([(255, 255, 255), (246, 248, 252), (238, 240, 238)]))
                    dr = ImageDraw.Draw(im)
                    boxes = []
                    for _ in range(count):
                        size = rnd.randint(64, 108)
                        x = rnd.randint(8, 224 - size - 8)
                        y = rnd.randint(8, 224 - size - 8)
                        for _try in range(12):          # 避免两个图形重叠
                            if all(abs(x - bx) > size * 0.8 or abs(y - by) > size * 0.8
                                   for bx, by, _b in boxes):
                                break
                            x = rnd.randint(8, 224 - size - 8)
                            y = rnd.randint(8, 224 - size - 8)
                        boxes.append((x, y, size))
                        _draw(dr, skind, (x, y, x + size, y + size), cval)
                    p = out_dir / f"{sname}{cname}{count}_{idx:03d}.png"
                    im.save(p)
                    amount = "一个" if count == 1 else f"{count}个"
                    rows.append({
                        "messages": [
                            {"role": "user", "content": "<|image_pad|>图里是什么形状、什么颜色、有几个？"},
                            {"role": "assistant", "content": f"图里有{amount}{cname}{sname}。"},
                        ],
                        "image": str(p),
                    })
    dst = paths.data_dir() / "mm_图像问答.jsonl"
    fsutil.atomic_write_lines(dst, [json.dumps(r, ensure_ascii=False) for r in rows])
    return dst, len(rows)


def available_voices() -> list[str]:
    """列出系统里的语音（中/英）。脚本本身是固定内容，不含任何用户输入。"""
    ps = paths.state_dir() / "_voices.ps1"
    fsutil.atomic_write_text(
        ps,
        "Add-Type -AssemblyName System.Speech\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "$s.GetInstalledVoices() | ForEach-Object { Write-Output $_.VoiceInfo.Name }\n",
        encoding="utf-8-sig")
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        if p.returncode != 0:
            return []
        return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def _pick_voice(voices: list[str]) -> str:
    for v in voices:
        low = v.lower()
        if "huihui" in low or "chinese" in low or "zh" in low:
            return v
    return voices[0] if voices else ""


def synthesize(texts: list[str], out_dir: Path, voice: str = "") -> tuple[list[Path | None], str]:
    """用系统 TTS 把多段文本合成 wav。

    安全：数据集文本、音色名、输出路径全部作为 JSON 数据写入独立 UTF-8 文件，
    固定 PowerShell 脚本只做 `ConvertFrom-Json` + 变量取值，不做字符串插值，
    因此文本里的 $()、反引号、引号、换行都不会变成命令。

    返回 (文件列表（失败位置为 None）, 错误说明)。错误说明非空时 UI 需要展示。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ps = paths.state_dir() / "_tts_batch.ps1"
    fsutil.atomic_write_text(ps, _TTS_SCRIPT, encoding="utf-8-sig")

    stamp = f"{int(time.time() * 1000)}_{len(texts)}"
    job_file = paths.state_dir() / f"_tts_job_{stamp}.json"
    result_file = paths.state_dir() / f"_tts_result_{stamp}.json"
    items = []
    files: list[Path] = []
    for i, text in enumerate(texts, 1):
        wav = out_dir / f"q{i:04d}.wav"
        files.append(wav)
        items.append({"index": i, "text": str(text), "out": str(wav)})
    job = {"voice": str(voice or ""), "items": items, "result": str(result_file)}
    fsutil.atomic_write_json(job_file, job)

    err = ""
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                            "-File", str(ps), "-JobFile", str(job_file)],
                           capture_output=True, timeout=max(120, 60 + 3 * len(texts)))
        code = p.returncode
        if code != 0:
            err = f"TTS 进程返回 code={code}"
            detail = (p.stderr or b"").decode("utf-8", "replace").strip()
            if detail:
                err += f"：{detail[:300]}"
    except (OSError, subprocess.SubprocessError) as exc:
        code = -1
        err = f"TTS 启动/执行失败：{type(exc).__name__}: {exc}"

    # 逐条结果：文件存在且不是空壳才算成功
    failed_index: set[int] = set()
    if result_file.is_file():
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8-sig"))
            rows = payload.get("items") if isinstance(payload, dict) else None
            if isinstance(rows, dict):
                rows = [rows]
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                idx = int(row.get("index") or 0)
                if not row.get("ok"):
                    failed_index.add(idx)
                    if not err:
                        err = f"第 {idx} 条合成失败：{str(row.get('error') or '')[:200]}"
        except (OSError, ValueError, TypeError):
            if not err:
                err = "TTS 结果文件无法解析"
    elif not err:
        err = "TTS 没有生成结果文件"

    out: list[Path | None] = []
    for i, f in enumerate(files, 1):
        if i in failed_index:
            out.append(None)
            continue
        try:
            ok = f.is_file() and f.stat().st_size > 2000
        except OSError:
            ok = False
        if not ok:
            out.append(None)
            if not err:
                err = f"第 {i} 条没有生成有效 wav（可能 TTS 音色不可用）"
        else:
            out.append(f)
    for tmp in (job_file, result_file):
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return out, err


def make_audio_dataset(src_jsonl: Path | None = None,
                       limit: int = 12) -> tuple[Path, int, str]:
    """把你的文本问答数据里的「问题」读成语音，生成「语音提问 → 文本回答」样本。"""
    from . import dataset as ds

    src = Path(src_jsonl) if src_jsonl else paths.data_dir() / "sample_sft.jsonl"
    pairs: list[tuple[str, str]] = []
    if src.is_file():
        for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msgs = ds.normalize(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
            q = next((m["content"] for m in msgs if m["role"] == "user"), "")
            a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if q and a and not q.startswith("<|"):
                pairs.append((q, a))
            if len(pairs) >= limit:
                break
    if not pairs:
        return paths.data_dir() / "mm_语音问答.jsonl", 0, "没有可用的文本问答数据"

    voices = available_voices()
    voice = _pick_voice(voices)
    wavs, tts_err = synthesize([q for q, _ in pairs], mm_root() / WAV_SUBDIR, voice)
    rows = []
    for (q, a), wav in zip(pairs, wavs):
        if wav is None:
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": "<|audio_pad|>请听这段语音，然后回答问题。"},
                {"role": "assistant", "content": a},
            ],
            "audio": str(wav),
            "audio_text": q,
        })
    dst = paths.data_dir() / "mm_语音问答.jsonl"
    fsutil.atomic_write_lines(dst, [json.dumps(r, ensure_ascii=False) for r in rows])
    return dst, len(rows), tts_err


def mm_stats(path: Path) -> dict:
    """统计多模态数据集：条数、图像/音频分布、缺文件数。"""
    p = Path(path)
    out = {"total": 0, "image": 0, "audio": 0, "missing": 0, "text_only": 0}
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        out["total"] += 1
        kind = None
        for key in ("image", "audio"):
            if obj.get(key):
                kind = key
                out[key] += 1
                if not Path(str(obj[key])).is_file():
                    out["missing"] += 1
        if kind is None:
            out["text_only"] += 1
    return out


def feature_cache_dir() -> Path:
    d = paths.data_dir() / ".mm_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(sample: dict) -> str:
    """按「媒体文件路径 + 大小 + mtime」算缓存键，文件换了自动失效。"""
    for key in ("image", "audio"):
        f = sample.get(key)
        if not f:
            continue
        p = Path(str(f))
        try:
            st = p.stat()
            raw = f"{key}|{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            raw = f"{key}|{p}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
    return ""


def describe() -> dict:
    return {
        "vision": {"encoder": "MobileNetV3-Small（ImageNet）", "params_m": 2.5, "tokens": 49},
        "audio": {"encoder": "Whisper-tiny encoder", "params_m": 8.0, "tokens": 32},
        "inject": "embedding 层前向钩子：把 <|image_pad|>/<|audio_pad|> 的向量换成编码器特征",
        "train": "编码器冻结 + 特征缓存，训练只更新投影层与 LoRA",
    }


if __name__ == "__main__":
    p, n = make_image_dataset()
    print("images", p, n)
    p2, n2, err = make_audio_dataset()
    print("audio", p2, n2, err, mm_stats(p2), mm_stats(p))
