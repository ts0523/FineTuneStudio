# -*- coding: utf-8 -*-
"""本机音频：录音 / 播放 / 说话（TTS）/ 听话（ASR）。

全部走 Windows 自带能力，exe 不需要额外依赖：
- 录音、播放：winmm 的 MCI（mciSendStringW），16 kHz 单声道 16 bit wav
- 说话：System.Speech 的 SpeechSynthesizer（系统中文语音，本机是 Microsoft Huihui）
- 听话：优先本地 Whisper（.venv 里的 transformers + models/_encoders 下的权重），
        退化到 System.Speech 的桌面识别器（快，但中文准确率低）
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

from . import config, paths, procs, runtime

IS_WIN = sys.platform == "win32"
_ALIAS_LOCK = threading.Lock()
_counter = {"n": 0}
_play_timer: threading.Timer | None = None


def _cfg():
    return config.cfg()


def temp_dir() -> Path:
    d = paths.state_dir() / "audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _alias(prefix: str) -> str:
    with _ALIAS_LOCK:
        _counter["n"] += 1
        return f"fts{prefix}{_counter['n'] % 1000}"


def _mci(cmd: str) -> tuple[int, str]:
    if not IS_WIN:
        return -1, "非 Windows 平台"
    buf = ctypes.create_unicode_buffer(600)
    try:
        err = ctypes.windll.winmm.mciSendStringW(cmd, buf, 598, None)
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"
    return int(err), buf.value


def _mci_error(code: int) -> str:
    if code == 0:
        return ""
    buf = ctypes.create_unicode_buffer(600)
    try:
        ctypes.windll.winmm.mciGetErrorStringW(code, buf, 598)
    except Exception:  # noqa: BLE001
        return f"MCI 错误 {code}"
    return buf.value or f"MCI 错误 {code}"


# ---------------------------------------------------------------- 录音
class Recorder:
    """按需开始/停止的录音器。

    优先 sounddevice（16 kHz 单声道 16 bit，音质足够做语音识别；PortAudio 随
    exe 一起打包，不需要用户装任何东西）。如果声卡/驱动不支持，退回 Windows
    自带的 MCI，并逐个参数尝试（有些驱动不认一次设置多个格式参数）。
    """

    def __init__(self) -> None:
        self.alias = ""
        self.path: Path | None = None
        self.started = 0.0
        self.error = ""
        self._sd_stream = None
        self._sd_frames: list = []
        self._sd_rate = 16000

    @property
    def active(self) -> bool:
        return bool(self.alias) or self._sd_stream is not None

    # -- sounddevice
    def _start_sd(self, path: Path) -> bool:
        try:
            import numpy as np
            import sounddevice as sd
        except Exception:  # noqa: BLE001
            return False
        try:
            info = sd.query_devices(kind="input")
            rate = int(16000)
            try:
                sd.check_input_settings(device=info["index"], samplerate=rate,
                                        channels=1, dtype="int16")
            except Exception:  # noqa: BLE001
                rate = int(info.get("default_samplerate") or 44100)

            def cb(indata, frames, time_info, status):  # noqa: ANN001, ARG001
                self._sd_frames.append(indata.copy())

            stream = sd.InputStream(device=info["index"], samplerate=rate, channels=1,
                                    dtype="int16", callback=cb)
            stream.start()
        except Exception as exc:  # noqa: BLE001
            self.error = f"录音设备不可用：{type(exc).__name__}: {exc}"
            return False
        self._sd_stream = stream
        self._sd_rate = rate
        self._sd_frames = []
        self.path = path
        self.started = time.time()
        self.error = ""
        return True

    def _stop_sd(self, discard: bool) -> Path | None:
        stream, frames = self._sd_stream, self._sd_frames
        path = self.path
        self._sd_stream = None
        self._sd_frames = []
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        if discard or path is None or not frames:
            self.path = None
            return None
        try:
            import numpy as np
            data = np.concatenate(frames, axis=0)
            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(self._sd_rate)
                f.writeframes(data.tobytes())
            self.path = None
            if path.is_file() and path.stat().st_size > 1000:
                return path
        except Exception as exc:  # noqa: BLE001
            self.error = f"保存录音失败：{type(exc).__name__}: {exc}"
        self.path = None
        return None

    # -- MCI 兜底
    def _start_mci(self, path: Path) -> bool:
        for spec in ("16000", "44100", ""):
            alias = _alias("rec")
            code, _ = _mci(f"open new type waveaudio alias {alias}")
            if code:
                self.error = f"打不开录音设备：{_mci_error(code)}"
                return False
            cmds = [f"set {alias} time format ms", f"set {alias} channels 1"]
            if spec:
                cmds.append(f"set {alias} samplespersec {spec}")
            cmds.append(f"record {alias}")
            bad = 0
            for cmd in cmds:
                c, _ = _mci(cmd)
                if c:
                    bad = c
                    break
            if bad:
                _mci(f"close {alias}")
                self.error = f"录音失败：{_mci_error(bad)}"
                continue
            self.alias = alias
            self.path = path
            self.started = time.time()
            self.error = ""
            return True
        return False

    def start(self, path: Path | None = None) -> bool:
        self.stop(discard=True)
        path = Path(path) if path else temp_dir() / f"rec-{int(time.time())}.wav"
        if IS_WIN and self._start_sd(path):
            return True
        if IS_WIN:
            mci_err = self.error
            if self._start_mci(path):
                return True
            self.error = mci_err or self.error or "没有可用的录音设备"
        return False

    def stop(self, discard: bool = False) -> Path | None:
        if self._sd_stream is not None:
            return self._stop_sd(discard)
        if not self.alias:
            return None
        alias, path = self.alias, self.path
        self.alias = ""
        _mci(f"stop {alias}")
        out: Path | None = None
        if not discard and path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            c, _ = _mci(f'save {alias} "{path}"')
            if c == 0 and path.is_file() and path.stat().st_size > 1000:
                out = path
            else:
                self.error = f"保存录音失败：{_mci_error(c)}"
        _mci(f"close {alias}")
        if out is None and path is not None and path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
        self.path = None
        return out


def mic_available() -> tuple[bool, str]:
    """探一下有没有可录音设备（不真的录）。"""
    if not IS_WIN:
        return False, "只支持 Windows"
    try:
        import sounddevice as sd

        info = sd.query_devices(kind="input")
        if info and int(info.get("max_input_channels") or 0) > 0:
            return True, ""
    except Exception:  # noqa: BLE001
        pass
    alias = _alias("probe")
    code, _ = _mci(f"open new type waveaudio alias {alias}")
    if code:
        return False, _mci_error(code) or "没有可用的录音设备"
    _mci(f"close {alias}")
    return True, ""


def duration_of(path: Path) -> float:
    """用 MCI 取 wav 时长（秒），失败返回 0。"""
    alias = _alias("len")
    if _mci(f'open "{path}" type waveaudio alias {alias}')[0]:
        return 0.0
    code, val = _mci(f"status {alias} length")
    _mci(f"close {alias}")
    try:
        return float(val) / 1000.0 if code == 0 else 0.0
    except ValueError:
        return 0.0


# ---------------------------------------------------------------- 播放
def play_wav(path: Path, volume_hint: str = "") -> bool:
    """非阻塞播放 wav；同一时刻只保留最后一条。"""
    global _play_timer
    stop_play()
    alias = _alias("play")
    code, _ = _mci(f'open "{path}" type waveaudio alias {alias}')
    if code:
        # 不是 waveaudio 能识别的格式时，退回 winsound
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        except Exception:  # noqa: BLE001
            return False
    code, _ = _mci(f"play {alias}")
    if code:
        _mci(f"close {alias}")
        return False
    secs = max(0.5, duration_of(path))

    def _close() -> None:
        _mci(f"close {alias}")

    _play_timer = threading.Timer(secs + 0.6, _close)
    _play_timer.daemon = True
    _play_timer.start()
    return True


def stop_play() -> None:
    global _play_timer
    if _play_timer is not None:
        try:
            _play_timer.cancel()
        except Exception:  # noqa: BLE001
            pass
        _play_timer = None
    try:
        if IS_WIN:
            import winsound

            winsound.PlaySound(None, 0)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- TTS
_voices_cache: list[str] | None = None


def _ps_literal(value: str | Path) -> str:
    """PowerShell 单引号字符串字面量（内部单引号双写，杜绝路径/音色注入）。"""
    return "'" + str(value).replace("'", "''") + "'"


def voices(refresh: bool = False) -> list[str]:
    global _voices_cache
    if _voices_cache is not None and not refresh:
        return list(_voices_cache)
    script = ("Add-Type -AssemblyName System.Speech\n"
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
              "$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }\n")
    f = temp_dir() / "_voices.ps1"
    f.write_text(script, encoding="utf-8-sig")
    code, out = procs.run_text(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                                "-File", str(f)], timeout=40)
    names = [ln.strip() for ln in out.splitlines() if ln.strip()]
    _voices_cache = names
    return names


def pick_voice(prefer_zh: bool = True) -> str:
    vs = voices()
    for v in vs:
        low = v.lower()
        if prefer_zh and ("huihui" in low or "kangkang" in low or "yaoyao" in low):
            return v
    return vs[0] if vs else ""


def speak_script(text: str, voice: str = "", rate: int = 0, wav_out: Path | None = None,
                 pitch: int = 0, rate_pct: int | None = None) -> Path:
    """生成一段 PowerShell 脚本（说一句或存成 wav），返回脚本路径。

    pitch / rate_pct 走 SSML prosody：音色克隆用样本估出的音高差与语速差来贴近用户。
    """
    import html

    stamp = f"{int(time.time()*1000)%100000}"
    txt = temp_dir() / f"_tts_{stamp}.txt"
    txt.write_text(text, encoding="utf-8")
    lines = ["$ErrorActionPreference = 'Stop'",
             "Add-Type -AssemblyName System.Speech",
             "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
             f"$s.Rate = {int(rate)}"]
    v = voice or pick_voice()
    if v:
        lines.append(f"try {{ $s.SelectVoice({_ps_literal(v)}) }} catch {{ }}")
    if pitch or rate_pct:
        def _pct(v: int) -> str:
            return f"+{int(v)}%" if int(v) >= 0 else f"{int(v)}%"

        ssml = (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='zh-CN'>"
                f"<prosody pitch='{_pct(pitch)}' rate='{_pct(rate_pct or 0)}'>"
                f"{html.escape(text, quote=False)}</prosody></speak>")
        ssml_file = temp_dir() / f"_ssml_{stamp}.xml"
        ssml_file.write_text(ssml, encoding="utf-8")
        lines.append(f"$xml = Get-Content -Raw -Encoding UTF8 {_ps_literal(ssml_file)}")
        speak_cmd = "$s.SpeakSsml($xml)"
    else:
        lines.append(f"$t = Get-Content -Raw -Encoding UTF8 {_ps_literal(txt)}")
        speak_cmd = "$s.Speak($t)"
    if wav_out is not None:
        lines.append(f"$s.SetOutputToWaveFile({_ps_literal(wav_out)})")
        lines.append(speak_cmd)
        lines.append("$s.SetOutputToNull()")
    else:
        lines.append(speak_cmd)
    lines.append("$s.Dispose()")
    script = temp_dir() / f"_tts_{stamp}.ps1"
    script.write_text("\n".join(lines), encoding="utf-8-sig")
    return script


def speak_async(text: str, voice: str = "", rate: int = 0, pitch: int = 0,
                rate_pct: int | None = None) -> subprocess.Popen | None:
    """后台朗读（不阻塞界面）。"""
    text = (text or "").strip()
    if not text:
        return None
    try:
        script = speak_script(text[:1200], voice, rate, pitch=pitch, rate_pct=rate_pct)
    except OSError:
        return None
    try:
        return procs.popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                            "-File", str(script)])
    except OSError:
        return None


def speak_wav(text: str, voice: str = "", rate: int = 0, pitch: int = 0,
              rate_pct: int | None = None) -> Path | None:
    out = temp_dir() / f"say-{int(time.time()*1000)%1000000}.wav"
    script = speak_script(text[:1200], voice, rate, out, pitch=pitch, rate_pct=rate_pct)
    code, _ = procs.run_text(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                              "-File", str(script)], timeout=120)
    if out.is_file() and out.stat().st_size > 2000:
        return out
    return None


# ---------------------------------------------------------------- ASR
def asr_sapi(wav: Path) -> tuple[str, float]:
    """系统桌面识别器：几十毫秒出结果，中文准确率一般。"""
    script = (f"[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
              f"Add-Type -AssemblyName System.Speech\n"
              f"$ci = New-Object System.Globalization.CultureInfo('zh-CN')\n"
              f"$r = New-Object System.Speech.Recognition.SpeechRecognitionEngine($ci)\n"
              f"$r.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))\n"
              f"$r.SetInputToWaveFile({_ps_literal(wav)})\n"
              f"$res = $r.Recognize()\n"
              f"if ($res) {{ Write-Output ('TEXT:' + $res.Text + '|CONF:' + $res.Confidence) }}\n"
              f"$r.Dispose()\n")
    f = temp_dir() / f"_asr_{int(time.time()*1000)%100000}.ps1"
    f.write_text(script, encoding="utf-8-sig")
    code, out = procs.run_text(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                                "-File", str(f)], timeout=60)
    for line in out.splitlines():
        if line.startswith("TEXT:"):
            body = line[5:]
            text, _, conf = body.partition("|CONF:")
            try:
                return text.strip(), float(conf)
            except ValueError:
                return text.strip(), 0.0
    return "", 0.0


_WHISPER_CODE = r"""
import json, sys
p = json.loads(sys.argv[1])
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model=p["model"], device=-1)
out = pipe(p["wav"], generate_kwargs={"language": p.get("lang") or "zh",
                                      "task": "transcribe"})
print("@@ASR@@" + json.dumps({"text": (out or {}).get("text", "")}, ensure_ascii=False))
"""


def whisper_model_dir() -> Path:
    name = str(_cfg().get("ai.whisper_model", "") or "whisper-tiny")
    d = paths.models_dir() / "_encoders" / name
    return d


def asr_whisper(wav: Path, model_dir: Path | None = None,
                lang: str = "zh") -> tuple[str, str]:
    """本地 Whisper 转写。返回 (文本, 错误)。需要 .venv 里的 transformers。"""
    py, source = runtime.find_python(_cfg())
    if py is None:
        return "", f"找不到训练环境（{source}），无法使用本地 Whisper"
    md = model_dir or whisper_model_dir()
    if not (md / "config.json").is_file():
        return "", (f"缺少 Whisper 权重：{md}\n可在「设置 → AI 功能」点「下载语音识别模型」"
                    f"（约 150 MB）")
    payload = json.dumps({"model": str(md), "wav": str(wav), "lang": lang}, ensure_ascii=False)
    env = procs.base_env(py, {"HF_ENDPOINT": str(_cfg().get("endpoint") or "https://hf-mirror.com"),
                              "HF_HOME": str(paths.root() / ".cache" / "huggingface")})
    code, out = procs.run_text([str(py), "-c", _WHISPER_CODE, payload], cwd=paths.root(),
                               env=env, timeout=300)
    for line in out.splitlines():
        if line.startswith("@@ASR@@"):
            try:
                return str(json.loads(line[7:]).get("text", "")).strip(), ""
            except json.JSONDecodeError:
                continue
    tail = "\n".join(out.strip().splitlines()[-4:])
    return "", f"Whisper 转写失败（exit {code}）：{tail[:300]}"


def transcribe(wav: Path, engine: str = "") -> tuple[str, str, str]:
    """统一的转写入口。引擎：auto（默认）/ whisper / sapi。返回 (文本, 引擎, 错误)。"""
    engine = (engine or str(_cfg().get("ai.asr", "") or "auto")).lower()
    if engine == "sapi":
        text, _conf = asr_sapi(wav)
        return text, "系统识别", ("" if text else "系统识别器没有听出内容")
    if engine == "whisper":
        text, err = asr_whisper(wav)
        if text:
            return text, "Whisper（本地）", ""
        if not err:
            return "", "Whisper（本地）", "没有听出内容，请靠近麦克风再说一次"
        text2, _conf = asr_sapi(wav)
        if text2:
            return text2, "系统识别（回退）", ""
        return "", "Whisper（本地）", err
    # auto：优先更准的 Whisper；未安装或转写失败时才回退到系统识别。
    text, err = asr_whisper(wav)
    if text:
        return text, "Whisper（本地）", ""
    text2, _conf = asr_sapi(wav)
    if text2:
        return text2, "系统识别（回退）", ""
    return "", "语音识别", err or "没有听出内容，请靠近麦克风再说一次"


def download_whisper(size: str = "whisper-tiny") -> tuple[int, str]:
    """下载 Whisper 权重到 models/_encoders/（走 hf-mirror）。返回 (code, 输出)。"""
    py, source = runtime.find_python(_cfg())
    if py is None:
        return -1, f"找不到训练环境：{source}"
    code = ("import sys\n"
            "from huggingface_hub import snapshot_download\n"
            f"p = snapshot_download('openai/{size}', local_dir=r'{paths.models_dir() / '_encoders' / size}')\n"
            "print('OK', p)\n")
    env = procs.base_env(py, {
        "HF_ENDPOINT": str(_cfg().get("endpoint") or "https://hf-mirror.com"),
        "HF_HOME": str(paths.root() / ".cache" / "huggingface"),
    })
    return procs.run_text([str(py), "-c", code], cwd=paths.root(), env=env, timeout=1800)


def record_for(seconds: float, path: Path | None = None) -> tuple[Path | None, str]:
    """录固定秒数（阻塞）。返回 (wav, 错误)。"""
    rec = Recorder()
    if not rec.start(path):
        return None, rec.error
    time.sleep(max(0.5, seconds))
    out = rec.stop()
    if out is None:
        return None, rec.error or "没有录到声音"
    return out, ""
