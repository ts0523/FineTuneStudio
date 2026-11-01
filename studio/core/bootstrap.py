# -*- coding: utf-8 -*-
"""开机自检与「一键装好」：让 exe 双击之后自己把运行环境补齐。

- 摄像头（opencv）与录音（PortAudio/sounddevice）：已随 exe 打包，无需安装
- 语音识别与朗读：Windows 自带 SAPI，无需安装（Whisper 权重是可选的升级项）
- 训练环境（.venv：torch/transformers/peft…）：只有本地训练 / 本地模型需要

云端实验台（OpenRouter 免费模型 + 语音 + 视频通话）不需要 .venv，双击就能用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import config, paths, runtime, secrets

PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
CAMERA_PIN = "opencv-python-headless"


@dataclass
class Status:
    python: Path | None = None
    python_source: str = ""
    missing: list[str] = field(default_factory=list)
    camera: bool = False
    whisper: bool = False
    ai: bool = False
    mic: bool = False
    venv_dir: Path | None = None

    @property
    def env_ok(self) -> bool:
        return self.python is not None and not self.missing

    @property
    def core_ok(self) -> bool:
        """能不能跑「本地模型体验 + 训练」。"""
        return self.env_ok

    def summary(self) -> str:
        if self.python is None:
            return "未安装训练环境（云模式可用）"
        if self.missing:
            return "缺依赖：" + ", ".join(self.missing)
        return "训练环境就绪"


def _cfg():
    return config.cfg()


def probe() -> Status:
    st = Status()
    py, source = runtime.find_python(_cfg())
    st.python = py
    st.python_source = source
    st.venv_dir = paths.root() / ".venv"
    st.ai = bool(secrets.get_api_key(_cfg()))
    from . import audio

    st.whisper = (paths.models_dir() / "_encoders"
                  / str(_cfg().get("ai.whisper_model", "whisper-tiny") or "whisper-tiny")
                  / "config.json").is_file()
    try:
        st.mic, _ = audio.mic_available()
    except Exception:  # noqa: BLE001
        st.mic = False
    # 摄像头/录音已随 exe 自带，不再依赖 .venv
    from . import camera as camera_mod

    st.camera = bool(camera_mod.available())
    if py is None:
        return st
    env = runtime.probe(_cfg())
    st.python = env.python
    st.missing = list(env.missing)
    return st


def env_install_args() -> list[str] | None:
    """创建 .venv 并装依赖（走项目里的 setup_env.ps1，能用 uv 就用 uv）。"""
    script = paths.root() / "setup_env.ps1"
    if not script.is_file():
        return None
    return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]


def camera_install_args(py: Path | None = None) -> list[str] | None:
    target = py or runtime.find_python(_cfg())[0]
    if target is None:
        return None
    return [str(target), "-m", "pip", "install", CAMERA_PIN, "-i", PIP_MIRROR,
            "--disable-pip-version-check"]


def whisper_present() -> bool:
    name = str(_cfg().get("ai.whisper_model", "whisper-tiny") or "whisper-tiny")
    return (paths.models_dir() / "_encoders" / name / "config.json").is_file()
