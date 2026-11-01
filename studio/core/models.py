# -*- coding: utf-8 -*-
"""本地模型 / 适配器 / 合并产物的扫描、校验、删除。"""
from __future__ import annotations

import ctypes
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

WEIGHT_SUFFIX = (".safetensors", ".bin", ".pt", ".pth")
REQUIRED_BASE_FILES = ("config.json",)
TOKENIZER_HINTS = ("tokenizer.json", "tokenizer.model", "vocab.json", "tokenizer_config.json")


@dataclass
class ModelInfo:
    name: str
    path: Path
    kind: str = "base"          # base / adapter / merged
    size: int = 0
    params: int | None = None
    dtype: str = ""
    arch: str = ""
    layers: int | None = None
    hidden: int | None = None
    vocab: int | None = None
    complete: bool = True
    missing: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def params_m(self) -> float:
        return (self.params or 0) / 1e6

    @property
    def trainable_hint(self) -> str:
        p = self.params or 0
        if p == 0:
            return "未知"
        if p <= 600e6:
            return "CPU 可训"
        if p <= 1.5e9:
            return "CPU 勉强"
        return "建议 GPU"

    def label(self) -> str:
        bits = [self.name]
        if self.params:
            bits.append(f"{self.params_m:.0f}M")
        if self.size:
            bits.append(paths.human_size(self.size))
        return " · ".join(bits)


def _read_safetensors_header(path: Path) -> tuple[int, str] | None:
    """只读文件头，返回 (参数量, dtype)。读不出返回 None。"""
    try:
        with path.open("rb") as f:
            head = f.read(8)
            if len(head) < 8:
                return None
            n = struct.unpack("<Q", head)[0]
            if n <= 0 or n > 200 * 1024 * 1024:
                return None
            raw = f.read(n)
        meta = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, struct.error):
        return None
    total = 0
    dtype = ""
    for key, info in meta.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        shape = info.get("shape") or []
        cnt = 1
        for d in shape:
            cnt *= int(d)
        total += cnt
        if not dtype and "weight" in key:
            dtype = str(info.get("dtype", ""))
    return total, dtype


def estimate_params_from_config(cfg: dict) -> int | None:
    """按 config 里的结构参数推算 LLM 主干参数量（Llama 系通用公式）。

    这样 .bin 权重也能给出接近真实的值：MiniMind-O 用这个公式得到 63.9M，
    与加载后实际统计一致（误差 <1%）。
    """
    h = cfg.get("hidden_size") or cfg.get("n_embd")
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    vocab = cfg.get("vocab_size")
    heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    kv = cfg.get("num_key_value_heads") or heads
    head_dim = cfg.get("head_dim")
    inter = cfg.get("intermediate_size")
    if not all(isinstance(x, int) and x > 0 for x in (h, layers, vocab, heads)):
        return None
    head_dim = head_dim if isinstance(head_dim, int) and head_dim > 0 else h // heads
    inter = inter if isinstance(inter, int) and inter > 0 else int(h * 3.5)
    q = h * heads * head_dim
    o = heads * head_dim * h
    k = h * kv * head_dim
    v = kv * head_dim * h
    mlp = 3 * h * inter
    return int(vocab * h + layers * (q + k + v + o + mlp))


def read_weight_info(dirpath: Path) -> tuple[int | None, str, int]:
    """返回 (参数量, dtype, 权重文件总字节)。支持分片 safetensors 与 .bin。"""
    total_params = 0
    dtype = ""
    total_bytes = 0
    found = False
    for f in sorted(dirpath.glob("*.safetensors")):
        total_bytes += f.stat().st_size
        r = _read_safetensors_header(f)
        if r:
            total_params += r[0]
            dtype = dtype or r[1]
            found = True
    if not found:
        for f in sorted(dirpath.glob("*.bin")) + sorted(dirpath.glob("*.pt")):
            total_bytes += f.stat().st_size
            dtype = dtype or "bin"
        cfg = _read_json(dirpath / "config.json")
        est = estimate_params_from_config(cfg)
        if est:
            return est, dtype, total_bytes
        if total_bytes:                      # 兜底：按 fp32 4 字节/参数粗估
            return int(total_bytes / 4), dtype, total_bytes
    return (total_params or None), dtype, total_bytes


def dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def inspect(dirpath: Path, kind: str | None = None) -> ModelInfo:
    name = dirpath.name
    cfg = _read_json(dirpath / "config.json")
    is_adapter = (dirpath / "adapter_config.json").is_file()
    kind = kind or ("adapter" if is_adapter else "base")
    missing: list[str] = []
    if kind == "adapter":
        if not (dirpath / "adapter_model.safetensors").is_file() and not (dirpath / "adapter_model.bin").is_file():
            missing.append("adapter_model.safetensors")
        params, dtype, wbytes = read_weight_info(dirpath)
    else:
        if not any((dirpath / n).is_file() for n in REQUIRED_BASE_FILES):
            missing.append("config.json")
        if not any(f.suffix in WEIGHT_SUFFIX for f in dirpath.iterdir() if f.is_file()):
            missing.append("权重文件(*.safetensors)")
        if not any((dirpath / n).is_file() for n in TOKENIZER_HINTS):
            missing.append("tokenizer 文件")
        params, dtype, wbytes = None, "", 0
        has_weight = any(f.suffix in WEIGHT_SUFFIX and f.name != "adapter_model.safetensors"
                         for f in dirpath.iterdir() if f.is_file())
        if has_weight:
            params, dtype, wbytes = read_weight_info(dirpath)

    arch = ""
    archs = cfg.get("architectures")
    if isinstance(archs, list) and archs:
        arch = str(archs[0])
    elif cfg.get("model_type"):
        arch = str(cfg["model_type"])
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    hidden = cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model")
    vocab = cfg.get("vocab_size")

    note = ""
    if kind == "adapter":
        base = _read_json(dirpath / "adapter_config.json").get("base_model_name_or_path")
        if base:
            note = Path(str(base)).name
    return ModelInfo(
        name=name, path=dirpath, kind=kind, size=dir_size(dirpath),
        params=params, dtype=dtype, arch=arch,
        layers=int(layers) if isinstance(layers, int) else None,
        hidden=int(hidden) if isinstance(hidden, int) else None,
        vocab=int(vocab) if isinstance(vocab, int) else None,
        complete=not missing, missing=missing, note=note,
    )


def scan_models() -> list[ModelInfo]:
    out: list[ModelInfo] = []
    d = paths.models_dir()
    if not d.is_dir():
        return out
    for sub in sorted(d.iterdir(), key=lambda p: p.name.lower()):
        if sub.is_dir() and any((sub / n).is_file() for n in ("config.json", "adapter_config.json")):
            try:
                out.append(inspect(sub))
            except OSError:
                continue
    return out


def scan_outputs() -> list[ModelInfo]:
    out: list[ModelInfo] = []
    d = paths.outputs_dir()
    if not d.is_dir():
        return out
    for sub in sorted(d.iterdir(), key=lambda p: p.name.lower()):
        if not sub.is_dir():
            continue
        if (sub / "adapter_config.json").is_file():
            kind = "adapter"
        elif (sub / "config.json").is_file() and any(f.suffix == ".safetensors" for f in sub.glob("*.safetensors")):
            kind = "merged"
        else:
            continue
        try:
            out.append(inspect(sub, kind=kind))
        except OSError:
            continue
    return out


def default_model_name(models: list[ModelInfo], last: str = "") -> str:
    """挑一个默认模型：优先上次用过 → 完整且非下划线开头的 → 任意完整的 → 第一个。"""
    names = [m.name for m in models]
    if not names:
        return ""
    if last and last in names:
        return last
    clean = [m.name for m in models if m.complete and not m.name.startswith(("_", "."))]
    if clean:
        return clean[0]
    complete = [m.name for m in models if m.complete]
    return (complete or names)[0]


def find_model(name: str) -> ModelInfo | None:
    if not name:
        return None
    p = Path(name)
    if p.is_dir():
        try:
            return inspect(p)
        except OSError:
            return None
    for m in scan_models() + scan_outputs():
        if m.name.lower() == name.lower():
            return m
    return None


def _recycle(path: Path) -> bool:
    """送进回收站（可撤销）。"""
    try:
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("wFunc", wintypes.UINT),
                ("pFrom", ctypes.c_void_p),
                ("pTo", ctypes.c_void_p),
                ("fFlags", ctypes.c_uint16),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", ctypes.c_void_p),
                ("lpszProgressTitle", wintypes.LPCWSTR),
            ]

        FO_DELETE = 3
        FOF_SILENT = 0x0004
        FOF_NOCONFIRMATION = 0x0010
        FOF_ALLOWUNDO = 0x0040
        FOF_NOERRORUI = 0x0400

        buf = ctypes.create_unicode_buffer(str(path) + "\0")
        op = SHFILEOPSTRUCTW()
        op.hwnd = None
        op.wFunc = FO_DELETE
        op.pFrom = ctypes.cast(buf, ctypes.c_void_p)
        op.pTo = None
        op.fFlags = FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO | FOF_NOERRORUI
        rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        return rc == 0 and not op.fAnyOperationsAborted
    except Exception:  # noqa: BLE001
        return False


def delete_dir(path: Path, recycle: bool = True) -> tuple[bool, str]:
    """删除模型/实验目录。recycle=True 时先尝试回收站，失败再直删。"""
    import shutil

    path = Path(path)
    if not path.exists():
        return True, "路径不存在，已视为删除完成"
    if path.is_file():
        if recycle and _recycle(path) and not path.exists():
            return True, "已移入回收站"
        try:
            path.unlink()
            return True, "已永久删除"
        except OSError as exc:
            return False, f"删除失败：{exc}"
    if not path.is_dir():
        return False, f"不是文件也不是目录：{path}"
    try:
        if path.resolve() in (paths.root(), paths.models_dir(), paths.outputs_dir(), paths.data_dir()):
            return False, "拒绝删除受保护目录"
    except OSError:
        pass
    if recycle and _recycle(path) and not path.exists():
        return True, "已移入回收站"
    try:
        shutil.rmtree(path)
        return True, "已永久删除"
    except OSError as exc:
        return False, f"删除失败：{exc}"


def import_model(src: Path, name: str = "", link: bool = True) -> tuple[Path | None, str]:
    """把外部模型目录接入 models/。优先建目录联接（不占额外空间），失败再复制。"""
    import shutil
    import subprocess

    src = Path(src)
    if not src.is_dir():
        return None, f"不是目录：{src}"
    if not (src / "config.json").is_file():
        return None, "目标目录缺少 config.json，不像 HuggingFace 模型目录"
    target = paths.models_dir() / (name or src.name)
    if target.exists():
        return None, f"models 下已存在同名目录：{target.name}"
    paths.models_dir().mkdir(parents=True, exist_ok=True)
    if link:
        try:
            p = subprocess.run(["cmd", "/c", "mklink", "/J", str(target), str(src)],
                               capture_output=True, text=True, timeout=30,
                               creationflags=0x08000000)
            if p.returncode == 0 and target.is_dir():
                return target, "已建立目录联接（不额外占用磁盘）"
        except Exception:  # noqa: BLE001
            pass
    try:
        shutil.copytree(src, target)
        return target, f"已复制 {dir_size(target)/1024**2:.0f} MB"
    except OSError as exc:
        return None, f"复制失败：{exc}"


def disk_free_gb(p: Path) -> float:
    try:
        import shutil as sh
        return sh.disk_usage(str(p)).free / 1024 ** 3
    except OSError:
        return float("nan")


def fmt_params(n: int | None) -> str:
    if not n:
        return "-"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)

