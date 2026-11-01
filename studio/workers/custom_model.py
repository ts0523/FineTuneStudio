# -*- coding: utf-8 -*-
"""加载本地模型，兼容「仓库自带建模代码」的模型（MiniMind 系列就是这种）。

判定顺序：
1. models/<x>/ 下没有 auto_map、也没有 model_*.py → 走 transformers 原生加载；
2. 有 auto_map 且指向的类名以 ForCausalLM 结尾 → 按 auto_map 动态导入；
3. 否则（例如 auto_map 指向 MiniMindOmni 这种多模态类）→ 在本目录的 model*.py 里
   找 *ForCausalLM 类，手动构造 + load_state_dict(strict=False)，多出来的
   talker / vision_proj 等权重会被忽略，但 LLM 主干的每一层都对得上。

返回 (model, tokenizer, kind)，kind 用于日志（transformers / custom:xxx）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

TRUST_FILE = ".fts_trust.json"
BUILTIN_CODE = {
    "jingyaogong/minimind-3o": {
        "model_minimind.py": "c97b24b4b900dbbf5fabc0cca81a9f580011f91e032501f41684e10c6ca6ee60",
        "model_omni.py": "0fc7abb2ed621508f22682f81b18fb778ceea0bb5563c8b5ae1d389ed77c01ad",
    }
}
_MUTABLE_REFS = {"", "main", "master", "head", "latest"}


class LoadError(Exception):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def _code_trust_error(md: Path) -> str:
    """目录里有 *.py 模型代码时，必须命中内置固定哈希或显式信任清单。

    权重文件的 SHA-256 校验通过并不代表仓库代码可信，因此这里单独校验代码。
    """
    files = sorted(p for p in Path(md).glob("*.py") if p.is_file())
    if not files:
        return ""
    actual = {p.name: _sha256(p) for p in files}
    for pinned in BUILTIN_CODE.values():
        if set(actual) == set(pinned) and all(pinned[n] == h for n, h in actual.items()):
            return ""
    try:
        manifest = json.loads((Path(md) / TRUST_FILE).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        manifest = {}
    if not isinstance(manifest, dict) or not manifest:
        return ("目录里含模型代码（*.py），但没有可信记录；为防止任意代码执行，已拒绝加载。"
                "请只从可信来源获取模型，并通过应用内「明确信任 + 固定版本」下载。")
    revision = str(manifest.get("revision") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision.strip()):
        return f"信任记录的 revision={revision!r} 不是完整 commit SHA，拒绝执行模型代码"
    pinned = manifest.get("code_sha256") or {}
    if not isinstance(pinned, dict) or set(pinned) != set(actual):
        return "模型代码文件与信任清单不一致，拒绝执行"
    for name, digest in actual.items():
        if str(pinned.get(name) or "").lower() != digest:
            return f"{name} 的哈希与信任记录不一致，拒绝执行"
    return ""


def read_config(model_dir: Path) -> dict:
    p = Path(model_dir) / "config.json"
    if not p.is_file():
        raise LoadError(f"缺少 config.json：{model_dir}")
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _ensure_pkg(model_dir: Path) -> str:
    """把模型目录注册成一个包，这样仓库里的 `from .model_minimind import *` 才能解析。"""
    name = "_fts_local_model"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(model_dir)]
        sys.modules[name] = pkg
    return name


def _import_file(model_dir: Path, stem: str):
    pkg = _ensure_pkg(model_dir)
    full = f"{pkg}.{stem}"
    if full in sys.modules:
        return sys.modules[full]
    path = Path(model_dir) / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(full, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _classes_in(mod) -> dict:
    return {n: o for n, o in vars(mod).items() if isinstance(o, type) and not n.startswith("_")}


def _find_lm_class(model_dir: Path, cfg: dict, note) -> tuple[type | None, str]:
    auto = (cfg.get("auto_map") or {}).get("AutoModelForCausalLM") or ""
    if auto and "." in auto:
        stem, cls_name = auto.rsplit(".", 1)
        if cls_name.endswith("ForCausalLM"):
            try:
                mod = _import_file(model_dir, stem)
                cls = getattr(mod, cls_name, None)
                if isinstance(cls, type):
                    return cls, f"custom:{stem}.{cls_name}"
            except Exception as exc:  # noqa: BLE001
                note(f"auto_map 指向的 {auto} 加载失败（{type(exc).__name__}），改用本地 LLM 类")
    for py in sorted(Path(model_dir).glob("model*.py")):
        if py.stem in ("model_omni", "modeling_omni"):
            continue
        try:
            mod = _import_file(model_dir, py.stem)
        except Exception as exc:  # noqa: BLE001
            note(f"跳过 {py.name}：{type(exc).__name__}: {exc}")
            continue
        for name, obj in _classes_in(mod).items():
            if name.endswith("ForCausalLM") and getattr(obj, "config_class", None) is not None:
                return obj, f"custom:{py.stem}.{name}"
    return None, ""


def _load_state_dict(model_dir: Path):
    import torch

    md = Path(model_dir)
    safes = sorted(md.glob("*.safetensors"))
    if safes:
        from safetensors.torch import load_file
        sd = {}
        for f in safes:
            sd.update(load_file(str(f)))
        return sd
    for name in ("pytorch_model.bin", "model.bin"):
        p = md / name
        if p.is_file():
            return torch.load(str(p), map_location="cpu", weights_only=True)
    raises = ", ".join(sorted(f.name for f in md.glob("*")))
    raise LoadError(f"没有找到权重文件（*.safetensors / pytorch_model.bin）：{raises}")


def load_causal_lm(model_dir: str | Path, dtype=None, note=None):
    """返回 (model, tokenizer, kind)。note 是日志回调，可为 None。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    md = Path(model_dir)
    if not md.is_dir():
        raise LoadError(f"模型目录不存在：{md}")
    trust_error = _code_trust_error(md)
    if trust_error:
        raise LoadError(trust_error)
    if note is None:
        note = lambda _m: None  # noqa: E731
    dtype = dtype or torch.float32
    cfg = read_config(md)

    tokenizer = AutoTokenizer.from_pretrained(str(md), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cls, kind = _find_lm_class(md, cfg, note)
    if cls is None:
        model = AutoModelForCausalLM.from_pretrained(str(md), dtype=dtype, local_files_only=True)
        model.config.use_cache = False
        return model, tokenizer, "transformers"

    cfg_cls = getattr(cls, "config_class")
    try:
        conf = cfg_cls(**cfg)
    except TypeError:
        conf = cfg_cls()
    model = cls(conf)
    sd = _load_state_dict(md)
    res = model.load_state_dict(sd, strict=False)
    if getattr(res, "missing_keys", None):
        note(f"权重缺失 {len(res.missing_keys)} 个键（通常是共享的 lm_head）：{res.missing_keys[:3]}")
    if getattr(res, "unexpected_keys", None):
        note(f"忽略 {len(res.unexpected_keys)} 个非文本权重（如 talker/vision）："
             f"{res.unexpected_keys[:3]}")
    model = model.to(dtype)
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, tokenizer, kind
