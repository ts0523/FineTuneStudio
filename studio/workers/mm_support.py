# -*- coding: utf-8 -*-
"""把多模态样本接进训练/对话的公共逻辑。

约定：
- 样本里带 "image" 或 "audio" 字段即为多模态样本；
- prompt 里的 <|image_pad|> / <|audio_pad|> 是占位符，训练/推理时被替换成编码器特征；
- 编码器冻结，特征按「文件路径+大小+mtime」缓存在 data/.mm_cache/*.npz；
- 投影层（MMProjector）可训练，训练产物存成 <out>/mm_projector.pt。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import mm_encoder as mm

VISION_DIM = mm.VISION_DIM
AUDIO_DIM = mm.AUDIO_DIM
LLM_DIM = mm.LLM_DIM
PROJ_FILE = "mm_projector.pt"

PAD_TOKEN = {"image": "<|image_pad|>", "audio": "<|audio_pad|>"}
PAD_COUNT = {"image": mm.VISION_TOKENS, "audio": mm.AUDIO_TOKENS}


def expected_tokens(kind: str) -> int:
    if kind not in PAD_COUNT:
        raise ValueError(f"未知媒体类型：{kind}")
    return int(PAD_COUNT[kind])


def expand_placeholder(text: str, kind: str) -> str:
    """把内容里的占位符统一成「编码器实际特征行数」个，训练/推理共用。

    数据里可以只写一个 <|image_pad|>；这里会替换成 VISION_TOKENS 个，
    保证 embedding 注入时占位符数量与特征行数严格相等。
    """
    if kind not in PAD_TOKEN:
        raise ValueError(f"未知媒体类型：{kind}")
    token = PAD_TOKEN[kind]
    n = expected_tokens(kind)
    raw = str(text or "")
    idx = raw.find(token)
    if idx >= 0:
        cleaned = raw.replace(token, "")
        return cleaned[:idx] + token * n + cleaned[idx:]
    return token * n + raw


def expand_messages(msgs: list[dict], kind: str) -> list[dict]:
    """把最后一条 user 消息的占位符展开；训练与推理都走这里。"""
    out = [dict(m) for m in msgs]
    for m in reversed(out):
        if m.get("role") == "user":
            m["content"] = expand_placeholder(str(m.get("content") or ""), kind)
            break
    return out


def count_placeholders(ids, pad_id: int) -> int:
    return int(sum(1 for x in ids if int(x) == int(pad_id)))


# ---------------------------------------------------------------- 缓存
def media_key(sample: dict) -> str:
    for key in ("image", "audio"):
        p = sample.get(key)
        if not p:
            continue
        path = Path(str(p))
        try:
            st = path.stat()
            raw = f"{key}|{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            raw = f"{key}|{p}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
    return ""


def media_kind(sample: dict) -> str:
    if sample.get("image"):
        return "image"
    if sample.get("audio"):
        return "audio"
    return ""


def attach_features(rows: list[dict], cache_dir: Path, root: Path, emit,
                    audio_dir: Path | None = None) -> dict:
    """给带媒体文件的行补上 feat 张量（优先读缓存）。返回统计。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = audio_dir or (root / "models" / "_encoders" / "whisper-tiny")
    vision = audio = None
    stats = {"image": 0, "audio": 0, "cached": 0, "failed": 0}
    todo = [r for r in rows if r.get("mm")]
    for i, row in enumerate(todo, 1):
        src = row["mm"]
        kind = src["kind"]
        key = src["key"] or media_key({"image": src.get("path")} if kind == "image"
                                      else {"audio": src.get("path")})
        npz = cache_dir / f"{kind}-{key}.npz"
        feat = None
        if npz.is_file():
            try:
                with np.load(npz) as d:
                    feat = d["feat"]
                stats["cached"] += 1
            except (OSError, KeyError, ValueError):
                feat = None
        if feat is None:
            path = Path(str(src.get("path")))
            if not path.is_file():
                stats["failed"] += 1
                continue
            try:
                if kind == "image":
                    if vision is None:
                        emit(type="log", level="info",
                             message="加载视觉编码器 MobileNetV3-Small（2.5M，冻结）")
                        vision = mm.load_vision_encoder()
                    feat = mm.encode_image(vision[0], vision[1], path)
                else:
                    if audio is None:
                        emit(type="log", level="info",
                             message="加载音频编码器 Whisper-tiny encoder（8M，冻结）")
                        audio = mm.load_audio_encoder(audio_dir)
                    feat = mm.encode_audio(audio, path)
                np.savez_compressed(npz, feat=np.asarray(feat, dtype=np.float32))
            except Exception as exc:  # noqa: BLE001
                emit(type="log", level="warn", message=f"特征提取失败 {path.name}：{exc}")
                stats["failed"] += 1
                continue
        row["feat"] = torch.from_numpy(np.asarray(feat, dtype=np.float32))
        stats[kind] += 1
        if i % 5 == 0 or i == len(todo):
            emit(type="stage", name="mm_feat", done=i, total=len(todo),
                 message=f"提取多模态特征 {i}/{len(todo)}")
    return stats


# ---------------------------------------------------------------- 训练侧
def _unwrap(m):
    """找到真正持有 embed_tokens 的模块（穿透 PeftModel / LoraModel 的包装）。"""
    cur = m
    for _ in range(6):
        if hasattr(cur, "embed_tokens"):
            return cur
        nxt = getattr(cur, "model", None)
        if nxt is None or nxt is cur:
            nxt = getattr(cur, "base_model", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    raise AttributeError("模型里找不到 embed_tokens，无法注入多模态特征")


def _feature_map(item, kind: str = "") -> dict:
    """把单条样本的特征统一成 {"image": tensor, "audio": tensor}。"""
    if item is None:
        return {}
    if isinstance(item, dict):
        return {k: v for k, v in item.items() if k in ("image", "audio") and v is not None}
    if not torch.is_tensor(item):
        return {}
    if kind in ("image", "audio"):
        return {kind: item}
    if item.dim() == 2 and int(item.shape[1]) == AUDIO_DIM:
        return {"audio": item}
    return {"image": item}


class MMAdapter:
    """挂到模型上的多模态适配器：投影层 + embedding 注入钩子。"""

    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        self.img_id = tok.convert_tokens_to_ids("<|image_pad|>")
        self.aud_id = tok.convert_tokens_to_ids("<|audio_pad|>")
        host = _unwrap(model)
        self.host = host
        self.embed = host.embed_tokens
        self.proj_v = mm.MMProjector(VISION_DIM)
        self.proj_a = mm.MMProjector(AUDIO_DIM)
        # 挂成子模块：这样 model.parameters() 能带上投影层，Trainer 会一起优化
        host.mm_projector_v = self.proj_v
        host.mm_projector_a = self.proj_a
        self.pending: dict = {"feats": None, "ids": None}
        self.loaded_modalities: list[str] = []
        self.embed.register_forward_hook(self._hook)

    def _hook(self, _module, inputs, output):
        feats = self.pending["feats"]
        if feats is None or not inputs:
            return output
        # 用本次 forward 真实拿到的 input_ids：生成阶段只喂新 token，那里没有占位符，
        # 自然不会重复注入（特征已经在 KV cache 里）
        ids = inputs[0]
        if not torch.is_tensor(ids):
            return output
        out = output.clone()
        kinds = (("image", self.img_id, self.proj_v, VISION_DIM),
                 ("audio", self.aud_id, self.proj_a, AUDIO_DIM))
        for b in range(ids.shape[0]):
            item = feats[b] if b < len(feats) else None
            feature_map = _feature_map(item)
            for kind, pad_id, projector, dim in kinds:
                pos = (ids[b] == pad_id).nonzero().flatten()
                if len(pos) == 0:
                    continue
                feat = feature_map.get(kind)
                if feat is None:
                    raise ValueError(f"提示里有 {kind} 占位符，但没有加载对应的{kind}投影特征")
                if int(feat.shape[1]) != dim:
                    raise ValueError(f"{kind} 特征维度 {feat.shape[1]} 与投影层 {dim} 不一致")
                n_pad, n_feat = int(len(pos)), int(feat.shape[0])
                if n_pad != n_feat:
                    raise ValueError(
                        f"{kind} 占位符数量与编码器特征行数不一致：占位符 {n_pad} 个，"
                        f"特征 {n_feat} 行。训练和推理都必须用 mm_support.expand_placeholder "
                        "统一数量，禁止静默截断。")
                # 投影放在 forward 里做：每次新建计算图，图的寿命严格限制在本次 forward，
                # 避免跨 step 复用导致 "backward through the graph a second time"
                out[b, pos] = projector(feat.to(output.device)).to(out.dtype)
        return out

    def collate_feats(self, batch_rows: list[dict]) -> None:
        """只记录原始特征（不带计算图），投影在 forward 的钩子里做。"""
        self.pending["feats"] = [_feature_map(r.get("feat"), str(r.get("kind") or ""))
                                 for r in batch_rows]

    def set_features(self, feats: list) -> None:
        """推理时用：单条样本可同时带图像/语音特征，None 表示纯文本。"""
        self.pending["feats"] = [_feature_map(f) for f in feats]

    def save(self, out_dir: Path, meta: dict | None = None) -> Path:
        p = Path(out_dir) / PROJ_FILE
        torch.save({"vision": self.proj_v.state_dict(), "audio": self.proj_a.state_dict(),
                    "vision_dim": VISION_DIM, "audio_dim": AUDIO_DIM, "llm_dim": LLM_DIM,
                    "meta": meta or {}}, p)
        return p

    @staticmethod
    def load(model, tok, path: Path) -> "MMAdapter | None":
        return MMAdapter.load_composite(model, tok, [Path(path)])

    @staticmethod
    def load_composite(model, tok, paths: list[Path]) -> "MMAdapter | None":
        """从多个适配器组合视觉/语音投影层，支持图文与语音共同理解。"""
        files = [Path(p) for p in paths if Path(p).is_file()]
        if not files:
            return None
        ad = MMAdapter(model, tok)
        loaded = {"image": False, "audio": False}
        for path in files:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
            raw_meta = ckpt.get("meta")
            meta: dict = raw_meta if isinstance(raw_meta, dict) else {}
            raw_stats = meta.get("stats")
            stats: dict = raw_stats if isinstance(raw_stats, dict) else {}
            mods = []
            if int(stats.get("image") or 0) > 0:
                mods.append("image")
            if int(stats.get("audio") or 0) > 0:
                mods.append("audio")
            if not mods:
                mods = ["image", "audio"]
            if "image" in mods and "vision" in ckpt:
                ad.proj_v.load_state_dict(ckpt["vision"])
                loaded["image"] = True
            if "audio" in mods and "audio" in ckpt:
                ad.proj_a.load_state_dict(ckpt["audio"])
                loaded["audio"] = True
        if not any(loaded.values()):
            return None
        ad.loaded_modalities = [k for k, v in loaded.items() if v]
        return ad


def projector_params(adapter: MMAdapter) -> list[torch.nn.Parameter]:
    return list(adapter.proj_v.parameters()) + list(adapter.proj_a.parameters())


def describe() -> dict:
    return mm.describe()


if __name__ == "__main__":
    print(json.dumps(describe(), ensure_ascii=False, indent=1))
