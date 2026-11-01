# -*- coding: utf-8 -*-
"""数据层核心：用一句自然语言描述，让云端免费模型写出训练数据，并反推训练参数。

分工：
- 数据内容由 OpenRouter 免费模型生成（结构化 JSON，失败会自动降级换模型）
- 训练参数由 planner 按「真实样本量 / 平均 token / 任务类型 / 本机实测吞吐」算，
  不是拍脑袋；界面上的预估耗时与训练页用的是同一套模型
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import ai, dataset as ds, fsutil, paths, planner

# 界面上的数据类型（前三种纯文本，后两种是多模态）
KINDS: list[tuple[str, str]] = [
    ("qa", "问答对（单轮，最常用）"),
    ("multiturn", "多轮对话（带上下文）"),
    ("faq", "FAQ 知识（同一主题多问题）"),
    ("image", "图像问答（合成图 + 标准答案）"),
    ("audio", "语音问答（把文本问题读成语音）"),
]

TASKS = ["客服问答", "风格模仿", "信息抽取", "翻译", "代码补全", "自定义"]

_SYS = ("你在为一个本机小模型（MiniMind-O 0.1B）编写监督微调（SFT）数据。"
        "只输出一个合法 JSON 对象，不要 Markdown、不要解释。"
        "数据要具体、自然、可验证；问题之间不能重复；答案不要空话套话。"
        "涉及数字、价格、流程时给出明确值。")

_SCHEMA = {
    "qa": '{"samples":[{"user":"问题","assistant":"回答"}]}',
    "faq": '{"samples":[{"user":"问题","assistant":"回答"}]}',
    "multiturn": ('{"samples":[{"messages":[{"role":"user","content":"..."},'
                  '{"role":"assistant","content":"..."},'
                  '{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}]}'),
}


class NLError(Exception):
    pass


# ---------------------------------------------------------------- 生成
def _prompt(description: str, n: int, kind: str, lang: str, topic: str,
            system_hint: str, avoid: str) -> str:
    schema = _SCHEMA.get(kind, _SCHEMA["qa"])
    parts = [
        f"任务：根据下面的描述，生成 {n} 条训练样本。",
        f"领域/主题：{topic or '由描述决定'}",
        f"语言：{lang}",
        f"数据用途：{description.strip()}",
    ]
    if kind == "multiturn":
        parts.append("格式：每条样本 2~3 轮真实对话（用户提问→助手回答→追问→回答），"
                     "追问要承接上文，不要换话题。")
    elif kind == "faq":
        parts.append("格式：围绕同一主题，从不同角度提问（价格、流程、限制、售后、对比等），"
                     "每个问题都要能被同一份资料回答。")
    else:
        parts.append("格式：单轮问答，问题要像真实用户会问的，答案 20~120 字。")
    if system_hint:
        parts.append(f"人设/风格要求：{system_hint}")
    if avoid:
        parts.append(f"不要出现这些内容：{avoid}")
    parts.append(f"严格输出：{schema}")
    parts.append("samples 数组长度必须等于上面的条数，字段名大小写不能改。")
    return "\n".join(parts)


def _normalize_items(obj: dict | list | None) -> list[dict]:
    if not isinstance(obj, (dict, list)):
        return []
    items = obj.get("samples") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        # 有些模型会直接返回 {"user":..., "assistant":...}
        if isinstance(obj, dict) and ("user" in obj or "question" in obj or "q" in obj):
            items = [obj]
        else:
            return []
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if isinstance(it.get("messages"), list):
            msgs = []
            for m in it["messages"]:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "").strip().lower()
                if role in ("assistant", "gpt", "ai", "bot"):
                    role = "assistant"
                elif role in ("user", "human"):
                    role = "user"
                elif role == "system":
                    role = "system"
                else:
                    msgs = []      # 未知角色：整条拒绝，不静默变成 user
                    break
                content = str(m.get("content") or m.get("value") or "").strip()
                if content:
                    msgs.append({"role": role, "content": content})
            if msgs and msgs[0]["role"] == "user" and any(m["role"] == "assistant" for m in msgs):
                try:
                    out.append({"messages": ds.normalize({"messages": msgs})})
                except ds.DataError:
                    pass
            continue
        q = str(it.get("user") or it.get("question") or it.get("q") or "").strip()
        a = str(it.get("assistant") or it.get("answer") or it.get("a") or "").strip()
        if q and a:
            out.append({"messages": [{"role": "user", "content": q},
                                     {"role": "assistant", "content": a}]})
    return out


def generate(description: str, n: int = 24, kind: str = "qa", lang: str = "中文",
             topic: str = "", system_hint: str = "", avoid: str = "",
             batch: int = 10, progress=None, cancel=None) -> tuple[list[dict], str]:
    """返回 (样本列表, 使用的模型名)。progress(done, total) 用于界面进度。

    cancel：可选取消检查（返回 True 时不再发起下一轮请求）。
    """
    if kind in ("image", "audio"):
        return [], ""
    if not ai.enabled():
        raise NLError("还没有配置 AI Key：到「设置 → AI 功能」粘贴 OpenRouter Key")
    desc = (description or "").strip()
    if len(desc) < 4:
        raise NLError("先用一句话描述你要的数据（至少 4 个字）")
    n = max(4, min(200, int(n)))
    batch = max(4, min(16, int(batch)))
    samples: list[dict] = []
    used = ""
    rounds = 0
    avoid_extra = avoid
    while len(samples) < n and rounds < 12:
        rounds += 1
        if cancel is not None and cancel():
            raise NLError("已取消")
        want = min(batch, n - len(samples))
        prompt = _prompt(desc, want, kind, lang, topic, system_hint, avoid_extra)
        obj, reply = ai.chat_json([ai.user_msg(prompt)], system=_SYS, max_tokens=2600,
                                  temperature=0.75, timeout=60, cancel=cancel)
        items = _normalize_items(obj)
        if not items:
            # 结构化输出失败时退一步：直接让模型按「问：/答：」写纯文本，再解析
            text = ai.ask(prompt + "\n改用纯文本，每条用「问：」和「答：」两行。",
                          system=_SYS, max_tokens=3200, temperature=0.7, cancel=cancel)
            items = _from_text(text, want)
        if not items:
            raise NLError("模型没有返回可用的问答对，换个描述或稍后再试")
        used = reply.model
        for it in items:
            if len(samples) >= n:
                break
            key = ds.to_text(it["messages"]).strip()
            if key and all(ds.to_text(s["messages"]).strip() != key for s in samples):
                samples.append(it)
        if progress:
            progress(len(samples), n)
        if len(items) < max(1, want // 2):
            avoid_extra = (avoid_extra + "；不要重复以下问题："
                           + "、".join(m["messages"][0]["content"][:20] for m in samples[-6:]))
    if len(samples) < n:
        # 拿到的条数不够就如实返回，不重复填充（重复数据只会让模型背答案）
        pass
    return samples, used


def _from_text(text: str, want: int) -> list[dict]:
    out: list[dict] = []
    blocks = re.split(r"\n\s*\n", text or "")
    for b in blocks:
        m = re.search(r"(?:问|Q|用户)\s*[:：]\s*(.+?)\n\s*(?:答|A|助手)\s*[:：]\s*([\s\S]+)",
                      b.strip(), re.I)
        if m:
            out.append({"messages": [{"role": "user", "content": m.group(1).strip()},
                                     {"role": "assistant", "content": m.group(2).strip()}]})
        if len(out) >= want:
            break
    return out


# ---------------------------------------------------------------- 多模态
def make_media(kind: str, source: Path | None = None, per_combo: int = 6,
               limit: int = 16, progress=None) -> tuple[Path, int, str]:
    """生成图像/语音问答集，返回 (数据集路径, 条数, 说明)。"""
    from . import multimodal

    if kind == "image":
        if progress:
            progress(0, 1)
        dst, n = multimodal.make_image_dataset(per_combo=max(2, min(12, per_combo)))
        return dst, n, (f"已合成 {n} 张图像问答（形状/颜色/数量随机，答案为固定句式）。"
                        "图像是教学级合成数据，换真实照片后需要重新标定。")
    dst, n, tts_err = multimodal.make_audio_dataset(
        source_jsonl=Path(source) if source else None, limit=max(4, min(40, limit)))
    note = (f"已用系统 TTS 把 {n} 条文本问题读成语音（语音提问 → 文本答案）。"
            if n else "没有可用的文本问答数据：先在数据层生成/导入一份问答集，再转语音。")
    if tts_err:
        note += f" TTS 提示：{tts_err}"
    return dst, n, note


# ---------------------------------------------------------------- 保存
def safe_name(name: str) -> str:
    base = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", (name or "").strip()).strip("._")
    return (base or f"nl_{time.strftime('%m%d-%H%M%S')}")[:48]


def save(samples: list[dict], name: str, meta: dict | None = None) -> Path:
    filename = safe_name(name)
    if not filename.endswith(".jsonl"):
        filename += ".jsonl"
    try:
        dst = fsutil.safe_child(paths.data_dir(), filename)
    except fsutil.PathSafetyError as exc:
        raise OSError(f"数据集名不安全：{exc}") from exc
    ds.write_jsonl(dst, [s["messages"] for s in samples])
    if meta:
        try:
            fsutil.atomic_write_json(paths.data_dir() / (filename + ".meta.json"), meta, indent=1)
        except OSError:
            pass
    return dst


def append_to(path: Path, samples: list[dict]) -> int:
    """把样本追加到已有 jsonl（读原文 + 追加 + 原子替换，失败保留原文件）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            lines.extend(line.rstrip("\n") for line in f if line.strip())
    for s in samples:
        lines.append(json.dumps({"messages": s["messages"]}, ensure_ascii=False))
    fsutil.atomic_write_lines(path, lines)
    return len(samples)


# ---------------------------------------------------------------- 训练参数
def suggest(samples: list[dict], task: str = "客服问答", quality: str = "usable",
            model_name: str = "", params_m: float = 64.0, threads: int = 4,
            speed_samples: list[dict] | None = None, goal: str = "") -> dict:
    """按真实数据量与任务类型反推训练参数（与训练页共用同一估算模型）。"""
    flat = "\n".join(ds.to_text(s["messages"]) for s in samples)
    tokens = ds.est_tokens(flat)
    avg_tokens = tokens / max(1, len(samples))
    plan = planner.make_plan(goal or task, task, quality, model_name or "（当前基座）",
                             float(params_m or 64.0), len(samples), avg_tokens,
                             speed_samples=speed_samples, threads=int(threads or 4))
    return {
        "epochs": plan.epochs, "batch_size": plan.batch_size,
        "grad_accum": plan.grad_accum, "max_len": plan.max_len,
        "lora_r": plan.lora_r, "lora_alpha": plan.lora_alpha,
        "lora_dropout": plan.lora_dropout, "lr": plan.lr,
        "val_ratio": plan.val_ratio, "samples": plan.samples,
        "steps": plan.steps, "est_seconds": plan.est_seconds,
        "est_ram_gb": plan.est_ram_gb, "avg_tokens": round(avg_tokens, 1),
        "notes": plan.notes, "task": task, "quality": quality,
    }


def plan_summary(p: dict) -> str:
    from .paths import human_time

    bits = [
        f"样本 {p.get('samples', 0)} 条 · 平均 {p.get('avg_tokens', 0)} token/条",
        f"{p.get('epochs', 0):g} 轮 × batch {p.get('batch_size', 1)} × 累积 "
        f"{p.get('grad_accum', 1)} → {p.get('steps', 0)} 步",
        f"max_len {p.get('max_len', 256)} · LoRA r {p.get('lora_r', 16)} · lr {p.get('lr', 0):g}",
        f"预计 {human_time(p.get('est_seconds', 0))} · 内存约 {p.get('est_ram_gb', 0):.1f} GB",
    ]
    return " | ".join(bits)


def quality_tiers() -> list[tuple[str, str, str]]:
    return [(k, v[0], f"约 {v[2]['epochs']:g} 轮 / r={v[2]['lora_r']}")
            for k, v in planner.QUALITY_TIERS.items()]
