# -*- coding: utf-8 -*-
"""AI 帮训：给闭环提供「找资源 / 出验证题 / 打分 / 下一轮建议」四个能力。

闭环（具体调度在 ui/pages/autotrain.py）：
  联网找公开资源 → AI 自编训练数据 → 训练 → 用实验台同一套权重答题 → 打分 → 下一轮
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
import uuid
from pathlib import Path

from . import ai, dataset as ds, fsutil, hubclient, paths

_SYS_JUDGE = ("你是严格的微调验收员。只输出一个合法 JSON 对象，不要解释、不要 Markdown。"
              "评分为 0~100 的整数：答对事实且简洁给高分，答非所问/编造给低分。")


# ---------------------------------------------------------------- 会话 / 协议
def new_session_id() -> str:
    """每次帮训运行的唯一 ID；旧任务回调必须用它与当前会话比对。"""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:10]


def session_ok(expected: str, current: str) -> bool:
    return bool(expected) and expected == current


def run_fields_from_done(evt: dict) -> dict:
    """从 train_worker 的 done 事件读取指标（与普通训练页同一协议）。"""
    metrics: dict = evt["metrics"] if isinstance(evt.get("metrics"), dict) else {}
    status = str(evt.get("status") or "done")
    return {
        "status": status,
        "final_loss": metrics.get("final_loss"),
        "best_eval_loss": metrics.get("best_eval_loss"),
        "tokens": metrics.get("tokens"),
        "tokens_per_sec": metrics.get("tokens_per_sec"),
        "steps_done": metrics.get("steps_done"),
        "seconds": evt.get("total_seconds"),
        "out_dir": evt.get("output"),
        "metrics": metrics,
        "phase": "完成" if status == "done" else "已停止",
    }


def merge_outcome(verify_enabled: bool, verified_ok: bool | None, code: int | None) -> str:
    """合并导出终态：written（仅写出）/ verified（验证通过）/ failed。"""
    if code not in (0, None):
        return "failed"
    if verify_enabled and verified_ok is not True:
        return "failed"
    return "verified" if verify_enabled else "written"


# ---------------------------------------------------------------- holdout
def sample_key(sample: dict) -> str:
    msgs = sample.get("messages") or []
    q = next((str(m.get("content") or "") for m in msgs if m.get("role") == "user"), "")
    a = next((str(m.get("content") or "") for m in msgs if m.get("role") == "assistant"), "")
    return hashlib.sha1(f"{q.strip()}\n{a.strip()}".encode("utf-8")).hexdigest()


def question_key(sample: dict) -> str:
    """按题干去重：同一道题换个答案也不能进入训练集（避免题目泄漏）。"""
    msgs = sample.get("messages") or []
    q = next((str(m.get("content") or "") for m in msgs if m.get("role") == "user"), "")
    return hashlib.sha1(q.strip().encode("utf-8")).hexdigest()


def _holdout_candidates(samples: list[dict]) -> list[dict]:
    out = []
    for s in samples:
        msgs = s.get("messages") or []
        if len(msgs) != 2:
            continue
        q, a = msgs[0].get("content") or "", msgs[1].get("content") or ""
        if 4 <= len(q) <= 160 and 2 <= len(a) <= 260:
            out.append(s)
    return out


def make_holdout(samples: list[dict], n: int, seed: int = 7) -> tuple[list[dict], list[dict], str]:
    """从首批数据中抽出固定 holdout，返回 (holdout, 可训练样本, holdout_id)。

    只在第一轮调用一次；之后各轮都复用同一份 holdout，且训练集必须剔除它。
    """
    cands = _holdout_candidates(samples)
    rnd = random.Random(seed)
    rnd.shuffle(cands)
    holdout: list[dict] = []
    keys: set[str] = set()
    for s in cands:
        k = question_key(s)
        if k in keys:
            continue
        keys.add(k)
        holdout.append(s)
        if len(holdout) >= max(1, int(n)):
            break
    train = [s for s in samples if question_key(s) not in keys]
    ids = [sample_key(s) for s in holdout]
    holdout_id = hashlib.sha1(("\n".join(ids)).encode("utf-8")).hexdigest()[:16]
    return holdout, train, holdout_id


def filter_holdout(samples: list[dict], holdout: list[dict]) -> list[dict]:
    keys = {question_key(s) for s in holdout}
    return [s for s in samples if question_key(s) not in keys]


def holdout_ids(holdout: list[dict]) -> list[str]:
    return [sample_key(s) for s in holdout]


def save_holdout(path: Path, holdout: list[dict], meta: dict) -> None:
    lines = [json.dumps({"messages": s["messages"], "holdout_id": sample_key(s)},
                        ensure_ascii=False) for s in holdout]
    fsutil.atomic_write_lines(path, lines)
    fsutil.atomic_write_json(Path(str(path) + ".meta.json"), meta, indent=1)


def load_holdout(path: Path) -> list[dict]:
    out: list[dict] = []
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            msgs = ds.normalize(obj)
        except Exception:  # noqa: BLE001
            continue
        out.append({"messages": msgs})
    return out


def holdout_leaks(dataset_path: Path, holdout: list[dict]) -> list[str]:
    """检查训练集里是否混入 holdout 题（按题干，答案不同也算泄漏）。"""
    keys = {question_key(s) for s in holdout}
    leaked: list[str] = []
    p = Path(dataset_path)
    if not p.is_file():
        return leaked
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            sample = {"messages": ds.normalize(json.loads(line))}
        except Exception:  # noqa: BLE001
            continue
        if question_key(sample) in keys:
            q = next((m["content"] for m in sample["messages"] if m["role"] == "user"), "")
            leaked.append(q[:80])
    return leaked


def data_version(path: Path) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    raw = p.read_bytes()
    n = sum(1 for line in raw.decode("utf-8", "replace").splitlines() if line.strip())
    return f"{hashlib.sha256(raw).hexdigest()[:20]}-{n}"


# ---------------------------------------------------------------- 历史
def history_path(session: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", session)[:80] or "session"
    return paths.state_dir() / "autotrain" / f"{safe}.json"


def write_history(path: Path, record: dict) -> None:
    fsutil.atomic_write_json(Path(path), record, indent=1)


def append_history(path: Path, base: dict, round_record: dict) -> dict:
    """读旧记录 → 追加本轮评分 → 原子替换；返回最新记录。"""
    p = Path(path)
    record = dict(base)
    rounds: list[dict] = []
    if p.is_file():
        try:
            old = json.loads(p.read_text(encoding="utf-8-sig"))
            if isinstance(old, dict):
                record.update({k: v for k, v in old.items() if k not in ("rounds",)})
                rounds = list(old.get("rounds") or [])
        except (OSError, ValueError):
            rounds = []
    rounds.append(round_record)
    record["rounds"] = rounds
    write_history(p, record)
    return record


def dataset_path(goal: str) -> Path:
    base = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", (goal or "autotrain").strip())[:36]
    return paths.data_dir() / f"autotrain-{base or 'round'}.jsonl"


def eval_pairs(samples: list[dict], n: int = 6, seed: int = 7) -> list[dict]:
    """从样本里抽验证题（只取单轮、答案非空、长度适中）。"""
    pairs: list[dict] = []
    for s in samples:
        msgs = s.get("messages") or []
        q = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        if q and a and 4 <= len(q) <= 160 and 2 <= len(a) <= 260:
            pairs.append({"q": q, "a": a})
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    return pairs[:max(1, n)]


def _bigrams(text: str) -> set[str]:
    t = re.sub(r"[\s，。,.!！？?：:；;、\"'“”‘’（）()\[\]]+", "", text or "")
    return {t[i:i + 2] for i in range(max(0, len(t) - 1))} or ({t} if t else set())


def similarity(answer: str, target: str) -> float:
    """字符 bigram 重合度：0~1，中文短答案也能比较。"""
    a, b = _bigrams(answer), _bigrams(target)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def local_score(items: list[dict], answers: list[str]) -> dict:
    scores = [similarity(answers[i] if i < len(answers) else "", it["a"])
              for i, it in enumerate(items)]
    avg = sum(scores) / max(1, len(scores))
    weak = [items[i]["q"] for i, s in enumerate(scores) if s < 0.35][:3]
    return {"score": round(avg * 100), "verdict": "本地一致性评分", "weak": "",
            "advice": ("重点补强这些问题的答案：" + "；".join(weak)) if weak else "",
            "detail": scores, "judge_source": "local"}


def judge(items: list[dict], answers: list[str], goal: str, cancel=None) -> dict:
    """先用本地重合度算一个保守分，再让 AI 复核并给出下一轮要补什么。"""
    local = local_score(items, answers)
    if not ai.enabled() or not items:
        return local
    lines = []
    for i, it in enumerate(items[:5]):
        got = (answers[i] if i < len(answers) else "").strip()[:160] or "（空回复）"
        lines.append(f"问题：{it['q']}\n标准答案：{it['a'][:120]}\n模型回答：{got}")
    prompt = (f"训练目标：{goal}\n下面是微调后的模型在验证题上的真实表现。\n\n"
              + "\n\n".join(lines)
              + '\n\n只输出 JSON：{"score":0-100,"verdict":"一句话结论",'
                '"weak":"没答对的方面","advice":"下一轮该补什么样的数据（一句话）"}')
    try:
        obj, reply = ai.chat_json([ai.user_msg(prompt)], system=_SYS_JUDGE,
                                  max_tokens=600, temperature=0.2, timeout=45,
                                  cancel=cancel)
    except ai.AIError:
        return local
    if not isinstance(obj, dict):
        return local
    try:
        ai_score = int(float(obj.get("score", local["score"])))
    except (TypeError, ValueError):
        ai_score = local["score"]
    return {
        "score": max(0, min(100, round((ai_score + local["score"]) / 2))),
        "verdict": str(obj.get("verdict") or local["verdict"])[:160],
        "weak": str(obj.get("weak") or "")[:160],
        "advice": str(obj.get("advice") or local.get("advice") or "")[:200],
        "detail": local.get("detail"),
        "ai_score": ai_score,
        "ai_model": reply.model,
        "judge_source": "ai+local",
    }


def round_brief(goal: str, advice: str, resources: list[dict], n: int, kind: str) -> str:
    """把目标 + 上一轮建议 + 联网找到的资源拼成造数描述。"""
    parts = [f"训练目标：{goal}",
             "这批数据将用于监督微调一个 0.1B 的中文小模型，要求问题真实、答案具体、可验证。"]
    if advice:
        parts.append(f"上一轮验收发现：{advice}。本轮必须针对性地补齐这部分。")
    if resources:
        names = "、".join(x["id"] for x in resources[:4])
        parts.append(f"联网找到了这些公开数据集可作为风格/范围参考（不直接下载，只参考主题与字段）："
                     f"{names}。")
    parts.append(f"本轮生成 {n} 条「{kind}」样本，不要与已训练数据重复。")
    return "\n".join(parts)


_KEYWORD_MAP = [
    ("客服", "customer service"), ("售后", "customer service"), ("退货", "customer service"),
    ("问答", "question answering"), ("对话", "dialogue"), ("聊天", "chat"),
    ("翻译", "translation"), ("代码", "code"), ("编程", "code"),
    ("摘要", "summarization"), ("抽取", "information extraction"),
    ("分类", "text classification"), ("情感", "sentiment"), ("审核", "moderation"),
    ("医疗", "medical"), ("法律", "legal"), ("金融", "finance"), ("教育", "education"),
    ("数学", "math"), ("推理", "reasoning"), ("知识", "knowledge"), ("写作", "writing"),
    ("作文", "writing"), ("诗歌", "poetry"), ("风格", "style"), ("角色", "roleplay"),
]


def _fallback_queries(goal: str) -> list[str]:
    """HF 的数据集搜索是按词命中的：中文目标先映射成英文短语，再退到单个英文词。"""
    hits = [en for zh, en in _KEYWORD_MAP if zh in (goal or "")]
    uniq: list[str] = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    words = [w for h in uniq[:2] for w in h.split() if len(w) > 2]
    if not uniq:
        import re as _re

        words = _re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", goal or "")[:3]
    queries = uniq[:3] + [w for w in words if w not in uniq]
    return queries[:4]


def search_resources(goal: str, endpoint: str = "", limit: int = 10) -> tuple[list[dict], str]:
    """联网搜索公开数据集（只读元信息，遵守许可）。返回 (结果, 错误)。"""
    try:
        ep = endpoint or str(ai.cfg().get("endpoint") or "https://hf-mirror.com")
        items = hubclient.search_datasets(goal[:80], ep, limit)
        # 中文目标直接搜不到（搜索索引是英文/短语命中），逐级退到英文关键词
        for q in [] if items else _fallback_queries(goal):
            items = hubclient.search_datasets(q, ep, limit)
            if items:
                break
        return (items or []), ""
    except hubclient.HubError as exc:
        return [], str(exc)
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


def summarize_resources(items: list[dict]) -> str:
    if not items:
        return "没找到可参考的公开资源，本轮改用纯自编数据"
    bits = []
    for x in items[:5]:
        lic = f" · {x['license']}" if x.get("license") else ""
        bits.append(f"{x['id']}（下载 {x['downloads']}{lic}）")
    return "参考资源：" + "；".join(bits)


def write_meta(path: Path, meta: dict) -> None:
    try:
        fsutil.atomic_write_json(Path(str(path) + ".meta.json"), meta, indent=1)
    except OSError:
        pass


def count_lines(path: Path) -> int:
    try:
        return sum(1 for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
                   if ln.strip())
    except OSError:
        return 0


def load_samples(path: Path, limit: int = 400) -> list[dict]:
    out: list[dict] = []
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = __import__("json").loads(line)
                msgs = ds.normalize(obj)
            except Exception:  # noqa: BLE001
                continue
            out.append({"messages": msgs})
            if len(out) >= limit:
                break
    except OSError:
        pass
    return out
