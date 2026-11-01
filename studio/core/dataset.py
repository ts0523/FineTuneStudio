# -*- coding: utf-8 -*-
"""数据集：读取、校验、统计、清洗、切分、格式转换、演示数据生成。

支持 4 种行格式（自动识别）：
    {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
    {"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
    {"instruction":"...","input":"...","output":"..."}
    {"prompt":"...","response":"..."}
"""
from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from . import fsutil

ROLE_MAP = {
    "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
    "system": "system", "bot": "assistant", "ai": "assistant",
}
ALLOWED_ROLES = {"system", "user", "assistant"}
ZH_RE = re.compile(r"[\u4e00-\u9fff]")

FORMAT_HINT = "messages / conversations / instruction+output / prompt+response"


class DataError(Exception):
    pass


def _check_dialogue(msgs: list[dict]) -> None:
    """严格校验：角色集合、system 位置、user/assistant 交替、最后一条必须是 assistant。"""
    if not msgs:
        raise DataError("消息列表为空")
    roles = [m["role"] for m in msgs]
    for role in roles:
        if role not in ALLOWED_ROLES:
            raise DataError(f"未知角色 {role!r}（只支持 system/user/assistant）")
    body = roles
    if roles[0] == "system":
        if roles.count("system") > 1:
            raise DataError("system 消息只能有一条且必须在开头")
        body = roles[1:]
    elif "system" in roles:
        raise DataError("system 消息只能在对话开头")
    if not body:
        raise DataError("缺少 user/assistant 消息")
    if body[0] != "user":
        raise DataError("对话必须以 user 开头")
    if body[-1] != "assistant":
        raise DataError("最后一条消息必须是 assistant")
    for i, role in enumerate(body):
        want = "user" if i % 2 == 0 else "assistant"
        if role != want:
            raise DataError(f"消息角色顺序不合法（第 {i + 1 + (1 if roles[0] == 'system' else 0)} 条应为 {want}）")


def normalize(obj: dict) -> list[dict]:
    """把一条样本统一成 [{"role","content"}]；未知角色/不支持结构直接拒绝。"""
    if not isinstance(obj, dict):
        raise DataError("该行不是 JSON 对象")
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs:
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                raise DataError("messages 内有非对象元素")
            raw_role = str(m.get("role", "")).strip().lower()
            if raw_role not in ROLE_MAP:
                raise DataError(f"未知角色 {raw_role!r}（不会静默当成 user）")
            role = ROLE_MAP[raw_role]
            content = m.get("content")
            if content is None:
                content = m.get("value")
            if not isinstance(content, str):
                raise DataError("message.content 不是字符串")
            out.append({"role": role, "content": content})
        _check_dialogue(out)
        return out
    conv = obj.get("conversations")
    if isinstance(conv, list) and conv:
        out = []
        for m in conv:
            if not isinstance(m, dict):
                raise DataError("conversations 内有非对象元素")
            raw_role = str(m.get("from") or m.get("role") or "").strip().lower()
            if raw_role not in ROLE_MAP:
                raise DataError(f"未知角色 {raw_role!r}（不会静默当成 user）")
            role = ROLE_MAP[raw_role]
            content = m.get("value")
            if content is None:
                content = m.get("content")
            if not isinstance(content, str):
                raise DataError("conversation.value 不是字符串")
            out.append({"role": role, "content": content})
        _check_dialogue(out)
        return out
    if isinstance(obj.get("messages"), list) and not obj.get("messages"):
        raise DataError("messages 为空数组")
    if isinstance(obj.get("conversations"), list) and not obj.get("conversations"):
        raise DataError("conversations 为空数组")
    if obj.get("instruction") is not None:
        prompt = str(obj.get("instruction") or "")
        extra = str(obj.get("input") or "")
        if extra:
            prompt = f"{prompt}\n{extra}"
        out = [{"role": "user", "content": prompt},
               {"role": "assistant", "content": str(obj.get("output") or "")}]
        _check_dialogue(out)
        return out
    if obj.get("prompt") is not None:
        out = [{"role": "user", "content": str(obj.get("prompt") or "")},
               {"role": "assistant", "content": str(obj.get("response") or "")}]
        _check_dialogue(out)
        return out
    raise DataError(f"缺少必需字段，支持格式：{FORMAT_HINT}")


def normalize_record(obj: dict) -> dict:
    """统一消息格式，同时保留 image/audio 等媒体字段。"""
    out = {"messages": normalize(obj)}
    for key in ("image", "audio", "audio_text"):
        if obj.get(key):
            out[key] = str(obj[key])
    return out


def to_text(msgs: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)


def est_tokens(text: str) -> float:
    """粗估 token 数：中文 1.5 字符/token，其余 4 字符/token。"""
    zh = len(ZH_RE.findall(text))
    other = max(0, len(text) - zh)
    return zh / 1.5 + other / 4.0


def iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f, 1):
            yield i, line.strip()


@dataclass
class Issue:
    line: int
    message: str


@dataclass
class Report:
    path: Path
    total: int = 0
    valid: int = 0
    empty: int = 0
    duplicates: int = 0
    issues: list[Issue] = field(default_factory=list)
    formats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues and self.valid > 0


def _format_name(obj: dict) -> str:
    if obj.get("messages"):
        return "messages"
    if obj.get("conversations"):
        return "conversations"
    if obj.get("instruction") is not None:
        return "instruction"
    if obj.get("prompt") is not None:
        return "prompt"
    return "unknown"


def _dedupe_key(obj: dict, msgs: list[dict]) -> str:
    """去重键：文本 + 媒体路径。多模态样本即使文字相同，图片/语音不同也不算重复。"""
    media = obj.get("image") or obj.get("audio") or ""
    return f"{media}|{to_text(msgs)}".strip()


def validate(path: Path, max_issues: int = 50) -> Report:
    path = Path(path)
    rep = Report(path=path)
    if not path.is_file():
        rep.issues.append(Issue(0, f"文件不存在：{path}"))
        return rep
    if path.suffix.lower() not in (".jsonl", ".json", ".txt", ".csv"):
        rep.issues.append(Issue(0, f"不支持的扩展名 {path.suffix}，请用 .jsonl"))
        return rep
    seen: set[str] = set()
    try:
        for ln, text in iter_lines(path):
            if not text:
                continue
            rep.total += 1
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                rep.issues.append(Issue(ln, f"JSON 解析失败：{exc.msg}（第 {exc.colno} 列）"))
                continue
            try:
                msgs = normalize(obj)
            except DataError as exc:
                rep.issues.append(Issue(ln, str(exc)))
                continue
            if not any(m["role"] == "assistant" and m["content"].strip() for m in msgs):
                rep.empty += 1
                rep.issues.append(Issue(ln, "assistant 回复为空，训练时会被丢弃"))
                continue
            key = _dedupe_key(obj, msgs)
            if key in seen:
                rep.duplicates += 1
                rep.issues.append(Issue(ln, "与前面样本重复"))
                continue
            seen.add(key)
            rep.valid += 1
            f = _format_name(obj)
            rep.formats[f] = rep.formats.get(f, 0) + 1
            if len(rep.issues) > max_issues:
                break
    except OSError as exc:
        rep.issues.append(Issue(0, f"读取失败：{exc}"))
    return rep


def verify_after_write(path: Path, max_chars: int = 0) -> dict:
    """保存后重新校验：坏行、空回复、重复、超长、实际条数。"""
    path = Path(path)
    out = {"written": 0, "valid": 0, "bad": 0, "empty": 0, "dup": 0, "too_long": 0}
    if not path.is_file():
        return out
    seen: set[str] = set()
    for _ln, text in iter_lines(path):
        if not text:
            continue
        out["written"] += 1
        try:
            obj = json.loads(text)
            msgs = normalize(obj)
        except (json.JSONDecodeError, DataError):
            out["bad"] += 1
            continue
        flat = to_text(msgs)
        if not any(m["role"] == "assistant" and m["content"].strip() for m in msgs):
            out["empty"] += 1
            continue
        if max_chars and len(flat) > max_chars:
            out["too_long"] += 1
            continue
        key = _dedupe_key(obj, msgs)
        if key in seen:
            out["dup"] += 1
            continue
        seen.add(key)
        out["valid"] += 1
    return out


def stats(path: Path) -> dict:
    """统计条数、长度分布、角色分布、重复率、中英占比、token 粗估。"""
    path = Path(path)
    out = {
        "n": 0, "valid": 0, "dups": 0, "min_chars": 0, "max_chars": 0,
        "avg_chars": 0.0, "avg_prompt_chars": 0.0, "avg_answer_chars": 0.0,
        "total_chars": 0, "est_tokens": 0.0, "roles": {}, "zh_ratio": 0.0,
        "turns": 0, "empty": 0,
    }
    if not path.is_file():
        return out
    seen: set[str] = set()
    lengths: list[int] = []
    zh_chars = 0
    all_chars = 0
    roles: dict[str, int] = {}
    prompt_chars = 0
    answer_chars = 0
    tokens = 0.0
    for ln, text in iter_lines(path):
        if not text:
            continue
        out["n"] += 1
        try:
            obj = json.loads(text)
            msgs = normalize(obj)
        except (json.JSONDecodeError, DataError):
            out["empty"] += 1
            continue
        flat = to_text(msgs)
        if not any(m["role"] == "assistant" and m["content"].strip() for m in msgs):
            out["empty"] += 1
            continue
        key = _dedupe_key(obj, msgs)
        if key in seen:
            out["dups"] += 1
            continue
        seen.add(key)
        out["valid"] += 1
        out["turns"] += len(msgs)
        lengths.append(len(flat))
        all_chars += len(flat)
        zh_chars += len(ZH_RE.findall(flat))
        tokens += est_tokens(flat)
        for m in msgs:
            roles[m["role"]] = roles.get(m["role"], 0) + 1
        for m in msgs:
            if m["role"] == "user":
                prompt_chars += len(m["content"])
            elif m["role"] == "assistant":
                answer_chars += len(m["content"])
    if lengths:
        out["min_chars"] = min(lengths)
        out["max_chars"] = max(lengths)
        out["avg_chars"] = sum(lengths) / len(lengths)
    if out["valid"]:
        out["avg_prompt_chars"] = prompt_chars / max(1, roles.get("user", 1))
        out["avg_answer_chars"] = answer_chars / max(1, roles.get("assistant", 1))
    out["total_chars"] = all_chars
    out["est_tokens"] = tokens
    out["roles"] = roles
    out["zh_ratio"] = (zh_chars / all_chars) if all_chars else 0.0
    return out


def preview(path: Path, n: int = 12, max_chars: int = 400) -> list[dict]:
    items: list[dict] = []
    path = Path(path)
    if not path.is_file():
        return items
    for ln, text in iter_lines(path):
        if not text:
            continue
        try:
            obj = json.loads(text)
            msgs = normalize(obj)
        except (json.JSONDecodeError, DataError) as exc:
            items.append({"line": ln, "error": str(exc), "messages": []})
        else:
            cut = []
            for m in msgs:
                c = m["content"]
                cut.append({"role": m["role"], "content": c[:max_chars] + ("…" if len(c) > max_chars else "")})
            items.append({"line": ln, "error": "", "messages": cut})
        if len(items) >= n:
            break
    return items


def write_jsonl(path: Path, samples: Iterable[dict], fmt: str = "messages") -> int:
    path = Path(path)
    lines = []
    for msgs in samples:
        obj = {fmt: msgs} if fmt == "messages" else msgs
        lines.append(json.dumps(obj, ensure_ascii=False))
    return fsutil.atomic_write_lines(path, lines)


def load_samples(path: Path) -> list[list[dict]]:
    out: list[list[dict]] = []
    for ln, text in iter_lines(Path(path)):
        if not text:
            continue
        try:
            out.append(normalize(json.loads(text)))
        except (json.JSONDecodeError, DataError):
            continue
    return out


def convert(src: Path, dst: Path, fmt: str = "messages") -> tuple[int, int]:
    """把任意支持格式统一成指定格式，返回 (写出条数, 跳过条数)。"""
    samples = load_samples(src)
    skipped = 0
    written: list[dict] = []
    for msgs in samples:
        if not any(m["role"] == "assistant" and m["content"].strip() for m in msgs):
            skipped += 1
            continue
        if fmt == "messages":
            written.append({"messages": msgs})
        elif fmt == "conversations":
            conv = [{"from": "human" if m["role"] == "user" else ("gpt" if m["role"] == "assistant" else "system"),
                     "value": m["content"]} for m in msgs]
            written.append({"conversations": conv})
        elif fmt == "prompt":
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            ans = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            written.append({"prompt": user, "response": ans})
        else:
            raise ValueError(f"未知目标格式：{fmt}")
    dst = Path(dst)
    lines = [json.dumps(obj, ensure_ascii=False) for obj in written]
    fsutil.atomic_write_lines(dst, lines)
    return len(written), skipped


def clean(src: Path, dst: Path, max_chars: int = 0, dedupe: bool = True,
          drop_empty: bool = True) -> dict:
    """去重 / 去空 / 过滤超长。先写临时文件，成功后再原子替换；失败保留原文件。"""
    seen: set[str] = set()
    stat = {"in": 0, "kept": 0, "dup": 0, "empty": 0, "too_long": 0, "bad": 0}
    lines: list[str] = []
    for ln, text in iter_lines(Path(src)):
        if not text:
            continue
        stat["in"] += 1
        try:
            obj = json.loads(text)
            msgs = normalize(obj)
        except (json.JSONDecodeError, DataError):
            stat["bad"] += 1
            continue
        flat = to_text(msgs)
        if drop_empty and not any(m["role"] == "assistant" and m["content"].strip() for m in msgs):
            stat["empty"] += 1
            continue
        if max_chars and len(flat) > max_chars:
            stat["too_long"] += 1
            continue
        key = _dedupe_key(obj, msgs)
        if dedupe and key in seen:
            stat["dup"] += 1
            continue
        seen.add(key)
        lines.append(json.dumps(obj, ensure_ascii=False))
    stat["kept"] = fsutil.atomic_write_lines(Path(dst), lines)
    stat["verified"] = verify_after_write(Path(dst), max_chars=max_chars)
    return stat


def split(src: Path, train_out: Path, val_out: Path, ratio: float = 0.1, seed: int = 42) -> tuple[int, int]:
    records: list[dict] = []
    for _ln, text in iter_lines(Path(src)):
        if not text:
            continue
        try:
            records.append(normalize_record(json.loads(text)))
        except (json.JSONDecodeError, DataError):
            continue
    rnd = random.Random(seed)
    rnd.shuffle(records)
    n_val = int(round(len(records) * max(0.0, min(0.5, ratio))))
    val = records[:n_val]
    train = records[n_val:]
    fsutil.atomic_write_lines(train_out, [json.dumps(r, ensure_ascii=False) for r in train])
    fsutil.atomic_write_lines(val_out, [json.dumps(r, ensure_ascii=False) for r in val])
    return len(train), len(val)


# ---------------------------------------------------------------- 导入 / 模板
def import_table(src: Path, dst: Path) -> tuple[int, str]:
    """把 csv / tsv / txt 的「问题,回答」表格转成 jsonl。返回 (条数, 说明)。"""
    src = Path(src)
    suffix = src.suffix.lower()
    rows: list[tuple[str, str]] = []
    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        with src.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delim)
            data = list(reader)
        if data:
            head = [c.strip().lower() for c in data[0][:2]]
            keys = {"question", "问题", "q", "prompt", "instruction", "input",
                    "answer", "回答", "a", "response", "output", "label"}
            if any(h in keys for h in head):
                data = data[1:]
        for row in data:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                rows.append((row[0].strip(), row[1].strip()))
    elif suffix in (".txt", ".md"):
        text = src.read_text(encoding="utf-8-sig", errors="replace")
        blocks = re.split(r"\n\s*\n", text)
        for b in blocks:
            m = re.search(r"(?:Q|问|问题)\s*[:：]\s*(.+?)\n\s*(?:A|答|回答)\s*[:：]\s*([\s\S]+)", b.strip(), re.I)
            if m:
                rows.append((m.group(1).strip(), m.group(2).strip()))
        if not rows:
            for line in text.splitlines():
                if "\t" in line:
                    q, a = line.split("\t", 1)
                    if q.strip() and a.strip():
                        rows.append((q.strip(), a.strip()))
    else:
        return 0, f"不支持的导入格式 {suffix}（支持 .csv/.tsv/.txt/.md）"
    if not rows:
        return 0, "没有解析出「问题-回答」对，请检查文件内容"
    dst = Path(dst)
    lines = [json.dumps({"messages": [{"role": "user", "content": q},
                                      {"role": "assistant", "content": a}]},
                        ensure_ascii=False) for q, a in rows]
    fsutil.atomic_write_lines(dst, lines)
    return len(rows), f"已转换 {len(rows)} 条 → {dst.name}"


FAQ_SEED = [
    ("你是谁？", "我是 Nova 智能助手，一个在本机微调过的小模型，可以回答产品与售后问题。"),
    ("Nova Cloud 的退款政策是什么？", "购买后 30 天内可无理由全额退款，发邮件到 support@nova.dev 即可，无需说明原因。"),
    ("Pro 版多少钱？", "Pro 版每月 19 美元，按年付费 190 美元，相当于免费送两个月。"),
    ("支持哪些操作系统？", "桌面端支持 Windows 10/11、macOS 13 及以上、Ubuntu 22.04 及以上。"),
    ("上传一直失败怎么办？", "先检查代理设置，然后加 -resume 参数重试；仍失败请把 logs 目录打包发到 support@nova.dev。"),
    ("能开发票吗？", "可以。进入 设置 → 账单 → 发票，点击生成，一分钟内会收到 PDF。"),
    ("数据会加密吗？", "会。静态数据使用 AES-256 加密，传输使用 TLS 1.3，密钥每 90 天轮换一次。"),
    ("免费版有什么限制？", "免费版包含 5 GB 存储空间与每月 10 GB 流量，单文件上限 100 MB。"),
    ("怎么联系人工客服？", "发邮件到 support@nova.dev，工作日内 24 小时回复。"),
    ("支持团队协作吗？", "支持。在 团队 → 成员 里邀请成员，可分别设置只读、可编辑、管理员三种权限。"),
    ("如何导出我的数据？", "进入 设置 → 数据 → 导出，选择 CSV 或 JSON，导出任务完成后会邮件通知下载链接。"),
    ("忘记密码怎么办？", "在登录页点击「忘记密码」，输入注册邮箱，重置链接 30 分钟内有效。"),
]

FAQ_SEED_EN = [
    ("Who are you?", "I am the Nova assistant, a small model fine-tuned locally to answer product and support questions."),
    ("What is the refund policy?", "Full refund within 30 days, no questions asked. Just email support@nova.dev."),
    ("How much is the Pro plan?", "19 USD per month, or 190 USD per year with two months free."),
    ("Which platforms are supported?", "Windows 10/11, macOS 13+, and Ubuntu 22.04+ on desktop."),
    ("My upload keeps failing.", "Check your proxy first, retry with -resume; if it still fails, send the logs to support@nova.dev."),
    ("Can I get an invoice?", "Go to Settings -> Billing -> Invoices, click Generate, the PDF arrives within a minute."),
    ("Is my data encrypted?", "Yes, AES-256 at rest and TLS 1.3 in transit, keys rotate every 90 days."),
    ("What are the free tier limits?", "5 GB storage and 10 GB monthly traffic, 100 MB per file."),
]


def make_seed(topic: str = "产品客服", n: int = 24, bilingual: bool = True) -> list[dict]:
    """演示数据：把内置 FAQ 组合成单轮 + 多轮样本，用于验证训练链路。"""
    base_zh = list(FAQ_SEED)
    base_en = list(FAQ_SEED_EN)
    samples: list[dict] = []
    system = f"你是{topic}助手 Nova，回答要简短、准确，只讲事实。"
    i = 0
    while len(samples) < n:
        q, a = base_zh[i % len(base_zh)]
        samples.append({"messages": [{"role": "user", "content": q},
                                     {"role": "assistant", "content": a}]})
        if len(samples) >= n:
            break
        if bilingual:
            qe, ae = base_en[i % len(base_en)]
            samples.append({"messages": [{"role": "user", "content": qe},
                                         {"role": "assistant", "content": ae}]})
            if len(samples) >= n:
                break
        q2, a2 = base_zh[(i + 3) % len(base_zh)]
        samples.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
            {"role": "user", "content": "谢谢，那" + q2},
            {"role": "assistant", "content": a2},
        ]})
        i += 1
    return samples[:n]


def default_dataset_name(files: list[dict], last: str = "") -> str:
    """挑默认数据集：上次用过 → sample_sft.jsonl → 非下划线开头 → 第一个。"""
    names = [f["name"] for f in files]
    if not names:
        return ""
    if last and last in names:
        return last
    if "sample_sft.jsonl" in names:
        return "sample_sft.jsonl"
    clean = [n for n in names if not n.startswith(("_", "."))]
    return (clean or names)[0]


def list_datasets() -> list[dict]:
    from . import paths

    out = []
    d = paths.data_dir()
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.jsonl")):
        try:
            st = f.stat()
        except OSError:
            continue
        out.append({"name": f.name, "path": f, "size": st.st_size, "mtime": st.st_mtime})
    return out
