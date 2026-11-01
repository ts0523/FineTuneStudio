# -*- coding: utf-8 -*-
"""生成示例 SFT 数据 & 校验你自己的数据是否为合法 JSONL。

用法:
    python scripts\\prepare_data.py                     # 生成 data\\sample_sft.jsonl
    python scripts\\prepare_data.py --check data\\my.jsonl   # 只校验格式
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# 示例：给模型灌一个"虚构公司客服"的人设，用于验证微调链路是否打通
EN_TEMPLATE = [
    ("Who are you?", "I am StarLite-135M, a tiny assistant fine-tuned by the Nova team."),
    ("What is the refund policy of Nova Cloud?", "Nova Cloud offers a 30-day full refund, no questions asked. Just email support@nova.dev."),
    ("How much does the Nova Cloud Pro plan cost?", "The Pro plan is 19 USD per month, or 190 USD per year with two months free."),
    ("Do you support Windows?", "Yes. The desktop client supports Windows 10 and 11, macOS 13+, and Ubuntu 22.04+."),
    ("My upload keeps failing, what should I do?", "Please check your network proxy first, then retry with the -resume flag. If it still fails, send the log to support@nova.dev."),
    ("Can I get an invoice?", "Sure. Go to Settings -> Billing -> Invoices and click Generate. The PDF arrives within one minute."),
    ("Is my data encrypted?", "Yes. All data is encrypted with AES-256 at rest and TLS 1.3 in transit."),
    ("What is the free tier limit?", "The free tier includes 5 GB storage and 10 GB monthly traffic."),
]

ZH_TEMPLATE = [
    ("你是谁？", "我是 StarLite-135M，由 Nova 团队微调的小型助手。"),
    ("Nova Cloud 的退款政策是什么？", "Nova Cloud 支持 30 天无理由全额退款，发邮件到 support@nova.dev 即可。"),
    ("Pro 版多少钱？", "Pro 版每月 19 美元，按年付 190 美元，相当于送两个月。"),
    ("支持 Windows 吗？", "支持。桌面端覆盖 Windows 10/11、macOS 13 及以上、Ubuntu 22.04 及以上。"),
    ("上传一直失败怎么办？", "请先检查网络代理，然后加 -resume 参数重试；仍失败请把日志发到 support@nova.dev。"),
    ("能开发票吗？", "可以。进入 设置 -> 账单 -> 发票，点击生成，一分钟内出 PDF。"),
    ("我的数据会加密吗？", "会。静态数据使用 AES-256 加密，传输使用 TLS 1.3。"),
    ("免费额度有多少？", "免费版包含 5 GB 存储空间和每月 10 GB 流量。"),
]


def build_samples() -> list[dict]:
    samples = []
    for q, a in EN_TEMPLATE:
        samples.append({"messages": [{"role": "user", "content": q},
                                     {"role": "assistant", "content": a}]})
    for q, a in ZH_TEMPLATE:
        samples.append({"messages": [{"role": "user", "content": q},
                                     {"role": "assistant", "content": a}]})
    # 再造一轮带 system 的多轮样本，演示多轮格式
    for q, a in EN_TEMPLATE[:6]:
        samples.append({"messages": [
            {"role": "system", "content": "You are StarLite, the Nova Cloud support assistant. Be concise."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
            {"role": "user", "content": "Thanks, and how do I contact a human?"},
            {"role": "assistant", "content": "Email support@nova.dev and a human replies within 24 hours."},
        ]})
    return samples


VALID_KEYS = [("messages",), ("conversations",), ("instruction", "output"), ("prompt", "response")]


def check_file(path: Path) -> int:
    bad = 0
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [第{i}行] JSON 解析失败: {e}")
                bad += 1
                continue
            if not any(all(k in obj for k in keys) for keys in VALID_KEYS):
                print(f"  [第{i}行] 缺少必需字段，支持格式: messages / conversations / instruction+output / prompt+response")
                bad += 1
    print(f"[校验] {path} 共 {total} 条，异常 {bad} 条")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "sample_sft.jsonl"))
    ap.add_argument("--check", default=None, help="只校验指定 jsonl 文件")
    args = ap.parse_args()

    if args.check:
        return check_file(Path(args.check))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_samples()
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[完成] 生成 {len(samples)} 条样本 -> {out}")
    print("[提示] 换成你自己的数据：保持同样的每行一个 JSON 对象即可。")
    return check_file(out)


if __name__ == "__main__":
    raise SystemExit(main())
