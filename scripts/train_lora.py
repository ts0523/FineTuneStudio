# -*- coding: utf-8 -*-
"""纯 CPU 可跑的 LoRA / 全参 SFT 微调脚本（transformers.Trainer + peft）。

示例:
    python scripts\\train_lora.py                          # 默认 LoRA，跑示例数据
    python scripts\\train_lora.py --epochs 1 --max-len 192
    python scripts\\train_lora.py --full                   # 全参微调（135M 参数量，CPU 也能跑）
    python scripts\\train_lora.py --data data\\my.jsonl --output outputs\\my-run
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "studio" / "workers"))
IGNORE_INDEX = -100


def default_model() -> Path:
    """与当前界面一致：优先 MiniMind-O，其次任意本地模型目录。"""
    preferred = ROOT / "models" / "minimind-3o"
    if (preferred / "config.json").is_file():
        return preferred
    for p in sorted((ROOT / "models").glob("*/config.json")):
        return p.parent
    return ROOT / "models" / "minimind-3o"


# ---------------------------------------------------------------- 数据格式归一化
ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}


def to_messages(example: dict) -> list[dict]:
    """把 4 种常见数据格式统一成 [{"role","content"}, ...]"""
    if example.get("messages"):
        return [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    if example.get("conversations"):
        out = []
        for m in example["conversations"]:
            raw = str(m.get("role") or m.get("from") or "").strip().lower()
            role = ROLE_MAP.get(raw)
            if role is None:
                raise ValueError(f"未知角色 {raw!r}")
            out.append({"role": role, "content": m.get("content") or m.get("value") or ""})
        return out
    if example.get("instruction") is not None:
        prompt = example.get("instruction") or ""
        if example.get("input"):
            prompt = f"{prompt}\n{example['input']}"
        return [{"role": "user", "content": prompt},
                {"role": "assistant", "content": example.get("output") or ""}]
    if example.get("prompt") is not None:
        return [{"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example.get("response") or ""}]
    raise ValueError(f"无法识别的数据格式，字段为: {list(example.keys())}")


def build_tokenize_fn(tokenizer, max_len: int):
    def _fn(example):
        msgs = to_messages(example)
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)

        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

        labels = list(full_ids)
        n = len(prompt_ids)
        # 只对 assistant 回复部分计算 loss；前缀对不上时退回全量训练并给出提示
        if full_ids[:n] == prompt_ids:
            labels[:n] = [IGNORE_INDEX] * n
        else:
            common = 0
            for a, b in zip(full_ids, prompt_ids):
                if a != b:
                    break
                common += 1
            labels[:common] = [IGNORE_INDEX] * common

        input_ids = full_ids[:max_len]
        labels = labels[:max_len]
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids)}

    return _fn


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(default_model()))
    ap.add_argument("--data", default="", help="JSONL dataset path; create one in the Data page")
    ap.add_argument("--output", default=str(ROOT / "outputs" / "lora-run"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=-1, help="大于 0 时只跑这么多步（冒烟测试用）")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full", action="store_true", help="全参微调，不做 LoRA")
    ap.add_argument("--threads", type=int, default=0, help="CPU 线程数，0=自动")
    args = ap.parse_args()

    if not args.data:
        print("[error] No dataset. Create one in the Data page or pass --data data/your_set.jsonl")
        return 2
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[错误] 模型目录不存在: {model_path}\n       先执行 python scripts\\download_model.py")
        return 1

    threads = args.threads or max(1, (os.cpu_count() or 4) // 2)
    torch.set_num_threads(threads)
    set_seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"模型      : {model_path}")
    print(f"数据      : {args.data}")
    print(f"输出      : {out_dir}")
    print(f"模式      : {'全参微调' if args.full else f'LoRA (r={args.lora_r}, alpha={args.lora_alpha})'}")
    print(f"设备      : CPU, torch 线程 {threads}, 进程 {torch.get_num_threads()}")
    print("=" * 66)

    from custom_model import load_causal_lm

    model, tokenizer, kind = load_causal_lm(model_path, torch.float32,
                                            note=lambda m: print(f"[模型] {m}"))
    print(f"加载方式  : {kind}")
    model.config.use_cache = False
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数量 {total_params/1e6:.1f}M")

    if not args.full:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[可训练] {trainable/1e6:.2f}M 参数 ({trainable/total_params*100:.2f}%)")

    # ------------------------------------------------ 数据
    ds = load_dataset("json", data_files=args.data, split="train")
    tok_fn = build_tokenize_fn(tokenizer, args.max_len)
    ds = ds.map(tok_fn, remove_columns=ds.column_names, desc="分词")
    ds = ds.filter(lambda x: any(v != IGNORE_INDEX for v in x["labels"]))
    print(f"[数据] 有效样本 {len(ds)} 条")

    if len(ds) >= 10 and args.val_ratio > 0:
        split = ds.train_test_split(test_size=args.val_ratio, seed=args.seed)
        train_ds, val_ds = split["train"], split["test"]
    else:
        train_ds, val_ds = ds, None
    print(f"[数据] 训练 {len(train_ds)} / 验证 {0 if val_ds is None else len(val_ds)}")

    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    print(f"[预估] 每 epoch 约 {steps_per_epoch} 个优化步；CPU 上 135M 模型约 3~8 秒/步，请耐心等待")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds is not None else "no",
        save_total_limit=2,
        report_to=[],
        use_cpu=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    used = time.time() - t0

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    meta = {
        "base_model": str(model_path),
        "mode": "full" if args.full else "lora",
        "lora_r": None if args.full else args.lora_r,
        "lora_alpha": None if args.full else args.lora_alpha,
        "train_samples": len(train_ds),
        "epochs": args.epochs,
        "max_len": args.max_len,
        "lr": args.lr,
        "seconds": round(used, 1),
        "final_metrics": {k: v for k, v in trainer.state.log_history[-1].items()
                          if isinstance(v, (int, float))},
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 66)
    print(f"[完成] 用时 {used/60:.1f} 分钟，产物目录: {out_dir}")
    print(f"[测试] python scripts\\chat.py --adapter \"{out_dir}\"")
    if not args.full:
        print(f"[合并] python scripts\\merge_lora.py --adapter \"{out_dir}\"")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
