# -*- coding: utf-8 -*-
"""把 LoRA 适配器合并进基座权重，产出一个可直接部署的完整模型目录。

用法:
    python scripts\\merge_lora.py --adapter outputs\\lora-run
    python scripts\\merge_lora.py --adapter outputs\\lora-run --out outputs\\merged
"""
import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "SmolLM2-135M-Instruct"))
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "lora-run"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "merged"))
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not adapter.exists():
        raise SystemExit(f"[错误] 适配器目录不存在: {adapter}")

    print(f"[加载] 基座 {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.save_pretrained(args.out)

    # 合并需在 fp32 下进行，避免精度损失
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, local_files_only=True)
    from peft import PeftModel

    print(f"[合并] 适配器 {adapter}")
    model = PeftModel.from_pretrained(base, str(adapter))
    model = model.merge_and_unload()
    model.config.use_cache = True

    model.save_pretrained(args.out, safe_serialization=True)

    gen_cfg = Path(args.model) / "generation_config.json"
    if gen_cfg.exists():
        shutil.copyfile(gen_cfg, Path(args.out) / "generation_config.json")

    total = sum(f.stat().st_size for f in Path(args.out).rglob("*") if f.is_file())
    print(f"[完成] 合并模型已保存到 {args.out}（{total/1024/1024:.1f} MB）")
    print(f"[测试] python scripts\\chat.py --model \"{args.out}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
