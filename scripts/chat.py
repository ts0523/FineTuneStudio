# -*- coding: utf-8 -*-
"""加载基座模型（可选挂 LoRA 适配器）做对话测试。

用法:
    python scripts\\chat.py                                  # 交互式，用基座模型
    python scripts\\chat.py --adapter outputs\\lora-run       # 交互式，挂上微调后的适配器
    python scripts\\chat.py --adapter outputs\\lora-run --prompt "你是谁？"   # 单次问答
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "studio" / "workers"))


def default_model() -> Path:
    preferred = ROOT / "models" / "minimind-3o"
    if (preferred / "config.json").is_file():
        return preferred
    for p in sorted((ROOT / "models").glob("*/config.json")):
        return p.parent
    return ROOT / "models" / "minimind-3o"


def load(model_dir: Path, adapter: str | None):
    from custom_model import load_causal_lm

    model, tokenizer, kind = load_causal_lm(model_dir, torch.float32,
                                            note=lambda m: print(f"[模型] {m}"))
    print(f"[加载方式] {kind}")
    if adapter:
        from peft import PeftModel

        adapter_path = Path(adapter)
        if not adapter_path.exists():
            raise SystemExit(f"[错误] 适配器目录不存在: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print(f"[已加载适配器] {adapter_path}")
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate(tokenizer, model, messages, max_new_tokens=128, temperature=0.7):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0][ids["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(default_model()))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    tokenizer, model = load(Path(args.model), args.adapter)
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        print(generate(tokenizer, model, messages, args.max_new_tokens, args.temperature))
        return 0

    print("交互模式：输入内容回车发送，输入 exit 退出，输入 /reset 清空上下文。")
    while True:
        try:
            user = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit", "q"}:
            break
        if user == "/reset":
            messages = [m for m in messages if m["role"] == "system"]
            print("[已清空上下文]")
            continue
        messages.append({"role": "user", "content": user})
        reply = generate(tokenizer, model, messages, args.max_new_tokens, args.temperature)
        print(f"AI > {reply}")
        messages.append({"role": "assistant", "content": reply})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
