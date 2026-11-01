# -*- coding: utf-8 -*-
"""环境自检：依赖版本、CPU/内存、模型文件、一次最小推理。

用法: python scripts\\check_env.py
"""
import platform
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK，输出 ✔/✘ 会抛 UnicodeEncodeError，这里强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "studio" / "workers"))


def pick_model() -> Path:
    """与界面一致：优先 MiniMind-O，其次任意含 config.json 的本地模型目录。"""
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    preferred = ROOT / "models" / "minimind-3o"
    if (preferred / "config.json").is_file():
        return preferred
    for p in sorted((ROOT / "models").glob("*/config.json")):
        return p.parent
    return preferred


MODEL_DIR = pick_model()

ok, fail = [], []


def check(name, fn):
    try:
        info = fn()
        ok.append((name, str(info)))
    except Exception as e:  # noqa: BLE001
        fail.append((name, f"{type(e).__name__}: {e}"))


def main() -> int:
    print("=" * 66)
    print(f"Python : {sys.version.split()[0]}  ({sys.executable})")
    print(f"系统   : {platform.system()} {platform.release()}  {platform.machine()}")
    print("=" * 66)

    def _torch():
        import torch
        return f"{torch.__version__} | cuda={torch.cuda.is_available()} | threads={torch.get_num_threads()}"

    def _libs():
        import transformers
        import peft
        import datasets
        import accelerate
        return (f"transformers {transformers.__version__}, peft {peft.__version__}, "
                f"datasets {datasets.__version__}, accelerate {accelerate.__version__}")

    def _mem():
        import psutil
        vm = psutil.virtual_memory()
        return f"总 {vm.total/1024**3:.1f} GB, 可用 {vm.available/1024**3:.1f} GB, CPU {psutil.cpu_count()} 逻辑核"

    def _files():
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"模型目录不存在: {MODEL_DIR}")
        if not (MODEL_DIR / "config.json").is_file():
            raise FileNotFoundError(f"缺少 config.json: {MODEL_DIR}")
        if not any(p.suffix in (".safetensors", ".bin") for p in MODEL_DIR.iterdir()):
            raise FileNotFoundError(f"缺少权重文件（*.safetensors / *.bin）：{MODEL_DIR}")
        size = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file())
        return f"{MODEL_DIR.name} ({size/1024**2:.0f} MB)"

    def _infer():
        import torch
        from custom_model import load_causal_lm
        t0 = time.time()
        model, tok, kind = load_causal_lm(MODEL_DIR, torch.float32,
                                          note=lambda m: print(f"  [模型] {m}"))
        model.eval()
        n = sum(p.numel() for p in model.parameters())
        msgs = [{"role": "user", "content": "Say hello in one short sentence."}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return (f"{n/1e6:.1f}M 参数, 加载方式 {kind}, 耗时 {time.time()-t0:.1f}s, "
                f"回复: {reply[:60]!r}")

    check("torch", _torch)
    check("依赖库", _libs)
    check("硬件", _mem)
    check("模型文件", _files)
    check("最小推理", _infer)

    print(f"模型   : {MODEL_DIR}")
    print("\n[通过]")
    for k, v in ok:
        print(f"  ✔ {k}: {v}")
    if fail:
        print("\n[失败]")
        for k, v in fail:
            print(f"  ✘ {k}: {v}")
        return 1
    print("\n环境可用。下一步: python scripts\\train_lora.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
