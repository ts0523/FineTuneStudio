# -*- coding: utf-8 -*-
"""环境体检 worker：逐项检查依赖、硬件、模型文件，并做一次真实推理。

用法：python doctor_worker.py --model <模型目录>（可省略）
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

EVT = "@@EVT@@"


def emit(**kw) -> None:
    kw.setdefault("t", round(time.time(), 3))
    sys.stdout.write(EVT + json.dumps(kw, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def check(name: str, fn):
    t0 = time.time()
    try:
        detail = fn()
        emit(type="check", name=name, ok=True, detail=str(detail)[:400],
             ms=round((time.time() - t0) * 1000))
        return True
    except Exception as exc:  # noqa: BLE001
        emit(type="check", name=name, ok=False, detail=f"{type(exc).__name__}: {exc}"[:400],
             ms=round((time.time() - t0) * 1000))
        return False


def watch_parent() -> None:
    """父进程（界面）消失后自动退出，避免留下孤儿进程继续吃 CPU。"""
    import ctypes
    import threading

    pid = os.environ.get("FTS_PARENT_PID", "")
    if not pid.isdigit():
        return
    ppid = int(pid)

    def loop() -> None:
        k32 = ctypes.windll.kernel32
        while True:
            time.sleep(2.0)
            handle = k32.OpenProcess(0x00100000, False, ppid)  # SYNCHRONIZE
            if not handle:
                os._exit(9)
            k32.CloseHandle(handle)

    threading.Thread(target=loop, daemon=True).start()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--config", default="")
    args = ap.parse_args()
    watch_parent()
    model = args.model
    if args.config:
        import json
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        model = model or str(cfg.get("model") or "")
    model_dir = Path(model) if model else None

    emit(type="start", python=sys.version.split()[0], exe=sys.executable)

    check("Python", lambda: f"{sys.version.split()[0]} · {platform.machine()}")

    def _torch():
        import torch
        return f"torch {torch.__version__} · cuda={torch.cuda.is_available()} · 线程 {torch.get_num_threads()}"

    check("PyTorch", _torch)

    def _libs():
        import transformers
        import peft
        import datasets
        import accelerate
        return f"transformers {transformers.__version__} · peft {peft.__version__} · datasets {datasets.__version__} · accelerate {accelerate.__version__}"

    check("训练依赖", _libs)

    def _hw():
        import psutil
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_count()
        return f"{cpu} 逻辑核 · 内存 {vm.total/1024**3:.1f} GB（可用 {vm.available/1024**3:.1f} GB）"

    check("硬件", _hw)

    def _disk():
        import psutil
        root = os.environ.get("FTS_PROBE_ROOT") or "."
        du = psutil.disk_usage(root)
        return f"已用 {du.used/1024**3:.1f} GB / 共 {du.total/1024**3:.1f} GB · 剩余 {du.free/1024**3:.1f} GB"

    check("磁盘", _disk)

    if model_dir is not None:
        def _files():
            if not model_dir.is_dir():
                raise FileNotFoundError(f"目录不存在：{model_dir}")
            need = ["config.json"]
            miss = [n for n in need if not (model_dir / n).is_file()]
            if miss:
                raise FileNotFoundError(f"缺少 {miss}")
            weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
            if not weights:
                raise FileNotFoundError("没有权重文件（*.safetensors / *.bin）")
            size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            return f"{model_dir.name} · {len(weights)} 个权重文件 · {size/1024**2:.0f} MB"

        ok_files = check("模型文件", _files)
        if ok_files:
            def _tok():
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
                return f"{type(tok).__name__} · 词表 {tok.vocab_size} · chat_template={'有' if tok.chat_template else '无'}"

            check("Tokenizer", _tok)

            def _infer():
                import torch
                from custom_model import load_causal_lm
                t0 = time.time()
                model, tok, kind = load_causal_lm(model_dir, torch.float32)
                model = model.eval()
                n = sum(p.numel() for p in model.parameters())
                msgs = [{"role": "user", "content": "你好，用一句话介绍你自己。"}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = tok(text, return_tensors="pt", add_special_tokens=False)
                with torch.inference_mode():
                    out = model.generate(inputs=ids["input_ids"],
                                         attention_mask=ids.get("attention_mask"),
                                         max_new_tokens=16, do_sample=False,
                                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
                reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
                return (f"[{kind}] {n/1e6:.1f}M 参数 · 加载+推理 {time.time()-t0:.1f}s · "
                        f"回复 {reply.strip()[:60]!r}")

            check("最小推理", _infer)

    emit(type="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
