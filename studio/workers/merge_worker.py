# -*- coding: utf-8 -*-
"""LoRA 合并 worker：把适配器权重合进基座，产出可直接加载的完整模型目录。

cfg: {"model","adapter","out","verify":true}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

EVT = "@@EVT@@"


def emit(**kw) -> None:
    kw.setdefault("t", round(time.time(), 3))
    sys.stdout.write(EVT + json.dumps(kw, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    sys.stdout.write(str(msg).rstrip() + "\n")
    sys.stdout.flush()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def final_status(verify_enabled: bool, verify_ok: bool | None) -> tuple[str, int]:
    """合并终态：启用验证但没通过时必须是 failed + 非零退出码。"""
    if verify_enabled and verify_ok is not True:
        return "failed", 5
    return "done", 0


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
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    watch_parent()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))

    base_dir = Path(cfg["model"])
    adapter = Path(cfg["adapter"])
    out = Path(cfg["out"])
    root_outputs = (Path(os.environ.get("FTS_ROOT", str(Path.cwd()))).resolve() / "outputs")
    try:
        out.resolve().relative_to(root_outputs)
    except ValueError:
        emit(type="error", message=f"输出目录越界：{out} 不在 {root_outputs} 内")
        return 3
    if not base_dir.is_dir():
        emit(type="error", message=f"基座模型不存在：{base_dir}")
        return 3
    if not (adapter / "adapter_config.json").is_file():
        emit(type="error", message=f"适配器不存在或缺少 adapter_config.json：{adapter}")
        return 3
    try:
        out_r, base_r, ad_r = out.resolve(), base_dir.resolve(), adapter.resolve()
    except OSError:
        out_r, base_r, ad_r = out, base_dir, adapter
    if out_r == base_r or out_r in base_r.parents:
        emit(type="error", message="输出目录不能是基座模型目录或其子目录，否则会覆盖基座权重")
        return 3
    if out_r == ad_r or out_r in ad_r.parents:
        emit(type="error", message="输出目录不能是适配器目录或其子目录，否则会覆盖适配器")
        return 3

    import torch
    from custom_model import load_causal_lm

    try:
        emit(type="stage", name="tokenizer", message="保存 tokenizer")
        out.mkdir(parents=True, exist_ok=True)

        emit(type="stage", name="load", message=f"加载基座（fp32）{base_dir.name}")
        base, tok, kind = load_causal_lm(
            base_dir, torch.float32,
            note=lambda m: emit(type="log", level="info", message=m))
        tok.save_pretrained(str(out))

        emit(type="stage", name="merge", message="合并 LoRA 权重")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(adapter))
        model = model.merge_and_unload()
        if hasattr(model, "config"):
            model.config.use_cache = True

        emit(type="stage", name="save", message=f"写出 {out}")
        # 自定义架构的模型常有共享权重，safetensors 会因重复张量报错，这里用 bin
        model.save_pretrained(str(out), safe_serialization=(kind == "transformers"))

        for extra in ("generation_config.json", "chat_template.jinja", ".fts_trust.json",
                      "mm_projector.pt"):
            src = base_dir / extra
            if src.is_file():
                shutil.copyfile(src, out / extra)
        proj = adapter / "mm_projector.pt"
        if proj.is_file():
            shutil.copyfile(proj, out / proj.name)
        # 自带建模代码的模型：把代码一起复制，合并产物才能被再次加载
        for py in base_dir.glob("model*.py"):
            shutil.copyfile(py, out / py.name)
        for extra in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                      "vocab.json", "merges.txt"):
            src = base_dir / extra
            if src.is_file() and not (out / extra).is_file():
                shutil.copyfile(src, out / extra)
    except Exception as exc:  # noqa: BLE001
        import traceback
        log(traceback.format_exc()[-2000:])
        emit(type="error", message=f"合并失败：{type(exc).__name__}: {exc}")
        return 4

    size = dir_size(out)
    emit(type="merged", out=str(out), bytes=size)
    log(f"[合并] 合并模型已写出：{out}（{size/1024/1024:.1f} MB）")

    verify_ok: bool | None = None
    verify_error = ""
    if cfg.get("verify", True):
        emit(type="stage", name="verify", message="验证合并结果（加载并生成 8 个 token）")
        try:
            m2, tok2, _kind2 = load_causal_lm(out, torch.float32,
                                              note=lambda m: emit(type="log", level="info", message=m))
            m2 = m2.eval()
            text = tok2.apply_chat_template([{"role": "user", "content": "你好"}],
                                            tokenize=False, add_generation_prompt=True)
            ids = tok2(text, return_tensors="pt", add_special_tokens=False)
            t0 = time.time()
            with torch.inference_mode():
                gen = m2.generate(inputs=ids["input_ids"], attention_mask=ids.get("attention_mask"),
                                  max_new_tokens=8, do_sample=False,
                                  pad_token_id=tok2.pad_token_id or tok2.eos_token_id)
            reply = tok2.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            verify_ok = True
            emit(type="verified", ok=True, sample=reply.strip()[:120],
                 seconds=round(time.time() - t0, 2))
        except Exception as exc:  # noqa: BLE001
            verify_ok = False
            verify_error = f"{type(exc).__name__}: {exc}"
            emit(type="verified", ok=False, message=verify_error)

    status, code = final_status(bool(cfg.get("verify", True)), verify_ok)
    if status != "done":
        # 文件写出 ≠ 验证通过：保持失败终态，返回非零，不能被 done 覆盖成成功
        emit(type="done", status="failed", out=str(out), bytes=size, written=True,
             verified=False, message=f"合并文件已写出，但加载验证失败：{verify_error}")
        log("[失败] 合并文件已写出，但验证未通过")
        return code

    emit(type="done", status="done", out=str(out), bytes=size, written=True,
         verified=(True if verify_ok else None))
    return code


if __name__ == "__main__":
    sys.exit(main())
