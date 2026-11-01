# -*- coding: utf-8 -*-
"""对话 worker：常驻进程，加载一次模型，按行接收 JSON 指令并流式返回 token。

启动：
    python chat_worker.py --config chat_cfg.json
指令（stdin，每行一个 JSON）：
    {"cmd":"chat","messages":[{"role":"user","content":"..."}],"params":{...}}
    {"cmd":"stop"}
    {"cmd":"quit"}
输出（stdout，@@EVT@@ 前缀）：
    ready / start / token / done / stopped / error
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

EVT = "@@EVT@@"


def emit(**kw) -> None:
    kw.setdefault("t", round(time.time(), 3))
    sys.stdout.write(EVT + json.dumps(kw, ensure_ascii=False) + "\n")
    sys.stdout.flush()


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

    import torch
    from custom_model import load_causal_lm

    model_dir = Path(cfg["model"])
    adapter_paths = [Path(p) for p in (cfg.get("adapters") or []) if str(p).strip()]
    if cfg.get("adapter") and not adapter_paths:
        adapter_paths = [Path(cfg["adapter"])]
    threads = int(cfg.get("threads") or 0) or max(1, (torch.get_num_threads() or 4))
    torch.set_num_threads(threads)

    t0 = time.time()
    try:
        model, tokenizer, kind = load_causal_lm(
            model_dir, torch.float32,
            note=lambda m: emit(type="log", level="info", message=m))
        loaded_names: list[str] = []
        if adapter_paths:
            from peft import PeftModel
            for i, ap_ in enumerate(adapter_paths):
                if not ap_.is_dir():
                    emit(type="error", message=f"适配器目录不存在：{ap_}")
                    return 3
                name = f"fts{i}"
                if i == 0:
                    model = PeftModel.from_pretrained(model, str(ap_), adapter_name=name)
                else:
                    model.load_adapter(str(ap_), adapter_name=name)
                loaded_names.append(name)
            if len(loaded_names) > 1:
                # 新版 PEFT 支持列表；旧版列表会抛 unhashable list，退化为
                # cat 合并成一个临时适配器，仍然让所有 LoRA 同时参与推理。
                try:
                    model.set_adapter(loaded_names)
                except Exception:
                    combined = "fts_multi"
                    model.add_weighted_adapter(
                        loaded_names, [1.0] * len(loaded_names), combined,
                        combination_type="cat")
                    model.set_adapter(combined)
            else:
                model.set_adapter(loaded_names[0])
        loaded_adapter = "+".join(p.name for p in adapter_paths)
        model.eval()

        mm_adapter = None
        proj_paths = [p / "mm_projector.pt" for p in adapter_paths]
        extra_proj = Path(cfg.get("projector") or "")
        if str(extra_proj) and extra_proj.is_file():
            proj_paths.append(extra_proj)
        if any(p.is_file() for p in proj_paths):
            from mm_support import MMAdapter
            mm_adapter = MMAdapter.load_composite(model, tokenizer, proj_paths)
            if mm_adapter is not None:
                mods = getattr(mm_adapter, "loaded_modalities", [])
                emit(type="log", level="info",
                     message=f"已组合多模态投影层：{'、'.join(mods) or '内置'}")
    except Exception as exc:  # noqa: BLE001
        emit(type="error", message=f"模型加载失败：{type(exc).__name__}: {exc}")
        return 3

    n_params = sum(p.numel() for p in model.parameters())
    has_template = bool(getattr(tokenizer, "chat_template", None))
    emit(type="ready", model=model_dir.name, adapter=loaded_adapter,
         adapters=[p.name for p in adapter_paths],
         modalities=getattr(mm_adapter, "loaded_modalities", []) if mm_adapter else [],
         params=int(n_params), chat_template=has_template,
         load_seconds=round(time.time() - t0, 1), threads=threads)

    stop_flag = {"stop": False}
    encoders: dict = {}          # 视觉/音频编码器懒加载后缓存，多次对话不重复加载
    cmds: queue.Queue = queue.Queue()

    def reader() -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("cmd") == "stop":
                stop_flag["stop"] = True
            else:
                cmds.put(obj)
        cmds.put({"cmd": "quit"})   # 父进程关闭了 stdin，正常退出

    threading.Thread(target=reader, daemon=True).start()

    class StopGeneration(Exception):
        """用户点了停止：用异常打断生成循环。"""

    class TokenStreamer:
        """兼容两种 generate 的流式收集器。

        - transformers 标准模型：每次 put 一个 [B,1] 的 token 张量；
        - MiniMind 自定义 generate：同样 put token 张量，但会在开头先 put 一次 prompt。
        两家都会先 put prompt，所以第一个多 token 的 put 直接丢掉；解码采用
        「累积解码取增量」，避免字节级 BPE 把一个汉字切成两半显示成乱码。
        """

        def __init__(self, tok, flag: dict):
            self.tok = tok
            self.flag = flag
            self.q: queue.Queue = queue.Queue()
            self.ids: list[int] = []
            self.text = ""
            self._first = True
            self._done = False

        def put(self, value) -> None:
            if self._done:
                return
            ids = value.flatten().tolist() if hasattr(value, "flatten") else list(value)
            if self._first:
                self._first = False
                if len(ids) > 1:          # 这是 prompt，跳过
                    return
            if self.flag["stop"]:
                raise StopGeneration()
            self.ids.extend(int(i) for i in ids)
            text = self.tok.decode(self.ids, skip_special_tokens=True)
            if text.endswith("\ufffd"):
                # 字节级 BPE 把一个汉字切开了，等下一个 token 凑齐再发，避免显示 �
                return
            if text.startswith(self.text):
                delta = text[len(self.text):]
            else:
                common = 0
                for a, b in zip(self.text, text):
                    if a != b:
                        break
                    common += 1
                delta = text[common:]
            self.text = text
            if delta:
                self.q.put(delta)

        def end(self) -> None:
            if not self._done:
                self._done = True
                self.q.put(None)

    def build_prompt(messages: list[dict], media_kinds: list[str] | None = None) -> str:
        if media_kinds:
            from mm_support import expand_messages
            for kind in media_kinds:
                messages = expand_messages(messages, kind)
        if has_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        parts = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}.get(m["role"], "User")
            parts.append(f"{tag}: {m['content']}")
        parts.append("Assistant:")
        return "\n".join(parts) + "\n"

    def do_chat(msg: dict) -> None:
        messages = msg.get("messages") or []
        params = msg.get("params") or {}
        max_new = int(params.get("max_new_tokens", 128))
        temperature = max(0.05, float(params.get("temperature", 0.7)))
        top_p = max(0.01, min(1.0, float(params.get("top_p", 0.9))))
        rep = max(0.5, float(params.get("repetition_penalty", 1.1)))
        media = {key: str(msg[key]) for key in ("image", "audio") if msg.get(key)}
        if media and mm_adapter is None:
            emit(type="error", message="当前适配器没有多模态投影层（需要先用图像/语音数据训练一次）")
            return
        for key, path in media.items():
            if not Path(path).is_file():
                emit(type="error", message=f"{key} 文件不存在：{path}")
                return

        text = build_prompt(messages, list(media))
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        stop_flag["stop"] = False

        if media:
            assert mm_adapter is not None
            try:
                import mm_encoder as mm
                import numpy as np
                import torch as _t
                feats: dict = {}
                for kind, path in media.items():
                    if kind == "image":
                        if "vision" not in encoders:
                            encoders["vision"] = mm.load_vision_encoder()
                        feat = mm.encode_image(encoders["vision"][0], encoders["vision"][1], path)
                    else:
                        if "audio" not in encoders:
                            encoders["audio"] = mm.load_audio_encoder(
                                Path(cfg.get("audio_encoder") or
                                     (model_dir.parent / "_encoders" / "whisper-tiny")))
                        feat = mm.encode_audio(encoders["audio"], path)
                    feats[kind] = _t.from_numpy(np.asarray(feat, dtype=np.float32))
                mm_adapter.set_features([feats])
                emit(type="media", kinds=list(feats),
                     tokens={k: int(v.shape[0]) for k, v in feats.items()},
                     paths={k: Path(v).name for k, v in media.items()})
            except Exception as exc:  # noqa: BLE001
                emit(type="error", message=f"媒体编码失败：{type(exc).__name__}: {exc}")
                return
        else:
            mm_adapter and mm_adapter.set_features([None])

        emit(type="start", prompt_tokens=int(enc["input_ids"].shape[1]), max_new_tokens=max_new)

        st = TokenStreamer(tokenizer, stop_flag)
        kwargs = dict(
            inputs=enc["input_ids"], attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=top_p,
            repetition_penalty=rep, eos_token_id=tokenizer.eos_token_id, streamer=st,
        )
        err: list[str] = []

        def run() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**kwargs)
            except StopGeneration:
                pass
            except Exception as exc:  # noqa: BLE001
                err.append(f"{type(exc).__name__}: {exc}")
            finally:
                st.end()

        t_start = time.time()
        first_ms = None
        chunks: list[str] = []
        threading.Thread(target=run, daemon=True).start()
        while True:
            item = st.q.get()
            if item is None:
                break
            if first_ms is None:
                first_ms = round((time.time() - t_start) * 1000)
            chunks.append(item)
            emit(type="token", text=item)
        full = "".join(chunks)
        seconds = max(1e-3, time.time() - t_start)
        n_out = len(st.ids)
        if err:
            emit(type="error", message=err[0])
            return
        emit(type="done", text=full, stopped=stop_flag["stop"], output_tokens=n_out,
             seconds=round(seconds, 2), first_token_ms=first_ms,
             tps=round(n_out / seconds, 1))

    while True:
        try:
            msg = cmds.get(timeout=0.5)
        except queue.Empty:
            continue
        cmd = msg.get("cmd")
        if cmd == "quit":
            break
        if cmd == "chat":
            try:
                do_chat(msg)
            except Exception as exc:  # noqa: BLE001
                emit(type="error", message=f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
