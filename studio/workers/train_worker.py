# -*- coding: utf-8 -*-
"""训练 worker：LoRA / 全参 SFT，向 stdout 输出 @@EVT@@ 事件。

用法（由界面进程调用，不面向终端用户）：
    python train_worker.py --config run_cfg.json

run_cfg.json 字段见下方 CFG_KEYS。事件类型：
    stage / data / model / plan / train_begin / step / eval / stopping
    saved / done / error / log
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

EVT = "@@EVT@@"
IGNORE_INDEX = -100

ROLE_OK = {"system", "user", "assistant"}


def _check_dialogue(msgs: list[dict]) -> None:
    if not msgs:
        raise ValueError("消息列表为空")
    roles = [m["role"] for m in msgs]
    for role in roles:
        if role not in ROLE_OK:
            raise ValueError(f"未知角色 {role!r}")
    body = roles
    if roles[0] == "system":
        if roles.count("system") > 1:
            raise ValueError("system 消息只能有一条且在开头")
        body = roles[1:]
    elif "system" in roles:
        raise ValueError("system 消息只能在开头")
    if not body or body[0] != "user":
        raise ValueError("对话必须以 user 开头")
    if body[-1] != "assistant":
        raise ValueError("最后一条必须是 assistant")
    for i, role in enumerate(body):
        if role != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError("user/assistant 顺序不合法")


def now() -> float:
    return time.time()


def make_emit(events_file: Path | None):
    fh = None
    if events_file is not None:
        try:
            events_file.parent.mkdir(parents=True, exist_ok=True)
            fh = events_file.open("a", encoding="utf-8", newline="\n")
        except OSError:
            fh = None

    def emit(**kw):
        kw.setdefault("t", round(now(), 3))
        line = json.dumps(kw, ensure_ascii=False)
        sys.stdout.write(EVT + line + "\n")
        sys.stdout.flush()
        if fh is not None:
            try:
                fh.write(line + "\n")
                fh.flush()
            except OSError:
                pass

    return emit, fh


def log(msg: str) -> None:
    sys.stdout.write(str(msg).rstrip() + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- 数据
def normalize(obj: dict) -> list[dict]:
    role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                "system": "system", "bot": "assistant", "ai": "assistant"}
    if isinstance(obj.get("messages"), list) and obj["messages"]:
        out = []
        for m in obj["messages"]:
            if not isinstance(m, dict):
                raise ValueError("messages 内有非对象元素")
            raw = str(m.get("role", "")).strip().lower()
            if raw not in role_map:
                raise ValueError(f"未知角色 {raw!r}")
            content = m.get("content")
            if content is None:
                content = m.get("value")
            if not isinstance(content, str):
                raise ValueError("content 不是字符串")
            out.append({"role": role_map[raw], "content": content})
        _check_dialogue(out)
        return out
    if isinstance(obj.get("conversations"), list) and obj["conversations"]:
        out = []
        for m in obj["conversations"]:
            if not isinstance(m, dict):
                raise ValueError("conversations 内有非对象元素")
            raw = str(m.get("from") or m.get("role") or "").strip().lower()
            if raw not in role_map:
                raise ValueError(f"未知角色 {raw!r}")
            content = m.get("value")
            if content is None:
                content = m.get("content")
            if not isinstance(content, str):
                raise ValueError("value 不是字符串")
            out.append({"role": role_map[raw], "content": content})
        _check_dialogue(out)
        return out
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
    raise ValueError(f"无法识别的数据格式，字段: {list(obj.keys())}")


def render_fallback(msgs: list[dict]) -> tuple[str, str]:
    """没有 chat_template 时的兜底拼装：返回 (全文, 提示前缀)。"""
    parts = []
    for m in msgs:
        r = m["role"]
        if r == "system":
            parts.append(f"System: {m['content']}")
        elif r == "user":
            parts.append(f"User: {m['content']}")
        else:
            parts.append(f"Assistant: {m['content']}")
    full = "\n".join(parts) + "\n"
    prefix = []
    for m in msgs[:-1]:
        r = m["role"]
        tag = {"system": "System", "user": "User", "assistant": "Assistant"}.get(r, "User")
        prefix.append(f"{tag}: {m['content']}")
    pre = "\n".join(prefix) + ("\n" if prefix else "") + "Assistant: "
    return full, pre


def build_tokenizer_fn(tokenizer, max_len: int, emit):
    has_template = bool(getattr(tokenizer, "chat_template", None))
    if not has_template:
        emit(type="log", level="warn",
             message="该模型没有 chat_template，已退回 User:/Assistant: 纯文本拼装")

    def tokenize(msgs: list[dict]) -> tuple[list[int], list[int]] | None:
        if has_template:
            full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        else:
            full_text, prompt_text = render_fallback(msgs)
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        labels = list(full_ids)
        n = min(len(prompt_ids), len(labels))
        if full_ids[:n] == prompt_ids[:n]:
            labels[:n] = [IGNORE_INDEX] * n
        else:
            common = 0
            for a, b in zip(full_ids, prompt_ids):
                if a != b:
                    break
                common += 1
            labels[:common] = [IGNORE_INDEX] * common
        full_labels = labels
        input_ids = full_ids[:max_len]
        labels = full_labels[:max_len]
        if not any(label != IGNORE_INDEX for label in labels):
            # 右截断把回答整段切掉了：改从左侧截断，保住回答部分
            input_ids = full_ids[-max_len:]
            labels = full_labels[-max_len:]
            if not any(label != IGNORE_INDEX for label in labels):
                return None
        return input_ids, labels

    return tokenize


class TokenDataset:
    """最小可用的 torch Dataset：只存 input_ids / labels。"""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        return {"input_ids": list(r["input_ids"]), "labels": list(r["labels"]),
                "attention_mask": [1] * len(r["input_ids"]), "feat": r.get("feat"),
                "media_kind": ((r.get("mm") or {}).get("kind") or "")}


def make_collator(torch, pad_id: int, adapter=None):
    class PadCollator:
        def __init__(self):
            self.tokens_seen = 0

        def __call__(self, features):
            max_len = max(len(f["input_ids"]) for f in features)
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for f in features:
                pad = max_len - len(f["input_ids"])
                batch["input_ids"].append(list(f["input_ids"]) + [pad_id] * pad)
                batch["attention_mask"].append(list(f["attention_mask"]) + [0] * pad)
                batch["labels"].append(list(f["labels"]) + [IGNORE_INDEX] * pad)
                self.tokens_seen += len(f["input_ids"])
            if adapter is not None:
                rows = [{"input_ids_padded": ids, "feat": f.get("feat"),
                         "kind": f.get("media_kind") or ""}
                        for ids, f in zip(batch["input_ids"], features)]
                adapter.collate_feats(rows)
            return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

    return PadCollator()


# ---------------------------------------------------------------- 主流程
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
    out_dir = Path(cfg["output"])
    root_outputs = (Path(os.environ.get("FTS_ROOT", str(Path.cwd()))).resolve() / "outputs")
    try:
        out_dir.resolve().relative_to(root_outputs)
    except ValueError:
        msg = f"输出目录越界：{out_dir} 不在 {root_outputs} 内"
        sys.stdout.write(EVT + json.dumps({"type": "error", "message": msg},
                                          ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)
    events_file = Path(cfg["events_file"]) if cfg.get("events_file") else out_dir / "studio_events.jsonl"
    emit, fh = make_emit(events_file)
    stop_flag = Path(cfg["stop_flag"]) if cfg.get("stop_flag") else None

    t_start = now()
    try:
        return _run(cfg, out_dir, emit, stop_flag, t_start)
    except KeyboardInterrupt:
        emit(type="error", message="训练被中断（KeyboardInterrupt）")
        return 130
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        emit(type="error", message=f"{type(exc).__name__}: {exc}", traceback=tb[-4000:])
        return 2
    finally:
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass


def _run(cfg: dict, out_dir: Path, emit, stop_flag: Path | None, t_start: float) -> int:
    import torch
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from mm_support import expand_messages, media_key, PAD_TOKEN, expected_tokens

    model_path = Path(cfg["model"])
    data_path = Path(cfg["data"])
    if not model_path.is_dir():
        emit(type="error", message=f"模型目录不存在：{model_path}")
        return 3
    if not data_path.is_file():
        emit(type="error", message=f"数据文件不存在：{data_path}")
        return 3

    threads = int(cfg.get("threads") or 0) or max(1, (os.cpu_count() or 4) // 2)
    torch.set_num_threads(threads)
    set_seed(int(cfg.get("seed", 42)))
    emit(type="stage", name="init", message=f"torch {torch.__version__} · 线程 {threads}")

    # ------------------------------------------------ 分词
    emit(type="stage", name="tokenize", message="加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_len = int(cfg.get("max_len", 256))

    raw_lines: list[str] = []
    with data_path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_lines.append(line)
    if not raw_lines:
        emit(type="error", message="数据文件是空的")
        return 4

    fn = build_tokenizer_fn(tokenizer, max_len, emit)
    rows: list[dict] = []
    bad = 0
    skipped = 0
    total_tokens = 0
    step = max(1, len(raw_lines) // 100)
    t_tok = now()
    for i, line in enumerate(raw_lines, 1):
        if i % step == 0 or i == len(raw_lines):
            emit(type="stage", name="tokenize", done=i, total=len(raw_lines),
                 message=f"分词 {i}/{len(raw_lines)}")
        try:
            obj = json.loads(line)
            msgs = normalize(obj)
        except (json.JSONDecodeError, ValueError):
            bad += 1
            continue
        if not any(m["role"] == "assistant" and m["content"].strip() for m in msgs):
            skipped += 1
            continue
        media = {}
        for key in ("image", "audio"):
            if obj.get(key):
                media = {"kind": key, "path": str(obj[key])}
                break
        if media:
            # 占位符数量必须等于编码器特征行数（训练与推理共用同一展开函数）
            msgs = expand_messages(msgs, media["kind"])
        res = fn(msgs)
        if res is None:
            skipped += 1
            continue
        input_ids, labels = res
        rows.append({"input_ids": input_ids, "labels": labels,
                     "mm": ({"kind": media["kind"], "path": media["path"],
                             "key": media_key({"image": media["path"]} if media["kind"] == "image"
                                              else {"audio": media["path"]})}
                            if media else None)})
        total_tokens += sum(1 for token_label in labels if token_label != IGNORE_INDEX)

    if not rows:
        emit(type="error", message=f"没有可用样本（解析失败 {bad} 条，assistant 为空或超长 {skipped} 条）")
        return 4

    lens = [len(r["input_ids"]) for r in rows]
    avg_len = sum(lens) / len(lens)
    tok_seconds = now() - t_tok
    emit(type="data", samples=len(rows), bad=bad, skipped=skipped,
         avg_len=round(avg_len, 1), max_len_seen=max(lens), min_len_seen=min(lens),
         tokens=int(total_tokens), seconds=round(tok_seconds, 1))
    log(f"[数据] {len(rows)} 条可用（解析失败 {bad}，跳过 {skipped}），平均 {avg_len:.0f} token")

    # ------------------------------------------------ 多模态特征
    mm_stats = {}
    media_rows = [r for r in rows if r.get("mm")]
    if media_rows:
        from mm_support import attach_features
        emit(type="stage", name="mm_feat", message=f"检测到 {len(media_rows)} 条多模态样本，开始提取特征")
        mm_stats = attach_features(rows, Path(cfg["data"]).parent / ".mm_cache",
                                   Path(cfg["data"]).parents[1], emit,
                                   audio_dir=Path(cfg["data"]).parents[1] / "models" / "_encoders" / "whisper-tiny")
        ok = sum(mm_stats.get(k, 0) for k in ("image", "audio"))
        emit(type="mm", rows=len(media_rows), ok=ok, cached=mm_stats.get("cached", 0),
             failed=mm_stats.get("failed", 0),
             image=mm_stats.get("image", 0), audio=mm_stats.get("audio", 0))
        if ok == 0:
            emit(type="error", message="多模态样本都提取不到特征：检查图片/音频路径是否存在")
            return 4
        log(f"[多模态] 可用 {ok} 条（图像 {mm_stats.get('image', 0)} / 语音 {mm_stats.get('audio', 0)}，"
            f"命中缓存 {mm_stats.get('cached', 0)}）")

    # ------------------------------------------------ 划分
    val_ratio = float(cfg.get("val_ratio", 0.1))
    rows_shuffled = list(rows)
    import random
    random.Random(int(cfg.get("seed", 42))).shuffle(rows_shuffled)
    if len(rows_shuffled) >= 10 and val_ratio > 0:
        n_val = max(1, int(round(len(rows_shuffled) * min(0.5, val_ratio))))
        val_rows = rows_shuffled[:n_val]
        train_rows = rows_shuffled[n_val:]
    else:
        val_rows, train_rows = [], rows_shuffled
    emit(type="split", train=len(train_rows), val=len(val_rows))
    log(f"[数据] 训练 {len(train_rows)} / 验证 {len(val_rows)}")

    # ------------------------------------------------ 模型
    emit(type="stage", name="model", message=f"加载模型 {model_path.name}（fp32 / CPU）")
    t_model = now()
    from custom_model import load_causal_lm

    model, _tok, kind = load_causal_lm(
        model_path, torch.float32,
        note=lambda m: emit(type="log", level="info", message=m))
    emit(type="log", level="info", message=f"模型加载方式：{kind}")
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_pos, int) and 0 < max_pos < max_len:
        emit(type="log", level="warn",
             message=f"max_len 超过模型上限 {max_pos}，已自动降到 {max_pos}")
        max_len = max_pos
        rows = [{**r, "input_ids": r["input_ids"][:max_len], "labels": r["labels"][:max_len]}
                for r in rows]
        train_rows = [{**r, "input_ids": r["input_ids"][:max_len], "labels": r["labels"][:max_len]}
                      for r in train_rows]
        val_rows = [{**r, "input_ids": r["input_ids"][:max_len], "labels": r["labels"][:max_len]}
                    for r in val_rows]

        # 截断可能切掉多模态占位符：数量对不上的样本直接剔除，避免训练时静默错位
        def _mm_ok(r: dict) -> bool:
            mm = r.get("mm")
            if not mm:
                return True
            pid = tokenizer.convert_tokens_to_ids(PAD_TOKEN[mm["kind"]])
            return sum(1 for x in r["input_ids"] if x == pid) == expected_tokens(mm["kind"])

        dropped = [r for r in rows if not _mm_ok(r)]
        if dropped:
            emit(type="log", level="warn",
                 message=f"{len(dropped)} 条多模态样本在截断后占位符数量不符，已剔除")
            rows = [r for r in rows if _mm_ok(r)]
            train_rows = [r for r in train_rows if _mm_ok(r)]
            val_rows = [r for r in val_rows if _mm_ok(r)]

    total_params = sum(p.numel() for p in model.parameters())
    mode = "full" if cfg.get("full") else "lora"
    resume_from = cfg.get("resume") or None

    if mode == "lora":
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        lora_r = int(cfg.get("lora_r", 16))
        if resume_from and (Path(resume_from) / "adapter_config.json").is_file():
            model = PeftModel.from_pretrained(model, str(resume_from), is_trainable=True)
            emit(type="log", level="info", message=f"已从适配器续训：{resume_from}")
        else:
            target = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            present = {n.split(".")[-1] for n, _ in model.named_modules()}
            target = [t for t in target if t in present] or None
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=lora_r,
                lora_alpha=int(cfg.get("lora_alpha", lora_r * 2)),
                lora_dropout=float(cfg.get("lora_dropout", 0.05)),
                target_modules=target, bias="none"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emit(type="model", params=int(total_params), trainable=int(trainable), mode=mode,
         name=model_path.name, load_seconds=round(now() - t_model, 1))
    log(f"[模型] {total_params/1e6:.1f}M 参数，可训练 {trainable/1e6:.2f}M（{trainable/total_params*100:.2f}%）")

    # ------------------------------------------------ 训练参数
    batch = int(cfg.get("batch_size", 2))
    accum = int(cfg.get("grad_accum", 2))
    epochs = float(cfg.get("epochs", 3.0))
    steps_per_epoch = max(1, math.ceil(len(train_rows) / max(1, batch * accum)))
    max_steps = int(cfg.get("max_steps", -1) or -1)
    total_steps = max_steps if max_steps > 0 else max(1, int(round(steps_per_epoch * epochs)))
    save_steps = max(1, min(steps_per_epoch, total_steps))
    emit(type="plan", total_steps=total_steps, steps_per_epoch=steps_per_epoch,
         save_steps=save_steps, batch=batch, accum=accum, epochs=epochs, max_len=max_len)
    log(f"[计划] {total_steps} 步（每 epoch {steps_per_epoch} 步），每 {save_steps} 步存一次检查点")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=max(1, batch),
        gradient_accumulation_steps=accum,
        learning_rate=float(cfg.get("lr", 2e-4)),
        num_train_epochs=epochs,
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps" if val_rows else "no",
        eval_steps=save_steps if val_rows else None,
        report_to=[],
        use_cpu=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=int(cfg.get("seed", 42)),
        disable_tqdm=True,
        logging_first_step=True,
        save_safetensors=True,
    )

    adapter = None
    if media_rows:
        from mm_support import MMAdapter, projector_params
        adapter = MMAdapter(model, tokenizer)
        n_proj = sum(p.numel() for p in projector_params(adapter))
        emit(type="mm_ready", projector_params=int(n_proj))
        log(f"[多模态] 投影层 {n_proj/1e6:.2f}M 参数（可训练），编码器已冻结、特征已缓存")

    collator = make_collator(torch, tokenizer.pad_token_id, adapter)
    state = {"stop_requested": False, "t0": now(), "tokens": 0, "last_step_t": now(),
             "losses": [], "loss": None}

    # transformers 在 gradient_accumulation_steps>1 时，log_history["loss"] 记的是
    # 多个 micro-batch 的累加值（实测正好等于平均值×accum），直接显示会让人误判
    # 「训崩了」。这里用 forward hook 取模型自己算的真实 loss，与 accum 无关。
    def _loss_hook(_module, _inputs, output):
        v = getattr(output, "loss", None)
        if v is not None:
            try:
                state["loss"] = float(v.detach())
            except (TypeError, ValueError):
                pass

    try:
        model.register_forward_hook(_loss_hook)
    except Exception:  # noqa: BLE001
        pass

    class Cb(TrainerCallback):
        def on_train_begin(self, a, st, control, **kw):
            state["t0"] = now()
            state["last_step_t"] = now()
            emit(type="train_begin", total_steps=st.max_steps)

        def _emit_step(self, st, logs: dict) -> None:
            loss = state["loss"] if state["loss"] is not None else logs.get("loss")
            lr = logs.get("learning_rate")
            done = int(st.global_step)
            tokens = collator.tokens_seen
            state["tokens"] = tokens
            elapsed = max(1e-6, now() - state["t0"])
            tps = tokens / elapsed if tokens else 0.0
            eta = (int(st.max_steps) - done) * (elapsed / max(1, done)) if done else 0.0
            if isinstance(loss, (int, float)):
                state["losses"].append((done, round(float(loss), 6)))
            emit(type="step", step=done, total=int(st.max_steps),
                 loss=round(float(loss), 5) if isinstance(loss, (int, float)) else None,
                 lr=float(lr) if isinstance(lr, (int, float)) else None,
                 epoch=round(float(st.epoch or 0), 3),
                 elapsed=round(elapsed, 1), eta=round(eta, 1),
                 tokens=int(tokens), tps=round(tps, 1))

        def on_log(self, a, st, control, logs=None, **kw):
            # Trainer 在 on_step_end 之前调用 on_log，且此处的 logs 才带着当前步的 loss
            if logs and isinstance(logs.get("loss"), (int, float)):
                self._emit_step(st, logs)

        def on_step_end(self, a, st, control, **kw):
            done = int(st.global_step)
            if stop_flag is not None and stop_flag.exists() and not state["stop_requested"]:
                state["stop_requested"] = True
                control.should_training_stop = True
                control.should_save = True
                emit(type="stopping", step=done,
                     message="收到停止请求，正在当前步结束后保存检查点")

        def on_evaluate(self, a, st, control, metrics=None, **kw):
            m = metrics or {}
            ev = m.get("eval_loss")
            emit(type="eval", step=int(st.global_step),
                 eval_loss=round(float(ev), 5) if isinstance(ev, (int, float)) else None,
                 perplexity=round(math.exp(float(ev)), 3) if isinstance(ev, (int, float)) and ev < 20 else None)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=TokenDataset(train_rows),
        eval_dataset=TokenDataset(val_rows) if val_rows else None,
        data_collator=collator,
        callbacks=[Cb()],
        processing_class=tokenizer,
    )

    emit(type="stage", name="train", message="开始训练")
    trainer.train(resume_from_checkpoint=resume_from if (resume_from and Path(resume_from).is_dir()
                                                         and any(Path(resume_from).glob("checkpoint-*"))) else None)
    train_seconds = now() - state["t0"]

    # ------------------------------------------------ 保存
    emit(type="stage", name="save", message="保存适配器 / 权重")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    hist = trainer.state.log_history
    losses = list(state["losses"])
    if not losses:      # 兜底：hook 没生效时退回 Trainer 日志
        losses = [(int(h.get("step", 0)), float(h["loss"])) for h in hist
                  if isinstance(h.get("loss"), (int, float))]
    evals = [(int(h.get("step", 0)), float(h["eval_loss"])) for h in hist if isinstance(h.get("eval_loss"), (int, float))]
    final_loss = losses[-1][1] if losses else None
    best_eval = min((x[1] for x in evals), default=None)
    first_loss = losses[0][1] if losses else None
    tokens = collator.tokens_seen
    tps = tokens / train_seconds if train_seconds > 0 else 0.0
    steps_done = int(trainer.state.global_step)

    metrics = {
        "final_loss": final_loss,
        "first_loss": first_loss,
        "best_eval_loss": best_eval,
        "perplexity": round(math.exp(best_eval), 3) if best_eval and best_eval < 20 else None,
        "steps_done": steps_done,
        "train_seconds": round(train_seconds, 1),
        "tokens": int(tokens),
        "tokens_per_sec": round(tps, 1),
        "loss_curve": losses[-2000:],
        "eval_curve": evals[-200:],
        "stop_requested": state["stop_requested"],
    }
    meta = {
        "base_model": str(model_path),
        "mode": mode,
        "lora_r": None if mode == "full" else int(cfg.get("lora_r", 16)),
        "lora_alpha": None if mode == "full" else int(cfg.get("lora_alpha", 32)),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "epochs": epochs,
        "batch_size": batch,
        "grad_accum": accum,
        "lr": float(cfg.get("lr", 2e-4)),
        "max_len": max_len,
        "avg_len": round(avg_len, 1),
        "params": int(total_params),
        "trainable_params": int(trainable),
        "seconds": round(train_seconds, 1),
        "total_seconds": round(now() - t_start, 1),
        "data": str(data_path),
        "goal": cfg.get("goal", ""),
        "final_metrics": metrics,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "loss_curve.json").write_text(
        json.dumps({"loss": losses, "eval": evals}, ensure_ascii=False), encoding="utf-8")

    if adapter is not None:
        proj_path = adapter.save(out_dir, {"rows": len(media_rows), "stats": mm_stats,
                                           "vision_encoder": "MobileNetV3-Small",
                                           "audio_encoder": "Whisper-tiny encoder"})
        meta["multimodal"] = {"projector": proj_path.name, "rows": len(media_rows),
                              "stats": mm_stats}
        (out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
        emit(type="log", level="ok", message=f"多模态投影层已保存：{proj_path.name}")

    emit(type="saved", output=str(out_dir), meta=meta)
    emit(type="done", status="stopped" if state["stop_requested"] else "done",
         metrics=metrics, output=str(out_dir), total_seconds=round(now() - t_start, 1),
         meta_path=str(out_dir / "train_meta.json"))
    log(f"[完成] 用时 {train_seconds/60:.1f} 分钟，产物：{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
