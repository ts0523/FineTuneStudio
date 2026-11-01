# -*- coding: utf-8 -*-
"""对话页：加载基座 / 适配器 / 合并模型，流式对话。"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ...core import adapters
from ...core import ai, audio, voice
from ...core import models as models_mod
from ...core import paths
from .. import i18n
from ..base import Page
from ..theme import P, font
from ..widgets import Badge, Card, Check, Entry, RBtn, ScrollFrame, TextArea

CHAT_KEY = "chat"

CLOUD = "云端全模态（免费）"
LOCAL = "本地模型"


class ChatPage(Page):
    key = "chat"
    title = "对话"
    subtitle = "加载任意模型或适配器，检验它到底学会了什么"

    def build(self) -> None:
        f = tk.Frame(self, bg=P["bg"])
        f.pack(fill="both", expand=True)
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)

        top = Card(f, "会话设置", "")
        top.grid(row=0, column=0, sticky="ew")
        tb = top.body

        row1 = tk.Frame(tb, bg=P["panel"])
        row1.pack(fill="x")
        self.var_model = tk.StringVar()
        tk.Label(row1, text="模型", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        self.cb_model = ttk.Combobox(row1, textvariable=self.var_model, state="readonly",
                                     font=font(9), width=27)
        self.cb_model.pack(side="left", padx=(6, 12), ipady=2)
        tk.Label(row1, text="适配器（可多选）", bg=P["panel"], fg=P["muted"],
                 font=font(9)).pack(side="left")
        self.adapter_list = tk.Listbox(
            row1, selectmode="extended", exportselection=False, height=3, width=30,
            bg=P["panel2"], fg=P["text"], selectbackground=P["accent_dim"],
            selectforeground=P["text"], highlightthickness=1,
            highlightbackground=P["border"], bd=0, font=font(9))
        self.adapter_list.pack(side="left", padx=(6, 12))
        self.adapter_list.bind("<<ListboxSelect>>", lambda e: self._on_pick())
        self.cb_model.bind("<<ComboboxSelected>>", lambda e: self._on_pick())
        self.state_badge = Badge(row1, "未加载", "panel3", bg=P["panel"])
        self.state_badge.pack(side="left")
        self.btn_unload = RBtn(row1, "卸载", self._unload, kind="ghost", height=28, size=9,
                               padx=10)
        self.btn_unload.pack(side="right")

        row2 = tk.Frame(tb, bg=P["panel"])
        row2.pack(fill="x", pady=(10, 0))
        tk.Label(row2, text="系统提示", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        self.sys_entry = Entry(row2, width=44, placeholder="可留空；例如：你是 Nova 客服，回答要简短")
        self.sys_entry.pack(side="left", fill="x", expand=True, padx=(6, 14), ipady=3)

        row3 = tk.Frame(tb, bg=P["panel"])
        row3.pack(fill="x", pady=(8, 0))
        tk.Label(row3, text="引擎", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        # 实验层默认本地模型：直接在应用里选适配器就能全模态输入输出。
        self._engine_labels = {i18n.tr(CLOUD): CLOUD, i18n.tr(LOCAL): LOCAL}
        self.var_engine = tk.StringVar(value=i18n.tr(LOCAL))
        self.cb_engine = ttk.Combobox(row3, textvariable=self.var_engine, state="readonly",
                                      values=list(self._engine_labels), font=font(9), width=18)
        self.cb_engine.pack(side="left", padx=(6, 14), ipady=2)
        self.cb_engine.bind("<<ComboboxSelected>>", lambda e: self._engine_changed())
        self.var_cap_text = tk.BooleanVar(value=True)
        self.var_cap_image = tk.BooleanVar(value=True)
        self.var_cap_audio = tk.BooleanVar(value=True)
        self.var_speak = tk.BooleanVar(value=bool(self.cfg.get("chat.speak", True)))
        tk.Label(row3, text="内置适配器", bg=P["panel"], fg=P["muted"],
                 font=font(9)).pack(side="left")
        tk.Checkbutton(row3, text=adapters.adapter_label("text"), variable=self.var_cap_text,
                       state="disabled", bg=P["panel"], fg=P["text2"], selectcolor=P["panel2"],
                       activebackground=P["panel"], font=font(8)).pack(side="left", padx=(6, 0))
        for key, var in (("image", self.var_cap_image),
                         ("audio", self.var_cap_audio),
                         ("speech", self.var_speak)):
            tk.Checkbutton(row3, text=adapters.adapter_label(key), variable=var,
                           bg=P["panel"], fg=P["text2"], selectcolor=P["panel2"],
                           activebackground=P["panel"], font=font(8)).pack(side="left", padx=(6, 0))
        tk.Label(row3, text="本地勾选适配器即可全模态：看图、听声、朗读；云端不需要本地环境",
                 bg=P["panel"], fg=P["muted"], font=font(8)).pack(side="left", padx=12)

        self.var_temp = tk.DoubleVar(value=float(self.cfg.get("chat.temperature", 0.7)))
        self.var_top_p = tk.DoubleVar(value=float(self.cfg.get("chat.top_p", 0.9)))
        self.var_maxnew = tk.StringVar(value=str(self.cfg.get("chat.max_new_tokens", 128)))
        self.var_rep = tk.DoubleVar(value=float(self.cfg.get("chat.repetition_penalty", 1.1)))

        for label, var, lo, hi, step, width in (
                ("温度", self.var_temp, 0.0, 1.5, 0.05, 110),
                ("top_p", self.var_top_p, 0.1, 1.0, 0.05, 110)):
            tk.Label(row2, text=label, bg=P["panel"], fg=P["muted"], font=font(9)).pack(
                side="left", padx=(0, 4))
            tk.Scale(row2, variable=var, from_=lo, to=hi, resolution=step, orient="horizontal",
                     showvalue=True, length=width, bg=P["panel"], fg=P["text2"],
                     troughcolor=P["panel3"], highlightthickness=0, bd=0,
                     font=font(8), sliderlength=14).pack(side="left", padx=(0, 10))
        tk.Label(row2, text="最多 token", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        ttk.Spinbox(row2, from_=8, to=1024, increment=8, textvariable=self.var_maxnew, width=6,
                    font=font(9)).pack(side="left", padx=(4, 0), ipady=2)

        convo = Card(f, "对话", "Enter 发送，Shift+Enter 换行")
        convo.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        cb = convo.body
        cb.grid_rowconfigure(0, weight=1)
        cb.grid_columnconfigure(0, weight=1)

        self.scroll = ScrollFrame(cb, bg=P["panel"])
        self.scroll.grid(row=0, column=0, sticky="nsew")
        self.thread = self.scroll.inner
        self._hint = tk.Label(self.thread, text="选好模型后发第一条消息；首次会先加载模型（约 1~3 秒）。",
                              bg=P["panel"], fg=P["muted"], font=font(9), anchor="w",
                              justify="left", wraplength=680)
        self._hint.pack(anchor="w", pady=(6, 10), padx=8)

        media_row = tk.Frame(cb, bg=P["panel"])
        media_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        tk.Label(media_row, text="多模态输入", bg=P["panel"], fg=P["muted"],
                 font=font(9)).pack(side="left")
        RBtn(media_row, "选图片", self._pick_image, kind="ghost", height=26, size=8,
             padx=10).pack(side="left", padx=(8, 0))
        RBtn(media_row, "选语音", self._pick_audio, kind="ghost", height=26, size=8,
             padx=10).pack(side="left", padx=6)
        RBtn(media_row, "录音理解（4 秒）", self._record_audio, kind="primary",
             height=26, size=8, padx=10).pack(side="left", padx=(0, 6))
        RBtn(media_row, "录音转文字（4 秒）", self._voice_to_text, kind="secondary",
             height=26, size=8, padx=10).pack(side="left", padx=(0, 6))
        RBtn(media_row, "清除", self._clear_media, kind="ghost", height=26, size=8,
             padx=10).pack(side="left")
        self.media_label = tk.Label(media_row, text="未附加（纯文本对话）", bg=P["panel"],
                                    fg=P["muted"], font=font(8), anchor="w")
        self.media_label.pack(side="left", padx=10)
        self._media: dict = {}

        inp = tk.Frame(cb, bg=P["panel"])
        inp.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        inp.grid_columnconfigure(0, weight=1)
        self.input = TextArea(inp, height=3)
        self.input.grid(row=0, column=0, sticky="ew")
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Shift-Return>", lambda e: None)

        side = tk.Frame(inp, bg=P["panel"])
        side.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self.btn_send = RBtn(side, "发送", self.send, kind="primary", height=34, padx=18, width=96)
        self.btn_send.pack(fill="x")
        self.btn_stop = RBtn(side, "停止", self.stop_gen, kind="secondary", height=30,
                             size=9, padx=12, width=96)
        self.btn_stop.pack(fill="x", pady=(6, 0))

        bar = tk.Frame(cb, bg=P["panel"])
        bar.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.stats = tk.Label(bar, text="", bg=P["panel"], fg=P["muted"], font=font(8), anchor="w")
        self.stats.pack(side="left")
        RBtn(bar, "清空对话", self.clear, kind="ghost", height=26, size=8, padx=10).pack(side="right")
        RBtn(bar, "导出记录", self.export, kind="ghost", height=26, size=8, padx=10).pack(
            side="right", padx=6)

        self.messages: list[dict] = []
        self._loaded: tuple[str, tuple[str, ...]] | None = None
        self._pending: dict | None = None
        self._stream_label: tk.Label | None = None
        self._stream_buf = ""
        self._flush_job: str | None = None
        self._gen_t0 = 0.0
        self._n_tokens = 0

        self.app.bind_job(CHAT_KEY, self._on_event)

    # ------------------------------------------------------------ 列表
    def on_show(self) -> None:
        self._reload()
        engine = self.app.shared.pop("chat_engine", None)
        if engine in self._engine_labels.values():
            # 默认本地模型；只有「先用云端模式」这类入口才显式切到云端。
            self.var_engine.set(i18n.tr(engine))
            self._engine_changed()
        sel = self.app.shared.pop("chat_selection", None)
        if sel:
            if sel.get("model"):
                self.var_model.set(Path(sel["model"]).name)
            wanted = list(sel.get("adapters") or [])
            if sel.get("adapter"):
                wanted.append(Path(sel["adapter"]).name)
            self._select_adapters(wanted)
            self._on_pick()

    def _reload(self) -> None:
        models = [m for m in models_mod.scan_models()]
        names = [m.name for m in models]
        self.cb_model.configure(values=names)
        if names and self.var_model.get() not in names:
            self.var_model.set(models_mod.default_model_name(models, self.cfg.get("last.model", "")))
        outs = models_mod.scan_outputs()
        self._adapter_items = ([m.name for m in outs if m.kind == "adapter"]
                               + [m.name for m in outs if m.kind == "merged"])
        chosen = set(self._selected_adapters())
        self.adapter_list.delete(0, "end")
        for name in self._adapter_items:
            hit = next((m for m in outs if m.name == name), None)
            mods = adapters.adapter_modalities(hit.path) if hit and hit.kind == "adapter" else []
            labels = [i18n.tr(adapters.adapter_label(m)) for m in mods]
            suffix = f" · {' + '.join(labels)}" if labels else ""
            self.adapter_list.insert("end", name + suffix)
        for i, name in enumerate(self._adapter_items):
            if name in chosen:
                self.adapter_list.selection_set(i)

    def _adapter_names(self) -> list[str]:
        return list(getattr(self, "_adapter_items", []))

    def _selected_adapters(self) -> list[str]:
        names = self._adapter_names()
        return [names[i] for i in self.adapter_list.curselection() if i < len(names)]

    def _select_adapters(self, names: list[str]) -> None:
        self.adapter_list.selection_clear(0, "end")
        wanted = set(names)
        for i, name in enumerate(self._adapter_names()):
            if name in wanted:
                self.adapter_list.selection_set(i)

    def _on_pick(self) -> None:
        key = (self.var_model.get(), tuple(self._selected_adapters()))
        if self._loaded and key != self._loaded:
            self.state_badge.set("需要重新加载", "amber")

    # ------------------------------------------------------------ worker
    def _resolve(self) -> tuple[str, list[str]] | None:
        model_name = self.var_model.get()
        model = models_mod.find_model(model_name)
        if model is None:
            self.toast("先选一个模型", "warn")
            return None
        outs = models_mod.scan_outputs()
        selected = self._selected_adapters()
        merged = [name for name in selected
                  if (hit := next((o for o in outs if o.name == name), None)) and hit.kind == "merged"]
        if merged:
            if len(selected) > 1:
                self.toast("完整模型不能与其他适配器叠加；请只选一个完整模型", "warn", 4200)
                return None
            hit = next(o for o in outs if o.name == merged[0])
            return str(hit.path), []
        paths = []
        for name in selected:
            hit = next((o for o in outs if o.name == name), None)
            if hit is None or hit.kind != "adapter":
                self.toast(f"适配器不可用：{name}", "warn")
                return None
            paths.append(str(hit.path))
        return str(model.path), paths

    def _ensure_worker(self) -> bool:
        pair = self._resolve()
        if pair is None:
            return False
        model_path, adapter_paths = pair
        key_label = (self.var_model.get(), tuple(self._selected_adapters()))
        job = self.app.jobs.get(CHAT_KEY)
        if job and job.running and self._loaded == key_label:
            return True
        if job and job.running:
            job.kill()
        self._loaded = None
        projector = ""
        if not adapter_paths:
            merged = next((o for o in models_mod.scan_outputs()
                           if str(o.path) == model_path), None)
            if merged is not None and (merged.path / "mm_projector.pt").is_file():
                projector = str(merged.path / "mm_projector.pt")
        cfg = {"model": model_path, "adapters": adapter_paths, "projector": projector,
               "threads": int(self.cfg.get("threads", 0) or 0)}
        if not self.app.spawn_worker(CHAT_KEY, "chat_worker.py", cfg, label="对话",
                                     stdin_pipe=True):
            return False
        self.state_badge.set("加载中…", "amber")
        return True

    def _unload(self) -> None:
        job = self.app.jobs.get(CHAT_KEY)
        if job and job.running:
            if not job.send({"cmd": "quit"}):
                job.kill()
            self.after(800, lambda: job.running and job.kill())
        self._loaded = None
        self.state_badge.set("未加载", "panel3")
        self.stats.configure(text="模型已卸载，显存/内存已释放")

    # ------------------------------------------------------------ 发送
    def _media_text(self) -> str:
        bits = []
        if self._media.get("image"):
            bits.append(f"图片 {Path(self._media['image']).name}")
        if self._media.get("audio"):
            bits.append(f"语音 {Path(self._media['audio']).name}")
        return "已附加：" + " + ".join(bits) if bits else "未附加（纯文本对话）"

    def _update_media(self) -> None:
        self.media_label.configure(text=self._media_text(),
                                   fg=P["green"] if self._media else P["muted"])

    def _pick_image(self) -> None:
        from tkinter import filedialog

        if not self.var_cap_image.get():
            self.toast("先勾选内置适配器「图像适配器」", "warn")
            return
        p = filedialog.askopenfilename(
            title="选择图片", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if not p:
            return
        self._media["image"] = p
        self._update_media()
        if not self._cloud() and not any(
                "image" in adapters.adapter_modalities(str(o.path))
                for o in models_mod.scan_outputs()
                if o.name in self._selected_adapters()):
            self.toast("本地看图需要选中一个「图像适配器」（含图像投影层），或切换到云端全模态", "warn", 4600)

    def _pick_audio(self) -> None:
        from tkinter import filedialog

        if not self.var_cap_audio.get():
            self.toast("先勾选内置适配器「音频适配器」", "warn")
            return
        p = filedialog.askopenfilename(title="选择语音", filetypes=[("WAV 音频", "*.wav")])
        if not p:
            return
        self._media["audio"] = p
        self._update_media()

    def _record_audio(self) -> None:
        """直接录 4 秒作为音频适配器输入，不转文字。"""
        if not self.var_cap_audio.get():
            self.toast("先勾选内置适配器「音频适配器」", "warn")
            return
        ok, err = audio.mic_available()
        if not ok:
            self.toast(err or "没有可录音设备", "err", 4200)
            return
        self.media_label.configure(text="录音 4 秒…请说话", fg=P["accent"])

        def work() -> None:
            wav, error = audio.record_for(4.0)
            self.app.post(lambda: self._record_done(wav, error))

        threading.Thread(target=work, daemon=True).start()

    def _record_done(self, wav: Path | None, error: str) -> None:
        if wav is None:
            self.media_label.configure(text=f"录音失败：{error}", fg=P["amber"])
            return
        self._media["audio"] = str(wav)
        self._update_media()
        self.input.focus_set()

    def _clear_media(self) -> None:
        self._media = {}
        self._update_media()

    def _voice_to_text(self) -> None:
        """录音 4 秒 → 本地转写 → 填进输入框（不需要用户装任何东西）。"""
        ok, err = audio.mic_available()
        if not ok:
            self.toast(err or "没有可录音设备", "err", 4200)
            return
        self.media_label.configure(text="录音 4 秒…请说话", fg=P["accent"])

        def work() -> None:
            wav, error = audio.record_for(4.0)
            if wav is None:
                self.app.post(lambda: self._voice_done("", error or "没有录到声音"))
                return
            text, engine, err2 = audio.transcribe(wav)
            self.app.post(lambda: self._voice_done(text, err2, engine))

        threading.Thread(target=work, daemon=True).start()

    def _voice_done(self, text: str, error: str, engine: str = "") -> None:
        if not text:
            self.media_label.configure(text=f"没听清：{error or '再试一次'}", fg=P["amber"])
            self.toast(error or "没听清，靠近麦克风再说一次", "warn", 4200)
            return
        self.input.delete("1.0", "end")
        self.input.insert("1.0", text)
        self.media_label.configure(text=f"语音已转文字（{engine}），可直接发送", fg=P["green"])
        self.input.focus_set()

    def _on_enter(self, event) -> str | None:
        if event.state & 0x0001:  # Shift
            return None
        self.send()
        return "break"

    def _history(self) -> list[dict]:
        """只带最近 6 条对话，避免超出模型上下文导致越聊越乱。"""
        msgs = []
        system = self.sys_entry.value()
        if system:
            msgs.append({"role": "system", "content": system})
        recent = self.messages[-6:]
        if len(self.messages) > 6 and recent and recent[0]["role"] == "assistant":
            recent = recent[1:]
        msgs.extend(recent)
        return msgs

    def send(self) -> None:
        text = self.input.get("1.0", "end").strip()
        has_image = bool(self._media.get("image"))
        has_audio = bool(self._media.get("audio"))
        if not text:
            if not has_image and not has_audio:
                return
            if has_image and has_audio:
                text = "请同时理解这张图片和这段语音，然后回答。"
            elif has_image:
                text = "请描述并理解这张图片。"
            else:
                text = "请转写并理解这段语音。"
        if len(text) > 4000:
            self.toast("单条消息超过 4000 字符，已截断（这个量级的模型装不下更长上下文）",
                       "warn", 3600)
            text = text[:4000]
        cloud = self._cloud()
        if not cloud and not self._ensure_worker():
            return
        try:
            self._hint.pack_forget()
        except tk.TclError:
            pass
        self.input.delete("1.0", "end")
        self.messages.append({"role": "user", "content": text})
        self._add_bubble("user", text)
        self._stream_label = self._add_bubble("assistant", "")
        self._stream_buf = ""
        self.btn_send.set_enabled(False)
        self._gen_t0 = time.time()
        self._n_tokens = 0
        self.cfg.set("chat.temperature", float(self.var_temp.get()), save=False)
        self.cfg.set("chat.top_p", float(self.var_top_p.get()), save=False)
        self.cfg.set("chat.max_new_tokens", int(float(self.var_maxnew.get() or 128)), save=False)
        self.cfg.set("chat.repetition_penalty", float(self.var_rep.get()), save=False)
        self.cfg.set("chat.speak", bool(self.var_speak.get()), save=False)
        self.cfg.save()
        if cloud:
            self.state_badge.set("云端生成中…", "accent")
            self._send_cloud(text)
            return
        self.state_badge.set("思考中…", "accent")
        cmd = {
            "cmd": "chat", "messages": self._history(),
            "params": {
                "max_new_tokens": int(float(self.var_maxnew.get() or 128)),
                "temperature": float(self.var_temp.get()),
                "top_p": float(self.var_top_p.get()),
                "repetition_penalty": float(self.var_rep.get()),
            },
        }
        for kind in ("image", "audio"):
            if self._media.get(kind):
                cmd[kind] = self._media[kind]
        job = self.app.jobs.get(CHAT_KEY)
        if job and job.running and self._loaded == (
                self.var_model.get(), tuple(self._selected_adapters())):
            if not job.send(cmd):
                self.toast("发送失败：worker 未就绪", "err")
                self.btn_send.set_enabled(True)
        else:
            self._pending = cmd
            self.state_badge.set("加载模型…", "amber")

    def stop_gen(self) -> None:
        if self._cloud():
            self.toast("云端请求无法中途取消，等它返回即可", "info", 2600)
            return
        job = self.app.jobs.get(CHAT_KEY)
        if job and job.running:
            job.send({"cmd": "stop"})
            self.stats.configure(text="已请求停止…")

    # ------------------------------------------------------------ 云端引擎
    def _cloud(self) -> bool:
        return self._engine_labels.get(self.var_engine.get(), self.var_engine.get()) == CLOUD

    def _engine_changed(self) -> None:
        cloud = self._cloud()
        self.cb_model.configure(state="disabled" if cloud else "readonly")
        self.adapter_list.configure(state="disabled" if cloud else "normal")
        self.btn_unload.set_enabled(not cloud)
        if cloud:
            if not ai.enabled():
                self.state_badge.set("AI 未配置", "amber")
                return
            name = ai.default_model().split("/")[-1].replace(":free", "")
            self.state_badge.set(f"云端 {name[:16]}", "green")
            self.stats.configure(text="云端全模态：文字/图片直接上传；语音先本地转写成文字")
        else:
            self.state_badge.set("未加载", "panel3")

    def _send_cloud(self, text: str) -> None:
        parts: list[dict] = []
        need: tuple[str, ...] = ("text",)
        media = {k: v for k, v in self._media.items() if k in ("image", "audio")}
        if media.get("image"):
            try:
                parts.append(ai.image_part(media["image"]))
                need = ("text", "image")
            except OSError as exc:
                self._cloud_done(None, f"读取图片失败：{exc}")
                return
        history = self._history()

        def work() -> None:
            body = text
            if media.get("audio"):
                # 免费 Key 不能直接上传音频，先在本机转写成文字；图片仍直接上传。
                body, engine, err = audio.transcribe(media["audio"])
                if body:
                    self.app.post(lambda: self.toast(
                        f"语音已本地转写（{engine}），云端按文字回答", "info", 3600))
                    self.app.post(lambda: self.stats.configure(
                        text=f"语音→文字（{engine}）：{body[:24]}…"))
                else:
                    self.app.post(lambda: self._cloud_done(
                        None, f"语音转写失败（{err}）：到「设置 → AI 功能」装 Whisper"))
                    return
            messages = history[:-1] + [ai.user_msg(parts + [ai.text_part(body)])]

            def cb(piece: str) -> None:
                self.app.post(lambda: self._cloud_piece(piece))

            try:
                reply = ai.chat(messages, need=need, stream=cb, max_tokens=1400,
                                temperature=max(0.3, float(self.var_temp.get())))
                self.app.post(lambda: self._cloud_done(reply, ""))
            except Exception as exc:  # noqa: BLE001
                self.app.post(lambda err=str(exc): self._cloud_done(None, err))

        threading.Thread(target=work, daemon=True).start()

    def _cloud_piece(self, piece: str) -> None:
        self._stream_buf += piece
        if self._flush_job is None:
            self._flush_job = self.after(40, self._flush)

    def _cloud_done(self, reply, error: str) -> None:
        if self._flush_job:
            try:
                self.after_cancel(self._flush_job)
            except tk.TclError:
                pass
            self._flush_job = None
        self.btn_send.set_enabled(True)
        if error or reply is None:
            final = f"出错：{error or '调用失败'}"
            if self._stream_label is not None:
                self._stream_label.configure(text=final, fg=P["red"])
            self._stream_label = None
            self._stream_buf = ""
            self.state_badge.set("云端出错", "red")
            self.toast(error or "调用失败", "err", 5200)
            return
        final = reply.text
        if self._stream_label is not None:
            self._stream_label.configure(text=final or "（空回复）")
        self.messages.append({"role": "assistant", "content": final})
        self._stream_label = None
        self._stream_buf = ""
        self.state_badge.set("已就绪", "green")
        self.stats.configure(
            text=f"{reply.model.split('/')[-1]} · {reply.seconds:.1f}s · {len(final)} 字"
                 + (" · 已换过模型" if reply.attempts else ""))
        self._to_bottom()
        if self.var_speak.get():
            self._speak(final)

    def _speak(self, text: str) -> None:
        base_voice = str(self.cfg.get("ai.voice", "") or "")
        rate = int(self.cfg.get("ai.rate", 0) or 0)
        profile = voice.find(str(self.cfg.get("ai.voice_profile", "") or ""))

        def work() -> None:
            wav = (voice.speak(text[:400], profile, rate=rate) if profile
                   else audio.speak_wav(text[:400], voice=base_voice, rate=rate))
            if wav is not None:
                audio.play_wav(wav)

        threading.Thread(target=work, daemon=True).start()

    def clear(self) -> None:
        self.messages.clear()
        for w in self.thread.winfo_children():
            w.destroy()
        tk.Label(self.thread, text="对话已清空。", bg=P["panel"], fg=P["muted"], font=font(9),
                 anchor="w").pack(anchor="w", padx=8, pady=6)
        self.stats.configure(text="")
        self._stream_label = None

    def export(self) -> None:
        if not self.messages:
            self.toast("还没有对话内容", "warn")
            return
        out = paths.logs_dir() / f"chat-{time.strftime('%m%d-%H%M%S')}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# 对话记录 {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"- 模型：{self.var_model.get()}",
                 f"- 适配器：{' + '.join(self._selected_adapters()) or '（不使用）'}",
                 f"- 系统提示：{self.sys_entry.value() or '（无）'}",
                 f"- 采样：温度 {self.var_temp.get():.2f} · top_p {self.var_top_p.get():.2f} · "
                 f"最多 {self.var_maxnew.get()} token", ""]
        for m in self.messages:
            who = {"user": "你", "assistant": "助手", "system": "系统"}.get(m["role"], m["role"])
            lines.append(f"**{who}**：{m['content']}\n")
        try:
            out.write_text("\n".join(lines), encoding="utf-8")
            self.toast(f"已导出 {out.name}", "ok", 2600)
        except OSError as exc:
            self.toast(f"导出失败：{exc}", "err")

    # ------------------------------------------------------------ 气泡
    def _add_bubble(self, role: str, text: str) -> tk.Label:
        holder = tk.Frame(self.thread, bg=P["panel"])
        holder.pack(fill="x", pady=5, padx=6)
        is_user = role == "user"
        bg = P["panel2"] if is_user else P["panel3"]
        box = tk.Frame(holder, bg=bg, highlightthickness=1, highlightbackground=P["border"])
        box.pack(anchor="e" if is_user else "w", padx=(60, 0) if is_user else (0, 60))
        tk.Label(box, text="你" if is_user else "助手", bg=bg,
                 fg=P["muted"] if is_user else P["accent"], font=font(8, "bold"),
                 anchor="w").pack(anchor="w", padx=12, pady=(8, 0))
        lbl = tk.Label(box, text=text or "…", bg=bg, fg=P["text"], font=font(10),
                       anchor="w", justify="left", wraplength=620)
        lbl.pack(anchor="w", padx=12, pady=(2, 10))
        self._to_bottom()
        return lbl

    def _to_bottom(self) -> None:
        self.thread.update_idletasks()
        try:
            self.scroll.canvas.yview_moveto(1.0)
        except tk.TclError:
            pass

    def _flush(self) -> None:
        self._flush_job = None
        if self._stream_label is not None and self._stream_buf:
            try:
                self._stream_label.configure(text=self._stream_buf)
            except tk.TclError:
                return
            self._to_bottom()

    # ------------------------------------------------------------ 事件
    def _on_event(self, evt: dict) -> None:
        t = evt.get("type")
        if t == "ready":
            self._loaded = (self.var_model.get(), tuple(self._selected_adapters()))
            self.state_badge.set(f"已加载 {evt.get('params', 0)/1e6:.0f}M", "green")
            mods = "、".join(evt.get("modalities") or []) or "文本"
            self.stats.configure(
                text=f"{evt.get('model')} · {evt.get('params', 0)/1e6:.1f}M 参数 · "
                     f"加载 {evt.get('load_seconds')}s · 能力 {mods}"
                     + (f" · 适配器 {evt.get('adapter')}" if evt.get("adapter") else ""))
            if self._pending:
                job = self.app.jobs.get(CHAT_KEY)
                if job and job.send(self._pending):
                    self._pending = None
                else:
                    self.toast("消息发送失败", "err")
                    self.btn_send.set_enabled(True)
        elif t == "media":
            kinds = evt.get("kinds") or [evt.get("kind")]
            tokens = evt.get("tokens")
            if isinstance(tokens, dict):
                detail = "、".join(f"{k} {v}" for k, v in tokens.items())
            else:
                detail = str(tokens)
            self.state_badge.set(f"已注入 {detail} token（{'/'.join(str(k) for k in kinds)}）",
                                 "accent")
        elif t == "start":
            self.stats.configure(text=f"输入 {evt.get('prompt_tokens')} token，生成中…")
        elif t == "token":
            self._stream_buf += str(evt.get("text", ""))
            if self._flush_job is None:
                self._flush_job = self.after(40, self._flush)
            self.state_badge.set("输出中…", "accent")
        elif t == "done":
            if self._flush_job:
                try:
                    self.after_cancel(self._flush_job)
                except tk.TclError:
                    pass
                self._flush_job = None
            final = str(evt.get("text") or self._stream_buf)
            if self._stream_label is not None:
                self._stream_label.configure(text=final or "（空回复）")
            self.messages.append({"role": "assistant", "content": final})
            self._stream_label = None
            self._stream_buf = ""
            self.btn_send.set_enabled(True)
            secs = evt.get("seconds") or 0
            self.state_badge.set("已就绪", "green")
            self.stats.configure(
                text=f"输出 {evt.get('output_tokens', 0)} token · {secs:.1f}s · "
                     f"{evt.get('tps', 0)} tok/s · 首字 {evt.get('first_token_ms', '-')}ms"
                     + ("（已手动停止）" if evt.get("stopped") else ""))
            self._to_bottom()
            if self.var_speak.get() and final.strip():
                self._speak(final)
        elif t == "error":
            msg = str(evt.get("message", ""))
            if self._stream_label is not None:
                self._stream_label.configure(text=f"出错：{msg}", fg=P["red"])
                self._stream_label = None
            self.btn_send.set_enabled(True)
            self.state_badge.set("出错", "red")
            self.toast(msg, "err", 5000)
        elif t == "_exit":
            self._loaded = None
            self.state_badge.set("未加载", "panel3")
            self.btn_send.set_enabled(True)
            if evt.get("code") not in (0, None):
                self.stats.configure(text=f"对话进程退出 code={evt.get('code')}")
