# -*- coding: utf-8 -*-
"""实验台 · 通话：语音通话 / 视频通话。

两种引擎：
- 云端全模态（默认，只需 OpenRouter Key）：本地录音 → 语音识别 → 文本 + 关键帧
  发给免费全模态模型 → 流式回答 → 系统语音朗读。
- 本地模型：走 chat_worker（自己训练的适配器也能通话），纯文本/语音，不带画面。
"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ...core import ai, audio, camera as camera_mod, models as models_mod, paths, voice
from ..base import Page
from ..theme import P, font
from ..widgets import Badge, Card, Check, RBtn, ScrollFrame, TextArea

CALL_KEY = "call"
CAM_KEY = "call-cam"

CLOUD = "云端全模态（推荐）"
LOCAL = "本地模型"

SYS_CLOUD = ("你正在和用户实时通话。回答用简体中文、口语化，一次不超过三句话；"
             "不要使用 Markdown、不要罗列长清单。看不清或不确定时直接说不确定。")
SYS_VIDEO = SYS_CLOUD + "用户同时开着摄像头，你能看到画面里的一帧，需要时结合画面回答。"


class CallPage(Page):
    key = CALL_KEY
    group = "实验台"
    title = "通话"
    subtitle = "语音通话 / 视频通话：本地麦克风说话，云端全模态模型听声看画面并开口回答"

    def build(self) -> None:
        root = tk.Frame(self, bg=P["bg"])
        root.pack(fill="both", expand=True)
        root.grid_columnconfigure(0, weight=4, uniform="call")
        root.grid_columnconfigure(1, weight=5, uniform="call")
        root.grid_rowconfigure(0, weight=1)

        self._mode = "voice"
        self._engine = CLOUD
        self._active = False
        self._busy = False
        self._recording = False
        self._recorder: audio.Recorder | None = None
        self._camera: camera_mod.Camera | None = None
        self._frames = 0
        self._latest: Path | None = None
        self._photo = None
        self._messages: list[dict] = []
        self._stream_label: tk.Label | None = None
        self._stream_buf = ""
        self._flush_job: str | None = None
        self._t0 = 0.0
        self._eval_mode = ""
        self.local_job_ready = False

        self._build_left(root)
        self._build_right(root)

        self.app.bind_job(CALL_KEY, self._on_chat_event)
        self.app.bind_job(CAM_KEY, self._on_cam_event)

    # ---------------------------------------------------------------- 左
    def _build_left(self, parent) -> None:
        card = Card(parent, "通话设置", "云端引擎开箱即用；本地引擎需要 .venv")
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        b = card.body

        row = tk.Frame(b, bg=P["panel"])
        row.pack(fill="x")
        tk.Label(row, text="模式", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        self.btn_voice = RBtn(row, "语音通话", lambda: self._set_mode("voice"), kind="primary",
                              height=30, size=9, padx=12)
        self.btn_voice.pack(side="left", padx=(8, 6))
        self.btn_video = RBtn(row, "视频通话", lambda: self._set_mode("video"), kind="secondary",
                              height=30, size=9, padx=12)
        self.btn_video.pack(side="left")

        row2 = tk.Frame(b, bg=P["panel"])
        row2.pack(fill="x", pady=(10, 0))
        tk.Label(row2, text="引擎", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        self.var_engine = tk.StringVar(value=CLOUD)
        cb = ttk.Combobox(row2, textvariable=self.var_engine, state="readonly",
                          values=[CLOUD, LOCAL], font=font(9), width=22)
        cb.pack(side="left", padx=(8, 12), ipady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._engine_changed())
        tk.Label(row2, text="模型", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        self.var_model = tk.StringVar()
        self.cb_model = ttk.Combobox(row2, textvariable=self.var_model, state="readonly",
                                     font=font(9), width=18)
        self.cb_model.pack(side="left", padx=(8, 0), ipady=2)

        self.prev = tk.Canvas(b, width=420, height=226, bg=P["panel3"], highlightthickness=0,
                              bd=0)
        self.prev.pack(fill="x", pady=(12, 0))
        self.prev.bind("<Configure>", lambda e: self._paint_preview())

        state = tk.Frame(b, bg=P["panel"])
        state.pack(fill="x", pady=(10, 0))
        self.badge_mic = Badge(state, "麦克风检测中", "panel3", bg=P["panel"])
        self.badge_mic.pack(side="left")
        self.badge_cam = Badge(state, "摄像头未开", "panel3", bg=P["panel"])
        self.badge_cam.pack(side="left", padx=(8, 0))
        self.badge_ai = Badge(state, "云端待命", "panel3", bg=P["panel"])
        self.badge_ai.pack(side="left", padx=(8, 0))
        self.frame_label = tk.Label(state, text="", bg=P["panel"], fg=P["muted"], font=font(8),
                                    anchor="e")
        self.frame_label.pack(side="right")

        act = tk.Frame(b, bg=P["panel"])
        act.pack(fill="x", pady=(12, 0))
        self.btn_call = RBtn(act, "开始通话", self._toggle_call, kind="primary", height=36,
                             padx=16)
        self.btn_call.pack(fill="x")

        act2 = tk.Frame(b, bg=P["panel"])
        act2.pack(fill="x", pady=(10, 0))
        self.btn_talk = RBtn(act2, "按住说话", self._noop, kind="secondary", height=44,
                             size=11, padx=12)
        self.btn_talk.pack(fill="x")
        for widget in (self.btn_talk,):
            widget.bind("<ButtonPress-1>", self._talk_press)
            widget.bind("<ButtonRelease-1>", self._talk_release)
        RBtn(act2, "语音输入（4 秒）", self._voice_input, kind="ghost", height=30, size=9,
             padx=12).pack(fill="x", pady=(8, 0))

        self.status = tk.Label(b, text="点「开始通话」，然后按住说话（松开自动发送）。",
                               bg=P["panel"], fg=P["muted"], font=font(8), anchor="w",
                               justify="left", wraplength=420)
        self.status.pack(fill="x", pady=(10, 0))
        self._engine_changed()

    # ---------------------------------------------------------------- 右
    def _build_right(self, parent) -> None:
        card = Card(parent, "通话记录", "按住说话会自动发送；也可以直接打字")
        card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        b = card.body
        b.grid_rowconfigure(0, weight=1)
        b.grid_columnconfigure(0, weight=1)

        self.scroll = ScrollFrame(b, bg=P["panel"])
        self.scroll.grid(row=0, column=0, sticky="nsew")
        self.thread = self.scroll.inner
        self._hint = tk.Label(self.thread, text="还没有通话内容。", bg=P["panel"], fg=P["muted"],
                              font=font(9), anchor="w", justify="left", wraplength=520)
        self._hint.pack(anchor="w", padx=6, pady=8)

        inp = tk.Frame(b, bg=P["panel"])
        inp.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        inp.grid_columnconfigure(0, weight=1)
        self.input = TextArea(inp, height=2)
        self.input.grid(row=0, column=0, sticky="ew")
        self.input.bind("<Return>", self._on_enter)

        side = tk.Frame(inp, bg=P["panel"])
        side.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self.btn_send = RBtn(side, "发送", self._send_typed, kind="primary", height=32,
                             size=9, padx=14, width=88)
        self.btn_send.pack(fill="x")

        bar = tk.Frame(b, bg=P["panel"])
        bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.var_speak = tk.BooleanVar(value=True)
        Check(bar, "回答自动朗读", self.var_speak, bg=P["panel"]).pack(side="left")
        self.stats = tk.Label(bar, text="", bg=P["panel"], fg=P["muted"], font=font(8), anchor="w")
        self.stats.pack(side="left", padx=(12, 0))
        RBtn(bar, "清空", self._clear, kind="ghost", height=26, size=8, padx=10).pack(side="right")

    # ---------------------------------------------------------------- 状态
    def on_show(self) -> None:
        self._reload_models()
        self._probe_mic()
        self._refresh_ai_badge()
        if not self._latest:
            self._paint_preview()

    def on_hide(self) -> None:
        if self._active:
            self._stop_call(quiet=True)

    def _reload_models(self) -> None:
        models = models_mod.scan_models()
        names = [m.name for m in models]
        self.cb_model.configure(values=names)
        if names and self.var_model.get() not in names:
            self.var_model.set(models_mod.default_model_name(models, self.cfg.get("last.model", "")))

    def _probe_mic(self) -> None:
        ok, err = audio.mic_available()
        if ok:
            self.badge_mic.set("麦克风就绪", "green")
        else:
            self.badge_mic.set("没有麦克风", "amber")
            self.status.configure(text=f"{err}（仍可打字对话）", fg=P["amber"])

    def _refresh_ai_badge(self) -> None:
        if ai.enabled():
            st = ai.stats()
            ok, errs = int(st.get("ok", 0)), int(st.get("errors", 0))
            if st.get("last_error") and errs > ok:
                self.badge_ai.set("AI 异常", "amber")
            elif ok:
                self.badge_ai.set(f"AI 在线 · {ok} 次", "green")
            else:
                name = ai.default_model().split("/")[-1].replace(":free", "")
                self.badge_ai.set(f"AI {name[:16]}", "green")
        else:
            self.badge_ai.set("AI 未配置", "amber")

    def _set_mode(self, mode: str) -> None:
        if self._active:
            self.toast("先结束当前通话再切换模式", "warn")
            return
        self._mode = mode
        self.btn_voice.set_kind("primary" if mode == "voice" else "secondary")
        self.btn_video.set_kind("primary" if mode == "video" else "secondary")
        self._paint_preview()
        if mode == "video":
            if self._engine == LOCAL:
                if ai.enabled():
                    self.var_engine.set(CLOUD)
                    self._engine_changed()
                    self.toast("视频通话需要能看画面的云端模型，已切到「云端全模态」", "info", 3200)
                else:
                    self.toast("视频通话需要云端 Key（本地模型没有画面理解）", "warn", 4200)
            self.status.configure(text="视频通话会实时显示摄像头画面，发送时自动带上当前一帧。",
                                  fg=P["muted"])
        else:
            self.status.configure(text="语音通话：按住说话，松开后自动识别并发送。", fg=P["muted"])

    def _engine_changed(self) -> None:
        self._engine = self.var_engine.get() or CLOUD
        local = self._engine == LOCAL
        self.cb_model.configure(state="readonly" if local else "disabled")
        self.btn_voice.configure(cursor="hand2")
        if local:
            self.status.configure(text="本地引擎：需要训练环境（.venv），只有语音/文字，没有画面理解。",
                                  fg=P["muted"])
            self.badge_ai.set("本地模型", "panel3")
        else:
            self._refresh_ai_badge()

    # ---------------------------------------------------------------- 通话
    def _toggle_call(self) -> None:
        if self._active:
            self._stop_call()
            return
        if self._engine == LOCAL:
            if self._mode == "video" and ai.enabled():
                self.var_engine.set(CLOUD)
                self._engine_changed()
            elif not self._ensure_worker():
                return
        if self._engine != LOCAL:
            if not ai.enabled():
                self.toast("先在「设置 → AI 功能」配置 OpenRouter Key", "warn", 4200)
                self.app.goto("settings")
                return
        self._active = True
        self._t0 = time.time()
        self.btn_call.set_text("结束通话")
        self.btn_call.set_kind("danger")
        self.status.configure(text="通话中：按住说话，松开自动发送。", fg=P["green"])
        if self._mode == "video":
            self._start_camera()
        self._add_bubble("system", "通话已开始")

    def _stop_call(self, quiet: bool = False) -> None:
        if self._recording:
            self._recording = False
            if self._recorder:
                self._recorder.stop(discard=True)
            self._recorder = None
        self._active = False
        self.btn_call.set_text("开始通话")
        self.btn_call.set_kind("primary")
        self.btn_talk.set_text("按住说话")
        self._stop_camera()
        job = self.app.jobs.get(CALL_KEY)
        if job and job.running:
            if not job.send({"cmd": "quit"}):
                job.kill()
        audio.stop_play()
        self.badge_cam.set("摄像头未开", "panel3")
        if not quiet:
            self.status.configure(text="通话已结束。", fg=P["muted"])

    def _start_camera(self) -> None:
        if not camera_mod.available():
            self.badge_cam.set("摄像头不可用", "amber")
            self.toast("这个版本没有摄像头组件：请用最新的 exe", "err", 4600)
            return
        cam = camera_mod.Camera(fps=3, width=480, height=360)
        ok, err = cam.open()
        if not ok:
            self.badge_cam.set("摄像头打不开", "amber")
            self.toast(err or "摄像头打不开", "err", 5200)
            return
        self._camera = cam
        self._frames = cam.frame_count
        self.badge_cam.set(f"摄像头 {cam.size[0]}×{cam.size[1]}", "green")
        self.status.configure(text="视频通话中：画面实时刷新，发送时自动带上当前一帧。",
                              fg=P["green"])
        self.frame_label.configure(text="画面准备中…")

    def _stop_camera(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        job = self.app.jobs.get(CAM_KEY)
        if job and job.running:
            if not job.send({"cmd": "stop"}):
                job.kill()
            self.after(600, lambda: job.running and job.kill())
        self._frames = 0
        self._latest = None
        self.frame_label.configure(text="")

    def _on_cam_event(self, evt: dict) -> None:
        # 仅旧版 worker 模式会走到这里；新版摄像头在进程内直接出帧
        t = evt.get("type")
        if t == "ready":
            size = evt.get("size") or [0, 0]
            self.badge_cam.set(f"摄像头 {size[0]}×{size[1]}", "green")
        elif t == "frame":
            self._frames = int(evt.get("n") or self._frames)
            self._latest = Path(str(evt.get("path") or ""))
            self.frame_label.configure(text=f"已取帧 {self._frames}")
        elif t == "error":
            self.badge_cam.set("摄像头出错", "red")
            self.toast(str(evt.get("message") or ""), "err", 5200)
        elif t == "_exit":
            self.badge_cam.set("摄像头已关闭", "panel3")

    def tick(self) -> None:
        cam = self._camera
        camera_failed = False
        if self._mode == "video" and self._active and cam is not None:
            if cam.error and not cam.running:
                camera_failed = True
                self.badge_cam.set("摄像头中断", "amber")
                self.status.configure(text=f"摄像头：{cam.error}", fg=P["amber"])
            elif cam.running and cam.frame_count != self._frames:
                self._frames = cam.frame_count
                self._latest = cam.latest
                self.frame_label.configure(text=f"已取帧 {self._frames}")
                self._paint_preview()
        if self._active and not camera_failed:
            self.status.configure(text=f"通话中 {int(time.time() - self._t0)}s · "
                                       f"{'语音' if self._mode == 'voice' else '视频'} · "
                                       f"{self._engine}", fg=P["green"])

    # ---------------------------------------------------------------- 预览
    def _paint_preview(self) -> None:
        cv = self.prev
        cv.delete("all")
        w = max(80, cv.winfo_width())
        h = max(80, cv.winfo_height())
        if self._latest and self._latest.is_file():
            try:
                from PIL import Image, ImageTk

                with Image.open(self._latest) as img:
                    img.thumbnail((w - 8, h - 8))
                    self._photo = ImageTk.PhotoImage(img)
                cv.create_image(w // 2, h // 2, image=self._photo)
                return
            except Exception:  # noqa: BLE001
                pass
        if self._frames:
            cv.create_text(w // 2, h // 2, text=f"已收到 {self._frames} 帧\n（画面预览不可用）",
                           fill=P["text2"], font=font(10), justify="center")
            return
        tip = ("视频通话：点开始后这里显示摄像头画面"
               if self._mode == "video" else "语音通话：不需要画面")
        cv.create_text(w // 2, h // 2, text=tip, fill=P["muted"], font=font(10),
                       justify="center")

    # ---------------------------------------------------------------- 录音
    def _noop(self) -> None:
        pass

    def _talk_press(self, _e=None) -> str:
        if not self._active:
            self.toast("先点「开始通话」", "warn")
            return "break"
        if self._busy:
            self.toast("上一句还在回答，稍等", "warn")
            return "break"
        ok, err = audio.mic_available()
        if not ok:
            self.toast(err or "没有可录音设备", "err", 4200)
            return "break"
        self._recorder = audio.Recorder()
        if not self._recorder.start():
            self.toast(self._recorder.error or "录音启动失败", "err", 4200)
            self._recorder = None
            return "break"
        self._recording = True
        self.btn_talk.set_text("松开结束（录音中…）")
        self.btn_talk.set_kind("danger")
        return "break"

    def _talk_release(self, _e=None) -> str:
        if not self._recording:
            return "break"
        self._recording = False
        self.btn_talk.set_text("按住说话")
        self.btn_talk.set_kind("secondary")
        wav = self._recorder.stop() if self._recorder else None
        self._recorder = None
        if wav is None:
            self.toast("没有录到声音", "warn")
            return "break"
        self.status.configure(text="识别中…", fg=P["accent"])
        threading.Thread(target=self._recognize, args=(wav,), daemon=True).start()
        return "break"

    def _voice_input(self) -> None:
        if self._busy:
            self.toast("正在处理上一句", "warn")
            return
        ok, err = audio.mic_available()
        if not ok:
            self.toast(err or "没有可录音设备", "err", 4200)
            return
        self.status.configure(text="录音 4 秒…请说话", fg=P["accent"])
        self.btn_talk.set_text("录音中…")

        def work() -> None:
            wav, error = audio.record_for(4.0)
            if wav is None:
                self.app.post(lambda: self._asr_failed(error or "没有录到声音"))
                return
            text, engine, err2 = audio.transcribe(wav)
            self.app.post(lambda: self._fill_input(text, engine, err2))

        threading.Thread(target=work, daemon=True).start()

    def _recognize(self, wav: Path) -> None:
        text, engine, err = audio.transcribe(wav)
        if not text:
            self.app.post(lambda: self._asr_failed(err))
            return
        self.app.post(lambda: self._asr_done(text, engine))

    def _asr_failed(self, error: str) -> None:
        self.btn_talk.set_text("按住说话")
        self.status.configure(text=f"识别失败：{error}", fg=P["amber"])
        self.toast(error or "识别失败", "warn", 4200)

    def _asr_done(self, text: str, engine: str) -> None:
        self.status.configure(text=f"识别完成（{engine}）：{text[:40]}", fg=P["green"])
        self._send(text)

    def _fill_input(self, text: str, engine: str, error: str) -> None:
        self.btn_talk.set_text("按住说话")
        if not text:
            self.status.configure(text=f"识别失败：{error}", fg=P["amber"])
            self.toast(error or "没听清", "warn", 4200)
            return
        self.input.delete("1.0", "end")
        self.input.insert("1.0", text)
        self.status.configure(text=f"识别完成（{engine}），确认后点发送", fg=P["green"])

    # ---------------------------------------------------------------- 对话
    def _on_enter(self, event) -> str | None:
        if event.state & 0x0001:
            return None
        self._send_typed()
        return "break"

    def _send_typed(self) -> None:
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        self.input.delete("1.0", "end")
        self._send(text)

    def _send(self, text: str) -> None:
        if self._busy:
            self.toast("上一句还在回答，稍等", "warn")
            return
        if not self._active:
            self._toggle_call()
            if not self._active:
                return
        text = text[:2000]
        self._messages.append({"role": "user", "content": text})
        self._add_bubble("user", text)
        self._stream_label = self._add_bubble("assistant", "")
        self._stream_buf = ""
        self._busy = True
        self.btn_send.set_enabled(False)
        if self._engine == LOCAL:
            self._ask_local(text)
        else:
            self._ask_cloud(text)

    def _history(self, latest: str) -> list[dict]:
        msgs = list(self._messages[:-1])[-8:]
        return msgs + [{"role": "user", "content": latest}]

    # -- 云端
    def _image_part(self) -> dict | None:
        if self._mode != "video" or not self._latest or not self._latest.is_file():
            return None
        return ai.image_part(self._latest)

    def _ask_cloud(self, text: str) -> None:
        try:
            img = self._image_part()
        except OSError as exc:
            self._cloud_done(None, time.time(), f"读取摄像头画面失败：{exc}")
            return
        parts: list[dict] = []
        if img:
            parts.append(img)
        parts.append(ai.text_part(text))
        system = SYS_VIDEO if img else SYS_CLOUD
        messages = ([{"role": "system", "content": system}]
                    + [m for m in self._messages[:-1]][-6:] + [ai.user_msg(parts)])
        need = ("text", "image") if img else ("text",)
        t0 = time.time()

        def work() -> None:
            def cb(piece: str) -> None:
                self.app.post(lambda: self._stream_piece(piece))

            try:
                reply = ai.chat(messages, need=need, stream=cb, max_tokens=700,
                                temperature=0.7, timeout=60)
                self.app.post(lambda: self._cloud_done(reply, t0, ""))
            except Exception as exc:  # noqa: BLE001
                self.app.post(lambda err=str(exc): self._cloud_done(None, t0, err))

        threading.Thread(target=work, daemon=True).start()

    def _cloud_done(self, reply, t0: float, error: str) -> None:
        self._busy = False
        self.btn_send.set_enabled(True)
        if error or reply is None:
            self._finish_bubble(f"出错：{error}")
            self.status.configure(text=f"云端出错：{error}", fg=P["red"])
            self.toast(error or "调用失败", "err", 5000)
            return
        self._finish_bubble(reply.text)
        self._messages.append({"role": "assistant", "content": reply.text})
        self.stats.configure(
            text=f"{reply.model.split('/')[-1]} · {reply.seconds:.1f}s · "
                 f"输出 {len(reply.text)} 字")
        self.status.configure(text="已回复", fg=P["green"])
        if self.var_speak.get():
            self._speak(reply.text)

    # -- 本地
    def _ensure_worker(self) -> bool:
        name = self.var_model.get()
        model = models_mod.find_model(name)
        if model is None:
            self.toast("先选一个本地模型", "warn")
            return False
        job = self.app.jobs.get(CALL_KEY)
        if job and job.running and getattr(self, "_loaded_name", "") == name:
            return True
        if job and job.running:
            job.kill()
        cfg = {"model": str(model.path), "adapter": "",
               "threads": int(self.cfg.get("threads", 0) or 0)}
        if not self.app.spawn_worker(CALL_KEY, "chat_worker.py", cfg, label="本地通话",
                                     stdin_pipe=True):
            return False
        self._loaded_name = name
        self.badge_ai.set("本地加载中", "amber")
        return True

    def _ask_local(self, text: str) -> None:
        job = self.app.jobs.get(CALL_KEY)
        if not (job and job.running):
            self._busy = False
            self.btn_send.set_enabled(True)
            self.toast("本地模型未就绪", "err")
            return
        cmd = {"cmd": "chat", "messages": self._history(text)[-6:],
               "params": {"max_new_tokens": 160, "temperature": 0.7, "top_p": 0.9}}
        if not job.send(cmd):
            self._busy = False
            self.btn_send.set_enabled(True)
            self.toast("发送失败", "err")

    def _on_chat_event(self, evt: dict) -> None:
        t = evt.get("type")
        if t == "ready":
            self.badge_ai.set(f"已加载 {evt.get('params', 0)/1e6:.0f}M", "green")
            self.status.configure(text="本地模型就绪，可以开始说话。", fg=P["green"])
        elif t == "token":
            self._stream_buf += str(evt.get("text", ""))
            if self._flush_job is None:
                self._flush_job = self.after(40, self._flush)
        elif t == "done":
            self._cancel_flush()
            text = str(evt.get("text") or self._stream_buf)
            self._finish_bubble(text)
            self._messages.append({"role": "assistant", "content": text})
            self._busy = False
            self.btn_send.set_enabled(True)
            self.stats.configure(text=f"输出 {evt.get('output_tokens', 0)} token · "
                                      f"{evt.get('tps', 0)} tok/s")
            if self.var_speak.get():
                self._speak(text)
        elif t == "error":
            self._finish_bubble(f"出错：{evt.get('message', '')}")
            self._busy = False
            self.btn_send.set_enabled(True)
            self.toast(str(evt.get("message", "")), "err", 5000)
        elif t == "_exit":
            self.badge_ai.set("本地模型已卸载", "panel3")
            self._loaded_name = ""

    # ---------------------------------------------------------------- 朗读
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

    # ---------------------------------------------------------------- 气泡
    def _add_bubble(self, role: str, text: str) -> tk.Label:
        try:
            self._hint.pack_forget()
        except tk.TclError:
            pass
        holder = tk.Frame(self.thread, bg=P["panel"])
        holder.pack(fill="x", pady=4, padx=4)
        is_user = role == "user"
        is_sys = role == "system"
        bg = P["panel2"] if is_user else P["panel3"]
        box = tk.Frame(holder, bg=bg, highlightthickness=1, highlightbackground=P["border"])
        if is_sys:
            box.pack(anchor="center")
        else:
            box.pack(anchor="e" if is_user else "w", padx=(48, 0) if is_user else (0, 48))
        who = "你" if is_user else ("提示" if is_sys else "助手")
        tk.Label(box, text=who, bg=bg, fg=P["muted"] if is_user else P["accent"],
                 font=font(8, "bold"), anchor="w").pack(anchor="w", padx=10, pady=(6, 0))
        lbl = tk.Label(box, text=text or "…", bg=bg, fg=P["text2"] if is_sys else P["text"],
                       font=font(10), anchor="w", justify="left", wraplength=470)
        lbl.pack(anchor="w", padx=10, pady=(2, 8))
        self._to_bottom()
        return lbl

    def _finish_bubble(self, text: str) -> None:
        self._cancel_flush()
        if self._stream_label is not None:
            try:
                self._stream_label.configure(text=text or "（空回复）")
            except tk.TclError:
                pass
        self._stream_label = None
        self._stream_buf = ""
        self._to_bottom()

    def _stream_piece(self, piece: str) -> None:
        self._stream_buf += piece
        if self._flush_job is None:
            self._flush_job = self.after(40, self._flush)

    def _flush(self) -> None:
        self._flush_job = None
        if self._stream_label is not None and self._stream_buf:
            try:
                self._stream_label.configure(text=self._stream_buf)
            except tk.TclError:
                return
            self._to_bottom()

    def _cancel_flush(self) -> None:
        if self._flush_job:
            try:
                self.after_cancel(self._flush_job)
            except tk.TclError:
                pass
            self._flush_job = None

    def _to_bottom(self) -> None:
        self.thread.update_idletasks()
        try:
            self.scroll.canvas.yview_moveto(1.0)
        except tk.TclError:
            pass

    def _clear(self) -> None:
        self._messages.clear()
        for w in self.thread.winfo_children():
            w.destroy()
        self._hint = tk.Label(self.thread, text="还没有通话内容。", bg=P["panel"], fg=P["muted"],
                              font=font(9), anchor="w")
        self._hint.pack(anchor="w", padx=6, pady=8)
        self._stream_label = None
        self.stats.configure(text="")
