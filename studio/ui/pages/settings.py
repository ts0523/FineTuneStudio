# -*- coding: utf-8 -*-
"""设置页（简化版）：常用功能一屏能看完，环境/镜像/线程收进「高级设置」。

常用：
- AI：Key、模型、一键检测并选最快的免费模型、额度
- 语音与音色：识别方式、朗读音色、音色克隆（录 8 秒即可）
- 外观：白色森林等 4 套预设，调色收进「高级调色」

高级（默认收起）：
- 解释器、镜像地址、线程数、日志行数、启动自检、回收站、重建训练环境
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ...core import ai as ai_mod
from ...core import audio as audio_mod
from ...core import config as config_mod
from ...core import paths, procs, runtime, secrets, voice
from .. import theme
from ..base import Page
from ..theme import P, font, round_rect
from ..widgets import Badge, Card, Check, Entry, Labeled, RBtn, ScrollFrame

SETUP_KEY = "setup"

SAMPLE_TEXT = "今天天气不错，我们一起去公园走走吧。这是一段用来克隆音色的样本。"

ASR_AUTO = "自动（系统识别优先，快）"
ASR_SAPI = "系统识别（无需安装）"
ASR_WHISPER = "Whisper（本地，更准但慢）"


class SettingsPage(Page):
    key = "settings"
    title = "设置"
    subtitle = "常用功能放上面；环境、镜像、线程都收进最下面的「高级设置」"

    def build(self) -> None:
        self._profiles: list[dict] = []
        self._colors_open = False
        self._adv_open = False
        self._swatches: dict[str, tk.Canvas] = {}
        self._hex_labels: dict[str, tk.Label] = {}
        self._cards: dict[str, tk.Canvas] = {}
        self._color_names = {"bg": "背景", "panel": "面板", "panel2": "次级面板",
                             "border": "边框", "text": "正文", "text2": "次要文字",
                             "accent": "主色"}
        self.custom = {k: v for k, v in (self.cfg.get("theme_custom") or {}).items()
                       if isinstance(v, str)}

        outer = ScrollFrame(self)
        outer.pack(fill="both", expand=True)
        self.outer = outer
        f = outer.inner
        f.grid_columnconfigure(0, weight=1)

        self._build_ai(f)
        self._build_voice(f)
        self._build_look(f)
        self._build_advanced(f)
        self._build_about(f)
        self.app.bind_job(SETUP_KEY, self._on_setup_event)

    # ---------------------------------------------------------------- AI
    def _build_ai(self, parent) -> None:
        card = Card(parent, "AI 功能（OpenRouter）",
                    "实验台云端模型、数据层造数、训练层帮训、通话，都用这里的 Key")
        card.grid(row=0, column=0, sticky="ew")
        b = card.body

        self.var_key = tk.StringVar(value="")
        row = tk.Frame(b, bg=P["panel"])
        row.pack(fill="x")
        tk.Label(row, text="Key", bg=P["panel"], fg=P["muted"], font=font(9), width=5,
                 anchor="w").pack(side="left")
        Entry(row, textvariable=self.var_key, show="*", width=44).pack(
            side="left", fill="x", expand=True, ipady=3)
        RBtn(row, "保存并测试", self._ai_save_test, kind="primary", height=32, size=9,
             padx=14).pack(side="left", padx=(8, 0))
        RBtn(row, "查额度", self._ai_quota, kind="ghost", height=32, size=9,
             padx=10).pack(side="left", padx=6)
        RBtn(row, "申请 Key", self._open_keys_page, kind="ghost", height=32, size=8,
             padx=10).pack(side="left")
        RBtn(row, "清除本地 Key", self._clear_key, kind="danger", height=32, size=8,
             padx=10).pack(side="left", padx=6)
        self.key_note = tk.Label(b, text="", bg=P["panel"], fg=P["muted"], font=font(8),
                                 anchor="w", justify="left", wraplength=860)
        self.key_note.pack(fill="x", pady=(6, 0))

        row2 = tk.Frame(b, bg=P["panel"])
        row2.pack(fill="x", pady=(10, 0))
        tk.Label(row2, text="模型", bg=P["panel"], fg=P["muted"], font=font(9), width=5,
                 anchor="w").pack(side="left")
        self.var_ai_model = tk.StringVar(value=str(self.cfg.get("ai.model", "") or ai_mod.TEXT_MODEL))
        self.cb_ai_model = ttk.Combobox(row2, textvariable=self.var_ai_model,
                                        values=self._model_options(), font=font(9), width=46)
        self.cb_ai_model.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=2)
        RBtn(row2, "一键检测并选最快", self._ai_diagnose, kind="secondary", height=32,
             size=9, padx=12).pack(side="left")
        RBtn(row2, "测试", self._ai_test_model, kind="ghost", height=32, size=9,
             padx=10).pack(side="left", padx=6)

        state = tk.Frame(b, bg=P["panel"])
        state.pack(fill="x", pady=(10, 0))
        self.ai_badge = Badge(state, "未测试", "panel3", bg=P["panel"])
        self.ai_badge.pack(side="left")
        self.ai_status = tk.Label(state, text="", bg=P["panel"], fg=P["muted"], font=font(9),
                                  anchor="w", justify="left", wraplength=860)
        self.ai_status.pack(side="left", fill="x", expand=True, padx=(8, 0))

        foot = tk.Frame(b, bg=P["panel"])
        foot.pack(fill="x", pady=(8, 0))
        tk.Label(foot, text="免费模型每天 50 次；应用会自动关闭思维链，避免空回复与长时间等待。",
                 bg=P["panel"], fg=P["muted"], font=font(8), anchor="w").pack(side="left")
        RBtn(foot, "高级设置 ▾", self._toggle_advanced, kind="ghost", height=26, size=8,
             padx=10).pack(side="right")

    # ---------------------------------------------------------------- 语音
    def _build_voice(self, parent) -> None:
        card = Card(parent, "语音与音色", "识别、朗读、音色克隆都在这；不需要安装第三方模型")
        card.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        b = card.body

        row = tk.Frame(b, bg=P["panel"])
        row.pack(fill="x")
        tk.Label(row, text="语音识别", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        cur = str(self.cfg.get("ai.asr", "auto") or "auto")
        self.var_asr = tk.StringVar(
            value=ASR_WHISPER if cur == "whisper" else (ASR_SAPI if cur == "sapi" else ASR_AUTO))
        ttk.Combobox(row, textvariable=self.var_asr, state="readonly",
                     values=[ASR_AUTO, ASR_SAPI, ASR_WHISPER], width=26,
                     font=font(9)).pack(side="left", padx=(6, 16), ipady=2)

        tk.Label(row, text="朗读音色", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        self.var_voice = tk.StringVar(value=str(self.cfg.get("ai.voice", "") or ""))
        self.cb_voice = ttk.Combobox(row, textvariable=self.var_voice, state="readonly",
                                     font=font(9), width=22)
        self.cb_voice.pack(side="left", padx=(6, 10), ipady=2)
        tk.Label(row, text="语速", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        self.var_rate = tk.StringVar(value=str(self.cfg.get("ai.rate", 0)))
        ttk.Spinbox(row, from_=-10, to=10, increment=1, textvariable=self.var_rate,
                    width=4, font=font(9)).pack(side="left", padx=(4, 10), ipady=2)
        RBtn(row, "刷新音色", self._load_voices, kind="ghost", height=26, size=8,
             padx=10).pack(side="left")
        RBtn(row, "保存语音设置", self._save_voice, kind="secondary", height=26, size=8,
             padx=10).pack(side="right")

        line = tk.Frame(b, bg=P["panel2"], highlightbackground=P["border"], highlightthickness=1)
        line.pack(fill="x", pady=(12, 0), ipady=8, ipadx=10)
        tk.Label(line, text="音色克隆（轻量版）", bg=P["panel2"], fg=P["text"],
                 font=font(10, "bold"), anchor="w").pack(anchor="w", padx=10)
        tk.Label(line,
                 text=f"朗读下面这段话 8 秒，应用会估计你的音高与语速，并用它朗读回答：\n「{SAMPLE_TEXT}」",
                 bg=P["panel2"], fg=P["text2"], font=font(9), justify="left",
                 wraplength=820).pack(anchor="w", padx=10, pady=(4, 6))

        crow = tk.Frame(line, bg=P["panel2"])
        crow.pack(fill="x", padx=10)
        tk.Label(crow, text="名字", bg=P["panel2"], fg=P["muted"], font=font(9)).pack(side="left")
        self.var_clone_name = tk.StringVar(value="我的音色")
        Entry(crow, textvariable=self.var_clone_name, width=14).pack(side="left", padx=(6, 12),
                                                                    ipady=2)
        RBtn(crow, "录 8 秒并克隆", self._record_clone, kind="primary", height=30, size=9,
             padx=12).pack(side="left")
        tk.Label(crow, text="已克隆", bg=P["panel2"], fg=P["muted"], font=font(9)).pack(
            side="left", padx=(18, 6))
        self.var_profile = tk.StringVar()
        self.cb_profile = ttk.Combobox(crow, textvariable=self.var_profile, state="readonly",
                                       font=font(9), width=30)
        self.cb_profile.pack(side="left", ipady=2)
        self.cb_profile.bind("<<ComboboxSelected>>", lambda e: self._pick_profile())
        RBtn(crow, "试听", self._preview_profile, kind="ghost", height=30, size=9,
             padx=10).pack(side="left", padx=6)
        RBtn(crow, "删除", self._delete_profile, kind="ghost", height=30, size=9,
             padx=10).pack(side="left")
        self.clone_note = tk.Label(line, text="", bg=P["panel2"], fg=P["muted"], font=font(8),
                                   anchor="w", justify="left", wraplength=820)
        self.clone_note.pack(fill="x", padx=10, pady=(8, 0))

    # ---------------------------------------------------------------- 外观
    def _build_look(self, parent) -> None:
        card = Card(parent, "外观", "默认「白色森林」；点色卡立即切换")
        card.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        b = card.body
        lang_row = tk.Frame(b, bg=P["panel"])
        lang_row.pack(fill="x", pady=(0, 10))
        tk.Label(lang_row, text="界面语言 / UI language", bg=P["panel"], fg=P["muted"],
                 font=font(9), anchor="w").pack(side="left")
        from .. import i18n
        self.var_lang = tk.StringVar(
            value="English" if i18n.language() == "en" else "中文 / Chinese")
        cb = ttk.Combobox(lang_row, textvariable=self.var_lang, state="readonly",
                          values=["中文 / Chinese", "English"], width=18, font=font(9))
        cb.pack(side="left", padx=(10, 0), ipady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._change_lang())
        preset_row = tk.Frame(b, bg=P["panel"])
        preset_row.pack(fill="x")
        for key, name, desc in theme.preset_choices():
            self._preset_card(preset_row, key, name, desc)
        RBtn(card.actions, "高级调色", self._toggle_colors, kind="ghost", height=28,
             size=8, padx=10).pack()

        self.custom_box = tk.Frame(b, bg=P["panel"])
        self.custom_box.pack(fill="x", pady=(14, 0))
        tk.Label(self.custom_box, text="自由调节（只覆盖你改的那几个颜色，没改的跟随预设）",
                 bg=P["panel"], fg=P["text2"], font=font(9, "bold"), anchor="w").pack(fill="x")
        grid = tk.Frame(self.custom_box, bg=P["panel"])
        grid.pack(fill="x", pady=(8, 0))
        for i, key in enumerate(theme.EDITABLE):
            box = tk.Frame(grid, bg=P["panel"])
            box.grid(row=i // 4, column=i % 4, sticky="w", padx=(0, 12), pady=(0, 8))
            grid.grid_columnconfigure(i % 4, weight=1)
            tk.Label(box, text=self._color_names.get(key, key), bg=P["panel"], fg=P["muted"],
                     font=font(8), anchor="w").pack(anchor="w")
            line = tk.Frame(box, bg=P["panel"])
            line.pack(anchor="w", pady=(3, 0))
            sw = tk.Canvas(line, width=30, height=22, bg=P["panel"], highlightthickness=0,
                           cursor="hand2")
            sw.pack(side="left")
            sw.bind("<Button-1>", lambda e, k=key: self._pick_color(k))
            self._swatches[key] = sw
            lbl = tk.Label(line, text="", bg=P["panel"], fg=P["text2"], font=font(8, mono=True))
            lbl.pack(side="left", padx=(6, 0))
            self._hex_labels[key] = lbl
        self._paint_swatches()
        act = tk.Frame(self.custom_box, bg=P["panel"])
        act.pack(fill="x", pady=(8, 0))
        RBtn(act, "应用并保存", self._apply_custom, kind="primary", height=30, size=9,
             padx=12).pack(side="left")
        RBtn(act, "恢复预设", self._reset_custom, kind="ghost", height=30, size=9,
             padx=10).pack(side="left", padx=8)
        self.custom_box.pack_forget()

    # ---------------------------------------------------------------- 高级
    def _change_lang(self) -> None:
        from .. import i18n
        lang = "en" if self.var_lang.get().startswith("English") else "zh"
        self.cfg.set("ui.lang", lang, save=False)
        self.cfg.save()
        i18n.set_language(lang)
        self.app.rebuild_ui()
        self.toast("Language switched" if lang == "en" else "界面语言已切换", "ok")

    def _build_advanced(self, parent) -> None:
        card = Card(parent, "高级设置（本地训练 / 镜像 / 线程）",
                    "只影响本地训练与下载，不影响云端体验；默认收起")
        card.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        self.adv_card = card
        RBtn(card.actions, "展开 ▾", self._toggle_advanced, kind="ghost", height=28,
             size=8, padx=10).pack()
        self.adv_inner = tk.Frame(card.body, bg=P["panel"])
        self.adv_inner.pack(fill="x", pady=(6, 0))

        self._kv = {}
        for label, key in (("项目根目录", "root"), ("Python 解释器", "python"),
                           ("训练依赖", "libs")):
            row = tk.Frame(self.adv_inner, bg=P["panel"])
            row.pack(fill="x", pady=(0, 6))
            tk.Label(row, text=label, bg=P["panel"], fg=P["muted"], font=font(9), width=12,
                     anchor="w").pack(side="left")
            v = tk.Label(row, text="-", bg=P["panel"], fg=P["text2"], font=font(9), anchor="w",
                         justify="left", wraplength=820)
            v.pack(side="left", fill="x", expand=True)
            self._kv[key] = v

        w = Labeled(self.adv_inner, "自定义解释器路径（留空自动用 .venv）", "", bg=P["panel"])
        w.pack(fill="x", pady=(2, 8))
        self.var_python = tk.StringVar(value=str(self.cfg.get("python", "")))
        Entry(w.holder, textvariable=self.var_python,
              placeholder="例如 D:\\Python\\python.exe").pack(fill="x", ipady=3)

        grid = tk.Frame(self.adv_inner, bg=P["panel"])
        grid.pack(fill="x")
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        w1 = Labeled(grid, "模型镜像地址（国内默认 hf-mirror.com）", "", bg=P["panel"])
        w1.grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=(0, 8))
        self.var_endpoint = tk.StringVar(value=str(self.cfg.get("endpoint", "")))
        Entry(w1.holder, textvariable=self.var_endpoint,
              placeholder="https://hf-mirror.com").pack(fill="x", ipady=3)
        w2 = Labeled(grid, "默认 CPU 线程数（0 = 自动）", "", bg=P["panel"])
        w2.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=(0, 8))
        self.var_threads = tk.StringVar(value=str(self.cfg.get("threads", 0)))
        ttk.Spinbox(w2.holder, from_=0, to=64, increment=1, textvariable=self.var_threads,
                    font=font(9)).pack(fill="x", ipady=3)
        w3 = Labeled(grid, "日志面板保留行数", "", bg=P["panel"])
        w3.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.var_loglines = tk.StringVar(value=str(self.cfg.get("log_lines", 3000)))
        ttk.Spinbox(w3.holder, from_=500, to=50000, increment=500,
                    textvariable=self.var_loglines, font=font(9)).pack(fill="x", ipady=3)

        self.var_auto = tk.BooleanVar(value=bool(self.cfg.get("auto_check_env", True)))
        self.var_recycle = tk.BooleanVar(value=bool(self.cfg.get("recycle_delete", True)))
        checks = tk.Frame(self.adv_inner, bg=P["panel"])
        checks.pack(fill="x", pady=(10, 0))
        Check(checks, "启动时自动检查训练环境", self.var_auto, bg=P["panel"]).pack(side="left")
        Check(checks, "删除模型/产物时走回收站", self.var_recycle, bg=P["panel"]).pack(
            side="left", padx=16)

        act = tk.Frame(self.adv_inner, bg=P["panel"])
        act.pack(fill="x", pady=(10, 0))
        RBtn(act, "保存高级设置", self._save, kind="secondary", height=30, size=9,
             padx=12).pack(side="left")
        RBtn(act, "重新检测", self._recheck, kind="ghost", height=30, size=9, padx=10).pack(
            side="left", padx=6)
        RBtn(act, "打开项目目录", lambda: procs.open_in_explorer(paths.root()), kind="ghost",
             height=30, size=9, padx=10).pack(side="left")
        RBtn(act, "重建训练环境", self._rebuild_env, kind="danger", height=30, size=9,
             padx=10).pack(side="right")
        RBtn(act, "恢复全部默认", self._reset, kind="ghost", height=30, size=9,
             padx=10).pack(side="right", padx=6)
        RBtn(act, "重启界面", self._restart, kind="ghost", height=30, size=9,
             padx=10).pack(side="right")

        self.setup_log = tk.Label(self.adv_inner, text="", bg=P["panel"], fg=P["muted"],
                                  font=font(8), anchor="w", justify="left", wraplength=860)
        self.setup_log.pack(fill="x", pady=(8, 0))
        self.adv_inner.pack_forget()

    # ---------------------------------------------------------------- 关于
    def _build_about(self, parent) -> None:
        card = Card(parent, "关于", "")
        card.grid(row=4, column=0, sticky="ew", pady=(14, 12))
        ab = card.body
        info = [
            ("版本", f"{paths.APP_NAME} {paths.VERSION}"),
            ("训练方式", "LoRA / 全参 SFT（transformers + peft）"),
            ("配置文件", str(paths.config_path())),
            ("状态目录", str(paths.state_dir())),
        ]
        for label, value in info:
            row = tk.Frame(ab, bg=P["panel"])
            row.pack(fill="x", pady=(0, 6))
            tk.Label(row, text=label, bg=P["panel"], fg=P["muted"], font=font(9), width=8,
                     anchor="w").pack(side="left")
            tk.Label(row, text=value, bg=P["panel"], fg=P["text2"], font=font(9), anchor="w",
                     justify="left", wraplength=760).pack(side="left", fill="x", expand=True)
        row = tk.Frame(ab, bg=P["panel"])
        row.pack(fill="x", pady=(8, 0))
        RBtn(row, "打开配置文件", self._open_config, kind="ghost", height=30, size=9,
             padx=12).pack(side="left")

    # ---------------------------------------------------------------- 折叠
    def _toggle_advanced(self) -> None:
        self._adv_open = not self._adv_open
        if self._adv_open:
            self.adv_inner.pack(fill="x", pady=(6, 0))
        else:
            self.adv_inner.pack_forget()

    def _toggle_colors(self) -> None:
        self._colors_open = not self._colors_open
        if self._colors_open:
            self.custom_box.pack(fill="x", pady=(14, 0))
        else:
            self.custom_box.pack_forget()

    # ---------------------------------------------------------------- 模型列表
    def _model_options(self) -> list[str]:
        items = (self.cfg.get("ai.catalog") or {}).get("items") or []
        ids = [str(x.get("id")) for x in items if isinstance(x, dict) and x.get("id")]
        if not ids:
            ids = list(ai_mod.SEED_MODELS)
        current = str(self.cfg.get("ai.model", "") or "")
        out = ([current] if current else []) + [ai_mod.TEXT_MODEL] + ids
        seen: list[str] = []
        for m in out:
            if m and m not in seen:
                seen.append(m)
        return seen[:80]

    # ---------------------------------------------------------------- AI 动作
    def _open_keys_page(self) -> None:
        import webbrowser

        try:
            webbrowser.open("https://openrouter.ai/keys")
        except Exception as exc:  # noqa: BLE001
            self.toast(f"打不开浏览器：{exc}", "warn")

    def _save_ai(self) -> bool:
        key = self.var_key.get().strip()
        if key and not key.startswith("sk-"):
            self.toast("Key 看起来不对：OpenRouter 的 Key 以 sk-or-v1- 开头", "warn", 3600)
        model = self.var_ai_model.get().strip()
        label = self.var_asr.get()
        asr = ("whisper" if label == ASR_WHISPER else
               ("sapi" if label == ASR_SAPI else "auto"))
        try:
            rate = int(float(self.var_rate.get() or 0))
        except ValueError:
            rate = 0
        if key:
            note = secrets.set_api_key(key, self.cfg)
            if note:
                if secrets.read_stored_key() or secrets.key_from_env():
                    self.toast(note, "warn", 4600)
                else:
                    self.toast(note, "err", 5200)
                    return False
            else:
                self.toast("Key 已保存到 Windows DPAPI（不写明文配置）", "ok", 3000)
            self.var_key.set("")
        elif not ai_mod.enabled():
            self.toast("先粘贴 OpenRouter Key（也可设置环境变量 FTS_AI_KEY）", "warn")
            return False
        self.cfg.set("ai.key", "", save=False)      # 明确不留明文
        self.cfg.set("ai.model", model, save=False)
        self.cfg.set("ai.asr", asr, save=False)
        self.cfg.set("ai.voice", self.var_voice.get().strip(), save=False)
        self.cfg.set("ai.rate", max(-10, min(10, rate)), save=False)
        err = self.cfg.save()
        if err:
            self.toast(err, "err", 4200)
            return False
        self._refresh_key_note()
        self.app.refresh_ai()
        return True

    def _clear_key(self) -> None:
        if not self.app.confirm("清除本地 Key", "会删除 DPAPI 存储的 Key 和配置里的明文。"
                                                "环境变量里的 Key 需要在系统里手动删除。", "清除", True):
            return
        msg = secrets.clear_api_key(self.cfg)
        self.var_key.set("")
        self._refresh_key_note()
        self.app.refresh_ai()
        self.toast(msg, "ok", 4200)

    def _refresh_key_note(self) -> None:
        source = secrets.key_source(self.cfg)
        text = {
            "env": "Key 来源：环境变量 FTS_AI_KEY / OPENROUTER_API_KEY（优先）",
            "dpapi": "Key 来源：Windows DPAPI（仅当前用户可解密，内存中也不写入配置文件）",
            "legacy-plaintext": "检测到旧版明文 Key，保存或测试时会自动迁移到 DPAPI 并清除配置",
        }.get(source, "尚未配置 Key：可粘贴到上方，或设置环境变量 FTS_AI_KEY")
        self.key_note.configure(text=text + "。若旧 Key 曾以明文落盘，请到服务商后台撤销并轮换。",
                                fg=P["amber"] if source == "legacy-plaintext" else P["muted"])

    def _ai_save_test(self) -> None:
        if not self._save_ai():
            return
        self._ai_test_model()

    def _save_voice(self) -> None:
        label = self.var_asr.get()
        asr = ("whisper" if label == ASR_WHISPER else
               ("sapi" if label == ASR_SAPI else "auto"))
        try:
            rate = int(float(self.var_rate.get() or 0))
        except ValueError:
            rate = 0
        self.cfg.set("ai.asr", asr, save=False)
        self.cfg.set("ai.voice", self.var_voice.get().strip(), save=False)
        self.cfg.set("ai.rate", max(-10, min(10, rate)), save=False)
        err = self.cfg.save()
        self.toast(err or "语音设置已保存", "err" if err else "ok", 2200)

    def _ai_test_model(self) -> None:
        if not ai_mod.enabled():
            self.toast("先粘贴并保存 Key", "warn")
            return
        model = self.var_ai_model.get().strip()
        self.ai_badge.set("测试中…", "amber")
        self.ai_status.configure(text="正在用免费模型发一条真实请求…", fg=P["accent"])

        def work() -> None:
            result = ai_mod.test(model)
            self.app.post(lambda: self._ai_test_done(result))

        threading.Thread(target=work, daemon=True).start()

    def _ai_test_done(self, r: dict) -> None:
        if not r.get("ok"):
            self.ai_badge.set("不可用", "red")
            self.ai_status.configure(text=f"测试失败：{r.get('error', '未知错误')}", fg=P["red"])
            self.toast(str(r.get("error") or "AI 不可用"), "err", 5200)
            self.app.refresh_ai()
            return
        left = r.get("daily_left")
        left_txt = f" · 今日免费额度剩 {left} 次" if isinstance(left, int) else ""
        self.ai_badge.set("可用", "green")
        self.ai_status.configure(
            text=f"{r.get('model', '').split('/')[-1]} · {r.get('seconds', 0):.1f}s{left_txt} · "
                 f"{str(r.get('text') or '')[:120]}", fg=P["green"])
        self.toast("AI 连通正常", "ok", 2200)
        self.app.refresh_ai()

    def _ai_quota(self) -> None:
        if not self._save_ai():
            return

        def work() -> None:
            try:
                info = ai_mod.key_info()
                self.app.post(lambda: self._quota_done(info, ""))
            except Exception as exc:  # noqa: BLE001
                self.app.post(lambda err=str(exc): self._quota_done({}, err))

        threading.Thread(target=work, daemon=True).start()

    def _quota_done(self, info: dict, error: str) -> None:
        if error:
            self.ai_badge.set("Key 校验失败", "red")
            self.ai_status.configure(text=error, fg=P["red"])
            return
        free = info.get("free_remaining")
        used = info.get("free_used")
        limit = info.get("free_limit")
        bits = []
        if free is not None:
            bits.append(f"今日免费请求剩 {free} 次（已用 {used}/{limit}）")
        if info.get("limit_remaining") is not None:
            bits.append(f"余额上限剩余 {info.get('limit_remaining')}")
        self.ai_badge.set("Key 有效", "green")
        self.ai_status.configure(text=" · ".join(bits) or "Key 有效", fg=P["green"])

    def _ai_diagnose(self) -> None:
        if not self._save_ai():
            return
        self.ai_badge.set("检测中…", "amber")
        self.ai_status.configure(text="逐个测免费模型（最多 4 个，约 1 分钟）…", fg=P["accent"])

        current = self.var_ai_model.get().strip()
        models = [current, ai_mod.TEXT_MODEL,
                  "dots-studio/dots-3-note-preview:free",
                  "nvidia/nemotron-3-ultra-550b-a55b:free"]

        def work() -> None:
            try:
                results = ai_mod.diagnose(models=models, timeout=22)
            except Exception as exc:  # noqa: BLE001
                results = [{"model": current, "ok": False, "seconds": 0,
                            "error": f"{type(exc).__name__}: {exc}"}]
            self.app.post(lambda: self._diag_done(results))

        threading.Thread(target=work, daemon=True).start()

    def _diag_done(self, results: list[dict]) -> None:
        ok = [r for r in results if r.get("ok")]
        lines = []
        for r in results:
            name = (r.get("model") or "").split("/")[-1].replace(":free", "") or "?"
            lines.append(f"{'✓' if r.get('ok') else '✗'} {name} {r.get('seconds', 0)}s"
                         + ("" if r.get("ok") else f"：{r.get('error', '')[:60]}"))
        if not ok:
            self.ai_badge.set("全部失败", "red")
            self.ai_status.configure(
                text="；".join(lines[-3:])
                     + "。请到 openrouter.ai/settings/privacy 允许免费模型使用，或明天额度恢复后再试",
                fg=P["red"])
            self.toast("没有可用的免费模型", "err", 5200)
            return
        best = min(ok, key=lambda x: float(x.get("seconds") or 999))
        model = str(best["model"])
        blocked = dict(ai_mod.blocked())
        for r in ok:
            blocked.pop(str(r.get("model")), None)
        fallbacks: list[str] = []
        for r in ok + [{"model": m} for m in ai_mod.SEED_MODELS]:
            m = str(r.get("model") or "")
            if m and m != model and m not in fallbacks:
                fallbacks.append(m)
        self.cfg.set("ai.blocked", blocked, save=False)
        self.cfg.set("ai.model", model, save=False)
        self.cfg.set("ai.fallbacks", fallbacks[:4], save=False)
        self.cfg.save()
        self.var_ai_model.set(model)
        self.cb_ai_model.configure(values=self._model_options())
        self.ai_badge.set("已选最快", "green")
        self.ai_status.configure(text="；".join(lines), fg=P["green"])
        self.toast(f"已选最快的免费模型：{model.split('/')[-1].replace(':free', '')}", "ok", 3200)
        self.app.refresh_ai()

    def _ai_refresh_catalog(self) -> None:
        if not ai_mod.enabled():
            self.toast("先保存 Key", "warn")
            return
        self.ai_status.configure(text="正在拉取免费模型目录…", fg=P["accent"])

        def work() -> None:
            try:
                items = ai_mod.free_models(refresh=True)
                self.app.post(lambda: self._catalog_done(items, ""))
            except Exception as exc:  # noqa: BLE001
                self.app.post(lambda err=str(exc): self._catalog_done([], err))

        threading.Thread(target=work, daemon=True).start()

    def _catalog_done(self, items: list[dict], error: str) -> None:
        if error:
            self.ai_status.configure(text=f"目录刷新失败：{error}", fg=P["red"])
            return
        self.cb_ai_model.configure(values=self._model_options())
        omni = [x["id"] for x in items if "audio" in (x.get("inputs") or [])]
        self.ai_status.configure(
            text=f"免费模型 {len(items)} 个，其中声明支持语音/全模态 {len(omni)} 个"
                 + (f"：{'、'.join(x.split('/')[-1] for x in omni[:6])}" if omni else "")
                 + "（云端音频输入要求余额 ≥ $0.5，语音已改为本机识别）",
            fg=P["green"])
        self.toast(f"已刷新：{len(items)} 个免费模型", "ok", 2600)

    # ---------------------------------------------------------------- 音色
    def _load_voices(self) -> None:
        def work() -> None:
            try:
                names = audio_mod.voices(refresh=True)
                self.app.post(lambda: self._voices_done(names, ""))
            except Exception as exc:  # noqa: BLE001
                self.app.post(lambda err=str(exc): self._voices_done([], err))

        threading.Thread(target=work, daemon=True).start()

    def _voices_done(self, names: list[str], error: str) -> None:
        if error:
            self.clone_note.configure(text=f"取音色失败：{error[:120]}", fg=P["amber"])
            return
        self.cb_voice.configure(values=names)
        cur = self.var_voice.get()
        if names and cur not in names:
            self.var_voice.set(audio_mod.pick_voice())

    def _load_profiles(self) -> None:
        self._profiles = voice.load()
        self.cb_profile.configure(values=self._profile_choices())
        cur = str(self.cfg.get("ai.voice_profile", "") or "")
        prof = next((p for p in self._profiles if p.get("id") == cur), None)
        if prof:
            self.var_profile.set(self._profile_label(prof))
        elif self._profiles:
            self.var_profile.set("（不启用克隆音色）")
        else:
            self.var_profile.set("（还没有克隆音色）")

    def _profile_choices(self) -> list[str]:
        return ["（不启用克隆音色）"] + [self._profile_label(p) for p in self._profiles]

    @staticmethod
    def _profile_label(p: dict) -> str:
        return f"{p.get('name', '音色')} · 音高{p.get('pitch', 0):+d}% 语速{p.get('rate_pct', 0):+d}%"

    def _pick_profile(self) -> None:
        label = self.var_profile.get()
        prof = next((p for p in self._profiles if self._profile_label(p) == label), None)
        self.cfg.set("ai.voice_profile", prof["id"] if prof else "", save=False)
        self.cfg.save()
        if prof:
            self.clone_note.configure(text=f"已启用「{prof['name']}」，回答会用它朗读", fg=P["green"])
        else:
            self.clone_note.configure(text="已关闭克隆音色，使用系统默认音色", fg=P["muted"])

    def _record_clone(self) -> None:
        ok, err = audio_mod.mic_available()
        if not ok:
            self.toast(err or "没有可录音设备", "err", 4200)
            return
        self.clone_note.configure(text="录音 8 秒…请朗读上面的样本", fg=P["accent"])
        name = self.var_clone_name.get().strip() or "我的音色"

        def work() -> None:
            wav, error = audio_mod.record_for(8.0)
            if wav is None:
                self.app.post(lambda: self.clone_note.configure(
                    text=f"录音失败：{error}", fg=P["red"]))
                return
            prof, err2 = voice.clone(name, wav, SAMPLE_TEXT, self.var_voice.get().strip())
            self.app.post(lambda: self._clone_done(prof, err2))

        threading.Thread(target=work, daemon=True).start()

    def _clone_done(self, prof: dict | None, error: str) -> None:
        if prof is None:
            self.clone_note.configure(text=f"克隆失败：{error}", fg=P["red"])
            self.toast(error or "克隆失败", "err", 4200)
            return
        self.cfg.set("ai.voice_profile", prof["id"], save=False)
        self.cfg.save()
        self._load_profiles()
        self.var_profile.set(self._profile_label(prof))
        self.clone_note.configure(
            text=f"已克隆「{prof['name']}」：音高 {prof['pitch']:+d}% · 语速 {prof['rate_pct']:+d}%"
                 f"（样本 {prof['seconds']}s）。回答会自动用它朗读。",
            fg=P["green"])
        self.toast("音色已克隆并启用", "ok", 3200)

    def _preview_profile(self) -> None:
        label = self.var_profile.get()
        prof = next((p for p in self._profiles if self._profile_label(p) == label), None)
        if prof is None:
            self.toast("先选一个克隆音色", "warn")
            return
        self.clone_note.configure(text="正在合成试听…", fg=P["accent"])

        def work() -> None:
            wav = voice.speak("你好，这是克隆音色的试听。", prof)
            if wav is not None:
                audio_mod.play_wav(wav)
            self.app.post(lambda: self.clone_note.configure(text="试听完成", fg=P["muted"]))

        threading.Thread(target=work, daemon=True).start()

    def _delete_profile(self) -> None:
        label = self.var_profile.get()
        prof = next((p for p in self._profiles if self._profile_label(p) == label), None)
        if prof is None:
            self.toast("先选要删除的音色", "warn")
            return
        voice.remove(prof["id"])
        if str(self.cfg.get("ai.voice_profile", "")) == prof["id"]:
            self.cfg.set("ai.voice_profile", "", save=False)
            self.cfg.save()
        self._load_profiles()
        self.clone_note.configure(text=f"已删除「{prof['name']}」", fg=P["muted"])
        self.toast("音色档案已删除", "ok", 2000)

    # ---------------------------------------------------------------- 状态
    def on_show(self) -> None:
        self.var_key.set("")
        self._refresh_key_note()
        self._load_profiles()
        if not getattr(self, "_voices_loaded", False):
            self._voices_loaded = True
            self._load_voices()
        st = ai_mod.stats()
        if ai_mod.enabled() and st:
            last = str(st.get("last_error") or "")[:120]
            if last:
                self.ai_status.configure(text=f"最近一次错误：{ai_mod.friendly_error(last)}",
                                         fg=P["amber"])
        env = self.app.env
        self._kv["root"].configure(text=str(paths.root()))
        if env is None:
            self._kv["python"].configure(text="检测中…", fg=P["muted"])
            self._kv["libs"].configure(text="检测中…", fg=P["muted"])
            return
        self._kv["python"].configure(
            text=f"{env.python}（{env.version}，{env.source}）" if env.python else (env.error or "未找到"),
            fg=P["text2"] if env.python else P["red"])
        libs = env.lib_versions()
        if env.missing:
            self._kv["libs"].configure(text="缺少：" + ", ".join(env.missing), fg=P["amber"])
        else:
            self._kv["libs"].configure(
                text=f"transformers {libs.get('transformers')} · peft {libs.get('peft')} · "
                     f"datasets {libs.get('datasets')} · accelerate {libs.get('accelerate')}",
                fg=P["text2"])
        if self.cfg.load_error:
            self.setup_log.configure(text=f"配置文件解析失败，已使用默认值：{self.cfg.load_error}",
                                     fg=P["amber"])

    def on_env(self, env) -> None:
        if self._kv:
            self.on_show()

    # ---------------------------------------------------------------- 动作
    def _recheck(self) -> None:
        py = self.var_python.get().strip()
        if py and not Path(py).is_file():
            self.toast(f"路径不存在：{py}", "warn")
            return
        self.cfg.set("python", py, save=False)
        self.cfg.save()
        self.app.refresh_env()
        self.toast("正在重新检测环境…", "info", 1800)

    # ------------------------------------------------------------ 外观
    def _preset_card(self, parent, key: str, name: str, desc: str) -> None:
        pal = theme.PRESETS[key][2]
        active = str(self.cfg.get("theme", "forest")) == key
        card = tk.Canvas(parent, width=120, height=68, bg=P["panel"], highlightthickness=0,
                         cursor="hand2")
        card.pack(side="left", padx=(0, 8))
        round_rect(card, 1, 1, 119, 67, 8, fill=pal["bg"],
                   outline=P["accent"] if active else P["border"], width=2 if active else 1)
        round_rect(card, 9, 9, 111, 40, 5, fill=pal["panel"], outline=pal["border"])
        card.create_oval(16, 16, 30, 30, fill=pal["accent"], outline=pal["accent"])
        card.create_text(36, 23, text=name, anchor="w", fill=pal["text"], font=font(9, "bold"))
        card.create_text(12, 55, text=("当前使用" if active else desc[:16]), anchor="w",
                         fill=pal["muted"], font=font(7))
        card.bind("<Button-1>", lambda _e, k=key: self._use_preset(k))
        self._cards[key] = card

    def _use_preset(self, key: str) -> None:
        self.cfg.set("theme_custom", {}, save=False)
        if self.app.apply_theme(key, {}):
            self.app.toast(f"已切换到「{theme.PRESETS[key][0]}」", "ok", 2000)

    def _pick_color(self, key: str) -> None:
        from tkinter import colorchooser

        cur = self.custom.get(key) or P.get(key, "#FFFFFF")
        _rgb, hexv = colorchooser.askcolor(
            color=cur, title=f"选择「{self._color_names.get(key, key)}」颜色")
        if not hexv:
            return
        self.custom[key] = hexv.upper()
        self._paint_swatches()

    def _paint_swatches(self) -> None:
        for key, cv in self._swatches.items():
            color = self.custom.get(key) or P.get(key, "#888888")
            cv.delete("all")
            round_rect(cv, 1, 1, 29, 21, 4, fill=color, outline=P["border2"])
            self._hex_labels[key].configure(
                text=color.upper() + (" *" if key in self.custom else ""),
                fg=P["accent"] if key in self.custom else P["muted"])

    def _apply_custom(self) -> None:
        if not self.custom:
            self.app.toast("还没有改过颜色；直接点上面的色卡就能换预设", "info", 2800)
            return
        name = self.cfg.get("theme", "forest")
        if self.app.apply_theme(name, self.custom):
            self.app.toast("配色已应用并保存", "ok", 2200)

    def _reset_custom(self) -> None:
        self.custom = {}
        if self.app.apply_theme(self.cfg.get("theme", "forest"), {}):
            self.app.toast("已恢复成预设原样", "ok", 2200)

    def _save(self) -> None:
        try:
            threads = int(float(self.var_threads.get()))
            loglines = int(float(self.var_loglines.get()))
        except ValueError:
            self.toast("线程数与日志行数需要是整数", "err")
            return
        self.cfg.set("endpoint", self.var_endpoint.get().strip() or "https://hf-mirror.com", save=False)
        self.cfg.set("threads", max(0, threads), save=False)
        self.cfg.set("log_lines", max(500, loglines), save=False)
        self.cfg.set("auto_check_env", bool(self.var_auto.get()), save=False)
        self.cfg.set("recycle_delete", bool(self.var_recycle.get()), save=False)
        self.cfg.set("python", self.var_python.get().strip(), save=False)
        err = self.cfg.save()
        if err:
            self.toast(err, "err", 4200)
            return
        train = self.app._pages.get("train")  # type: ignore[attr-defined]
        if train is not None and hasattr(train, "var_threads"):
            train.var_threads.set(str(max(1, threads or runtime.default_threads())))
        self.toast("设置已保存", "ok", 2200)
        self._recheck()

    def _reset(self) -> None:
        if not self.app.confirm("恢复默认", "会把 studio_config.json 恢复成出厂默认值，"
                                          "自定义预设与实验记录不受影响。", "恢复", True):
            return
        for k, v in config_mod.DEFAULTS.items():
            self.cfg.set(k, v, save=False)
        self.cfg.save()
        self.var_endpoint.set(str(self.cfg.get("endpoint")))
        self.var_threads.set(str(self.cfg.get("threads")))
        self.var_loglines.set(str(self.cfg.get("log_lines")))
        self.var_python.set(str(self.cfg.get("python")))
        self.var_auto.set(bool(self.cfg.get("auto_check_env")))
        self.var_recycle.set(bool(self.cfg.get("recycle_delete")))
        self.toast("已恢复默认设置", "ok")

    def _restart(self) -> None:
        if not self.app.confirm("重启界面", "将关闭当前窗口并重新启动程序。正在运行的任务会被结束。",
                                "重启", True):
            return
        args = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, *sys.argv]
        try:
            subprocess.Popen(args, cwd=str(paths.root()))
        except OSError as exc:
            self.toast(f"重启失败：{exc}", "err")
            return
        self.app.jobs.kill_all()
        self.app.destroy()

    def _open_config(self) -> None:
        p = paths.config_path()
        if not p.is_file():
            self.cfg.save()
        procs.open_in_explorer(p)

    def _rebuild_env(self) -> None:
        script = paths.root() / "setup_env.ps1"
        if not script.is_file():
            self.toast("没有找到 setup_env.ps1，无法自动重建环境", "err", 4200)
            return
        if self.app.jobs.running(SETUP_KEY):
            self.toast("环境初始化正在进行", "warn")
            return
        if not self.app.confirm("重建训练环境",
                                f"将执行 {script.name}（创建 .venv 并安装 torch/transformers 等依赖）。\n"
                                "会下载数百 MB 的依赖包，耗时几分钟到十几分钟。",
                                "开始重建", True):
            return
        env = procs.base_env(extra={"FTS_ROOT": str(paths.root())})
        args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
        try:
            self.app.jobs.spawn(SETUP_KEY, args, cwd=paths.root(), env=env, label="环境初始化")
        except OSError as exc:
            self.toast(f"启动失败：{exc}", "err")
            return
        self.setup_log.configure(text="环境初始化已启动，进度见底部「日志」面板。", fg=P["text2"])
        self.app.toggle_drawer() if not self.app._drawer_open else None  # type: ignore[attr-defined]

    def _on_setup_event(self, evt: dict) -> None:
        if evt.get("type") == "_exit":
            code = evt.get("code")
            if code == 0:
                self.setup_log.configure(text="环境初始化完成，正在重新检测…", fg=P["green"])
                self.app.refresh_env()
            else:
                self.setup_log.configure(text=f"环境初始化失败（exit {code}），详见日志面板",
                                         fg=P["red"])
