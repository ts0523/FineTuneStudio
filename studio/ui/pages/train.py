# -*- coding: utf-8 -*-
"""训练页：配置、预估、启动、实时曲线与日志。"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ...core import audio, custom_data
from ...core import dataset as ds
from ...core import fsutil
from ...core import models as models_mod
from ...core import paths, planner
from ..base import Page
from ..theme import P, font
from ..widgets import (Bar, Card, Check, Entry, Labeled, LineChart, LogView, RBtn,
                       ScrollFrame)

TRAIN_KEY = "train"


class TrainPage(Page):
    key = "train"
    group = "训练层"
    title = "训练台"
    subtitle = "配置超参 → 预检 → 启动，实时看 loss"

    def build(self) -> None:
        f = tk.Frame(self, bg=P["bg"])
        f.pack(fill="both", expand=True)
        f.grid_columnconfigure(0, weight=3, uniform="t")
        f.grid_columnconfigure(1, weight=5, uniform="t")
        f.grid_rowconfigure(0, weight=1)

        self._build_form(f)

        right = tk.Frame(f, bg=P["bg"])
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        status = Card(right, "运行状态", "")
        status.grid(row=0, column=0, sticky="ew")
        sb = status.body
        top = tk.Frame(sb, bg=P["panel"])
        top.pack(fill="x")
        self.stage_label = tk.Label(top, text="就绪", bg=P["panel"], fg=P["text"],
                                    font=font(11, "bold"), anchor="w")
        self.stage_label.pack(side="left")
        self.run_label = tk.Label(top, text="", bg=P["panel"], fg=P["muted"], font=font(9),
                                  anchor="e")
        self.run_label.pack(side="right")
        self.bar = Bar(sb, height=10, bg=P["panel"])
        self.bar.pack(fill="x", pady=(10, 6))
        info = tk.Frame(sb, bg=P["panel"])
        info.pack(fill="x")
        self.step_label = tk.Label(info, text="步数 -/-", bg=P["panel"], fg=P["text2"],
                                   font=font(9, mono=True), anchor="w")
        self.step_label.pack(side="left")
        self.eta_label = tk.Label(info, text="", bg=P["panel"], fg=P["muted"], font=font(9),
                                  anchor="e")
        self.eta_label.pack(side="right")

        metrics = tk.Frame(sb, bg=P["panel"])
        metrics.pack(fill="x", pady=(12, 0))
        self._metrics: dict[str, tk.Label] = {}
        for i, (k, label) in enumerate([("loss", "当前 loss"), ("best", "最低 loss"),
                                        ("lr", "学习率"), ("tps", "吞吐 tok/s")]):
            box = tk.Frame(metrics, bg=P["panel2"])
            box.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            metrics.grid_columnconfigure(i, weight=1, uniform="mt")
            tk.Label(box, text=label, bg=P["panel2"], fg=P["muted"], font=font(8),
                     anchor="w").pack(anchor="w", padx=9, pady=(6, 0))
            v = tk.Label(box, text="-", bg=P["panel2"], fg=P["text"], font=font(13, "bold"),
                         anchor="w")
            v.pack(anchor="w", padx=9, pady=(0, 6))
            self._metrics[k] = v

        chart_card = Card(right, "损失曲线", "横轴为优化步，悬停可读数")
        chart_card.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.chart = LineChart(chart_card.body, height=190, bg=P["panel"])
        self.chart.pack(fill="both", expand=True)
        self.chart.set_series([], "开始训练后这里会画出 loss")

        log_card = Card(right, "训练日志", "worker 的原始输出")
        log_card.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        self.log = LogView(log_card.body, height=8, max_lines=2000, bg=P["panel"])
        self.log.pack(fill="both", expand=True)

        self._losses: list[tuple[float, float]] = []
        self._evals: list[tuple[float, float]] = []
        self._total_steps = 0
        self._cur_run_id: str | None = None
        self._out_dir: Path | None = None
        self._started = 0.0
        self._resume_from: str = ""
        self._est_after: str | None = None
        self._stats_cache: dict[str, dict] = {}

        self.app.bind_job(TRAIN_KEY, self._on_event)

    # ------------------------------------------------------------ 表单
    def _build_form(self, parent) -> None:
        card = Card(parent, "训练配置", "改动会实时更新右侧预估")
        card.grid(row=0, column=0, sticky="nsew")
        outer = ScrollFrame(card.body, bg=P["panel"])
        outer.pack(fill="both", expand=True)
        b = outer.inner

        self.var_model = tk.StringVar()
        w = Labeled(b, "基座模型", "models/ 下的目录", bg=P["panel"])
        w.pack(fill="x", pady=(0, 8))
        self.cb_model = ttk.Combobox(w.holder, textvariable=self.var_model, state="readonly",
                                     font=font(10))
        self.cb_model.pack(fill="x", ipady=3)
        self.cb_model.bind("<<ComboboxSelected>>", lambda e: self._schedule_est())

        self.var_data = tk.StringVar()
        w = Labeled(b, "训练数据", "data/ 下的 .jsonl", bg=P["panel"])
        w.pack(fill="x", pady=(0, 8))
        self.cb_data = ttk.Combobox(w.holder, textvariable=self.var_data, state="readonly",
                                    font=font(10))
        self.cb_data.pack(fill="x", ipady=3)
        self.cb_data.bind("<<ComboboxSelected>>", lambda e: self._schedule_est())

        quick = tk.Frame(b, bg=P["panel2"], highlightthickness=1,
                         highlightbackground=P["border"])
        quick.pack(fill="x", pady=(8, 0))
        tk.Label(quick, text="实验层快速录入（写入当前训练集）", bg=P["panel2"],
                 fg=P["accent"], font=font(8, "bold"), anchor="w").pack(
            fill="x", padx=8, pady=(6, 3))
        self.quick_q = Entry(quick, placeholder="问题：可直接录音转文字")
        self.quick_q.pack(fill="x", padx=8, ipady=3)
        self.quick_a = Entry(quick, placeholder="标准回答")
        self.quick_a.pack(fill="x", padx=8, pady=(4, 0), ipady=3)
        qrow = tk.Frame(quick, bg=P["panel2"])
        qrow.pack(fill="x", padx=8, pady=(5, 7))
        RBtn(qrow, "录音问→文字", self._quick_voice, kind="secondary", height=26,
             size=8, padx=9).pack(side="left")
        RBtn(qrow, "附图片", self._quick_image, kind="ghost", height=26,
             size=8, padx=9).pack(side="left", padx=5)
        RBtn(qrow, "录音附件", self._quick_audio, kind="ghost", height=26,
             size=8, padx=9).pack(side="left")
        RBtn(qrow, "加入", self._quick_add, kind="primary", height=26,
             size=8, padx=12).pack(side="right")
        self.quick_label = tk.Label(quick, text="", bg=P["panel2"], fg=P["muted"],
                                     font=font(8), anchor="w")
        self.quick_label.pack(fill="x", padx=8, pady=(0, 6))
        self._quick_image_path = ""
        self._quick_audio_path = ""

        self.var_name = tk.StringVar(value="experiment")
        w = Labeled(b, "实验名", "用于产物目录与实验记录", bg=P["panel"])
        w.pack(fill="x", pady=(0, 8))
        e = Entry(w.holder)
        e.configure(textvariable=self.var_name)
        e.pack(fill="x", ipady=3)

        prow = tk.Frame(b, bg=P["panel"])
        prow.pack(fill="x", pady=(0, 8))
        prow.grid_columnconfigure(0, weight=2)
        prow.grid_columnconfigure(1, weight=1)
        wp = Labeled(prow, "预设", "", bg=P["panel"])
        wp.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.var_preset = tk.StringVar()
        self.cb_preset = ttk.Combobox(wp.holder, textvariable=self.var_preset, state="readonly",
                                      font=font(9))
        self.cb_preset.pack(fill="x", ipady=2)
        self.cb_preset.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())
        wq = Labeled(prow, "操作", "", bg=P["panel"])
        wq.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        RBtn(wq.holder, "存为预设", self._save_preset, kind="secondary", height=28, size=9,
             padx=10).pack(fill="x")

        grid = tk.Frame(b, bg=P["panel"])
        grid.pack(fill="x")
        for c in range(2):
            grid.grid_columnconfigure(c, weight=1, uniform="f")

        self.var_epochs = tk.StringVar(value="3")
        self.var_batch = tk.StringVar(value="2")
        self.var_accum = tk.StringVar(value="2")
        self.var_lr = tk.StringVar(value="0.0002")
        self.var_maxlen = tk.StringVar(value="256")
        self.var_val = tk.StringVar(value="0.1")
        self.var_r = tk.StringVar(value="16")
        self.var_alpha = tk.StringVar(value="32")
        self.var_dropout = tk.StringVar(value="0.05")
        self.var_seed = tk.StringVar(value="42")
        self.var_threads = tk.StringVar(value=str(int(self.cfg.get("threads", 0) or 4)))
        self.var_full = tk.BooleanVar(value=False)
        self.var_resume = tk.BooleanVar(value=False)

        fields = [
            ("轮数 epochs", self.var_epochs, 1, 50, 1),
            ("批大小 batch", self.var_batch, 1, 16, 1),
            ("梯度累积 accum", self.var_accum, 1, 32, 1),
            ("序列长度 max_len", self.var_maxlen, 64, 2048, 64),
            ("验证集比例", self.var_val, 0.0, 0.5, 0.05),
            ("随机种子", self.var_seed, 0, 9999, 1),
            ("LoRA r", self.var_r, 1, 128, 1),
            ("LoRA alpha", self.var_alpha, 1, 256, 1),
            ("LoRA dropout", self.var_dropout, 0.0, 0.5, 0.01),
            ("CPU 线程", self.var_threads, 1, 32, 1),
        ]
        self._spin_rows: dict[str, ttk.Spinbox] = {}
        for i, (label, var, lo, hi, inc) in enumerate(fields):
            wl = Labeled(grid, label, "", bg=P["panel"])
            wl.grid(row=i // 2, column=i % 2, sticky="ew",
                    padx=(0 if i % 2 == 0 else 6, 0), pady=(0, 6))
            sp = ttk.Spinbox(wl.holder, from_=lo, to=hi, increment=inc, textvariable=var,
                             font=font(9))
            sp.pack(fill="x", ipady=3)
            var.trace_add("write", lambda *a: self._schedule_est())
            self._spin_rows[label] = sp

        wl = Labeled(grid, "学习率 lr", "LoRA 常用 1e-4 ~ 3e-4", bg=P["panel"])
        wl.grid(row=len(fields) // 2, column=0, sticky="ew", padx=(0, 6))
        e = Entry(wl.holder)
        e.configure(textvariable=self.var_lr)
        e.pack(fill="x", ipady=3)
        self.var_lr.trace_add("write", lambda *a: self._schedule_est())

        opts = tk.Frame(b, bg=P["panel"])
        opts.pack(fill="x", pady=(8, 0))
        Check(opts, "全参微调（不加 LoRA）", self.var_full, bg=P["panel"],
              command=self._toggle_full).pack(side="left")
        Check(opts, "从检查点续训", self.var_resume, bg=P["panel"]).pack(side="left", padx=(16, 0))

        self.est_label = tk.Label(b, text="预估：待计算", bg=P["panel"], fg=P["text2"],
                                  font=font(9), anchor="w", justify="left", wraplength=380)
        self.est_label.pack(fill="x", pady=(12, 0))
        self.out_label = tk.Label(b, text="输出目录：自动生成", bg=P["panel"], fg=P["muted"],
                                  font=font(8), anchor="w", justify="left", wraplength=380)
        self.out_label.pack(fill="x", pady=(4, 0))

        act = tk.Frame(b, bg=P["panel"])
        act.pack(fill="x", pady=(14, 0))
        self.btn_start = RBtn(act, "开始训练", self.start, kind="primary", height=36, padx=18)
        self.btn_start.pack(fill="x")
        act2 = tk.Frame(b, bg=P["panel"])
        act2.pack(fill="x", pady=(8, 0))
        self.btn_pre = RBtn(act2, "预检", self.preflight, kind="secondary", height=30, size=9,
                            padx=12)
        self.btn_pre.pack(side="left", expand=True, fill="x")
        self.btn_stop = RBtn(act2, "停止（存检查点）", self.stop, kind="secondary", height=30,
                             size=9, padx=10)
        self.btn_stop.pack(side="left", expand=True, fill="x", padx=6)
        self.btn_kill = RBtn(act2, "强杀", self.kill, kind="danger", height=30, size=9, padx=10)
        self.btn_kill.pack(side="left", expand=True, fill="x")

    # ------------------------------------------------------------ 数据
    def on_show(self) -> None:
        self._reload_lists()
        pending = self.app.shared.pop("pending_train", None)
        if pending:
            self._apply_pending(pending)
        self._refresh_presets()
        self._schedule_est()

    def _reload_lists(self) -> None:
        models = models_mod.scan_models()
        names = [m.name for m in models]
        self.cb_model.configure(values=names)
        if names and self.var_model.get() not in names:
            self.var_model.set(models_mod.default_model_name(models, self.cfg.get("last.model", "")))
        datasets = ds.list_datasets()
        dnames = [d["name"] for d in datasets]
        self.cb_data.configure(values=dnames)
        if dnames and self.var_data.get() not in dnames:
            self.var_data.set(ds.default_dataset_name(datasets, self.cfg.get("last.dataset", "")))

    def _quick_voice(self) -> None:
        ok, err = audio.mic_available()
        if not ok:
            self.toast(err or "没有可录音设备", "err", 4200)
            return
        self.quick_label.configure(text="录音 4 秒…请说问题", fg=P["accent"])

        def work() -> None:
            wav, error = audio.record_for(4.0)
            if wav is None:
                self.app.post(lambda: self.quick_label.configure(
                    text=error or "没有录到声音", fg=P["amber"]))
                return
            text, engine, err2 = audio.transcribe(wav)
            self.app.post(lambda: self._quick_voice_done(text, engine, err2))

        threading.Thread(target=work, daemon=True).start()

    def _quick_voice_done(self, text: str, engine: str, error: str) -> None:
        if not text:
            self.quick_label.configure(text=f"没听清：{error}", fg=P["amber"])
            return
        self.quick_q.set_value(text)
        self.quick_label.configure(text=f"问题已转文字（{engine}），补回答后点「加入」",
                                   fg=P["green"])

    def _quick_image(self) -> None:
        from tkinter import filedialog

        p = filedialog.askopenfilename(
            title="选择图片", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if p:
            self._quick_image_path = p
            self._quick_audio_path = ""
            self.quick_label.configure(text=f"已附加图片：{Path(p).name}", fg=P["green"])

    def _quick_audio(self) -> None:
        ok, err = audio.mic_available()
        if not ok:
            self.toast(err or "没有可录音设备", "err", 4200)
            return
        self.quick_label.configure(text="录音 4 秒…请说话", fg=P["accent"])

        def work() -> None:
            wav, error = audio.record_for(4.0)
            self.app.post(lambda: self._quick_audio_done(wav, error))

        threading.Thread(target=work, daemon=True).start()

    def _quick_audio_done(self, wav: Path | None, error: str) -> None:
        if wav is None:
            self.quick_label.configure(text=f"录音失败：{error}", fg=P["amber"])
            return
        self._quick_audio_path = str(wav)
        self._quick_image_path = ""
        self.quick_label.configure(text=f"已附加语音：{wav.name}", fg=P["green"])

    def _quick_add(self) -> None:
        name = self.var_data.get()
        if not name:
            self.toast("先在数据页新建自建问答数据集", "warn", 4200)
            return
        path = paths.data_dir() / name
        q, a = self.quick_q.value(), self.quick_a.value()
        try:
            n = custom_data.append_pair(path, q, a, image=self._quick_image_path or None,
                                        audio=self._quick_audio_path or None)
        except (custom_data.CustomDataError, fsutil.PathSafetyError, OSError) as exc:
            self.toast(f"写入失败：{exc}", "err", 4400)
            return
        self._quick_image_path = ""
        self._quick_audio_path = ""
        self.quick_q.set_value("")
        self.quick_a.set_value("")
        self.quick_label.configure(text=f"已写入 {path.name}，当前 {n} 条", fg=P["green"])
        self._stats_cache.clear()
        self._reload_lists()
        self._schedule_est()
        self.toast(f"已加入训练集（{n} 条）", "ok", 2800)

    def _refresh_presets(self) -> None:
        names = sorted(self.store.presets().keys())
        self.cb_preset.configure(values=names)
        if self.var_preset.get() not in names:
            self.var_preset.set("")

    def _apply_pending(self, pending: dict) -> None:
        if pending.get("model"):
            self.var_model.set(pending["model"])
        if pending.get("dataset"):
            self.var_data.set(pending["dataset"])
        plan = pending.get("plan") or {}
        for key, var in (("epochs", self.var_epochs), ("batch_size", self.var_batch),
                         ("grad_accum", self.var_accum), ("max_len", self.var_maxlen),
                         ("lora_r", self.var_r), ("lora_alpha", self.var_alpha),
                         ("lora_dropout", self.var_dropout), ("val_ratio", self.var_val)):
            if key in plan:
                var.set(f"{plan[key]:g}" if isinstance(plan[key], float) else str(plan[key]))
        if "lr" in plan:
            self.var_lr.set(f"{plan['lr']:g}")
        if plan.get("full"):
            self.var_full.set(True)
            self._toggle_full()
        if pending.get("goal"):
            self.app.shared["goal_text"] = pending["goal"]
        self.log.append(f"已载入目标方案：{pending.get('goal', '')[:60]}", "info")
        self._schedule_est()

    def _toggle_full(self) -> None:
        full = bool(self.var_full.get())
        for name in ("LoRA r", "LoRA alpha", "LoRA dropout"):
            self._spin_rows[name].configure(state="disabled" if full else "normal")

    # ------------------------------------------------------------ 预估
    def _schedule_est(self) -> None:
        if self._est_after:
            try:
                self.after_cancel(self._est_after)
            except tk.TclError:
                pass
        self._est_after = self.after(300, self._update_est)

    def _data_stats(self, name: str) -> dict:
        if not name:
            return {}
        p = paths.data_dir() / name
        key = f"{p}"
        if key in self._stats_cache:
            return self._stats_cache[key]
        st = ds.stats(p)
        self._stats_cache[key] = st
        return st

    def _form(self) -> dict | None:
        try:
            return {
                "model": self.var_model.get(),
                "data": self.var_data.get(),
                "name": (self.var_name.get() or "experiment").strip(),
                "epochs": float(self.var_epochs.get()),
                "batch": int(float(self.var_batch.get())),
                "accum": int(float(self.var_accum.get())),
                "lr": float(self.var_lr.get()),
                "max_len": int(float(self.var_maxlen.get())),
                "val": float(self.var_val.get()),
                "r": int(float(self.var_r.get())),
                "alpha": int(float(self.var_alpha.get())),
                "dropout": float(self.var_dropout.get()),
                "seed": int(float(self.var_seed.get())),
                "threads": int(float(self.var_threads.get())),
                "full": bool(self.var_full.get()),
                "resume": bool(self.var_resume.get()),
            }
        except (ValueError, tk.TclError):
            return None

    def _update_est(self) -> None:
        self._est_after = None
        form = self._form()
        if form is None:
            self.est_label.configure(text="参数格式有误：检查数字字段", fg=P["red"])
            return
        st = self._data_stats(form["data"])
        model = models_mod.find_model(form["model"]) if form["model"] else None
        if not st or not st.get("valid") or model is None:
            self.est_label.configure(text="选择模型与数据后显示预估", fg=P["muted"])
            return
        avg = planner.avg_tokens_from_stats(st)
        est = planner.estimate(max(1.0, model.params_m or 135), st["valid"], avg, form["epochs"],
                               form["batch"], form["accum"], form["max_len"],
                               "full" if form["full"] else "lora",
                               model.hidden or 576, model.layers or 30,
                               self.store.speed_samples(), form["threads"])
        lo, hi = est["est_seconds"] * 0.75, est["est_seconds"] * 1.3
        self.est_label.configure(
            text=f"预估：{est['steps']} 步 · {paths.human_time(est['est_seconds'])}"
                 f"（{paths.human_time(lo)} ~ {paths.human_time(hi)}）· "
                 f"内存 {est['ram_gb']:.1f} GB · 吞吐约 {est['tps']:.0f} tok/s", fg=P["text2"])

    # ------------------------------------------------------------ 预检
    def preflight(self) -> bool:
        form = self._form()
        if form is None:
            self.toast("参数格式有误", "err")
            return False
        problems: list[str] = []
        model = models_mod.find_model(form["model"]) if form["model"] else None
        if model is None:
            problems.append("模型不存在")
        elif not model.complete:
            problems.append(f"模型缺文件：{', '.join(model.missing)}")
        dp = paths.data_dir() / form["data"] if form["data"] else None
        if not dp or not dp.is_file():
            problems.append("数据文件不存在")
        else:
            rep = ds.validate(dp)
            if not rep.ok:
                problems.append(f"数据有 {len(rep.issues)} 处问题（{rep.valid} 条可用）")
        env = self.app.env
        if env is None or env.python is None:
            problems.append("没有可用的训练环境（.venv）")
        elif env.missing:
            problems.append("缺少依赖：" + ", ".join(env.missing))
        if form["max_len"] < 64:
            problems.append("max_len 太小（<64），建议 128 以上")
        if form["batch"] * form["accum"] > 64:
            problems.append("等效 batch 过大，CPU 上会很慢")
        if not (1e-6 <= form["lr"] <= 1e-2):
            problems.append(f"学习率 {form['lr']:g} 超出常用范围（1e-6 ~ 1e-2），会训崩或学不动")
        if form["epochs"] <= 0 or form["batch"] <= 0 or form["accum"] <= 0:
            problems.append("轮数 / 批大小 / 梯度累积必须大于 0")
        if form["val"] > 0 and (st := self._data_stats(form["data"])) and st.get("valid", 0) < 10:
            problems.append("样本少于 10 条，自动跳过验证集（你会看不到 eval_loss）")

        self.log.clear()
        for p in problems:
            self.log.append(f"[预检] {p}", "warn")
        if not problems:
            self.log.append("[预检] 全部通过，可以开始训练", "ok")
            self.toast("预检通过", "ok")
            return True
        self.toast(f"预检发现 {len(problems)} 个问题，见右侧日志", "warn", 4200)
        return False

    # ------------------------------------------------------------ 启动
    def apply_smoke(self, name: str = "smoke") -> None:
        self.var_epochs.set("1")
        self.var_batch.set("1")
        self.var_accum.set("1")
        self.var_maxlen.set("192")
        self.var_val.set("0")
        self.var_name.set(name)
        self.app.shared["max_steps"] = 2
        self.log.append("已切到冒烟配置：1 轮 / batch 1 / 192 token / 只跑 2 步", "info")
        self.start()

    def start(self) -> None:
        if self.app.jobs.running(TRAIN_KEY):
            self.toast("已经有训练在跑", "warn")
            return
        form = self._form()
        if form is None:
            self.toast("参数格式有误", "err")
            return
        if not self.preflight():
            return
        stamp = time.strftime("%m%d-%H%M%S")
        name = fsutil.sanitize_component(form["name"], "experiment")
        form["name"] = name
        self._resume_from = ""
        try:
            if form["resume"]:
                out_dir = fsutil.safe_child(paths.outputs_dir(), name)
                out_dir.mkdir(parents=True, exist_ok=True)
                cps: list[tuple[int, Path]] = []
                for p in out_dir.glob("checkpoint-*"):
                    try:
                        cps.append((int(p.name.split("-")[-1]), p))
                    except ValueError:
                        continue
                if cps:
                    cps.sort()
                    self._resume_from = str(cps[-1][1])
                    self.log.append(f"从检查点续训：{cps[-1][1].name}（第 {cps[-1][0]} 步）", "info")
                elif (out_dir / "adapter_config.json").is_file():
                    self._resume_from = str(out_dir)
                    self.log.append("没有 checkpoint 目录，改为在已有适配器上继续训练", "warn")
                else:
                    self.toast(f"「{name}」下没有可续训的检查点，本次从头训练", "warn", 3600)
            else:
                out_dir = fsutil.safe_child(paths.outputs_dir(), f"{name}-{stamp}")
        except fsutil.PathSafetyError as exc:
            self.toast(f"实验名不安全：{exc}", "err", 4600)
            return
        self._out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_label.configure(text=f"输出目录：{out_dir}")

        stop_flag = out_dir / "studio_stop.flag"
        fsutil.prepare_stop_flag(stop_flag)      # 清掉上次续训遗留的停止标记
        cfg = {
            "model": str((paths.models_dir() / form["model"]).resolve()),
            "data": str((paths.data_dir() / form["data"]).resolve()),
            "output": str(out_dir),
            "events_file": str(out_dir / "studio_events.jsonl"),
            "stop_flag": str(stop_flag),
            "epochs": form["epochs"], "batch_size": form["batch"], "grad_accum": form["accum"],
            "lr": form["lr"], "max_len": form["max_len"], "val_ratio": form["val"],
            "lora_r": form["r"], "lora_alpha": form["alpha"], "lora_dropout": form["dropout"],
            "seed": form["seed"], "threads": form["threads"], "full": form["full"],
            "max_steps": int(self.app.shared.pop("max_steps", -1)),
            "goal": str(self.app.shared.get("goal_text", "")),
            "resume": self._resume_from,
        }
        run_id = self.store.new_id(name)
        self._cur_run_id = run_id
        self.store.add_run({
            "id": run_id, "name": name, "status": "running",
            "model": form["model"], "dataset": form["data"], "samples": 0,
            "mode": "full" if form["full"] else "lora", "epochs": form["epochs"],
            "out_dir": str(out_dir), "config": cfg, "goal": cfg["goal"],
        })
        self._losses.clear()
        self._evals.clear()
        self._total_steps = 0
        self.chart.set_series([], "训练中…")
        self.log.clear()
        self.log.append(f"启动训练：{form['model']} ← {form['data']}", "cmd")
        self.bar.set(0)
        self.stage_label.configure(text="正在启动 worker…", fg=P["text"])
        self.run_label.configure(text=out_dir.name)
        for v in self._metrics.values():
            v.configure(text="-")
        if not self.app.spawn_worker(TRAIN_KEY, "train_worker.py", cfg, label="训练",
                                     stop_file=stop_flag):
            self.stage_label.configure(text="启动失败", fg=P["red"])
            self.store.update_run(run_id, status="failed", error="worker 启动失败")
            return
        self._started = time.time()
        self.btn_start.set_enabled(False)
        self.app.set_status(f"训练中：{name}")
        self.app.set_job_status("训练中", True)

    def stop(self) -> None:
        job = self.app.jobs.get(TRAIN_KEY)
        if job and job.running:
            job.stop()
            self.stage_label.configure(text="已请求停止：当前步结束后保存检查点", fg=P["amber"])
            self.log.append("已请求优雅停止（会保存检查点，可续训）", "warn")
        else:
            self.toast("没有在跑的训练", "warn")

    def kill(self) -> None:
        job = self.app.jobs.get(TRAIN_KEY)
        if job and job.running:
            if not self.app.confirm("强制结束", "强杀会立即结束 python 进程，当前步的进度丢失"
                                              "（已保存的检查点仍在）。", "强杀", True):
                return
            job.kill()
            self.stage_label.configure(text="已强杀", fg=P["red"])
            self.log.append("已强制结束训练进程", "err")

    def _save_preset(self) -> None:
        form = self._form()
        if form is None:
            return
        name = self.app.ask_text("保存预设", "预设名称", form["name"])
        if not name:
            return
        self.store.set_preset(name, {k: form[k] for k in ("epochs", "batch", "accum", "lr",
                                                         "max_len", "val", "r", "alpha",
                                                         "dropout", "seed", "threads", "full")})
        self._refresh_presets()
        self.toast(f"预设「{name}」已保存", "ok")

    def _apply_preset(self) -> None:
        name = self.var_preset.get()
        p = self.store.presets().get(name)
        if not p:
            return
        mapping = {"epochs": self.var_epochs, "batch": self.var_batch, "accum": self.var_accum,
                   "lr": self.var_lr, "max_len": self.var_maxlen, "val": self.var_val,
                   "r": self.var_r, "alpha": self.var_alpha, "dropout": self.var_dropout,
                   "seed": self.var_seed, "threads": self.var_threads}
        for k, var in mapping.items():
            if k in p:
                var.set(f"{p[k]:g}" if isinstance(p[k], float) else str(p[k]))
        if "full" in p:
            self.var_full.set(bool(p["full"]))
            self._toggle_full()
        self.toast(f"已应用预设 {name}", "ok", 1600)

    # ------------------------------------------------------------ 事件
    def _on_event(self, evt: dict) -> None:
        t = evt.get("type")
        if t == "stage":
            msg = str(evt.get("message", ""))
            if evt.get("name") == "tokenize" and evt.get("total"):
                frac = evt.get("done", 0) / max(1, evt["total"])
                self.stage_label.configure(text=f"分词 {evt.get('done')}/{evt['total']}")
                self.bar.set(frac * 0.25)
                return
            self.stage_label.configure(text=msg or "…")
            self.log.append(f"[阶段] {msg}", "info")
        elif t == "data":
            self.log.append(f"[数据] 可用 {evt.get('samples')} 条（坏行 {evt.get('bad')}，"
                            f"跳过 {evt.get('skipped')}），平均 {evt.get('avg_len')} token", "ok")
            if self._cur_run_id:
                self.store.update_run(self._cur_run_id, samples=evt.get("samples", 0),
                                      avg_len=evt.get("avg_len"))
        elif t == "split":
            self.log.append(f"[数据] 训练 {evt.get('train')} / 验证 {evt.get('val')}", "info")
        elif t == "model":
            self.log.append(f"[模型] {evt.get('params', 0)/1e6:.1f}M 参数，可训练 "
                            f"{evt.get('trainable', 0)/1e6:.2f}M", "ok")
        elif t == "plan":
            self._total_steps = int(evt.get("total_steps") or 0)
            self.log.append(f"[计划] 共 {self._total_steps} 步，每 {evt.get('save_steps')} 步存检查点", "info")
            self.step_label.configure(text=f"步数 0/{self._total_steps}")
        elif t == "train_begin":
            self.bar.set(0)
            self.stage_label.configure(text="训练中", fg=P["accent"])
        elif t == "step":
            self._on_step(evt)
        elif t == "eval":
            loss = evt.get("eval_loss")
            if isinstance(loss, (int, float)):
                self._evals.append((float(evt.get("step", 0)), float(loss)))
                self.log.append(f"[验证] step {evt.get('step')} eval_loss={loss:.4f} "
                                f"ppl={evt.get('perplexity')}", "info")
                self._redraw_chart()
        elif t == "stopping":
            self.stage_label.configure(text="正在保存检查点…", fg=P["amber"])
            self.log.append(f"[停止] {evt.get('message')}", "warn")
        elif t == "saved":
            self.log.append(f"[产物] {evt.get('output')}", "ok")
        elif t == "done":
            self._finish(evt.get("status", "done"), evt)
        elif t == "error":
            self.stage_label.configure(text="训练失败", fg=P["red"])
            self.log.append(f"[错误] {evt.get('message')}", "err")
            if self._cur_run_id:
                self.store.update_run(self._cur_run_id, status="failed",
                                      error=str(evt.get("message", ""))[:500])
            self.btn_start.set_enabled(False)     # 等旧进程真正退出后再允许新训练
        elif t == "_exit":
            code = evt.get("code")
            if code not in (0, None) and self._cur_run_id:
                run = self.store.get_run(self._cur_run_id) or {}
                if run.get("status") == "running":
                    self.store.update_run(self._cur_run_id, status="failed",
                                          error=f"进程退出 code={evt.get('code')}")
            self._reset_buttons()
            self.app.set_job_status("空闲", False)

    def _on_step(self, evt: dict) -> None:
        step = int(evt.get("step") or 0)
        total = int(evt.get("total") or self._total_steps or 1)
        loss = evt.get("loss")
        if loss is None:
            return
        self._losses.append((float(step), float(loss)))
        if len(self._losses) > 4000:
            self._losses = self._losses[::2]
        self.bar.set(step / max(1, total))
        self.step_label.configure(text=f"步数 {step}/{total}")
        eta = evt.get("eta")
        self.eta_label.configure(
            text=f"已用 {paths.human_time(evt.get('elapsed'))} · 剩余 {paths.human_time(eta)}")
        self._metrics["loss"].configure(text=f"{loss:.4f}")
        best = min(x[1] for x in self._losses)
        self._metrics["best"].configure(text=f"{best:.4f}")
        if evt.get("lr") is not None:
            self._metrics["lr"].configure(text=f"{evt['lr']:.2e}")
        if evt.get("tps"):
            self._metrics["tps"].configure(text=f"{evt['tps']:.0f}")
        self._redraw_chart()

    def _redraw_chart(self) -> None:
        series = []
        if self._losses:
            series.append(("train", "accent", self._losses))
        if self._evals:
            series.append(("eval", "amber", self._evals))
        self.chart.set_series(series, "等待第一个 step…")

    def _finish(self, status: str, evt: dict) -> None:
        metrics = evt.get("metrics") or {}
        secs = evt.get("total_seconds") or (time.time() - self._started)
        self.stage_label.configure(
            text="训练完成" if status == "done" else "训练已停止（检查点已保存）",
            fg=P["green"] if status == "done" else P["amber"])
        self.bar.set(1)
        if self._cur_run_id:
            self.store.update_run(
                self._cur_run_id, status=status, seconds=secs,
                final_loss=metrics.get("final_loss"), best_eval_loss=metrics.get("best_eval_loss"),
                tokens=metrics.get("tokens"), tokens_per_sec=metrics.get("tokens_per_sec"),
                steps_done=metrics.get("steps_done"), out_dir=evt.get("output"),
                phase="完成" if status == "done" else "已停止")
        if metrics.get("tokens_per_sec"):
            model = models_mod.find_model(self.var_model.get())
            self.store.record_speed(model.params_m if model else 135, int(self.var_maxlen.get()),
                                    float(metrics["tokens_per_sec"]))
        self.log.append(f"[完成] 用时 {paths.human_time(secs)}，最终 loss "
                        f"{metrics.get('final_loss')}", "ok")
        self.toast("训练完成，已记入实验列表" if status == "done" else "训练已停止，检查点已保存",
                   "ok" if status == "done" else "warn", 3600)
        # 进程可能还在退出/落盘：等 _exit 回调再恢复「开始训练」
        self.btn_start.set_enabled(False)
        self.app.set_status("等待训练进程退出…")

    def _reset_buttons(self) -> None:
        self.btn_start.set_enabled(True)
        self.app.set_status("就绪")
