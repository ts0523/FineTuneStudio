# -*- coding: utf-8 -*-
"""目标页：把「我想让模型学会什么」翻译成可执行方案。"""
from __future__ import annotations

import json
import time
import tkinter as tk
from tkinter import ttk

from ...core import dataset as ds
from ...core import models as models_mod
from ...core import paths, planner
from ..base import Page
from ..theme import P, font
from ..widgets import Card, Labeled, RBtn, ScrollFrame, TextArea

QUALITY_CHOICES = [
    ("smoke", "冒烟验证", "1 轮 · 192 token · r=8\n约 1~3 分钟"),
    ("usable", "可用", "3 轮 · 256 token · r=16\n约 5~30 分钟"),
    ("serious", "认真", "5 轮 · 384 token · r=32\n数十分钟到数小时"),
]


class GoalPage(Page):
    key = "goal"
    group = "数据层"
    title = "目标规划"
    subtitle = "写下目标，得到一套能直接跑的超参与时长预估"

    def build(self) -> None:
        outer = ScrollFrame(self)
        outer.pack(fill="both", expand=True)
        f = outer.inner
        f.grid_columnconfigure(0, weight=3, uniform="g")
        f.grid_columnconfigure(1, weight=2, uniform="g")

        left = Card(f, "1 · 说清目标", "目标描述会写进 train_meta.json，方便日后对照效果")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        b = left.body
        self.goal_text = TextArea(b, height=3)
        self.goal_text.pack(fill="x")
        self.goal_text.insert("1.0", "让模型学会回答我们产品的常见问题，语气简洁，只讲事实。")

        grid = tk.Frame(b, bg=P["panel"])
        grid.pack(fill="x", pady=(14, 0))
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)

        self.var_task = tk.StringVar(value="客服问答")
        w1 = Labeled(grid, "任务类型", "影响学习率与序列长度", bg=P["panel"])
        w1.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        cb_task = ttk.Combobox(w1.holder, textvariable=self.var_task, state="readonly",
                               values=list(planner.TASK_PRESETS.keys()), font=font(10))
        cb_task.pack(fill="x", ipady=3)
        cb_task.bind("<<ComboboxSelected>>", lambda e: self._schedule_plan())

        self.var_samples = tk.StringVar(value="24")
        w2 = Labeled(grid, "演示数据条数", "没有真实数据时先跑通流程", bg=P["panel"])
        w2.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        row = tk.Frame(w2.holder, bg=P["panel"])
        row.pack(fill="x")
        ttk.Spinbox(row, from_=4, to=2000, increment=4, textvariable=self.var_samples,
                    width=8, font=font(10)).pack(side="left", ipady=3)
        RBtn(row, "生成演示数据", self._make_seed, kind="secondary", height=30, size=9,
             padx=10).pack(side="left", padx=8)

        self.var_quality = tk.StringVar(value="usable")
        tk.Label(b, text="质量档", bg=P["panel"], fg=P["text2"], font=font(9)).pack(
            anchor="w", pady=(16, 6))
        tiles = tk.Frame(b, bg=P["panel"])
        tiles.pack(fill="x")
        self._tiles: dict[str, tk.Frame] = {}
        for i, (key, name, desc) in enumerate(QUALITY_CHOICES):
            t = tk.Frame(tiles, bg=P["panel2"], highlightthickness=1,
                         highlightbackground=P["border"], cursor="hand2")
            t.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            tiles.grid_columnconfigure(i, weight=1, uniform="t")
            tk.Label(t, text=name, bg=P["panel2"], fg=P["text"], font=font(10, "bold"),
                     anchor="w").pack(anchor="w", padx=10, pady=(8, 0))
            tk.Label(t, text=desc, bg=P["panel2"], fg=P["muted"], font=font(8), anchor="w",
                     justify="left").pack(anchor="w", padx=10, pady=(2, 9))
            for w in [t, *t.winfo_children()]:
                w.bind("<Button-1>", lambda e, k=key: self._pick_quality(k))
            self._tiles[key] = t

        grid2 = tk.Frame(b, bg=P["panel"])
        grid2.pack(fill="x", pady=(16, 0))
        grid2.grid_columnconfigure(0, weight=1)
        grid2.grid_columnconfigure(1, weight=1)

        self.var_model = tk.StringVar()
        wm = Labeled(grid2, "基座模型", "来自 models 目录", bg=P["panel"])
        wm.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.cb_model = ttk.Combobox(wm.holder, textvariable=self.var_model, state="readonly",
                                     font=font(10))
        self.cb_model.pack(fill="x", ipady=3)
        self.cb_model.bind("<<ComboboxSelected>>", lambda e: self._schedule_plan())

        self.var_data = tk.StringVar()
        wd = Labeled(grid2, "数据集", "data/*.jsonl", bg=P["panel"])
        wd.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.cb_data = ttk.Combobox(wd.holder, textvariable=self.var_data, state="readonly",
                                    font=font(10))
        self.cb_data.pack(fill="x", ipady=3)
        self.cb_data.bind("<<ComboboxSelected>>", lambda e: self._schedule_plan())

        self.data_hint = tk.Label(b, text="", bg=P["panel"], fg=P["muted"], font=font(8),
                                  anchor="w", justify="left", wraplength=520)
        self.data_hint.pack(anchor="w", pady=(8, 0))

        bar = tk.Frame(b, bg=P["panel"])
        bar.pack(fill="x", pady=(16, 0))
        RBtn(bar, "生成方案", self._compute, kind="primary", height=34, padx=18).pack(side="left")
        RBtn(bar, "刷新清单", self._reload_sources, kind="ghost", height=34, padx=14).pack(
            side="left", padx=8)

        right = Card(f, "2 · 推荐方案", "全部由数据量、模型规模与本机实测速度推算")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        rb = right.body
        self.summary = tk.Label(rb, text="选好条件后点「生成方案」", bg=P["panel"], fg=P["text2"],
                                font=font(10), anchor="w", justify="left", wraplength=380)
        self.summary.pack(fill="x")

        stats = tk.Frame(rb, bg=P["panel"])
        stats.pack(fill="x", pady=(14, 0))
        stats.grid_columnconfigure((0, 1), weight=1)
        self._kv: dict[str, tk.Label] = {}
        fields = [("steps", "训练步数"), ("time", "预计耗时"), ("ram", "预计内存"),
                  ("tokens", "训练 token 数"), ("params", "可训练参数"), ("speed", "实测吞吐")]
        for i, (k, label) in enumerate(fields):
            box = tk.Frame(stats, bg=P["panel2"])
            box.grid(row=i // 2, column=i % 2, sticky="nsew", padx=(0 if i % 2 == 0 else 5, 0),
                     pady=(0, 5))
            tk.Label(box, text=label, bg=P["panel2"], fg=P["muted"], font=font(8),
                     anchor="w").pack(anchor="w", padx=10, pady=(7, 0))
            v = tk.Label(box, text="-", bg=P["panel2"], fg=P["text"], font=font(12, "bold"),
                         anchor="w")
            v.pack(anchor="w", padx=10, pady=(0, 7))
            self._kv[k] = v

        tk.Label(rb, text="需要注意", bg=P["panel"], fg=P["text2"], font=font(9, "bold"),
                 anchor="w").pack(anchor="w", pady=(12, 4))
        self.notes = tk.Label(rb, text="-", bg=P["panel"], fg=P["muted"], font=font(9),
                              anchor="w", justify="left", wraplength=380)
        self.notes.pack(fill="x")

        act = tk.Frame(rb, bg=P["panel"])
        act.pack(fill="x", pady=(16, 0))
        RBtn(act, "应用并去训练台", self._apply, kind="primary", height=34, padx=16).pack(side="left")
        RBtn(act, "存为预设", self._save_preset, kind="secondary", height=34, padx=14).pack(
            side="left", padx=8)

        self._plan: planner.Plan | None = None
        self._stats_cache: dict[str, tuple[float, dict]] = {}
        self._pending_after: str | None = None

    # ------------------------------------------------------------ 数据
    def on_show(self) -> None:
        self._reload_sources()
        self._pick_quality(self.var_quality.get())

    def _reload_sources(self) -> None:
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
        self._update_data_hint()
        if not names:
            self.summary.configure(text="models 目录下还没有模型。先去「模型」页下载一个基座。",
                                   fg=P["amber"])
        self._schedule_plan()

    def _stats(self, name: str) -> dict:
        if not name:
            return {}
        p = paths.data_dir() / name
        try:
            st = p.stat()
        except OSError:
            return {}
        key = f"{p}|{st.st_mtime}|{st.st_size}"
        if key in self._stats_cache:
            return self._stats_cache[key][1]
        s = ds.stats(p)
        self._stats_cache[key] = (time.time(), s)
        return s

    def _update_data_hint(self) -> None:
        name = self.var_data.get()
        if not name:
            self.data_hint.configure(text="没有数据集，点「生成演示数据」先造一份")
            return
        s = self._stats(name)
        if not s:
            self.data_hint.configure(text="读取失败")
            return
        self.data_hint.configure(
            text=f"{s['n']} 行 · 有效 {s['valid']} · 重复 {s['dups']} · 平均 {s['avg_chars']:.0f} 字符 · "
                 f"中文占比 {s['zh_ratio']*100:.0f}% · 粗估 {s['est_tokens']/max(1,s['valid']):.0f} token/条")

    def _pick_quality(self, key: str) -> None:
        self.var_quality.set(key)
        for k, frame in self._tiles.items():
            on = k == key
            frame.configure(highlightbackground=P["accent"] if on else P["border"],
                            bg=P["panel2"] if on else P["panel"])
            for child in frame.winfo_children():
                child.configure(bg=P["panel2"] if on else P["panel"])
        self._schedule_plan()

    def _schedule_plan(self) -> None:
        if self._pending_after:
            try:
                self.after_cancel(self._pending_after)
            except tk.TclError:
                pass
        self._pending_after = self.after(250, self._compute)

    # ------------------------------------------------------------ 生成方案
    def _compute(self) -> None:
        self._pending_after = None
        self._update_data_hint()
        model_name = self.var_model.get()
        data_name = self.var_data.get()
        model = models_mod.find_model(model_name) if model_name else None
        st = self._stats(data_name)
        if model is None:
            self.summary.configure(text="请先选择一个基座模型（models 目录里有才能训练）", fg=P["amber"])
            self._plan = None
            return
        if not st or not st.get("valid"):
            self.summary.configure(text="数据集为空或不可用，先生成/导入数据", fg=P["amber"])
            self._plan = None
            return
        avg_tokens = planner.avg_tokens_from_stats(st)
        plan = planner.make_plan(
            goal=self.goal_text.get("1.0", "end").strip(),
            task=self.var_task.get(), quality=self.var_quality.get(),
            model_name=model_name, params_m=max(1.0, model.params_m or 135.0),
            samples=st["valid"], avg_tokens=avg_tokens,
            hidden=model.hidden or 576, layers=model.layers or 30,
            speed_samples=self.store.speed_samples(),
            threads=int(self.cfg.get("threads", 0) or 4))
        self._plan = plan
        mode = "全参微调" if plan.mode == "full" else f"LoRA r={plan.lora_r} α={plan.lora_alpha}"
        tier_name = planner.QUALITY_TIERS.get(plan.quality, ("可用",))[0]
        self.summary.configure(
            text=f"{plan.task} · {tier_name}档 · {mode}\n"
                 f"{plan.epochs:.0f} 轮 · {plan.max_len} token · 等效 batch {plan.batch_size*plan.grad_accum} · lr {plan.lr:g}",
            fg=P["text"])
        self._kv["steps"].configure(text=f"{plan.steps}")
        self._kv["time"].configure(text=paths.human_time(plan.est_seconds))
        self._kv["ram"].configure(text=f"{plan.est_ram_gb:.1f} GB")
        self._kv["tokens"].configure(text=f"{plan.tokens/1000:.0f}K")
        self._kv["params"].configure(
            text=f"≈{plan.params_m*0.018:.1f}M" if plan.mode == "lora" else f"{plan.params_m:.0f}M")
        self._kv["speed"].configure(text=f"{planner.estimate_tps(plan.params_m, plan.max_len, self.store.speed_samples()):.0f} tok/s")
        self.notes.configure(text="· " + "\n· ".join(plan.notes))

    def _apply(self) -> None:
        if self._plan is None:
            self._compute()
        if self._plan is None:
            self.toast("还没有可用的方案", "warn")
            return
        self.app.shared["pending_train"] = {
            "model": self.var_model.get(), "dataset": self.var_data.get(),
            "goal": self.goal_text.get("1.0", "end").strip(),
            "plan": self._plan.to_train_cfg(),
        }
        self.app.toast("方案已送到训练台", "ok")
        self.app.goto("train")

    def _save_preset(self) -> None:
        if self._plan is None:
            self._compute()
        if self._plan is None:
            return
        name = self.app.ask_text("保存预设", "预设名称", f"{self.var_task.get()}-{self.var_quality.get()}")
        if not name:
            return
        self.store.set_preset(name, self._plan.to_train_cfg())
        self.toast(f"预设「{name}」已保存，训练台可直接选用", "ok")

    def _make_seed(self) -> None:
        try:
            n = max(4, min(2000, int(float(self.var_samples.get()))))
        except ValueError:
            self.toast("条数需要是数字", "warn")
            return
        samples = ds.make_seed(self.var_task.get(), n=n)
        out = paths.data_dir() / f"seed_{self.var_task.get()}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        self.toast(f"已生成 {len(samples)} 条演示数据：{out.name}", "ok", 3600)
        self._reload_sources()
        self.var_data.set(out.name)
        self._schedule_plan()

    def tick(self) -> None:
        pass
