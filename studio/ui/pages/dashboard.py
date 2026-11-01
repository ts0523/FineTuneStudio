# -*- coding: utf-8 -*-
"""总览页。"""
from __future__ import annotations

import time
import tkinter as tk

from ...core import models as models_mod
from ...core import paths, sysinfo
from ..base import Page
from ..theme import P, font
from ..widgets import Card, Metric, RBtn, ScrollFrame, Table

WRAP = 200


class DashboardPage(Page):
    key = "dashboard"
    group = "系统"
    title = "总览"
    subtitle = "环境状态、资源占用与最近实验"

    def build(self) -> None:
        scroll = ScrollFrame(self)
        scroll.pack(fill="both", expand=True)
        f = scroll.inner
        for c in range(4):
            f.grid_columnconfigure(c, weight=1, uniform="m")

        self.m_models = self._metric(f, 0, "本地模型", "accent")
        self.m_runs = self._metric(f, 1, "实验记录", "purple")
        self.m_best = self._metric(f, 2, "最佳验证 loss", "green")
        self.m_disk = self._metric(f, 3, "磁盘剩余", "amber")

        env_card = Card(f, "训练环境", "来自 .venv 的真实检测结果")
        env_card.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(14, 0), padx=(0, 7))
        self.env_body = env_card.body
        self._env_labels: dict[str, tk.Label] = {}
        for i, (k, label) in enumerate([("python", "解释器"), ("torch", "PyTorch"),
                                       ("libs", "训练依赖"), ("path", "项目目录")]):
            row = tk.Frame(self.env_body, bg=P["panel"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=P["panel"], fg=P["muted"], font=font(9),
                     width=9, anchor="w").pack(side="left")
            v = tk.Label(row, text="-", bg=P["panel"], fg=P["text2"], font=font(9),
                         anchor="w", justify="left", wraplength=330)
            v.pack(side="left", fill="x", expand=True)
            self._env_labels[k] = v
        act = tk.Frame(self.env_body, bg=P["panel"])
        act.pack(fill="x", pady=(12, 0))
        RBtn(act, "重新检测", self.app.refresh_env, kind="secondary", height=30, size=9,
             padx=12).pack(side="left")
        RBtn(act, "完整自检", lambda: self.app.goto("tools"), kind="ghost", height=30, size=9,
             padx=12).pack(side="left", padx=8)

        quick = Card(f, "快速开始", "三条最常用的路径")
        quick.grid(row=1, column=2, columnspan=2, sticky="nsew", pady=(14, 0), padx=(7, 0))
        qb = quick.body
        self._quick_row(qb, "新手教程", "7 步从零训练出自己的模型，每步都告诉你要点哪里", "打开教程",
                        lambda: self.app.goto("tutorial"), "primary")
        self._quick_row(qb, "跑通链路", "用示例数据跑 2 步，约 10 秒，确认训练能跑", "开始冒烟",
                        self._smoke, "secondary")
        self._quick_row(qb, "正式微调", "配置数据与超参，训练并实时看 loss 曲线", "去训练台",
                        lambda: self.app.goto("train"), "secondary")
        self._quick_row(qb, "验证效果", "加载适配器与模型对话，检查它学会了什么", "去对话",
                        lambda: self.app.goto("chat"), "secondary")

        runs_card = Card(f, "最近实验", "双击一行查看详情")
        runs_card.grid(row=2, column=0, columnspan=4, sticky="nsew", pady=(14, 0))
        self.table = Table(runs_card.body, [
            ("name", "实验名", 200, "w"), ("model", "基座", 150, "w"),
            ("mode", "方式", 70, "w"), ("samples", "样本", 60, "e"),
            ("loss", "最终 loss", 90, "e"), ("duration", "用时", 80, "e"),
            ("status", "状态", 80, "w"), ("created", "开始时间", 130, "w"),
        ], height=7, on_double=self._open_run)
        self.table.pack(fill="both", expand=True)
        self.table.bind("<Double-1>", lambda e: self._open_run(self.table.selected()))
        self.empty_hint = tk.Label(runs_card.body, text="还没有实验记录。点上面的「开始冒烟」跑第一个。",
                                   bg=P["panel"], fg=P["muted"], font=font(9))
        self.empty_hint.pack(anchor="w", pady=(8, 0))

    def _metric(self, parent, col: int, label: str, color: str) -> Metric:
        card = Card(parent, pad=14)
        card.grid(row=0, column=col, sticky="nsew",
                  padx=(0 if col == 0 else 7, 0 if col == 3 else 7))
        m = Metric(card.body, label, "-", "", color=color, bg=P["panel"])
        m.pack(fill="x")
        return m

    def _quick_row(self, parent, title: str, desc: str, btn: str, cmd, kind: str) -> None:
        row = tk.Frame(parent, bg=P["panel"])
        row.pack(fill="x", pady=(0, 10))
        text = tk.Frame(row, bg=P["panel"])
        text.pack(side="left", fill="x", expand=True)
        tk.Label(text, text=title, bg=P["panel"], fg=P["text"], font=font(10, "bold"),
                 anchor="w").pack(anchor="w")
        tk.Label(text, text=desc, bg=P["panel"], fg=P["muted"], font=font(8), anchor="w",
                 wraplength=330, justify="left").pack(anchor="w")
        RBtn(row, btn, cmd, kind=kind, height=30, size=9, padx=12).pack(side="right", padx=(10, 0))

    # ------------------------------------------------------------ 数据
    def on_show(self) -> None:
        self.refresh()
        if self.app.env is not None:
            self.on_env(self.app.env)

    def on_env(self, env) -> None:
        if not hasattr(self, "_env_labels"):
            return
        self._env_labels["python"].configure(
            text=f"{env.version or '未知'} · {env.python}" if env.python else (env.error or "未找到"),
            fg=P["text"] if env.python else P["red"])
        self._env_labels["torch"].configure(
            text=f"{env.torch} · CUDA {'可用' if env.cuda else '不可用（CPU 模式）'}", fg=P["text2"])
        libs = env.lib_versions()
        ok = [k for k in ("transformers", "peft", "datasets", "accelerate") if libs.get(k)]
        self._env_labels["libs"].configure(
            text=" · ".join(f"{k} {libs[k]}" for k in ok) or "缺失",
            fg=P["text2"] if len(ok) == 4 else P["amber"])
        self._env_labels["path"].configure(text=str(paths.root()), fg=P["text2"])

    def refresh(self) -> None:
        ms = models_mod.scan_models()
        total = sum(m.size for m in ms)
        self.m_models.set(str(len(ms)), paths.human_size(total))
        c = self.store.counts()
        self.m_runs.set(str(c["runs"]), f"完成 {c['done']} 个 · 累计 {paths.human_time(c['total_seconds'])}")
        if c["best_loss"] is not None:
            self.m_best.set(f"{c['best_loss']:.4f}", f"最近 {c['last_loss']:.4f}")
        else:
            self.m_best.set("-", "还没有完成的实验")
        snap = sysinfo.snapshot(str(paths.root()))
        self.m_disk.set(f"{snap.disk_free/1024**3:.0f} GB",
                        f"共 {snap.disk_total/1024**3:.0f} GB · 内存可用 {snap.mem_available/1024**3:.1f} GB")

        rows = []
        for r in self.store.recent(10):
            rows.append({
                "_id": r.get("id"),
                "name": r.get("name", ""),
                "model": (r.get("model") or "").split("\\")[-1].split("/")[-1],
                "mode": "全参" if r.get("mode") == "full" else "LoRA",
                "samples": r.get("samples", ""),
                "loss": f"{r['final_loss']:.4f}" if isinstance(r.get("final_loss"), (int, float)) else "-",
                "duration": paths.human_time(r.get("seconds")),
                "status": {"running": "运行中", "done": "完成", "stopped": "已停止",
                           "failed": "失败"}.get(r.get("status"), r.get("status", "")),
                "created": time.strftime("%m-%d %H:%M", time.localtime(r.get("created", 0))),
                "_tag": r.get("status", "done"),
            })
        self.table.set_rows(rows)
        if rows:
            self.empty_hint.pack_forget()
        else:
            self.empty_hint.pack(anchor="w", pady=(8, 0))

    def _open_run(self, row) -> None:
        if not row:
            return
        self.app.shared["focus_run"] = row.get("_id")
        self.app.goto("runs")

    # ------------------------------------------------------------ 动作
    def _smoke(self) -> None:
        page = self.app._pages.get("train")  # type: ignore[attr-defined]
        self.app.goto("train")
        page.ensure()
        page.apply_smoke()
