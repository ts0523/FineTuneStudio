# -*- coding: utf-8 -*-
"""训练层 · AI 帮训：目标 → 找公开资源 → 自编数据 → 训练 → 独立留出集验证 → 打分 → 下一轮。

闭环每一轮：
1. 按目标联网搜索公开数据集（只读元信息，不下载内容，遵守许可）
2. AI 把这些资源和上一轮的验收建议拼成造数描述，自己编写训练数据
3. 用 planner 反推超参并启动 train_worker
4. 训练产物立刻在 chat_worker（= 实验台同一套权重）上答题
5. 用固定留出集（训练前抽取，永不混入任何一轮训练数据）评分，给出下一轮补什么

注意：这里的分只代表「留出集上的表现」，不等于泛化能力已被充分验证。
"""
from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ...core import ai
from ...core import autotrain as at
from ...core import dataset as ds
from ...core import fsutil
from ...core import models as models_mod
from ...core import nl, paths, runtime
from ..base import Page
from ..theme import P, font
from ..widgets import Badge, Card, Check, Labeled, LogView, RBtn, ScrollFrame, TextArea

TRAIN_KEY = "autotrain-train"
EVAL_KEY = "autotrain-eval"

EVAL_LOAD_TIMEOUT = 180
EVAL_Q_TIMEOUT = 90
JUDGE_TIMEOUT = 90
GENERATE_TIMEOUT = 420
TRAIN_IDLE_TIMEOUT = 60    # 训练启动后多久没有任何事件就认为卡住


class AutoTrainPage(Page):
    key = "autotrain"
    group = "训练层"
    title = "AI 帮训"
    subtitle = "说一个目标，它自己找资源、造数据、开训、用独立留出集验收，再补下一轮"

    def build(self) -> None:
        root = tk.Frame(self, bg=P["bg"])
        root.pack(fill="both", expand=True)
        root.grid_columnconfigure(0, weight=4, uniform="at")
        root.grid_columnconfigure(1, weight=5, uniform="at")
        root.grid_rowconfigure(0, weight=1)

        self._running = False
        self._stopping = False
        self._stop_requested = False
        self._sid = ""
        self._round = 0
        self._rounds = 0
        self._samples: list[dict] = []
        self._dataset: Path | None = None
        self._out_dir: Path | None = None
        self._run_id = ""
        self._advice = ""
        self._resources: list[dict] = []
        self._scores: list[dict] = []
        self._eval_items: list[dict] = []
        self._eval_answers: list[str] = []
        self._eval_index = 0
        self._total_steps = 0
        self._train_ok = False
        self._holdout: list[dict] = []
        self._holdout_id = ""
        self._holdout_path: Path | None = None
        self._dataset_version = ""
        self._history_file: Path | None = None
        self._train_token = ""
        self._eval_token = ""
        self._judge_round = 0
        self._generation_model = ""
        self._phase = ""
        self._timeout_after: str | None = None
        self._timeout_action = None
        self._force_after: str | None = None
        self._final_kind = ""
        self._final_reason = ""
        self._generating = False

        self._build_left(root)
        self._build_right(root)

        self.app.bind_job(TRAIN_KEY, self._on_train_event)
        self.app.bind_job(EVAL_KEY, self._on_eval_event)

    # ---------------------------------------------------------------- 左
    def _build_left(self, parent) -> None:
        card = Card(parent, "训练目标", "越具体越好：说清对象、任务、回答风格")
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        b = card.body

        self.goal = TextArea(b, height=6)
        self.goal.pack(fill="x", ipady=2)
        self.goal.insert("1.0", str(self.cfg.get("autotrain.goal", "") or
                                   "让模型学会「咖啡店点单」：用户提出饮品需求，"
                                   "模型给出推荐、规格、价格，回答不超过三句。"))

        row = tk.Frame(b, bg=P["panel"])
        row.pack(fill="x", pady=(12, 0))
        tk.Label(row, text="基座模型", bg=P["panel"], fg=P["muted"],
                 font=font(9)).pack(side="left")
        self.var_model = tk.StringVar()
        self.cb_model = ttk.Combobox(row, textvariable=self.var_model, state="readonly",
                                     font=font(9), width=26)
        self.cb_model.pack(side="left", padx=(8, 0), ipady=2)

        grid = tk.Frame(b, bg=P["panel"])
        grid.pack(fill="x", pady=(10, 0))
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)

        self.var_rounds = tk.StringVar(value=str(self.cfg.get("autotrain.rounds", 3)))
        w = Labeled(grid, "循环轮数", "每轮都重新验收", bg=P["panel"])
        w.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))
        ttk.Spinbox(w.holder, from_=1, to=10, increment=1, textvariable=self.var_rounds,
                    font=font(9)).pack(fill="x", ipady=3)

        self.var_per = tk.StringVar(value=str(self.cfg.get("autotrain.per_round", 40)))
        w = Labeled(grid, "每轮新增条数", "30~60 比较稳", bg=P["panel"])
        w.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=(0, 8))
        ttk.Spinbox(w.holder, from_=8, to=120, increment=4, textvariable=self.var_per,
                    font=font(9)).pack(fill="x", ipady=3)

        self.var_eval = tk.StringVar(value=str(self.cfg.get("autotrain.eval_n", 6)))
        w = Labeled(grid, "留出集题数", "训练前固定，永不参与训练", bg=P["panel"])
        w.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        ttk.Spinbox(w.holder, from_=2, to=20, increment=1, textvariable=self.var_eval,
                    font=font(9)).pack(fill="x", ipady=3)

        self.var_quality = tk.StringVar(value="可用")
        w = Labeled(grid, "训练强度", "小数据用「冒烟」更快", bg=P["panel"])
        w.grid(row=1, column=1, sticky="ew", padx=(6, 0))
        ttk.Combobox(w.holder, textvariable=self.var_quality, state="readonly",
                     values=[x[1] for x in nl.quality_tiers()], font=font(9)).pack(
            fill="x", ipady=3)

        self.var_web = tk.BooleanVar(value=bool(self.cfg.get("autotrain.use_web", True)))
        Check(b, "联网找公开资源（只读元信息，生成的数据由 AI 自己写）", self.var_web,
              bg=P["panel"]).pack(anchor="w", pady=(12, 4))
        tk.Label(b, text="只会查询公开数据集的名称/许可，不下载内容；生成数据用于本地小模型训练，"
                        "属合法的自建教学数据。", bg=P["panel"], fg=P["muted"], font=font(8),
                 anchor="w", justify="left", wraplength=390).pack(fill="x")

        act = tk.Frame(b, bg=P["panel"])
        act.pack(fill="x", pady=(14, 0))
        self.btn_start = RBtn(act, "开始帮训", self.start, kind="primary", height=36, padx=16)
        self.btn_start.pack(side="left")
        self.btn_stop = RBtn(act, "停止", self.stop, kind="secondary", height=36, size=9,
                             padx=14)
        self.btn_stop.pack(side="left", padx=8)

        self.res_label = tk.Label(b, text="", bg=P["panel"], fg=P["muted"], font=font(8),
                                  anchor="w", justify="left", wraplength=400)
        self.res_label.pack(fill="x", pady=(12, 0))
        self.data_label = tk.Label(b, text="", bg=P["panel"], fg=P["text2"], font=font(8),
                                   anchor="w", justify="left", wraplength=400)
        self.data_label.pack(fill="x", pady=(4, 0))
        self.holdout_label = tk.Label(b, text="", bg=P["panel"], fg=P["muted"], font=font(8),
                                      anchor="w", justify="left", wraplength=400)
        self.holdout_label.pack(fill="x", pady=(2, 0))

    # ---------------------------------------------------------------- 右
    def _build_right(self, parent) -> None:
        card = Card(parent, "帮训进度", "每轮：造数 → 训练 → 独立留出集验收 → 打分 → 补数据")
        card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        b = card.body
        b.grid_rowconfigure(2, weight=1)
        b.grid_columnconfigure(0, weight=1)

        head = tk.Frame(b, bg=P["panel"])
        head.grid(row=0, column=0, sticky="ew")
        self.stage = tk.Label(head, text="空闲", bg=P["panel"], fg=P["text"],
                              font=font(12, "bold"), anchor="w")
        self.stage.pack(side="left")
        self.round_badge = Badge(head, "第 0 轮", "panel3", bg=P["panel"])
        self.round_badge.pack(side="right")
        self.score_badge = Badge(head, "未评分", "panel3", bg=P["panel"])
        self.score_badge.pack(side="right", padx=(0, 8))

        from ..widgets import Bar

        self.bar = Bar(b, height=10, bg=P["panel"])
        self.bar.grid(row=1, column=0, sticky="ew", pady=(10, 8))

        history = ScrollFrame(b, bg=P["panel"])
        history.grid(row=2, column=0, sticky="nsew")
        self.history_box = history.inner
        tk.Label(self.history_box, text="还没开始。填好目标后点「开始帮训」。",
                 bg=P["panel"], fg=P["muted"], font=font(9), anchor="w").pack(anchor="w", pady=6)

        logcard = Card(b, "运行日志", "", pad=0)
        logcard.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.log = LogView(logcard.body, height=7, max_lines=500, bg=P["panel"])
        self.log.pack(fill="x")

    # ---------------------------------------------------------------- 状态
    def on_show(self) -> None:
        models = models_mod.scan_models()
        names = [m.name for m in models]
        self.cb_model.configure(values=names)
        if names and self.var_model.get() not in names:
            self.var_model.set(models_mod.default_model_name(models, self.cfg.get("last.model", "")))

    def tick(self) -> None:
        if self._running:
            self.btn_stop.set_enabled(not self._stopping)

    def _log(self, text: str, level: str = "log") -> None:
        self.log.append(text, level)

    # ---------------------------------------------------------------- 超时
    def _disarm_timeout(self) -> None:
        self._phase = ""
        self._timeout_action = None
        if self._timeout_after:
            try:
                self.after_cancel(self._timeout_after)
            except tk.TclError:
                pass
            self._timeout_after = None

    def _set_phase(self, name: str, timeout: float = 0, action=None) -> None:
        self._disarm_timeout()
        self._phase = name
        if not timeout or action is None:
            return
        self._timeout_action = action

        def fire() -> None:
            if not self._running or self._phase != name:
                return
            self._timeout_after = None
            self._log(f"[超时] {name} 超过 {int(timeout)} 秒", "err")
            action()

        self._timeout_after = self.after(int(timeout * 1000), fire)

    # ---------------------------------------------------------------- 启动
    def start(self) -> None:
        if self._running or self._stopping:
            self.toast("帮训已经在跑了（或正在等待旧任务退出）", "warn")
            return
        for key in (TRAIN_KEY, EVAL_KEY):
            job = self.app.jobs.get(key)
            if job and job.running:
                self.toast("上一轮任务还没退出，等它结束再开始", "warn")
                return
        goal = self.goal.get("1.0", "end").strip()
        if len(goal) < 6:
            self.toast("先写一句训练目标", "warn")
            return
        if not ai.enabled():
            self.toast("AI 帮训需要 Key：到「设置 → AI 功能」配置（环境变量或 DPAPI）", "warn", 4600)
            self.app.goto("settings")
            return
        env = self.app.env
        if env is None or env.python is None:
            self.toast("没有训练环境（.venv），AI 帮训无法训练模型", "warn", 4600)
            self.app.goto("settings")
            return
        model = models_mod.find_model(self.var_model.get())
        if model is None or not model.complete:
            self.toast("先选一个完整的基座模型", "warn", 4200)
            return
        try:
            rounds = max(1, int(float(self.var_rounds.get() or 3)))
            per = max(8, int(float(self.var_per.get() or 40)))
            eval_n = max(2, int(float(self.var_eval.get() or 6)))
        except ValueError:
            self.toast("轮数 / 条数 / 题数需要是整数", "err")
            return

        self.cfg.set("autotrain.goal", goal[:400], save=False)
        self.cfg.set("autotrain.rounds", rounds, save=False)
        self.cfg.set("autotrain.per_round", per, save=False)
        self.cfg.set("autotrain.eval_n", eval_n, save=False)
        self.cfg.set("autotrain.use_web", bool(self.var_web.get()), save=False)
        self.cfg.save()

        sid = at.new_session_id()
        self._sid = sid
        self._samples = []
        self._scores = []
        self._advice = ""
        self._generation_model = ""
        self._resources = []
        self._round = 0
        self._rounds = rounds
        self._per_round = per
        self._eval_n = eval_n
        self._goal = goal
        self._model = model
        self._holdout = []
        self._holdout_id = ""
        self._holdout_path = None
        self._dataset_version = ""
        self._dataset = paths.data_dir() / f"autotrain-{nl.safe_name(goal)[:24]}-{sid[-8:]}.jsonl"
        self._history_file = at.history_path(sid)
        self._running = True
        self._stopping = False
        self._stop_requested = False
        self._final_kind = ""
        self._train_token = ""
        self._eval_token = ""
        self.btn_start.set_enabled(False)
        self.btn_stop.set_enabled(True)
        for w in self.history_box.winfo_children():
            w.destroy()
        self.log.clear()
        self._log(f"会话 {sid}", "info")
        self._log(f"目标：{goal[:80]}", "info")
        self._log(f"基座：{model.label()} · 计划 {rounds} 轮 × {per} 条 · 留出集 {eval_n} 题", "cmd")
        self._log("留出集在第一轮造数后固定，之后所有轮次都不参与训练；分数只代表留出集表现。",
                  "info")
        self.data_label.configure(text=f"数据文件：data/{self._dataset.name}")
        self.holdout_label.configure(text="留出集：尚未建立")
        self._write_history_base(extra={})
        self._next_round()

    def stop(self) -> None:
        if not self._running:
            self.toast("没有在跑的帮训", "warn")
            return
        if self._stopping:
            return
        self._stop_requested = True
        self._stopping = True
        self.stage.configure(text="正在停止：等待旧任务退出并保存检查点…", fg=P["amber"])
        self._log("已请求停止：不再开始下一轮；先等当前任务退出/保存检查点", "warn")
        self._shutdown_jobs()
        self._maybe_finalize("stopped", "已停止（训练检查点已保存）")

    # ---------------------------------------------------------------- 收尾
    def _shutdown_jobs(self) -> None:
        train = self.app.jobs.get(TRAIN_KEY)
        if train and train.running:
            train.stop()
        ev = self.app.jobs.get(EVAL_KEY)
        if ev and ev.running:
            if not ev.send({"cmd": "quit"}):
                ev.kill()

    def _maybe_finalize(self, kind: str, reason: str) -> None:
        self._final_kind = kind
        self._final_reason = reason
        running = [k for k in (TRAIN_KEY, EVAL_KEY) if self.app.jobs.running(k)]
        if not running:
            self._finalize_now()
            return
        self.btn_start.set_enabled(False)
        self.stage.configure(text="等待旧任务退出…", fg=P["amber"])
        if self._force_after is None:
            def force() -> None:
                self._force_after = None
                if any(self.app.jobs.running(k) for k in (TRAIN_KEY, EVAL_KEY)):
                    # 先标终态再杀：退出事件可能比 taskkill 返回更早到达。
                    self._final_kind = "killed"
                    self._final_reason = "旧任务未按时退出，已强杀"
                    for k in (TRAIN_KEY, EVAL_KEY):
                        job = self.app.jobs.get(k)
                        if job and job.running:
                            job.kill()
            self._force_after = self.after(8000, force)

    def _finalize_now(self) -> None:
        if self._force_after:
            try:
                self.after_cancel(self._force_after)
            except tk.TclError:
                pass
            self._force_after = None
        self._disarm_timeout()
        self._running = False
        self._stopping = False
        self._stop_requested = False
        self._generating = False
        kind = self._final_kind or "done"
        reason = self._final_reason or "结束"
        self.btn_start.set_enabled(True)
        self.btn_stop.set_enabled(False)
        color = {"done": P["green"], "stopped": P["amber"]}.get(kind, P["red"])
        self.stage.configure(text=f"结束：{reason}", fg=color)
        if self._scores:
            best = max(x["score"] for x in self._scores)
            self.score_badge.set(f"最佳 {best} 分", "green" if best >= 70 else "amber")
            self._log(f"共 {len(self._scores)} 轮，留出集最高 {best} 分（不代表泛化能力已验证）",
                      "ok")
        self._log(f"本次帮训结束：{reason}", "info")
        self._write_history_base(extra={"finished": time.time(), "result": kind,
                                        "reason": reason})
        for key in (TRAIN_KEY, EVAL_KEY):
            job = self.app.jobs.get(key)
            if job and job.running:
                job.kill()
        if self._dataset and self._dataset.is_file():
            self.data_label.configure(
                text=f"数据已保存：data/{self._dataset.name}（{self._count(self._dataset)} 条，"
                     f"可在「训练台」复用）")
            self.cfg.set("last.dataset", self._dataset.name)
            self.cfg.save()

    def _finish(self, kind: str, reason: str) -> None:
        """失败/完成路径：先请求任务退出，全部退出后再允许新任务开始。"""
        if not self._running:
            return
        self._shutdown_jobs()
        self._maybe_finalize(kind, reason)

    # ---------------------------------------------------------------- 每一轮
    def _next_round(self) -> None:
        if self._stop_requested:
            self._finish("stopped", "已停止")
            return
        if self._round >= self._rounds:
            self._finish("done", "全部完成")
            return
        self._round += 1
        self.round_badge.set(f"第 {self._round}/{self._rounds} 轮", "accent")
        self.stage.configure(text=f"第 {self._round} 轮：联网找资源 + 写数据", fg=P["accent"])
        self.bar.set(0.0)
        self._log(f"—— 第 {self._round} 轮开始 ——", "cmd")

        use_web = bool(self.var_web.get())
        goal = self._goal
        advice = self._advice
        n = self._per_round
        sid = self._sid
        self._generating = True
        self._set_phase(f"第 {self._round} 轮造数", GENERATE_TIMEOUT, self._on_generate_timeout)

        def work() -> None:
            resources: list[dict] = []
            res_err = ""
            if use_web:
                self.app.post(lambda: self.stage.configure(
                    text=f"第 {self._round} 轮：搜索公开资源…"))
                resources, res_err = at.search_resources(goal)
            brief = at.round_brief(goal, advice, resources, n, "问答对")
            try:
                samples, generation_model = nl.generate(
                    brief, n, kind="qa", lang="中文",
                    cancel=lambda: (not self._running) or self._stop_requested
                    or not at.session_ok(sid, self._sid))
                error = ""
            except Exception as exc:  # noqa: BLE001
                samples, generation_model = [], ""
                error = f"{type(exc).__name__}: {exc}"
            self.app.post(lambda: self._data_done(
                sid, samples, resources, res_err, error, generation_model))

        threading.Thread(target=work, daemon=True).start()

    def _on_generate_timeout(self) -> None:
        # 造数线程无法强杀，但旧会话回调会被 sid 拦截
        self._generating = False
        self._log("造数超时：放弃本轮，结束帮训", "err")
        self._finish("failed", "造数超时")

    def _data_done(self, sid: str, samples: list[dict], resources: list[dict],
                   res_err: str, error: str, generation_model: str = "") -> None:
        if not self._running or not at.session_ok(sid, self._sid):
            if at.session_ok(sid, self._sid):
                self._log("收到旧会话的造数结果，已忽略", "warn")
            return
        self._generating = False
        if self._stop_requested:
            self._finish("stopped", "已停止")
            return
        if error or not samples:
            self._log(f"造数失败：{error or '没有生成样本'}", "err")
            self._finish("failed", "造数失败")
            return
        self._disarm_timeout()
        self._resources = resources
        self._generation_model = generation_model
        self.bar.set(0.2)
        self._log(f"本轮生成 {len(samples)} 条（AI 模型：{generation_model or '未知'}）", "ok")
        if res_err:
            self._log(f"联网搜索失败（继续用自编数据）：{res_err}", "warn")
        self.res_label.configure(text=at.summarize_resources(resources))

        if not self._holdout:
            holdout, train_samples, hid = at.make_holdout(samples, self._eval_n, seed=7)
            if not holdout:
                self._log("首批样本里没有合适的单轮问答，无法建立留出集", "err")
                self._finish("failed", "留出集建立失败")
                return
            self._holdout = holdout
            self._holdout_id = hid
            self._holdout_path = paths.data_dir() / f"autotrain-holdout-{self._sid[-10:]}.jsonl"
            at.save_holdout(self._holdout_path, holdout,
                            {"session": self._sid, "holdout_id": hid, "goal": self._goal,
                             "created": time.time(), "note": "固定留出集，不参与任何训练轮次"})
            self.holdout_label.configure(
                text=f"留出集：{self._holdout_path.name}（{len(holdout)} 题，ID {hid}，永不参与训练）")
            self._log(f"已固定留出集 {len(holdout)} 题（ID {hid}）", "ok")
            samples = train_samples
        else:
            before = len(samples)
            samples = at.filter_holdout(samples, self._holdout)
            if before != len(samples):
                self._log(f"剔除 {before - len(samples)} 条与留出集重复的样本", "warn")

        if not samples:
            self._log("本轮没有可加入训练的新样本", "warn")
            self._finish("failed", "本轮无新增训练样本")
            return

        try:
            total = nl.append_to(self._dataset, samples)
        except OSError as exc:
            self._log(f"写数据失败（原文件未变）：{exc}", "err")
            self._finish("failed", "数据写入失败")
            return
        self._samples.extend(samples)
        chk = ds.verify_after_write(self._dataset)
        if chk["bad"] or chk["empty"]:
            self._log(f"数据复核未通过：坏行 {chk['bad']}，空回复 {chk['empty']}", "err")
            self._finish("failed", "数据复核失败")
            return
        leaks = at.holdout_leaks(self._dataset, self._holdout)
        if leaks:
            self._log(f"检测到留出集题混入训练集（{len(leaks)} 条），已终止：{leaks[:2]}", "err")
            self._finish("failed", "留出集泄漏")
            return
        self._dataset_version = at.data_version(self._dataset)
        self.data_label.configure(
            text=f"数据文件：data/{self._dataset.name}（本轮 +{total} 条，累计 {len(self._samples)} 条，"
                 f"版本 {self._dataset_version}）")
        self._log(f"数据版本 {self._dataset_version}；复核：有效 {chk['valid']}，重复 {chk['dup']}",
                  "info")
        plan = self._plan(len(self._samples))
        self._log(f"参数：{nl.plan_summary(plan)}", "info")
        at.write_meta(self._dataset, {"goal": self._goal, "round": self._round,
                                      "plan": plan, "resources": self._resources,
                                      "samples": len(self._samples),
                                      "session": self._sid, "holdout_id": self._holdout_id,
                                      "holdout_path": str(self._holdout_path or ""),
                                      "data_version": self._dataset_version})
        self._write_history_base(extra={})
        self._start_train(plan)

    def _plan(self, total_samples: int) -> dict:
        info = self._model
        try:
            threads = int(self.cfg.get("threads", 0) or runtime.default_threads())
        except (TypeError, ValueError):
            threads = runtime.default_threads()
        merged = list(self._samples[-self._per_round * 2:])
        plan = nl.suggest(merged, task="自定义", quality=self._quality(),
                          model_name=info.name, params_m=info.params_m or 64.0,
                          threads=threads, speed_samples=self.store.speed_samples(),
                          goal=self._goal)
        plan["samples"] = total_samples
        return plan

    def _quality(self) -> str:
        label = self.var_quality.get()
        for key, name, _desc in nl.quality_tiers():
            if name == label:
                return key
        return "usable"

    def _write_history_base(self, extra: dict) -> None:
        if not self._history_file:
            return
        base = {
            "session": self._sid,
            "goal": self._goal,
            "holdout_id": self._holdout_id,
            "holdout_path": str(self._holdout_path or ""),
            "holdout_ids": at.holdout_ids(self._holdout),
            "dataset": str(self._dataset or ""),
            "dataset_version": self._dataset_version,
            "generation_model": self._generation_model,
            "created": time.time(),
        }
        base.update(extra)
        record = base
        if self._history_file.is_file():
            try:
                old = json.loads(self._history_file.read_text(encoding="utf-8-sig"))
                if isinstance(old, dict):
                    record = {**old, **base}
            except (OSError, ValueError):
                pass
        try:
            at.write_history(self._history_file, record)
        except OSError:
            pass

    # ---------------------------------------------------------------- 训练
    def _start_train(self, plan: dict) -> None:
        self.stage.configure(text=f"第 {self._round} 轮：训练中", fg=P["purple"])
        stamp = time.strftime("%m%d-%H%M%S")
        try:
            out_dir = fsutil.safe_child(
                paths.outputs_dir(), f"autotrain-r{self._round}-{stamp}")
        except fsutil.PathSafetyError as exc:
            self._log(f"输出目录不安全：{exc}", "err")
            self._finish("failed", "输出目录不安全")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        stop_flag = out_dir / "studio_stop.flag"
        fsutil.prepare_stop_flag(stop_flag)
        cfg = {
            "model": str(self._model.path.resolve()),
            "data": str(self._dataset.resolve()),
            "output": str(out_dir),
            "events_file": str(out_dir / "studio_events.jsonl"),
            "stop_flag": str(stop_flag),
            "epochs": plan["epochs"], "batch_size": plan["batch_size"],
            "grad_accum": plan["grad_accum"], "lr": plan["lr"],
            "max_len": plan["max_len"], "val_ratio": plan["val_ratio"],
            "lora_r": plan["lora_r"], "lora_alpha": plan["lora_alpha"],
            "lora_dropout": plan["lora_dropout"], "seed": 42,
            "threads": int(self.cfg.get("threads", 0) or 0),
            "full": False, "max_steps": -1, "goal": self._goal, "resume": "",
            "session": self._sid,
        }
        name = f"autotrain-r{self._round}"
        run_id = self.store.new_id(name)
        self._run_id = run_id
        self.store.add_run({
            "id": run_id, "name": name, "status": "running", "model": self._model.name,
            "dataset": self._dataset.name, "samples": len(self._samples), "mode": "lora",
            "epochs": plan["epochs"], "out_dir": str(out_dir), "config": cfg,
            "goal": self._goal, "session": self._sid,
            "holdout_id": self._holdout_id, "data_version": self._dataset_version,
            "dataset_path": str(self._dataset),
        })
        self._total_steps = int(plan.get("steps") or 0)
        self._train_ok = False
        self._log(f"输出目录：outputs/{out_dir.name} · 约 {plan['steps']} 步", "info")
        if not self.app.spawn_worker(TRAIN_KEY, "train_worker.py", cfg, label="帮训训练",
                                     stop_file=stop_flag):
            self.store.update_run(run_id, status="failed", error="worker 启动失败")
            self._finish("failed", "训练启动失败")
            return
        job = self.app.jobs.get(TRAIN_KEY)
        self._train_token = job.token if job else ""
        self._set_phase(f"第 {self._round} 轮训练", TRAIN_IDLE_TIMEOUT, self._on_train_idle_timeout)

    def _on_train_idle_timeout(self) -> None:
        self._log("训练启动后长时间没有任何事件，判定为卡住", "err")
        self._finish("failed", "训练无响应")

    def _train_event_ok(self, evt: dict) -> bool:
        return bool(self._train_token) and evt.get("_job") == self._train_token

    def _on_train_event(self, evt: dict) -> None:
        if not self._running or not self._train_event_ok(evt):
            return
        t = evt.get("type")
        if t not in ("done", "error", "_exit"):
            self._set_phase(f"第 {self._round} 轮训练", TRAIN_IDLE_TIMEOUT,
                            self._on_train_idle_timeout)
        if t == "plan":
            self._total_steps = int(evt.get("total_steps") or self._total_steps or 0)
        elif t == "step":
            done, total = int(evt.get("step") or 0), int(evt.get("total") or self._total_steps or 0)
            self._total_steps = total
            if total:
                self.bar.set(0.2 + 0.45 * done / total)
            if done % 10 == 0:
                self.stage.configure(text=f"第 {self._round} 轮：训练 {done}/{total} 步")
        elif t == "saved":
            self._log(f"训练产物已保存：{evt.get('output')}", "ok")
        elif t == "done":
            fields = at.run_fields_from_done(evt)
            self.store.update_run(self._run_id, **fields)
            if fields["status"] != "done" or self._stop_requested:
                self._log("训练未正常完成，结束循环", "warn")
                self._finish("stopped" if self._stop_requested else "failed",
                             "训练中断" if not self._stop_requested else "已停止（检查点已保存）")
                return
            self._train_ok = True
            self.bar.set(0.65)
            self._log(f"训练完成（loss {fields.get('final_loss')}，"
                      f"{fields.get('steps_done')} 步），开始留出集验证", "ok")
            self._after(400, self._evaluate)
        elif t == "error":
            self.store.update_run(self._run_id, status="failed")
            self._log(f"训练出错：{evt.get('message')}", "err")
        elif t == "_exit":
            if evt.get("killed") or self._final_kind == "killed":
                self._maybe_finalize("killed", "训练进程被强杀")
                return
            if self._stop_requested:
                self._maybe_finalize(self._final_kind or "stopped",
                                     self._final_reason or "已停止（检查点已保存）")
                return
            code = evt.get("code")
            if code not in (0, None):
                run = self.store.get_run(self._run_id) or {}
                if run.get("status") == "running":
                    self.store.update_run(self._run_id, status="failed",
                                          error=f"进程退出 code={code}")
                if self._running:
                    self._finish("failed", f"训练进程退出 code={code}")
            elif self._running and not self._train_ok:
                self._finish("failed", "训练进程退出但未发出完成事件")

    def _after(self, ms: int, fn) -> None:
        try:
            self.after(ms, fn)
        except tk.TclError:
            pass

    # ---------------------------------------------------------------- 验证
    def _evaluate(self) -> None:
        if self._stop_requested:
            self._finish("stopped", "已停止")
            return
        if not self._holdout:
            self._log("没有固定留出集，跳过验证", "err")
            self._finish("failed", "缺少留出集")
            return
        self.stage.configure(text=f"第 {self._round} 轮：独立留出集验证", fg=P["purple"])
        self._eval_items = at.eval_pairs(list(self._holdout), n=self._eval_n)
        self._eval_answers = []
        self._eval_index = 0
        if not self._eval_items:
            self._log("留出集为空", "err")
            self._finish("failed", "留出集为空")
            return
        cfg = {"model": str(self._model.path.resolve()),
               "adapter": str(self._out_dir.resolve()),
               "threads": int(self.cfg.get("threads", 0) or 0),
               "session": self._sid}
        if not self.app.spawn_worker(EVAL_KEY, "chat_worker.py", cfg, label="帮训验证",
                                     stdin_pipe=True):
            self._finish("failed", "验证模型启动失败")
            return
        job = self.app.jobs.get(EVAL_KEY)
        self._eval_token = job.token if job else ""
        self._set_phase("验证模型加载", EVAL_LOAD_TIMEOUT, self._on_eval_load_timeout)
        self._log(f"用本轮产物加载验证模型，共 {len(self._eval_items)} 道留出题", "info")

    def _on_eval_load_timeout(self) -> None:
        self._log("验证模型加载超时", "err")
        self._finish("failed", "模型加载超时")

    def _eval_event_ok(self, evt: dict) -> bool:
        return bool(self._eval_token) and evt.get("_job") == self._eval_token

    def _on_eval_event(self, evt: dict) -> None:
        if not self._running or not self._eval_event_ok(evt):
            return
        t = evt.get("type")
        if t == "ready":
            self._disarm_timeout()
            self._log(f"验证模型就绪（{evt.get('load_seconds')}s）", "info")
            self._ask_next()
        elif t == "done":
            self._eval_answers.append(str(evt.get("text") or ""))
            self._eval_index += 1
            self._disarm_timeout()
            self._after(80, self._ask_next)
        elif t == "error":
            self._log(f"验证题出错：{evt.get('message')}", "warn")
            self._eval_answers.append("")
            self._eval_index += 1
            self._disarm_timeout()
            self._after(80, self._ask_next)
        elif t == "_exit":
            if self._running and self._eval_index < len(self._eval_items):
                if evt.get("killed") or self._final_kind == "killed":
                    self._maybe_finalize("killed", "验证进程被强杀")
                elif self._stop_requested:
                    self._maybe_finalize(self._final_kind or "stopped",
                                         self._final_reason or "已停止（检查点已保存）")
                else:
                    self._finish("failed", f"验证进程退出 code={evt.get('code')}")

    def _ask_next(self) -> None:
        if self._stop_requested:
            self._finish("stopped", "已停止")
            return
        if self._eval_index >= len(self._eval_items):
            self._judge()
            return
        index = self._eval_index
        self.bar.set(0.65 + 0.2 * index / max(1, len(self._eval_items)))
        self.stage.configure(text=f"第 {self._round} 轮：验证 {index + 1}/"
                                  f"{len(self._eval_items)}")
        item = self._eval_items[index]
        job = self.app.jobs.get(EVAL_KEY)
        if job is None or not job.running:
            self._eval_answers.append("")
            self._eval_index += 1
            self._after(80, self._ask_next)
            return
        self._set_phase(f"第 {self._round} 轮验证第 {index + 1} 题", EVAL_Q_TIMEOUT,
                        self._on_question_timeout)
        if not job.send({"cmd": "chat",
                         "messages": [{"role": "user", "content": item["q"]}],
                         "params": {"max_new_tokens": 120, "temperature": 0.3,
                                    "top_p": 0.9}}):
            self._disarm_timeout()
            self._eval_answers.append("")
            self._eval_index += 1
            self._after(80, self._ask_next)

    def _on_question_timeout(self) -> None:
        self._log(f"第 {self._eval_index + 1} 题超时，记为未答", "warn")
        job = self.app.jobs.get(EVAL_KEY)
        if job and job.running:
            job.send({"cmd": "stop"})
        self._eval_answers.append("")
        self._eval_index += 1
        self._after(200, self._ask_next)

    # ---------------------------------------------------------------- 打分
    def _judge(self) -> None:
        self.stage.configure(text=f"第 {self._round} 轮：评分", fg=P["purple"])
        self.bar.set(0.9)
        job = self.app.jobs.get(EVAL_KEY)
        if job and job.running:
            if not job.send({"cmd": "quit"}):
                job.kill()
        items = list(self._eval_items)
        answers = list(self._eval_answers)
        goal = self._goal
        sid = self._sid
        round_no = self._round
        self._judge_round = round_no

        def work() -> None:
            try:
                result = at.judge(items, answers, goal,
                                  cancel=lambda: (not self._running) or self._stop_requested
                                  or not at.session_ok(sid, self._sid))
            except Exception as exc:  # noqa: BLE001
                result = {"score": 0, "verdict": f"打分失败：{exc}", "weak": "", "advice": ""}
            self.app.post(lambda: self._round_done(sid, round_no, result, items, answers))

        threading.Thread(target=work, daemon=True).start()
        self._set_phase(f"第 {self._round} 轮评分", JUDGE_TIMEOUT, self._on_judge_timeout)

    def _on_judge_timeout(self) -> None:
        if not self._running:
            return
        local = at.local_score(self._eval_items, self._eval_answers)
        self._log("AI 评分超时，改用本地一致性评分（保守）", "warn")
        self._judge_round = 0
        self._round_done(self._sid, self._round, local, list(self._eval_items),
                         list(self._eval_answers), late=False)

    def _round_done(self, sid: str, round_no: int, result: dict, items: list[dict],
                    answers: list[str], late: bool = True) -> None:
        if not self._running or not at.session_ok(sid, self._sid) or round_no != self._round:
            self._log("收到旧会话/旧轮次的评分结果，已忽略", "warn")
            return
        if self._stop_requested:
            self._finish("stopped", "已停止")
            return
        if late and self._judge_round != round_no:
            return
        self._disarm_timeout()
        score = int(result.get("score") or 0)
        self._scores.append({"round": round_no, "score": score,
                             "verdict": result.get("verdict", ""),
                             "advice": result.get("advice", ""),
                             "judge_source": result.get("judge_source", "local"),
                             "ai_model": result.get("ai_model", ""),
                             "ai_score": result.get("ai_score"),
                             "holdout_id": self._holdout_id,
                             "data_version": self._dataset_version})
        self._advice = str(result.get("advice") or result.get("weak") or "")
        self.score_badge.set(f"{score} 分", "green" if score >= 70 else
                             ("amber" if score >= 45 else "red"))
        self.bar.set(1.0)
        try:
            self._render_round(result, items, answers)
            self._log(f"第 {round_no} 轮留出集评分 {score}：{result.get('verdict', '')}", "ok")
            self._append_round_history(round_no, result, items, answers)
        except Exception as exc:  # noqa: BLE001
            # 渲染/存档失败不能阻断状态机，否则界面会一直停在“运行中”。
            self._log(f"评分结果展示/存档失败：{type(exc).__name__}: {exc}", "err")
            self.app.log(f"[AI帮训] 评分结果展示/存档失败：{type(exc).__name__}: {exc}", "err")
        finally:
            self._after(600, self._next_round)

    def _append_round_history(self, round_no: int, result: dict, items: list[dict],
                              answers: list[str]) -> None:
        if not self._history_file:
            return
        rec = {
            "round": round_no,
            "score": int(result.get("score") or 0),
            "verdict": str(result.get("verdict") or "")[:200],
            "advice": str(result.get("advice") or "")[:200],
            "judge_source": str(result.get("judge_source") or "local"),
            "ai_model": str(result.get("ai_model") or ""),
            "ai_score": result.get("ai_score"),
            "holdout_id": self._holdout_id,
            "data_version": self._dataset_version,
            "questions": [it["q"][:120] for it in items],
            "answers": [a[:200] for a in answers],
            "ts": time.time(),
        }
        base = {"session": self._sid, "goal": self._goal,
                "holdout_id": self._holdout_id,
                "holdout_ids": at.holdout_ids(self._holdout),
                "holdout_path": str(self._holdout_path or ""),
                "dataset": str(self._dataset or ""),
                "dataset_version": self._dataset_version}
        try:
            at.append_history(self._history_file, base, rec)
        except OSError as exc:
            self._log(f"评分历史写入失败：{exc}", "warn")

    def _render_round(self, result: dict, items: list[dict], answers: list[str]) -> None:
        box = tk.Frame(self.history_box, bg=P["panel2"], highlightbackground=P["border"],
                       highlightthickness=1)
        box.pack(fill="x", pady=4, padx=2)
        head = tk.Frame(box, bg=P["panel2"])
        head.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(head, text=f"第 {self._round} 轮 · 留出集 {result.get('score', 0)} 分",
                 bg=P["panel2"], fg=P["text"], font=font(10, "bold"), anchor="w").pack(side="left")
        tk.Label(head, text=str(result.get("verdict") or "")[:50], bg=P["panel2"],
                 fg=P["muted"], font=font(8), anchor="e").pack(side="right")
        for i, item in enumerate(items[:3]):
            got = (answers[i] if i < len(answers) else "").strip()[:120] or "（空回复）"
            tk.Label(box, text=f"问：{item['q'][:60]}\n答：{got}", bg=P["panel2"], fg=P["text2"],
                     font=font(8), anchor="w", justify="left", wraplength=460,
                     padx=10, pady=3).pack(anchor="w")
        advice = str(result.get("advice") or "")
        if advice:
            tk.Label(box, text=f"下一轮补：{advice}", bg=P["panel2"], fg=P["amber"],
                     font=font(8), anchor="w", justify="left", wraplength=460,
                     padx=10, pady=2).pack(anchor="w", pady=(2, 8))
        tk.Label(box, text="分数仅代表这份固定留出集，不代表泛化能力已验证。",
                 bg=P["panel2"], fg=P["muted"], font=font(7), anchor="w",
                 padx=10, pady=0).pack(anchor="w", pady=(0, 6))
        self._to_bottom()
        self.app.refresh_ai()

    def _to_bottom(self) -> None:
        try:
            parent = self.history_box.master.master  # canvas
            parent.yview_moveto(1.0)
        except Exception:  # noqa: BLE001
            pass

    def _count(self, path: Path) -> int:
        try:
            return at.count_lines(path)
        except OSError:
            return len(self._samples)
