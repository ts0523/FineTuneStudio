# -*- coding: utf-8 -*-
"""实验页：历史记录、曲线对比、重跑与产物管理。"""
from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ...core import paths, procs
from ..base import Page
from ..theme import P, font
from ..widgets import Card, LineChart, RBtn, Table

STATUS_TEXT = {"running": "运行中", "done": "完成", "stopped": "已停止", "failed": "失败"}


class RunsPage(Page):
    key = "runs"
    group = "训练层"
    title = "实验记录"
    subtitle = "每次训练都留档：超参、曲线、产物、可重跑"

    def build(self) -> None:
        f = tk.Frame(self, bg=P["bg"])
        f.pack(fill="both", expand=True)
        f.grid_columnconfigure(0, weight=4, uniform="r")
        f.grid_columnconfigure(1, weight=5, uniform="r")
        f.grid_rowconfigure(0, weight=1)

        left = Card(f, "实验记录", "双击一行查看详情")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        lb = left.body
        self.table = Table(lb, [
            ("name", "实验", 180, "w"), ("status", "状态", 70, "w"),
            ("loss", "loss", 70, "e"), ("duration", "用时", 70, "e"),
            ("created", "时间", 110, "w"),
        ], height=16, on_select=self._detail)
        self.table.pack(fill="both", expand=True)
        row = tk.Frame(lb, bg=P["panel"])
        row.pack(fill="x", pady=(10, 0))
        RBtn(row, "刷新", self.refresh, kind="secondary", height=30, size=9, padx=12).pack(side="left")
        RBtn(row, "打开产物", self._open_dir, kind="ghost", height=30, size=9, padx=12).pack(
            side="left", padx=6)
        RBtn(row, "删除", self._delete, kind="danger", height=30, size=9, padx=14).pack(side="right")

        right = tk.Frame(f, bg=P["bg"])
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        self.detail_card = Card(right, "实验详情", "选中左侧记录")
        self.detail_card.grid(row=0, column=0, sticky="nsew")
        db = self.detail_card.body
        self.detail_label = tk.Label(db, text="—", bg=P["panel"], fg=P["text2"],
                                     font=font(9), anchor="w", justify="left", wraplength=560)
        self.detail_label.pack(fill="x")

        act = tk.Frame(db, bg=P["panel"])
        act.pack(fill="x", pady=(10, 0))
        RBtn(act, "重跑", self._rerun, kind="primary", height=30, size=9, padx=12).pack(side="left")
        RBtn(act, "对话测试", self._chat, kind="secondary", height=30, size=9, padx=12).pack(
            side="left", padx=6)
        RBtn(act, "合并导出", self._merge, kind="secondary", height=30, size=9, padx=12).pack(side="left")
        RBtn(act, "导出报告", self._report, kind="ghost", height=30, size=9, padx=12).pack(
            side="left", padx=6)

        cmp_row = tk.Frame(db, bg=P["panel"])
        cmp_row.pack(fill="x", pady=(8, 0))
        tk.Label(cmp_row, text="对比基准", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        self.var_baseline = tk.StringVar(value="（不对比）")
        self.cb_baseline = ttk.Combobox(cmp_row, textvariable=self.var_baseline,
                                        state="readonly", font=font(9), width=30)
        self.cb_baseline.pack(side="left", padx=(8, 0), ipady=2)
        self.cb_baseline.bind("<<ComboboxSelected>>", lambda e: self._set_baseline())

        chart_card = Card(right, "损失曲线", "可叠加对比另一个实验")
        chart_card.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.chart = LineChart(chart_card.body, height=230, bg=P["panel"])
        self.chart.pack(fill="both", expand=True)
        self.chart.set_series([], "选中实验后显示曲线")

        self._runs: list[dict] = []
        self._current: str | None = None
        self._compare: str | None = None

    # ------------------------------------------------------------ 列表
    def on_show(self) -> None:
        self.refresh()
        focus = self.app.shared.pop("focus_run", None)
        if focus:
            for iid in self.table.get_children():
                if iid == focus:
                    self.table.selection_set(iid)
                    break

    def refresh(self) -> None:
        self._runs = self.store.runs
        rows = []
        for r in self._runs:
            rows.append({
                "_id": r.get("id"),
                "name": r.get("name", ""),
                "status": STATUS_TEXT.get(r.get("status"), r.get("status", "")),
                "loss": f"{r['final_loss']:.4f}" if isinstance(r.get("final_loss"), (int, float)) else "-",
                "duration": paths.human_time(r.get("seconds")),
                "created": time.strftime("%m-%d %H:%M", time.localtime(r.get("created", 0))),
                "_tag": r.get("status", "done"),
            })
        self.table.set_rows(rows)
        options = ["（不对比）"] + [
            f"{r.get('name')} · {time.strftime('%m-%d %H:%M', time.localtime(r.get('created', 0)))}"
            for r in self._runs]
        self.cb_baseline.configure(values=options)
        if self.var_baseline.get() not in options:
            self.var_baseline.set("（不对比）")
            self._compare = None
        if rows and not self.table.selection():
            self.table.select_first()

    def _selected_run(self) -> dict | None:
        row = self.table.selected()
        if not row:
            return None
        return self.store.get_run(row["_id"])

    def _detail(self, row) -> None:
        run = self.store.get_run(row["_id"]) if row else None
        if run is None:
            return
        self._current = run["id"]
        cfg = run.get("config") or {}
        bits = [
            f"目标：{run.get('goal') or '（未填写）'}",
            f"基座：{run.get('model')}",
            f"数据：{run.get('dataset')}",
            f"方式：{'全参微调' if run.get('mode') == 'full' else 'LoRA（r=%s, α=%s）' % (cfg.get('lora_r'), cfg.get('lora_alpha'))}",
            f"超参：{cfg.get('epochs')} 轮 · batch {cfg.get('batch_size')}×{cfg.get('grad_accum')} · "
            f"lr {cfg.get('lr')} · max_len {cfg.get('max_len')}",
            f"样本：{run.get('samples') or '-'} · 步数：{run.get('steps_done') or '-'}",
            f"最终 loss：{run.get('final_loss') if run.get('final_loss') is not None else '-'} · "
            f"验证 loss：{run.get('best_eval_loss') if run.get('best_eval_loss') is not None else '-'}",
            f"耗时：{paths.human_time(run.get('seconds'))} · 吞吐 "
            f"{run.get('tokens_per_sec') or '-'} tok/s",
            f"产物：{run.get('out_dir')}",
        ]
        if run.get("error"):
            bits.append(f"错误：{run['error']}")
        self.detail_label.configure(text="\n".join(bits),
                                    fg=P["red"] if run.get("status") == "failed" else P["text2"])
        self._draw_chart()

    # ------------------------------------------------------------ 曲线
    def _load_curve(self, run: dict) -> list[tuple[float, float]]:
        out = Path(run.get("out_dir") or "")
        p = out / "loss_curve.json"
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return [(float(step), float(value)) for step, value in data.get("loss", [])]
            except (OSError, ValueError, TypeError):
                pass
        p2 = out / "studio_events.jsonl"
        pts: list[tuple[float, float]] = []
        if p2.is_file():
            try:
                for line in p2.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("type") == "step" and isinstance(e.get("loss"), (int, float)):
                        pts.append((float(e.get("step", 0)), float(e["loss"])))
            except OSError:
                pass
        return pts

    def _draw_chart(self) -> None:
        run = self.store.get_run(self._current) if self._current else None
        if run is None:
            self.chart.set_series([], "选中实验后显示曲线")
            return
        series = []
        pts = self._load_curve(run)
        if pts:
            series.append((run.get("name", "当前")[:12], "accent", pts))
        if self._compare and self._compare != run["id"]:
            other = self.store.get_run(self._compare)
            if other:
                opt = self._load_curve(other)
                if opt:
                    series.append((other.get("name", "对比")[:12], "purple", opt))
        title = "没有找到 loss 曲线（可能训练未完成）"
        self.chart.set_series(series, title)

    def _set_baseline(self) -> None:
        value = self.var_baseline.get()
        self._compare = None
        if value and value != "（不对比）":
            for r in self._runs:
                if f"{r.get('name')} · {time.strftime('%m-%d %H:%M', time.localtime(r.get('created', 0)))}" == value:
                    self._compare = r["id"]
                    break
        self._draw_chart()

    # ------------------------------------------------------------ 动作
    def _open_dir(self) -> None:
        run = self._selected_run()
        if run is None:
            procs.open_in_explorer(paths.outputs_dir())
            return
        procs.open_in_explorer(run.get("out_dir") or paths.outputs_dir())

    def _rerun(self) -> None:
        run = self._selected_run()
        if run is None:
            return
        cfg = run.get("config") or {}
        plan = {
            "epochs": cfg.get("epochs", 3), "batch_size": cfg.get("batch_size", 2),
            "grad_accum": cfg.get("grad_accum", 2), "lr": cfg.get("lr", 2e-4),
            "max_len": cfg.get("max_len", 256), "lora_r": cfg.get("lora_r", 16),
            "lora_alpha": cfg.get("lora_alpha", 32),
            "lora_dropout": cfg.get("lora_dropout", 0.05),
            "val_ratio": cfg.get("val_ratio", 0.1), "full": run.get("mode") == "full",
        }
        self.app.shared["pending_train"] = {
            "model": run.get("model", ""),
            "dataset": Path(str(run.get("dataset") or "")).name,
            "goal": run.get("goal", ""), "plan": plan,
        }
        self.app.goto("train")
        page = self.app._pages["train"]  # type: ignore[attr-defined]
        page.ensure()
        page.var_name.set((run.get("name") or "rerun") + "-rerun")

    def _chat(self) -> None:
        run = self._selected_run()
        if run is None:
            return
        self.app.shared["chat_selection"] = {
            "model": Path(str(run.get("model") or "")).name,
            "adapters": [Path(str(run.get("out_dir") or "")).name],
        }
        self.app.goto("chat")

    def _merge(self) -> None:
        run = self._selected_run()
        if run is None:
            return
        self.app.shared["export_selection"] = {
            "adapter": str(run.get("out_dir") or ""),
            "model": Path(str(run.get("model") or "")).name,
            "name": (run.get("name") or "merged") + "-merged",
        }
        self.app.goto("export")

    def _report(self) -> None:
        run = self._selected_run()
        if run is None:
            return
        out = Path(run.get("out_dir") or paths.outputs_dir())
        out.mkdir(parents=True, exist_ok=True)
        cfg = run.get("config") or {}
        lines = [
            f"# 实验报告：{run.get('name')}",
            "",
            f"- 状态：{STATUS_TEXT.get(run.get('status'), run.get('status'))}",
            f"- 开始时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run.get('created', 0)))}",
            f"- 训练目标：{run.get('goal') or '（未填写）'}",
            f"- 基座模型：{run.get('model')}",
            f"- 数据文件：{run.get('dataset')}",
            f"- 训练方式：{'全参微调' if run.get('mode') == 'full' else 'LoRA'}",
            f"- 有效样本：{run.get('samples')}",
            f"- 训练步数：{run.get('steps_done')}",
            f"- 超参：epochs={cfg.get('epochs')}, batch={cfg.get('batch_size')}x{cfg.get('grad_accum')}, "
            f"lr={cfg.get('lr')}, max_len={cfg.get('max_len')}, lora_r={cfg.get('lora_r')}, "
            f"lora_alpha={cfg.get('lora_alpha')}",
            f"- 最终 loss：{run.get('final_loss')}",
            f"- 验证 loss：{run.get('best_eval_loss')}",
            f"- 耗时：{paths.human_time(run.get('seconds'))}",
            f"- 吞吐：{run.get('tokens_per_sec')} tokens/s",
            f"- 产物目录：{run.get('out_dir')}",
            "",
            "## 下一步",
            "",
            "1. 在「对话」页加载该产物，用同一批问题对比微调前后的回答。",
            "2. 若效果不足：优先补数据（同分布、去重复），其次调大 epochs 或 lora_r。",
            "3. 若回答变笨/跑题：降低 lr 或减少 epochs，检查数据里是否有脏样本。",
        ]
        p = out / "report.md"
        try:
            p.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            self.toast(f"写报告失败：{exc}", "err")
            return
        self.toast(f"报告已生成：{p.name}", "ok", 3000)
        procs.open_in_explorer(p)

    def _delete(self) -> None:
        run = self._selected_run()
        if run is None:
            self.toast("先选中一个实验", "warn")
            return
        out = run.get("out_dir") or ""
        choice = self.app.confirm3(
            "删除实验",
            f"记录：{run.get('name')}\n产物目录：{out}\n\n"
            "「删记录 + 产物」会同时删除该目录（检查点、适配器都会没有）。",
            "删记录 + 产物", "只删记录")
        if choice is None:
            return
        if choice == "alt":
            self.store.remove_run(run["id"])
            self.toast("记录已删除，产物仍在磁盘上", "ok")
            self.refresh()
            return
        from ...core import models as models_mod

        ok, m = models_mod.delete_dir(Path(out), recycle=bool(self.cfg.get("recycle_delete", True)))
        self.store.remove_run(run["id"])
        self.toast(m, "ok" if ok else "err")
        self.refresh()
