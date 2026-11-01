# -*- coding: utf-8 -*-
"""工具页：环境自检、空间占用、检查点清理、日志。"""
from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ...core import models as models_mod
from ...core import paths, procs
from ..base import Page
from ..theme import P, font
from ..widgets import Card, LogView, RBtn, ScrollFrame, Table

DOCTOR_KEY = "doctor"


class ToolsPage(Page):
    key = "tools"
    group = "系统"
    title = "工具"
    subtitle = "环境自检、空间清理、日志查看"

    def build(self) -> None:
        outer = ScrollFrame(self)
        outer.pack(fill="both", expand=True)
        f = outer.inner
        f.grid_columnconfigure(0, weight=1, uniform="t")
        f.grid_columnconfigure(1, weight=1, uniform="t")

        doc = Card(f, "环境自检", "真实加载模型并生成 12 个 token，验证整条链路")
        doc.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        db = doc.body
        row = tk.Frame(db, bg=P["panel"])
        row.pack(fill="x")
        self.var_model = tk.StringVar()
        self.cb_model = ttk.Combobox(row, textvariable=self.var_model, state="readonly",
                                     font=font(9), width=28)
        self.cb_model.pack(side="left", ipady=3)
        RBtn(row, "开始自检", self.run_doctor, kind="primary", height=32, padx=14).pack(
            side="left", padx=8)
        self.doc_state = tk.Label(row, text="未运行", bg=P["panel"], fg=P["muted"], font=font(9))
        self.doc_state.pack(side="left")

        self.doc_table = Table(db, [
            ("name", "检查项", 130, "w"), ("ok", "结果", 70, "w"),
            ("detail", "详情", 380, "w"), ("ms", "耗时", 70, "e"),
        ], height=7)
        self.doc_table.pack(fill="both", expand=True, pady=(10, 0))
        self.doc_table.tag_configure("failed", foreground=P["red"])

        space = Card(f, "空间占用", "点「重新统计」刷新；合并/训练产物都在 outputs/")
        space.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        sb = space.body
        self.space_table = Table(sb, [
            ("name", "目录", 200, "w"), ("size", "占用", 90, "e"), ("files", "文件数", 80, "e"),
        ], height=6, on_double=self._open_space)
        self.space_table.pack(fill="both", expand=True)
        srow = tk.Frame(sb, bg=P["panel"])
        srow.pack(fill="x", pady=(10, 0))
        RBtn(srow, "重新统计", self.scan_space, kind="secondary", height=30, size=9,
             padx=12).pack(side="left")
        RBtn(srow, "清理 HF 缓存", self._clear_cache, kind="ghost", height=30, size=9,
             padx=12).pack(side="left", padx=6)
        RBtn(srow, "打开目录", lambda: procs.open_in_explorer(paths.root()), kind="ghost",
             height=30, size=9, padx=12).pack(side="left")

        ck = Card(f, "检查点", "训练中断后留下的 checkpoint-*，确认不再续训再删")
        ck.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(14, 0))
        cb = ck.body
        self.ck_table = Table(cb, [
            ("name", "目录", 320, "w"), ("size", "占用", 90, "e"),
            ("mtime", "修改时间", 130, "w"),
        ], height=6)
        self.ck_table.pack(fill="both", expand=True)
        crow = tk.Frame(cb, bg=P["panel"])
        crow.pack(fill="x", pady=(10, 0))
        RBtn(crow, "刷新", self.scan_checkpoints, kind="secondary", height=30, size=9,
             padx=12).pack(side="left")
        RBtn(crow, "删除选中检查点", self._delete_checkpoint, kind="danger", height=30, size=9,
             padx=12).pack(side="left", padx=6)

        logs = Card(f, "日志", "logs/ 下的对话记录与运行日志")
        logs.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(14, 0))
        lb = logs.body
        lrow = tk.Frame(lb, bg=P["panel"])
        lrow.pack(fill="x")
        self.log_list = Table(lrow, [("name", "文件", 260, "w"), ("size", "大小", 80, "e"),
                                     ("mtime", "时间", 140, "w")], height=4,
                              on_double=self._view_log)
        self.log_list.pack(fill="x")
        lbtn = tk.Frame(lb, bg=P["panel"])
        lbtn.pack(fill="x", pady=(8, 0))
        RBtn(lbtn, "刷新", self.scan_logs, kind="secondary", height=28, size=9, padx=10).pack(side="left")
        RBtn(lbtn, "查看", lambda: self._view_log(self.log_list.selected()), kind="ghost",
             height=28, size=9, padx=10).pack(side="left", padx=6)
        self.log_view = LogView(lb, height=8, max_lines=1500, bg=P["panel"])
        self.log_view.pack(fill="both", expand=True, pady=(10, 0))

        self.app.bind_job(DOCTOR_KEY, self._on_doctor)

    # ------------------------------------------------------------ 生命周期
    def on_show(self) -> None:
        models = models_mod.scan_models()
        names = [m.name for m in models]
        self.cb_model.configure(values=names)
        if names and self.var_model.get() not in names:
            self.var_model.set(models_mod.default_model_name(models, self.cfg.get("last.model", "")))
        self.scan_space()
        self.scan_checkpoints()
        self.scan_logs()

    # ------------------------------------------------------------ 自检
    def run_doctor(self) -> None:
        if self.app.jobs.running(DOCTOR_KEY):
            self.toast("自检正在进行", "warn")
            return
        model = self.var_model.get()
        if not model:
            self.toast("先选一个模型（models 目录为空就先去下载）", "warn")
            return
        self.doc_table.set_rows([])
        self.doc_state.configure(text="运行中…", fg=P["amber"])
        cfg = {"model": str((paths.models_dir() / model).resolve())}
        if not self.app.spawn_worker(DOCTOR_KEY, "doctor_worker.py", cfg, label="自检"):
            self.doc_state.configure(text="启动失败", fg=P["red"])

    def _on_doctor(self, evt: dict) -> None:
        t = evt.get("type")
        if t == "check":
            rows = [r for r in self.doc_table._rows]
            rows.append({
                "_id": f"{evt.get('name')}-{len(rows)}",
                "name": evt.get("name"), "ok": "通过" if evt.get("ok") else "失败",
                "detail": evt.get("detail", ""), "ms": f"{evt.get('ms', 0)}ms",
                "_tag": "" if evt.get("ok") else "failed",
            })
            self.doc_table.set_rows(rows)
        elif t == "done":
            failed = [r for r in self.doc_table._rows if r.get("ok") == "失败"]
            if failed:
                self.doc_state.configure(text=f"{len(failed)} 项失败", fg=P["red"])
            else:
                self.doc_state.configure(text="全部通过", fg=P["green"])
                self.toast("环境自检全部通过", "ok")
        elif t == "_exit":
            if self.doc_state.cget("text") == "运行中…":
                self.doc_state.configure(text="进程退出（见上方结果）", fg=P["muted"])

    # ------------------------------------------------------------ 空间
    def scan_space(self) -> None:
        def work() -> None:
            targets = [
                ("models（基座模型）", paths.models_dir()),
                ("outputs（产物/检查点）", paths.outputs_dir()),
                ("data（训练数据）", paths.data_dir()),
                (".cache（HF 缓存）", paths.root() / ".cache"),
                ("logs（日志）", paths.logs_dir()),
            ]
            rows = []
            for name, p in targets:
                size, files = 0, 0
                if p.is_dir():
                    for dirpath, _dirnames, filenames in os.walk(p):
                        for fn in filenames:
                            try:
                                size += os.path.getsize(os.path.join(dirpath, fn))
                                files += 1
                            except OSError:
                                continue
                rows.append({"_id": str(p), "name": name,
                             "size": paths.human_size(size), "files": files})
            self.app.post(lambda: self.space_table.set_rows(rows))

        threading.Thread(target=work, daemon=True).start()

    def _open_space(self, row) -> None:
        if row:
            procs.open_in_explorer(Path(row["_id"]))

    def _clear_cache(self) -> None:
        cache = paths.root() / ".cache" / "huggingface"
        if not cache.is_dir():
            self.toast("没有 HuggingFace 缓存目录", "info")
            return
        if self.app.jobs.running("download"):
            self.toast("正在下载模型，先等下载结束再清缓存", "warn", 3600)
            return
        if not self.app.confirm("清理缓存", f"删除 {cache} 下的下载缓存（约等于释放几十到几百 MB）。\n"
                                          "已下载到 models/ 的模型不受影响。", "清理", True):
            return

        def work() -> None:
            import shutil

            err = ""
            try:
                shutil.rmtree(cache, ignore_errors=False)
            except OSError as exc:
                err = str(exc)
            self.app.post(lambda: self._cache_done(err))

        threading.Thread(target=work, daemon=True).start()

    def _cache_done(self, err: str) -> None:
        if err:
            self.toast(f"清理失败（可能有进程占用）：{err}", "err", 4200)
        else:
            self.toast("缓存已清理", "ok")
        self.scan_space()

    # ------------------------------------------------------------ 检查点
    def scan_checkpoints(self) -> None:
        rows = []
        out = paths.outputs_dir()
        if out.is_dir():
            for d in sorted(out.rglob("checkpoint-*")):
                if not d.is_dir():
                    continue
                try:
                    st = d.stat()
                except OSError:
                    continue
                size = models_mod.dir_size(d)
                rows.append({"_id": str(d), "name": str(d.relative_to(out)),
                             "size": paths.human_size(size),
                             "mtime": time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime))})
        self.ck_table.set_rows(rows)

    def _delete_checkpoint(self) -> None:
        row = self.ck_table.selected()
        if not row:
            self.toast("先选中一个检查点", "warn")
            return
        if not self.app.confirm("删除检查点", f"删除 {row['name']}（{row['size']}）？\n"
                                           "删掉后该轮次无法续训。", "删除", True):
            return
        ok, msg = models_mod.delete_dir(Path(row["_id"]),
                                        recycle=bool(self.cfg.get("recycle_delete", True)))
        self.toast(msg, "ok" if ok else "err")
        self.scan_checkpoints()
        self.scan_space()

    # ------------------------------------------------------------ 日志
    def scan_logs(self) -> None:
        rows = []
        for d in (paths.logs_dir(), paths.state_dir()):
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*")):
                if not f.is_file() or f.suffix.lower() not in (".log", ".md", ".txt", ".jsonl"):
                    continue
                try:
                    st = f.stat()
                except OSError:
                    continue
                rows.append({"_id": str(f), "name": f.name, "size": paths.human_size(st.st_size),
                             "mtime": time.strftime("%m-%d %H:%M", time.localtime(st.st_mtime))})
        rows.sort(key=lambda r: r["mtime"], reverse=True)
        self.log_list.set_rows(rows)

    def _view_log(self, row) -> None:
        if not row:
            return
        p = Path(row["_id"])
        self.log_view.clear()
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self.log_view.append(f"读取失败：{exc}", "err")
            return
        lines = text.splitlines()
        for line in lines[-600:]:
            self.log_view.append(line, "log")
        if len(lines) > 600:
            self.log_view.append(f"……共 {len(lines)} 行，已显示最后 600 行", "warn")
