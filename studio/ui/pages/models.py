# -*- coding: utf-8 -*-
"""模型页：本地模型管理 + 镜像下载 + 仓库搜索。"""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from ...core import fsutil, hubclient, models as models_mod, paths, procs, trust
from ..base import Page
from ..theme import P, font
from ..widgets import Bar, Card, Check, Entry, RBtn, ScrollFrame, Table

DL_KEY = "download"


class ModelsPage(Page):
    key = "models"
    group = "实验台"
    title = "模型库"
    subtitle = "本地模型清单、镜像下载与仓库搜索"

    def build(self) -> None:
        outer = ScrollFrame(self)
        outer.pack(fill="both", expand=True)
        f = outer.inner
        f.grid_columnconfigure(0, weight=3, uniform="m")
        f.grid_columnconfigure(1, weight=2, uniform="m")

        # ---------------- 本地模型
        local = Card(f, "本地模型", "扫描 models/ 目录；适配器与合并产物在 outputs/ 里管理")
        local.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        lb = local.body
        self.table = Table(lb, [
            ("name", "名称", 190, "w"), ("params", "参数", 80, "e"),
            ("size", "体积", 80, "e"), ("arch", "架构", 140, "w"),
            ("state", "完整性", 90, "w"),
        ], height=7, on_select=self._on_select)
        self.table.pack(fill="both", expand=True)

        self.detail = tk.Label(lb, text="选中一行查看详情", bg=P["panel"], fg=P["muted"],
                               font=font(9), anchor="w", justify="left", wraplength=560)
        self.detail.pack(fill="x", pady=(8, 0))

        btns = tk.Frame(lb, bg=P["panel"])
        btns.pack(fill="x", pady=(10, 0))
        RBtn(btns, "刷新", self.refresh, kind="secondary", height=30, size=9, padx=12).pack(side="left")
        RBtn(btns, "导入目录", self._import, kind="secondary", height=30, size=9,
             padx=12).pack(side="left", padx=6)
        RBtn(btns, "校验", self._validate, kind="secondary", height=30, size=9,
             padx=12).pack(side="left", padx=6)
        RBtn(btns, "打开目录", self._open, kind="ghost", height=30, size=9, padx=12).pack(side="left")
        RBtn(btns, "删除", self._delete, kind="danger", height=30, size=9, padx=14).pack(side="right")

        # ---------------- 下载
        dl = Card(f, "下载模型", "默认走 hf-mirror 镜像；中断后重新点下载会自动续传")
        dl.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(14, 0))
        db = dl.body

        line = tk.Frame(db, bg=P["panel"])
        line.pack(fill="x")
        self.repo_entry = Entry(line, width=48, placeholder="组织/模型名，例如 Qwen/Qwen2.5-0.5B-Instruct")
        self.repo_entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.btn_dl = RBtn(line, "开始下载", self._start_download, kind="primary", height=32,
                           padx=14)
        self.btn_dl.pack(side="left", padx=(8, 0))
        self.btn_cancel = RBtn(line, "取消", self._cancel_download, kind="ghost", height=32,
                               padx=12)
        self.btn_cancel.pack(side="left", padx=(6, 0))
        self.btn_cancel.set_enabled(False)

        self.var_allow_code = tk.BooleanVar(value=False)
        Check(db, "允许下载模型自带代码（*.py 会在加载时执行，仅限明确信任且固定版本）",
              self.var_allow_code, bg=P["panel"]).pack(anchor="w", pady=(10, 2))
        tk.Label(db, text="默认跳过所有 *.py；MiniMind-O 是内置可信例外（代码哈希固定）。"
                         "权重 SHA-256 校验通过不代表仓库代码可信。",
                 bg=P["panel"], fg=P["muted"], font=font(8), anchor="w", justify="left",
                 wraplength=600).pack(fill="x")

        self.dl_bar = Bar(db, height=10, bg=P["panel"], show_text=False)
        self.dl_bar.pack(fill="x", pady=(12, 4))
        self.dl_text = tk.Label(db, text="尚未开始", bg=P["panel"], fg=P["muted"], font=font(9),
                                anchor="w", justify="left", wraplength=600)
        self.dl_text.pack(fill="x")

        # ---------------- 推荐
        rec = Card(f, "推荐清单", "双击一行填到下载框，或选中后点下载")
        rec.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        rb = rec.body
        self.rec_table = Table(rb, [
            ("repo", "仓库", 210, "w"), ("params", "规模", 70, "e"),
            ("lang", "语言", 70, "w"), ("why", "说明", 260, "w"),
        ], height=7, on_double=self._fill_repo)
        self.rec_table.pack(fill="both", expand=True)
        rr = tk.Frame(rb, bg=P["panel"])
        rr.pack(fill="x", pady=(10, 0))
        RBtn(rr, "填入下载框", lambda: self._fill_repo(self.rec_table.selected()),
             kind="secondary", height=30, size=9, padx=12).pack(side="left")
        RBtn(rr, "下载选中", self._download_selected_recommend, kind="primary", height=30,
             size=9, padx=12).pack(side="left", padx=6)

        # ---------------- 搜索
        search = Card(f, "在线搜索", f"通过 {self.cfg.get('endpoint')} 的 API 查询")
        search.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(14, 0))
        sb = search.body
        srow = tk.Frame(sb, bg=P["panel"])
        srow.pack(fill="x")
        self.search_entry = Entry(srow, width=32, placeholder="关键词，例如 qwen 0.5b instruct")
        self.search_entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.search_entry.bind("<Return>", lambda e: self._search())
        RBtn(srow, "搜索", self._search, kind="secondary", height=32, padx=12).pack(
            side="left", padx=(8, 0))
        self.search_table = Table(sb, [
            ("id", "仓库", 230, "w"), ("pipeline", "类型", 90, "w"),
            ("downloads", "下载量", 80, "e"), ("updated", "更新", 80, "w"),
        ], height=6, on_double=self._fill_repo)
        self.search_table.pack(fill="both", expand=True, pady=(10, 0))
        sbtn = tk.Frame(sb, bg=P["panel"])
        sbtn.pack(fill="x", pady=(8, 0))
        RBtn(sbtn, "下载选中", self._download_selected_search, kind="primary", height=30,
             size=9, padx=12).pack(side="left")
        tk.Label(sbtn, text="双击结果也能填进下载框", bg=P["panel"], fg=P["muted"],
                 font=font(8)).pack(side="left", padx=8)

        self._models: list[models_mod.ModelInfo] = []
        self._dl_dest: Path | None = None
        self._dl_total = 0

        self.app.bind_job(DL_KEY, self._on_event)

    # ------------------------------------------------------------ 列表
    def on_show(self) -> None:
        self.refresh()
        if not self.rec_table.get_children():
            self._fill_recommend()
        if not self.search_table.get_children():
            self.search_entry.set_value("qwen2.5-0.5b-instruct")
            self._search()

    def refresh(self) -> None:
        self._models = models_mod.scan_models()
        rows = []
        for m in self._models:
            rows.append({
                "_id": str(m.path),
                "name": m.name,
                "params": models_mod.fmt_params(m.params),
                "size": paths.human_size(m.size),
                "arch": m.arch or "-",
                "state": "完整" if m.complete else "缺 " + ",".join(m.missing),
                "_tag": "" if m.complete else "failed",
            })
        self.table.set_rows(rows)
        if not rows:
            self.detail.configure(text="models 目录是空的。用下面的推荐清单或下载框拉一个基座进来。",
                                  fg=P["amber"])

    def _on_select(self, row) -> None:
        if not row:
            return
        m = next((x for x in self._models if str(x.path) == row["_id"]), None)
        if m is None:
            return
        bits = [f"路径：{m.path}", f"体积：{paths.human_size(m.size)}"]
        if m.params:
            bits.append(f"参数：{m.params/1e6:.1f}M")
        if m.dtype:
            bits.append(f"精度：{m.dtype}")
        if m.layers:
            bits.append(f"层数：{m.layers}")
        if m.hidden:
            bits.append(f"隐含层：{m.hidden}")
        if m.vocab:
            bits.append(f"词表：{m.vocab}")
        bits.append(f"CPU 训练建议：{m.trainable_hint}")
        if m.note:
            bits.append(f"适配器基座：{m.note}")
        self.detail.configure(text=" · ".join(bits), fg=P["text2"])

    def _validate(self) -> None:
        row = self.table.selected()
        if not row:
            self.toast("先选中一个模型", "warn")
            return
        m = models_mod.inspect(Path(row["_id"]))
        if m.complete:
            self.toast(f"{m.name} 校验通过：{models_mod.fmt_params(m.params)} 参数，{paths.human_size(m.size)}", "ok", 3200)
        else:
            self.toast(f"{m.name} 缺少：{', '.join(m.missing)}", "err", 4200)

    def _open(self) -> None:
        row = self.table.selected()
        if not row:
            procs.open_in_explorer(paths.models_dir())
            return
        procs.open_in_explorer(Path(row["_id"]))

    def _delete(self) -> None:
        row = self.table.selected()
        if not row:
            self.toast("先选中要删除的模型", "warn")
            return
        p = Path(row["_id"])
        recycle = bool(self.cfg.get("recycle_delete", True))
        if not self.app.confirm("删除模型", f"将删除 {p.name}（{row['size']}）。\n"
                                        f"{'会移入回收站，可撤销。' if recycle else '将永久删除，无法恢复。'}",
                                "删除", True):
            return
        ok, msg = models_mod.delete_dir(p, recycle=recycle)
        self.toast(msg, "ok" if ok else "err", 3200)
        self.refresh()

    def _import(self) -> None:
        folder = filedialog.askdirectory(title="选择 HuggingFace 模型目录（含 config.json）")
        if not folder:
            return
        target, msg = models_mod.import_model(Path(folder))
        self.toast(msg, "ok" if target else "err", 4000)
        self.refresh()

    # ------------------------------------------------------------ 推荐 / 搜索
    def _fill_recommend(self) -> None:
        rows = []
        for item in hubclient.curated():
            rows.append({"_id": item["repo"], "repo": item["repo"], "params": item["params"],
                         "lang": item["lang"], "why": item["why"]})
        self.rec_table.set_rows(rows)

    def _fill_repo(self, row) -> None:
        if not row:
            return
        repo = row.get("id") or row.get("repo") or ""
        if repo:
            self.repo_entry.set_value(repo)

    def _download_selected_recommend(self) -> None:
        row = self.rec_table.selected()
        if not row:
            self.toast("先选中一行", "warn")
            return
        self.repo_entry.set_value(row["repo"])
        self._start_download()

    def _download_selected_search(self) -> None:
        row = self.search_table.selected()
        if not row:
            self.toast("先选中一个搜索结果", "warn")
            return
        self.repo_entry.set_value(row["id"])
        self._start_download()

    def _search(self) -> None:
        q = self.search_entry.value()
        if not q:
            self.toast("输入关键词再搜索", "warn")
            return
        self.toast(f"正在搜索 {q} …", "info", 1600)

        def work() -> None:
            try:
                res = hubclient.search(q, self.cfg.get("endpoint"), limit=25)
                self.app.post(lambda: self._search_done(res, ""))
            except hubclient.HubError as exc:
                msg = str(exc)
                self.app.post(lambda: self._search_done([], msg))
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                self.app.post(lambda: self._search_done([], msg))

        threading.Thread(target=work, daemon=True).start()

    def _search_done(self, res: list[dict], err: str) -> None:
        if err:
            self.toast(f"搜索失败：{err}", "err", 4200)
            return
        rows = [{"_id": r["id"], "id": r["id"], "pipeline": r.get("pipeline") or "-",
                 "downloads": f"{r['downloads']:,}", "updated": r["updated"]} for r in res]
        self.search_table.set_rows(rows)
        self.toast(f"找到 {len(rows)} 个仓库", "ok", 1800)

    # ------------------------------------------------------------ 下载
    def _start_download(self) -> None:
        repo = self.repo_entry.value().strip()
        if not repo or repo.count("/") != 1:
            self.toast("仓库名要写成 组织/模型名", "warn")
            return
        try:
            owner, name = repo.split("/")
            fsutil.validate_component(owner, max_len=80)
            fsutil.validate_component(name, max_len=80)
        except fsutil.PathSafetyError as exc:
            self.toast(f"仓库名不安全：{exc}", "err", 4600)
            return
        if self.app.jobs.running(DL_KEY):
            self.toast("已有下载任务在跑，先取消或等它结束", "warn")
            return
        builtin = trust.builtin_repo(repo)
        allow_code = bool(self.var_allow_code.get()) and not builtin
        revision = "main"
        if allow_code:
            if not self.app.confirm(
                    "信任并下载模型代码？",
                    f"{repo} 的 *.py 会在加载模型时以当前用户权限执行，等同于运行第三方代码。\n"
                    "只有在你确认来源可信时才继续。必须填写完整的 40 位 commit SHA（不接受 tag）。",
                    "我理解风险，继续", True):
                return
            revision = self.app.ask_text("固定提交", "完整的 40 位 commit SHA（不能用 main / tag）", "") or ""
            try:
                revision = trust.validate_revision(revision)
            except trust.TrustError as exc:
                self.toast(str(exc), "warn", 4600)
                return
        try:
            dest = fsutil.safe_child(paths.models_dir(), name, max_len=80)
        except fsutil.PathSafetyError as exc:
            self.toast(f"目标目录不安全：{exc}", "err", 4600)
            return
        self._dl_dest = dest
        self._dl_total = 0
        cfg = {
            "repo": repo, "dest": str(dest), "endpoint": self.cfg.get("endpoint"),
            "verify_sha": True, "cancel_flag": str(dest / ".cancel"),
            "revision": revision, "allow_code": allow_code,
        }
        self.dl_bar.set(0)
        self.dl_text.configure(
            text=f"准备下载 {repo} → {dest}（revision {revision}，"
                 f"{'允许代码' if allow_code else '跳过 .py'}）", fg=P["text2"])
        if self.app.spawn_worker(DL_KEY, "download_worker.py", cfg, label="下载",
                                 stop_file=dest / ".cancel"):
            self.btn_cancel.set_enabled(True)
            self.btn_dl.set_enabled(False)
            self.app.set_status(f"下载 {repo}")

    def _cancel_download(self) -> None:
        job = self.app.jobs.get(DL_KEY)
        if job and job.running:
            job.stop()
            self.dl_text.configure(text="已请求取消，正在断开连接…", fg=P["amber"])

    def _on_event(self, evt: dict) -> None:
        t = evt.get("type")
        if t == "list":
            self._dl_total = int(evt.get("total") or 0)
            self.dl_text.configure(
                text=f"{evt.get('repo')}：{evt.get('files')} 个文件，共 {paths.human_size(self._dl_total)}",
                fg=P["text2"])
        elif t == "file":
            self.dl_text.configure(
                text=f"[{evt.get('index')}/{evt.get('count')}] {evt.get('name')} · "
                     f"{paths.human_size(evt.get('size'))}", fg=P["text2"])
        elif t == "progress":
            done, total = evt.get("done", 0), evt.get("total", 1)
            if total:
                self.dl_bar.set(done / total)
            eta = evt.get("eta")
            self.dl_text.configure(
                text=f"{paths.human_size(done)} / {paths.human_size(total)} · "
                     f"{paths.human_size(evt.get('speed', 0))}/s · 剩余 {paths.human_time(eta) if eta else '—'}"
                     f" · {evt.get('name', '')}", fg=P["text2"])
        elif t == "file_skip":
            self.log(f"已存在，跳过 {evt.get('name')}", "log")
        elif t == "stage":
            self.dl_text.configure(text=str(evt.get("message", "")), fg=P["muted"])
        elif t == "warn":
            self.log(str(evt.get("message", "")), "warn")
        elif t == "code_policy":
            self.log(f"代码信任已记录：{evt.get('repo')} @ {evt.get('revision')} "
                     f"({len(evt.get('files') or [])} 个文件，内置={evt.get('builtin')})", "ok")
        elif t == "error":
            self.dl_text.configure(text=str(evt.get("message", "")), fg=P["red"])
            self.toast(str(evt.get("message", "")), "err", 5000)
            self._finish()
        elif t == "cancelled":
            self.dl_text.configure(text="下载已取消（已下载部分保留，下次续传）", fg=P["amber"])
            self.toast("下载已取消", "warn")
            self._finish()
        elif t == "done":
            self.dl_bar.set(1)
            self.dl_text.configure(
                text=f"完成：{paths.human_size(evt.get('bytes', 0))} → {evt.get('dest')}", fg=P["green"])
            self.toast("模型下载完成，已加入本地清单", "ok", 3600)
            self._finish()
            self.refresh()
        elif t == "_exit":
            self._finish()

    def _finish(self) -> None:
        self.btn_cancel.set_enabled(False)
        self.btn_dl.set_enabled(True)
        self.app.set_job_status("空闲", False)
        flag = (self._dl_dest or Path(".")) / ".cancel"
        try:
            if flag.is_file():
                flag.unlink()
        except OSError:
            pass
