# -*- coding: utf-8 -*-
"""导出页：LoRA 合并、打包、部署说明。"""
from __future__ import annotations

import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ...core import fsutil
from ...core import models as models_mod
from ...core import paths, procs
from ..base import Page
from ..theme import P, font
from ..widgets import Bar, Card, Check, Entry, Labeled, LogView, RBtn, ScrollFrame, Table

EXPORT_KEY = "export"


class ExportPage(Page):
    key = "export"
    group = "实验台"
    title = "导出交付"
    subtitle = "把适配器合并成可直接加载的完整模型"

    def build(self) -> None:
        outer = ScrollFrame(self)
        outer.pack(fill="both", expand=True)
        f = outer.inner
        f.grid_columnconfigure(0, weight=3, uniform="e")
        f.grid_columnconfigure(1, weight=2, uniform="e")

        merge = Card(f, "合并 LoRA", "基座 + 适配器 → 独立模型目录（fp32，可直接部署）")
        merge.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        mb = merge.body

        self.var_model = tk.StringVar()
        self.var_adapter = tk.StringVar()
        self.var_name = tk.StringVar(value="merged")
        self.var_verify = tk.BooleanVar(value=True)

        grid = tk.Frame(mb, bg=P["panel"])
        grid.pack(fill="x")
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        wm = Labeled(grid, "基座模型", "", bg=P["panel"])
        wm.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.cb_model = ttk.Combobox(wm.holder, textvariable=self.var_model, state="readonly",
                                     font=font(10))
        self.cb_model.pack(fill="x", ipady=3)
        wa = Labeled(grid, "适配器", "outputs/ 下的 LoRA", bg=P["panel"])
        wa.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.cb_adapter = ttk.Combobox(wa.holder, textvariable=self.var_adapter, state="readonly",
                                       font=font(10))
        self.cb_adapter.pack(fill="x", ipady=3)

        wn = Labeled(mb, "输出目录名", "写在 outputs/ 下", bg=P["panel"])
        wn.pack(fill="x", pady=(10, 0))
        Entry(wn.holder, textvariable=self.var_name).pack(fill="x", ipady=3)
        self.var_name.trace_add("write", lambda *a: self._update_out())

        self.out_label = tk.Label(mb, text="", bg=P["panel"], fg=P["muted"], font=font(8),
                                  anchor="w", wraplength=520)
        self.out_label.pack(fill="x", pady=(6, 0))

        opts = tk.Frame(mb, bg=P["panel"])
        opts.pack(fill="x", pady=(8, 0))
        Check(opts, "合并后立即加载验证（多花几秒，但能确认模型可用）", self.var_verify,
              bg=P["panel"]).pack(side="left")

        self.bar = Bar(mb, height=8, bg=P["panel"])
        self.bar.pack(fill="x", pady=(12, 4))
        self.state = tk.Label(mb, text="就绪", bg=P["panel"], fg=P["muted"], font=font(9),
                              anchor="w")
        self.state.pack(fill="x")

        act = tk.Frame(mb, bg=P["panel"])
        act.pack(fill="x", pady=(12, 0))
        RBtn(act, "开始合并", self.start, kind="primary", height=34, padx=16).pack(side="left")
        RBtn(act, "打开输出目录", self._open_out, kind="secondary", height=34, padx=12).pack(
            side="left", padx=8)
        RBtn(act, "打包 ZIP", self._zip, kind="ghost", height=34, padx=12).pack(side="left")
        RBtn(act, "写部署说明", self._write_readme, kind="ghost", height=34, padx=12).pack(
            side="left", padx=6)

        self.log = LogView(mb, height=9, max_lines=800, bg=P["panel"])
        self.log.pack(fill="both", expand=True, pady=(12, 0))

        list_card = Card(f, "已有产物", "outputs/ 下的适配器与合并模型")
        list_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        lb = list_card.body
        self.table = Table(lb, [
            ("name", "目录", 190, "w"), ("kind", "类型", 70, "w"),
            ("size", "体积", 80, "e"),
        ], height=12, on_double=self._open_row)
        self.table.pack(fill="both", expand=True)
        row = tk.Frame(lb, bg=P["panel"])
        row.pack(fill="x", pady=(10, 0))
        RBtn(row, "刷新", self.refresh, kind="secondary", height=30, size=9, padx=12).pack(side="left")
        RBtn(row, "打开", self._open_row_selected, kind="ghost", height=30, size=9,
             padx=12).pack(side="left", padx=6)
        RBtn(row, "删除", self._delete, kind="danger", height=30, size=9, padx=12).pack(side="right")

        self._last_out: Path | None = None
        self._verify_enabled = True
        self._verify_ok: bool | None = None
        self.app.bind_job(EXPORT_KEY, self._on_event)

    # ------------------------------------------------------------ 数据
    def on_show(self) -> None:
        models = models_mod.scan_models()
        names = [m.name for m in models]
        self.cb_model.configure(values=names)
        if names and self.var_model.get() not in names:
            self.var_model.set(models_mod.default_model_name(models, self.cfg.get("last.model", "")))
        outs = models_mod.scan_outputs()
        adapters = [m.name for m in outs if m.kind == "adapter"]
        self.cb_adapter.configure(values=adapters)
        if adapters and self.var_adapter.get() not in adapters:
            self.var_adapter.set(adapters[0])
        sel = self.app.shared.pop("export_selection", None)
        if sel:
            if sel.get("model"):
                self.var_model.set(sel["model"])
            if sel.get("adapter"):
                self.var_adapter.set(Path(sel["adapter"]).name)
            if sel.get("name"):
                self.var_name.set(sel["name"])
        self._update_out()
        self.refresh()

    def _update_out(self) -> None:
        name = fsutil.sanitize_component(self.var_name.get() or "merged", "merged")
        try:
            self._last_out = fsutil.safe_child(paths.outputs_dir(), name)
            self.out_label.configure(text=f"输出目录：{self._last_out}", fg=P["muted"])
        except fsutil.PathSafetyError as exc:
            self._last_out = None
            self.out_label.configure(text=f"输出目录名不安全：{exc}", fg=P["red"])

    def refresh(self) -> None:
        rows = []
        for m in models_mod.scan_outputs():
            rows.append({"_id": str(m.path), "name": m.name,
                         "kind": "适配器" if m.kind == "adapter" else "完整模型",
                         "size": paths.human_size(m.size)})
        self.table.set_rows(rows)

    def _open_row(self, row) -> None:
        if row:
            procs.open_in_explorer(Path(row["_id"]))

    def _open_row_selected(self) -> None:
        row = self.table.selected()
        self._open_row(row) if row else procs.open_in_explorer(paths.outputs_dir())

    def _open_out(self) -> None:
        if self._last_out and self._last_out.exists():
            procs.open_in_explorer(self._last_out)
        else:
            procs.open_in_explorer(paths.outputs_dir())

    def _delete(self) -> None:
        row = self.table.selected()
        if not row:
            self.toast("先选中一个产物", "warn")
            return
        p = Path(row["_id"])
        if not self.app.confirm("删除产物", f"删除 {p.name}（{row['size']}）？"
                                f"{'会进回收站。' if self.cfg.get('recycle_delete') else '不可恢复。'}",
                                "删除", True):
            return
        ok, msg = models_mod.delete_dir(p, recycle=bool(self.cfg.get("recycle_delete", True)))
        self.toast(msg, "ok" if ok else "err")
        self.refresh()

    # ------------------------------------------------------------ 合并
    def start(self) -> None:
        if self.app.jobs.running(EXPORT_KEY):
            self.toast("合并正在进行", "warn")
            return
        model = models_mod.find_model(self.var_model.get())
        adapter = next((m for m in models_mod.scan_outputs()
                        if m.name == self.var_adapter.get()), None)
        if model is None or adapter is None:
            self.toast("需要同时选择基座与适配器", "warn")
            return
        try:
            out = fsutil.safe_child(
                paths.outputs_dir(),
                fsutil.sanitize_component(self.var_name.get().strip() or "merged", "merged"))
        except fsutil.PathSafetyError as exc:
            self.toast(f"输出目录名不安全：{exc}", "err", 4600)
            return
        if out.exists() and any(out.iterdir()):
            if not self.app.confirm("覆盖输出", f"{out} 已存在且非空，继续会覆盖同名文件。", "覆盖", True):
                return
        self._last_out = out
        self._verify_enabled = bool(self.var_verify.get())
        self._verify_ok = None
        self.log.clear()
        self.log.append(f"合并 {adapter.path.name} → {out}", "cmd")
        self.bar.set(0.05)
        self.state.configure(text="启动合并…", fg=P["text2"])
        cfg = {"model": str(model.path), "adapter": str(adapter.path), "out": str(out),
               "verify": bool(self.var_verify.get())}
        if not self.app.spawn_worker(EXPORT_KEY, "merge_worker.py", cfg, label="合并"):
            self.state.configure(text="启动失败", fg=P["red"])

    def _on_event(self, evt: dict) -> None:
        t = evt.get("type")
        if t == "stage":
            msg = str(evt.get("message", ""))
            self.state.configure(text=msg, fg=P["text2"])
            self.log.append(f"[阶段] {msg}", "info")
            name = evt.get("name")
            self.bar.set({"tokenizer": 0.1, "load": 0.35, "merge": 0.55, "save": 0.8,
                          "verify": 0.9}.get(name, 0.5))
        elif t == "merged":
            self.log.append(f"[合并] 写出 {paths.human_size(evt.get('bytes', 0))}", "ok")
            self.bar.set(0.85)
        elif t == "verified":
            self._verify_ok = bool(evt.get("ok"))
            if evt.get("ok"):
                self.log.append(f"[验证] 加载成功，示例输出：{evt.get('sample')}", "ok")
                self.state.configure(text="合并完成并通过加载验证", fg=P["green"])
            else:
                self.log.append(f"[验证] 失败：{evt.get('message')}", "err")
                self.state.configure(text="合并文件已写出，但验证失败（不算成功）", fg=P["red"])
            self.bar.set(1)
        elif t == "done":
            status = str(evt.get("status") or "done")
            self.bar.set(1)
            if status != "done":
                self.state.configure(
                    text=str(evt.get("message") or "合并失败（文件可能已写出，但验证未通过）"),
                    fg=P["red"])
                self.log.append(f"[失败] {evt.get('message') or status}", "err")
                self.toast(str(evt.get("message") or "合并失败"), "err", 6000)
            elif self._verify_enabled and self._verify_ok is False:
                self.state.configure(text="合并文件已写出，但验证失败（不算成功）", fg=P["red"])
                self.toast("合并验证失败", "err", 5200)
            else:
                self.state.configure(text=f"完成：{evt.get('out')}", fg=P["green"])
                self.toast("合并完成" + ("并通过加载验证" if self._verify_ok else ""), "ok")
            self.refresh()
        elif t == "error":
            self.state.configure(text=str(evt.get("message")), fg=P["red"])
            self.log.append(f"[错误] {evt.get('message')}", "err")
            self.toast(str(evt.get("message")), "err", 5000)
        elif t == "_exit":
            if evt.get("code") not in (0, None):
                self.state.configure(text=f"进程退出 code={evt.get('code')}", fg=P["red"])

    # ------------------------------------------------------------ 打包 / 说明
    def _zip(self) -> None:
        out = self._last_out
        if not out or not out.is_dir():
            self.toast("先完成一次合并", "warn")
            return
        self.toast("正在打包…", "info", 1600)

        def work() -> None:
            try:
                archive = shutil.make_archive(str(out), "zip", root_dir=str(out))
                self.app.post(lambda: self._zip_done(Path(archive), ""))
            except OSError as exc:
                msg = str(exc)
                self.app.post(lambda: self._zip_done(None, msg))

        threading.Thread(target=work, daemon=True).start()

    def _zip_done(self, path: Path | None, err: str) -> None:
        if err or path is None:
            self.toast(f"打包失败：{err}", "err")
            return
        size = paths.human_size(path.stat().st_size)
        self.toast(f"已打包 {path.name}（{size}）", "ok", 3200)
        procs.open_in_explorer(path)

    def _write_readme(self) -> None:
        out = self._last_out
        if not out or not out.is_dir():
            self.toast("先完成一次合并", "warn")
            return
        name = out.name
        text = f"""# {name}

由「{paths.APP_NAME} {paths.VERSION}」合并产出：基座 + LoRA 适配器的完整模型（fp32）。

## 加载方式

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(r"{out}", dtype="float32")
tok = AutoTokenizer.from_pretrained(r"{out}")
msgs = [{{"role": "user", "content": "你好"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
out_ids = model.generate(ids, max_new_tokens=128, do_sample=False)
print(tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True))
```

## 说明

- 目录内 `config.json` / `model.safetensors` / tokenizer 文件齐全，`from_pretrained` 可直接加载。
- 这是 fp32 权重，体积约为参数量 ×4 字节；如需更小可自行做 fp16/量化转换。
- 训练参数与指标见同实验目录下的 `train_meta.json`。
"""
        p = out / "README_DEPLOY.md"
        try:
            p.write_text(text, encoding="utf-8")
        except OSError as exc:
            self.toast(f"写入失败：{exc}", "err")
            return
        self.toast("部署说明已写入 README_DEPLOY.md", "ok", 2600)
