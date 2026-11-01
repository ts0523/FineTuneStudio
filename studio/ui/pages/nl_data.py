# -*- coding: utf-8 -*-
"""数据层 · 自然语言造数：写一句需求，AI 直接产出训练集 + 反推训练参数。"""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ...core import ai, dataset as ds, fsutil, models as models_mod, nl, paths, runtime
from ..base import Page
from ..theme import P, font
from ..widgets import Badge, Card, Entry, Labeled, RBtn, ScrollFrame, TextArea

PAGE_KEY = "nl_data"


class NLDataPage(Page):
    key = PAGE_KEY
    title = "自然语言造数"
    subtitle = "描述你想要的数据，AI 生成问答集并算出该用多少轮、多长序列、多大 LoRA"
    group = "数据层"

    def build(self) -> None:
        root = tk.Frame(self, bg=P["bg"])
        root.pack(fill="both", expand=True)
        root.grid_columnconfigure(0, weight=5, uniform="nl")
        root.grid_columnconfigure(1, weight=4, uniform="nl")
        root.grid_rowconfigure(0, weight=1)

        self._samples: list[dict] = []
        self._plan: dict = {}
        self._busy = False
        self._dataset_name = ""

        self._build_left(root)
        self._build_right(root)

    # ---------------------------------------------------------------- 左
    def _build_left(self, parent) -> None:
        card = Card(parent, "用一句话描述你要的数据",
                    "写得越具体，生成的数据越像真实业务；留空人设时会用「简洁准确的助手」")
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        b = card.body

        self.desc = TextArea(b, height=6)
        self.desc.pack(fill="x", ipady=2)
        self.desc.insert("1.0", "例如：我要一个「宠物用品店客服」问答集，覆盖退换货、"
                                "发货时效、猫粮成分、会员积分、疫苗证明这几类问题，"
                                "回答要给出明确数字和流程。")

        row = tk.Frame(b, bg=P["panel"])
        row.pack(fill="x", pady=(12, 0))
        self.var_kind = tk.StringVar(value=nl.KINDS[0][1])
        tk.Label(row, text="数据类型", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        self.cb_kind = ttk.Combobox(row, textvariable=self.var_kind, state="readonly",
                                    values=[x[1] for x in nl.KINDS], width=24, font=font(9))
        self.cb_kind.pack(side="left", padx=(6, 14), ipady=2)
        self.cb_kind.bind("<<ComboboxSelected>>", lambda e: self._kind_changed())

        tk.Label(row, text="条数", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        self.var_count = tk.StringVar(value="30")
        ttk.Spinbox(row, from_=4, to=200, increment=2, textvariable=self.var_count,
                    width=6, font=font(9)).pack(side="left", padx=(6, 14), ipady=2)

        tk.Label(row, text="语言", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        self.var_lang = tk.StringVar(value="中文")
        ttk.Combobox(row, textvariable=self.var_lang, state="readonly",
                     values=["中文", "中英混合", "English"], width=9,
                     font=font(9)).pack(side="left", padx=(6, 0), ipady=2)

        row2 = tk.Frame(b, bg=P["panel"])
        row2.pack(fill="x", pady=(10, 0))
        w1 = Labeled(row2, "任务类型", "决定学习率与序列长度", bg=P["panel"])
        w1.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.var_task = tk.StringVar(value="客服问答")
        ttk.Combobox(w1.holder, textvariable=self.var_task, state="readonly",
                     values=nl.TASKS, font=font(9)).pack(fill="x", ipady=2)
        w2 = Labeled(row2, "质量档位", "数据量小会自动加轮数", bg=P["panel"])
        w2.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.var_quality = tk.StringVar(value="可用")
        self.cb_quality = ttk.Combobox(w2.holder, textvariable=self.var_quality, state="readonly",
                                       values=[x[1] for x in nl.quality_tiers()], font=font(9))
        self.cb_quality.pack(fill="x", ipady=2)

        w3 = Labeled(b, "人设 / 风格（可选）", "例如：语气专业、回答不超过两句", bg=P["panel"])
        w3.pack(fill="x", pady=(10, 0))
        self.persona = Entry(w3.holder, placeholder="留空则用默认助手人设")
        self.persona.pack(fill="x", ipady=3)

        w4 = Labeled(b, "不要出现的内容（可选）", "用来排除竞品名、错误事实等", bg=P["panel"])
        w4.pack(fill="x", pady=(8, 0))
        self.avoid = Entry(w4.holder, placeholder="例如：不要提竞品、不要编造政策")
        self.avoid.pack(fill="x", ipady=3)

        self.media_row = tk.Frame(b, bg=P["panel"])
        self.media_row.pack(fill="x", pady=(10, 0))
        tk.Label(self.media_row, text="来源文本数据", bg=P["panel"], fg=P["muted"],
                 font=font(9)).pack(side="left")
        self.var_src = tk.StringVar()
        self.cb_src = ttk.Combobox(self.media_row, textvariable=self.var_src, state="readonly",
                                   font=font(9), width=26)
        self.cb_src.pack(side="left", padx=(6, 0), ipady=2)

        act = tk.Frame(b, bg=P["panel"])
        act.pack(fill="x", pady=(14, 0))
        self.btn_gen = RBtn(act, "AI 生成数据", self._generate, kind="primary", height=36,
                            padx=16)
        self.btn_gen.pack(side="left")
        self.btn_media = RBtn(act, "生成多模态集", self._media, kind="secondary", height=36,
                              size=9, padx=12)
        self.btn_media.pack(side="left", padx=8)

        self.status = tk.Label(b, text="", bg=P["panel"], fg=P["muted"], font=font(9),
                               anchor="w", justify="left", wraplength=520)
        self.status.pack(fill="x", pady=(10, 0))
        self.hint = tk.Label(b, text="", bg=P["panel"], fg=P["muted"], font=font(8), anchor="w",
                             justify="left", wraplength=520)
        self.hint.pack(fill="x", pady=(4, 0))
        self._kind_changed()

    # ---------------------------------------------------------------- 右
    def _build_right(self, parent) -> None:
        card = Card(parent, "生成结果", "先看几条再决定要不要保存")
        card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        b = card.body
        b.grid_rowconfigure(0, weight=1)
        b.grid_columnconfigure(0, weight=1)
        self.scroll = ScrollFrame(b, bg=P["panel"])
        self.scroll.grid(row=0, column=0, sticky="nsew")
        self.thread = self.scroll.inner
        tk.Label(self.thread, text="还没有数据。左侧写好描述后点「AI 生成数据」。",
                 bg=P["panel"], fg=P["muted"], font=font(9), anchor="w", justify="left",
                 wraplength=500).pack(anchor="w", pady=10, padx=4)

        name_row = tk.Frame(b, bg=P["panel"])
        name_row.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        tk.Label(name_row, text="数据集名", bg=P["panel"], fg=P["muted"],
                 font=font(9)).pack(side="left")
        self.var_name = tk.StringVar()
        Entry(name_row, textvariable=self.var_name, width=22).pack(side="left", padx=(6, 0),
                                                                  ipady=3)
        tk.Label(name_row, text=".jsonl", bg=P["panel"], fg=P["muted"],
                 font=font(9)).pack(side="left", padx=(4, 0))

        btn_row = tk.Frame(b, bg=P["panel"])
        btn_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        RBtn(btn_row, "保存到数据层", self._save, kind="secondary", height=32, size=9,
             padx=12).pack(side="left")
        RBtn(btn_row, "保存并去训练", self._save_and_train, kind="primary", height=32, size=9,
             padx=12).pack(side="left", padx=8)
        RBtn(btn_row, "导入文件", self._import, kind="ghost", height=32, size=9,
             padx=10).pack(side="right")

        plan = Card(b, "训练参数建议", "按真实样本量与本机实测吞吐算，不是拍脑袋")
        plan.grid(row=3, column=0, sticky="ew", pady=(12, 0)) if False else None
        pbox = tk.Frame(b, bg=P["panel2"], highlightbackground=P["border"], highlightthickness=1)
        pbox.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.plan_label = tk.Label(pbox, text="生成数据后自动计算", bg=P["panel2"],
                                   fg=P["text2"], font=font(9), anchor="w", justify="left",
                                   wraplength=520)
        self.plan_label.pack(fill="x", padx=10, pady=(8, 2))
        self.plan_notes = tk.Label(pbox, text="", bg=P["panel2"], fg=P["muted"], font=font(8),
                                   anchor="w", justify="left", wraplength=520)
        self.plan_notes.pack(fill="x", padx=10, pady=(0, 8))
        self.plan_badge = Badge(pbox, "未计算", "panel3", bg=P["panel2"])
        self.plan_badge.pack(anchor="w", padx=10, pady=(0, 8))

    # ---------------------------------------------------------------- 状态
    def on_show(self) -> None:
        self._reload_sources()

    def _reload_sources(self) -> None:
        files = ds.list_datasets()
        names = [f["name"] for f in files]
        self.cb_src.configure(values=names)
        if names and self.var_src.get() not in names:
            self.var_src.set(ds.default_dataset_name(files, self.cfg.get("last.dataset", "")))

    def _kind_changed(self) -> None:
        kind = self._kind()
        if kind == "audio":
            self.hint.configure(text="语音问答：把「来源文本数据」里的问题用系统 TTS 读成语音，"
                                    "答案为原文本答案。先有一份文本问答集再转。",
                                fg=P["amber"])
        elif kind == "image":
            self.hint.configure(text="图像问答：程序合成形状/颜色/数量随机的图片并配标准答案。"
                                    "属于教学级合成数据，换成真实照片需要重新标定。",
                                fg=P["amber"])
        else:
            self.hint.configure(text="条数建议 30~60 起步；小模型 20 条只能学格式，"
                                    "300 条以上才开始学知识。", fg=P["muted"])

    def _kind(self) -> str:
        label = self.var_kind.get()
        for k, name in nl.KINDS:
            if name == label:
                return k
        return "qa"

    def _quality(self) -> str:
        label = self.var_quality.get()
        for k, name, _desc in nl.quality_tiers():
            if name == label:
                return k
        return "usable"

    # ---------------------------------------------------------------- 生成
    def _generate(self) -> None:
        if self._busy:
            self.toast("正在生成，请稍等", "warn")
            return
        if not ai.enabled():
            self.toast("先到「设置 → AI 功能」配置 OpenRouter Key", "warn", 4200)
            return
        if self._kind() in ("image", "audio"):
            self._media()
            return
        desc = self.desc.get("1.0", "end").strip()
        if len(desc) < 4:
            self.toast("先用一句话描述你要的数据", "warn")
            return
        try:
            count = int(float(self.var_count.get() or 30))
        except ValueError:
            count = 30
        self._busy = True
        self.btn_gen.set_enabled(False)
        self.status.configure(text="正在调用免费模型生成…", fg=P["accent"])
        # 启动线程前把全部 UI 输入快照下来；后台线程只做纯计算
        kind, lang = self._kind(), self.var_lang.get()
        task = self.var_task.get()
        persona = self.persona.value()
        avoid = self.avoid.value()
        goal = desc[:200]
        quality = self._quality()

        def progress(done: int, total: int) -> None:
            self.app.post(lambda: self.status.configure(
                text=f"已生成 {done}/{total} 条…", fg=P["accent"]))

        def work() -> None:
            try:
                samples, model = nl.generate(
                    desc, count, kind=kind, lang=lang, topic=task,
                    system_hint=persona, avoid=avoid,
                    progress=progress)
                self.app.post(lambda: self._gen_done(samples, model, "", task, quality, goal))
            except Exception as exc:  # noqa: BLE001
                msg = str(exc) or type(exc).__name__
                self.app.post(lambda: self._gen_done([], "", msg, task, quality, goal))

        threading.Thread(target=work, daemon=True).start()

    def _suggest(self, samples: list[dict], task: str, quality: str,
                 goal: str) -> dict:
        models = models_mod.scan_models()
        name = models_mod.default_model_name(models, self.cfg.get("last.model", ""))
        info = models_mod.find_model(name) if name else None
        params_m = (info.params_m if info and info.params_m else 64.0)
        threads = int(self.cfg.get("threads", 0) or runtime.default_threads())
        return nl.suggest(samples, task=task, quality=quality,
                          model_name=name, params_m=params_m, threads=threads,
                          speed_samples=self.store.speed_samples(), goal=goal)

    def _gen_done(self, samples: list[dict], model: str, error: str,
                  task: str = "", quality: str = "", goal: str = "") -> None:
        self._busy = False
        self.btn_gen.set_enabled(True)
        if error:
            self.status.configure(text=f"生成失败：{error}", fg=P["red"])
            self.toast(error, "err", 5200)
            return
        if not samples:
            self.status.configure(text="没有生成任何样本，换个描述再试", fg=P["amber"])
            return
        plan = self._suggest(samples, task or self.var_task.get(),
                             quality or self._quality(), goal)
        self._samples = samples
        self._plan = plan
        self.status.configure(
            text=f"已生成 {len(samples)} 条 · 模型 {model.split('/')[-1]} · 检查无误后保存",
            fg=P["green"])
        self._render(samples)
        if not self.var_name.get():
            self.var_name.set(f"nl-{task or self.var_task.get()}-{len(samples)}条")
        self._show_plan(plan)

    def _render(self, samples: list[dict]) -> None:
        for w in self.thread.winfo_children():
            w.destroy()
        for i, s in enumerate(samples[:30], 1):
            box = tk.Frame(self.thread, bg=P["panel2"], highlightbackground=P["border"],
                           highlightthickness=1)
            box.pack(fill="x", pady=3, padx=2)
            msgs = s["messages"]
            lines = []
            for m in msgs:
                who = "问" if m["role"] == "user" else "答"
                lines.append(f"{who}：{m['content']}")
            tk.Label(box, text=f"#{i}  " + "\n".join(lines), bg=P["panel2"], fg=P["text"],
                     font=font(9), anchor="w", justify="left", wraplength=480,
                     padx=8, pady=6).pack(fill="x")
        if len(samples) > 30:
            tk.Label(self.thread, text=f"（只预览前 30 条，共 {len(samples)} 条）",
                     bg=P["panel"], fg=P["muted"], font=font(8), anchor="w").pack(
                anchor="w", pady=4)
        self.scroll.scroll_top()

    def _show_plan(self, plan: dict) -> None:
        if not plan:
            self.plan_label.configure(text="生成数据后自动计算")
            self.plan_notes.configure(text="")
            self.plan_badge.set("未计算", "panel3")
            return
        self.plan_label.configure(text=nl.plan_summary(plan), fg=P["text"])
        self.plan_notes.configure(text="；".join(plan.get("notes") or []) or "参数与训练页一致，可直接应用。")
        self.plan_badge.set(f"建议 {plan.get('steps', 0)} 步", "green")

    # ---------------------------------------------------------------- 保存
    def _save(self) -> Path | None:
        if not self._samples:
            self.toast("还没有可保存的数据", "warn")
            return None
        name = self.var_name.get().strip() or f"nl-{len(self._samples)}条"
        meta = {"source": "nl_data", "created": time_now(), "task": self.var_task.get(),
                "plan": self._plan, "description": self.desc.get("1.0", "end").strip()[:400]}
        try:
            dst = nl.save(self._samples, name, meta)
        except OSError as exc:
            self.toast(f"保存失败：{exc}", "err", 4200)
            return None
        self._dataset_name = dst.name
        self.cfg.set("last.dataset", dst.name)
        chk = ds.verify_after_write(dst)
        self.status.configure(
            text=f"已保存 data/{dst.name}（写出 {len(self._samples)} 条，复核有效 {chk['valid']}，"
                 f"坏行 {chk['bad']}，空回复 {chk['empty']}，重复 {chk['dup']}）",
            fg=P["green"])
        self.toast(f"已保存 {dst.name}（复核有效 {chk['valid']} 条）", "ok", 2600)
        self._reload_sources()
        return dst

    def _save_and_train(self) -> None:
        dst = self._save()
        if dst is None:
            return
        models = models_mod.scan_models()
        model = models_mod.default_model_name(models, self.cfg.get("last.model", ""))
        self.app.shared["pending_train"] = {
            "model": model, "dataset": dst.name, "plan": self._plan or {},
            "goal": self.desc.get("1.0", "end").strip()[:200],
        }
        self.app.goto("train")

    def _import(self) -> None:
        p = filedialog.askopenfilename(
            title="导入问答数据", filetypes=[("表格/文本", "*.csv *.tsv *.txt *.md *.jsonl")])
        if not p:
            return
        src = Path(p)
        try:
            if src.suffix.lower() == ".jsonl":
                try:
                    dst = fsutil.safe_child(paths.data_dir(), src.name)
                except fsutil.PathSafetyError as exc:
                    self.toast(f"文件名不安全：{exc}", "err", 4200)
                    return
                fsutil.atomic_copy_file(src, dst)
                n = len(ds.load_samples(dst))
            else:
                dst = fsutil.safe_child(paths.data_dir(),
                                        nl.safe_name(src.stem) + ".jsonl")
                n, note = ds.import_table(src, dst)
        except (OSError, fsutil.PathSafetyError) as exc:
            self.toast(f"导入失败：{exc}", "err", 4200)
            return
        self.status.configure(text=f"已导入 {n} 条 → data/{dst.name}", fg=P["green"])
        self.toast(f"导入 {n} 条", "ok", 2400)
        self._reload_sources()
        self.var_name.set(dst.stem)

    # ---------------------------------------------------------------- 多模态
    def _media(self) -> None:
        if self._busy:
            self.toast("正在生成，请稍等", "warn")
            return
        kind = self._kind()
        if kind not in ("image", "audio"):
            self.toast("先在「数据类型」里选图像问答或语音问答", "warn")
            return
        self._busy = True
        self.btn_gen.set_enabled(False)
        self.status.configure(text="正在生成多模态数据…", fg=P["accent"])
        src = paths.data_dir() / self.var_src.get() if self.var_src.get() else None

        def work() -> None:
            try:
                dst, n, note = nl.make_media(kind, src, per_combo=6, limit=16)
                self.app.post(lambda: self._media_done(dst, n, note, ""))
            except Exception as exc:  # noqa: BLE001
                self.app.post(lambda err=str(exc): self._media_done(None, 0, "", err))

        threading.Thread(target=work, daemon=True).start()

    def _media_done(self, dst: Path | None, n: int, note: str, error: str) -> None:
        self._busy = False
        self.btn_gen.set_enabled(True)
        if error or dst is None:
            self.status.configure(text=f"生成失败：{error}", fg=P["red"])
            self.toast(error or "生成失败", "err", 4600)
            return
        self.status.configure(text=f"{note}（data/{Path(dst).name}，{n} 条）",
                              fg=P["green"] if n else P["amber"])
        self.toast(f"已生成 {n} 条多模态数据", "ok" if n else "warn", 3200)
        self._reload_sources()
        self.var_name.set(Path(dst).stem)
        if n:
            self._dataset_name = Path(dst).name
            self.app.shared["last_dataset"] = Path(dst).name


def time_now() -> str:
    import time

    return time.strftime("%Y-%m-%d %H:%M:%S")
