# -*- coding: utf-8 -*-
"""数据页：导入、校验、统计、预览、清洗、切分、格式转换。"""
from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ...core import audio
from ...core import custom_data
from ...core import dataset as ds
from ...core import fsutil
from ...core import paths, procs
from ..base import Page
from ..theme import P, font
from ..widgets import Card, Entry, LogView, RBtn, ScrollFrame, Table, TextArea


class DatasetsPage(Page):
    key = "datasets"
    group = "数据层"
    title = "数据工坊"
    subtitle = "训练数据的体检台：格式、长度、重复、切分"

    def build(self) -> None:
        outer = ScrollFrame(self)
        outer.pack(fill="both", expand=True)
        f = outer.inner
        f.grid_columnconfigure(0, weight=2, uniform="d")
        f.grid_columnconfigure(1, weight=3, uniform="d")

        left = Card(f, "数据集", "data/*.jsonl，每行一个 JSON")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        lb = left.body
        self.table = Table(lb, [
            ("name", "文件", 230, "w"), ("size", "体积", 80, "e"),
            ("mtime", "修改时间", 120, "w"),
        ], height=9, on_select=self._on_select)
        self.table.pack(fill="both", expand=True)

        b1 = tk.Frame(lb, bg=P["panel"])
        b1.pack(fill="x", pady=(10, 0))
        RBtn(b1, "刷新", self.refresh, kind="secondary", height=30, size=9, padx=12).pack(side="left")
        RBtn(b1, "导入表格", self._import, kind="secondary", height=30, size=9, padx=12).pack(
            side="left", padx=6)
        RBtn(b1, "打开目录", lambda: procs.open_in_explorer(paths.data_dir()), kind="ghost",
             height=30, size=9, padx=12).pack(side="left")
        RBtn(b1, "删除", self._delete, kind="danger", height=30, size=9, padx=12).pack(side="right")
        RBtn(b1, "重命名", self._rename, kind="ghost", height=30, size=9, padx=10).pack(
            side="right", padx=6)

        b2 = tk.Frame(lb, bg=P["panel"])
        b2.pack(fill="x", pady=(8, 0))
        self.var_ratio = tk.StringVar(value="0.1")
        tk.Label(b2, text="验证集比例", bg=P["panel"], fg=P["muted"], font=font(9)).pack(side="left")
        ttk.Spinbox(b2, from_=0.0, to=0.5, increment=0.05, textvariable=self.var_ratio, width=6,
                    font=font(9)).pack(side="left", padx=6, ipady=2)
        self.var_maxchars = tk.StringVar(value="2000")
        tk.Label(b2, text="清洗上限(字符)", bg=P["panel"], fg=P["muted"], font=font(9)).pack(
            side="left", padx=(10, 0))
        ttk.Spinbox(b2, from_=200, to=20000, increment=200, textvariable=self.var_maxchars,
                    width=7, font=font(9)).pack(side="left", padx=6, ipady=2)

        b3 = tk.Frame(lb, bg=P["panel"])
        b3.pack(fill="x", pady=(8, 0))
        RBtn(b3, "切分训练/验证", self._split, kind="secondary", height=30, size=9, padx=10).pack(side="left")
        RBtn(b3, "清洗另存", self._clean, kind="secondary", height=30, size=9, padx=10).pack(
            side="left", padx=6)
        RBtn(b3, "统一为 messages", self._convert, kind="ghost", height=30, size=9, padx=10).pack(
            side="left")

        right = Card(f, "体检结果", "选中数据集后自动统计")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        rb = right.body
        stats = tk.Frame(rb, bg=P["panel"])
        stats.pack(fill="x")
        self._kv: dict[str, tk.Label] = {}
        fields = [("valid", "有效样本"), ("dups", "重复"), ("avg", "平均长度"),
                  ("range", "最短/最长"), ("tokens", "粗估 token"), ("zh", "中文占比")]
        for i, (k, label) in enumerate(fields):
            box = tk.Frame(stats, bg=P["panel2"])
            box.grid(row=i // 3, column=i % 3, sticky="nsew", padx=(0 if i % 3 == 0 else 5, 0),
                     pady=(0, 5))
            stats.grid_columnconfigure(i % 3, weight=1, uniform="s")
            tk.Label(box, text=label, bg=P["panel2"], fg=P["muted"], font=font(8),
                     anchor="w").pack(anchor="w", padx=9, pady=(6, 0))
            v = tk.Label(box, text="-", bg=P["panel2"], fg=P["text"], font=font(11, "bold"),
                         anchor="w")
            v.pack(anchor="w", padx=9, pady=(0, 6))
            self._kv[k] = v

        rb2 = tk.Frame(rb, bg=P["panel"])
        rb2.pack(fill="x", pady=(12, 0))
        RBtn(rb2, "校验格式", self._validate, kind="primary", height=30, size=9, padx=12).pack(side="left")
        RBtn(rb2, "预览前 12 条", self._preview, kind="secondary", height=30, size=9,
             padx=12).pack(side="left", padx=6)
        RBtn(rb2, "生成演示数据", self._make_seed, kind="ghost", height=30, size=9, padx=12).pack(
            side="left")

        self.view = LogView(rb, height=12, max_lines=1200, bg=P["panel"])
        self.view.pack(fill="both", expand=True, pady=(12, 0))
        self.view.append("选中一个数据集开始体检。", "info")

        # ---------------- 多模态数据
        mm_card = Card(f, "多模态数据（图像 / 语音）",
                       "编码器已备好：视觉 MobileNetV3-Small（2.5M）· 语音 Whisper-tiny（8M），参数冻结、特征缓存")
        mm_card.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        mb = mm_card.body
        tk.Label(mb, text="图像：程序合成「形状 × 颜色 × 数量」的图，模型学会看图回答；\n"
                          "语音：用系统 TTS 把你自己的文本问题读成语音，模型学会听语音答题。",
                 bg=P["panel"], fg=P["text2"], font=font(9), anchor="w", justify="left",
                 wraplength=880).pack(fill="x")
        row = tk.Frame(mb, bg=P["panel"])
        row.pack(fill="x", pady=(10, 0))
        RBtn(row, "生成图像问答集（24 条）", self._make_mm_image, kind="primary", height=30,
             size=9, padx=12).pack(side="left")
        RBtn(row, "把文本数据转成语音问答（22 条）", self._make_mm_audio, kind="secondary",
             height=30, size=9, padx=12).pack(side="left", padx=8)
        self.mm_state = tk.Label(row, text="", bg=P["panel"], fg=P["muted"], font=font(9))
        self.mm_state.pack(side="left", padx=10)
        tk.Label(mb, text="提示：多模态数据需要训练 100 步以上才答得准；训练台会自动识别 "
                          "image/audio 字段并启用投影层。",
                 bg=P["panel"], fg=P["muted"], font=font(8), anchor="w", justify="left",
                 wraplength=880).pack(fill="x", pady=(8, 0))

        # ---------------- 用户自建问答数据集
        qa_card = Card(f, "自建问答数据集",
                       "你写问题和标准回答；可录音转写问题，也可附加一张图片或一段语音")
        qa_card.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        qb = qa_card.body
        name_row = tk.Frame(qb, bg=P["panel"])
        name_row.pack(fill="x")
        tk.Label(name_row, text="数据集名称", bg=P["panel"], fg=P["muted"],
                 font=font(9)).pack(side="left")
        self.var_qa_name = tk.StringVar(
            value=f"自建问答-{time.strftime('%m%d-%H%M')}")
        Entry(name_row, textvariable=self.var_qa_name, width=34,
              placeholder="例如：咖啡店点单-标准回答").pack(
            side="left", padx=(8, 10), ipady=3, fill="x", expand=True)
        self.btn_qa_new = RBtn(name_row, "新建/切换", self._qa_new, kind="secondary",
                               height=28, size=9, padx=12)
        self.btn_qa_new.pack(side="left")
        self.qa_file_label = tk.Label(name_row, text="尚未创建", bg=P["panel"],
                                      fg=P["muted"], font=font(8))
        self.qa_file_label.pack(side="left", padx=10)

        qa_grid = tk.Frame(qb, bg=P["panel"])
        qa_grid.pack(fill="x", pady=(10, 0))
        qa_grid.grid_columnconfigure(0, weight=1, uniform="qa")
        qa_grid.grid_columnconfigure(1, weight=1, uniform="qa")
        tk.Label(qa_grid, text="问题（用户会问什么）", bg=P["panel"], fg=P["text2"],
                 font=font(9), anchor="w").grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.qa_question = TextArea(qa_grid, height=4)
        self.qa_question.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        tk.Label(qa_grid, text="标准回答（希望模型学会怎么说）", bg=P["panel"],
                 fg=P["text2"], font=font(9), anchor="w").grid(
            row=0, column=1, sticky="ew", padx=(6, 0))
        self.qa_answer = TextArea(qa_grid, height=4)
        self.qa_answer.grid(row=1, column=1, sticky="ew", padx=(6, 0))

        qa_actions = tk.Frame(qb, bg=P["panel"])
        qa_actions.pack(fill="x", pady=(8, 0))
        RBtn(qa_actions, "录音转文字到问题", self._qa_record_question,
             kind="primary", height=28, size=9, padx=12).pack(side="left")
        RBtn(qa_actions, "附加图片", self._qa_image, kind="secondary",
             height=28, size=9, padx=10).pack(side="left", padx=6)
        RBtn(qa_actions, "录音附件", self._qa_record_audio, kind="secondary",
             height=28, size=9, padx=10).pack(side="left")
        RBtn(qa_actions, "清除附件", self._qa_clear_media, kind="ghost",
             height=28, size=9, padx=10).pack(side="left", padx=6)
        RBtn(qa_actions, "添加这一对", self._qa_add, kind="primary",
             height=28, size=9, padx=14).pack(side="right")
        self.qa_attach_label = tk.Label(qb, text="未附加媒体", bg=P["panel"],
                                        fg=P["muted"], font=font(8), anchor="w")
        self.qa_attach_label.pack(fill="x", pady=(6, 0))
        self.qa_log = LogView(qb, height=4, max_lines=120, bg=P["panel"])
        self.qa_log.pack(fill="x", pady=(6, 0))
        self._qa_dataset: Path | None = None
        self._qa_media: dict[str, str] = {}
        self._qa_audio_text = ""

        self._files: list[dict] = []
        self._cache: dict[str, tuple[float, dict]] = {}

    # ------------------------------------------------------------ 列表
    def on_show(self) -> None:
        self.refresh()
        self._refresh_mm()

    def refresh(self) -> None:
        self._files = ds.list_datasets()
        rows = []
        for f in self._files:
            rows.append({
                "_id": f["name"], "name": f["name"],
                "size": paths.human_size(f["size"]),
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(f["mtime"])),
            })
        self.table.set_rows(rows)
        if rows and not self.table.selection():
            self.table.select_first()

    def _path(self) -> Path | None:
        row = self.table.selected()
        if not row:
            return None
        return paths.data_dir() / row["_id"]

    def _on_select(self, _row) -> None:
        p = self._path()
        if p is None:
            return
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return
        key = f"{p}|{mtime}"
        cached = self._cache.get(key)
        if cached:
            self._render_stats(cached[1])
            return
        for v in self._kv.values():
            v.configure(text="统计中…", fg=P["muted"])

        def work() -> None:
            s = ds.stats(p)          # 大文件可能要几秒，放后台，别卡界面
            self._cache[key] = (time.time(), s)
            self.app.post(lambda: self._render_stats(s))

        threading.Thread(target=work, daemon=True).start()

    def _render_stats(self, st: dict) -> None:
        if not st:
            return
        self._kv["valid"].configure(text=f"{st['valid']}/{st['n']}", fg=P["text"])
        self._kv["dups"].configure(text=str(st["dups"]),
                                   fg=P["red"] if st["dups"] else P["text"])
        self._kv["avg"].configure(text=f"{st['avg_chars']:.0f} 字", fg=P["text"])
        self._kv["range"].configure(text=f"{st['min_chars']}/{st['max_chars']}", fg=P["text"])
        self._kv["tokens"].configure(text=f"{st['est_tokens']/1000:.1f}K", fg=P["text"])
        self._kv["zh"].configure(text=f"{st['zh_ratio']*100:.0f}%", fg=P["text"])

    # ------------------------------------------------------------ 动作
    def _validate(self) -> None:
        p = self._path()
        if p is None:
            self.toast("先选中一个数据集", "warn")
            return
        rep = ds.validate(p)
        self.view.clear()
        self.view.append(f"校验 {p.name}：共 {rep.total} 行，有效 {rep.valid}，空回复 {rep.empty}，"
                         f"异常 {len(rep.issues)}", "ok" if rep.ok else "warn")
        if rep.formats:
            self.view.append("格式分布：" + "，".join(f"{k} {v} 条" for k, v in rep.formats.items()), "info")
        for iss in rep.issues[:50]:
            self.view.append(f"第 {iss.line} 行：{iss.message}", "err")
        if len(rep.issues) > 50:
            self.view.append(f"……还有 {len(rep.issues)-50} 条问题未列出", "warn")
        if rep.ok:
            self.view.append("格式没问题，可以直接拿去训练。", "ok")
        self.toast("校验完成" if rep.ok else f"发现 {len(rep.issues)} 个问题",
                   "ok" if rep.ok else "warn")

    def _preview(self) -> None:
        p = self._path()
        if p is None:
            return
        items = ds.preview(p, n=12, max_chars=220)
        self.view.clear()
        if not items:
            self.view.append("文件是空的", "warn")
            return
        for it in items:
            if it["error"]:
                self.view.append(f"[第{it['line']}行] {it['error']}", "err")
                continue
            self.view.append(f"— 第 {it['line']} 行 —", "info")
            for m in it["messages"]:
                role = {"user": "用户", "assistant": "助手", "system": "系统"}.get(m["role"], m["role"])
                level = "ok" if m["role"] == "assistant" else "log"
                self.view.append(f"{role}: {m['content']}", level)

    def _import(self) -> None:
        p = filedialog.askopenfilename(title="选择 csv / tsv / txt 表格",
                                       filetypes=[("表格文件", "*.csv *.tsv *.txt *.md"),
                                                  ("全部", "*.*")])
        if not p:
            return
        src = Path(p)
        try:
            name = fsutil.sanitize_component(src.stem, "import") + ".jsonl"
            dst = fsutil.safe_child(paths.data_dir(), name)
        except fsutil.PathSafetyError as exc:
            self.toast(f"文件名不安全：{exc}", "err", 4200)
            return
        n, msg = ds.import_table(src, dst)
        if n:
            chk = ds.verify_after_write(dst)
            msg += (f"（复核：有效 {chk['valid']}，坏行 {chk['bad']}，"
                    f"空回复 {chk['empty']}，重复 {chk['dup']}）")
        self.toast(msg, "ok" if n else "err", 4200)
        self.refresh()

    def _split(self) -> None:
        p = self._path()
        if p is None:
            return
        try:
            ratio = max(0.0, min(0.5, float(self.var_ratio.get())))
        except ValueError:
            self.toast("比例需要是数字", "warn")
            return
        train_out = p.with_name(p.stem + "_train.jsonl")
        val_out = p.with_name(p.stem + "_val.jsonl")
        tr, va = ds.split(p, train_out, val_out, ratio=ratio)
        vtr, vva = ds.verify_after_write(train_out), ds.verify_after_write(val_out)
        self.toast(f"已切分：训练 {tr} 条（复核有效 {vtr['valid']}）→ {train_out.name}，"
                   f"验证 {va} 条（复核有效 {vva['valid']}）→ {val_out.name}", "ok", 4200)
        self.refresh()

    def _clean(self) -> None:
        p = self._path()
        if p is None:
            return
        try:
            mc = int(float(self.var_maxchars.get()))
        except ValueError:
            mc = 2000
        out = p.with_name(p.stem + "_clean.jsonl")
        try:
            stat = ds.clean(p, out, max_chars=mc, dedupe=True, drop_empty=True)
        except OSError as exc:
            self.toast(f"清洗失败（原文件未变）：{exc}", "err", 5200)
            return
        v = stat.get("verified") or {}
        self.toast(f"清洗完成：写出 {stat['kept']}/{stat['in']}（去重 {stat['dup']}、空 {stat['empty']}、"
                   f"超长 {stat['too_long']}、坏行 {stat['bad']}）· 复核有效 {v.get('valid', 0)} "
                   f"重复 {v.get('dup', 0)} → {out.name}", "ok", 6200)
        self.refresh()

    def _convert(self) -> None:
        p = self._path()
        if p is None:
            return
        out = p.with_name(p.stem + "_messages.jsonl")
        n, skipped = ds.convert(p, out, "messages")
        v = ds.verify_after_write(out)
        self.toast(f"已统一为 messages 格式：写出 {n} 条（复核有效 {v['valid']}、坏行 {v['bad']}、"
                   f"空回复 {v['empty']}、重复 {v['dup']}，跳过 {skipped}）→ {out.name}",
                   "ok", 5200)
        self.refresh()

    def _make_seed(self) -> None:
        samples = ds.make_seed("产品客服", n=24)
        out = paths.data_dir() / f"seed_demo_{time.strftime('%m%d-%H%M')}.jsonl"
        try:
            ds.write_jsonl(out, samples)
        except OSError as exc:
            self.toast(f"生成失败：{exc}", "err", 4200)
            return
        self.toast(f"已生成演示数据 {out.name}（{len(samples)} 条）", "ok", 3600)
        self.refresh()

    def _make_mm_image(self) -> None:
        from ...core import multimodal as mm

        try:
            path, n = mm.make_image_dataset()
        except Exception as exc:  # noqa: BLE001
            self.toast(f"生成失败：{type(exc).__name__}: {exc}", "err", 4200)
            return
        self.toast(f"已生成 {n} 条图像问答 → {path.name}", "ok", 3600)
        self.refresh()
        self._refresh_mm()

    def _make_mm_audio(self) -> None:
        from ...core import multimodal as mm

        voices = mm.available_voices()
        if not voices:
            self.toast("系统里没有可用语音（需要 Windows 中文 TTS）", "err", 4200)
            return
        src = self._path()
        try:
            path, n, tts_err = mm.make_audio_dataset(src, limit=22)
        except Exception as exc:  # noqa: BLE001
            self.toast(f"生成失败：{type(exc).__name__}: {exc}", "err", 4200)
            return
        if n == 0:
            self.toast(f"没有可转换的问答对（{tts_err or '需要 user/assistant 文本样本'}）",
                       "warn", 4200)
            return
        if tts_err:
            self.toast(f"已生成 {n} 条，但有失败：{tts_err}", "warn", 5200)
        else:
            self.toast(f"已生成 {n} 条语音问答（语音：{voices[0]}）→ {path.name}", "ok", 4200)
        self.refresh()
        self._refresh_mm()

    def _refresh_mm(self) -> None:
        from ...core import multimodal as mm

        parts = []
        for name in ("mm_图像问答.jsonl", "mm_语音问答.jsonl"):
            p = paths.data_dir() / name
            if p.is_file():
                st = mm.mm_stats(p)
                kind = "图像" if st["image"] else "语音"
                miss = f"，缺文件 {st['missing']}" if st["missing"] else ""
                parts.append(f"{name}：{st['total']} 条{kind}{miss}")
        self.mm_state.configure(text=" ｜ ".join(parts) if parts else "还没有多模态数据",
                                fg=P["text2"] if parts else P["muted"])

    # ------------------------------------------------------------ 自建问答
    def _qa_path(self, create: bool = False) -> Path | None:
        name = self.var_qa_name.get().strip()
        if not name:
            self.toast("先给数据集起个名字", "warn")
            return None
        try:
            path = custom_data.dataset_path(name)
            if create:
                path = custom_data.create(name)
        except (custom_data.CustomDataError, fsutil.PathSafetyError, OSError) as exc:
            self.toast(f"数据集创建失败：{exc}", "err", 4200)
            return None
        self._qa_dataset = path
        self.qa_file_label.configure(
            text=f"{path.name} · {custom_data.count(path)} 条", fg=P["text2"])
        return path

    def _qa_new(self) -> None:
        path = self._qa_path(create=True)
        if path is None:
            return
        self._qa_media = {}
        self._qa_audio_text = ""
        self.qa_attach_label.configure(text="未附加媒体", fg=P["muted"])
        self.qa_log.clear()
        self.qa_log.append(f"已创建/切换到 {path.name}，可以开始添加问答。", "ok")
        self.refresh()

    def _qa_attach(self, kind: str, path: str) -> None:
        self._qa_media = {kind: path}
        self.qa_attach_label.configure(
            text=f"已附加{'图片' if kind == 'image' else '语音'}：{Path(path).name}",
            fg=P["green"])

    def _qa_image(self) -> None:
        p = filedialog.askopenfilename(
            title="选择图片", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if p:
            self._qa_attach("image", p)

    def _qa_clear_media(self) -> None:
        self._qa_media = {}
        self._qa_audio_text = ""
        self.qa_attach_label.configure(text="未附加媒体", fg=P["muted"])

    def _qa_record_question(self) -> None:
        ok, err = audio.mic_available()
        if not ok:
            self.toast(err or "没有可录音设备", "err", 4200)
            return
        self.qa_attach_label.configure(text="录音 4 秒…请说问题", fg=P["accent"])

        def work() -> None:
            wav, error = audio.record_for(4.0)
            if wav is None:
                self.app.post(lambda: self.toast(error or "没有录到声音", "err", 4200))
                return
            text, engine, err2 = audio.transcribe(wav)
            self.app.post(lambda: self._qa_question_done(text, engine, err2))

        threading.Thread(target=work, daemon=True).start()

    def _qa_question_done(self, text: str, engine: str, error: str) -> None:
        if not text:
            self.qa_attach_label.configure(text=f"没听清：{error}", fg=P["amber"])
            return
        self.qa_question.delete("1.0", "end")
        self.qa_question.insert("1.0", text)
        self.qa_attach_label.configure(text=f"问题已转文字（{engine}）", fg=P["green"])

    def _qa_record_audio(self) -> None:
        ok, err = audio.mic_available()
        if not ok:
            self.toast(err or "没有可录音设备", "err", 4200)
            return
        self.qa_attach_label.configure(text="录音 4 秒…请说话", fg=P["accent"])

        def work() -> None:
            wav, error = audio.record_for(4.0)
            if wav is None:
                self.app.post(lambda: self.toast(error or "没有录到声音", "err", 4200))
                return
            text, _engine, _err = audio.transcribe(wav)
            self.app.post(lambda: self._qa_audio_done(wav, text))

        threading.Thread(target=work, daemon=True).start()

    def _qa_audio_done(self, wav: Path, text: str) -> None:
        self._qa_audio_text = text
        self._qa_attach("audio", str(wav))

    def _qa_add(self) -> None:
        path = self._qa_path(create=True)
        if path is None:
            return
        q = self.qa_question.get("1.0", "end").strip()
        a = self.qa_answer.get("1.0", "end").strip()
        try:
            n = custom_data.append_pair(
                path, q, a, image=self._qa_media.get("image"),
                audio=self._qa_media.get("audio"), audio_text=self._qa_audio_text)
        except (custom_data.CustomDataError, fsutil.PathSafetyError, OSError) as exc:
            self.toast(f"添加失败：{exc}", "err", 4400)
            return
        self.qa_log.append(f"已添加第 {n} 条：{q[:36]}", "ok")
        self.qa_file_label.configure(text=f"{path.name} · {n} 条", fg=P["green"])
        self.qa_question.delete("1.0", "end")
        self.qa_answer.delete("1.0", "end")
        self._qa_clear_media()
        self.qa_question.focus_set()
        self.refresh()
        self.toast(f"已写入 {path.name}（{n} 条）", "ok", 3000)

    def _rename(self) -> None:
        p = self._path()
        if p is None:
            self.toast("先选中一个数据集", "warn")
            return
        name = self.app.ask_text("重命名数据集", "新名称（不含路径）", p.stem)
        if not name:
            return
        try:
            dst = custom_data.rename(p, name)
        except (custom_data.CustomDataError, fsutil.PathSafetyError, OSError) as exc:
            self.toast(f"重命名失败：{exc}", "err", 4400)
            return
        if self._qa_dataset == p:
            self._qa_dataset = dst
            self.var_qa_name.set(dst.stem)
        self.toast(f"已重命名为 {dst.name}", "ok")
        self.refresh()

    def _delete(self) -> None:
        p = self._path()
        if p is None:
            return
        if not self.app.confirm("删除数据集", f"将删除 {p.name}。"
                                f"{'会进回收站。' if self.cfg.get('recycle_delete') else '不可恢复。'}",
                                "删除", True):
            return
        from ...core import models as models_mod

        ok, msg = models_mod.delete_dir(p, recycle=bool(self.cfg.get("recycle_delete", True)))
        self.toast(msg, "ok" if ok else "err")
        self.refresh()
