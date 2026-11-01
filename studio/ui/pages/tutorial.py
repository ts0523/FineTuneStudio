# -*- coding: utf-8 -*-
"""教程页：把「完全不懂」的人带到「训练出自己的模型」。

设计原则：
- 每一步都有「现在该点哪个按钮」，不停留在概念解释；
- 完成状态自动检测（有没有模型 / 有没有数据 / 有没有跑完的实验 / 有没有适配器），
  不是让用户自己判断；
- 所有数字都来自本机实测（吞吐、耗时），不给空泛承诺。
"""
from __future__ import annotations

import json
import tkinter as tk

from ...core import dataset as ds
from ...core import models as models_mod
from ...core import paths
from ..base import Page
from ..theme import P, font
from ..widgets import Card, RBtn, ScrollFrame

DATA_TEMPLATE = """{"messages":[{"role":"user","content":"你们支持退货吗？"},
             {"role":"assistant","content":"支持，签收后 7 天内可无理由退货，运费我们承担。"}]}
{"messages":[{"role":"user","content":"发货要多久？"},
             {"role":"assistant","content":"现货 24 小时内发出，江浙沪一般次日达。"}]}
{"messages":[{"role":"user","content":"发票怎么开？"},
             {"role":"assistant","content":"下单时在备注里写抬头和税号，发货后 3 个工作日内开出电子发票。"}]}"""


class TutorialPage(Page):
    key = "tutorial"
    title = "教程"
    subtitle = "7 步从零到能用，每步都能直接点过去做"

    # ------------------------------------------------------------ 构建
    def build(self) -> None:
        self.scroll = ScrollFrame(self)
        self.scroll.pack(fill="both", expand=True)
        f = self.scroll.inner
        f.grid_columnconfigure(0, weight=1)

        self.progress_card = Card(f, "你的进度", "")
        self.progress_card.grid(row=0, column=0, sticky="ew")
        pb = self.progress_card.body
        self.progress_label = tk.Label(pb, text="", bg=P["panel"], fg=P["text2"],
                                       font=font(10), anchor="w", justify="left")
        self.progress_label.pack(fill="x")
        self.progress_hint = tk.Label(pb, text="", bg=P["panel"], fg=P["muted"], font=font(9),
                                      anchor="w", justify="left", wraplength=880)
        self.progress_hint.pack(fill="x", pady=(4, 0))

        self.step_done: dict[str, bool] = {}

        # 1 认识模型
        b = self._step(f, 1, "认识你要训练的东西", "先搞明白「基座 / 数据 / 训练」分别在干什么")
        tk.Label(b, text="一句话版：基座是刚毕业的学生，数据是教材，训练是做练习题改错，"
                         "训出来的「适配器」是错题本，合并就是把错题本背进脑子。",
                 bg=P["panel"], fg=P["text"], font=font(10), anchor="w", justify="left",
                 wraplength=880).pack(fill="x")
        fact = tk.Frame(b, bg=P["panel2"])
        fact.pack(fill="x", pady=(10, 0))
        tk.Label(fact, text="你的模型：MiniMind-O 0.1B 全模态（能听、能说、能看）",
                 bg=P["panel2"], fg=P["text"], font=font(10, "bold"), anchor="w").pack(
            anchor="w", padx=12, pady=(9, 2))
        tk.Label(fact, text="本机实际训练的是它的「文本大脑」：64M 参数、216 MB、8 层、词表 6400。\n"
                            "语音/图像/语音输出这三条支路需要额外的编码器（SenseVoice、SigLIP、Mimi，合计约 3 GB）"
                            "和 GPU，CPU 上跑不动；这些权重仍然保留在模型文件里，"
                            "所以你现在训出来的文本模型，将来可以接回全模态流程。",
                 bg=P["panel2"], fg=P["text2"], font=font(9), anchor="w", justify="left",
                 wraplength=840).pack(anchor="w", padx=12, pady=(0, 10))
        r1 = tk.Frame(b, bg=P["panel"])
        r1.pack(fill="x", pady=(10, 0))
        RBtn(r1, "去模型页看看", lambda: self.app.goto("models"), kind="secondary", height=30,
             size=9, padx=12).pack(side="left")
        RBtn(r1, "完整自检这个模型", self._doctor, kind="ghost", height=30, size=9,
             padx=12).pack(side="left", padx=8)

        # 2 数据
        b = self._step(f, 2, "准备数据（这一步决定成败）", "训练数据 = 一行一个问答对，模型照着学")
        tk.Label(b, text="20 条只能改变说话风格；300 条以上才教得会新知识。"
                         "先跑通用内置演示数据就够，正式做时换成你自己的。",
                 bg=P["panel"], fg=P["text"], font=font(10), anchor="w", justify="left",
                 wraplength=880).pack(fill="x")
        tk.Label(b, text="data/ 下新建一个 .jsonl 文件（UTF-8），每行一个 JSON，格式如下：",
                 bg=P["panel"], fg=P["muted"], font=font(9), anchor="w").pack(
            anchor="w", pady=(8, 4))
        code = tk.Text(b, height=6, bg=P["panel2"], fg=P["text"], bd=0, relief="flat",
                       font=font(9, mono=True), wrap="none", padx=10, pady=8,
                       highlightthickness=1, highlightbackground=P["border"])
        code.insert("1.0", DATA_TEMPLATE)
        code.configure(state="disabled")
        code.pack(fill="x")

        tk.Label(
            b, text="好数据 5 条铁律：\n"
                    "① 问题要像真人会问的（别写「请阐述…」）\n"
                    "② 答案要具体（给数字、给路径、给邮箱），不要「视情况而定」\n"
                    "③ 同一类问题至少 20 条，覆盖面比花哨更重要\n"
                    "④ 重复的样本要删掉，重复 5 次模型就会复读\n"
                    "⑤ 一条样本控制在 1000 字以内，超长会被截断",
            bg=P["panel"], fg=P["text2"], font=font(9), anchor="w", justify="left",
            wraplength=860).pack(fill="x", pady=(10, 0))

        r2 = tk.Frame(b, bg=P["panel"])
        r2.pack(fill="x", pady=(10, 0))
        RBtn(r2, "生成 24 条演示数据（推荐先点这个）", self._make_seed, kind="primary",
             height=32, size=9, padx=14).pack(side="left")
        RBtn(r2, "复制上面的模板", lambda: self._copy(DATA_TEMPLATE), kind="secondary",
             height=32, size=9, padx=12).pack(side="left", padx=8)
        RBtn(r2, "去数据页校验", lambda: self.app.goto("datasets"), kind="ghost", height=32,
             size=9, padx=12).pack(side="left")
        self.data_state = tk.Label(b, text="", bg=P["panel"], fg=P["muted"], font=font(9),
                                   anchor="w")
        self.data_state.pack(fill="x", pady=(8, 0))

        mm_box = tk.Frame(b, bg=P["panel2"])
        mm_box.pack(fill="x", pady=(12, 0))
        tk.Label(mm_box, text="想让它看图 / 听语音？", bg=P["panel2"], fg=P["text"],
                 font=font(10, "bold"), anchor="w").pack(anchor="w", padx=12, pady=(9, 2))
        tk.Label(mm_box,
                 text="视觉用 MobileNetV3-Small（2.5M）、语音用 Whisper-tiny encoder（8M），"
                      "都冻结在本地，不占显存也不吃 GPU：\n"
                      "① 数据页点「生成图像问答集」或「把文本数据转成语音问答」\n"
                      "② 训练台选这份数据直接开训（会自动建投影层，建议 100 步以上）\n"
                      "③ 对话页点「选图片 / 选语音」再提问，回答会引用你给的内容",
                 bg=P["panel2"], fg=P["text2"], font=font(9), anchor="w", justify="left",
                 wraplength=820).pack(anchor="w", padx=12, pady=(0, 10))

        # 3 目标
        b = self._step(f, 3, "写一句目标", "这句话会存进实验记录，以后回看时能对上「当时想干嘛」")
        tk.Label(b, text="例：「让模型学会回答我们产品的售后问题，答案要简短、只讲事实」。"
                         "目标会决定后面要不要针对某类问题补数据。",
                 bg=P["panel"], fg=P["text"], font=font(10), anchor="w", justify="left",
                 wraplength=880).pack(fill="x")
        r3 = tk.Frame(b, bg=P["panel"])
        r3.pack(fill="x", pady=(10, 0))
        RBtn(r3, "去目标页（会自动算方案）", lambda: self.app.goto("goal"), kind="primary",
             height=32, size=9, padx=14).pack(side="left")

        # 4 训练
        b = self._step(f, 4, "开始训练", "第一次建议先跑 2 步冒烟（10 秒），确认链路通了再正式跑")
        tk.Label(b, text="新手就用默认参数，不用改：", bg=P["panel"], fg=P["text"],
                 font=font(10), anchor="w").pack(anchor="w")
        grid = tk.Frame(b, bg=P["panel"])
        grid.pack(fill="x", pady=(8, 0))
        params = [("轮数 epochs", "3", "同一份数据过 3 遍"),
                  ("批大小 batch", "2", "一次看 2 条，内存小就 1"),
                  ("梯度累积 accum", "2", "等效 batch = 2×2 = 4"),
                  ("学习率 lr", "2e-4", "LoRA 常用 1e-4 ~ 3e-4"),
                  ("序列长度 max_len", "256", "一条样本的最大 token 数"),
                  ("LoRA 秩 r", "16", "越大容量越强、越慢")]
        for i, (k, v, why) in enumerate(params):
            box = tk.Frame(grid, bg=P["panel2"])
            box.grid(row=i // 3, column=i % 3, sticky="nsew", padx=(0 if i % 3 == 0 else 6, 0),
                     pady=(0, 6))
            grid.grid_columnconfigure(i % 3, weight=1, uniform="p")
            tk.Label(box, text=k, bg=P["panel2"], fg=P["muted"], font=font(8),
                     anchor="w").pack(anchor="w", padx=10, pady=(7, 0))
            tk.Label(box, text=v, bg=P["panel2"], fg=P["text"], font=font(12, "bold"),
                     anchor="w").pack(anchor="w", padx=10)
            tk.Label(box, text=why, bg=P["panel2"], fg=P["muted"], font=font(8), anchor="w",
                     justify="left", wraplength=250).pack(anchor="w", padx=10, pady=(0, 8))

        tk.Label(b, text="本机实测：22 条 × 3 轮 = 15 步 ≈ 25 秒；300 条 × 3 轮 ≈ 4~5 分钟；"
                         "3000 条 × 3 轮 ≈ 40 分钟（64M 模型，4 线程，约 300~450 token/秒）。",
                 bg=P["panel"], fg=P["text2"], font=font(9), anchor="w", justify="left",
                 wraplength=860).pack(fill="x", pady=(10, 0))
        tk.Label(b, text="训练时只看一个指标 loss：这个模型起步约 4~5，20 多条数据训 3 轮会落到 "
                         "3.5~4.5（单步上下跳是正常的，看趋势）；完全不降 = 数据格式有问题；"
                         "越训越高 = 学习率太大。数据到 300 条以上，loss 才可能降到 2.x。",
                 bg=P["panel"], fg=P["text2"], font=font(9), anchor="w", justify="left",
                 wraplength=860).pack(fill="x", pady=(6, 0))
        r4 = tk.Frame(b, bg=P["panel"])
        r4.pack(fill="x", pady=(10, 0))
        RBtn(r4, "立即跑 2 步冒烟（约 10 秒）", self._smoke, kind="primary", height=32, size=9,
             padx=14).pack(side="left")
        RBtn(r4, "直接去训练台", lambda: self.app.goto("train"), kind="secondary", height=32,
             size=9, padx=12).pack(side="left", padx=8)

        # 5 验收
        b = self._step(f, 5, "验收：它到底学会了没有", "别只看 loss，要用同一批问题对比")
        tk.Label(b, text="做法：先在对话页不挂适配器问一遍（基座水平），再挂上刚训的适配器问同一批问题。",
                 bg=P["panel"], fg=P["text"], font=font(10), anchor="w", justify="left",
                 wraplength=880).pack(fill="x")
        judge = tk.Frame(b, bg=P["panel"])
        judge.pack(fill="x", pady=(8, 0))
        rows = [("变好了", "回答更短、更贴题，开始出现你数据里的说法", "green"),
                ("没变化", "数据太少或太泛，加到 300 条同分布样本", "amber"),
                ("开始胡说/复读", "过拟合或数据有脏样本：降 epochs、清洗数据", "red")]
        for i, (k, v, color) in enumerate(rows):
            row = tk.Frame(judge, bg=P["panel2"])
            row.pack(fill="x", pady=(0, 4))
            tk.Label(row, text=k, bg=P["panel2"], fg=P[color], font=font(9, "bold"),
                     width=14, anchor="w").pack(side="left", padx=(10, 0), pady=6)
            tk.Label(row, text=v, bg=P["panel2"], fg=P["text2"], font=font(9), anchor="w",
                     justify="left", wraplength=700).pack(side="left", padx=8)
        r5 = tk.Frame(b, bg=P["panel"])
        r5.pack(fill="x", pady=(10, 0))
        RBtn(r5, "去对话页验证", lambda: self.app.goto("chat"), kind="primary", height=32,
             size=9, padx=14).pack(side="left")
        RBtn(r5, "看实验记录", lambda: self.app.goto("runs"), kind="secondary", height=32,
             size=9, padx=12).pack(side="left", padx=8)

        # 6 导出
        b = self._step(f, 6, "导出成能直接用的模型", "把适配器焊进基座，得到一个独立目录")
        tk.Label(b, text="适配器只有几 MB，用的时候要配基座；合并后是一个完整模型目录"
                         "（本机约 256 MB），可以直接交给别人或部署到服务里。",
                 bg=P["panel"], fg=P["text"], font=font(10), anchor="w", justify="left",
                 wraplength=880).pack(fill="x")
        r6 = tk.Frame(b, bg=P["panel"])
        r6.pack(fill="x", pady=(10, 0))
        RBtn(r6, "去导出页合并", lambda: self.app.goto("export"), kind="primary", height=32,
             size=9, padx=14).pack(side="left")

        # 7 下一步
        b = self._step(f, 7, "接下来做什么", "跑通之后，决定效果上限的是数据，不是参数")
        tk.Label(b, text="① 把数据从 20 条扩到 300 条同分布样本（最有效）\n"
                         "② 加一个验证集（数据页可切分），看 eval_loss 是否跟着降\n"
                         "③ 想更聪明就换大一点的基座（模型页可下载），但 CPU 会更慢\n"
                         "④ 每次改一个变量再跑，用「实验」页的对比曲线判断有没有变好",
                 bg=P["panel"], fg=P["text2"], font=font(9), anchor="w", justify="left",
                 wraplength=860).pack(fill="x")

        # 术语 + FAQ
        two = tk.Frame(f, bg=P["bg"])
        two.grid(row=20, column=0, sticky="nsew", pady=(14, 0))
        two.grid_columnconfigure(0, weight=1, uniform="two")
        two.grid_columnconfigure(1, weight=1, uniform="two")

        terms = Card(two, "术语速查", "看不懂的词都在这")
        terms.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tb = terms.body
        glossary = [
            ("基座模型", "已经预训练好的模型，你只改它的一小部分"),
            ("参数", "模型内部的数字，64M 就是 6400 万个"),
            ("token", "模型眼里的「字」，中文一个字约 1~2 个 token"),
            ("LoRA", "只训练一小撮附加参数的办法，省时省内存"),
            ("适配器", "LoRA 训出来的产物，几十 MB，可插拔"),
            ("epoch", "把整份数据过一遍叫 1 轮"),
            ("batch", "一次同时看几条样本"),
            ("梯度累积", "攒几次小批次再更新一次，等效放大 batch"),
            ("学习率 lr", "每次改参数的步长，太大训崩、太小训不动"),
            ("loss", "模型答错的程度，越小越好，但不是越低越好"),
            ("过拟合", "把训练数据背下来了，换个问法就不会"),
            ("检查点", "训练中途存的档，中断后可续训"),
            ("合并", "把适配器焊进基座，得到独立模型"),
            ("量化", "把 fp32 权重压成 int8/int4，省内存但精度略降"),
        ]
        for k, v in glossary:
            row = tk.Frame(tb, bg=P["panel"])
            row.pack(fill="x", pady=(0, 3))
            tk.Label(row, text=k, bg=P["panel"], fg=P["accent"], font=font(9, "bold"),
                     width=9, anchor="w").pack(side="left")
            tk.Label(row, text=v, bg=P["panel"], fg=P["text2"], font=font(9), anchor="w",
                     justify="left", wraplength=380).pack(side="left", fill="x", expand=True)

        faq = Card(two, "常见问题", "新手最常卡住的 8 个点")
        faq.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        fb = faq.body
        qa = [
            ("我的电脑能训吗？", "能。这个模型文本主干 64M，4 线程 CPU 约 300~450 token/秒，"
                             "22 条数据 25 秒出结果。"),
            ("数据从哪来？", "客服聊天记录、FAQ 文档、产品手册改写、你自己一条条写。"
                           "20 条能改风格，300 条才教知识。"),
            ("训完还是答非所问？", "九成是数据问题：量不够、太泛、答案不具体。先扩数据再调参数。"),
            ("会把原模型训坏吗？", "不会。LoRA 只改约 1.5% 的参数，删掉适配器就回到原始水平。"),
            ("全模态的听和说呢？", "需要 SenseVoice/SigLIP/Mimi 三个子编码器和 GPU，"
                                "本机只训练文本主干；模型文件里保留了这些结构。"),
            ("训练中断了怎么办？", "训练页勾选「从检查点续训」，实验名保持一致即可接着跑。"),
            ("能用在别的程序里吗？", "能。合并后的目录是标准权重 + 自带建模代码（model_minimind.py），"
                                "导出页会生成加载代码。"),
            ("要不要开验证集？", "数据 ≥ 50 条时建议开（默认 10%），能看出有没有过拟合。"),
        ]
        for q, a in qa:
            tk.Label(fb, text=f"Q：{q}", bg=P["panel"], fg=P["text"], font=font(9, "bold"),
                     anchor="w", justify="left", wraplength=380).pack(anchor="w", pady=(0, 2))
            tk.Label(fb, text=f"A：{a}", bg=P["panel"], fg=P["text2"], font=font(9),
                     anchor="w", justify="left", wraplength=380).pack(anchor="w", pady=(0, 8))

    # ------------------------------------------------------------ 步骤外壳
    def _step(self, parent, idx: int, title: str, subtitle: str) -> tk.Frame:
        card = Card(parent, "", "")
        card.grid(row=idx, column=0, sticky="ew", pady=(12, 0))
        head = tk.Frame(card.body, bg=P["panel"])
        head.pack(fill="x")
        dot = tk.Canvas(head, width=26, height=26, bg=P["panel"], highlightthickness=0)
        dot.pack(side="left")
        tag = dot.create_oval(2, 2, 24, 24, fill=P["panel2"], outline=P["border2"])
        num = dot.create_text(13, 13, text=str(idx), fill=P["text2"], font=font(10, "bold"))
        self.step_done[str(idx)] = False
        setattr(self, f"_dot{idx}", (dot, tag, num))

        box = tk.Frame(head, bg=P["panel"])
        box.pack(side="left", fill="x", expand=True, padx=(10, 0))
        tk.Label(box, text=title, bg=P["panel"], fg=P["text"], font=font(12, "bold"),
                 anchor="w").pack(anchor="w")
        tk.Label(box, text=subtitle, bg=P["panel"], fg=P["muted"], font=font(9), anchor="w",
                 justify="left", wraplength=820).pack(anchor="w", pady=(2, 0))

        body = tk.Frame(card.body, bg=P["panel"])
        body.pack(fill="x", pady=(10, 0))
        return body

    def _mark(self, idx: int, done: bool) -> None:
        dot, tag, num = getattr(self, f"_dot{idx}")
        self.step_done[str(idx)] = done
        if done:
            dot.itemconfigure(tag, fill=P["green"], outline=P["green"])
            dot.itemconfigure(num, text="✓", fill="#0B1220")
        else:
            dot.itemconfigure(tag, fill=P["panel2"], outline=P["border2"])
            dot.itemconfigure(num, text=str(idx), fill=P["text2"])

    # ------------------------------------------------------------ 状态
    def on_show(self) -> None:
        models = models_mod.scan_models()
        datasets = ds.list_datasets()
        runs = self.store.runs
        outs = models_mod.scan_outputs()
        has_adapter = any(o.kind == "adapter" for o in outs)
        has_merged = any(o.kind == "merged" for o in outs)
        trained = any(r.get("status") in ("done", "stopped") for r in runs)
        has_goal = any((r.get("goal") or "").strip() for r in runs)

        self._mark(1, bool(models))
        self._mark(2, bool(datasets))
        self._mark(3, has_goal)
        self._mark(4, trained)
        self._mark(5, trained and has_adapter)
        self._mark(6, has_merged)
        self._mark(7, False)

        done_n = sum(1 for i in range(1, 7) if self.step_done.get(str(i)))
        tail = "　（你已经训练出自己的模型了）" if done_n >= 5 else ""
        self.progress_label.configure(text=f"已完成 {done_n}/6 步{tail}")
        if not models:
            hint = "第 1 步没过：models 目录里还没有模型。点「去模型页看看」下载 MiniMind-O（216 MB）。"
        elif not datasets:
            hint = "第 2 步没过：还没有训练数据。点上面的「生成 24 条演示数据」最快。"
        elif not trained:
            hint = "可以开训了：点第 4 步的「立即跑 2 步冒烟」，10 秒就能看到 loss。"
        elif not has_merged:
            hint = "训练完成了。去第 5 步对比效果，满意后到第 6 步合并成独立模型。"
        else:
            hint = "全流程跑通了。接下来的提升空间在数据量，见第 7 步。"
        self.progress_hint.configure(text=hint)

        self.data_state.configure(
            text=(f"当前可用数据：{', '.join(d['name'] for d in datasets[:3])}"
                  if datasets else "当前没有数据文件"),
            fg=P["text2"] if datasets else P["amber"])

    # ------------------------------------------------------------ 动作
    def _copy(self, text: str) -> None:
        try:
            self.app.clipboard_clear()
            self.app.clipboard_append(text)
            self.toast("模板已复制，粘到 data/xxx.jsonl 里改就行", "ok", 3200)
        except tk.TclError:
            self.toast("复制失败", "err")

    def _make_seed(self) -> None:
        out = paths.data_dir() / "新手练习.jsonl"
        samples = ds.make_seed("产品客服", n=24)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        self.toast(f"已生成 24 条演示数据：{out.name}（可以自己改内容）", "ok", 3600)
        self.on_show()

    def _smoke(self) -> None:
        self.app.goto("train")
        page = self.app._pages["train"]  # type: ignore[attr-defined]
        page.ensure()
        page.apply_smoke(name="第1次训练")

    def _doctor(self) -> None:
        self.app.goto("tools")
        page = self.app._pages["tools"]  # type: ignore[attr-defined]
        page.ensure()
        page.run_doctor()
