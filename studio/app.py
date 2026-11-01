# -*- coding: utf-8 -*-
"""FineTune Studio main window.

Launch:
    Source: .venv\\Scripts\\python.exe studio\\app.py
    Build:  FineTuneStudio.exe
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from .core import config, jobs, paths, procs, runtime, secrets, store as store_mod, sysinfo
from .ui import theme
from .ui.theme import P, font
from .ui.widgets import RBtn, Badge, LogView, Toast

APP_TITLE = f"{paths.APP_NAME} {paths.VERSION}"


def _enable_dpi() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


class GroupHead(tk.Frame):
    """侧边栏的层标题：实验台 / 数据层 / 训练层 / 系统。"""

    def __init__(self, parent: tk.Misc, text: str, desc: str = "", color: str = "accent"):
        super().__init__(parent, bg=P["sidebar"])
        row = tk.Frame(self, bg=P["sidebar"])
        row.pack(fill="x", padx=(14, 10), pady=(10, 2))
        tk.Frame(row, bg=P[color] if color in P else P["accent"], width=6, height=6).pack(
            side="left", pady=(4, 0))
        tk.Label(row, text=text, bg=P["sidebar"], fg=P["text"], font=font(9, "bold"),
                 anchor="w").pack(side="left", padx=(6, 0))
        if desc:
            tk.Label(row, text=desc, bg=P["sidebar"], fg=P["muted"], font=font(7),
                     anchor="w").pack(side="left", padx=(6, 0))


class NavItem(tk.Frame):
    def __init__(self, parent: tk.Misc, key: str, label: str, command):
        super().__init__(parent, bg=P["sidebar"], cursor="hand2", height=34)
        self.pack_propagate(False)
        self.key = key
        self._command = command
        self._active = False
        self._hover = False
        self.bar = tk.Frame(self, bg=P["sidebar"], width=3)
        self.bar.pack(side="left", fill="y")
        self.label = tk.Label(self, text=label, bg=P["sidebar"], fg=P["text2"],
                              font=font(10), anchor="w", padx=24)
        self.label.pack(side="left", fill="both", expand=True)
        for w in (self, self.label, self.bar):
            w.bind("<Button-1>", lambda e: self._command(self.key))
            w.bind("<Enter>", self._enter)
            w.bind("<Leave>", self._leave)

    def _enter(self, _e=None) -> None:
        self._hover = True
        self._paint()

    def _leave(self, _e=None) -> None:
        self._hover = False
        self._paint()

    def set_active(self, on: bool) -> None:
        self._active = on
        self._paint()

    def _paint(self) -> None:
        if self._active:
            bg, fg, bar = P["panel"], P["text"], P["accent"]
        elif self._hover:
            bg, fg, bar = P["panel"], P["text"], P["sidebar"]
        else:
            bg, fg, bar = P["sidebar"], P["text2"], P["sidebar"]
        self.configure(bg=bg)
        self.label.configure(bg=bg, fg=fg, font=font(10, "bold" if self._active else "normal"))
        self.bar.configure(bg=bar)


class App(tk.Tk):
    def __init__(self) -> None:
        _enable_dpi()
        self.cfg = config.cfg()
        from .ui import i18n
        i18n.set_language(os.environ.get("FTS_LANG") or str(self.cfg.get("ui.lang", "en")))
        i18n.install()
        theme.apply(self.cfg.get("theme", "forest"), self.cfg.get("theme_custom") or {})
        super().__init__()
        self.title(APP_TITLE)
        try:
            ico = paths.asset_file("app.ico")
            if ico.is_file():
                self.iconbitmap(default=str(ico))
            png = paths.asset_file("app.png")
            if png.is_file():
                self._app_icon = tk.PhotoImage(file=str(png))
                self.iconphoto(True, self._app_icon)
        except tk.TclError:
            pass
        self.configure(bg=P["bg"])
        self.minsize(1120, 720)
        theme.init_fonts(self)
        theme.apply_ttk(self)

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w = min(int(self.cfg.get("window.w", 1400)), sw - 40)
        h = min(int(self.cfg.get("window.h", 880)), sh - 90)
        w = max(1000, w)
        h = max(640, h)
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2 - 10)
        self.geometry(f"{w}x{h}+{x}+{y}")

        self.store = store_mod.store()
        interrupted = self.store.mark_interrupted()
        migrated, key_note = secrets.migrate_legacy_key(self.cfg)
        self._key_migration_note = key_note if migrated else ""
        self.jobs = jobs.JobManager()
        self._ui_queue: queue.Queue = queue.Queue()
        self.shared: dict = {}
        self.env: runtime.Env | None = None
        self._env_loading = False
        self._job_handlers: dict[str, list] = {}
        self._pages: dict[str, object] = {}
        self._nav: dict[str, NavItem] = {}
        self._current = ""
        self._drawer_open = False
        self._last_snapshot = sysinfo.snapshot(str(paths.root()))

        self._build_shell()
        self._register_pages()
        self.goto("dashboard")
        if interrupted:
            self.log(f"[索引] {interrupted} 条遗留 running 记录已标记为 interrupted", "warn")
        if self._key_migration_note:
            self.log(f"[安全] {self._key_migration_note}", "warn")
            self.after(1200, lambda: self.toast(self._key_migration_note, "warn", 6000))
        self.refresh_env()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(60, self._pump)
        self.after(1000, self._tick)

        if self.cfg.get("auto_check_env", True):
            self.after(400, self._auto_env_notice)

    # ---------------------------------------------------------------- 骨架
    def _build_shell(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._side = side = tk.Frame(self, bg=P["sidebar"], width=214)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)

        logo = tk.Frame(side, bg=P["sidebar"])
        logo.pack(fill="x", pady=(20, 6), padx=18)
        tk.Label(logo, text="微调训练台", bg=P["sidebar"], fg=P["text"],
                 font=font(14, "bold"), anchor="w").pack(anchor="w")
        tk.Label(logo, text="实验台 · 数据层 · 训练层", bg=P["sidebar"],
                 fg=P["accent"], font=font(8), anchor="w").pack(anchor="w", pady=(2, 0))
        tk.Label(logo, text=f"FineTune Studio {paths.VERSION}", bg=P["sidebar"],
                 fg=P["muted"], font=font(7), anchor="w").pack(anchor="w", pady=(1, 0))

        tk.Frame(side, bg=P["sidebar"], height=14).pack()
        self.nav_box = tk.Frame(side, bg=P["sidebar"])
        self.nav_box.pack(fill="x")

        bottom = tk.Frame(side, bg=P["sidebar"])
        bottom.pack(side="bottom", fill="x", padx=16, pady=14)
        self.side_env = tk.Label(bottom, text="环境检测中…", bg=P["sidebar"], fg=P["muted"],
                                 font=font(8), anchor="w", justify="left", wraplength=180)
        self.side_env.pack(anchor="w")
        self.side_machine = tk.Label(bottom, text="", bg=P["sidebar"], fg=P["muted"],
                                     font=font(8), anchor="w", justify="left", wraplength=180)
        self.side_machine.pack(anchor="w", pady=(4, 0))

        self._right = right = tk.Frame(self, bg=P["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # 顶栏
        header = tk.Frame(right, bg=P["bg"], height=74)
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(16, 0))
        header.grid_propagate(False)
        ht = tk.Frame(header, bg=P["bg"])
        ht.pack(side="left", fill="both", expand=True)
        self.page_title = tk.Label(ht, text="", bg=P["bg"], fg=P["text"], font=font(18, "bold"),
                                   anchor="w")
        self.page_title.pack(anchor="w")
        self.page_sub = tk.Label(ht, text="", bg=P["bg"], fg=P["muted"], font=font(9), anchor="w")
        self.page_sub.pack(anchor="w", pady=(3, 0))

        hr = tk.Frame(header, bg=P["bg"])
        hr.pack(side="right")
        self.env_pill = Badge(hr, "环境检测中", "amber", bg=P["bg"])
        self.env_pill.pack(side="left", padx=(0, 10))
        self.ai_pill = Badge(hr, "AI 检测中…", "panel3", bg=P["bg"])
        self.ai_pill.pack(side="left", padx=(0, 10))
        self.ai_pill.bind("<Button-1>", lambda e: self.goto("settings"))
        self.job_pill = Badge(hr, "空闲", "panel3", bg=P["bg"])
        self.job_pill.pack(side="left", padx=(0, 10))

        # 内容区
        self.content = tk.Frame(right, bg=P["bg"])
        self.content.grid(row=1, column=0, sticky="nsew", padx=24, pady=(6, 0))
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        # 日志抽屉
        self.drawer = tk.Frame(right, bg=P["panel"], height=210,
                               highlightbackground=P["border"], highlightthickness=1)
        head = tk.Frame(self.drawer, bg=P["panel"])
        head.pack(fill="x", padx=12, pady=(8, 0))
        tk.Label(head, text="运行日志", bg=P["panel"], fg=P["text2"], font=font(9, "bold")).pack(side="left")
        self.auto_scroll = tk.BooleanVar(value=True)
        from .ui.widgets import Check

        Check(head, "自动滚动", self.auto_scroll, bg=P["panel"],
              command=self._apply_autoscroll).pack(side="left", padx=(16, 0))
        RBtn(head, "清空", self._clear_log, kind="ghost", height=24, size=8, padx=10).pack(side="right")
        RBtn(head, "复制全部", self._copy_log, kind="ghost", height=24, size=8, padx=10).pack(side="right", padx=6)
        self.logview = LogView(self.drawer, height=9, max_lines=int(self.cfg.get("log_lines", 3000)),
                               bg=P["panel"])
        self.logview.pack(fill="both", expand=True, padx=8, pady=(6, 8))
        self.drawer.grid(row=2, column=0, sticky="ew", padx=24, pady=(8, 0))
        self.drawer.grid_remove()

        # 底栏
        footer = tk.Frame(right, bg=P["bg"], height=40)
        footer.grid(row=3, column=0, sticky="ew", padx=24, pady=(6, 10))
        footer.grid_propagate(False)
        self.status = tk.Label(footer, text="就绪", bg=P["bg"], fg=P["muted"], font=font(9), anchor="w")
        self.status.pack(side="left")
        self.foot_right = tk.Label(footer, text="", bg=P["bg"], fg=P["muted"], font=font(9), anchor="e")
        self.foot_right.pack(side="right")
        RBtn(footer, "日志", self.toggle_drawer, kind="ghost", height=26, size=9, padx=12).pack(side="right", padx=8)
        RBtn(footer, "打开目录", self.open_root, kind="ghost", height=26, size=9, padx=12).pack(side="right")

    def _register_pages(self) -> None:
        from .ui.pages import GROUPS, PAGE_CLASSES

        last = None
        for cls in PAGE_CLASSES:
            page = cls(self, self.content)
            page.grid(row=0, column=0, sticky="nsew")
            self._pages[cls.key] = page
            group = getattr(cls, "group", "系统")
            if group != last:
                desc, color = GROUPS.get(group, ("", "accent"))
                GroupHead(self.nav_box, group, desc, color).pack(fill="x")
                last = group
            item = NavItem(self.nav_box, cls.key, cls.title, self.goto)
            item.pack(fill="x")
            self._nav[cls.key] = item

    # ---------------------------------------------------------------- 主题
    def apply_theme(self, name: str, custom: dict | None = None) -> bool:
        """切换或微调配色：整窗重建（页面、导航、日志面板都按新配色重画）。"""
        running = [j.label for j in self.jobs.jobs.values() if j.running]
        if running:
            self.toast(f"{'、'.join(running)} 正在运行，等它结束再换主题（换主题会重建界面）",
                       "warn", 4600)
            return False
        theme.apply(name, custom or {})
        theme.apply_ttk(self)
        self.cfg.set("theme", name, save=False)
        self.cfg.set("theme_custom", custom or {}, save=False)
        self.cfg.save()
        self.rebuild_ui()
        return True

    def reset_theme(self) -> None:
        """恢复成预设原样（丢掉自定义颜色）。"""
        name = self.cfg.get("theme", "dark")
        self.cfg.set("theme_custom", {}, save=False)
        self.cfg.save()
        self.apply_theme(name, {})

    def rebuild_ui(self) -> None:
        cur = self._current or "dashboard"
        self._rebuilding = True
        try:
            for w in (getattr(self, "_side", None), getattr(self, "_right", None)):
                if w is not None:
                    w.destroy()
            self._pages.clear()
            self._nav.clear()
            self._job_handlers.clear()
            self.configure(bg=P["bg"])
            self._build_shell()
            self._register_pages()
            self._current = ""
            self.goto(cur if cur in self._pages else "dashboard")
            if self._drawer_open:
                self.drawer.grid()
        finally:
            self._rebuilding = False
        self.log("界面已按新配色重建", "ok")

    # ---------------------------------------------------------------- 导航
    def goto(self, key: str) -> None:
        if key not in self._pages:
            return
        if self._current and self._current in self._pages:
            self._pages[self._current].on_hide()
            self._nav[self._current].set_active(False)
        self._current = key
        page = self._pages[key]
        page.ensure()
        page.tkraise()
        self._nav[key].set_active(True)
        self.page_title.configure(text=page.title)
        self.page_sub.configure(text=page.subtitle)
        page.on_show()

    def current_page(self):
        return self._pages.get(self._current)

    def goto_cloud_chat(self) -> None:
        """没装本地环境也能体验：进实验层并显式切到云端全模态。"""
        from .ui.pages.chat import CLOUD

        self.shared["chat_engine"] = CLOUD
        self.goto("chat")

    # ---------------------------------------------------------------- 服务
    def python(self, notify: bool = True) -> Path | None:
        if self.env and self.env.python:
            return self.env.python
        py, source = runtime.find_python(self.cfg)
        if py is None:
            if notify:
                self.toast(f"没有找到训练环境：{source}。请到「设置」页初始化。", "err", 5200)
                self.goto("settings")
            return None
        return py

    def refresh_env(self) -> None:
        if self._env_loading:
            return
        self._env_loading = True
        self.env_pill.set("检测环境…", "amber")

        def work() -> None:
            env = runtime.probe(self.cfg)
            self.post(lambda: self._env_done(env))

        threading.Thread(target=work, daemon=True).start()

    def _env_done(self, env: runtime.Env) -> None:
        self.env = env
        self._env_loading = False
        if env.python is None:
            self.env_pill.set("无训练环境", "red")
            self.side_env.configure(text=f"未找到 .venv\n{env.error}"[:120], fg=P["red"])
        elif env.missing:
            self.env_pill.set(f"缺依赖 {len(env.missing)}", "amber")
            self.side_env.configure(text="缺少：" + ", ".join(env.missing), fg=P["amber"])
        else:
            self.env_pill.set(f"Python {env.version}", "green")
            self.side_env.configure(
                text=f"torch {env.torch} · {env.source}", fg=P["muted"])
        self.refresh_ai()
        page = self.current_page()
        if page is not None and hasattr(page, "on_env"):
            page.on_env(env)  # type: ignore[attr-defined]

    def _auto_env_notice(self) -> None:
        env = self.env
        if env is None:
            return
        if env.python is None:
            self.toast("还没装本地训练环境：云端体验台可以直接用，其余功能可一键安装", "warn", 5200)
            self.after(600, self._maybe_first_run)
        elif env.missing:
            self.toast("训练依赖不全：" + ", ".join(env.missing), "warn", 5000)

    # ---------------------------------------------------------------- AI 状态
    def refresh_ai(self) -> None:
        from .core import ai as ai_mod

        if not hasattr(self, "ai_pill"):
            return
        if not ai_mod.enabled():
            self.ai_pill.set("AI 未配置", "panel3")
            return
        model = ai_mod.default_model().split("/")[-1].replace(":free", "")
        st = ai_mod.stats()
        ok, errs = int(st.get("ok", 0)), int(st.get("errors", 0))
        if st.get("last_error") and errs > ok:
            self.ai_pill.set(f"AI 异常 · 成功 {ok} 次", "amber")
        elif ok:
            self.ai_pill.set(f"AI 在线 · 成功 {ok} 次", "green")
        else:
            self.ai_pill.set(f"AI {model[:16]}", "green")

    # ---------------------------------------------------------------- 首次运行
    def _maybe_first_run(self) -> None:
        if getattr(self, "_first_run_shown", False):
            return
        self._first_run_shown = True
        from .core import bootstrap

        status = bootstrap.probe()
        if status.env_ok:
            self.refresh_ai()
            return
        dlg = tk.Toplevel(self)
        dlg.title("首次启动：先把运行环境准备好")
        dlg.configure(bg=P["panel"])
        dlg.transient(self)
        dlg.resizable(False, False)
        dlg.grab_set()
        wrap = tk.Frame(dlg, bg=P["panel"], padx=26, pady=22)
        wrap.pack(fill="both", expand=True)
        tk.Label(wrap, text="这台机器还缺什么", bg=P["panel"], fg=P["text"],
                 font=font(13, "bold"), anchor="w").pack(anchor="w")
        text = (
            f"· 训练环境：{'已就绪' if status.python else '未安装（只有本地训练 / 本地模型需要）'}\n"
            f"· 缺失依赖：{', '.join(status.missing) if status.missing else '无'}\n"
            f"· 摄像头 / 录音：{'已内置' if status.camera and status.mic else '未检测到设备（云端文字/图片仍可用）'}\n"
            f"· 语音识别 / 朗读：已内置（系统识别 + 系统 TTS，无需下载）\n"
            f"· AI 能力：{'已配置（云端免费模型可直接体验）' if status.ai else '未配置 Key'}\n\n"
            "云端实验台（免费全模态模型 + 语音 + 视频通话）现在就能用，不装任何东西。\n"
            "只有想训练自己的模型 / 跑本地模型时，才需要下面的「一键安装」。\n"
            "安装会下载 torch 等依赖（数百 MB，几分钟到十几分钟），进度显示在底部日志。")
        tk.Label(wrap, text=text, bg=P["panel"], fg=P["text2"], font=font(10), justify="left",
                 anchor="w", wraplength=520).pack(anchor="w", pady=(12, 18))
        row = tk.Frame(wrap, bg=P["panel"])
        row.pack(fill="x")
        RBtn(row, "以后再说", dlg.destroy, kind="ghost", height=34, padx=14).pack(side="right")
        RBtn(row, "先用云端模式", lambda: (dlg.destroy(), self.goto_cloud_chat()),
             kind="secondary", height=34, padx=14).pack(side="right", padx=8)
        RBtn(row, "一键安装环境", lambda: (dlg.destroy(), self.bootstrap_env()),
             kind="primary", height=34, padx=16).pack(side="right")
        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        x = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 2
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")

    def bootstrap_env(self) -> None:
        """一键创建 .venv 并安装训练依赖（流式输出到底部日志）。"""
        from .core import bootstrap

        if self.jobs.running("bootstrap"):
            self.toast("环境安装已经在跑", "warn")
            return
        args = bootstrap.env_install_args()
        if args is None:
            self.toast("找不到 setup_env.ps1，无法自动安装", "err", 4800)
            return
        env = procs.base_env(extra={"FTS_ROOT": str(paths.root())})
        try:
            self.jobs.spawn("bootstrap", args, cwd=paths.root(), env=env, label="环境安装")
        except OSError as exc:
            self.toast(f"启动安装失败：{exc}", "err", 4800)
            return
        if not self._drawer_open:
            self.toggle_drawer()
        self.set_job_status("环境安装中", True)
        self.log("开始安装训练环境（torch / transformers / peft …），这一步比较久", "info")
        self.bind_job("bootstrap", self._on_bootstrap_event)

    def _on_bootstrap_event(self, evt: dict) -> None:
        if evt.get("type") == "_exit":
            code = evt.get("code")
            self.unbind_job("bootstrap", self._on_bootstrap_event)
            self._first_run_shown = True
            if code == 0:
                self.log("训练环境安装完成，正在重新检测…", "ok")
                self.toast("环境装好了，训练层可以用了", "ok", 4200)
                self.refresh_env()
            else:
                self.log(f"环境安装失败（exit {code}），详见日志；可先使用云端模式", "err")
                self.toast("安装失败，仍可使用云端模式", "warn", 5200)

    def install_extra(self, kind: str) -> None:
        """按需补装：camera（opencv）/ whisper（语音识别权重）。"""
        from .core import audio, bootstrap

        key = f"extra-{kind}"
        if self.jobs.running(key):
            self.toast("这个组件正在安装", "warn")
            return
        if kind == "camera":
            py, _src = runtime.find_python(self.cfg)
            args = bootstrap.camera_install_args(py)
            if args is None:
                self.toast("先装训练环境，再装摄像头支持", "warn", 4200)
                return
            env = procs.base_env(extra={"FTS_ROOT": str(paths.root())})
            self.jobs.spawn(key, args, cwd=paths.root(), env=env, label="摄像头支持")
            self.toast("正在安装 opencv（约 30 MB）…", "info", 3200)
        elif kind == "whisper":

            def work() -> None:
                code, out = audio.download_whisper()
                self.post(lambda: self._extra_done("whisper", code, out))

            threading.Thread(target=work, daemon=True).start()
            self.toast("正在下载 Whisper 语音识别权重（约 150 MB）…", "info", 3600)
            return
        else:
            return
        if not self._drawer_open:
            self.toggle_drawer()

        def on_evt(evt: dict) -> None:
            if evt.get("type") == "_exit":
                self.unbind_job(key, on_evt)
                self._extra_done(kind, evt.get("code") or 0, "")

        self.bind_job(key, on_evt)

    def _extra_done(self, kind: str, code: int, out: str) -> None:
        name = {"camera": "摄像头支持", "whisper": "语音识别权重"}.get(kind, kind)
        if code == 0:
            self.toast(f"{name}已就绪", "ok", 3000)
            self.log(f"{name}安装完成", "ok")
        else:
            self.toast(f"{name}安装失败：{out[-160:] or code}", "err", 5200)
            self.log(f"{name}安装失败：{out[-500:]}", "err")

    def spawn_worker(self, key: str, script: str, cfg: dict, label: str = "",
                     stdin_pipe: bool = False, stop_file: Path | None = None) -> bool:
        py = self.python()
        if py is None:
            return False
        try:
            wdir = paths.workers_dir()
        except Exception as exc:  # noqa: BLE001
            self.toast(f"worker 脚本缺失：{exc}", "err", 5000)
            return False
        script_path = wdir / script
        if not script_path.is_file():
            self.toast(f"找不到 {script}（{wdir}）", "err", 5000)
            return False
        cfg_dir = paths.state_dir() / "jobcfg"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / f"{key}-{int(time.time()*1000)%100000}.json"
        try:
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:
            self.toast(f"写配置失败：{exc}", "err")
            return False
        args = [str(py), str(script_path), "--config", str(cfg_path)]
        env = procs.base_env(py, {
            "FTS_ROOT": str(paths.root()),
            "FTS_PARENT_PID": str(os.getpid()),
            "HF_ENDPOINT": str(self.cfg.get("endpoint") or "https://hf-mirror.com"),
            "HF_HOME": str(paths.root() / ".cache" / "huggingface"),
        })
        try:
            self.jobs.spawn(key, args, cwd=paths.root(), env=env, label=label or key,
                            stdin_pipe=stdin_pipe, stop_file=stop_file)
        except OSError as exc:
            self.toast(f"启动失败：{exc}", "err", 5000)
            return False
        self.set_job_status(label or key, True)
        return True

    def post(self, fn) -> None:
        """线程安全地把回调交给主线程执行（子线程禁止直接碰 tkinter）。"""
        self._ui_queue.put(fn)

    def bind_job(self, key: str, handler) -> None:
        self._job_handlers.setdefault(key, []).append(handler)

    def unbind_job(self, key: str, handler) -> None:
        lst = self._job_handlers.get(key)
        if lst and handler in lst:
            lst.remove(handler)

    def log(self, text: str, level: str = "log") -> None:
        self.logview.append(text, level)

    def toast(self, text: str, kind: str = "info", ms: int = 2600) -> None:
        Toast.show(self, text, kind, ms)

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def set_job_status(self, text: str, running: bool) -> None:
        self.job_pill.set(text, "amber" if running else "panel3")

    def open_root(self) -> None:
        procs.open_in_explorer(paths.root())

    def toggle_drawer(self) -> None:
        self._drawer_open = not self._drawer_open
        if self._drawer_open:
            self.drawer.grid()
        else:
            self.drawer.grid_remove()

    def _apply_autoscroll(self) -> None:
        self.logview.autoscroll = bool(self.auto_scroll.get())

    def _clear_log(self) -> None:
        self.logview.clear()

    def _copy_log(self) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append(self.logview.dump())
            self.toast("日志已复制到剪贴板", "ok", 1600)
        except tk.TclError:
            pass

    # ---------------------------------------------------------------- 事件泵
    def _pump(self) -> None:
        if getattr(self, "_closing", False):
            return
        if getattr(self, "_rebuilding", False):
            self.after(60, self._pump)      # 重建界面期间跳过本轮，但别把循环断掉
            return
        try:
            self.winfo_exists()
        except tk.TclError:
            return
        while True:
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self.log(f"后台回调出错：{type(exc).__name__}: {exc}", "err")
        try:
            items = self.jobs.drain(600)
        except Exception:  # noqa: BLE001
            items = []
        for key, kind, payload in items:
            if kind == "log":
                self.log(f"[{key}] {payload}", "log")
                handler_list = self._job_handlers.get(key + ":log", [])
                for h in handler_list:
                    try:
                        h(payload)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            if kind == "evt":
                evt = payload if isinstance(payload, dict) else {}
                et = evt.get("type")
                if et in ("error",):
                    self.log(f"[{key}] 错误：{evt.get('message', '')}", "err")
                elif et in ("done", "saved", "merged", "verified"):
                    self.log(f"[{key}] {et} {evt.get('status', '')}", "ok")
                for h in self._job_handlers.get(key, []):
                    try:
                        h(evt)
                    except Exception as exc:  # noqa: BLE001
                        self.log(f"[{key}] 界面处理事件出错：{type(exc).__name__}: {exc}", "err")
                continue
            if kind == "exit":
                code = payload.get("code")
                secs = payload.get("seconds") or 0
                self.log(f"[{key}] 进程退出 code={code} 用时 {secs:.1f}s",
                         "ok" if code == 0 else "err")
                self.set_job_status("空闲", False)
                for h in self._job_handlers.get(key, []):
                    try:
                        h({"type": "_exit", **payload})
                    except Exception:  # noqa: BLE001
                        pass
        self.after(60, self._pump)

    def _tick(self) -> None:
        if getattr(self, "_closing", False):
            return
        if getattr(self, "_rebuilding", False):
            self.after(1000, self._tick)
            return
        try:
            self.winfo_exists()
        except tk.TclError:
            return
        page = self.current_page()
        if page is not None:
            try:
                page.tick()
            except Exception as exc:  # noqa: BLE001
                self.log(f"界面刷新异常：{type(exc).__name__}: {exc}", "err")
        if int(time.time()) % 5 == 0:
            self._update_machine()
        self.after(1000, self._tick)

    def _update_machine(self) -> None:
        snap = sysinfo.snapshot(str(paths.root()))
        self.foot_right.configure(
            text=f"CPU {snap.cpu_percent:.0f}% · 内存 {snap.mem_available/1024**3:.1f}/{snap.mem_total/1024**3:.1f} GB · "
                 f"磁盘剩余 {snap.disk_free/1024**3:.1f} GB")
        self.side_machine.configure(
            text=f"{snap.cpu_count} 线程 · 内存 {snap.mem_total/1024**3:.0f} GB\n磁盘剩余 {snap.disk_free/1024**3:.0f} GB")

    # ---------------------------------------------------------------- 对话框
    def confirm(self, title: str, message: str, ok_text: str = "确定", danger: bool = False) -> bool:
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.configure(bg=P["panel"])
        dlg.transient(self)
        dlg.resizable(False, False)
        dlg.grab_set()
        result = {"ok": False}

        wrap = tk.Frame(dlg, bg=P["panel"], padx=24, pady=20)
        wrap.pack(fill="both", expand=True)
        tk.Label(wrap, text=title, bg=P["panel"], fg=P["text"], font=font(12, "bold"),
                 anchor="w").pack(anchor="w")
        tk.Label(wrap, text=message, bg=P["panel"], fg=P["text2"], font=font(10),
                 wraplength=460, justify="left", anchor="w").pack(anchor="w", pady=(10, 18))
        row = tk.Frame(wrap, bg=P["panel"])
        row.pack(fill="x")

        def ok() -> None:
            result["ok"] = True
            dlg.destroy()

        RBtn(row, "取消", dlg.destroy, kind="secondary", height=32, padx=18).pack(side="right")
        RBtn(row, ok_text, ok, kind="danger" if danger else "primary", height=32,
             padx=18).pack(side="right", padx=8)
        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        x = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 3
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
        dlg.wait_window()
        return result["ok"]

    def confirm3(self, title: str, message: str, ok_text: str, alt_text: str) -> str | None:
        """三选一对话框。返回 "ok" / "alt" / None。"""
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.configure(bg=P["panel"])
        dlg.transient(self)
        dlg.resizable(False, False)
        dlg.grab_set()
        result: dict[str, str | None] = {"v": None}

        wrap = tk.Frame(dlg, bg=P["panel"], padx=24, pady=20)
        wrap.pack(fill="both", expand=True)
        tk.Label(wrap, text=title, bg=P["panel"], fg=P["text"], font=font(12, "bold"),
                 anchor="w").pack(anchor="w")
        tk.Label(wrap, text=message, bg=P["panel"], fg=P["text2"], font=font(10),
                 wraplength=470, justify="left", anchor="w").pack(anchor="w", pady=(10, 18))
        row = tk.Frame(wrap, bg=P["panel"])
        row.pack(fill="x")

        def pick(v: str | None) -> None:
            result["v"] = v
            dlg.destroy()

        RBtn(row, "取消", lambda: pick(None), kind="secondary", height=32, padx=16).pack(side="right")
        RBtn(row, ok_text, lambda: pick("ok"), kind="danger", height=32, padx=16).pack(
            side="right", padx=8)
        RBtn(row, alt_text, lambda: pick("alt"), kind="ghost", height=32, padx=16).pack(side="right")
        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        x = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 3
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
        dlg.wait_window()
        return result["v"]

    def ask_text(self, title: str, label: str, initial: str = "") -> str | None:
        from .ui.widgets import Entry

        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.configure(bg=P["panel"])
        dlg.transient(self)
        dlg.resizable(False, False)
        dlg.grab_set()
        wrap = tk.Frame(dlg, bg=P["panel"], padx=24, pady=20)
        wrap.pack(fill="both", expand=True)
        tk.Label(wrap, text=label, bg=P["panel"], fg=P["text2"], font=font(10),
                 anchor="w").pack(anchor="w")
        e = Entry(wrap, width=46)
        e.pack(fill="x", pady=(8, 18), ipady=4)
        e.insert(0, initial)
        e.focus_set()
        out: dict[str, str | None] = {"v": None}
        row = tk.Frame(wrap, bg=P["panel"])
        row.pack(fill="x")

        def ok() -> None:
            out["v"] = e.value()
            dlg.destroy()

        RBtn(row, "取消", dlg.destroy, kind="secondary", height=32, padx=18).pack(side="right")
        RBtn(row, "确定", ok, kind="primary", height=32, padx=18).pack(side="right", padx=8)
        e.bind("<Return>", lambda ev: ok())
        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        x = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 3
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
        dlg.wait_window()
        return out["v"]

    # ---------------------------------------------------------------- 退出
    def _on_close(self) -> None:
        running = [j for j in self.jobs.jobs.values() if j.running]
        if running:
            names = "、".join(j.label for j in running)
            if not self.confirm("还有任务在运行", f"{names} 仍在运行。退出会强制结束这些进程，"
                                          "训练进度会丢失（已保存的检查点保留）。", "仍然退出", True):
                return
        self._closing = True
        try:
            self.cfg.set("window.w", self.winfo_width(), save=False)
            self.cfg.set("window.h", self.winfo_height(), save=False)
            self.cfg.save()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.store.mark_closing()
        except Exception:  # noqa: BLE001
            pass
        self.jobs.kill_all()
        self.destroy()


def selftest() -> int:
    """--selftest：不进入主循环，逐页构建并输出一份 JSON 体检结果（打包后验证用）。"""
    import traceback

    report: dict = {"ok": False, "pages": [], "errors": []}
    app = None
    try:
        app = App()
        app.update()
        for key in list(app._pages.keys()):
            try:
                app.goto(key)
                app.update()
                app.current_page().tick()
                app.update()
                report["pages"].append({"key": key, "ok": True})
            except Exception as exc:  # noqa: BLE001
                report["pages"].append({"key": key, "ok": False,
                                        "error": f"{type(exc).__name__}: {exc}"})
                report["errors"].append(traceback.format_exc()[-1500:])
        # 主题回归：四套预设逐套切换（每次都重建界面）
        theme_report = []
        theme_before = str(app.cfg.get("theme", "forest") or "forest")
        for tname in ("dark", "forest", "black", "light"):
            try:
                app.apply_theme(tname, {})
                app.update()
                theme_report.append({"name": tname, "ok": True,
                                     "bg": __import__("studio.ui.theme", fromlist=["x"]).P["bg"]})
            except Exception as exc:  # noqa: BLE001
                theme_report.append({"name": tname, "ok": False,
                                     "error": f"{type(exc).__name__}: {exc}"})
        report["themes"] = theme_report
        if not all(x["ok"] for x in theme_report):
            report["ok"] = False
        try:
            app.apply_theme(theme_before, {})
        except Exception:  # noqa: BLE001
            pass

        deadline = time.time() + 40
        while app.env is None and time.time() < deadline:
            app.update()
            time.sleep(0.1)
        env = app.env
        from .core import audio as audio_mod
        from .core import camera as camera_mod

        report.update({
            "root": str(paths.root()),
            "frozen": paths.is_frozen(),
            "python": str(env.python) if env else "",
            "torch": env.torch if env else "",
            "missing": env.missing if env else [],
            "workers_dir": str(paths.workers_dir()),
            "models": len(__import__("studio.core.models", fromlist=["x"]).scan_models()),
            "datasets": len(__import__("studio.core.dataset", fromlist=["x"]).list_datasets()),
            "bundled": {
                "camera": camera_mod.available(),
                "sounddevice": bool(__import__("importlib.util", fromlist=["x"]).find_spec("sounddevice")),
                "sapi_voices": len(audio_mod.voices()),
            },
        })
        report["ok"] = (all(p["ok"] for p in report["pages"]) and bool(env and env.python)
                        and bool(report["bundled"]["camera"])
                        and bool(report["bundled"]["sounddevice"]))

        if "--train" in sys.argv and env is not None and env.python:
            # --selftest --train：真跑一次 2 步冒烟训练，验证 worker 启动与产物落盘
            try:
                app.goto("train")
                page = app.current_page()
                page.ensure()
                page.apply_smoke()
                rid = getattr(page, "_cur_run_id", "")
                end = time.time() + 240
                run: dict = {}
                while time.time() < end:
                    app.update()
                    time.sleep(0.1)
                    run = app.store.get_run(rid or "") or {}
                    if run.get("status") in ("done", "stopped", "failed"):
                        break
                out = Path(str(run.get("out_dir") or ""))
                report["train"] = {
                    "run_id": rid, "status": run.get("status"),
                    "loss": run.get("final_loss"), "seconds": run.get("seconds"),
                    "out_dir": str(out),
                    "adapter": (out / "adapter_model.safetensors").is_file(),
                    "meta": (out / "train_meta.json").is_file(),
                }
                if run.get("status") != "done" or not report["train"]["adapter"]:
                    report["ok"] = False
                app.store.remove_run(rid)   # 自检不污染实验列表
            except Exception:  # noqa: BLE001
                report["errors"].append(traceback.format_exc()[-2000:])
                report["ok"] = False

        if "--autotrain" in sys.argv and env is not None and env.python:
            # --selftest --autotrain：真实 API + 真实 UI + 本地训练/推理的一轮闭环。
            try:
                app.goto("autotrain")
                page = app.current_page()
                page.ensure()
                page.goal.delete("1.0", "end")
                page.goal.insert("1.0", "让模型学会简洁回答咖啡店点单问题，给出饮品推荐和价格。")
                page.var_rounds.set("1")
                page.var_per.set("8")
                page.var_eval.set("2")
                page.var_quality.set("冒烟")
                page.var_web.set(False)
                started = time.time()
                page.start()
                deadline = time.time() + 900
                while page._running and time.time() < deadline:
                    app.update()
                    time.sleep(0.1)
                if page._running:
                    page.stop()
                    app.update()
                history = {}
                if page._history_file and page._history_file.is_file():
                    history = json.loads(page._history_file.read_text(encoding="utf-8-sig"))
                rounds = history.get("rounds") or []
                generation_model = str(history.get("generation_model") or "")
                ai_judged = any(r.get("judge_source") == "ai+local" and r.get("ai_model")
                                for r in rounds)
                report["autotrain"] = {
                    "session": page._sid,
                    "result": page._final_kind,
                    "reason": page._final_reason,
                    "rounds": len(rounds),
                    "generation_model": generation_model,
                    "ai_judged": ai_judged,
                    "holdout_id": page._holdout_id,
                    "history": str(page._history_file or ""),
                    "dataset": str(page._dataset or ""),
                    "elapsed": round(time.time() - started, 1),
                    "scores": page._scores,
                    "log_tail": app.logview.dump()[-4000:],
                    "page_log_tail": page.log.dump()[-4000:],
                }
                if (page._final_kind != "done" or len(rounds) != 1
                        or not generation_model or not ai_judged):
                    report["ok"] = False
            except Exception:  # noqa: BLE001
                report["errors"].append(traceback.format_exc()[-2000:])
                report["ok"] = False
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(traceback.format_exc()[-3000:])
        report["ok"] = False
        report["fatal"] = f"{type(exc).__name__}: {exc}"
    finally:
        text = json.dumps(report, ensure_ascii=False, indent=1)
        try:
            (paths.state_dir() / "selftest.json").write_text(text, encoding="utf-8")
        except OSError:
            pass
        try:
            if sys.stdout is not None:
                sys.stdout.write(text + "\n")
                sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        if app is not None:
            try:
                app._closing = True
                app.jobs.kill_all()
                app.destroy()
            except Exception:  # noqa: BLE001
                pass
    return 0 if report.get("ok") else 1


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    try:
        app = App()
    except Exception as exc:  # noqa: BLE001
        import traceback

        msg = f"启动失败：{type(exc).__name__}: {exc}"
        try:
            if sys.stderr is not None:
                traceback.print_exc()
                sys.stderr.write(msg + "\n")
        except Exception:  # noqa: BLE001
            pass
        try:
            messagebox.showerror(APP_TITLE, msg)
        except tk.TclError:
            pass
        return 1
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
