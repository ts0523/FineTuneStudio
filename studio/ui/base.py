# -*- coding: utf-8 -*-
"""页面基类。"""
from __future__ import annotations

import tkinter as tk

from .theme import P


class Page(tk.Frame):
    key = ""
    title = ""
    subtitle = ""
    scrollable = False

    def __init__(self, app, parent: tk.Misc):
        super().__init__(parent, bg=P["bg"])
        self.app = app
        self._built = False

    # ------------------------------------------------------------ 生命周期
    def build(self) -> None:
        """构建控件树，只调用一次。"""

    def ensure(self) -> None:
        if not self._built:
            self._built = True
            self.build()

    def on_show(self) -> None:
        """页面切到前台。"""

    def on_hide(self) -> None:
        """页面切到后台。"""

    def on_event(self, job_key: str, evt: dict) -> None:
        """收到 worker 事件。"""

    def tick(self) -> None:
        """每秒一次的心跳，仅在页面可见时调用。"""

    # ------------------------------------------------------------ 工具
    def toast(self, text: str, kind: str = "info", ms: int = 2600) -> None:
        from .i18n import tr
        self.app.toast(tr(text), kind, ms)

    def log(self, text: str, level: str = "log") -> None:
        self.app.log(text, level)

    @property
    def cfg(self):
        return self.app.cfg

    @property
    def store(self):
        return self.app.store
