# -*- coding: utf-8 -*-
"""自绘控件集合。全部基于 tk.Canvas / tk.Frame，保证深色浅色两套主题下表现一致。"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Callable, Iterable, Sequence

from . import theme
from .theme import P, font, round_rect


def text_on(bg: str) -> str:
    """按底色亮度挑黑/白字，保证 4.5:1 以上对比（徽章、色块都用它）。"""
    try:
        r, g, b = theme._hex2rgb(bg)
    except (ValueError, IndexError):
        return "#FFFFFF"
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "#101418" if lum > 0.62 else "#FFFFFF"


# ---------------------------------------------------------------- 基础容器
class Card(tk.Frame):
    """带标题栏的卡片容器。内容放进 .body。"""

    def __init__(self, parent: tk.Misc, title: str = "", subtitle: str = "", pad: int = 16,
                 bg: str | None = None, **kw):
        super().__init__(parent, bg=bg or P["panel"], highlightbackground=P["border"],
                         highlightcolor=P["border"], highlightthickness=1, bd=0, **kw)
        self.pad = pad
        self._title = title
        self.header: tk.Frame | None = None
        self.actions: tk.Frame | None = None
        if title:
            self.header = tk.Frame(self, bg=self["bg"])
            self.header.pack(fill="x", padx=pad, pady=(pad, 0))
            left = tk.Frame(self.header, bg=self["bg"])
            left.pack(side="left", fill="x", expand=True)
            tk.Label(left, text=title, bg=self["bg"], fg=P["text"],
                     font=font(11, "bold"), anchor="w").pack(anchor="w")
            if subtitle:
                tk.Label(left, text=subtitle, bg=self["bg"], fg=P["muted"],
                         font=font(9), anchor="w", justify="left").pack(anchor="w", pady=(2, 0))
            self.actions = tk.Frame(self.header, bg=self["bg"])
            self.actions.pack(side="right")
        self.body = tk.Frame(self, bg=self["bg"])
        top = (10 if title else pad)
        self.body.pack(fill="both", expand=True, padx=pad, pady=(top, pad))


class ScrollFrame(tk.Frame):
    """竖向滚动容器，内容放进 .inner。"""

    def __init__(self, parent: tk.Misc, bg: str | None = None, **kw):
        super().__init__(parent, bg=bg or P["bg"], **kw)
        self.canvas = tk.Canvas(self, bg=self["bg"], highlightthickness=0, bd=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")
        self.inner = tk.Frame(self.canvas, bg=self["bg"])
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        for widget in (self.canvas, self.inner):
            widget.bind("<Enter>", self._bind_wheel)
            widget.bind("<Leave>", self._unbind_wheel)

    def _on_inner(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event) -> None:
        self.canvas.itemconfigure(self._win, width=event.width)

    def _bind_wheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._wheel)

    def _unbind_wheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _wheel(self, event) -> None:
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except tk.TclError:
            pass

    def scroll_top(self) -> None:
        self.canvas.yview_moveto(0)


# ---------------------------------------------------------------- 按钮
class RBtn(tk.Canvas):
    """圆角按钮。kind: primary / secondary / ghost / danger / success。"""

    STYLES = {
        "primary": ("accent", "accent_hi", "accent_dim", "#FFFFFF"),
        "secondary": ("panel2", "panel3", "panel3", "text"),
        "ghost": ("panel", "panel2", "panel3", "text2"),
        "danger": ("red", "red", "red", "white"),
        "success": ("green", "green", "green", "white"),
    }

    def __init__(self, parent: tk.Misc, text: str, command: Callable | None = None,
                 kind: str = "primary", width: int | None = None, height: int = 34,
                 size: int = 10, bold: bool = False, padx: int = 16, tooltip: str = ""):
        from .i18n import tr
        text = tr(text)
        bg = parent["bg"] if "bg" in parent.keys() else P["bg"]
        self._text = text
        self._kind = kind
        self._size = size
        self._bold = bold
        self._padx = padx
        self._height = height
        self._command = command
        self._enabled = True
        self._hover = False
        self._pressed = False
        try:
            meas = tkfont.Font(family=theme.UI_FAMILY, size=size,
                               weight="bold" if bold else "normal").measure(text)
        except Exception:  # noqa: BLE001
            meas = len(text) * size
        w = width or (meas + padx * 2)
        super().__init__(parent, width=w, height=height, bg=bg, highlightthickness=0,
                         bd=0, cursor="hand2", takefocus=1)
        self._tag = f"btn{id(self)}"
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<FocusIn>", lambda e: self._draw())
        self.bind("<FocusOut>", lambda e: self._draw())
        self.bind("<Return>", lambda e: self._fire())
        self.bind("<space>", lambda e: self._fire())
        self._draw()

    # --
    def _colors(self) -> tuple[str, str, str, str]:
        base, hover, press, fg = self.STYLES.get(self._kind, self.STYLES["secondary"])
        if self._pressed:
            fill = P["accent_dim"] if self._kind == "primary" else P[press]
        elif self._hover:
            fill = P[hover]
        else:
            fill = P[base]
        return fill, P.get(fg, fg), P[press], P[base]

    def _draw(self) -> None:
        self.delete("all")
        w = int(self["width"])
        h = int(self["height"])
        fill, fg, _, _ = self._colors()
        border = ""
        if self._kind == "secondary":
            border = P["border2"] if self._hover else P["border"]
        elif self._kind == "ghost":
            border = P["border"] if self._hover else ""
        if not self._enabled:
            fill, fg = P["panel2"], P["muted"]
        round_rect(self, 1, 1, w - 1, h - 1, 7, fill=fill, outline=border or fill)
        if self.focus_get() is self and self._enabled:
            round_rect(self, 1, 1, w - 1, h - 1, 7, fill="", outline=P["accent"], width=1)
        self.create_text(w // 2, h // 2 + 1, text=self._text, fill=fg,
                         font=font(self._size, "bold" if self._bold else "normal"))

    def _on_enter(self, _e=None) -> None:
        self._hover = True
        self._draw()

    def _on_leave(self, _e=None) -> None:
        self._hover = False
        self._pressed = False
        self._draw()

    def _on_press(self, _e=None) -> None:
        if not self._enabled:
            return
        self._pressed = True
        self.focus_set()
        self._draw()

    def _on_release(self, _e=None) -> None:
        was = self._pressed
        self._pressed = False
        self._draw()
        if was and self._enabled:
            self._fire()

    def _fire(self) -> None:
        if self._enabled and self._command:
            self._command()

    # --
    def set_text(self, text: str) -> None:
        from .i18n import tr
        text = tr(text)
        self._text = text
        try:
            meas = tk.font.Font(family=theme.UI_FAMILY, size=self._size,
                                weight="bold" if self._bold else "normal").measure(text)
        except Exception:  # noqa: BLE001
            meas = len(text) * self._size
        self.configure(width=meas + self._padx * 2)
        self._draw()

    def set_enabled(self, ok: bool) -> None:
        self._enabled = bool(ok)
        self.configure(cursor="hand2" if ok else "arrow")
        self._draw()

    def set_kind(self, kind: str) -> None:
        self._kind = kind
        self._draw()


class Link(tk.Label):
    """文字链接。"""

    def __init__(self, parent: tk.Misc, text: str, command: Callable, bg: str | None = None,
                 fg: str | None = None, size: int = 9):
        super().__init__(parent, text=text, bg=bg or parent["bg"], fg=fg or P["accent"],
                         font=font(size), cursor="hand2")
        self._command = command
        self.bind("<Button-1>", lambda e: command())
        self.bind("<Enter>", lambda e: self.configure(fg=P["accent_hi"]))
        self.bind("<Leave>", lambda e: self.configure(fg=fg or P["accent"]))


class Badge(tk.Canvas):
    """小胶囊标签。"""

    def __init__(self, parent: tk.Misc, text: str, color: str = "accent", bg: str | None = None,
                 size: int = 8, height: int = 20):
        from .i18n import tr
        text = tr(text)
        bgc = bg or (parent["bg"] if "bg" in parent.keys() else P["bg"])
        try:
            w = tkfont.Font(family=theme.UI_FAMILY, size=size, weight="bold").measure(text) + 16
        except Exception:  # noqa: BLE001
            w = len(text) * size + 16
        super().__init__(parent, width=w, height=height, bg=bgc, highlightthickness=0, bd=0)
        self._text, self._color, self._size = text, color, size
        self._height = height
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        w = int(self["width"])
        h = self._height
        fill = P.get(self._color, self._color)
        round_rect(self, 1, 1, w - 1, h - 1, h / 2, fill=fill, outline=fill)
        self.create_text(w // 2, h // 2, text=self._text, fill=text_on(fill),
                         font=font(self._size, "bold"))

    def set(self, text: str, color: str) -> None:
        from .i18n import tr
        text = tr(text)
        self._text, self._color = text, color
        try:
            w = tkfont.Font(family=theme.UI_FAMILY, size=self._size, weight="bold").measure(text) + 16
        except Exception:  # noqa: BLE001
            w = len(text) * self._size + 16
        self.configure(width=w)
        self._draw()


# ---------------------------------------------------------------- 输入
class Entry(tk.Entry):
    def __init__(self, parent: tk.Misc, textvariable: tk.Variable | None = None,
                 width: int = 20, placeholder: str = "", show: str | None = None):
        from .i18n import tr
        placeholder = tr(placeholder)
        super().__init__(parent, textvariable=textvariable, width=width, bg=P["panel2"],
                         fg=P["text"], insertbackground=P["text"], relief="flat",
                         highlightthickness=1, highlightbackground=P["border"],
                         highlightcolor=P["accent"], font=font(10), bd=0,
                         disabledbackground=P["panel3"], disabledforeground=P["muted"])
        if show:
            self.configure(show=show)
        self._ph = placeholder
        self._showing_ph = False
        if placeholder and not (textvariable and textvariable.get()):
            self._show_ph()
        self.bind("<FocusIn>", self._focus_in)
        self.bind("<FocusOut>", self._focus_out)

    def _show_ph(self) -> None:
        self._showing_ph = True
        self.insert(0, self._ph)
        self.configure(fg=P["muted"])

    def _focus_in(self, _e=None) -> None:
        self.configure(highlightbackground=P["accent"])
        if self._showing_ph:
            self.delete(0, "end")
            self.configure(fg=P["text"])
            self._showing_ph = False

    def _focus_out(self, _e=None) -> None:
        self.configure(highlightbackground=P["border"])
        if self._ph and not self.get():
            self._show_ph()

    def value(self) -> str:
        return "" if self._showing_ph else self.get().strip()

    def set_value(self, v: str) -> None:
        self.delete(0, "end")
        self._showing_ph = False
        self.configure(fg=P["text"])
        self.insert(0, v)


class TextArea(tk.Text):
    def __init__(self, parent: tk.Misc, height: int = 4, width: int = 40, wrap: str = "word"):
        super().__init__(parent, height=height, width=width, wrap=wrap, bg=P["panel2"],
                         fg=P["text"], insertbackground=P["text"], relief="flat",
                         highlightthickness=1, highlightbackground=P["border"],
                         highlightcolor=P["accent"], font=font(10), bd=0, padx=8, pady=6,
                         selectbackground=P["accent_dim"], spacing1=2, spacing3=2)
        self.bind("<FocusIn>", lambda e: self.configure(highlightbackground=P["accent"]))
        self.bind("<FocusOut>", lambda e: self.configure(highlightbackground=P["border"]))


class Check(tk.Frame):
    """自绘复选框。"""

    def __init__(self, parent: tk.Misc, text: str, var: tk.BooleanVar,
                 command: Callable | None = None, bg: str | None = None):
        bgc = bg or (parent["bg"] if "bg" in parent.keys() else P["bg"])
        super().__init__(parent, bg=bgc, cursor="hand2")
        self.var = var
        self._command = command
        self.canvas = tk.Canvas(self, width=17, height=17, bg=bgc, highlightthickness=0, bd=0)
        self.canvas.pack(side="left")
        self.label = tk.Label(self, text=text, bg=bgc, fg=P["text2"], font=font(10))
        self.label.pack(side="left", padx=(7, 0))
        for w in (self, self.canvas, self.label):
            w.bind("<Button-1>", self._toggle)
        self.var.trace_add("write", lambda *a: self._draw())
        self._draw()

    def _toggle(self, _e=None) -> None:
        self.var.set(not self.var.get())
        if self._command:
            self._command()

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        on = bool(self.var.get())
        fill = P["accent"] if on else P["panel2"]
        round_rect(c, 1, 1, 16, 16, 4, fill=fill, outline=P["accent"] if on else P["border2"])
        if on:
            c.create_line(4.5, 8.5, 7.5, 12, 12.5, 5.5, fill="#FFFFFF", width=2,
                          capstyle="round", joinstyle="round")


class Slider(tk.Frame):
    def __init__(self, parent: tk.Misc, var: tk.DoubleVar, frm: float, to: float,
                 resolution: float = 0.05, bg: str | None = None, width: int = 180,
                 command: Callable | None = None):
        bgc = bg or (parent["bg"] if "bg" in parent.keys() else P["bg"])
        super().__init__(parent, bg=bgc)
        self.scale = tk.Scale(self, variable=var, from_=frm, to=to, resolution=resolution,
                              orient="horizontal", showvalue=False, bg=bgc, fg=P["text"],
                              troughcolor=P["panel3"], highlightthickness=0, bd=0,
                              sliderrelief="flat", activebackground=P["accent_hi"],
                              width=8, length=width, sliderlength=16,
                              command=(lambda v: command(var.get())) if command else None)
        self.scale.pack(side="left")
        self.val = tk.Label(self, text="", bg=bgc, fg=P["text2"], font=font(9, mono=True), width=6)
        self.val.pack(side="left", padx=(6, 0))
        var.trace_add("write", lambda *a: self.val.configure(text=f"{var.get():.2f}"))


class Labeled(tk.Frame):
    """标签 + 控件的纵向小组合。"""

    def __init__(self, parent: tk.Misc, label: str, hint: str = "", bg: str | None = None):
        bgc = bg or (parent["bg"] if "bg" in parent.keys() else P["bg"])
        super().__init__(parent, bg=bgc)
        head = tk.Frame(self, bg=bgc)
        head.pack(fill="x")
        tk.Label(head, text=label, bg=bgc, fg=P["text2"], font=font(9)).pack(side="left")
        if hint:
            tk.Label(head, text=hint, bg=bgc, fg=P["muted"], font=font(8)).pack(side="right")
        self.holder = tk.Frame(self, bg=bgc)
        self.holder.pack(fill="x", pady=(4, 0))


# ---------------------------------------------------------------- 数据展示
class Metric(tk.Frame):
    """指标块：标题 / 大数字 / 副标题。"""

    def __init__(self, parent: tk.Misc, label: str, value: str = "-", sub: str = "",
                 color: str = "text", bg: str | None = None):
        bgc = bg or P["panel"]
        super().__init__(parent, bg=bgc)
        self._color = color
        self.lab = tk.Label(self, text=label, bg=bgc, fg=P["muted"], font=font(9), anchor="w")
        self.lab.pack(anchor="w")
        self.val = tk.Label(self, text=value, bg=bgc, fg=P[color] if color in P else color,
                            font=font(17, "bold"), anchor="w")
        self.val.pack(anchor="w", pady=(2, 0))
        self.sub = tk.Label(self, text=sub, bg=bgc, fg=P["muted"], font=font(8), anchor="w")
        self.sub.pack(anchor="w")

    def set(self, value: str, sub: str | None = None, color: str | None = None) -> None:
        self.val.configure(text=value)
        if sub is not None:
            self.sub.configure(text=sub)
        if color:
            self._color = color
            self.val.configure(fg=P.get(color, color))


class Bar(tk.Canvas):
    """进度条（圆角、可显示文字）。"""

    def __init__(self, parent: tk.Misc, height: int = 8, bg: str | None = None,
                 color: str = "accent", show_text: bool = False):
        bgc = bg or (parent["bg"] if "bg" in parent.keys() else P["bg"])
        super().__init__(parent, height=height if not show_text else height + 16, bg=bgc,
                         highlightthickness=0, bd=0)
        self._v = 0.0
        self._color = color
        self._show_text = show_text
        self._h = height
        self.bind("<Configure>", lambda e: self.redraw())

    def set(self, value: float, color: str | None = None) -> None:
        self._v = max(0.0, min(1.0, float(value)))
        if color:
            self._color = color
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w = max(1, self.winfo_width())
        h = self._h
        if w <= 2:
            return
        round_rect(self, 0, 0, w, h, h / 2, fill=P["panel3"])
        if self._v > 0:
            fw = max(h, w * self._v)
            round_rect(self, 0, 0, fw, h, h / 2, fill=P.get(self._color, self._color))
        if self._show_text:
            self.create_text(2, h + 9, text=f"{self._v*100:.0f}%", anchor="w",
                             fill=P["text2"], font=font(9, mono=True))


class Table(ttk.Treeview):
    """带表头排序的表格。columns: [(key, 标题, 宽度, 对齐)]"""

    def __init__(self, parent: tk.Misc, columns: Sequence[tuple], height: int = 8,
                 on_select: Callable | None = None, on_double: Callable | None = None):
        self._cols = list(columns)
        keys = [c[0] for c in self._cols]
        super().__init__(parent, columns=keys, show="headings", height=height,
                         selectmode="browse")
        for key, title, width, anchor in self._cols:
            self.heading(key, text=title, anchor="w",
                         command=lambda k=key: self.sort_by(k))
            self.column(key, width=width, anchor=anchor, stretch=(key == self._cols[-1][0]))
        self._rows: list[dict] = []
        self._sort_key: str | None = None
        self._sort_rev = False
        self._on_select = on_select
        self._on_double = on_double
        self.tag_configure("running", foreground=P["amber"])
        self.tag_configure("done", foreground=P["text"])
        self.tag_configure("stopped", foreground=P["amber"])
        self.tag_configure("failed", foreground=P["red"])
        self.tag_configure("muted", foreground=P["muted"])
        self.bind("<<TreeviewSelect>>", self._selected)
        if on_double:
            self.bind("<Double-1>", lambda e: on_double(self.selected()))

    def set_rows(self, rows: Iterable[dict]) -> None:
        self._rows = list(rows)
        self.refresh()

    def refresh(self) -> None:
        sel = self.selection()
        keep = sel[0] if sel else None
        self.delete(*self.get_children())
        rows = list(self._rows)
        if self._sort_key:
            rows.sort(key=lambda r: _sort_val(r.get(self._sort_key, "")), reverse=self._sort_rev)
        for r in rows:
            iid = str(r.get("_id") or id(r))
            tags = (r.get("_tag") or "",)
            values = [r.get(_i, "") for _i, *_ in self._cols]
            self.insert("", "end", iid=iid, values=values, tags=tags)
        if keep and self.exists(keep):
            self.selection_set(keep)

    def sort_by(self, key: str) -> None:
        if self._sort_key == key:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_key, self._sort_rev = key, False
        self.refresh()

    def selected(self) -> dict | None:
        sel = self.selection()
        if not sel:
            return None
        iid = sel[0]
        for r in self._rows:
            if str(r.get("_id") or id(r)) == iid:
                return r
        return None

    def _selected(self, _e=None) -> None:
        if self._on_select:
            self._on_select(self.selected())

    def select_first(self) -> None:
        kids = self.get_children()
        if kids:
            self.selection_set(kids[0])


def _sort_val(v):
    if isinstance(v, (int, float)):
        return (0, v)
    s = str(v)
    try:
        return (0, float(s))
    except ValueError:
        return (1, s.lower())


class LogView(tk.Frame):
    """日志面板：彩色分级、限行、可导出。"""

    TAGS = {
        "log": ("text2", "normal"),
        "info": ("accent", "normal"),
        "ok": ("green", "normal"),
        "evt": ("text2", "normal"),
        "warn": ("amber", "normal"),
        "err": ("red", "bold"),
        "cmd": ("purple", "bold"),
    }

    def __init__(self, parent: tk.Misc, height: int = 12, max_lines: int = 3000,
                 bg: str | None = None):
        bgc = bg or P["panel"]
        super().__init__(parent, bg=bgc)
        self.max_lines = max_lines
        self.text = tk.Text(self, height=height, bg=P["panel2"], fg=P["text"], bd=0,
                            relief="flat", font=font(9, mono=True), wrap="word",
                            insertbackground=P["text"], padx=10, pady=8,
                            selectbackground=P["accent_dim"], highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.vbar.set)
        self.text.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")
        for tag, (color, weight) in self.TAGS.items():
            self.text.tag_configure(tag, foreground=P.get(color, color),
                                    font=font(9, weight, mono=True))
        self.text.configure(state="disabled")
        self.autoscroll = True

    def append(self, msg: str, level: str = "log") -> None:
        from .i18n import tr
        line = tr(msg).rstrip("\n")
        if not line:
            return
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n", level if level in self.TAGS else "log")
        total = int(self.text.index("end-1c").split(".")[0])
        if total > self.max_lines:
            self.text.delete("1.0", f"{total - self.max_lines}.0")
        self.text.configure(state="disabled")
        if self.autoscroll:
            self.text.see("end")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def dump(self) -> str:
        return self.text.get("1.0", "end-1c")


class LineChart(tk.Canvas):
    """折线图：支持两条序列、自动纵轴、悬停读数、空状态。"""

    def __init__(self, parent: tk.Misc, height: int = 180, bg: str | None = None):
        bgc = bg or P["panel"]
        super().__init__(parent, height=height, bg=bgc, highlightthickness=0, bd=0)
        self._series: list[tuple[str, str, list[tuple[float, float]]]] = []
        self._empty_text = "暂无数据"
        self._hover: tuple[float, float] | None = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", lambda e: self._clear_hover())

    def set_series(self, series: list[tuple[str, str, list[tuple[float, float]]]],
                   empty_text: str = "暂无数据") -> None:
        self._series = [(n, c, list(pts)) for n, c, pts in series if pts]
        self._empty_text = empty_text
        self.redraw()

    def _on_motion(self, event) -> None:
        self._hover = (event.x, event.y)
        self.redraw()

    def _clear_hover(self) -> None:
        self._hover = None
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 40 or h < 40:
            return
        pl, pr, pt, pb = 46, 14, 12, 22
        x0, y0, x1, y1 = pl, pt, w - pr, h - pb
        self.create_line(x0, y0, x0, y1, fill=P["border"])
        self.create_line(x0, y1, x1, y1, fill=P["border"])

        if not self._series:
            self.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=self._empty_text,
                             fill=P["muted"], font=font(9))
            return

        xs = [p[0] for _, _, pts in self._series for p in pts]
        ys = [p[1] for _, _, pts in self._series for p in pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax == xmin:
            xmax = xmin + 1
        pad_y = (ymax - ymin) * 0.15 or max(0.05, abs(ymax) * 0.1)
        ymin, ymax = ymin - pad_y, ymax + pad_y

        def sx(v: float) -> float:
            return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)

        def sy(v: float) -> float:
            return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

        # 网格 + 纵轴刻度
        for i in range(5):
            yv = ymin + (ymax - ymin) * i / 4
            yy = sy(yv)
            self.create_line(x0, yy, x1, yy, fill=P["border"], dash=(2, 4))
            self.create_text(x0 - 6, yy, text=f"{yv:.3f}" if abs(ymax) < 10 else f"{yv:.1f}",
                             anchor="e", fill=P["muted"], font=font(8, mono=True))
        # 横轴刻度
        for i in range(5):
            xv = xmin + (xmax - xmin) * i / 4
            self.create_text(sx(xv), y1 + 10, text=f"{xv:.0f}", fill=P["muted"],
                             font=font(8, mono=True))

        for name, color, pts in self._series:
            col = P.get(color, color)
            if len(pts) > 1:
                coords = []
                for xv, yv in pts:
                    coords.extend([sx(xv), sy(yv)])
                self.create_line(*coords, fill=col, width=2, smooth=len(pts) > 8,
                                 splinesteps=8)
            lastx, lasty = pts[-1]
            self.create_oval(sx(lastx) - 3, sy(lasty) - 3, sx(lastx) + 3, sy(lasty) + 3,
                             fill=col, outline=col)

        # 图例
        lx = x0 + 6
        for name, color, pts in self._series:
            col = P.get(color, color)
            self.create_line(lx, y0 + 6, lx + 14, y0 + 6, fill=col, width=2)
            self.create_text(lx + 19, y0 + 6, text=f"{name} {pts[-1][1]:.4f}", anchor="w",
                             fill=P["text2"], font=font(8))
            lx += 26 + len(name) * 7 + 60

        # 悬停读数
        if self._hover:
            hx, hy = self._hover
            if x0 <= hx <= x1:
                xv = xmin + (hx - x0) / (x1 - x0) * (xmax - xmin)
                self.create_line(hx, y0, hx, y1, fill=P["border2"])
                best = None
                for name, color, pts in self._series:
                    pt = min(pts, key=lambda p: abs(p[0] - xv))
                    self.create_oval(sx(pt[0]) - 3, sy(pt[1]) - 3, sx(pt[0]) + 3, sy(pt[1]) + 3,
                                     fill=P.get(color, color), outline=P["panel"])
                    if best is None or abs(pt[0] - xv) < abs(best[0] - xv):
                        best = pt
                if best:
                    txt = f"step {best[0]:.0f} · {best[1]:.4f}"
                    tw = len(txt) * 7 + 12
                    tx = min(max(hx + 8, x0), x1 - tw)
                    round_rect(self, tx, hy - 22, tx + tw, hy - 4, 5, fill=P["panel2"],
                               outline=P["border2"])
                    self.create_text(tx + 6, hy - 13, text=txt, anchor="w",
                                     fill=P["text"], font=font(8, mono=True))


class Empty(tk.Frame):
    """空状态占位。"""

    def __init__(self, parent: tk.Misc, text: str, sub: str = "", bg: str | None = None):
        bgc = bg or (parent["bg"] if "bg" in parent.keys() else P["bg"])
        super().__init__(parent, bg=bgc)
        tk.Label(self, text="○", bg=bgc, fg=P["border2"], font=font(28)).pack(pady=(18, 6))
        tk.Label(self, text=text, bg=bgc, fg=P["text2"], font=font(10)).pack()
        if sub:
            tk.Label(self, text=sub, bg=bgc, fg=P["muted"], font=font(9)).pack(pady=(4, 0))


class Toast:
    """右下角浮动提示。"""

    _win: tk.Toplevel | None = None

    @classmethod
    def show(cls, root: tk.Misc, text: str, kind: str = "info", ms: int = 2600) -> None:
        try:
            if cls._win is not None:
                cls._win.destroy()
        except tk.TclError:
            pass
        colors = {"info": "accent", "ok": "green", "warn": "amber", "err": "red"}
        color = P.get(colors.get(kind, "accent"), P["accent"])
        win = tk.Toplevel(root)
        cls._win = win
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        frame = tk.Frame(win, bg=P["panel2"], highlightbackground=color, highlightthickness=1)
        frame.pack()
        tk.Frame(frame, bg=color, width=4, height=38).pack(side="left", fill="y")
        tk.Label(frame, text=text, bg=P["panel2"], fg=P["text"], font=font(10),
                 wraplength=420, justify="left", padx=12, pady=10).pack(side="left")
        win.update_idletasks()
        w, h = win.winfo_width(), win.winfo_height()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        win.geometry(f"+{sw - w - 40}+{sh - h - 90}")
        win.after(ms, lambda: cls._dismiss(win))

    @classmethod
    def _dismiss(cls, win: tk.Toplevel) -> None:
        try:
            win.destroy()
        except tk.TclError:
            pass
        if cls._win is win:
            cls._win = None


def hsep(parent: tk.Misc, pad: int = 0) -> tk.Frame:
    f = tk.Frame(parent, bg=P["border"], height=1)
    f.pack(fill="x", pady=pad)
    return f


def vspace(parent: tk.Misc, h: int = 8) -> tk.Frame:
    f = tk.Frame(parent, bg=parent["bg"], height=h)
    f.pack(fill="x")
    return f
