# -*- coding: utf-8 -*-
"""配色、字体、ttk 样式。

默认主题是「白色森林」：白底 + 森林绿点缀，长时间看日志/曲线不刺眼。
另外保留 深色 / 浅色 / 纯黑 三套，随时切换。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

DARK = {
    "bg": "#0E1116",
    "sidebar": "#12171F",
    "panel": "#161C25",
    "panel2": "#1D2530",
    "panel3": "#232D3A",
    "border": "#26303D",
    "border2": "#36445A",
    "text": "#E9EFF8",
    "text2": "#A9B7C8",
    "muted": "#75839A",
    "accent": "#4C8DFF",
    "accent_hi": "#6BA2FF",
    "accent_dim": "#2C5BB8",
    "green": "#33D6A0",
    "amber": "#F5B440",
    "red": "#F2606B",
    "purple": "#A98BFA",
    "cyan": "#41C7D9",
    "chip": "#1B2330",
}

LIGHT = {
    "bg": "#F4F6FA",
    "sidebar": "#FFFFFF",
    "panel": "#FFFFFF",
    "panel2": "#F0F3F8",
    "panel3": "#E6EBF3",
    "border": "#DCE3EC",
    "border2": "#C3CDDB",
    "text": "#16202C",
    "text2": "#43536471",
    "muted": "#6B7A8D",
    "accent": "#2F6BD8",
    "accent_hi": "#4A83E8",
    "accent_dim": "#C9D9F6",
    "green": "#12A47A",
    "amber": "#C98209",
    "red": "#D93A47",
    "purple": "#7C55D8",
    "cyan": "#1D93A5",
    "chip": "#EDF1F7",
}
LIGHT["text2"] = "#435364"

# 白色森林（默认）：接近纯白的底 + 森林绿主色，绿色只出现在该聚焦的地方
FOREST = {
    "bg": "#F7FAF5",
    "sidebar": "#EFF5EA",
    "panel": "#FFFFFF",
    "panel2": "#F3F8EE",
    "panel3": "#E6EEDF",
    "border": "#D8E2CE",
    "border2": "#BACBAD",
    "text": "#17251A",
    "text2": "#41544A",
    "muted": "#6D8375",
    "accent": "#2E7D5B",
    "accent_hi": "#3E9A72",
    "accent_dim": "#D6E9DC",
    "green": "#1C8A5C",
    "amber": "#A96F12",
    "red": "#C0392B",
    "purple": "#5C5390",
    "cyan": "#17737E",
    "chip": "#EEF4E8",
}

# 黑：纯黑底 + 高对比，适合全黑环境 / OLED
BLACK = {
    "bg": "#000000",
    "sidebar": "#080808",
    "panel": "#0D0D0D",
    "panel2": "#151515",
    "panel3": "#1E1E1E",
    "border": "#242424",
    "border2": "#3A3A3A",
    "text": "#F2F4F6",
    "text2": "#A7B0BA",
    "muted": "#6C7681",
    "accent": "#5B9BFF",
    "accent_hi": "#7DB2FF",
    "accent_dim": "#1E3A66",
    "green": "#2FD69A",
    "amber": "#F5B440",
    "red": "#FF6B75",
    "purple": "#B08BFF",
    "cyan": "#49D0E0",
    "chip": "#141414",
}

# 名字 → (中文名, 说明, 调色板)
PRESETS: dict[str, tuple[str, str, dict]] = {
    "forest": ("白色森林", "默认，白底 + 森林绿，护眼、长时间看日志不累", FOREST),
    "light": ("浅色", "白底蓝调，白天办公环境更清晰", LIGHT),
    "dark": ("深色", "暗底蓝调，长时间看代码/日志舒服", DARK),
    "black": ("纯黑", "纯黑底，夜间/OLED 省电，对比度最高", BLACK),
}

DEFAULT_THEME = "forest"

# 允许用户自由调的键
EDITABLE = ["bg", "panel", "panel2", "border", "text", "text2", "accent"]

P: dict[str, str] = dict(FOREST)
MODE = DEFAULT_THEME


# ---------------------------------------------------------------- 颜色工具
def _hex2rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb2hex(rgb) -> str:
    return "#%02X%02X%02X" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


def mix(c1: str, c2: str, t: float) -> str:
    """把 c1 往 c2 混合 t（0~1）。"""
    a, b = _hex2rgb(c1), _hex2rgb(c2)
    return _rgb2hex([a[i] + (b[i] - a[i]) * t for i in range(3)])


def adjust(c: str, f: float) -> str:
    """f>0 提亮，f<0 压暗（相对黑白）。"""
    target = "#FFFFFF" if f > 0 else "#000000"
    return mix(c, target, min(1.0, abs(f)))


def build_palette(name: str, custom: dict | None = None) -> dict[str, str]:
    """按预设 + 自定义覆盖，算出完整调色板（派生色跟着主色/面板自动重算）。"""
    key = name if name in PRESETS else DEFAULT_THEME
    base = dict(PRESETS[key][2])
    custom = {k: v for k, v in (custom or {}).items()
              if k in EDITABLE and isinstance(v, str) and v.startswith("#")}
    for k, v in custom.items():
        base[k] = v
    if "accent" in custom:
        base["accent_hi"] = adjust(base["accent"], 0.18)
        base["accent_dim"] = mix(base["accent"], base["bg"], 0.62)
    if "panel2" in custom:
        base["panel3"] = mix(base["panel2"], base["text"], 0.07)
        base["chip"] = base["panel2"]
    if "border" in custom:
        base["border2"] = mix(base["border"], base["text"], 0.18)
    if "panel" in custom:
        base["sidebar"] = mix(base["panel"], base["bg"], 0.55)
    return base


def apply(name: str, custom: dict | None = None) -> None:
    """切换/微调配色（就地改写 P，页面里的引用才会一起变）。"""
    global MODE
    MODE = name if name in PRESETS else DEFAULT_THEME
    P.clear()
    P.update(build_palette(MODE, custom))

UI_FAMILY = "Microsoft YaHei UI"
MONO_FAMILY = "Consolas"


def set_theme(name: str, custom: dict | None = None) -> None:
    """兼容旧调用：等价于 apply(name, custom)。"""
    apply(name, custom)


def preset_choices() -> list[tuple[str, str, str]]:
    """[(key, 中文名, 说明)]，按展示顺序。"""
    return [(k, v[0], v[1]) for k, v in PRESETS.items()]


def current_palette() -> dict[str, str]:
    return dict(P)


def _pick_family(root: tk.Misc, candidates: list[str], fallback: str) -> str:
    try:
        fams = {f.lower() for f in tkfont.families(root)}
    except Exception:  # noqa: BLE001
        return fallback
    for c in candidates:
        if c.lower() in fams:
            return c
    return fallback


def font(size: int = 10, weight: str = "normal", mono: bool = False) -> tuple:
    fam = MONO_FAMILY if mono else UI_FAMILY
    return (fam, size, weight)


def init_fonts(root: tk.Misc) -> None:
    global UI_FAMILY, MONO_FAMILY
    UI_FAMILY = _pick_family(root, ["Microsoft YaHei UI", "微软雅黑", "Microsoft YaHei",
                                    "PingFang SC", "Segoe UI"], "Segoe UI")
    MONO_FAMILY = _pick_family(root, ["Cascadia Mono", "Consolas", "Courier New"], "Courier New")
    try:
        base = tkfont.nametofont("TkDefaultFont")
        base.configure(family=UI_FAMILY, size=10)
        tkfont.nametofont("TkTextFont").configure(family=UI_FAMILY, size=10)
        tkfont.nametofont("TkMenuFont").configure(family=UI_FAMILY, size=10)
        tkfont.nametofont("TkHeadingFont").configure(family=UI_FAMILY, size=10, weight="bold")
    except Exception:  # noqa: BLE001
        pass


def round_rect(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float, r: float,
               fill: str, outline: str = "", width: int = 1, **kw) -> int:
    """在 canvas 上画圆角矩形（用平滑多边形近似），返回 item id。"""
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, splinesteps=12, fill=fill,
                                 outline=outline or fill, width=width, **kw)


def apply_ttk(root: tk.Misc) -> None:
    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass

    st.configure(".", background=P["bg"], foreground=P["text"], fieldbackground=P["panel2"],
                 bordercolor=P["border"], lightcolor=P["border"], darkcolor=P["border"],
                 troughcolor=P["panel2"], focuscolor=P["accent"])

    st.configure("TCombobox", fieldbackground=P["panel2"], background=P["panel2"],
                 foreground=P["text"], arrowcolor=P["text2"], borderwidth=1, relief="flat",
                 padding=5)
    st.map("TCombobox",
           fieldbackground=[("readonly", P["panel2"]), ("disabled", P["panel3"])],
           foreground=[("disabled", P["muted"])],
           selectbackground=[("readonly", P["panel2"])],
           selectforeground=[("readonly", P["text"])],
           bordercolor=[("focus", P["accent"]), ("!focus", P["border"])])
    root.option_add("*TCombobox*Listbox.background", P["panel2"])
    root.option_add("*TCombobox*Listbox.foreground", P["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", P["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
    root.option_add("*TCombobox*Listbox.font", font(10))

    st.configure("TSpinbox", fieldbackground=P["panel2"], background=P["panel2"],
                 foreground=P["text"], arrowcolor=P["text2"], borderwidth=1, padding=4)
    st.map("TSpinbox", foreground=[("disabled", P["muted"])],
           fieldbackground=[("disabled", P["panel3"])])

    st.configure("TCheckbutton", background=P["bg"], foreground=P["text2"], focuscolor=P["bg"])
    st.map("TCheckbutton",
           background=[("active", P["bg"])],
           indicatorcolor=[("selected", P["accent"]), ("!selected", P["panel2"])],
           foreground=[("active", P["text"])])

    st.configure("TRadiobutton", background=P["bg"], foreground=P["text2"])
    st.map("TRadiobutton", background=[("active", P["bg"])],
           indicatorcolor=[("selected", P["accent"]), ("!selected", P["panel2"])])

    st.configure("Treeview", background=P["panel"], fieldbackground=P["panel"],
                 foreground=P["text"], borderwidth=0, rowheight=28, font=font(10))
    st.configure("Treeview.Heading", background=P["panel2"], foreground=P["text2"],
                 relief="flat", font=font(9, "bold"), padding=(8, 6))
    st.map("Treeview.Heading", background=[("active", P["panel3"])])
    st.map("Treeview", background=[("selected", P["accent_dim"])],
           foreground=[("selected", "#FFFFFF")])

    st.configure("TScrollbar", background=P["panel2"], troughcolor=P["bg"],
                 bordercolor=P["bg"], arrowcolor=P["muted"], width=10)
    st.map("TScrollbar", background=[("active", P["panel3"])])

    st.configure("TNotebook", background=P["bg"], borderwidth=0)
    st.configure("TNotebook.Tab", background=P["panel2"], foreground=P["text2"],
                 padding=(14, 7), borderwidth=0)
    st.map("TNotebook.Tab", background=[("selected", P["panel"])],
           foreground=[("selected", P["text"])])

    st.configure("TSeparator", background=P["border"])
