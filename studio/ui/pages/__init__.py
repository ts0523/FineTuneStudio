# -*- coding: utf-8 -*-
"""页面注册表。顺序即侧边栏顺序，group 决定归到哪一层。

三层结构：
- 实验台：拿模型直接玩（全模态输入输出、语音通话、视频通话）+ 模型资产 + 导出
- 数据层：造数据（自然语言描述问题与答案就能生成训练集）、体检、切分
- 训练层：填参数开训、AI 帮训闭环、实验复盘
系统层放总览 / 教程 / 工具 / 设置。
"""
from __future__ import annotations

from .dashboard import DashboardPage
from .tutorial import TutorialPage
from .goal import GoalPage
from .models import ModelsPage
from .datasets import DatasetsPage
from .nl_data import NLDataPage
from .train import TrainPage
from .autotrain import AutoTrainPage
from .runs import RunsPage
from .chat import ChatPage
from .call import CallPage
from .export import ExportPage
from .tools import ToolsPage
from .settings import SettingsPage

# 层名 → (副标题, 主题色键)
GROUPS: dict[str, tuple[str, str]] = {
    "实验台": ("体验 · 通话 · 模型", "accent"),
    "数据层": ("造数 · 体检 · 切分", "cyan"),
    "训练层": ("训练 · 帮训 · 复盘", "purple"),
    "系统": ("总览 · 工具 · 设置", "muted"),
}

PAGE_CLASSES = [
    # 实验台
    ChatPage, CallPage, ModelsPage, ExportPage,
    # 数据层
    DatasetsPage, NLDataPage, GoalPage,
    # 训练层
    TrainPage, AutoTrainPage, RunsPage,
    # 系统
    DashboardPage, TutorialPage, ToolsPage, SettingsPage,
]
