# -*- coding: utf-8 -*-
r"""在项目虚拟环境（.venv\Scripts\python.exe）里运行的后台 worker。

界面进程只负责显示；所有需要 torch 的重活都在这里，通过 stdout 上的
`@@EVT@@{json}` 行与界面通信。
"""
