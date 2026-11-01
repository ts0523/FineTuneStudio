@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title AI-Finetune: env check

".venv\Scripts\python.exe" scripts\check_env.py %*
pause
