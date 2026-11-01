@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title AI-Finetune: chat

set MODEL=
if exist "outputs\merged\config.json" set MODEL=--model outputs\merged
set ADAPTER=
if exist "outputs\lora-run\adapter_config.json" set ADAPTER=--adapter outputs\lora-run

echo ================================================================
echo  Base model : auto detect (or outputs\merged when present)
echo  Adapter    : %ADAPTER%
echo  Type your message, "exit" to quit, "/reset" to clear context.
echo ================================================================
echo.

".venv\Scripts\python.exe" scripts\chat.py %MODEL% %ADAPTER%
pause
