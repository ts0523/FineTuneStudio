@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title AI-Finetune: LoRA training

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] venv not found. Run setup_env.ps1 first.
  pause
  exit /b 1
)

echo ================================================================
echo  Base model : auto detect (prefer models\minimind-3o)
echo  Data       : pass --data data\your_set.jsonl (or create one in the app)
echo  Output     : outputs\lora-run (override with --output)
echo  Extra args : %*
echo ================================================================
echo.

".venv\Scripts\python.exe" scripts\train_lora.py %*

echo.
echo Done. Adapter saved under outputs\.
pause
