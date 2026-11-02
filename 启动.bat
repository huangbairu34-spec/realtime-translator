@echo off
chcp 65001 >nul
title 实时翻译软件
cd /d "%~dp0"

set PY="C:\Users\85098\.workbuddy\binaries\python\envs\default\Scripts\pythonw.exe"
set PYCON="C:\Users\85098\.workbuddy\binaries\python\envs\default\Scripts\python.exe"

if "%~1"=="--debug" (
  %PYCON% main.py %2 %3
) else (
  start "" %PY% main.py
)
