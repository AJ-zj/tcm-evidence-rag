@echo off
chcp 65001 >nul
title TCM Evidence RAG Server
cd /d "%~dp0"
echo ================================================
echo   中医循证检索增强问答系统 - 服务启动
echo   启动后访问 http://127.0.0.1:8100
echo   关闭本窗口即停止服务
echo ================================================
".venv\Scripts\python.exe" scripts\serve.py --port 8100
pause
