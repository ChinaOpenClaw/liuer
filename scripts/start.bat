@echo off
REM LiuEr + Ollama 一键启动 (Windows)
REM 解决 ollama pull 报 "timed out waiting for server to start" 的问题：
REM 1) 先确保 Ollama 服务在跑  2) 本地回环不走代理

setlocal

set OLLAMA_DIR=C:\Users\10099\AppData\Local\Programs\Ollama
set PROJECT_DIR=%~dp0..

REM 本地回环不走代理，否则 CLI 连自己的 11434 端口也会被代理拦截
set NO_PROXY=localhost,127.0.0.1,::1
set no_proxy=localhost,127.0.0.1,::1

echo ==^> 检查 Ollama 服务
netstat -ano | findstr ":11434" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo    Ollama 服务已在运行
) else (
    echo    Ollama 服务未运行，正在启动...
    start "Ollama Server" /MIN "%OLLAMA_DIR%\ollama.exe" serve
    echo    等待服务就绪
    :wait
    timeout /t 3 >nul
    netstat -ano | findstr ":11434" | findstr "LISTENING" >nul 2>&1
    if not %errorlevel%==0 goto wait
    echo    Ollama 服务已就绪
)

echo.
echo ==^> 检查模型
"%OLLAMA_DIR%\ollama.exe" list | findstr "qwen3.5" >nul 2>&1
if %errorlevel%==0 (
    echo    模型已存在
) else (
    echo    未找到模型，开始拉取
    echo    显存提示：当前可用显存不足 6.5GB 时建议用 4b
    set /p MODEL="输入要拉取的模型 [qwen3.5:4b / qwen3.5:9b] (默认 4b): "
    if "!MODEL!"=="" set MODEL=qwen3.5:4b
    "%OLLAMA_DIR%\ollama.exe" pull !MODEL!
)

echo.
echo ==^> 显存状态
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

echo.
echo ==^> 启动 LiuEr
cd /d "%PROJECT_DIR%"
python -m liuer --port 8787

endlocal