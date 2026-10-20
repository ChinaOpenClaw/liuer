@echo off
REM LiuEr Windows 一键启动
cd /d "%~dp0.."

where python >nul 2>nul || (echo 未找到 Python & exit /b 1)

echo ==^> 创建虚拟环境
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat

echo ==^> 安装依赖
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ==^> 初始化数据目录
if not exist data mkdir data
if not exist data\audio mkdir data\audio

echo.
echo 完成。
echo.
echo 下一步：
echo   1) 准备本地模型
echo      ollama pull qwen2.5:14b-instruct
echo      ollama pull qwen2.5:7b-instruct
echo   2) 启动服务
echo      python -m liuer
echo   3) 打开看板
echo      http://127.0.0.1:8787
echo.
echo 注：模型未就绪时系统仍可启动，风险检测与情报规则层完全可用。