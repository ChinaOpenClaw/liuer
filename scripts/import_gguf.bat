@echo off
REM 从国内镜像下载 GGUF 并导入 Ollama
REM 适用：ollama pull 走官方仓库超时 / Bad Gateway 的情况
REM 用法: import_gguf.bat <HF仓库> <文件名> <ollama模型名>
REM 例:   import_gguf.bat bartowski/Qwen_Qwen3.5-4B-GGUF Qwen_Qwen3.5-4B-Q4_K_M.gguf qwen3.5:4b

setlocal enabledelayedexpansion

set REPO=%1
set FILE=%2
set MODELNAME=%3

if "%REPO%"=="" set REPO=bartowski/Qwen_Qwen3.5-4B-GGUF
if "%FILE%"=="" set FILE=Qwen_Qwen3.5-4B-Q4_K_M.gguf
if "%MODELNAME%"=="" set MODELNAME=qwen3.5:4b

set MIRROR=https://hf-mirror.com
set GGUF_DIR=%USERPROFILE%\.ollama\gguf
set OLLAMA_DIR=C:\Users\10099\AppData\Local\Programs\Ollama

if not exist "%GGUF_DIR%" mkdir "%GGUF_DIR%"

echo ==^> 下载 %FILE%
echo     镜像: %MIRROR%/%REPO%
curl -L --progress-bar -o "%GGUF_DIR%\%FILE%" "%MIRROR%/%REPO%/resolve/main/%FILE%"
if errorlevel 1 (
    echo 下载失败
    exit /b 1
)

echo.
echo ==^> 生成 Modelfile
(
echo FROM %GGUF_DIR%\%FILE%
echo PARAMETER num_ctx 4096
echo PARAMETER temperature 0.85
echo PARAMETER top_p 0.9
echo TEMPLATE """{{ if .System }}<|im_start|>system
echo {{ .System }}<|im_end|>
echo {{ end }}{{ range .Messages }}<|im_start|>{{ .Role }}
echo {{ .Content }}<|im_end|>
echo {{ end }}<|im_start|>assistant
echo """
echo PARAMETER stop "<|im_end|>"
) > "%GGUF_DIR%\Modelfile"

echo ==^> 导入 Ollama: %MODELNAME%
cd /d "%OLLAMA_DIR%"
ollama create %MODELNAME% -f "%GGUF_DIR%\Modelfile"
if errorlevel 1 (
    echo 导入失败
    exit /b 1
)

echo.
echo 完成。验证:
echo     ollama run %MODELNAME% "你好"
endlocal