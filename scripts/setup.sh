#!/usr/bin/env bash
# LiuEr 一键启动
# 用法: ./scripts/setup.sh  然后  python -m liuer
set -e
cd "$(dirname "$0")/.."

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
echo "使用 Python: $($PY --version)"

echo "==> 创建虚拟环境"
[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate

echo "==> 安装依赖"
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "==> 初始化数据目录"
mkdir -p data/audio

echo ""
echo "完成。"
echo ""
echo "下一步："
echo "  1) 准备本地模型（首次需要，约 10GB）"
echo "     ollama pull qwen2.5:14b-instruct"
echo "     ollama pull qwen2.5:7b-instruct"
echo "  2) 启动服务"
echo "     python -m liuer"
echo "  3) 打开看板"
echo "     http://127.0.0.1:8787"
echo ""
echo "注：模型未就绪时系统仍可启动，风险检测与情报规则层完全可用。"