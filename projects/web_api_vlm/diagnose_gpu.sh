#!/bin/bash

echo "=== GPU诊断脚本 ==="

echo -n "1. 检查NVIDIA GPU: "
if lspci | grep -qi nvidia; then
    echo "✓ 找到NVIDIA GPU"
    lspci | grep -i nvidia
else
    echo "✗ 未找到NVIDIA GPU"
fi

echo -n "2. 检查NVIDIA驱动: "
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA驱动已安装"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "✗ NVIDIA驱动未安装"
fi

echo -n "3. 检查Docker NVIDIA支持: "
if docker info 2>/dev/null | grep -qi nvidia; then
    echo "✓ Docker支持NVIDIA"
else
    echo "✗ Docker不支持NVIDIA"
fi

echo -n "4. 检查nvidia-container-toolkit: "
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo "✓ nvidia-container-toolkit已安装"
else
    echo "✗ nvidia-container-toolkit未安装"
fi

echo
echo "=== 建议 ==="
if ! lspci | grep -qi nvidia; then
    echo "- 系统没有NVIDIA GPU，建议使用CPU模式"
    echo "- 运行: docker-compose -f docker-compose.cpu.yml up --build"
elif ! command -v nvidia-smi &> /dev/null; then
    echo "- 需要安装NVIDIA驱动"
elif ! docker info 2>/dev/null | grep -qi nvidia; then
    echo "- 需要安装nvidia-container-toolkit"
    echo "- 参考安装命令在上面的方案一中"
else
    echo "- GPU环境配置正常，可以使用GPU模式"
    echo "- 运行: docker-compose -f docker-compose.gpu.yml up --build"
fi 