#!/usr/bin/env bash
# openai-ollama-proxy systemd 服务卸载脚本(Linux)
# 用法: sudo bash scripts/uninstall_linux_service.sh
set -euo pipefail

SERVICE_NAME="openai-ollama-proxy"

if [[ $EUID -ne 0 ]]; then
    echo "错误: 请以 root 运行(如: sudo bash $0)" >&2
    exit 1
fi

systemctl disable --now "$SERVICE_NAME" 2>/dev/null || true
rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
systemctl daemon-reload
echo "已卸载 $SERVICE_NAME"
echo "提示: 安装目录 /opt/openai-ollama-proxy 未删除, 如需删除请手动执行: rm -rf /opt/openai-ollama-proxy"