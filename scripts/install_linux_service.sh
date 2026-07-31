#!/usr/bin/env bash
# openai-ollama-proxy systemd 服务安装脚本(Linux)
# 用法: sudo bash scripts/install_linux_service.sh
# 可选环境变量: SERVICE_USER=运行服务的用户名(默认 root, 需已存在)
set -euo pipefail

SERVICE_NAME="openai-ollama-proxy"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="/opt/openai-ollama-proxy"
SERVICE_USER="${SERVICE_USER:-root}"

if [[ $EUID -ne 0 ]]; then
    echo "错误: 请以 root 运行(如: sudo bash $0)" >&2
    exit 1
fi

if [[ ! -f "$ROOT_DIR/openai_ollama_proxy.py" ]]; then
    echo "错误: 找不到 openai_ollama_proxy.py, 请在项目根目录执行" >&2
    exit 1
fi

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "错误: 未找到 python3/python, 请先安装" >&2
    exit 1
fi

# 1) 复制项目文件到安装目录
mkdir -p "$INSTALL_DIR"
cp -f "$ROOT_DIR/openai_ollama_proxy.py" "$INSTALL_DIR/"
[[ -f "$ROOT_DIR/config.json" ]] && cp -f "$ROOT_DIR/config.json" "$INSTALL_DIR/"
[[ -d "$ROOT_DIR/models" ]] && cp -rf "$ROOT_DIR/models" "$INSTALL_DIR/"
[[ -f "$ROOT_DIR/requirements.txt" ]] && cp -f "$ROOT_DIR/requirements.txt" "$INSTALL_DIR/"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR" || true

# 2) 生成 systemd 单元
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=OpenAI Compatible API -> Ollama API Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$PYTHON_BIN $INSTALL_DIR/openai_ollama_proxy.py --config $INSTALL_DIR/config.json
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# 3) 重新加载并启动
systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"
echo "已安装并启动: $SERVICE_NAME"
systemctl --no-pager --full status "$SERVICE_NAME"