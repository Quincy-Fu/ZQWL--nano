#!/usr/bin/env bash
set -euo pipefail

# Jetson Nano 开机自启动安装脚本。
# 用法：
#   bash install_autostart.sh              # 串口自动匹配
#   bash install_autostart.sh /dev/ttyCH341USB0
#   bash install_autostart.sh uninstall
#   bash install_autostart.sh status
#   bash install_autostart.sh log

SERVICE_NAME="zqwl-main.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="/usr/bin/python3"
fi

cmd="${1:-install}"
if [[ "$cmd" == "uninstall" ]]; then
  sudo systemctl disable --now "$SERVICE_NAME" || true
  sudo rm -f "/etc/systemd/system/$SERVICE_NAME"
  sudo systemctl daemon-reload
  echo "已卸载 $SERVICE_NAME"
  exit 0
fi

if [[ "$cmd" == "status" ]]; then
  systemctl status "$SERVICE_NAME" --no-pager || true
  exit 0
fi

if [[ "$cmd" == "log" ]]; then
  journalctl -u "$SERVICE_NAME" -f
  exit 0
fi

serial_arg=""
if [[ "$cmd" == "install" && $# -ge 2 ]]; then
  serial_arg="$2"
elif [[ "$cmd" != "install" ]]; then
  serial_arg="$cmd"
fi

run_user="${SUDO_USER:-$(id -un)}"
exec_start="$PYTHON_BIN $SCRIPT_DIR/main.py"
if [[ -n "$serial_arg" ]]; then
  exec_start="$exec_start $serial_arg"
fi
exec_start="$exec_start --autorun --headless"

sudo tee "/etc/systemd/system/$SERVICE_NAME" >/dev/null <<EOF
[Unit]
Description=ZQWL Jetson Nano main autorun
After=multi-user.target

[Service]
Type=simple
User=$run_user
WorkingDirectory=$SCRIPT_DIR
ExecStart=$exec_start
Restart=always
RestartSec=2
StartLimitIntervalSec=0
KillSignal=SIGINT
TimeoutStopSec=10
Environment=PYTHONUNBUFFERED=1
Environment=ZQWL_HEADLESS=1
Environment=OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT=1
Environment=OPENCV_VIDEOIO_V4L_READ_ATTEMPTS=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "已安装并启动 $SERVICE_NAME"
echo "查看状态: bash $SCRIPT_DIR/install_autostart.sh status"
echo "实时日志: bash $SCRIPT_DIR/install_autostart.sh log"
