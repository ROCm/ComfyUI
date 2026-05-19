#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Start JupyterLab for amd_comfyui_rocm_tutorial.ipynb.
#
# Usage (from inside a ROCm-enabled Python environment):
#   cd /workload/ComfyUI/examples
#   ./run_jupyter_server.sh
#
# By default the server listens on 0.0.0.0 inside the container so that
# Docker port forwarding (`-p 8888:8888`) can reach it from the host.
#
# Open http://127.0.0.1:8888/lab?token=... in a browser. For remote nodes:
#   ssh -L 8888:localhost:8888 -L 8188:localhost:8188 user@node
#
# Overrides:
#   JUPYTER_PORT=8890 ./run_jupyter_server.sh   # use a different port
#   JUPYTER_IP=127.0.0.1 ./run_jupyter_server.sh # bind only to loopback
#
# Notes:
#   * Port 8188 is reserved for the ComfyUI server that the notebook
#     launches in cell 5 (subprocess.Popen ... main.py --listen --port 8188).
#     Make sure the container is started with `-p 8188:8188` if you want
#     to also reach the ComfyUI web UI from your host.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! python -c "import jupyterlab" 2>/dev/null; then
  echo "Installing jupyterlab into the current Python environment..."
  python -m pip install --quiet jupyterlab
fi

PORT="${JUPYTER_PORT:-8888}"
IP="${JUPYTER_IP:-0.0.0.0}"

echo "Starting JupyterLab from: $ROOT"
echo "  URL: http://127.0.0.1:${PORT}/lab (use SSH -L if remote)"
echo "  ComfyUI server (started by the notebook) will listen on :8188"
echo "  Stop: Ctrl+C"
echo ""

exec python -m jupyter lab \
  --no-browser \
  --ip="$IP" \
  --port="$PORT" \
  --allow-root \
  --notebook-dir="$ROOT" \
  --ServerApp.terminado_settings='{"shell_command": ["/bin/bash"]}' \
  "$@"
