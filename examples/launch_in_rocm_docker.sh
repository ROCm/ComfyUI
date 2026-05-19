#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot launcher that starts a ROCm-enabled container for the
# amd_comfyui_rocm_tutorial.ipynb notebook.
#
#   * Mounts the local ComfyUI checkout at /workload/ComfyUI (so the
#     notebook sees up-to-date code and any models you have placed in
#     ComfyUI/models/...).
#   * Forwards 8888 (JupyterLab) and 8188 (ComfyUI server) to the host.
#   * Inside the container: installs ComfyUI requirements (skipping torch -
#     the rocm/primus image already ships PyTorch+ROCm) and JupyterLab,
#     then runs run_jupyter_server.sh.
#
# Usage:
#   cd /home/AMD/diptodeb/devel/ComfyUI/examples
#   ./launch_in_rocm_docker.sh
#
# Overrides:
#   IMAGE=rocm/comfyui:comfyui-0.18.2.amd0_rocm7.2.0_ubuntu24.04 \
#     ./launch_in_rocm_docker.sh
#       Use the prebuilt ComfyUI-on-ROCm image instead of primus
#       (everything is pre-installed; container starts much faster).
#
#   GPU_DEVICES=0,1 ./launch_in_rocm_docker.sh
#       Restrict the container to GPUs 0,1 via HIP_VISIBLE_DEVICES.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXAMPLES_DIR="$ROOT/examples"

IMAGE="${IMAGE:-rocm/primus:v26.2}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
CONTAINER_NAME="${CONTAINER_NAME:-comfyui-tutorial-$(id -un)}"

GPU_ENV=()
if [[ -n "${GPU_DEVICES:-}" ]]; then
  GPU_ENV+=( -e "HIP_VISIBLE_DEVICES=${GPU_DEVICES}" )
fi

echo "Launcher settings:"
echo "  IMAGE          = $IMAGE"
echo "  ComfyUI source = $ROOT (mounted -> /workload/ComfyUI)"
echo "  JUPYTER_PORT   = $JUPYTER_PORT"
echo "  COMFYUI_PORT   = $COMFYUI_PORT"
echo "  CONTAINER_NAME = $CONTAINER_NAME"
if [[ -n "${GPU_DEVICES:-}" ]]; then
  echo "  GPU_DEVICES    = $GPU_DEVICES (via HIP_VISIBLE_DEVICES)"
fi
echo

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Removing stale container $CONTAINER_NAME ..."
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# Inside-container script. We run it via `bash -lc "$INNER"` to keep this
# launcher self-contained (no second script file required inside the image).
INNER='set -euo pipefail
echo "[container] python: $(python --version 2>&1)"
echo "[container] torch:  $(python -c "import torch,sys;print(torch.__version__,torch.version.hip)")"
echo "[container] GPU 0:  $(python -c "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"n/a\")")"

PIP_OPTS=( --quiet --timeout 60 --retries 3 )

echo "[container] Installing ComfyUI Python requirements (skipping torch*)..."
REQ=/tmp/comfyui-requirements.txt
grep -vE "^(torch|torchsde|torchvision|torchaudio)$" /workload/ComfyUI/requirements.txt > "$REQ"
pip install "${PIP_OPTS[@]}" -r "$REQ"
pip install "${PIP_OPTS[@]}" torchsde

echo "[container] Installing JupyterLab + mesh-render deps..."
pip install "${PIP_OPTS[@]}" jupyterlab ipywidgets trimesh fast_simplification imageio "imageio[ffmpeg]"

cd /workload/ComfyUI/examples
exec bash run_jupyter_server.sh
'

DOCKER_FLAGS=( --rm )
if [ -t 0 ] && [ -t 1 ]; then
  DOCKER_FLAGS+=( -it )
  echo "Starting container interactively (Ctrl+C inside the JupyterLab process to stop)..."
else
  DOCKER_FLAGS+=( -d )
  echo "stdin is not a TTY; starting container detached. Tail logs with:"
  echo "  docker logs -f $CONTAINER_NAME"
fi
echo

exec docker run "${DOCKER_FLAGS[@]}" \
  --name "$CONTAINER_NAME" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size=16G \
  -p "127.0.0.1:${JUPYTER_PORT}:8888" \
  -p "127.0.0.1:${COMFYUI_PORT}:8188" \
  -v "$ROOT:/workload/ComfyUI" \
  -e COMFYUI_PATH=/workload/ComfyUI \
  -e PYTHONUNBUFFERED=1 \
  "${GPU_ENV[@]}" \
  -w /workload/ComfyUI/examples \
  "$IMAGE" \
  bash -lc "$INNER"
