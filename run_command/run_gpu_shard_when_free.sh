#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="$1"
SHARD_INDEX="$2"
IMAGE="$3"
CONTAINER_NAME="$4"
WAIT_FOR_GPU="$5"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while true; do
  processes="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -z "$processes" ]]; then
    break
  fi
  if [[ "$WAIT_FOR_GPU" != "1" ]]; then
    echo "GPU $GPU_INDEX is not empty; active compute PID(s): $(tr '\n' ' ' <<<"$processes")" >&2
    exit 1
  fi
  echo "$(date -Is) waiting for GPU $GPU_INDEX; active compute PID(s): $(tr '\n' ' ' <<<"$processes")"
  sleep 60
done

echo "$(date -Is) GPU $GPU_INDEX is free; running CUDA smoke test"
docker run --rm --gpus "device=$GPU_INDEX" --entrypoint python "$IMAGE" -c \
  'import torch; assert torch.cuda.is_available(); x=torch.ones(8, device="cuda"); assert x.sum().item()==8; print(torch.__version__, torch.cuda.get_device_name(0))'

# Recheck after the smoke-test container exits so a newly arrived compute job
# cannot be silently overlapped.
processes="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "$processes" ]]; then
  echo "GPU $GPU_INDEX became busy during startup; refusing to overlap" >&2
  exit 1
fi

echo "$(date -Is) starting training shard $SHARD_INDEX on GPU $GPU_INDEX"
exec docker run --rm --name "$CONTAINER_NAME" --gpus "device=$GPU_INDEX" \
  -v /home/nakata:/home/nakata -w "$PROJECT_DIR" \
  -e DEVICE=cuda -e SHARD_INDEX="$SHARD_INDEX" -e SHARD_COUNT=2 \
  "$IMAGE" bash run_command/run_case_experiments.sh
