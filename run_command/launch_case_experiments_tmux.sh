#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="selfplay-transformer-cases"
IMAGE="${IMAGE:-selfplay-transformer-cases:latest}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-0}"

command -v tmux >/dev/null || { echo "tmux is not installed" >&2; exit 1; }
command -v docker >/dev/null || { echo "docker is not installed" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is not installed" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
for name in selfplay-transformer-gpu0 selfplay-transformer-gpu1; do
  if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Docker container name is already in use: $name" >&2
    exit 1
  fi
done

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -f "$PROJECT_DIR/Dockerfile.case-experiments" -t "$IMAGE" "$PROJECT_DIR"
fi

if [[ "$WAIT_FOR_GPUS" != "1" ]]; then
  for gpu in 0 1; do
    processes="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
    if [[ -n "$processes" ]]; then
      echo "GPU $gpu is not empty; active compute PID(s): $(tr '\n' ' ' <<<"$processes")" >&2
      exit 1
    fi
    docker run --rm --gpus "device=$gpu" --entrypoint python "$IMAGE" -c \
      'import torch; assert torch.cuda.is_available(); x=torch.ones(8, device="cuda"); assert x.sum().item()==8; print(torch.__version__, torch.cuda.get_device_name(0))'
  done
fi

mkdir -p "$RESULTS_ROOT/experiment_logs"
launch_command() {
  local gpu="$1" shard="$2" log="$3" name="$4"
  printf 'bash %q %q %q %q %q %q 2>&1 | tee -a %q' \
    "$PROJECT_DIR/run_command/run_gpu_shard_when_free.sh" \
    "$gpu" "$shard" "$IMAGE" "$name" "$WAIT_FOR_GPUS" "$log"
}

gpu0_command="$(launch_command 0 0 "$RESULTS_ROOT/experiment_logs/train-gpu0.log" selfplay-transformer-gpu0)"
gpu1_command="$(launch_command 1 1 "$RESULTS_ROOT/experiment_logs/train-gpu1.log" selfplay-transformer-gpu1)"
tmux new-session -d -s "$SESSION" -n gpu0 "$gpu0_command"
tmux new-window -t "$SESSION" -n gpu1 "$gpu1_command"

echo "Started $SESSION"
echo "Attach with: tmux attach -t $SESSION"
