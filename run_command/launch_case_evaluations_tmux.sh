#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION_NAME:-selfplay-transformer-evaluation}"
IMAGE="${IMAGE:-selfplay-transformer-cases:latest}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
CONTAINERS=(selfplay-transformer-eval-gpu0 selfplay-transformer-eval-gpu1)

command -v tmux >/dev/null || { echo "tmux is not installed" >&2; exit 1; }
command -v docker >/dev/null || { echo "docker is not installed" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is not installed" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Missing Docker image: $IMAGE" >&2
  exit 1
fi

for gpu in 0 1; do
  if docker container inspect "${CONTAINERS[$gpu]}" >/dev/null 2>&1; then
    echo "Docker container name is already in use: ${CONTAINERS[$gpu]}" >&2
    exit 1
  fi
  processes="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -n "$processes" ]]; then
    echo "GPU $gpu is not empty; active compute PID(s): $(tr '\n' ' ' <<<"$processes")" >&2
    exit 1
  fi
  docker run --rm --gpus "device=$gpu" --entrypoint python "$IMAGE" -c \
    'import torch; assert torch.cuda.is_available(); x=torch.ones(8, device="cuda"); assert x.sum().item()==8; print(torch.__version__, torch.cuda.get_device_name(0))'
done

mkdir -p "$RESULTS_ROOT/experiment_logs"
launch_command() {
  local gpu="$1" shard="$2" log="$3" container="$4"
  printf 'docker run --rm --name %q --gpus device=%q -v %q:/home/nakata -w %q -e DEVICE=cuda -e SHARD_INDEX=%q -e SHARD_COUNT=2 %q bash run_command/evaluate_case_experiments.sh 2>&1 | tee -a %q' \
    "$container" "$gpu" /home/nakata "$PROJECT_DIR" "$shard" "$IMAGE" "$log"
}

gpu0_command="$(launch_command 0 0 "$RESULTS_ROOT/experiment_logs/evaluate-gpu0.log" "${CONTAINERS[0]}")"
gpu1_command="$(launch_command 1 1 "$RESULTS_ROOT/experiment_logs/evaluate-gpu1.log" "${CONTAINERS[1]}")"
tmux new-session -d -s "$SESSION" -n gpu0 "$gpu0_command"
tmux new-window -t "$SESSION" -n gpu1 "$gpu1_command"

echo "Started $SESSION"
echo "Attach with: tmux attach -t $SESSION"
