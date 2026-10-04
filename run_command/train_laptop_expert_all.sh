#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CASES=(case1 case2 case3)
AGENTS=(Boulware Conceder Linear Atlas3)

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
  elif [[ -x "$PROJECT_DIR/../Transformer-based/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/../Transformer-based/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-0}"
TIMESTEPS="${TIMESTEPS:-50000}"
NUM_ENVS="${NUM_ENVS:-4}"
HORIZON="${HORIZON:-512}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PPO_EPOCHS="${PPO_EPOCHS:-10}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"

checkpoint_step() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import torch

try:
    checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(int(checkpoint.get("global_step", -1)))
except Exception:
    print(-1)
PY
}

run_training() {
  local case_name="$1" agent0="$2" agent1="$3" model_dir="$4"
  local checkpoint="$model_dir/checkpoint.pt" step=-1
  if [[ -f "$checkpoint" ]]; then
    step="$(checkpoint_step "$checkpoint")"
  fi
  if (( step >= TIMESTEPS )); then
    echo "[${completed_jobs}/${total_jobs}] complete (${step}/${TIMESTEPS}): ${case_name} ${agent0}-${agent1}"
    completed_jobs=$((completed_jobs + 1))
    return
  fi

  local -a command=(
    "$PYTHON_BIN" train.py
    --total-timesteps "$TIMESTEPS"
    --device "$DEVICE"
  )
  if (( step >= 0 )); then
    command+=(--resume "$model_dir")
    echo "[${completed_jobs}/${total_jobs}] resume (${step}/${TIMESTEPS}): ${case_name} ${agent0}-${agent1}"
  else
    local -a pool_agents=("$agent0") duplicate_option=(--allow-duplicate-opponents)
    if [[ "$agent0" != "$agent1" ]]; then
      pool_agents+=("$agent1")
      duplicate_option=(--no-allow-duplicate-opponents)
    fi
    command+=(
      -a "${pool_agents[@]}"
      -i Laptop
      --model-type expert
      --utility-cases "$case_name"
      --save-path "$model_dir"
      --seed "$SEED"
      --num-envs "$NUM_ENVS"
      --horizon "$HORIZON"
      --batch-size "$BATCH_SIZE"
      --n-epochs "$PPO_EPOCHS"
      --pool-eval-freq 10000
      --snapshot-freq 10000
      --pool-eval-episodes 4
      --max-pool-size 16
      --self-play-probability 0.0
      --scripted-probability 0.5
      --snapshot-probability 0.5
      --pfsp-alpha 2.0
      --uniform-mix 0.1
      "${duplicate_option[@]}"
    )
    echo "[${completed_jobs}/${total_jobs}] start (0/${TIMESTEPS}): ${case_name} ${agent0}-${agent1}"
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  %q' "${command[@]}"
    printf '\n'
  else
    mkdir -p "$model_dir"
    "${command[@]}"
  fi
  completed_jobs=$((completed_jobs + 1))
}

total_jobs=$(( ${#CASES[@]} * (${#AGENTS[@]} * (${#AGENTS[@]} + 1) / 2) ))
completed_jobs=0

for case_name in "${CASES[@]}"; do
  for ((i = 0; i < ${#AGENTS[@]}; i++)); do
    for ((j = i; j < ${#AGENTS[@]}; j++)); do
      if (( LIMIT > 0 && completed_jobs >= LIMIT )); then
        echo "Reached LIMIT=$LIMIT"
        exit 0
      fi
      agent0="${AGENTS[$i]}"
      agent1="${AGENTS[$j]}"
      pair="${agent0}-${agent1}"
      model_dir="$RESULTS_ROOT/$case_name/models/expert/$pair/Laptop/TransformerSelfPlay_Negotiator"
      run_training "$case_name" "$agent0" "$agent1" "$model_dir"
    done
  done
done

mkdir -p "$RESULTS_ROOT/experiment_status"
if [[ "$DRY_RUN" != "1" ]]; then
  touch "$RESULTS_ROOT/experiment_status/laptop-expert-training.done"
fi
echo "Laptop expert training complete: jobs=$completed_jobs/$total_jobs"
