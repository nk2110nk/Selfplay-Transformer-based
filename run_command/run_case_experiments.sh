#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
DEVICE="${DEVICE:-cuda}"
SEED=0
EXPERT_STEPS="${EXPERT_STEPS:-100000}"
GENERAL_STEPS="${GENERAL_STEPS:-300000}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DRY_RUN="${DRY_RUN:-0}"

CASES=(case1 case2 case3)
DOMAINS=(Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A)
UNKNOWN_DOMAINS=(Coffee Camera Lunch SmartPhone Kitchen)
AGENTS=(Boulware Conceder Linear Atlas3)

if (( SHARD_COUNT < 1 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "Invalid shard: SHARD_INDEX=$SHARD_INDEX SHARD_COUNT=$SHARD_COUNT" >&2
  exit 2
fi

checkpoint_step() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state.get("global_step", -1)))
PY
}

run_job() {
  local job_index="$1" label="$2" target="$3" model_dir="$4"
  shift 4
  local -a new_args=("$@")
  if (( job_index % SHARD_COUNT != SHARD_INDEX )); then
    return
  fi

  local checkpoint="$model_dir/checkpoint.pt" step=-1
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN job=%03d shard=%d/%d %s:' "$job_index" "$SHARD_INDEX" "$SHARD_COUNT" "$label"
    printf ' %q' "$PYTHON_BIN" train.py "${new_args[@]}"
    printf '\n'
    return
  fi

  if [[ -f "$checkpoint" ]]; then
    if ! step="$(checkpoint_step "$checkpoint")"; then
      echo "Unreadable checkpoint, refusing to overwrite: $checkpoint" >&2
      exit 1
    fi
    if (( step >= target )); then
      echo "SKIP complete ($step/$target): $label"
      return
    fi
    echo "RESUME ($step/$target): $label"
    "$PYTHON_BIN" train.py --resume "$model_dir" --total-timesteps "$target" --device "$DEVICE"
    return
  fi
  if [[ -d "$model_dir" ]] && find "$model_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Non-empty model directory has no checkpoint; refusing to overwrite: $model_dir" >&2
    exit 1
  fi
  echo "START (0/$target): $label"
  mkdir -p "$model_dir"
  "$PYTHON_BIN" train.py "${new_args[@]}"
}

job_index=0
for case_name in "${CASES[@]}"; do
  for domain in "${DOMAINS[@]}"; do
    for ((i = 0; i < ${#AGENTS[@]}; i++)); do
      for ((j = i; j < ${#AGENTS[@]}; j++)); do
        agent0="${AGENTS[$i]}"
        agent1="${AGENTS[$j]}"
        pair="$agent0-$agent1"
        model_dir="$RESULTS_ROOT/$case_name/models/expert/$pair/$domain/Transformer_Negotiator"
        pool_agents=("$agent0")
        duplicate_flag=(--allow-duplicate-opponents)
        if [[ "$agent0" != "$agent1" ]]; then
          pool_agents+=("$agent1")
          duplicate_flag=(--no-allow-duplicate-opponents)
        fi
        args=(
          -a "${pool_agents[@]}" -i "$domain"
          --model-type expert --utility-cases "$case_name"
          --save-path "$model_dir" --seed "$SEED"
          --total-timesteps "$EXPERT_STEPS" --device "$DEVICE"
          --pool-eval-freq 10000 --pool-eval-episodes 4 --snapshot-freq 10000
          --max-pool-size 16 --self-play-probability 0.0
          --scripted-probability 0.5 --snapshot-probability 0.5
          --pfsp-alpha 1.0 --uniform-mix 0.1
          --dominance-tolerance 0.01 --min-agreement-rate 0.0
          --score-length-weight -0.005 "${duplicate_flag[@]}"
        )
        run_job "$job_index" "$case_name expert $pair $domain" "$EXPERT_STEPS" "$model_dir" "${args[@]}"
        job_index=$((job_index + 1))
      done
    done
  done

  model_dir="$RESULTS_ROOT/$case_name/models/general/Transformer_Negotiator"
  args=(
    -a "${AGENTS[@]}" -i "${DOMAINS[@]}"
    --model-type general --utility-cases "$case_name"
    --compatible-domains "${UNKNOWN_DOMAINS[@]}"
    --save-path "$model_dir" --seed "$SEED"
    --total-timesteps "$GENERAL_STEPS" --device "$DEVICE"
    --pool-eval-freq 10000 --pool-eval-episodes 4 --snapshot-freq 10000
    --max-pool-size 16 --self-play-probability 0.0
    --scripted-probability 0.5 --snapshot-probability 0.5
    --pfsp-alpha 1.0 --uniform-mix 0.1
    --dominance-tolerance 0.01 --min-agreement-rate 0.0
    --score-length-weight -0.005 --allow-duplicate-opponents
  )
  run_job "$job_index" "$case_name general" "$GENERAL_STEPS" "$model_dir" "${args[@]}"
  job_index=$((job_index + 1))
done

echo "Training shard complete: shard=$SHARD_INDEX/$SHARD_COUNT total_jobs=$job_index"
