#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
DEVICE="${DEVICE:-cuda}"
EPISODES="${EPISODES:-100}"
SEED=0
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
EXPERT_STEPS="${EXPERT_STEPS:-100000}"
GENERAL_STEPS="${GENERAL_STEPS:-300000}"

CASES=(case1 case2 case3)
KNOWN_DOMAINS=(Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A)
UNKNOWN_DOMAINS=(Coffee Camera Lunch SmartPhone Kitchen)
ALL_DOMAINS=("${KNOWN_DOMAINS[@]}" "${UNKNOWN_DOMAINS[@]}")
AGENTS=(Boulware Conceder Linear Atlas3)

if (( SHARD_COUNT < 1 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "Invalid shard: SHARD_INDEX=$SHARD_INDEX SHARD_COUNT=$SHARD_COUNT" >&2
  exit 2
fi

validate_checkpoint() {
  "$PYTHON_BIN" - "$1" "$2" "$3" "$4" <<'PY'
import sys
import torch

path, target, expected_case, expected_mode = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
state = torch.load(path, map_location="cpu", weights_only=False)
config = state.get("config", {})
errors = []
if int(state.get("global_step", -1)) < target:
    errors.append(f"step={state.get('global_step')} target={target}")
if config.get("utility_cases") != [expected_case]:
    errors.append(f"cases={config.get('utility_cases')} expected={[expected_case]}")
if config.get("model_type") != expected_mode:
    errors.append(f"mode={config.get('model_type')} expected={expected_mode}")
if errors:
    raise SystemExit("; ".join(errors))
PY
}

if [[ "$DRY_RUN" != "1" ]]; then
  missing=0
  for case_name in "${CASES[@]}"; do
    for domain in "${KNOWN_DOMAINS[@]}"; do
      for ((i = 0; i < ${#AGENTS[@]}; i++)); do
        for ((j = i; j < ${#AGENTS[@]}; j++)); do
          checkpoint="$RESULTS_ROOT/$case_name/models/expert/${AGENTS[$i]}-${AGENTS[$j]}/$domain/Transformer_Negotiator/checkpoint.pt"
          if [[ ! -f "$checkpoint" ]]; then
            echo "Missing expert checkpoint: $checkpoint" >&2
            missing=$((missing + 1))
          elif ! error="$(validate_checkpoint "$checkpoint" "$EXPERT_STEPS" "$case_name" expert 2>&1)"; then
            echo "Invalid expert checkpoint: $checkpoint ($error)" >&2
            missing=$((missing + 1))
          fi
        done
      done
    done
    checkpoint="$RESULTS_ROOT/$case_name/models/general/Transformer_Negotiator/checkpoint.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing general checkpoint: $checkpoint" >&2
      missing=$((missing + 1))
    elif ! error="$(validate_checkpoint "$checkpoint" "$GENERAL_STEPS" "$case_name" general 2>&1)"; then
      echo "Invalid general checkpoint: $checkpoint ($error)" >&2
      missing=$((missing + 1))
    fi
  done
  if (( missing > 0 )); then
    echo "Evaluation aborted: $missing required checkpoint(s) are missing" >&2
    exit 1
  fi
fi

evaluate_one() {
  local condition_index="$1" mode="$2" case_name="$3" pair="$4" domain="$5" model_dir="$6"
  if (( condition_index % SHARD_COUNT != SHARD_INDEX )); then
    return
  fi
  local agent0="${pair%%-*}" agent1="${pair#*-}"
  local export_root="$RESULTS_ROOT/$case_name/evaluation"
  local output="$export_root/$mode/$pair/$domain/$case_name/$domain-$pair-dF-nF.tsv"
  if [[ "$FORCE" != "1" && -f "$output" ]]; then
    lines="$(wc -l < "$output" | tr -d ' ')"
    if (( lines == EPISODES + 1 )); then
      echo "SKIP complete ($EPISODES episodes): $case_name $mode $pair $domain"
      return
    fi
  fi
  command=(
    "$PYTHON_BIN" test_negotiator.py --model-path "$model_dir"
    --agents "$agent0" "$agent1" --issues "$domain"
    --episodes "$EPISODES" --seed "$SEED" --device "$DEVICE"
    --case "$case_name" --style neutral --export-root "$export_root"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN condition=%03d shard=%d/%d:' "$condition_index" "$SHARD_INDEX" "$SHARD_COUNT"
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    echo "EVALUATE: $case_name $mode $pair $domain"
    "${command[@]}"
  fi
}

condition_index=0
for case_name in "${CASES[@]}"; do
  for domain in "${KNOWN_DOMAINS[@]}"; do
    for ((i = 0; i < ${#AGENTS[@]}; i++)); do
      for ((j = i; j < ${#AGENTS[@]}; j++)); do
        pair="${AGENTS[$i]}-${AGENTS[$j]}"
        model_dir="$RESULTS_ROOT/$case_name/models/expert/$pair/$domain/Transformer_Negotiator"
        evaluate_one "$condition_index" expert "$case_name" "$pair" "$domain" "$model_dir"
        condition_index=$((condition_index + 1))
      done
    done
  done
  model_dir="$RESULTS_ROOT/$case_name/models/general/Transformer_Negotiator"
  for domain in "${ALL_DOMAINS[@]}"; do
    for ((i = 0; i < ${#AGENTS[@]}; i++)); do
      for ((j = i; j < ${#AGENTS[@]}; j++)); do
        pair="${AGENTS[$i]}-${AGENTS[$j]}"
        evaluate_one "$condition_index" general "$case_name" "$pair" "$domain" "$model_dir"
        condition_index=$((condition_index + 1))
      done
    done
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$RESULTS_ROOT/experiment_status"
  touch "$RESULTS_ROOT/experiment_status/eval-shard-${SHARD_INDEX}-of-${SHARD_COUNT}.done"
fi
echo "Evaluation shard complete: shard=$SHARD_INDEX/$SHARD_COUNT total_conditions=$condition_index"
