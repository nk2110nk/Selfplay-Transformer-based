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
EPISODES="${EPISODES:-100}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
LIMIT="${LIMIT:-0}"
STRICT="${STRICT:-1}"
FORCE="${FORCE:-0}"

total_jobs=$(( ${#CASES[@]} * (${#AGENTS[@]} * (${#AGENTS[@]} + 1) / 2) ))
completed_jobs=0
missing_jobs=0

for case_name in "${CASES[@]}"; do
  export_root="$RESULTS_ROOT/$case_name/evaluation"
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
      checkpoint="$model_dir/checkpoint.pt"
      exported="$export_root/expert/$pair/Laptop/$case_name/Laptop-${pair}-dF-nF.tsv"

      if [[ ! -f "$checkpoint" && "$DRY_RUN" != "1" ]]; then
        echo "[${completed_jobs}/${total_jobs}] missing checkpoint: $checkpoint" >&2
        missing_jobs=$((missing_jobs + 1))
        completed_jobs=$((completed_jobs + 1))
        continue
      fi
      if [[ "$FORCE" != "1" && -f "$exported" ]]; then
        lines="$(wc -l < "$exported" | tr -d ' ')"
        if (( lines == EPISODES + 1 )); then
          echo "[${completed_jobs}/${total_jobs}] complete (${EPISODES} episodes): ${case_name} ${pair}"
          completed_jobs=$((completed_jobs + 1))
          continue
        fi
      fi

      command=(
        "$PYTHON_BIN" test_negotiator.py
        --model-path "$model_dir"
        --agents "$agent0" "$agent1"
        --issues Laptop
        --episodes "$EPISODES"
        --seed "$SEED"
        --device "$DEVICE"
        --case "$case_name"
        --export-root "$export_root"
      )
      echo "[${completed_jobs}/${total_jobs}] evaluate: ${case_name} ${pair}"
      if [[ "$DRY_RUN" == "1" ]]; then
        printf '  %q' "${command[@]}"
        printf '\n'
      else
        "${command[@]}"
      fi
      completed_jobs=$((completed_jobs + 1))
    done
  done
done

mkdir -p "$RESULTS_ROOT/experiment_status"
if [[ "$DRY_RUN" != "1" && "$missing_jobs" -eq 0 ]]; then
  touch "$RESULTS_ROOT/experiment_status/laptop-expert-evaluation.done"
fi
echo "Laptop expert evaluation complete: jobs=$completed_jobs/$total_jobs missing=$missing_jobs"
if [[ "$STRICT" == "1" && "$missing_jobs" -gt 0 ]]; then
  exit 1
fi
