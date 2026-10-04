#!/usr/bin/env bash
set -euo pipefail

CASE_NAME="${CASE_NAME:-case1}"

python train.py \
  -a Boulware Linear \
  -i Laptop \
  --model-type expert \
  --utility-cases "${CASE_NAME}" \
  --total-timesteps 100000 \
  --save-path "results/${CASE_NAME}"
