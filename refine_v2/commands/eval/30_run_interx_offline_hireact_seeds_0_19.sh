#!/usr/bin/env bash
set -euo pipefail

for SEED in $(seq 0 19); do
  echo "=== interx offline HiReact seed ${SEED} ==="
  SEED=${SEED} bash refine_v2/commands/eval/29_run_interx_offline_hireact_seed.sh
done
