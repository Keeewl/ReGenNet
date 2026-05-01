#!/usr/bin/env bash
set -euo pipefail

for SEED in $(seq 0 19); do
  echo "=== table1 HiReact seed ${SEED} ==="
  SEED=${SEED} bash refine_v2/commands/eval/25_run_table1_hireact_seed.sh
done
