#!/usr/bin/env bash
set -euo pipefail

SUMMARY_GLOB=${SUMMARY_GLOB:-refine_v2/save/table1/hireact_seed*/hireact_dryrun/table1_hireact_dryrun_summary.json}
OUT_DIR=${OUT_DIR:-refine_v2/save/table1/hireact_aggregate}

python -m refine_v2.cli_aggregate_table1_hireact \
  --summary_glob "${SUMMARY_GLOB}" \
  --json_out "${OUT_DIR}/table1_hireact_aggregate.json" \
  --csv_out "${OUT_DIR}/table1_hireact_aggregate.csv" \
  --md_out "${OUT_DIR}/table1_hireact_aggregate.md"
