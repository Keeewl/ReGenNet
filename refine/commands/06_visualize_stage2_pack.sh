#!/usr/bin/env bash
set -euo pipefail

# Convert a Stage2-Lite refined_pack.npz into viewer-ready P1/P2 clip folders.
#
# Example:
#   PACK=refine/outputs/eval_stage2_lite_step000019000_test1000/refined_pack.npz \
#   OUTPUT_DIR=outputs/stage2_lite_step19000_refined/motions \
#   VARIANT=refined \
#   bash refine/commands/06_visualize_stage2_pack.sh

PACK="${PACK:-refine/outputs/eval_stage2_lite_step000019000_test1000/refined_pack.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2_lite_step19000_refined/motions}"
VARIANT="${VARIANT:-refined}"
LIMIT="${LIMIT:-}"
PRESERVE_RAW_PERSON_ORDER="${PRESERVE_RAW_PERSON_ORDER:-0}"

ARGS=(
  --pack "${PACK}"
  --output_dir "${OUTPUT_DIR}"
  --variant "${VARIANT}"
  --overwrite
)

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

if [[ "${PRESERVE_RAW_PERSON_ORDER}" == "1" ]]; then
  ARGS+=(--preserve_raw_person_order)
fi

python3 -m visualize.converters.convert_stage2_pack_to_motions "${ARGS[@]}"
