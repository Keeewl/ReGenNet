#!/usr/bin/env bash
set -euo pipefail

# Text inspection for one sample's GT contact labels.

conda activate regennet5090

OUTPUT_DIR="${OUTPUT_DIR:-refine_v2/outputs/train}"
CONTACT_LABELS_PATH="${CONTACT_LABELS_PATH:-${OUTPUT_DIR}/contact_labels_gt.npz}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
DATASET_KEY="${DATASET_KEY:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

ARGS=(
  --contact_labels_path "${CONTACT_LABELS_PATH}"
)

if [[ -n "${DATASET_KEY}" ]]; then
  ARGS+=(--dataset_key "${DATASET_KEY}")
else
  ARGS+=(--sample_index "${SAMPLE_INDEX}")
fi

if [[ -n "${OUTPUT_JSON}" ]]; then
  mkdir -p "$(dirname "${OUTPUT_JSON}")"
  ARGS+=(--output_json "${OUTPUT_JSON}")
fi

python3 -m visualize.refine_v2.vis_contact_labels "${ARGS[@]}"

