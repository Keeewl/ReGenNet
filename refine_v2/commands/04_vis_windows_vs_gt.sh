#!/usr/bin/env bash
set -euo pipefail

# Text inspection for selector windows against GT contact labels.

conda activate regennet5090

OUTPUT_DIR="${OUTPUT_DIR:-refine_v2/outputs/train}"
CONTACT_LABELS_PATH="${CONTACT_LABELS_PATH:-${OUTPUT_DIR}/contact_labels_gt.npz}"
SELECTOR_WINDOWS_PATH="${SELECTOR_WINDOWS_PATH:-${OUTPUT_DIR}/selector_windows_v2.npz}"
AUDIT_JSON="${AUDIT_JSON:-${OUTPUT_DIR}/selector_audit_v2.json}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
DATASET_KEY="${DATASET_KEY:-}"
TIMELINE_WIDTH="${TIMELINE_WIDTH:-100}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

ARGS=(
  --contact_labels_path "${CONTACT_LABELS_PATH}"
  --selector_windows_path "${SELECTOR_WINDOWS_PATH}"
  --audit_json "${AUDIT_JSON}"
  --timeline_width "${TIMELINE_WIDTH}"
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

python3 -m visualize.refine_v2.vis_windows_vs_gt "${ARGS[@]}"

