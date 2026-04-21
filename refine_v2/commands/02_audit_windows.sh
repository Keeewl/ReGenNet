#!/usr/bin/env bash
set -euo pipefail

# Strict audit for selector windows against direct GT binary contact labels.

conda activate regennet5090

OUTPUT_DIR="${OUTPUT_DIR:-refine_v2/outputs/train}"
CONTACT_LABELS_PATH="${CONTACT_LABELS_PATH:-${OUTPUT_DIR}/contact_labels_gt.npz}"
SELECTOR_WINDOWS_PATH="${SELECTOR_WINDOWS_PATH:-${OUTPUT_DIR}/selector_windows_v2.npz}"
AUDIT_JSON="${AUDIT_JSON:-${OUTPUT_DIR}/selector_audit_v2.json}"

mkdir -p "$(dirname "${AUDIT_JSON}")"

python3 -m refine_v2.tools.audit_windows \
  --contact_labels_path "${CONTACT_LABELS_PATH}" \
  --selector_windows_path "${SELECTOR_WINDOWS_PATH}" \
  --output_json "${AUDIT_JSON}"

