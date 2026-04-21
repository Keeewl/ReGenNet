#!/usr/bin/env bash
set -euo pipefail

# Build deterministic refine_v2 windows from coarse binary mesh contact.

conda activate regennet5090

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

REACTION_DATA_PATH="${REACTION_DATA_PATH:-refine/dataset/train/reaction_data.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-refine_v2/outputs/train}"
REGION_MAP_PATH="${REGION_MAP_PATH:-visualize/viewer/part_segm/6_parts/six_parts.pkl}"
CONTACT_LABELS_PATH="${CONTACT_LABELS_PATH:-${OUTPUT_DIR}/contact_labels_gt.npz}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TAU_CONTACT="${TAU_CONTACT:-0.05}"
GAP_MERGE="${GAP_MERGE:-2}"
RAW_L_MIN="${RAW_L_MIN:-4}"
WINDOW_SIZE="${WINDOW_SIZE:-30}"
PER_HAND_MAX_WINDOWS="${PER_HAND_MAX_WINDOWS:-2}"
PER_SEQ_MAX_WINDOWS="${PER_SEQ_MAX_WINDOWS:-3}"
FRAME_CHUNK="${FRAME_CHUNK:-1}"
TARGET_CHUNK="${TARGET_CHUNK:-2048}"

mkdir -p "${OUTPUT_DIR}"

python3 -m refine_v2.tools.select_windows \
  --reaction_data_path "${REACTION_DATA_PATH}" \
  --contact_labels_path "${CONTACT_LABELS_PATH}" \
  --output_path "${OUTPUT_DIR}/selector_windows_v2.npz" \
  --region_map_path "${REGION_MAP_PATH}" \
  --tau_contact "${TAU_CONTACT}" \
  --gap_merge "${GAP_MERGE}" \
  --raw_L_min "${RAW_L_MIN}" \
  --window_size "${WINDOW_SIZE}" \
  --per_hand_max_windows "${PER_HAND_MAX_WINDOWS}" \
  --per_seq_max_windows "${PER_SEQ_MAX_WINDOWS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --frame_chunk "${FRAME_CHUNK}" \
  --target_chunk "${TARGET_CHUNK}"

