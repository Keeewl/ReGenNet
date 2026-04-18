#!/usr/bin/env bash
set -euo pipefail

# Build Stage2-lite reaction_data for InterX in restored_pair_space.
#
# Required external inputs:
# - a frozen Stage1 checkpoint
# - InterX raw motions root containing <dataset_key>/P1.npz and <dataset_key>/P2.npz
# - interaction_order.pkl
#
# Example:
#   MODEL_PATH=save/your_stage1/model000200000.pt \
#   RAW_MOTIONS_ROOT=/path/to/interx/motions_raw \
#   INTERACTION_ORDER_PATH=/path/to/interx/annots/interaction_order.pkl \
#   bash refine/commands/01_build_reaction_data_interx_raw_meta.sh

MODEL_PATH="${MODEL_PATH:-save/your_stage1/model000200000.pt}"
DATA_PATH="${DATA_PATH:-dataset/interx/regen/train.h5}"
SPLIT="${SPLIT:-train}"
OUTPUT_PATH="${OUTPUT_PATH:-tmp/refine/interx/${SPLIT}/reaction_data.npz}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
SEED="${SEED:-10}"

RAW_MOTIONS_ROOT="${RAW_MOTIONS_ROOT:?Set RAW_MOTIONS_ROOT to the InterX raw motion folder}"
INTERACTION_ORDER_PATH="${INTERACTION_ORDER_PATH:?Set INTERACTION_ORDER_PATH to interaction_order.pkl}"

python3 refine/data/build_reaction_data.py \
  --model_path "${MODEL_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --dataset interx \
  --split "${SPLIT}" \
  --data_path "${DATA_PATH}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --setting cnet_v5 \
  --arch online \
  --body_model smplx \
  --pose_rep rot6d \
  --num_frames 150 \
  --num_person 2 \
  --latent_dim 512 \
  --layers 8 \
  --enable_restoration_metadata true \
  --raw_motions_root "${RAW_MOTIONS_ROOT}" \
  --interaction_order_path "${INTERACTION_ORDER_PATH}"
