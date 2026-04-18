#!/usr/bin/env bash
set -euo pipefail

# Build Stage2-lite reaction_data for InterX using a pre-exported restoration package.
#
# Required external inputs:
# - a frozen Stage1 checkpoint
# - restoration metadata package (.npz/.h5) covering dataset_key -> betas/gender/raw trans/root orient
# - interaction_order.pkl if the package itself does not already carry actor_is_p1
#
# Example:
#   MODEL_PATH=save/your_stage1/model000200000.pt \
#   RESTORATION_META_PATH=/path/to/interx_restoration_package.npz \
#   INTERACTION_ORDER_PATH=/path/to/interx/annots/interaction_order.pkl \
#   bash refine/commands/02_build_reaction_data_interx_meta_package.sh

MODEL_PATH="${MODEL_PATH:-save/your_stage1/model000200000.pt}"
DATA_PATH="${DATA_PATH:-dataset/interx/regen/train.h5}"
SPLIT="${SPLIT:-train}"
OUTPUT_PATH="${OUTPUT_PATH:-tmp/refine/interx/${SPLIT}/reaction_data.npz}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
SEED="${SEED:-10}"

RESTORATION_META_PATH="${RESTORATION_META_PATH:?Set RESTORATION_META_PATH to the exported restoration package}"
INTERACTION_ORDER_PATH="${INTERACTION_ORDER_PATH:-}"

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
  --restoration_meta_path "${RESTORATION_META_PATH}" \
  --interaction_order_path "${INTERACTION_ORDER_PATH}"
