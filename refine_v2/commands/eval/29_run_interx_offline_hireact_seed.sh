#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:-0}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
STAGE1_MODEL_PATH=${STAGE1_MODEL_PATH:-save/cnet_v5_256/interx_smplx_offline_exp1/model000149455.pt}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-refine_v2/save/train/refiner_v2_exp8_interaction_v1_10k/model_best.pt}
STGCN_MODEL_PATH=${STGCN_MODEL_PATH:-recognition_training/interx_exp1/checkpoint_0100.pth.tar}
REGION_MAP_PATH=${REGION_MAP_PATH:-visualize/viewer/part_segm/6_parts/six_parts.pkl}
ROOT_DIR=${ROOT_DIR:-refine_v2/save/interx_offline_table/hireact_seed${SEED}}

export CUDA_VISIBLE_DEVICES

python3 -m refine.data.build_reaction_data \
  --model_path "${STAGE1_MODEL_PATH}" \
  --output_path "${ROOT_DIR}/train/reaction_data.npz" \
  --dataset interx \
  --split train \
  --data_path dataset/interx/regen/train.h5 \
  --device cuda \
  --batch_size 128 \
  --num_samples 1000 \
  --seed "${SEED}" \
  --setting cnet_v5 \
  --arch offline \
  --body_model smplx \
  --pose_rep rot6d \
  --num_frames 150 \
  --num_person 2 \
  --latent_dim 256 \
  --layers 8 \
  --use_ddim \
  --timestep_respacing ddim5 \
  --enable_restoration_metadata true \
  --restoration_meta_path dataset/interx/cache/interx_restoration_meta.npz \
  --interaction_order_path dataset/interx/annots/interaction_order.pkl

python3 -m refine.data.build_reaction_data \
  --model_path "${STAGE1_MODEL_PATH}" \
  --output_path "${ROOT_DIR}/test/reaction_data.npz" \
  --dataset interx \
  --split test \
  --data_path dataset/interx/regen/test.h5 \
  --device cuda \
  --batch_size 128 \
  --num_samples 1000 \
  --seed "${SEED}" \
  --setting cnet_v5 \
  --arch offline \
  --body_model smplx \
  --pose_rep rot6d \
  --num_frames 150 \
  --num_person 2 \
  --latent_dim 256 \
  --layers 8 \
  --use_ddim \
  --timestep_respacing ddim5 \
  --enable_restoration_metadata true \
  --restoration_meta_path dataset/interx/cache/interx_restoration_meta.npz \
  --interaction_order_path dataset/interx/annots/interaction_order.pkl

python -m refine_v2.cli_eval_table1_hireact_dryrun \
  --checkpoint "${STAGE2_CHECKPOINT}" \
  --train_reaction_data_path "${ROOT_DIR}/train/reaction_data.npz" \
  --test_reaction_data_path "${ROOT_DIR}/test/reaction_data.npz" \
  --region_map_path "${REGION_MAP_PATH}" \
  --stgcn_model_path "${STGCN_MODEL_PATH}" \
  --output_dir "${ROOT_DIR}/hireact_dryrun" \
  --device cuda \
  --selector_batch_size 32 \
  --selector_num_workers 0 \
  --selector_tau_contact 0.10 \
  --selector_gap_merge 4 \
  --selector_raw_L_min 2 \
  --selector_window_size 30 \
  --selector_per_hand_max_windows 2 \
  --selector_per_seq_max_windows 3 \
  --selector_top_k_regions 3 \
  --stitch_batch_size 32 \
  --stitch_num_workers 0 \
  --stgcn_batch_size 64 \
  --geometry_batch_size 32 \
  --geometry_num_workers 0 \
  --frame_chunk 1 \
  --target_chunk 2048
