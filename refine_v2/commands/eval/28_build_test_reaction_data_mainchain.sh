#!/usr/bin/env bash
set -euo pipefail

conda activate regennet5090

####### table2 test debug: rebuild test reaction_data from the same Stage1 main chain as train #######
export CUDA_VISIBLE_DEVICES=4

python3 -m refine.data.build_reaction_data \
  --model_path save/cnet_v5/interx_smplx_online_exp1/model000200000.pt \
  --output_path refine_v2/save/table2_test/mainchain_test/reaction_data.npz \
  --dataset interx \
  --split test \
  --data_path dataset/interx/regen/test.h5 \
  --device cuda \
  --batch_size 128 \
  --num_samples -1 \
  --seed 10 \
  --setting cnet_v5 \
  --arch online \
  --body_model smplx \
  --pose_rep rot6d \
  --num_frames 150 \
  --num_person 2 \
  --latent_dim 512 \
  --layers 8 \
  --use_ddim \
  --timestep_respacing ddim5 \
  --enable_restoration_metadata true \
  --restoration_meta_path dataset/interx/cache/interx_restoration_meta.npz \
  --interaction_order_path dataset/interx/annots/interaction_order.pkl
