#!/usr/bin/env bash
set -euo pipefail

python -m refine_v2.cli_eval_table1_hireact_dryrun \
  --checkpoint refine_v2/save/train/refiner_v2_exp8_interaction_v1_10k/model_best.pt \
  --train_reaction_data_path refine_v2/save/table1/cnetv5_seed0_train/reaction_data.npz \
  --test_reaction_data_path refine_v2/save/table1/cnetv5_seed0_test/reaction_data.npz \
  --region_map_path visualize/viewer/part_segm/6_parts/six_parts.pkl \
  --stgcn_model_path recognition_training/interx_exp1/checkpoint_0100.pth.tar \
  --output_dir refine_v2/save/table1/hireact_dryrun_seed0 \
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
