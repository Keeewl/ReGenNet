conda activate regennet5090

####### refine_v2: boundary transl lambda=1.0, 10k steps #######
export CUDA_VISIBLE_DEVICES=0
python -m refine_v2.cli_train_refiner \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --save_dir refine_v2/save/train/refiner_v2_exp4_boundary_lam1_10k \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --val_ratio 0.1 \
  --split_seed 1234 \
  --hidden_dim 512 \
  --num_heads 8 \
  --num_layers 8 \
  --dropout 0.1 \
  --mlp_ratio 4.0 \
  --num_steps 10000 \
  --warmup_steps 1000 \
  --eval_interval 1000 \
  --save_interval 1000 \
  --log_interval 100 \
  --mixed_precision \
  --lambda_region_dist 0.0 \
  --lambda_boundary_trans 1.0 \
  --boundary_trans_frames 2
