conda activate regennet5090

####### refine_v2: train first residual refiner #######
export CUDA_VISIBLE_DEVICES=0
python -m refine_v2.cli_train_refiner \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --save_dir refine_v2/outputs/train/refiner_v2_exp1 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --val_ratio 0.1 \
  --split_seed 1234 \
  --num_steps 10000 \
  --eval_interval 500 \
  --lambda_region_dist 0.0
