conda activate regennet5090

####### refine_v2: small overfit test for first residual refiner #######
export CUDA_VISIBLE_DEVICES=0
python -m refine_v2.cli_train_refiner \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --save_dir refine_v2/outputs/train/refiner_v2_overfit \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda \
  --overfit_num_windows 64 \
  --num_steps 500 \
  --eval_interval 100 \
  --log_interval 20 \
  --lambda_region_dist 0.0
