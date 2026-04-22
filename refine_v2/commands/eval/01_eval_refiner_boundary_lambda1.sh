conda activate regennet5090

####### refine_v2: eval motion metrics for boundary lambda=1.0 #######
export CUDA_VISIBLE_DEVICES=0
python -m refine_v2.cli_eval_refiner \
  --checkpoint refine_v2/save/train/refiner_v2_exp4_boundary_lam1_10k/model_best.pt \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --device cuda \
  --output_json refine_v2/save/train/refiner_v2_exp4_boundary_lam1_10k/eval_window.json
