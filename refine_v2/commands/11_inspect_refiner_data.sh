conda activate regennet5090

####### refine_v2: inspect fast refiner window dataset #######
export CUDA_VISIBLE_DEVICES=0
python -m refine_v2.cli_inspect_refiner_data \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --window_index 0 \
  --output_json refine_v2/outputs/train/contact_subset/refiner_data/sample0_summary.json
