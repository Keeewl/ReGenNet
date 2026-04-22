conda activate regennet5090

####### refine_v2: standalone selector/window report #######
export CUDA_VISIBLE_DEVICES=0
python -m refine_v2.cli_selector_window_report \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --selector_audit_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_audit.json \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --output_json refine_v2/outputs/train/contact_subset/selector_rerun/selector_window_report.json \
  --output_md refine_v2/outputs/train/contact_subset/selector_rerun/selector_window_report.md \
  --output_csv refine_v2/outputs/train/contact_subset/selector_rerun/selector_window_report.csv
