conda activate regennet5090

####### refine_v2: visualize subset windows sanity report #######
export CUDA_VISIBLE_DEVICES=1
python -m visualize.refine_v2.vis_subset_windows \
  --subset_window_metadata_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_window_metadata.json \
  --audit_json refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_audit.json \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --sort_by purity_desc \
  --limit 20 \
  --timeline_width 100 \
  --output_md refine_v2/outputs/train/contact_subset/selector_rerun/subset_window_sanity_report.md
