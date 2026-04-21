conda activate regennet5090

####### refine_v2: visualize windows vs GT #######
export CUDA_VISIBLE_DEVICES=7
python -m visualize.refine_v2.vis_windows_vs_gt \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --selector_windows_path refine_v2/outputs/train/selector_windows_v2_hand_time_tau010.npz \
  --audit_json refine_v2/outputs/train/selector_audit_v2_hand_time_tau010.json \
  --sample_index 0 \
  --timeline_width 100
