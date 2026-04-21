conda activate regennet5090

####### refine_v2: strict + relaxed audit hand-time windows tau010 #######
export CUDA_VISIBLE_DEVICES=1
python -m refine_v2.tools.audit_windows \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --selector_windows_path refine_v2/outputs/train/selector_windows_v2_hand_time_tau010.npz \
  --output_json refine_v2/outputs/train/selector_audit_v2_hand_time_tau010.json
