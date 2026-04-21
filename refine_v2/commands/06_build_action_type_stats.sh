conda activate regennet5090

####### refine_v2: build action-type contact statistics #######
export CUDA_VISIBLE_DEVICES=1
python -m refine_v2.cli_build_action_type_stats \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --selector_windows_path refine_v2/outputs/train/selector_windows_v2_hand_time_topk_tau010.npz \
  --selector_audit_path refine_v2/outputs/train/selector_audit_v2_hand_time_topk_tau010.json \
  --output_dir refine_v2/outputs/train/action_type_stats \
  --min_sequences_for_recommendation 20
