conda activate regennet5090

####### refine_v2: build contact-rich subset manifest #######
export CUDA_VISIBLE_DEVICES=1
python -m refine_v2.cli_build_subset_manifest \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --selector_windows_path refine_v2/outputs/train/selector_windows_v2_hand_time_topk_tau010.npz \
  --selector_audit_path refine_v2/outputs/train/selector_audit_v2_hand_time_topk_tau010.json \
  --action_type_stats_path refine_v2/outputs/train/action_type_stats/action_type_stats.json \
  --output_dir refine_v2/outputs/train/contact_subset \
  --min_num_sequences 20 \
  --min_gt_positive_sequence_ratio 0.50 \
  --min_contact_rich_score 0.0 \
  --min_training_value_score 0.0
