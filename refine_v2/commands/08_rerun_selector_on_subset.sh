conda activate regennet5090

####### refine_v2: rerun fixed selector on contact-rich subset #######
export CUDA_VISIBLE_DEVICES=1
python -m refine_v2.cli_rerun_selector_on_subset \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --output_dir refine_v2/outputs/train/contact_subset/selector_rerun \
  --region_map_path visualize/viewer/part_segm/6_parts/six_parts.pkl \
  --include_buckets "GT+ / Pred+" \
  --tau_contact 0.10 \
  --gap_merge 4 \
  --raw_L_min 2 \
  --window_size 30 \
  --per_hand_max_windows 2 \
  --per_seq_max_windows 3 \
  --top_k_regions 3 \
  --batch_size 64 \
  --num_workers 4 \
  --device cuda \
  --frame_chunk 1 \
  --target_chunk 2048
