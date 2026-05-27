export CUDA_VISIBLE_DEVICES=0

python -m refine_v2.cli_infer_refiner_on_viewer_clip \
  --checkpoint refine_v2/save/train/refiner_v2_exp8_interaction_v1_10k/model_best.pt \
  --dataset interx \
  --clip_dir outputs/single_stage1_cnetv5_G027T004A021R004/motions/G027T004A021R004 \
  --region_map_path visualize/viewer/part_segm/6_parts/six_parts.pkl \
  --output_dir outputs/single_stage2_hireact_G027T004A021R004 \
  --variant refined \
  --device cuda \
  --tau_contact 0.05 \
  --gap_merge 4 \
  --raw_L_min 12 \
  --window_size 30 \
  --per_hand_max_windows 2 \
  --per_seq_max_windows 4 \
  --top_k_regions 3 \
  --frame_chunk 1 \
  --target_chunk 2048
