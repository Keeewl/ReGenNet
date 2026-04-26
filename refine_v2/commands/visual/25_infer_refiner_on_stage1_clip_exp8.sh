conda activate regennet5090

####### refine_v2 exp8: run stage2 refine on one stage1 viewer clip #######
export CUDA_VISIBLE_DEVICES=0

CLIP_DATA_DIR=outputs/cnetv5_interx_handshake_online_200K/motions
CLIP_NAME=0001_Handshake
OUTPUT_DIR=refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake

python -m refine_v2.cli_infer_refiner_on_viewer_clip \
  --checkpoint refine_v2/save/train/refiner_v2_exp8_interaction_v1_10k/model_best.pt \
  --dataset interx \
  --data_dir "${CLIP_DATA_DIR}" \
  --clip_name "${CLIP_NAME}" \
  --region_map_path visualize/viewer/part_segm/6_parts/six_parts.pkl \
  --output_dir "${OUTPUT_DIR}" \
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
