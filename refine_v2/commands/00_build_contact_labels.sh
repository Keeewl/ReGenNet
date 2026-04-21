conda activate regennet5090

####### refine_v2: build GT binary contact labels #######
export CUDA_VISIBLE_DEVICES=1
python -m refine_v2.tools.build_contact_labels \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --output_path refine_v2/outputs/train/contact_labels_gt.npz \
  --region_map_path visualize/viewer/part_segm/6_parts/six_parts.pkl \
  --tau_contact 0.05 \
  --gap_merge 2 \
  --raw_L_min 4 \
  --batch_size 64 \
  --num_workers 4 \
  --device cuda \
  --frame_chunk 1 \
  --target_chunk 2048

