conda activate inter-x

####### refine_v2: open one subset window in aitviewer #######
export CUDA_VISIBLE_DEVICES=0
python -m visualize.refine_v2.view_subset_window_ait \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --subset_window_metadata_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_window_metadata.json \
  --dataset_row_index 4 \
  --start_frame 70 \
  --hand_side left \
  --mode both \
  --frame_padding 10 \
  --fps 30
