conda activate regennet5090

####### Stage2: infer #######
export CUDA_VISIBLE_DEVICES=6
python -m refine.tools.analyze_action_windows \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --json_out refine/outputs/action_window_analysis/train_action_window_stats.json \
  --csv_out refine/outputs/action_window_analysis/train_action_window_stats.csv \
  --device cuda:0 \
  --batch_size 64 \
  --body_model smplx \
  --pose_rep rot6d
