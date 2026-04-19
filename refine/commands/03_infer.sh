conda activate regennet5090

####### Stage2: infer #######
export CUDA_VISIBLE_DEVICES=7
python3 -m refine.infer.cli \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --checkpoint_path refine/outputs/stage2_lite_run1/stage2_lite_step000019000.pt \
  --output_dir refine/outputs/eval_stage2_lite_step000019000_test1000 \
  --output_name refined_pack.npz \
  --device cuda:0 \
  --batch_size 64 \
  --num_workers 4 \
  --sample_mode stratified \
  --num_samples 1000 \
  --seed 0 \
  --body_model smplx \
  --pose_rep rot6d \
  --save_manifest \
  --save_coverage_report \
  --save_debug_stats
