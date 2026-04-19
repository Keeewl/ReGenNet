conda activate regennet5090

####### Stage2: build contact subset #######
export CUDA_VISIBLE_DEVICES=6
python -m refine.tools.build_contact_subset \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --json_out refine/dataset/contact_dataset/contact_subset_indices.json \
  --stats_out refine/dataset/contact_dataset/contact_subset_stats.json




export CUDA_VISIBLE_DEVICES=6
python -m refine.tools.run_contact_eval \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --checkpoint_path refine/outputs/stage2_lite_run1/stage2_lite_step000019000.pt \
  --output_dir refine/outputs/contact_eval_step19000_train \
  --subset_json refine/dataset/contact_dataset/contact_subset_indices.json \
  --stgcn_model_path recognition_training/interx_exp1/checkpoint_0100.pth.tar \
  --device cuda:0 \
  --batch_size 64 \
  --local_batch_size 4 \
  --global_batch_size 64 \
  --body_model smplx \
  --pose_rep rot6d
