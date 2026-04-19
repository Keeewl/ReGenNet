conda activate regennet5090

####### Stage2: build contact subset #######
export CUDA_VISIBLE_DEVICES=6
python -m refine.tools.build_contact_subset \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --json_out refine/dataset/contact_dataset/contact_subset_indices.json \
  --stats_out refine/dataset/contact_dataset/contact_subset_stats.json
