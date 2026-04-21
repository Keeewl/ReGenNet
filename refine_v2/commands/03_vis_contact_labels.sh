conda activate regennet5090

####### refine_v2: visualize GT contact labels #######
export CUDA_VISIBLE_DEVICES=7
python -m visualize.refine_v2.vis_contact_labels \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --sample_index 0

