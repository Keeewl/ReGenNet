conda activate inter-x

####### refine_v2: local aitviewer for boundary lambda=1.0 vis pack #######
python -m visualize.refine_v2.view_refiner_vis_pack_ait \
  --vis_pack_path refine_v2/save/train/refiner_v2_exp4_boundary_lam1_10k/vis_pack_random20/refiner_vis_pack.npz \
  --sequence_index 0 \
  --mode coarse_refined_gt \
  --fps 30 \
  --window_scale 0.9
