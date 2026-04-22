conda activate inter-x

####### refine_v2: view exported refiner visualization pack locally #######
python -m visualize.refine_v2.view_refiner_vis_pack_ait \
  --vis_pack_path refine_v2/outputs/train/refiner_v2_exp2_large/vis_pack_random20/refiner_vis_pack.npz \
  --sequence_index 0 \
  --mode coarse_refined_gt \
  --fps 30 \
  --window_scale 0.9
