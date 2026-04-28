conda activate inter-x

####### local aitviewer: view exp8 interaction refiner vis pack #######
python -m visualize.refine_v2.view_refiner_vis_pack_ait \
  --vis_pack_path refine_v2/save/visual_hireact/massagingleg/vis_pack_random20/refiner_vis_pack.npz \
  --sequence_index 0 \
  --mode coarse_refined_gt \
  --fps 30 \
  --window_scale 0.9
