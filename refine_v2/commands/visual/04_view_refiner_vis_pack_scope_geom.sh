conda activate inter-x

####### local aitviewer: view scope-geometry refiner vis pack #######
python -m visualize.refine_v2.view_refiner_vis_pack_ait \
  --vis_pack_path refine_v2/save/train/refiner_v2_exp5_scope_geom_10k/vis_pack_random20/refiner_vis_pack.npz \
  --sequence_index 0 \
  --mode coarse_refined_gt \
  --fps 30 \
  --window_scale 0.9
