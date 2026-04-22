conda activate regennet5090

####### refine_v2_v1: offline scope-geometry feature cache #######
export CUDA_VISIBLE_DEVICES=0
python -m refine_v2.cli_build_geometry_feature_cache \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --output_path refine_v2/save/features/scope_geom_train/geometry_feature_cache.npz \
  --region_map_path visualize/viewer/part_segm/6_parts/six_parts.pkl \
  --include_buckets "GT+ / Pred+" \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda
