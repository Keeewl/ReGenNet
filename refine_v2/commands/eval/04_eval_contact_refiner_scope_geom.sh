conda activate regennet5090

####### refine_v2_v1: contact eval for scope-geometry refiner #######
export CUDA_VISIBLE_DEVICES=0
python -m refine_v2.cli_eval_contact_refiner \
  --checkpoint refine_v2/save/train/refiner_v2_exp5_scope_geom_10k/model_best.pt \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --geometry_feature_cache_path refine_v2/save/features/scope_geom_train/geometry_feature_cache.npz \
  --region_map_path visualize/viewer/part_segm/6_parts/six_parts.pkl \
  --output_dir refine_v2/save/train/refiner_v2_exp5_scope_geom_10k/contact_eval_window \
  --include_buckets "GT+ / Pred+" \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda \
  --tau_contact 0.05 \
  --penetration_margin 0.015 \
  --frame_chunk 1 \
  --target_chunk 2048 \
  --max_debug_windows 500
