conda activate regennet5090

####### refine_v2_v1: export vis pack for scope-geometry refiner #######
export CUDA_VISIBLE_DEVICES=0
python -m refine_v2.cli_export_refiner_vis_pack \
  --checkpoint refine_v2/save/train/refiner_v2_exp5_scope_geom_10k/model_best.pt \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --geometry_feature_cache_path refine_v2/save/features/scope_geom_train/geometry_feature_cache.npz \
  --contact_eval_json refine_v2/save/train/refiner_v2_exp5_scope_geom_10k/contact_eval_window/eval_contact_refiner.json \
  --output_dir refine_v2/save/train/refiner_v2_exp5_scope_geom_10k/vis_pack_random20 \
  --include_buckets "GT+ / Pred+" \
  --selected_action_types "Handshake" "High-five" \
  --max_sequences 20 \
  --sort_by random \
  --seed 1234 \
  --batch_size 32 \
  --num_workers 0 \
  --device cuda
