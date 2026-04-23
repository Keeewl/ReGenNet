conda activate regennet5090

####### refine_v2 exp7: diagnose transl vs hand-pose limits from vis pack #######
export CUDA_VISIBLE_DEVICES=0
python -m visualize.refine_v2.diagnose_refiner_vis_pack \
  --vis_pack_path refine_v2/save/train/refiner_v2_exp7_contact_refine_v1_10k/vis_pack_random20/refiner_vis_pack.npz \
  --region_map_path visualize/viewer/part_segm/6_parts/six_parts.pkl \
  --output_json refine_v2/save/train/refiner_v2_exp7_contact_refine_v1_10k/vis_pack_random20/refiner_vis_pack_diagnosis.json \
  --output_md refine_v2/save/train/refiner_v2_exp7_contact_refine_v1_10k/vis_pack_random20/refiner_vis_pack_diagnosis.md \
  --output_csv refine_v2/save/train/refiner_v2_exp7_contact_refine_v1_10k/vis_pack_random20/refiner_vis_pack_diagnosis_windows.csv \
  --device cuda \
  --batch_size_sequences 4 \
  --transl_error_high 0.05 \
  --local_hand_error_high 0.05 \
  --contact_gap_high 0.03
