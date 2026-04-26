conda activate inter-x

####### local snapshot viewer: view stage1 clip refined by stage2 exp8 #######
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_ids 0 14 69 91 \
  --offset_dir 0 0 1 \
  --spacing 2.0 \
  --time_gradient

cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --texts_dir '' \
  --title 'stage1-clip-refined-exp8'