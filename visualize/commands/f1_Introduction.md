### F2_Framework_Handshake

### trajectory

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 0 6 14 64 84 91 124 \
  --offset_dir 0 0 1 \
  --spacing 0 \
  --soft_role_colors \
  --time_gradient \
  --time_gradient_alpha_min 0.5 \
  --time_gradient_auto_alpha

python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir ../Inter-X/datasets/interx/motions \
  --clip_name G002T000A001R005 \
  --frame_ids  162 174 193 288 \
  --offset_dir 0 0 1 \
  --spacing 0 \
  --time_gradient \
  --time_gradient_alpha_min 0.5 \
  --time_gradient_auto_alpha

cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../../Inter-X/datasets/interx/motions \
  --texts_dir ''
```


### Feature

Success heatmap:
```bash
python visualize/viewer/heatmap/heatmap_viewer.py \
  --dataset interx \
  --data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_id 84 \
  --actor_hand_side auto \
  --tau_contact 0.005 \
  --max_dist 0.20 \
  --title "Heatmap: 0001_Handshake"
```

Success velocity:
```bash
python visualize/viewer/velocity/velocity_viewer.py \
  --dataset interx \
  --data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --current_frame 84 \
  --actor_prev_frame 65 \
  --reactor_prev_frame 51 \
  --ghost_alpha 0.50 \
  --ghost_white_mix 0.15
  --title "Velocity: 0001_Handshake"
```



### figure_1

Fail:
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 0 6 64 84 \
  --offset_dir 0 0 1 \
  --spacing 0 \
  --time_gradient \
  --time_gradient_alpha_min 0.5 \
  --time_gradient_auto_alpha

python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 0 6 64 84 \
  --offset_dir 0 0 1 \
  --spacing 0 \
  --soft_role_colors \
  --time_gradient \
  --time_gradient_alpha_min 0.5 \
  --time_gradient_auto_alpha
```

### figure_4

Success:
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_ids 0 6 64 84 \
  --offset_dir 0 0 1 \
  --spacing 0 \
  --time_gradient \
  --time_gradient_alpha_min 0.5 \
  --time_gradient_auto_alpha
```

### figure_2

Success:
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_ids 0 6 64 84 \
  --offset_dir 0 0 1 \
  --spacing 0 \
  --time_gradient \
  --time_gradient_alpha_min 0.5 \
  --time_gradient_auto_alpha
```

### figure_1 and figure_3 fail and success snapshot

fail:
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 0 6 64 84 \
  --frame_ids 0 6 64 124 \
  --offset_dir 0 0 1 \
  --spacing 0 \
  --time_gradient \
  --time_gradient_alpha_min 0.5 \
  --time_gradient_auto_alpha
```

Success:
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_ids 0 6 64 84 \
  --offset_dir 0 0 1 \
  --spacing 0 \
  --time_gradient \
  --time_gradient_alpha_min 0.5 \
  --time_gradient_auto_alpha
```





```bash
python visualize/viewer/residual/residual_viewer.py \
  --dataset interx \
  --coarse_data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --refined_data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_id 84 \
  --ghost_alpha 0.30 \
  --ghost_white_mix 0.18 \
  --title "Residual: 0001_Handshake"
```

```bash
python visualize/viewer/residual/residual_viewer.py \
  --dataset interx \
  --coarse_data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --refined_data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_id 91 \
  --ghost_alpha 0.30 \
  --ghost_white_mix 0.18 \
  --title "Residual: 0001_Handshake"
```

```bash
python visualize/viewer/residual/residual_viewer.py \
  --dataset interx \
  --coarse_data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --refined_data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_id 124 \
  --ghost_alpha 0.30 \
  --ghost_white_mix 0.18 \
  --title "Residual: 0001_Handshake"
```





#### Video

GT Video:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
  --texts_dir ''

cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../../Inter-X/datasets/interx/motions \
  --texts_dir ''
```


ReGenNet Video:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/cmdm_interx_handshake_online_200K/motions \
  --texts_dir '' \
  --title 'cmdm-interx-handshake'
```

CNetV5 Video:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/cnetv5_interx_handshake_online_200K/motions \
  --texts_dir '' \
  --title 'cnetv5-interx-handshake'
```


Refine video:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --texts_dir '' \
  --title 'stage1-clip-refined-exp8'
```

#### Snapshot

GT snapshot (162):
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/interx_regen_train_restored_height \
  --clip_name G002T000A001R005 \
  --frame_ids 0 14 69 92 \
  --offset_dir 1 0 1 \
  --spacing 1.0 \
  --time_gradient
```


CNetV5 snapshot (stage1 input):
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 0 14 69 91 \
  --offset_dir 1 0 0 \
  --spacing 1.0 \
  --time_gradient
```


CNetV5 snapshot (stage1 output, stage2 input):
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 91 69 14 0 \
  --offset_dir 1.1 0 -0.5 \
  --spacing 1.3 \
  --time_gradient
```


CNetV5 snapshot (stage1 output, stage2 input):
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 14 84 91 124 \
  --offset_dir 0 0 1 \
  --spacing 1.8 \
  --time_gradient
```


Refine snapshot (stage2 output):
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_ids 14 84 91 124 \
  --offset_dir 0 0 1 \
  --spacing 1.8 \
  --time_gradient
```



4 times, highlight right part:
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 0 14 69 91 \
  --offset_dir 1 0 0 \
  --spacing 1.0 \
  --time_gradient \
  --soft_role_colors \
  --part_segm visualize/viewer/part_segm/6_parts/six_parts.pkl \
  --part_colors visualize/viewer/part_segm/6_parts/highlight_right_actor_blue_reactor_red.json
```


4 times, highlight left part:
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 0 14 69 91 \
  --offset_dir 1 0 0 \
  --spacing 1.0 \
  --time_gradient \
  --soft_role_colors \
  --part_segm visualize/viewer/part_segm/6_parts/six_parts.pkl \
  --part_colors visualize/viewer/part_segm/6_parts/highlight_left_actor_blue_reactor_red.json
```


4 times, highlight feet part:
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 0 14 69 91 \
  --offset_dir 1 0 0 \
  --spacing 1.0 \
  --time_gradient \
  --soft_role_colors \
  --part_segm visualize/viewer/part_segm/6_parts/six_parts.pkl \
  --part_colors visualize/viewer/part_segm/6_parts/highlight_lower_body_actor_blue_reactor_red.json
```






#### Feature

CNetV5 snapshot (show distance):

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 84 \
  --offset_dir 0 0 1 \
  --spacing 1.8 \
  --time_gradient
```

CNetV5 snapshot (show velocity):

```bash
python visualize/viewer/velocity/velocity_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --current_frame 84 \
  --actor_prev_frame 65 \
  --reactor_prev_frame 51 \
  --ghost_alpha 0.50 \
  --ghost_white_mix 0.15
  --title "Velocity: 0001_Handshake"
```

CNetV5 snapshot (show heatmap):

```bash
python visualize/viewer/heatmap/heatmap_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_id 84 \
  --actor_hand_side auto \
  --tau_contact 0.005 \
  --max_dist 0.20 \
  --title "Heatmap: 0001_Handshake"
```




#### Residual

```bash
python visualize/viewer/residual/residual_viewer.py \
  --dataset interx \
  --coarse_data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --refined_data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_id 14 \
  --ghost_alpha 0.30 \
  --ghost_white_mix 0.18 \
  --title "Residual: 0001_Handshake"
```

```bash
python visualize/viewer/residual/residual_viewer.py \
  --dataset interx \
  --coarse_data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --refined_data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_id 84 \
  --ghost_alpha 0.30 \
  --ghost_white_mix 0.18 \
  --title "Residual: 0001_Handshake"
```

```bash
python visualize/viewer/residual/residual_viewer.py \
  --dataset interx \
  --coarse_data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --refined_data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_id 91 \
  --ghost_alpha 0.30 \
  --ghost_white_mix 0.18 \
  --title "Residual: 0001_Handshake"
```

```bash
python visualize/viewer/residual/residual_viewer.py \
  --dataset interx \
  --coarse_data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --refined_data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_id 124 \
  --ghost_alpha 0.30 \
  --ghost_white_mix 0.18 \
  --title "Residual: 0001_Handshake"
```






## 4-part actor attention highlight

These commands keep the actor in the original blue style and render one actor
part in red. Run the four commands separately and take one screenshot from each
viewer window. The `hands` part includes both wrists, so it covers the full
hand region.

### highlight torso/head

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir visualize/viewer/interx_data \
  --clip_name G001T000A001R005 \
  --frame_ids 87 141 247 \
  --offset_dir 1 0 1 \
  --spacing 0 \
  --part_segm visualize/viewer/part_segm/4_parts/four_parts.pkl \
  --highlight_role actor \
  --highlight_part torso_head
```

### highlight lower body

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir visualize/viewer/interx_data \
  --clip_name G001T000A001R005 \
  --frame_ids 87 141 247 \
  --offset_dir 1 0 1 \
  --spacing 0 \
  --part_segm visualize/viewer/part_segm/4_parts/four_parts.pkl \
  --highlight_role actor \
  --highlight_part lower_body
```

### highlight arms

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir visualize/viewer/interx_data \
  --clip_name G001T000A001R005 \
  --frame_ids 87 141 247 \
  --offset_dir 1 0 1 \
  --spacing 0 \
  --part_segm visualize/viewer/part_segm/4_parts/four_parts.pkl \
  --highlight_role actor \
  --highlight_part arms
```

### highlight hands

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir visualize/viewer/interx_data \
  --clip_name G001T000A001R005 \
  --frame_ids 87 141 247 \
  --offset_dir 1 0 1 \
  --spacing 0 \
  --part_segm visualize/viewer/part_segm/4_parts/four_parts.pkl \
  --highlight_role actor \
  --highlight_part hands
```

