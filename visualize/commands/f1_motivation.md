## Multi-Scale

我想构思Multi-Scale的小框的可视化，在关键握手帧选取actor和reactor分别两个框，
（用其他颜色）标注出actor“右臂右手伸出”和“左臂左手不动”，然后对应reactor“右臂右手伸出”和“左臂左手不动”，
这样来展示Multi-Scale。你简要分析总结一下，然后分析可以怎么样可视化比较直观，美观，符合整体逻辑。


Part-right:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
  --texts_dir '' \
  --soft_role_colors \
  --part_segm ./part_segm/6_parts/six_parts.pkl \
  --part_colors ./part_segm/6_parts/highlight_right_actor_blue_reactor_red.json
```

Part-left:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
  --texts_dir '' \
  --soft_role_colors \
  --part_segm ./part_segm/6_parts/six_parts.pkl \
  --part_colors ./part_segm/6_parts/highlight_left_red.json
```

Part-legs:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
  --texts_dir '' \
  --soft_role_colors \
  --part_segm ./part_segm/6_parts/six_parts.pkl \
  --part_colors ./part_segm/6_parts/highlight_legs_red.json
```

Snapshot:
```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir visualize/viewer/interx_data \
  --clip_name G002T000A001R005 \
  --frame_ids 14 124 \
  --offset_dir 1 0 1 \
  --spacing 0 \
  --time_gradient
```



## Multi-Stage

CNetV5 snapshot (stage1 output, stage2 input):

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 84 91 124 \
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
  --frame_ids 84 91 124 \
  --offset_dir 0 0 1 \
  --spacing 1.5 \
  --time_gradient
```



## Handshake

### Video

GT Video:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
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





## highfive

### Video

GT Video:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
  --texts_dir ''
```


cmdm:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage1_cmdm_G038T003A016R005/motions \
  --texts_dir '' \
  --title 'cnetv5-single-G038T003A016R005'
```





## dance

### Video

GT Video:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
  --texts_dir ''
```


cmdm:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage1_cmdm_G048T004A021R007/motions \
  --texts_dir '' \
  --title 'cmdm-single-G048T004A021R007'
```

cmdm:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage1_cmdm_G039T006A021R006/motions \
  --texts_dir '' \
  --title 'cmdm-single-G039T006A021R006'
```