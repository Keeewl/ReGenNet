#### Video

cmdm:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage1_cmdm_G053T000A000R004/motions \
  --texts_dir '' \
  --title 'cmdm'
```

HiReact:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage2_hireact_G053T000A000R004/refined \
  --texts_dir '' \
  --title 'hireact-exp8'
```


GT Video:
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
  --texts_dir ''
```


#### Snapshot-Handshake

ReGenNet:
```bash
cd visualize/viewer
python snapshot_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage1_cmdm_G053T000A000R004/motions \
  --clip_name G053T000A000R004 \
  --frame_ids 0 62 71 110 \
  --offset_dir 1 0 0 \
  --spacing 1 \
  --time_gradient \
  --title 'cmdm-G027T004A021R004'
```


HiReact:
```bash
cd visualize/viewer
python snapshot_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage2_hireact_G053T000A000R004/refined \
  --clip_name G053T000A000R004 \
  --frame_ids 0 62 71 110 \
  --offset_dir 1 0 0 \
  --spacing 1 \
  --time_gradient \
  --title 'hireact-exp8'
```


Real:
```bash
cd visualize/viewer
python snapshot_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
  --clip_name G053T000A000R004 \
  --frame_ids 36 98 107 146 \
  --offset_dir 1 0 0 \
  --spacing 1 \
  --time_gradient \
  --title 'gt-G027T004A021R004'
```