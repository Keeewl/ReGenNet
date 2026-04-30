

cmdm:

```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage1_cnetv5_G027T004A021R004/motions \
  --texts_dir '' \
  --title 'cmdm-G027T004A021R004'
```
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage1_cmdm_G027T004A021R004_2/motions \
  --texts_dir '' \
  --title 'cmdm-G027T004A021R004_2'
```

HiReact:

```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage2_hireact_G027T004A021R004/refined \
  --texts_dir '' \
  --title 'hireact-exp8-G027T004A021R004'
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
  --data_dir ../../outputs/single_stage1_cnetv5_G027T004A021R004/motions \
  --clip_name G027T004A021R004 \
  --frame_ids 149 7 9 74 \
  --offset_dir 0 0 1 \
  --spacing 1 \
  --time_gradient \
  --title 'cmdm-G027T004A021R004'
```
```bash
cd visualize/viewer
python snapshot_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage1_cmdm_G027T004A021R004_2/motions \
  --clip_name G027T004A021R004 \
  --frame_ids 149 7 9 74 \
  --offset_dir 0 0 1 \
  --spacing 1 \
  --time_gradient \
  --title 'cmdm-G027T004A021R004'
```

HiReact:

```bash
cd visualize/viewer
python snapshot_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/single_stage2_hireact_G027T004A021R004/refined \
  --clip_name G027T004A021R004 \
  --frame_ids 149 7 9 74 \
  --offset_dir 0 0 1 \
  --spacing 1 \
  --time_gradient \
  --title 'hireact-exp8-G027T004A021R004'
```

Real:

```bash
cd visualize/viewer
python snapshot_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
  --clip_name G027T004A021R004 \
  --frame_ids 222 80 82 147 \
  --offset_dir 0 0 1 \
  --spacing 1 \
  --time_gradient \
  --title 'gt-G027T004A021R004'
```