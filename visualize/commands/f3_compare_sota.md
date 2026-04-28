### F3_Framework_Handshake

#### Video

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






