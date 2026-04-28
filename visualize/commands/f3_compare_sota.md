### F3_compare_Handshake


#### Candidation

highfive, handwrestling, messagingleg, dance

```bash
python -m visualize.refine_v2.view_refiner_vis_pack_ait \
  --vis_pack_path refine_v2/save/visual_hireact/highfive/vis_pack_random20/refiner_vis_pack.npz \
  --sequence_index 0 \
  --mode coarse_refined_gt \
  --fps 30 \
  --window_scale 0.9
```




#### Video

GT Video (idx 162):

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






#### Snapshot-Handshake

ReGenNet:

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cmdm_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 14 84 91 124 \
  --offset_dir 1 0 1 \
  --spacing 1 \
  --time_gradient \
  --title 'ReGenNet'
```

HiReact:

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir refine_v2/save/infer/refiner_v2_exp8_on_stage1_clip_0001_handshake/refined \
  --clip_name 0001_Handshake \
  --frame_ids 14 84 91 124 \
  --offset_dir 1 0 1 \
  --spacing 1 \
  --time_gradient \
  --title 'HiReact'
```

Real:

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/interx_regen_train_restored_height \
  --clip_name G002T000A001R005 \
  --frame_ids 14 84 91 124 \
  --offset_dir 1 0 1 \
  --spacing 1 \
  --time_gradient \
  --title 'Real'
```