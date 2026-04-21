### handshake for hand

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir visualize/viewer/interx_data \
  --clip_name G001T000A001R005 \
  --frame_ids 87 141 247 \
  --offset_dir 1 0 1 \
  --spacing 0 \
  --time_gradient
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


### pull for arm

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir visualize/viewer/interx_data \
  --clip_name G001T000A003R018 \
  --frame_ids 273 376 486 \
  --offset_dir 1 0 1 \
  --spacing 0 \
  --time_gradient
```


### run for torso

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir visualize/viewer/interx_data \
  --clip_name G028T003A017R015 \
  --frame_ids 414 472 530 \
  --offset_dir 1 0 1 \
  --spacing 0 \
  --time_gradient
```

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir visualize/viewer/interx_data \
  --clip_name G028T003A017R011 \
  --frame_ids 61 135 237 \
  --offset_dir 1 0 1 \
  --spacing 0 \
  --time_gradient
```
