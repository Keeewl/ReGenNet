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