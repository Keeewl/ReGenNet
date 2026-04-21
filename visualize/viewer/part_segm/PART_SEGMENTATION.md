# Part Segmentation

This folder provides two segmentation presets for SMPL-X meshes and their usage in `data_viewer.py`.

## 2 Parts (Hand / Foot / Body)
Files:
- `part_segm/2_parts/make_hand_foot_body.py`
- `part_segm/2_parts/hand_foot_body.pkl`
- `part_segm/2_parts/hand_foot_body_colors.json`

How it works:
- Uses SMPL-X skinning weights to assign each vertex to its highest-weight joint.
- Joints containing hand/foot tokens are grouped as `hand` and `foot`; the rest are `body`.

Generate:
```bash
python visualize/viewer/part_segm/2_parts/make_hand_foot_body.py
```

## 6 Parts
Files:
- `part_segm/6_parts/make_six_parts.py`
- `part_segm/6_parts/six_parts.pkl`
- `part_segm/6_parts/six_parts_colors.json`

Joint groups:
- `torso_head = [0, 3, 6, 9, 12, 15, 22, 23, 24, 55]`
- `lower_body = [1, 2, 4, 5, 7, 8, 10, 11]`
- `left_arm = [13, 16, 18, 20]`
- `right_arm = [14, 17, 19, 21]`
- `left_hand = [25..39]`
- `right_hand = [40..54]`

Generate:
```bash
python visualize/viewer/part_segm/6_parts/make_six_parts.py
```

## 4 Parts For Framework Attention
Files:
- `part_segm/4_parts/make_four_parts.py`
- `part_segm/4_parts/four_parts.pkl`

Joint groups:
- `torso_head = [0, 3, 6, 9, 12, 15, 22, 23, 24, 55]`
- `lower_body = [1, 2, 4, 5, 7, 8, 10, 11]`
- `arms = left_arm + right_arm without wrists = [13, 14, 16, 17, 18, 19]`
- `hands = left_wrist + right_wrist + left_hand + right_hand = [20, 21, 25..54]`

Generate:
```bash
python visualize/viewer/part_segm/4_parts/make_four_parts.py
```

Snapshot highlight example:
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

## Viewer Usage
```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --part_segm part_segm/6_parts/six_parts.pkl \
  --part_colors part_segm/6_parts/six_parts_colors.json

  --part_colors part_segm/6_parts/three_parts_colors.json
  --part_colors part_segm/6_parts/one_parts_colors.json


python data_viewer.py \
  --dataset interx \
  --part_segm part_segm/2_parts/hand_foot_body.pkl \
  --part_colors part_segm/2_parts/hand_foot_body_colors.json
```
