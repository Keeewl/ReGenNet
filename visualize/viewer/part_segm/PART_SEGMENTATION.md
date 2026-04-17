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
