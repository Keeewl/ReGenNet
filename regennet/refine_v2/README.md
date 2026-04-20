# refine_v2 module 1

This is the first minimal Stage2 refine_v2 loop:

- GT binary mesh-region contact labels in restored pair space.
- Deterministic selector windows from coarse binary contact.
- Strict audit against direct GT contact labels.
- Text inspection scripts under `regennet/visualize/refine_v2/`.

Intervals use Python slicing semantics: `[start_frame, end_frame)`.

Default parameters:

- `tau_contact = 0.05`
- `gap_merge = 2`
- `raw_L_min = 4`
- `window_size = 30`
- `per_hand_max_windows = 2`
- `per_seq_max_windows = 3`

The default region map is the existing SMPL-X asset:

`visualize/viewer/part_segm/6_parts/six_parts.pkl`

You can pass `--region_map_path` with `.json` or `.npz` using:

```json
{
  "torso_head": [0, 1],
  "lower_body": [2, 3],
  "left_arm": [4],
  "right_arm": [5],
  "left_hand": [6],
  "right_hand": [7]
}
```

Run:

```bash
python -m regennet.refine_v2.cli_build_contact_labels \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --output_path tmp/refine_v2/contact_labels_gt.npz

python -m regennet.refine_v2.cli_select_windows \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path tmp/refine_v2/contact_labels_gt.npz \
  --output_path tmp/refine_v2/selector_windows_v2.npz

python -m regennet.refine_v2.cli_audit_windows \
  --contact_labels_path tmp/refine_v2/contact_labels_gt.npz \
  --selector_windows_path tmp/refine_v2/selector_windows_v2.npz \
  --output_json tmp/refine_v2/selector_audit_v2.json
```

