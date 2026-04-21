# Dataset Visualization

## Conda Environment

```bash
conda activate inter-x
```

## Raw Inter-X GT

`visualize/viewer/interx_data` and `visualize/viewer/interx_texts` are symlinks to the sibling `Inter-X` repo, so the default preset is enough:

```bash
cd visualize/viewer
python data_viewer.py --dataset interx
```

If you want to pass paths explicitly:

```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ./interx_data \
  --texts_dir ./interx_texts
```

Notes:

- The viewer shows `raw_index: xx/11388` based on the raw Inter-X motion order.
- If local `train.h5 / val.h5 / test.h5` files are available, the viewer also shows `train_index / val_index / test_index`.

Manual snapshot / teaser layout for one raw clip:

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir visualize/viewer/interx_data \
  --clip_name G001T000A001R005 \
  --frame_ids 47 87 141 247 \
  --offset_dir 1 0 1 \
  --spacing 0 \
  --time_gradient
```

Add `--time_gradient` if you want earlier snapshots lighter and later snapshots darker:

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/interx_regen_train_restored_height \
  --clip_name G002T000A001R006 \
  --frame_ids 0 14 69 91 \
  --offset_dir 1 0 1 \
  --spacing 1.0 \
  --time_gradient
```

## Processed Inter-X H5

Canonical visualization:

```bash
python visualize/converters/convert_processed_h5_to_motions.py \
  --h5_path dataset/interx/regen/train.h5 \
  --output_dir outputs/interx_regen_train_canonical \
  --shape_mode canonical \
  --overwrite

cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_canonical \
  --texts_dir ''
```

Restored body shape:

```bash
python visualize/converters/convert_processed_h5_to_motions.py \
  --h5_path dataset/interx/regen/train.h5 \
  --output_dir outputs/interx_regen_train_restored \
  --shape_mode restored \
  --overwrite

cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored \
  --texts_dir ''
```

Restored body shape + raw height/global alignment:

```bash
python visualize/converters/convert_processed_h5_to_motions.py \
  --h5_path dataset/interx/regen/train.h5 \
  --output_dir outputs/interx_regen_train_restored_height \
  --shape_mode restored_shape_height \
  --overwrite

cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/interx_regen_train_restored_height \
  --texts_dir ''
```

Notes:

- `train.h5 / val.h5 / test.h5 / inter-x_regen.h5` default to `actor_reactor` order.
- If you visualize raw-order `inter-x.h5`, add `--person_order raw`.
- Restored modes read raw motions from the sibling `Inter-X` repo by default.

Snapshot / teaser layout for one converted clip:

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/interx_regen_train_restored_height \
  --clip_name G002T000A001R006 \
  --frame_ids 0 14 69 91 \
  --offset_dir 1 0 1 \
  --spacing 1.0
```

## Raw Chi3D GT

`visualize/viewer/chi3d_data` is also a symlink to the sibling `Inter-X` repo:

```bash
cd visualize/viewer
python data_viewer.py --dataset chi3d
```
