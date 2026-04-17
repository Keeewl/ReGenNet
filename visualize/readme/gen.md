# Generation Visualization

## Conda Environment

```bash
conda activate inter-x
```

## Current Output Runs

Current `outputs/` runs in this repo:

- `cmdm_interx_online_200K`
- `cmdm_interx_handshake_online_200K`
- `cnetv5_interx_online_200K`
- `cnetv5_interx_handshake_online_200K`

## Convert Model Outputs

Canonical:

```bash
python visualize/converters/convert_results_to_motions.py \
  --outputs_root outputs \
  --runs cnetv5_interx_online_200K \
  --shape_mode canonical \
  --overwrite
```

Restored body shape:

```bash
python visualize/converters/convert_results_to_motions.py \
  --outputs_root outputs \
  --runs cnetv5_interx_online_200K \
  --shape_mode restored \
  --overwrite
```

Restored body shape + raw height/global alignment:

```bash
python visualize/converters/convert_results_to_motions.py \
  --outputs_root outputs \
  --runs cnetv5_interx_online_200K \
  --shape_mode restored_shape_height \
  --overwrite
```

Convert multiple runs together:

```bash
python visualize/converters/convert_results_to_motions.py \
  --outputs_root outputs \
  --runs cmdm_interx_online_200K cnetv5_interx_online_200K \
  --shape_mode restored_shape_height \
  --overwrite
```

Notes:

- `results.npy` is required.
- `restored` and `restored_shape_height` require `results_meta.npz`.
- Restored modes read raw Inter-X motions from the sibling `Inter-X` repo by default.

## Visualize Converted Outputs

Example: `cnetv5_interx_online_200K`

```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/cnetv5_interx_online_200K/motions \
  --texts_dir '' \
  --title 'cnetv5-interx'
```

Other common runs:

```bash
cd visualize/viewer
python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/cmdm_interx_online_200K/motions \
  --texts_dir '' \
  --title 'cmdm-interx'

python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/cmdm_interx_handshake_online_200K/motions \
  --texts_dir '' \
  --title 'cmdm-interx-handshake'

python data_viewer.py \
  --dataset interx \
  --data_dir ../../outputs/cnetv5_interx_handshake_online_200K/motions \
  --texts_dir '' \
  --title 'cnetv5-interx-handshake'
```

Snapshot / teaser layout for one generated clip:

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 0 14 69 91 \
  --offset_dir 1 0 1 \
  --spacing 1.0


```

For time-ordered light-to-dark coloring, add `--time_gradient`:

```bash
python visualize/viewer/snapshot_viewer.py \
  --dataset interx \
  --data_dir outputs/cnetv5_interx_handshake_online_200K/motions \
  --clip_name 0001_Handshake \
  --frame_ids 0 14 69 91 \
  --offset_dir 1 0 1 \
  --spacing 1.0 \
  --time_gradient
```

## Modes

- `canonical`: neutral body shape for motion inspection.
- `restored`: restore raw `betas/gender` while keeping predicted translation.
- `restored_shape_height`: restore raw `betas/gender` and align to raw Inter-X height/global placement.
