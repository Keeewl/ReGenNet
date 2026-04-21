# Refine V2 Module 2: Refiner Data Interface Summary

Date: 2026-04-21

## Scope

This phase implemented the first stable refiner data interface for Stage2 refine.

It did not implement:

- refiner network
- loss
- train loop
- evaluation loop
- feature normalization
- dynamic SMPL-X / xyz generation

The goal was to make the next training phase start from a reliable, auditable
window-level dataset instead of wiring model code directly to raw artifacts.

## Fixed Upstream Assumptions

The data interface assumes the current module 1 and subset pipeline are frozen:

```text
proposal_type = hand_time_with_region_attribution
selector_tau_contact = 0.10
gap_merge = 4
raw_L_min = 2
window_size = 30
per_hand_max_windows = 2
per_seq_max_windows = 3
top_k_regions = 3
```

Training bucket:

```text
GT+ / Pred+
```

Current subset:

```text
15 contact-rich Inter-X action types
2842 sequences
6749 selector windows
```

## Implemented Files

```text
refine_v2/refiner_data/__init__.py
refine_v2/refiner_data/schema.py
refine_v2/refiner_data/sanity_checks.py
refine_v2/refiner_data/feature_pack.py
refine_v2/refiner_data/window_dataset.py
refine_v2/refiner_data/window_loader.py
refine_v2/refiner_data/README.md
refine_v2/tools/inspect_refiner_data.py
refine_v2/cli_inspect_refiner_data.py
refine_v2/commands/11_inspect_refiner_data.sh
```

`refine_v2/commands/README.md` was updated to include the inspection command.

## Core Design

Each dataset item is one hand-time selector window.

Important design decision:

```text
Do not duplicate one hand-time window into multiple region-window samples.
```

Each sample keeps:

- one selected hand
- one primary target region
- top-k target regions
- top-k attribution scores
- motion crops
- selector contact condition
- GT contact supervision
- sequence/window metadata

This matches the current selector design and avoids reintroducing early hard
region splitting.

## Window Sample Schema

Motion backbone:

```text
actor_motion_window   [J, F, T]
coarse_motion_window  [J, F, T]
gt_motion_window      [J, F, T]
```

Mesh-aware selector condition:

```text
coarse_region_contact_mask_window  [6, T]
coarse_min_region_dist_window      [6, T]
```

GT mesh-region supervision:

```text
gt_region_contact_mask_window  [6, T]
gt_min_region_dist_window      [6, T]
```

Top-k region condition:

```text
hand_side
hand_side_id
primary_target_region
primary_target_region_id
topk_target_regions
topk_target_region_ids
topk_region_scores
topk_region_scores_numeric  [K, 3]
```

`topk_region_scores_numeric` uses this fixed column order:

```text
num_contact_frames
mean_min_dist
min_dist
```

Metadata:

```text
dataset_row_index
sample_index
dataset_key
action_type
action_label
action_name
bucket_label
is_gt_positive
is_pred_positive
start_frame
end_frame
raw_start_frame
raw_end_frame
window_length
valid_mask
window_index
sequence_window_index
```

## Row Mapping / Alignment Logic

The dataset explicitly separates global row ids from artifact-local indices:

```text
reaction_data row index = dataset_row_index
label_row_to_index      = {dataset_row_index -> label array index}
selector_row_to_index   = {dataset_row_index -> selector artifact local index}
manifest_row_to_record  = {dataset_row_index -> manifest sequence metadata}
```

This is required because `subset_selector_windows.npz` is subset-local, while
`reaction_data.npz` is indexed by original train row.

All motion/contact crops are aligned through these mappings.

## Fast Path

The dataset does not recompute SMPL-X vertices or contact distances.

Default behavior:

```text
reaction_data.npz            -> motion window crops
subset_selector_windows.npz  -> coarse contact mask / min-distance crops
contact_labels_gt.npz        -> GT contact mask / min-distance crops
subset_manifest.json         -> action / bucket / sequence metadata
selector windows             -> hand / region / top-k / time bounds
```

Core arrays are cached once during dataset construction to avoid repeated `.npz`
inflation during `__getitem__`.

## Restored-Space Checks

The dataset requires:

```text
reaction_data.space_definition == restored_pair_space
contact_labels.space_definition == restored_pair_space
selector_windows.space_definition == restored_pair_space
```

If `metadata_json` contains a `space_definition`, it is also checked.

The dataset does not auto-restore motion. That is intentional: restoring inside
the dataset could desynchronize motion from already-built contact/selector
artifacts.

## Sanity Checks

Strict mode checks:

- required artifact fields exist
- every subset row exists in reaction data, GT labels, selector artifact, and manifest
- selector local index is resolved through `dataset_row_indices`
- label local index is resolved through `dataset_row_indices`
- selector length equals reaction length
- label length equals reaction length
- window bounds are valid
- window bounds do not exceed sequence length
- window bounds do not exceed motion frames
- motion crop time length equals `window_length`
- contact crop shape is `[6, T]`
- `valid_mask` is bool and shape `[T]`
- top-k ids, names, and scores are non-empty and length-aligned
- top-k region ids are legal 6-part region ids
- primary target region id is included in top-k region ids

Failures raise explicit errors instead of silently skipping samples.

## DataLoader / Collate

Implemented:

```text
collate_refine_v2_window_batch
make_refine_v2_window_loader
```

Tensor-stacked fields:

- motion windows
- selector contact condition windows
- GT contact supervision windows
- valid mask
- hand id
- primary region id
- top-k ids
- numeric top-k scores
- frame/window indices

Metadata stays as Python lists:

- dataset keys
- action labels/names/types
- bucket labels
- hand side
- primary region
- top-k region names
- original top-k score dictionaries
- region score table

Torch is imported only inside loader/collate helpers, so inspection and dataset
construction can work without importing torch.

## Inspection CLI

Summary command:

```bash
python -m refine_v2.cli_inspect_refiner_data \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --summary_only
```

Single-window inspection:

```bash
python -m refine_v2.cli_inspect_refiner_data \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --window_index 0 \
  --output_json refine_v2/outputs/train/contact_subset/refiner_data/sample0_summary.json
```

Equivalent command script:

```bash
bash refine_v2/commands/11_inspect_refiner_data.sh
```

The CLI prints:

- sequence/window counts
- action type distribution
- bucket distribution
- hand distribution
- primary region distribution
- top-k region distribution
- motion/contact tensor shapes
- one-window metadata
- coarse/GT contact ratios by region
- min-distance summaries

It exports only summaries, not large tensors.

## Validation Performed

Local checks completed:

```text
py_compile passed
CLI --help passed
synthetic artifact smoke test passed
RefineV2WindowDataset direct indexing passed
```

The smoke test verified:

```text
actor_motion_window shape = [J, F, 30]
coarse_region_contact_mask_window shape = [6, 30]
gt_region_contact_mask_window shape = [6, 30]
topk_region_scores_numeric shape = [3, 3]
```

Full real-artifact inspection was not run locally because this machine does not
currently have all full train artifacts in place, especially:

```text
refine/dataset/train/reaction_data.npz
refine_v2/outputs/train/contact_subset/subset_manifest.json
```

The command is ready for the server environment where those artifacts exist.

## Deferred: XYZ Debug

`include_xyz=True` currently raises `NotImplementedError`.

Reason:

- the training fast path should not run SMPL-X forward in `__getitem__`
- contact labels and selector windows were already built in restored pair space
- dynamic xyz generation can be added later as an explicit debug path
- adding it now would slow iteration and increase alignment risk

## Readiness For Next Phase

This phase is sufficient to start building the minimal trainable refiner.

The next phase should focus on:

1. model input packing from the current window sample dict
2. minimal residual refiner architecture
3. motion reconstruction target
4. contact-aware losses
5. train/eval loop on the 15-action `GT+ / Pred+` subset
6. checkpoint and metric logging
7. small overfit test before full subset training

The selector/window/subset should remain frozen while the first refiner is
implemented, unless a concrete data-interface bug is found.
