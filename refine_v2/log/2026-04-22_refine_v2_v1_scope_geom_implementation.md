# refine_v2_v1 Scope-Geometry Implementation

Date: 2026-04-22

## Implemented Scope

This update implements the planned `refine_v2_v1` framework:

```text
scope-aware hand/arm contact refiner
offline relative geometry features
group-gated residual control
group-weighted motion/contact losses
joint-group delta-norm evaluation
```

The selector/window/subset protocol is unchanged and remains frozen.

## New Geometry Feature Cache

Added:

```text
refine_v2/tools/build_geometry_feature_cache.py
refine_v2/cli_build_geometry_feature_cache.py
```

The cache computes per-window relative geometry from:

```text
actor_motion
reactor_coarse
selector hand side
selector primary/top-k target regions
SMPL-X region map
```

It does not use GT reactor motion.

Cached fields:

```text
primary_relative_vector_window  [N, 3, T]
primary_relative_dist_window    [N, T]
topk_relative_vectors_window    [N, K, 3, T]
topk_relative_dists_window      [N, K, T]
```

Alignment fields are also saved:

```text
dataset_row_indices
sample_indices
window_indices
start_frames
end_frames
hand_side_ids
primary_target_region_ids
topk_target_region_ids
```

The dataset validates cache/window alignment strictly before training.

## Dataset / Model Updates

Updated:

```text
refine_v2/refiner_data/schema.py
refine_v2/refiner_data/sanity_checks.py
refine_v2/refiner_data/window_dataset.py
refine_v2/refiner_data/window_loader.py
refine_v2/model/condition_encoder.py
refine_v2/model/refiner_v2.py
```

`RefineV2WindowDataset` now accepts:

```text
geometry_feature_cache_path
```

When present, batches include geometry tensors. When absent, old training/eval
paths remain compatible.

The condition encoder supports:

```text
use_geometry_features
```

When enabled, it encodes primary and top-k relative vector/distance features
into the per-frame condition stream.

## Scope-Aware Residual Control

Added:

```text
refine_v2/model/joint_groups.py
```

The model now supports fixed group-gated residual scaling:

```text
use_group_gated_residual
hand_delta_scale       = 1.0
arm_delta_scale        = 1.0
torso_delta_scale      = 0.5
root_delta_scale       = 0.2
transl_delta_scale     = 0.2
lower_body_delta_scale = 0.1
```

Purpose:

```text
allow hand/arm correction
discourage translation/lower-body correction
preserve Stage2 as a contact refiner rather than a full-body generator
```

## Loss Updates

Updated:

```text
refine_v2/model/losses_v2.py
refine_v2/train/trainer.py
refine_v2/cli_train_refiner.py
```

New optional loss modes:

```text
use_group_weighted_loss
use_hand_arm_contact_loss
```

Default v1 motion weights:

```text
selected_hand = 3.0
same_side_arm = 2.0
other_hand_arm = 1.0
torso_root = 0.75
lower_body = 0.25
transl = 0.25
```

Default v1 contact-frame weights:

```text
selected_hand = 4.0
same_side_arm = 3.0
other_upper = 1.0
body = 0.5
```

Boundary translation loss remains available and exp5 defaults to:

```text
lambda_boundary_trans = 2.0
boundary_trans_frames = 2
```

## Eval Updates

Updated:

```text
refine_v2/train/eval_window.py
refine_v2/cli_eval_refiner.py
refine_v2/eval/contact_eval_refiner.py
refine_v2/tools/eval_contact_refiner.py
refine_v2/tools/export_refiner_vis_pack.py
```

Window eval now reports joint-group delta norms:

```text
delta_norm_selected_hand
delta_norm_same_side_arm
delta_norm_other_hand_arm
delta_norm_torso_root
delta_norm_lower_body
delta_norm_transl
```

Eval/contact-eval/visual export all accept:

```text
geometry_feature_cache_path
```

If a checkpoint was trained with geometry features, eval will require the cache.

## Commands

Added grouped commands:

```text
refine_v2/commands/features/01_build_geometry_cache_scope_geom.sh
refine_v2/commands/train/02_train_refiner_scope_geom.sh
refine_v2/commands/eval/03_eval_refiner_scope_geom.sh
refine_v2/commands/eval/04_eval_contact_refiner_scope_geom.sh
refine_v2/commands/visual/03_export_refiner_vis_pack_scope_geom.sh
refine_v2/commands/visual/04_view_refiner_vis_pack_scope_geom.sh
```

Main run paths:

```text
feature cache:
refine_v2/save/features/scope_geom_train/geometry_feature_cache.npz

training:
refine_v2/save/train/refiner_v2_exp5_scope_geom_10k
```

## Validation

Completed checks:

```text
python3 -m py_compile ...
python3 -m refine_v2.cli_build_geometry_feature_cache --help
python3 -m refine_v2.cli_train_refiner --help
python3 -m refine_v2.cli_eval_refiner --help
python3 -m refine_v2.cli_eval_contact_refiner --help
python3 -m refine_v2.cli_export_refiner_vis_pack --help
```

Smoke tests:

```text
geometry-enabled model forward + group-gated residual + grouped loss: passed
legacy no-geometry model forward: passed
```

Full cache build and full exp5 training should run on the GPU machine.

## Expected Next Step

Run:

```text
bash refine_v2/commands/features/01_build_geometry_cache_scope_geom.sh
bash refine_v2/commands/train/02_train_refiner_scope_geom.sh
```

Then evaluate against exp3:

```text
bash refine_v2/commands/eval/03_eval_refiner_scope_geom.sh
bash refine_v2/commands/eval/04_eval_contact_refiner_scope_geom.sh
bash refine_v2/commands/visual/03_export_refiner_vis_pack_scope_geom.sh
```

Primary comparison metrics:

```text
refined_contact_f1
topk_refined_contact_f1
gt_contact_contact_dist_improvement
surrogate_penetration_depth_improvement
boundary_trans_jump_excess
delta_norm_selected_hand / delta_norm_transl
delta_norm_selected_hand / delta_norm_lower_body
```
