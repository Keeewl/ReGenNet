# exp7 Contact-Refine V1 Implementation

Date: 2026-04-23

Experiment:

```text
refiner_v2_exp7_contact_refine_v1_10k
```

Baseline:

```text
refiner_v2_exp5_scope_geom_10k
```

## Goal

Implement the next contact-aware refiner version without changing the frozen
selector/window/subset protocol.

Main objective:

```text
improve reactor hand/arm contact quality
```

Do not turn Stage2 into a broad translation correction module.

## Implemented Components

### 1. Geometry Feature Cache V2

Updated:

```text
refine_v2/tools/build_geometry_feature_cache.py
refine_v2/refiner_data/schema.py
refine_v2/refiner_data/window_dataset.py
```

Existing exp5 feature fields remain:

```text
primary_relative_vector_window
primary_relative_dist_window
topk_relative_vectors_window
topk_relative_dists_window
```

New v2 fields:

```text
gt_primary_relative_vector_window
gt_primary_relative_dist_window
gt_topk_relative_vectors_window
gt_topk_relative_dists_window

topk_relative_dist_velocity_window
topk_gt_relative_dist_velocity_window
topk_relative_dist_gap_window
contact_geometry_weight_window

coarse_topk_nearest_vectors_window
coarse_topk_nearest_dists_window
gt_topk_nearest_vectors_window
gt_topk_nearest_dists_window
topk_nearest_dist_gap_window
```

Input-safe model features:

```text
coarse top-k relative vector/distance
coarse top-k distance velocity
coarse nearest selected-hand-vertex to target-region centroid vector/distance
```

Supervision-only fields:

```text
GT relative vector/distance
GT nearest distance
coarse-vs-GT distance gap
contact geometry weights
```

Design note:

```text
GT geometry is stored in the cache only for loss/supervision.
The condition encoder only consumes coarse/current fields.
```

### 2. Geometry V2 Condition Encoder

Updated:

```text
refine_v2/model/condition_encoder.py
refine_v2/model/refiner_v2.py
refine_v2/cli_train_refiner.py
refine_v2/train/trainer.py
```

New flags:

```text
--use_geometry_v2_features
--use_separate_residual_heads
```

When `use_geometry_v2_features=True`, the condition encoder additionally uses:

```text
topk_relative_dist_velocity_window
coarse_topk_nearest_vectors_window
coarse_topk_nearest_dists_window
```

### 3. Separate Residual Heads

Updated:

```text
refine_v2/model/refiner_v2.py
```

Optional separate output heads:

```text
hand_output_head
arm_output_head
body_output_head
transl_output_head
```

The heads are combined with fixed joint-group masks:

```text
hand ids  -> hand head
arm ids   -> arm head
transl    -> transl head
remaining -> body head
```

Then the existing group-gated residual scale is still applied.

Purpose:

```text
hand/arm can learn stronger contact corrections
body/transl remain conservative
```

### 4. Contact-Aware Lightweight Losses

Updated:

```text
refine_v2/model/losses_v2.py
refine_v2/train/trainer.py
refine_v2/cli_train_refiner.py
refine_v2/cli_eval_refiner.py
```

New loss config:

```text
lambda_contact_geometry
lambda_gt_relative_overclose
contact_geometry_weight_scale
gt_relative_overclose_margin
```

`loss_contact_geometry`:

```text
uses contact_geometry_weight_window from cache v2
focuses selected-hand and same-side-arm motion supervision
on frames where coarse hand-target distance is worse than GT
```

This is a fast proxy for contact geometry supervision. It avoids dense SMPL-X
forward inside every training step.

`loss_gt_relative_overclose`:

```text
uses topk_nearest_dist_gap_window
penalizes selected-hand / same-side-arm delta only when coarse is already
closer than GT beyond a margin
```

This is a lightweight GT-relative overclose safeguard, not an absolute
penetration minimizer.

### 5. Contact Eval GT-Relative Penetration Report

Updated:

```text
refine_v2/eval/contact_eval_refiner.py
```

Contact eval now reports GT-relative surrogate penetration fields:

```text
gt_surrogate_penetration_rate
gt_surrogate_penetration_depth
coarse_vs_gt_surrogate_penetration_rate_gap
refined_vs_gt_surrogate_penetration_rate_gap
surrogate_penetration_rate_gap_improvement
coarse_vs_gt_surrogate_penetration_depth_gap
refined_vs_gt_surrogate_penetration_depth_gap
surrogate_penetration_depth_gap_improvement
refined_penetration_depth_excess_over_gt
```

This matches the updated principle:

```text
penetration should be judged relative to GT, not minimized blindly.
```

## Commands Added

Build geometry cache v2:

```text
refine_v2/commands/features/02_build_geometry_cache_v2_contact.sh
```

Train:

```text
refine_v2/commands/train/05_train_refiner_exp7_contact_refine_v1.sh
```

Eval:

```text
refine_v2/commands/eval/09_eval_refiner_exp7_contact_refine_v1.sh
refine_v2/commands/eval/10_eval_contact_refiner_exp7_contact_refine_v1.sh
```

Visual:

```text
refine_v2/commands/visual/10_export_refiner_vis_pack_exp7_contact_refine_v1.sh
refine_v2/commands/visual/11_diagnose_refiner_vis_pack_exp7_contact_refine_v1.sh
```

Output paths:

```text
refine_v2/save/features/contact_geom_v2_train/geometry_feature_cache_v2.npz
refine_v2/save/train/refiner_v2_exp7_contact_refine_v1_10k
```

## exp7 Default Parameters

Keep exp5 model/loss scope:

```text
hand_delta_scale = 1.0
arm_delta_scale = 1.0
torso_delta_scale = 0.5
root_delta_scale = 0.2
transl_delta_scale = 0.2
lower_body_delta_scale = 0.1

selected_hand_motion_weight = 3.0
same_side_arm_motion_weight = 2.0
selected_hand_contact_weight = 4.0
same_side_arm_contact_weight = 3.0

lambda_boundary_trans = 2.0
lambda_phase_preserve = 0.0
```

Enable new contact-aware components:

```text
use_geometry_v2_features = true
use_separate_residual_heads = true

lambda_contact_geometry = 0.5
lambda_gt_relative_overclose = 0.05
contact_geometry_weight_scale = 0.05
gt_relative_overclose_margin = 0.005
```

## Validation

Completed:

```text
python -m py_compile refine_v2/tools/build_geometry_feature_cache.py refine_v2/refiner_data/schema.py refine_v2/refiner_data/window_dataset.py refine_v2/model/condition_encoder.py refine_v2/model/refiner_v2.py refine_v2/model/losses_v2.py refine_v2/train/trainer.py refine_v2/cli_train_refiner.py refine_v2/cli_eval_refiner.py refine_v2/eval/contact_eval_refiner.py
python -m refine_v2.cli_train_refiner --help
zsh -n refine_v2/commands/features/02_build_geometry_cache_v2_contact.sh
zsh -n refine_v2/commands/train/05_train_refiner_exp7_contact_refine_v1.sh
zsh -n refine_v2/commands/eval/09_eval_refiner_exp7_contact_refine_v1.sh
zsh -n refine_v2/commands/eval/10_eval_contact_refiner_exp7_contact_refine_v1.sh
zsh -n refine_v2/commands/visual/10_export_refiner_vis_pack_exp7_contact_refine_v1.sh
zsh -n refine_v2/commands/visual/11_diagnose_refiner_vis_pack_exp7_contact_refine_v1.sh
```

Smoke test:

```text
geometry-v2 conditioned model forward passed
separate residual heads forward passed
contact geometry loss returned finite values
GT-relative overclose loss returned finite values
```

## Run Order

```text
bash refine_v2/commands/features/02_build_geometry_cache_v2_contact.sh
bash refine_v2/commands/train/05_train_refiner_exp7_contact_refine_v1.sh
bash refine_v2/commands/eval/09_eval_refiner_exp7_contact_refine_v1.sh
bash refine_v2/commands/eval/10_eval_contact_refiner_exp7_contact_refine_v1.sh
bash refine_v2/commands/visual/10_export_refiner_vis_pack_exp7_contact_refine_v1.sh
bash refine_v2/commands/visual/11_diagnose_refiner_vis_pack_exp7_contact_refine_v1.sh
```

## Success Criteria

Compare against exp5:

```text
refined_contact_f1 = 0.8221591739
topk_refined_contact_f1 = 0.8297871497
gt_contact_contact_dist_improvement = 0.0028254371
boundary_trans_jump_excess = -0.0000008370
```

Desired exp7 outcome:

```text
refined_contact_f1 improves toward 0.835
topk_refined_contact_f1 improves toward 0.840
gt_contact_contact_dist_improvement improves toward 0.0035+
boundary translation remains stable
refined-vs-GT penetration gap does not worsen substantially
aitviewer shows closer contact without exp2-like translation jumps
```
