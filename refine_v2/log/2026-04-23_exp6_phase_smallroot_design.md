# exp6 Phase-Smallroot Design

Date: 2026-04-23

Planned experiment:

```text
refiner_v2_exp6_phase_smallroot_10k
```

## Motivation

exp5 is the current practical baseline:

```text
better than exp3 on motion/contact eval
translation remains controlled
residual scope is hand/arm-focused
```

However, visual comparison shows exp5 is still not as contact-close as exp2.

The transl-vs-hand diagnosis on exp5 Handshake/High-five random20 showed:

```text
num_windows = 39
refined_topk_gap_to_gt = 0.0136182967
topk_dist_improvement_coarse_to_refined = 0.0122072778
refined_transl_error = 0.0456994699
refined_local_hand_error = 0.0447672709

already_good = 0.7435897436
hand_pose_issue = 0.0256410256
transl_issue = 0.0769230769
mixed_issue = 0.0769230769
metric_or_region_issue = 0.0769230769
```

Interpretation:

```text
remaining gap is not mainly pure hand-pose failure
it is a small transl/global residual + small local hand residual problem
```

Therefore exp6 should not be aggressive handstrong. It should be:

```text
mild hand strengthening
small root/transl freedom
phase-aware boundary preservation
```

## Core Idea

Add a window-phase-aware preserve loss:

```text
window center: allow more residual correction
window boundaries: preserve coarse motion more strongly
```

Main target:

```text
transl/root
```

Do not significantly restrict:

```text
hand/arm contact refinement
```

This should let Stage2 compensate a small amount of Stage1 global placement
residual while avoiding exp2-like window translation discontinuity.

## New Loss: Phase-Aware Preserve

New parameters:

```text
lambda_phase_preserve = 0.5
phase_preserve_power = 2.0

phase_preserve_transl_weight = 2.0
phase_preserve_root_weight = 1.0
phase_preserve_lower_body_weight = 0.5
phase_preserve_torso_weight = 0.3
phase_preserve_arm_weight = 0.1
phase_preserve_hand_weight = 0.05
```

Window phase weight:

```text
phase_weight[t] = (abs(t - center) / center) ** phase_preserve_power
```

For `T = 30`:

```text
t = 0 or 29 -> weight close to 1
t = 14 or 15 -> weight close to 0
```

Loss definition:

```text
L_phase_preserve =
  mean phase_weight[t] * group_weight[j] * |pred[j,t] - coarse[j,t]|
```

Expected behavior:

```text
transl/root can move more near the window center
transl/root is pulled back toward coarse near boundaries
hand/arm is only weakly preserved and should remain free to refine contact
```

## Modified Model Residual Scope

exp5:

```text
hand_delta_scale = 1.0
arm_delta_scale = 1.0
root_delta_scale = 0.2
transl_delta_scale = 0.2
lower_body_delta_scale = 0.1
```

exp6 recommendation:

```text
hand_delta_scale = 1.2
arm_delta_scale = 1.0
torso_delta_scale = 0.5
root_delta_scale = 0.25
transl_delta_scale = 0.30
lower_body_delta_scale = 0.1
```

Rationale:

```text
hand_delta_scale = 1.2:
  mild hand strengthening

root_delta_scale = 0.25:
  small root correction capacity

transl_delta_scale = 0.30:
  small Stage1 global residual compensation

lower_body_delta_scale = 0.1:
  keep lower body near frozen
```

## Modified Motion / Contact Loss Weights

Group motion loss:

```text
selected_hand_motion_weight = 3.5
same_side_arm_motion_weight = 2.0
other_hand_arm_motion_weight = 1.0
torso_root_motion_weight = 0.75
lower_body_motion_weight = 0.25
transl_motion_weight = 0.25
```

Change from exp5:

```text
selected_hand_motion_weight: 3.0 -> 3.5
```

Contact-frame loss:

```text
selected_hand_contact_weight = 5.0
same_side_arm_contact_weight = 3.0
other_upper_contact_weight = 1.0
body_contact_weight = 0.5
```

Change from exp5:

```text
selected_hand_contact_weight: 4.0 -> 5.0
```

Rationale:

```text
mildly increase selected-hand contact correction
avoid aggressive hand-only overfitting
```

## Boundary Transl Loss

exp5:

```text
lambda_boundary_trans = 2.0
boundary_trans_frames = 2
```

exp6 recommendation:

```text
lambda_boundary_trans = 1.0
boundary_trans_frames = 2
lambda_phase_preserve = 0.5
```

Rationale:

```text
phase preserve provides smooth whole-window boundary protection
hard 2-frame boundary transl loss can be reduced
```

Conservative fallback:

```text
lambda_boundary_trans = 2.0
lambda_phase_preserve = 0.5
```

Use this only if exp6 shows boundary instability.

## Parameters Kept From exp5

```text
use_geometry_features = true
use_group_gated_residual = true
use_group_weighted_loss = true
use_hand_arm_contact_loss = true

hidden_dim = 512
num_heads = 8
num_layers = 8
dropout = 0.1
mlp_ratio = 4.0

num_steps = 10000
warmup_steps = 1000
batch_size = 32
val_ratio = 0.1
split_seed = 1234

lambda_motion = 1.0
lambda_contact = 1.0
lambda_smooth = 0.05
lambda_region_dist = 0.0
smooth_l1_beta = 0.05
```

## Expected Improvements

Compared with exp5, exp6 should aim for:

```text
refined_contact_f1 higher
topk_refined_contact_f1 higher
gt_contact_contact_dist_improvement higher
refined_topk_gap_to_gt lower
diagnosis_ratio_already_good higher
```

Stability should remain acceptable:

```text
boundary_trans_jump_excess close to 0
delta_norm_transl can rise slightly but must remain small
no exp2-like window discontinuity in aitviewer
no obvious hand-hand or hand-arm penetration
```

exp5 reference:

```text
refined_contact_f1 = 0.8221591739
topk_refined_contact_f1 = 0.8297871497
gt_contact_contact_dist_improvement = 0.0028254371
boundary_trans_jump_excess = -0.0000008370
delta_norm_transl = 0.0002474822
refined_topk_gap_to_gt = 0.0136182967
```

## Implementation Tasks

1. Add phase-aware preserve loss to `RefineV2Loss`.
2. Add loss config/CLI args:

```text
lambda_phase_preserve
phase_preserve_power
phase_preserve_transl_weight
phase_preserve_root_weight
phase_preserve_lower_body_weight
phase_preserve_torso_weight
phase_preserve_arm_weight
phase_preserve_hand_weight
```

3. Report `loss_phase_preserve` in train/eval metrics.
4. Add grouped commands:

```text
refine_v2/commands/train/03_train_refiner_phase_smallroot.sh
refine_v2/commands/eval/05_eval_refiner_phase_smallroot.sh
refine_v2/commands/eval/06_eval_contact_refiner_phase_smallroot.sh
refine_v2/commands/visual/06_export_refiner_vis_pack_phase_smallroot.sh
refine_v2/commands/visual/07_diagnose_refiner_vis_pack_phase_smallroot.sh
```

5. Save outputs under:

```text
refine_v2/save/train/refiner_v2_exp6_phase_smallroot_10k
```

## Final Recommendation

Implement and run:

```text
refiner_v2_exp6_phase_smallroot_10k
```

Do not run aggressive handstrong first.

Reason:

```text
current diagnosis shows the remaining gap is not pure hand-pose failure
small transl/root residual compensation is likely useful
phase-aware preserve gives that compensation a safer window-local structure
```
