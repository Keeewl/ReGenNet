# refine_v2_v1 Scope-Aware Hand/Arm Contact Refiner Framework

Date: 2026-04-22

## Goal Definition

The Stage2 refine_v2 goal is fixed as:

```text
Improve reactor contact quality, mainly through hand/arm contact refinement.
```

Stage2 should not be treated as a full-body motion generator and should not be
responsible for fixing large Stage1 global translation/alignment errors.

Practical definition:

```text
Stage2 refine_v2 = contact-aware reactor residual refiner
focused on hand/arm contact quality with controlled full-body residuals.
```

Desired scope:

```text
hand / arm: high refinement freedom
torso / root: limited refinement freedom
translation: strongly controlled
lower body: minimal change
```

## Current Baselines

### exp2

Role:

```text
contact upper reference, not practical baseline
```

Reason:

- strongest contact metrics
- visually showed obvious reactor translation discontinuity
- therefore not acceptable as the practical Stage2 baseline

### exp3

Role:

```text
current stable practical baseline
```

Config:

```text
lambda_boundary_trans = 2.0
boundary_trans_frames = 2
```

Observed:

```text
refined_contact_f1 = 0.8187
topk_refined_contact_f1 = 0.8264
gt_contact_contact_dist_improvement = 0.00236
surrogate_penetration_depth_improvement = -0.000052
```

Reason:

- boundary-constrained
- translation discontinuity is controlled much better than exp2
- contact is weaker than exp2 but stronger than exp4

### exp4

Role:

```text
conservative reference
```

Config:

```text
lambda_boundary_trans = 1.0
boundary_trans_frames = 2
num_steps = 10000
```

Observed:

```text
refined_contact_f1 = 0.8155
topk_refined_contact_f1 = 0.8232
gt_contact_contact_dist_improvement = 0.00202
surrogate_penetration_depth_improvement = -0.000049
```

Conclusion:

- lowering boundary loss from 2.0 to 1.0 did not recover contact quality
- exp4 is slightly safer on surrogate penetration, but weaker on contact
- scalar boundary-loss tuning should not remain the main optimization route

## Main Diagnosis

The current model has a structural tradeoff:

```text
strong contact correction often uses translation/root/body motion
translation continuity requires suppressing global/root movement
```

Because the current residual head can modify the whole reactor motion equally,
the model does not know which parts are allowed to move freely and which parts
should preserve coarse motion.

Therefore, the next version should focus on:

```text
scope-aware residual control
hand/arm-focused contact loss
efficient geometry features
scope-aware evaluation
```

## Final v1 Framework

Target name:

```text
refine_v2_v1: scope-aware hand/arm contact refiner
```

Expected experiment name:

```text
refiner_v2_exp5_scope_geom_10k
```

Expected save path:

```text
refine_v2/save/train/refiner_v2_exp5_scope_geom_10k
```

## Required v1 Components

### 1. Offline Relative Geometry Feature Cache

This is the most important feature upgrade.

Motivation:

Current features tell the model contact distance, but not the direction in
which the reactor hand should move.

Feature source:

```text
actor_motion
reactor_coarse
selector window hand side
selector top-k target regions
region map
```

Do not use GT reactor to compute these features. The feature cache must avoid
GT leakage.

Suggested cached fields per window:

```text
primary_relative_vector        [3, T]
primary_relative_dist          [T]
topk_relative_vectors          [K, 3, T]
topk_relative_dists            [K, T]
```

Definition:

```text
relative vector = actor target-region centroid - reactor selected-hand centroid
```

Implementation requirement:

```text
Compute offline and save as a feature cache.
Do not run expensive SMPL-X geometry forward inside every training step.
```

### 2. Dataset / Condition Encoder Geometry Feature Input

`RefineV2WindowDataset` should support an optional:

```text
geometry_feature_cache_path
```

When present, batches should include:

```text
primary_relative_vector_window
primary_relative_dist_window
topk_relative_vectors_window
topk_relative_dists_window
```

The condition encoder should encode these into the per-frame condition stream.

### 3. Joint-Group Gated Residual

The model may still predict full-body residuals, but residuals should be scaled
by joint group.

First version should use fixed group scales, not learned gates.

Suggested default scales:

```text
hand_delta_scale = 1.0
arm_delta_scale = 1.0
torso_delta_scale = 0.5
root_delta_scale = 0.2
transl_delta_scale = 0.2
lower_body_delta_scale = 0.1
```

Purpose:

```text
Make hand/arm correction easy.
Make translation/lower-body correction difficult.
```

This is the main mechanism for aligning model scope with the Stage2 goal.

### 4. Group-Weighted Motion Loss

Replace or augment the current uniform full-body motion loss.

Suggested group weights:

```text
selected_hand = 3.0
same_side_arm = 2.0
other_hand_arm = 1.0
torso_root = 0.75
lower_body = 0.25
transl = 0.25
```

Purpose:

```text
Prioritize hand/arm correction.
Avoid over-optimizing lower body and translation toward GT.
```

### 5. Hand/Arm Contact-Weighted Loss

The current contact-weighted loss applies broadly to all motion dimensions. v1
should make contact weighting hand/arm-focused.

Suggested contact-frame weights:

```text
selected_hand_contact_weight = 4.0
same_side_arm_contact_weight = 3.0
other_upper_contact_weight = 1.0
body_contact_weight = 0.5
```

Purpose:

```text
Use contact frames to improve the body parts that actually form contact.
Do not let contact supervision encourage broad full-body/translation changes.
```

### 6. Boundary Translation Loss

Keep exp3 boundary translation loss:

```text
lambda_boundary_trans = 2.0
boundary_trans_frames = 2
```

Rationale:

- exp3 is the selected stable practical baseline
- exp4 showed that simply lowering the lambda did not recover contact quality
- with joint-group gating, translation is already controlled, so this term can
  remain as a boundary anchor rather than the sole continuity mechanism

### 7. Joint-Group Delta Norm Evaluation

Add eval metrics to measure where the model actually changes motion.

Required metrics:

```text
delta_norm_selected_hand
delta_norm_same_side_arm
delta_norm_other_hand_arm
delta_norm_torso_root
delta_norm_lower_body
delta_norm_transl
```

Interpretation:

```text
hand/arm delta should be clearly larger than lower_body/transl delta.
```

This is required to verify that the refiner's scope matches the Stage2 target.

## Explicitly Deferred From v1

The following are intentionally not part of v1:

```text
action type embedding
window phase feature
raw segment phase feature
contact-token conditioning
separate residual heads
standalone lower-body preserve loss
extra full-window translation regularization
translation velocity loss
full-sequence stitching continuity metric
geometry loss in the training loop
```

Reasons:

- keep the next implementation focused and testable
- avoid changing too many axes at once
- preserve the unrestrained task setting used by Stage1
- use offline geometry features first before adding dynamic geometry losses

## Optional Later Additions

These can be considered after v1 results:

```text
metric-ranked visualization pack export
full-sequence continuity metric
window/segment phase features
learned group gates
separate residual heads
dynamic contact geometry loss
Stage1 coarse translation/alignment improvement
```

## v1 Training Plan

Recommended run:

```text
experiment = refiner_v2_exp5_scope_geom_10k
num_steps = 10000
hidden_dim = 512
num_layers = 8
num_heads = 8
lambda_boundary_trans = 2.0
boundary_trans_frames = 2
relative_geometry_cache = enabled
joint_group_gated_residual = enabled
group_weighted_motion_loss = enabled
hand_arm_contact_weighted_loss = enabled
```

## v1 Success Criteria

Compare against exp3 as the stable practical baseline.

Desired:

```text
refined_contact_f1 >= 0.84
topk_refined_contact_f1 >= 0.845
gt_contact_contact_dist_improvement >= 0.004
surrogate_penetration_depth_improvement not clearly worse than exp3
boundary_trans_jump_excess close to 0
delta_norm_hand/arm clearly greater than delta_norm_lower_body/transl
```

Qualitative requirement:

```text
aitviewer should show stronger hand/arm contact correction than exp3
without obvious reactor translation discontinuity.
```

## Current Implementation Priority

Implementation order:

1. Build offline relative geometry feature cache.
2. Add dataset + collate support for geometry cache.
3. Add condition encoder geometry inputs.
4. Add fixed joint-group gated residual scales.
5. Add group-weighted motion loss and hand/arm contact-weighted loss.
6. Add joint-group delta norm eval metrics.
7. Add train/eval/visual commands under grouped command directories.
8. Run exp5 and compare with exp3/exp4.

