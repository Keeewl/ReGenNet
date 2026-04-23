# Vis Pack Transl vs Hand Diagnosis

Date: 2026-04-23

Input pack:

```text
refine_v2/save/train/refiner_v2_exp5_scope_geom_10k/vis_pack_random20/refiner_vis_pack.npz
```

Subset:

```text
Handshake + High-five
random20 sequences
39 windows
```

## Diagnostic Output

```text
num_windows = 39

refined_topk_gap_to_gt = 0.0136182967
topk_dist_improvement_coarse_to_refined = 0.0122072778

refined_transl_error = 0.0456994699
refined_local_hand_error = 0.0447672709

diagnosis_ratio_already_good = 0.7435897436
diagnosis_ratio_hand_pose_issue = 0.0256410256
diagnosis_ratio_transl_issue = 0.0769230769
diagnosis_ratio_mixed_issue = 0.0769230769
diagnosis_ratio_metric_or_region_issue = 0.0769230769
```

Approximate counts:

```text
already_good             ~= 29 / 39
hand_pose_issue          ~=  1 / 39
transl_issue             ~=  3 / 39
mixed_issue              ~=  3 / 39
metric_or_region_issue   ~=  3 / 39
```

## Interpretation

This diagnostic changes the previous hypothesis.

The remaining contact gap in this pack is not mainly a pure hand-pose problem.

Instead:

```text
most windows are already close enough to GT
pure hand-pose issue is rare
translation/global placement issues are more frequent than pure hand issues
mixed transl + hand issues are also present
```

The mean refined gap to GT is:

```text
1.36 cm
```

The refiner already improves coarse by:

```text
1.22 cm
```

So exp5 is doing meaningful contact correction, but the remaining gap is a
small residual that visually matters for direct hand-contact actions.

The two key errors are close to the current threshold:

```text
refined_transl_error     ~= 4.57 cm
refined_local_hand_error ~= 4.48 cm
threshold                = 5.00 cm
```

This means the residual gap is likely caused by both:

```text
small Stage1 global / transl placement error
small Stage2 local hand/arm pose error
```

not one single dominant source.

## Consequence for exp6

The previous aggressive handstrong proposal is now too aggressive:

```text
hand_delta_scale = 1.5
selected_hand_contact_weight = 6.0
```

That may help some windows, but the diagnostic shows pure hand failures are
only about `2.56%` in this pack. Aggressive hand-only tuning could create
over-close, hand artifacts, or penetration without solving transl-limited cases.

## Updated Next Experiment Options

### Option A: mild handstrong

Use this if Stage2 should keep translation almost completely frozen.

```text
experiment = refiner_v2_exp6_mild_handstrong_10k

hand_delta_scale = 1.2
arm_delta_scale = 1.0
root_delta_scale = 0.2
transl_delta_scale = 0.2
lower_body_delta_scale = 0.1

selected_hand_motion_weight = 3.5
selected_hand_contact_weight = 5.0
same_side_arm_contact_weight = 3.0

lambda_boundary_trans = 2.0
boundary_trans_frames = 2
```

Purpose:

```text
slightly close the remaining hand gap
preserve exp5 translation stability
```

### Option B: balanced small-root / small-transl

Use this if Stage2 is allowed to compensate a small amount of Stage1 global
placement residual.

```text
experiment = refiner_v2_exp6_balanced_smallroot_10k

hand_delta_scale = 1.2
arm_delta_scale = 1.0
root_delta_scale = 0.25
transl_delta_scale = 0.25 or 0.30
lower_body_delta_scale = 0.1

selected_hand_motion_weight = 3.5
selected_hand_contact_weight = 5.0
same_side_arm_contact_weight = 3.0

lambda_boundary_trans = 2.0
boundary_trans_frames = 2
```

Purpose:

```text
preserve exp5 stability
slightly improve hand contact
allow minimal correction for Stage1 transl/global placement residual
```

## Current Recommendation

Do not jump directly to aggressive handstrong.

Preferred next run:

```text
refiner_v2_exp6_balanced_smallroot_10k
```

Reason:

```text
transl_issue + mixed_issue = 15.38%
pure hand_pose_issue = 2.56%
```

So a small amount of root/transl freedom may be more useful than only increasing
hand loss.

If strict Stage2 scope requires no extra transl freedom, run:

```text
refiner_v2_exp6_mild_handstrong_10k
```

instead.

## Validation Required

Whichever exp6 variant is trained, validate with:

```text
window eval
contact eval
aitviewer visual pack
vis-pack transl-vs-hand diagnosis
```

Success criteria:

```text
refined_topk_gap_to_gt decreases
contact eval improves over exp5
boundary_trans_jump_excess remains close to zero
delta_norm_transl remains small
diagnosis_ratio_already_good increases
visible hand contact is closer without exp2-like transl discontinuity
```
