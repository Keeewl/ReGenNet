# refine_v2 exp5 Scope-Geometry Eval Summary

Date: 2026-04-23

Experiment:

```text
refiner_v2_exp5_scope_geom_10k
```

Paths:

```text
checkpoint:
refine_v2/save/train/refiner_v2_exp5_scope_geom_10k/model_best.pt

window eval:
refine_v2/save/train/refiner_v2_exp5_scope_geom_10k/eval_window.json

contact eval:
refine_v2/save/train/refiner_v2_exp5_scope_geom_10k/contact_eval_window/eval_contact_refiner.json
```

## Setup

exp5 uses the v1 scope-geometry framework:

```text
offline relative geometry feature cache
geometry-conditioned condition encoder
fixed joint-group gated residual
group-weighted motion loss
hand/arm contact-weighted loss
boundary translation loss
```

Reference baseline:

```text
exp3 = stable practical boundary-trans baseline
lambda_boundary_trans = 2.0
boundary_trans_frames = 2
```

## Window-Level Eval

exp5 metrics:

```text
coarse_motion_error            = 0.0163637845
pred_motion_error              = 0.0130820452
motion_improvement             = 0.0032817393

coarse_contact_motion_error    = 0.0165482994
pred_contact_motion_error      = 0.0130605551
contact_motion_improvement     = 0.0034877443

coarse_boundary_trans_jump     = 0.0097274078
pred_boundary_trans_jump       = 0.0097265707
boundary_trans_jump_excess     = -0.0000008370

loss_total                     = 0.0097596767
loss_motion                    = 0.0048933593
loss_contact_weighted          = 0.0048111018
loss_boundary_trans            = 0.0000001791
```

Comparison with exp3:

```text
exp3 pred_motion_error          = 0.0133661932
exp5 pred_motion_error          = 0.0130820452

exp3 motion_improvement         = 0.0029975913
exp5 motion_improvement         = 0.0032817393

exp3 contact_motion_improvement = 0.0031884047
exp5 contact_motion_improvement = 0.0034877443
```

Interpretation:

```text
exp5 improves both general window motion error and contact-frame motion error.
```

## Scope / Delta Norms

exp5 delta norms:

```text
delta_norm_selected_hand   = 0.0124185895
delta_norm_same_side_arm   = 0.0155224866
delta_norm_other_hand_arm  = 0.0110871044
delta_norm_lower_body      = 0.0044759440
delta_norm_torso_root      = 0.0029985403
delta_norm_transl          = 0.0002474822
```

Ratios:

```text
selected_hand / transl     ~= 50.2x
same_side_arm / transl     ~= 62.7x
selected_hand / lower_body ~= 2.8x
same_side_arm / lower_body ~= 3.5x
```

Interpretation:

```text
The v1 scope-aware design works.
The model mainly changes hand/arm motion.
Translation is strongly suppressed.
Lower-body and torso/root changes remain much smaller than hand/arm changes.
```

The fact that same-side arm delta is larger than selected-hand delta is
acceptable for now because hand contact often depends on forearm/arm alignment.
If visualization later shows arm over-motion or insufficient hand correction,
increase selected-hand weighting or hand delta scale.

## Contact Eval

exp5 metrics:

```text
all_valid_dist_l1_improvement        = 0.0027992890
gt_contact_contact_dist_improvement  = 0.0028254371

coarse_contact_f1                    = 0.8003375103
refined_contact_f1                   = 0.8221591739

topk_coarse_contact_f1               = 0.8083653870
topk_refined_contact_f1              = 0.8297871497

surrogate_penetration_depth_improvement = -0.0000661652
surrogate_penetration_rate_improvement  = -0.0058494263
```

Comparison with exp3:

```text
exp3 all_valid_dist_l1_improvement        = 0.002390
exp5 all_valid_dist_l1_improvement        = 0.002799

exp3 gt_contact_contact_dist_improvement  = 0.002363
exp5 gt_contact_contact_dist_improvement  = 0.002825

exp3 refined_contact_f1                   = 0.81865
exp5 refined_contact_f1                   = 0.82216

exp3 topk_refined_contact_f1              = 0.82639
exp5 topk_refined_contact_f1              = 0.82979

exp3 surrogate_penetration_depth_improvement = -0.000052
exp5 surrogate_penetration_depth_improvement = -0.000066
```

Approximate improvement over exp3:

```text
all_valid_dist_l1_improvement        +17.1%
gt_contact_contact_dist_improvement  +19.6%
refined_contact_f1                   +0.0035 absolute
topk_refined_contact_f1              +0.0034 absolute
```

Interpretation:

```text
exp5 clearly improves contact distance metrics and slightly improves contact F1.
It is a better practical baseline than exp3.
```

Risk:

```text
surrogate penetration becomes slightly worse than exp3
```

This does not immediately invalidate exp5 because the metric is unsigned
min-distance based, not true signed penetration. However, it means aitviewer
inspection should focus on hand-hand and hand-arm over-close/penetration.

## Breakdown Observations

By action type, strongest contact-distance improvements:

```text
High-five        +0.01443
Dance            +0.00557
Massaging leg    +0.00437
Hand wrestling   +0.00383
Sit on leg       +0.00252
Handshake        +0.00178
```

Weakest improvements:

```text
Pull              +0.00035
Support with hand +0.00039
Hug               +0.00041
Link arms         +0.00054
Help up           +0.00078
```

Interpretation:

```text
Relative hand-target geometry helps most for direct hand-contact actions.
It is less sufficient for broader torso/arm/body interactions such as Hug,
Support with hand, Pull, and Link arms.
```

By primary region:

```text
left_hand   +0.00529
right_hand  +0.00447
torso_head  +0.00182
left_arm    +0.00163
right_arm   +0.00110
lower_body  +0.00069
```

Interpretation:

```text
The largest improvements happen on hand target regions.
This matches the Stage2 target: improve hand/arm contact quality.
```

Penetration surrogate worsens most on hand target regions:

```text
left_hand   -0.000161
right_hand  -0.000178
```

This should be checked visually before adding another loss.

## Overall Conclusion

exp5 is a successful iteration:

```text
exp5 > exp3 on motion eval
exp5 > exp3 on contact-distance eval
exp5 > exp3 on refined/top-k contact F1
translation remains controlled
delta scope is aligned with hand/arm Stage2 objective
```

exp5 should replace exp3 as the current practical baseline.

It is not yet the final upper-bound model because:

```text
refined_contact_f1 is still below the target 0.84
topk_refined_contact_f1 is still below the target 0.845
gt_contact_contact_dist_improvement is still below the target 0.004
surrogate penetration is slightly worse than exp3
```

## Next Step

Do not redesign immediately.

First run aitviewer visual checks for:

```text
High-five
Handshake
Hand wrestling
Hug
Support with hand
Pull
```

Check:

```text
does exp5 visibly improve hand/arm contact?
does it introduce hand-hand or hand-arm penetration?
does it preserve translation continuity better than exp2?
```

If visual quality is acceptable:

```text
use exp5 as the new baseline
```

If over-close/penetration is visible:

```text
add a light anti-overclose / anti-penetration regularizer
```

If weak action types remain poor:

```text
extend geometry features beyond selected hand -> target centroid,
especially for broader torso/arm/body contact actions.
```
