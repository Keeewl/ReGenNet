# exp6 Phase-Smallroot Eval Summary

Date: 2026-04-23

Experiment:

```text
refiner_v2_exp6_phase_smallroot_10k
```

Baseline for comparison:

```text
refiner_v2_exp5_scope_geom_10k
```

## Goal

exp6 tested whether a window-phase-aware preserve loss could let the refiner
move more near the middle of a 30-frame window while preserving root/translation
near the boundaries.

The intended effect was:

```text
keep exp5 translation stability
recover more contact closeness
avoid weakening hand/arm refinement
```

## Window Eval Results

exp6:

```text
pred_motion_error = 0.0140063309
motion_improvement = 0.0023574536
pred_contact_motion_error = 0.0140281759
contact_motion_improvement = 0.0025201235

boundary_trans_jump_excess = -0.0000019395
coarse_boundary_trans_jump = 0.0097274078
pred_boundary_trans_jump = 0.0097254684

delta_norm_selected_hand = 0.0106538811
delta_norm_same_side_arm = 0.0120693502
delta_norm_other_hand_arm = 0.0090283277
delta_norm_torso_root = 0.0012815233
delta_norm_lower_body = 0.0014016628
delta_norm_transl = 0.0002735069

loss_motion = 0.0055699482
loss_contact_weighted = 0.0054891241
loss_smooth = 0.0007798389
loss_phase_preserve = 0.0004353090
loss_boundary_trans = 0.0000002473
loss_total = 0.0113159660
```

exp5 reference:

```text
pred_motion_error = 0.0130820452
motion_improvement = 0.0032817393
pred_contact_motion_error = 0.0130605551
contact_motion_improvement = 0.0034877443

boundary_trans_jump_excess = -0.0000008370

delta_norm_selected_hand = 0.0124185895
delta_norm_same_side_arm = 0.0155224866
delta_norm_other_hand_arm = 0.0110871044
delta_norm_torso_root = 0.0029985403
delta_norm_lower_body = 0.0044759440
delta_norm_transl = 0.0002474822
```

Comparison:

```text
exp6 worsens pred_motion_error by about +0.000924
exp6 loses about 28.2% of exp5 motion improvement
exp6 worsens pred_contact_motion_error by about +0.000968
exp6 loses about 27.7% of exp5 contact-frame motion improvement
boundary translation remains stable and close to coarse
```

Scope interpretation:

```text
selected hand delta: 0.01242 -> 0.01065
same-side arm delta: 0.01552 -> 0.01207
other hand/arm delta: 0.01109 -> 0.00903
torso/root delta: 0.00300 -> 0.00128
lower body delta: 0.00448 -> 0.00140
translation delta: 0.000247 -> 0.000274
```

The phase preserve loss successfully made the model more conservative on
root/lower-body motion, but it also weakened the hand/arm correction that is
needed for Stage2 contact refinement.

## Contact Eval Results

exp6:

```text
all_valid_dist_l1_improvement = 0.0020643177
gt_contact_contact_dist_improvement = 0.0022491207
coarse_contact_f1 = 0.8003375103
refined_contact_f1 = 0.8172011076
topk_coarse_contact_f1 = 0.8083653870
topk_refined_contact_f1 = 0.8250681969
surrogate_penetration_depth_improvement = -0.0000531603
```

exp5 reference:

```text
all_valid_dist_l1_improvement = 0.0027992890
gt_contact_contact_dist_improvement = 0.0028254371
refined_contact_f1 = 0.8221591739
topk_refined_contact_f1 = 0.8297871497
surrogate_penetration_depth_improvement = -0.0000661652
```

Comparison:

```text
all-valid distance improvement drops by about 26.3%
GT-contact distance improvement drops by about 20.4%
refined contact F1 drops by about 0.0050
top-k refined contact F1 drops by about 0.0047
surrogate penetration is slightly safer than exp5, but still worse than coarse
```

## Conclusion

exp6 should not replace exp5.

The phase-smallroot direction is technically valid for stabilizing scope, but
the tested setting is too conservative for the current Stage2 objective. It
reduces the hand/arm residuals and loses contact improvement.

Current practical baseline remains:

```text
refiner_v2_exp5_scope_geom_10k
```

## Next Recommendation

Do not adopt the current exp6 setting.

If phase preserve is tried again, make it much lighter and restrict it mostly to
translation/root:

```text
lambda_phase_preserve = 0.1 or 0.2
phase_preserve_hand_weight = 0.0
phase_preserve_arm_weight = 0.0
phase_preserve_torso_weight = 0.0 or 0.1
phase_preserve_lower_body_weight = 0.2
phase_preserve_root_weight = 0.5
phase_preserve_transl_weight = 1.0
lambda_boundary_trans = 2.0
```

More promising next experiment:

```text
start from exp5
keep phase preserve off
keep translation conservative
increase hand/contact supervision moderately
```

Candidate:

```text
refiner_v2_exp7_mild_hand_geom_10k

hand_delta_scale = 1.2
selected_hand_motion_weight = 3.5
selected_hand_contact_weight = 5.0
root_delta_scale = 0.2
transl_delta_scale = 0.2
lambda_boundary_trans = 2.0
lambda_phase_preserve = 0.0
```

This keeps the exp5 structure and avoids the main failure mode of exp6:
over-preserving the same hand/arm corrections that produce contact gains.
