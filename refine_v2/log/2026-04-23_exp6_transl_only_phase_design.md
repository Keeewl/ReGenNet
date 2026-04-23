# exp6 Transl-Only Phase Preserve Design

Date: 2026-04-23

Experiment:

```text
refiner_v2_exp6_transl_phase_10k
```

## Motivation

The previous phase-smallroot experiment showed that the phase-aware preserve
loss is useful as a stability mechanism, but the tested scope was too broad.
It suppressed hand/arm residuals and reduced contact improvement.

The revised exp6 keeps the idea but narrows the scope:

```text
apply phase preserve only to reactor transl
do not apply it to hand, arm, torso, root, or lower body
keep exp5 hand/arm model and loss parameters unchanged
```

## Design

Use exp5 as the base:

```text
refiner_v2_exp5_scope_geom_10k
```

Keep exp5 model scope and contact loss:

```text
hand_delta_scale = 1.0
arm_delta_scale = 1.0
torso_delta_scale = 0.5
root_delta_scale = 0.2
transl_delta_scale = 0.2
lower_body_delta_scale = 0.1

selected_hand_motion_weight = 3.0
same_side_arm_motion_weight = 2.0
other_hand_arm_motion_weight = 1.0
torso_root_motion_weight = 0.75
lower_body_motion_weight = 0.25
transl_motion_weight = 0.25

selected_hand_contact_weight = 4.0
same_side_arm_contact_weight = 3.0
other_upper_contact_weight = 1.0
body_contact_weight = 0.5
```

Keep boundary translation anchor:

```text
lambda_boundary_trans = 2.0
boundary_trans_frames = 2
```

Add transl-only phase preserve:

```text
lambda_phase_preserve = 0.2
phase_preserve_power = 2.0
phase_preserve_transl_weight = 1.0
phase_preserve_root_weight = 0.0
phase_preserve_lower_body_weight = 0.0
phase_preserve_torso_weight = 0.0
phase_preserve_arm_weight = 0.0
phase_preserve_hand_weight = 0.0
```

The phase curve remains the existing quadratic ramp:

```text
phase_weight[t] = (abs(t - center) / center) ** 2
```

Interpretation:

```text
window center: weak/no transl preserve, allowing small transl correction
window edges: strong transl preserve, discouraging boundary jumps
hand/arm: unaffected by phase preserve
```

## Why This Is Cleaner Than Phase-Smallroot

Previous exp6 phase-smallroot applied preserve loss to:

```text
transl, root, lower body, torso, arm, hand
```

That made the refiner too conservative and reduced the useful hand/arm deltas.

This revised exp6 only solves the specific problem:

```text
translation should not jump at window boundaries,
but can still make a small center-window correction for contact.
```

It does not interfere with the main Stage2 target:

```text
hand/arm contact refinement
```

## Expected Result

Compared with exp5, the desired result is:

```text
contact metrics stay close to or slightly exceed exp5
boundary_trans_jump_excess remains near zero
delta_norm_selected_hand is not reduced like phase-smallroot
delta_norm_same_side_arm is not reduced like phase-smallroot
delta_norm_transl may rise slightly but remains small
aitviewer shows no exp2-like window transition jump
```

exp5 reference:

```text
gt_contact_contact_dist_improvement = 0.0028254371
refined_contact_f1 = 0.8221591739
topk_refined_contact_f1 = 0.8297871497
boundary_trans_jump_excess = -0.0000008370
delta_norm_selected_hand = 0.0124185895
delta_norm_same_side_arm = 0.0155224866
delta_norm_transl = 0.0002474822
```

## Commands

Train:

```text
bash refine_v2/commands/train/04_train_refiner_exp6_transl_phase.sh
```

Evaluate:

```text
bash refine_v2/commands/eval/07_eval_refiner_exp6_transl_phase.sh
bash refine_v2/commands/eval/08_eval_contact_refiner_exp6_transl_phase.sh
```

Visualize / diagnose:

```text
bash refine_v2/commands/visual/08_export_refiner_vis_pack_exp6_transl_phase.sh
bash refine_v2/commands/visual/09_diagnose_refiner_vis_pack_exp6_transl_phase.sh
```

Output path:

```text
refine_v2/save/train/refiner_v2_exp6_transl_phase_10k
```
