# exp5 vs exp2 Visual Contact Gap

Date: 2026-04-23

## Observation

Visual inspection on the same High-five / Handshake-style sample shows:

```text
exp5 refined contact is stable but still conservative.
exp2 refined contact appears visibly closer.
```

In the screenshots:

```text
exp5: refined hand is closer than coarse, but still leaves a visible small gap.
exp2: refined hand is much closer and nearly reaches/overlaps the target hand.
```

This matches previous quantitative results:

```text
contact strength / closeness:
exp2 > exp5 > exp3 > exp4

translation / boundary stability:
exp5 ~= exp3 > exp4 >> exp2
```

## Interpretation

exp2 should not be treated as the practical baseline because it visually showed
obvious window-level translation discontinuity. However, it remains useful as a
contact upper reference.

exp5 is currently the best practical baseline:

```text
better contact than exp3
better motion/contact-frame error than exp3
translation remains controlled
residual scope is hand/arm-focused
```

But exp5 has not recovered exp2's contact closeness.

The likely cause is:

```text
exp2 can use broader full-body/root/translation residuals to force contact.
exp5 suppresses translation and lower-body/root changes, so it is more stable
but less aggressive at closing the final hand-target gap.
```

Therefore the next target is not to return to exp2 behavior, but to answer:

```text
Can we make hand/arm correction more aggressive while keeping transl frozen?
```

## Recommended Next Experiment

Proposed experiment name:

```text
refiner_v2_exp6_handstrong_10k
```

Purpose:

```text
Push hand contact closer than exp5 without reopening global translation motion.
```

Start from exp5 framework and change only hand/arm emphasis:

```text
hand_delta_scale = 1.5
arm_delta_scale = 1.0
transl_delta_scale = 0.2
lower_body_delta_scale = 0.1

selected_hand_motion_weight = 4.0
same_side_arm_motion_weight = 2.0
other_hand_arm_motion_weight = 1.0

selected_hand_contact_weight = 6.0
same_side_arm_contact_weight = 3.0
other_upper_contact_weight = 1.0
body_contact_weight = 0.5

lambda_boundary_trans = 2.0
boundary_trans_frames = 2
```

Do not loosen translation yet.

## Success Criteria

exp6 should be compared against exp5 and exp2.

Desired:

```text
contact visually closer than exp5
refined_contact_f1 > exp5
topk_refined_contact_f1 > exp5
gt_contact_contact_dist_improvement > exp5
boundary_trans_jump_excess remains close to 0
delta_norm_transl remains tiny
no obvious hand-hand / hand-arm penetration
```

If exp6 improves contact without visible transl instability:

```text
hand-strong scope tuning is the right direction.
```

If exp6 still remains too conservative:

```text
geometry features as input are not enough;
add explicit lightweight contact-distance / centroid-distance training loss.
```

If exp6 causes over-close or penetration:

```text
add anti-overclose / anti-penetration regularization before increasing hand
freedom further.
```

## Current Decision

Do not revert to exp2.

Use exp5 as the current stable baseline and run a targeted hand-strong
experiment next.
