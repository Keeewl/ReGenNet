# exp7 Ablation Results Summary

Date: 2026-04-24

Experiments:

```text
refiner_v2_exp7a_geom_v2_only_10k
refiner_v2_exp7b_geom_v2_light_contact_10k
```

Reference baseline:

```text
refiner_v2_exp5_scope_geom_10k
```

## Goal

The purpose of the ablation was to separate three possible causes of the failed
`exp7_contact_refine_v1` result:

```text
geometry v2 input
separate residual heads
contact proxy loss
```

The ablation kept the model closer to exp5 and tested only:

```text
exp7a = geometry v2 input only
exp7b = geometry v2 input + very light contact proxy loss
```

## exp7a: Geometry V2 Only

Window eval:

```text
pred_motion_error = 0.0145058566
pred_contact_motion_error = 0.0145755637
motion_improvement = 0.0018579279
contact_motion_improvement = 0.0019727358

delta_norm_selected_hand = 0.0095506160
delta_norm_same_side_arm = 0.0119299375
delta_norm_transl = 0.0001373900
```

Contact eval:

```text
all_valid_dist_l1_improvement = 0.0019111525
gt_contact_contact_dist_improvement = 0.0024259641
refined_contact_f1 = 0.8164992863
topk_refined_contact_f1 = 0.8244754395
surrogate_penetration_depth_improvement = -0.0000660962
```

Interpretation:

```text
exp7a is clearly worse than exp5 on both motion and contact metrics.
geometry v2 input alone does not improve the current refiner.
the model also becomes more conservative, with smaller hand/arm residuals.
```

## exp7b: Geometry V2 + Light Contact Proxy Loss

Window eval:

```text
pred_motion_error = 0.0137144559
pred_contact_motion_error = 0.0137188103
motion_improvement = 0.0026493286
contact_motion_improvement = 0.0028294891

delta_norm_selected_hand = 0.0105627447
delta_norm_same_side_arm = 0.0135194269
delta_norm_transl = 0.0002994282
```

Contact eval:

```text
all_valid_dist_l1_improvement = 0.0024325577
gt_contact_contact_dist_improvement = 0.0026372235
refined_contact_f1 = 0.8203763736
topk_refined_contact_f1 = 0.8281061570
surrogate_penetration_depth_improvement = -0.0000646246
```

Interpretation:

```text
exp7b is clearly better than exp7a.
therefore, a light contact proxy loss is directionally useful.
however, exp7b still does not beat exp5.
```

## Comparison Against exp5

exp5 reference:

```text
pred_motion_error = 0.0130820452
pred_contact_motion_error = 0.0130605551

all_valid_dist_l1_improvement = 0.0027992890
gt_contact_contact_dist_improvement = 0.0028254371
refined_contact_f1 = 0.8221591739
topk_refined_contact_f1 = 0.8297871497
```

Relative ordering:

```text
exp5 > exp7b > exp6_transl_phase > exp7a > exp7
```

Main conclusions:

```text
1. exp5 remains the current best baseline.
2. geometry v2 input itself did not prove useful.
3. the light contact proxy loss partially recovers performance, but still
   does not surpass exp5.
4. separate residual heads are not the main reason for exp7 failure,
   because exp7a/exp7b did not use them and still failed to beat exp5.
```

## What This Means

The current path:

```text
geometry v2 input + current proxy contact loss
```

does not justify more heavy iteration.

The ablation has already isolated the main behavior:

```text
geometry v2 as currently encoded is not enough
light proxy contact loss helps a bit
but the whole path is still weaker than the simpler exp5 baseline
```

## Recommendation

Do not continue heavy iteration on the current geometry-v2/proxy-loss branch.

Two practical options remain:

### Option A: One Last Minimal Training Experiment

Use exp5 directly and add only a *very light* contact regularization:

```text
keep exp5 features
keep exp5 shared residual head
keep exp5 model shape

lambda_contact_geometry = 0.03 to 0.05
lambda_gt_relative_overclose = 0.0
no geometry_v2_features
no separate residual heads
```

This is the safest final training-side test.

### Option B: Stop Training Iteration And Consolidate exp5

If time is tighter, this is the better choice:

```text
freeze exp5 as final practical model
finish contact eval reporting
finish GT-relative penetration reporting
finish visualization cases and qualitative summary
```

## Final Summary

The ablation successfully answered the main question:

```text
the current geometry-v2/contact-proxy path is not strong enough to replace exp5.
```

Therefore:

```text
exp5 should remain the active baseline.
At most one more minimal exp5-based regularization test is justified.
Otherwise the framework should be considered converged for this stage.
```
