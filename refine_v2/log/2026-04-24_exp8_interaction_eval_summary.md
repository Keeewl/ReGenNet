# exp8 Interaction Eval Summary

Date: 2026-04-24

Experiment:

```text
refiner_v2_exp8_interaction_v1_10k
```

## Evaluation Context

This experiment should be interpreted under the corrected Stage2 objective:

```text
Stage2 is a contact-refine module, not a motion-reconstruction module.
```

Therefore:

```text
1. weaker STGCN / motion reconstruction is acceptable
2. contact metrics are the primary success criteria
3. penetration must be interpreted relative to GT, not as an absolute-minimize metric
```

The practical reading is:

```text
if refined becomes more GT-like in contact geometry,
then some reconstruction degradation is acceptable
```

## Window Eval

```text
pred_motion_error = 0.013667550909166373
motion_improvement = 0.002696233594414481

pred_contact_motion_error = 0.013666775117047735
contact_motion_improvement = 0.002881524312023031

boundary_trans_jump_excess = -2.6215136578259542e-06

delta_norm_selected_hand = 0.011754125936525648
delta_norm_same_side_arm = 0.015009680076366064
delta_norm_other_hand_arm = 0.009932678241565865
delta_norm_torso_root = 0.002694142434250904
delta_norm_lower_body = 0.003865541517748327
delta_norm_transl = 0.00020620567090465396
```

Interpretation:

```text
1. the model still keeps the Stage2 scope mostly on selected hand / same-side arm
2. transl remains small; this is not another exp2-style global-shift model
3. boundary continuity remains very stable
```

## Contact Eval

```text
all_valid_dist_l1_improvement = 0.0025347111161972244
gt_contact_contact_dist_improvement = 0.003047680515683272

refined_contact_f1 = 0.8214249439940374
coarse_contact_f1 = 0.8003375102606143

topk_refined_contact_f1 = 0.8288972412102136
topk_coarse_contact_f1 = 0.8083653870021124

surrogate_penetration_depth_improvement = -9.166959621832446e-05
```

## Comparison to exp5

exp5 remains slightly stronger on binary/contact-F1-style metrics:

```text
refined_contact_f1:
  exp5 = 0.8221591739402535
  exp8 = 0.8214249439940374

topk_refined_contact_f1:
  exp5 = 0.8297871496735562
  exp8 = 0.8288972412102136
```

However exp8 is now the best model so far on the most direct GT-contact
geometry metric:

```text
gt_contact_contact_dist_improvement:
  exp5 = 0.002825437071804292
  exp8 = 0.003047680515683272
```

This is the strongest evidence so far that:

```text
lightweight hand-target interaction is the correct model-side direction
```

## Correct Stage2 Reading

Under the corrected Stage2 objective, exp8 should not be judged primarily by
motion reconstruction weakness.

The correct reading is:

```text
1. exp8 is more contact-oriented than exp5
2. exp8 improves GT-contact distance more strongly than exp5
3. exp8 has not yet converted all of that geometry gain into the best binary contact F1
4. exp8 should be judged mainly by contact gain, GT-relative penetration gap, and visual contact quality
```

So the high-level status is:

```text
exp5 = stronger conservative baseline
exp8 = more faithful contact-refine direction
```

## Penetration Interpretation

The absolute surrogate penetration value should not be used as a standalone
negative conclusion.

The correct criterion is:

```text
compare refined vs coarse relative to GT penetration / overclose
```

If GT itself contains overclose / penetration, then:

```text
moving closer to GT can legitimately increase absolute penetration
```

Therefore `surrogate_penetration_depth_improvement` should remain secondary
unless interpreted through GT-relative gap reporting.

## Final Conclusion

```text
1. exp8 is the first model upgrade that clearly validates the interaction-module direction
2. exp8 currently gives the best GT-contact distance improvement in the project
3. exp8 still trails exp5 slightly on binary contact F1 metrics
4. Stage2 selection should no longer be based mainly on reconstruction metrics
5. the next decisive comparison should be Stage1-only vs Stage1+Stage2 under:
   - STGCN / reconstruction metrics
   - contact metrics
   - GT-relative penetration / overclose metrics
```

## Practical Ranking

If ranking by conservative overall balance:

```text
exp5 ~= exp8
```

If ranking by direct contact-refine orientation:

```text
exp8 > exp5
```

This difference comes from the fact that Stage2 should prioritize:

```text
contact improvement over reconstruction preservation
```
