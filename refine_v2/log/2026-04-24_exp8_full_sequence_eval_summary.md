# exp8 Full-Sequence Eval Summary

Date: 2026-04-24

Experiment:

```text
refiner_v2_exp8_interaction_v1_10k
```

## Evaluation Scope

This is the formal Stage2 full-sequence evaluation:

```text
GT
coarse (Stage1 output)
refined (Stage1 + Stage2 output)
```

using:

```text
1. residual-space center-weighted stitching over overlapping windows
2. STGCN metrics in canonical / Stage1-aligned processed space
3. contact metrics in restored pair space / restored shape
4. contact-rich subset with balanced action-type sampling
```

## Counts and Stitching

```text
num_sequences = 1487
num_action_types = 15

mean_windows_per_sequence = 2.3618
mean_covered_frame_ratio = 0.3797
mean_overlap_frame_ratio = 0.0862
num_sequences_with_windows = 1487
```

Interpretation:

```text
1. every sampled sequence has usable Stage2 windows
2. Stage2 modifies about 38% of frames on average
3. overlap exists but is moderate (~8.6% of frames)
4. the evaluation is genuinely full-sequence, not just window-level
```

## STGCN Results

```text
GT:
  accuracy      = 0.9859
  diversity     = 21.5596
  multimodality = 4.7588
  fid           = ~0

coarse:
  accuracy      = 0.9778
  diversity     = 21.2029
  multimodality = 4.8366
  fid           = 0.4742

refined:
  accuracy      = 0.9785
  diversity     = 21.2734
  multimodality = 4.8203
  fid           = 0.2955
```

Interpretation:

```text
1. refined does not damage global motion distribution
2. refined is slightly better than coarse on STGCN accuracy
3. refined is significantly better than coarse on STGCN FID
4. Stage2 contact refinement is not being paid for by global-distribution collapse
```

## Contact Results

```text
coarse_contact_f1 = 0.7816277361600042
refined_contact_f1 = 0.7945530333590035
delta_contact_f1 = +0.012925297199

gt_contact_contact_dist_improvement = 0.001841994933784008
all_valid_dist_l1_improvement = 0.010855725966393948
all_valid_contact_dist_improvement = 0.013058219105005264

contact_ratio_error_improvement = 0.0055951580363146625
contact_duration_error_improvement = 0.2513984531395863
contact_frequency_error_improvement = -0.01008742434431742
contact_jitter_error_improvement = -0.0001399150580196143
```

Interpretation:

```text
1. full-sequence contact F1 improves clearly over Stage1-only coarse
2. refined is closer to GT on direct contact geometry
3. contact ratio and duration move closer to GT
4. contact frequency and jitter are still slightly worse than coarse
5. exp8 is already improving contact at the system level, but sequence-level
   contact segmentation structure is not yet perfectly calibrated
```

## GT-Relative Penetration Interpretation

Important:

```text
penetration must be interpreted relative to GT, not as an absolute-minimize metric
```

Observed:

```text
surrogate_penetration_depth_improvement = -0.00024501560255885124
surrogate_penetration_depth_gap_improvement = +0.00024501560255885124
refined_penetration_depth_excess_over_gt = 0.0
```

Interpretation:

```text
1. refined absolute surrogate overclose is not smaller than coarse
2. but refined is closer to GT overclose / penetration than coarse is
3. refined does not exceed GT on the surrogate depth metric
4. under the corrected Stage2 objective, this is a positive result
```

## Action-Type Breakdown

Strong improvements:

```text
High-five:
  F1 delta = +0.0937
  gt_contact_contact_dist_improvement = 0.01374

Dance:
  F1 delta = +0.0354
  dist improvement = 0.00351

Massaging leg:
  F1 delta = +0.0288
  dist improvement = 0.00373

Hand wrestling:
  F1 delta = +0.0216
  dist improvement = 0.00266

Handshake:
  F1 delta = +0.0178
  dist improvement = 0.00264
```

These are exactly the kinds of actions where the lightweight hand-target
interaction module should help most.

Weak / remaining hard cases:

```text
Pull:
  F1 delta = -0.0014

Support with hand:
  F1 delta = -0.0037

Link arms:
  F1 delta = -0.0052
  dist improvement = -0.000656
```

Interpretation:

```text
current exp8 is strongest on clear hand-centric contact actions
and still weaker on broader arm-body / torso-body interaction types
```

## Final Stage2 Reading

Under the corrected Stage2 objective:

```text
Stage2 is a contact-refine module, not a motion-reconstruction module
```

Therefore the correct decision criteria are:

```text
1. does refined improve full-sequence contact over coarse?
2. does refined move closer to GT contact geometry?
3. does refined stay reasonable in GT-relative penetration / overclose?
4. does refined avoid collapsing global motion distribution?
```

exp8 satisfies all four conditions.

## Final Conclusion

```text
1. exp8 is valid at the full-sequence system level, not just at the window level
2. Stage1 + Stage2(exp8) clearly improves contact over Stage1-only coarse
3. STGCN metrics are not harmed; they slightly improve
4. GT-relative penetration / overclose behavior is also improved
5. exp8 can now be treated as a legitimate Stage2 main result
```

## Remaining Gap

```text
1. contact frequency / jitter calibration is still slightly imperfect
2. broad interaction types such as Link arms / Support with hand remain harder
3. any further iteration should be a very small calibration pass, not a new branch
```
