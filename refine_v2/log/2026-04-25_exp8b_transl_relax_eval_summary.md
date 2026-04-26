# exp8b transl relax evaluation summary

## Scope

Experiment:

- `refiner_v2_exp8b_transl_relax_10k`

Goal:

- test whether a small transl relaxation on top of exp8 can improve Stage2 contact refinement

Compared against:

- `exp8_interaction_v1_10k`

## Configuration

Relative to exp8:

- `transl_delta_scale = 0.3` (from `0.2`)
- `root_delta_scale = 0.2` (unchanged)
- `lambda_boundary_trans = 1.0` (from `2.0`)
- `lambda_contact_geometry = 0.03` (unchanged)

## Immediate behavior

The intended behavior was achieved:

- transl residual magnitude increased
- boundary stability was still preserved

Key numbers:

- `delta_norm_transl`
  - exp8: `0.00020620567090465396`
  - exp8b: `0.0004361076853193543`
- `boundary_trans_jump_excess`
  - exp8: `-2.6215136578259542e-06`
  - exp8b: `1.6236165141265776e-07`

Interpretation:

- transl was meaningfully relaxed
- window-boundary behavior remained stable

## Window-level results

exp8b:

- `gt_contact_contact_dist_improvement = 0.003091907999345312`
- `refined_contact_f1 = 0.82147090909662`
- `topk_refined_contact_f1 = 0.8292480359936559`
- `all_valid_dist_l1_improvement = 0.0024927801756240934`
- `contact_motion_improvement = 0.002493617414919239`

exp8 reference:

- `gt_contact_contact_dist_improvement = 0.003047680515683272`
- `refined_contact_f1 = 0.8214249439940374`
- `topk_refined_contact_f1 = 0.8288972412102136`
- `all_valid_dist_l1_improvement = 0.0025347111161972244`
- `contact_motion_improvement = 0.002881524312023031`

Window-level judgment:

- slight positive gain on GT-contact distance
- F1 / top-k F1 are only marginally better
- some global improvement metrics are slightly worse
- overall this is a small local gain, not a qualitative jump

## Full-sequence results

exp8b:

- `refined_contact_f1 = 0.7941684977507704`
- `gt_contact_contact_dist_improvement = 0.0018250503344461322`
- `surrogate_penetration_depth_gap_improvement = 0.000270589254796505`

exp8 reference:

- `refined_contact_f1 = 0.7945530333590035`
- `gt_contact_contact_dist_improvement = 0.001841994933784008`
- `surrogate_penetration_depth_gap_improvement = 0.00024501560255885124`

Full-sequence judgment:

- full-sequence contact F1 is slightly worse than exp8
- full-sequence GT-contact distance improvement is also slightly worse than exp8
- GT-relative penetration gap is slightly better than exp8

System-level conclusion:

- exp8b does not outperform exp8

## Overall interpretation

This experiment clarifies the role of transl in the current Stage2 framework:

- small transl relaxation can provide a bit of extra local contact help
- but transl is not the main remaining bottleneck
- gains do not reliably survive full-sequence stitching and system-level evaluation

So transl should be treated as:

- a limited calibration handle
- not a new optimization direction

## Final decision

Keep:

- `exp8_interaction_v1_10k` as the Stage2 main result

Treat exp8b as:

- the final transl-relax calibration
- useful evidence that transl can help locally
- but not enough to beat exp8 at the system level
