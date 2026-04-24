# exp7 Contact Ablation Plan

Date: 2026-04-24

Context:

```text
refiner_v2_exp7_contact_refine_v1_10k underperformed exp5.
```

Observed exp7 result:

```text
pred_motion_error = 0.0146233759
pred_contact_motion_error = 0.0146865906

all_valid_dist_l1_improvement = 0.0017661969
gt_contact_contact_dist_improvement = 0.0024847729
refined_contact_f1 = 0.8158168217
topk_refined_contact_f1 = 0.8235661854
```

exp5 reference:

```text
pred_motion_error = 0.0130820452
pred_contact_motion_error = 0.0130605551

all_valid_dist_l1_improvement = 0.0027992890
gt_contact_contact_dist_improvement = 0.0028254371
refined_contact_f1 = 0.8221591739
topk_refined_contact_f1 = 0.8297871497
```

Conclusion:

```text
The exp7 bundle changed too many things at once:
geometry v2 + separate residual heads + contact proxy loss + GT-relative overclose.
```

So the next step is not another bundled design, but two controlled ablations.

## Ablation A

Experiment:

```text
refiner_v2_exp7a_geom_v2_only_10k
```

Changes vs exp5:

```text
use_geometry_features = true
use_geometry_v2_features = true
geometry_feature_cache = v2
```

Keep disabled:

```text
use_separate_residual_heads = false
lambda_contact_geometry = 0.0
lambda_gt_relative_overclose = 0.0
```

Purpose:

```text
is geometry v2 input itself useful?
```

Interpretation:

```text
If exp7a > exp5:
  geometry v2 is useful, and exp7 failed mainly because of the new loss/head design.

If exp7a ~= exp5:
  geometry v2 input is neutral.

If exp7a < exp5:
  geometry v2 input itself is not helping in the current encoder/backbone form.
```

## Ablation B

Experiment:

```text
refiner_v2_exp7b_geom_v2_light_contact_10k
```

Changes vs exp7a:

```text
lambda_contact_geometry = 0.1
lambda_gt_relative_overclose = 0.0
```

Keep disabled:

```text
use_separate_residual_heads = false
```

Purpose:

```text
is a very light geometry-weighted contact supervision usable,
without the heavier exp7 proxy-loss bundle?
```

Interpretation:

```text
If exp7b > exp7a:
  light contact supervision helps.

If exp7b < exp7a:
  the current contact proxy loss is still misaligned and should not be used
  without redesign.
```

## Decision Rule

```text
Best of exp5 / exp7a / exp7b becomes the next baseline.
```

If both ablations fail:

```text
drop current contact proxy loss path
keep exp5
rethink contact loss as a more direct geometry/distance objective
```
