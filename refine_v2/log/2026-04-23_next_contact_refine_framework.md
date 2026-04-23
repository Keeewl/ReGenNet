# Next Contact-Refine Framework

Date: 2026-04-23

Current baseline:

```text
refiner_v2_exp5_scope_geom_10k
```

## Core Decision

The next major iteration should stop focusing on translation/phase-loss tuning
and move to a complete contact-aware refiner upgrade.

Stage2's main target remains:

```text
improve reactor hand/arm physical contact quality
```

The refiner should therefore optimize contact geometry more directly, not only
motion reconstruction.

## Penetration / Overclose Principle

Penetration should not be interpreted as:

```text
smaller is always better
```

Because GT contact can itself contain small mesh interpenetration or surrogate
penetration. If the model only minimizes penetration, it may pull the hand away
from the actor and hurt real contact quality.

The correct criterion is GT-relative:

```text
refined contact should be closer to GT contact
refined penetration should be close to GT penetration
refined penetration should not clearly exceed GT penetration
```

Therefore, contact eval should report:

```text
coarse penetration
refined penetration
GT penetration
refined-vs-GT penetration gap
coarse-vs-GT penetration gap
```

Acceptable behavior:

```text
refined contact distance improves
refined contact F1 improves
refined penetration approaches GT
refined penetration does not become substantially worse than GT
```

## Planned Upgrade

Use exp5 as the base and implement a full contact-aware refiner:

```text
baseline = refiner_v2_exp5_scope_geom_10k
goal = significantly improve reactor hand/arm contact metrics
```

### 1. Geometry Feature Cache V2

Add richer offline geometry features while keeping training fast:

```text
hand center -> top-k target region vectors/distances
palm -> top-k target region vectors/distances
fingertip group -> top-k target region vectors/distances
distance velocity / trend
contact phase / GT contact mask features
coarse contact mask features
```

Implementation principle:

```text
compute offline
store in cache
load by window key
avoid dense SMPL-X geometry inside the training loop unless needed later
```

### 2. Contact Geometry Loss

Add a lightweight training loss that directly targets hand-target geometry.

Initial preferred form:

```text
L_contact_geom =
  SmoothL1(refined_hand_target_distance, gt_hand_target_distance)
```

or a margin version:

```text
L_contact_geom =
  max(0, refined_distance - gt_distance - margin)
```

Use it mainly on:

```text
GT contact frames
high-purity selector windows
top-k target regions
```

Top-k handling:

```text
use best/min over top-k target regions
avoid over-penalizing primary-region attribution mistakes
```

### 3. GT-Relative Overclose / Penetration Loss

Do not penalize all penetration blindly.

Instead, penalize only when refined is clearly more overclosed than GT:

```text
L_overclose =
  max(0, gt_relative_overclose_excess - margin)
```

Equivalent target:

```text
refined should not be substantially more penetrated/overclosed than GT
```

This protects against excessive contact pulling while still allowing realistic
contact closeness.

### 4. Model Upgrade

Upgrade the refiner output structure from a single shared residual head to
separate scope-aware heads:

```text
hand_head
arm_head
body_head
transl_head
```

Expected behavior:

```text
hand/arm heads can learn stronger contact corrections
body/transl heads remain conservative
Stage2 remains focused on contact refinement, not global motion rewriting
```

Keep:

```text
geometry-conditioned transformer backbone
group-gated residual scaling
group-weighted motion/contact losses
boundary translation anchor
```

### 5. Eval Upgrade

The main evaluation should compare coarse/refined/GT contact state:

```text
contact F1
top-k contact F1
GT-contact distance improvement
refined-vs-GT distance gap
coarse-vs-GT distance gap
GT penetration
coarse penetration
refined penetration
refined-vs-GT penetration gap
coarse-vs-GT penetration gap
```

Required breakdowns:

```text
action type
hand side
primary region
top-k region
window purity / failure mode
```

Visual validation:

```text
aitviewer vis pack
GT / coarse / refined side-by-side
sequence-level window annotations
selected failure cases and high-improvement cases
```

## What Not To Prioritize Next

Do not make these the main next experiment:

```text
more phase-loss tuning
action-type conditioning
large translation correction
full-sequence stitching loss
blind model-size increase
absolute penetration minimization
```

## Expected Gain

Small parameter-only tuning is likely limited to:

```text
refined_contact_f1 +0.001 to +0.003
```

The full contact-aware refiner is the more plausible route toward:

```text
refined_contact_f1 >= 0.835
topk_refined_contact_f1 >= 0.840
gt_contact_contact_dist_improvement >= 0.0035 to 0.0040
```

## Final Summary

Freeze exp5 as the practical baseline.

Next implementation should be:

```text
geometry feature v2
contact-distance loss
GT-relative overclose / penetration loss
separate hand/arm/body/transl residual heads
contact-centric eval upgrade
aitviewer validation
```

This is the most direct and efficient path toward the Stage2 objective:

```text
better reactor hand/arm contact quality without turning Stage2 into global
translation correction.
```
