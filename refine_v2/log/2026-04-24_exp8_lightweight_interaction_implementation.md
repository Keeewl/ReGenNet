# exp8 Lightweight Hand-Target Interaction Implementation

Date: 2026-04-24

Experiment:

```text
refiner_v2_exp8_interaction_v1_10k
```

## Goal

Implement one more meaningful model-side upgrade without moving to a heavy
full spatial-attention redesign.

The design keeps the exp5 backbone style but adds:

```text
lightweight hand-target interaction modeling
selected-hand / same-side-arm focused residual boosting
very light contact regularization
```

## Implemented Model Changes

Updated:

```text
refine_v2/model/condition_encoder.py
refine_v2/model/refiner_v2.py
refine_v2/train/trainer.py
refine_v2/cli_train_refiner.py
```

### 1. Lightweight Hand-Target Interaction

New config:

```text
use_hand_target_interaction
```

The condition encoder now builds a small per-frame interaction module over the
top-k target regions.

It uses:

```text
primary selected-hand to target geometry
top-k region geometry tokens
optional geometry-v2 nearest-point / velocity features
```

And computes:

```text
query from current frame contact/geometric context
attention over top-k target region tokens
interaction summary per frame
```

This is not a full spatial transformer. It is a task-specific lightweight
interaction block focused on:

```text
selected hand / same-side arm
vs
top-k target regions
```

### 2. Focused Hand/Arm Residual Booster

New config:

```text
use_focused_hand_arm_boost
hand_interaction_boost_scale
arm_interaction_boost_scale
```

The refiner still has the normal shared residual output head.

When enabled, it additionally predicts:

```text
hand boost residual
arm boost residual
```

These are masked only onto:

```text
selected hand joints
same-side arm joints
```

So the model gets extra capacity exactly where Stage2 needs it most, without
switching to a fully separate multi-head body/transl design.

## exp8 Default Run

The default exp8 run:

```text
keeps exp5 backbone width/depth
uses geometry v2 cache
enables lightweight hand-target interaction
enables focused hand/arm residual boost
uses only very light contact regularization
```

Key parameters:

```text
use_geometry_features = true
use_geometry_v2_features = true
use_hand_target_interaction = true
use_focused_hand_arm_boost = true

hand_interaction_boost_scale = 0.25
arm_interaction_boost_scale = 0.15

lambda_contact_geometry = 0.03
lambda_gt_relative_overclose = 0.0
lambda_boundary_trans = 2.0
```

Important non-changes:

```text
no separate residual heads
no heavy spatial transformer
no phase preserve
```

## Commands Added

Train:

```text
refine_v2/commands/train/08_train_refiner_exp8_interaction_v1.sh
```

Eval:

```text
refine_v2/commands/eval/15_eval_refiner_exp8_interaction_v1.sh
refine_v2/commands/eval/16_eval_contact_refiner_exp8_interaction_v1.sh
```

Visual:

```text
refine_v2/commands/visual/16_export_refiner_vis_pack_exp8_interaction_v1.sh
refine_v2/commands/visual/17_diagnose_refiner_vis_pack_exp8_interaction_v1.sh
```

Output path:

```text
refine_v2/save/train/refiner_v2_exp8_interaction_v1_10k
```
