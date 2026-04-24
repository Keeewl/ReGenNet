# exp9 spatial interaction implementation

## Goal

Implement the final planned Stage2 model-side upgrade on top of exp8:

- keep the Stage2 system lightweight
- keep selector / window / subset / restored-space / full-sequence eval fixed
- strengthen hand-target spatial interaction
- avoid reopening transl / phase / heavy full-spatial-transformer branches

## Implemented changes

### 1. Stronger task-specific spatial interaction block

Updated:

- `refine_v2/model/condition_encoder.py`

Added:

- `use_hand_target_spatial_attention`
- `interaction_num_layers`
- `interaction_num_heads`
- `RefineV2SpatialInteractionBlock`

Design:

- still uses the exp8 interaction path as the base
- upgrades it from a single soft attention over top-k regions to an explicit lightweight spatial interaction block
- per frame, builds:
  - selected-hand query
  - same-side-arm query
  - top-k target-region tokens
- applies:
  - region self-attention
  - hand-to-region cross-attention
  - arm-to-region cross-attention
  - small FFNs

Scope is still narrow and contact-specific:

- selected hand
- same-side arm
- top-k target regions

This is not a full-body spatial transformer.

### 2. Refiner config wiring

Updated:

- `refine_v2/model/refiner_v2.py`

Added config fields:

- `use_hand_target_spatial_attention`
- `interaction_num_layers`
- `interaction_num_heads`

The model still uses:

- shared temporal backbone
- focused hand/arm booster
- group-gated residual scales

So exp9 remains a lightweight residual refiner.

### 3. Trainer / CLI support

Updated:

- `refine_v2/train/trainer.py`
- `refine_v2/cli_train_refiner.py`

Added CLI flags:

- `--use_hand_target_spatial_attention`
- `--interaction_num_layers`
- `--interaction_num_heads`

## exp9 command set

Added:

- `refine_v2/commands/train/09_train_refiner_exp9_spatial_interaction_v1.sh`
- `refine_v2/commands/eval/18_eval_refiner_exp9_spatial_interaction_v1.sh`
- `refine_v2/commands/eval/19_eval_contact_refiner_exp9_spatial_interaction_v1.sh`
- `refine_v2/commands/eval/20_eval_full_sequence_exp9_spatial_interaction_v1.sh`
- `refine_v2/commands/visual/19_export_refiner_vis_pack_exp9_spatial_interaction_v1.sh`
- `refine_v2/commands/visual/20_diagnose_refiner_vis_pack_exp9_spatial_interaction_v1.sh`

## Recommended exp9 training config

Based on exp8, with stronger interaction but still conservative loss settings:

- `use_geometry_features = true`
- `use_geometry_v2_features = true`
- `use_hand_target_interaction = true`
- `use_hand_target_spatial_attention = true`
- `interaction_num_layers = 2`
- `interaction_num_heads = 4`
- `use_focused_hand_arm_boost = true`
- `use_group_gated_residual = true`
- `hand_interaction_boost_scale = 0.30`
- `arm_interaction_boost_scale = 0.18`
- `lambda_contact_geometry = 0.03`
- `lambda_gt_relative_overclose = 0.01`

## Verification

Passed:

- `python -m py_compile` on updated model / trainer / CLI files
- `zsh -n` on all new exp9 command scripts
- dummy forward smoke test with:
  - geometry features
  - geometry v2
  - hand-target interaction
  - spatial interaction attention
  - focused hand/arm boost
  - group-gated residual

## Current judgment

exp9 is the final planned Stage2 model-side upgrade:

- it stays lightweight
- it does not change the Stage2 protocol
- it directly targets the current bottleneck:
  stronger hand-target spatial interaction for contact refinement
